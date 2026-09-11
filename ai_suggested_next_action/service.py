"""Orchestration: load -> tiers -> window -> decide -> word -> respond.

Owns no clients or connections. Everything it needs from the process arrives in
SuggestDeps, which is what lets the tests run it with fakes and no network.

House convention: always a well-formed response. A lead that cannot be read
comes back with `availability.available: false` and a reason, never a 500 and
never an invented recommendation.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from . import data as _data
from . import rules as _rules
from . import writer as _writer
from .framework import load_framework
from .tiers import infer_tiers

UNAVAILABLE_LEAD_NOT_FOUND = "lead_not_found"
UNAVAILABLE_DB_ERROR = "database_unavailable"

_MESSAGES = {
    UNAVAILABLE_LEAD_NOT_FOUND: "No lead with this id exists.",
    UNAVAILABLE_DB_ERROR: "The lead database could not be read. Try again shortly.",
}


@dataclass
class SuggestDeps:
    run_sync: Callable[..., Awaitable[Any]]           # threadpool runner
    load_lead: Callable[[str], Optional[dict]]        # sync, raises DataUnavailable
    brand_payments: Callable[[Optional[str]], list]   # sync
    brand_windows: Callable[[Optional[str]], dict]    # sync
    load_brand_brain: Callable[[Optional[str]], Awaitable[Optional[dict]]]
    load_call_analysis: Callable[[str], Awaitable[Optional[dict]]]
    store: Any                                        # NextActionStore or None
    llm_client: Any
    llm_model: str
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


def _versions(cfg: dict, model: str) -> dict:
    return {"framework_version": cfg["framework_version"],
            "prompt_version": _writer.PROMPT_VERSION, "llm_model": model}


def _unavailable(lead_id: str, reason: str, cfg: dict, model: str, now: datetime) -> dict:
    return {
        "lead_id": lead_id, "brand_id": None, "brand_name": None,
        "ai_suggested_next_action": None,
        "lead_state": None, "tiers": None, "conversion_window": None,
        "confidence": None, "data_flags": [],
        "wording": None, "versions": _versions(cfg, model),
        "generated_at": now.isoformat(timespec="seconds"), "cached": False,
        "availability": {"available": False, "reason": reason, "message": _MESSAGES[reason]},
        "fallback": True,
    }


def _call_insights(report: Optional[dict]) -> Optional[dict]:
    """The parts of a sales-call analysis that make a follow-up specific. Quotes
    and transcript text are deliberately left out."""
    if not report:
        return None
    r = report.get("report") or report
    insights = {
        "summary": (r.get("summary") or "")[:600] or None,
        "objections": [{"summary": o.get("summary"), "handled": o.get("handled")}
                       for o in (r.get("objections") or [])[:3]],
        "buying_signals": [{"summary": s.get("summary"), "strength": s.get("strength")}
                           for s in (r.get("buying_signals") or [])[:3]],
        "customer_needs": [n.get("summary") for n in (r.get("customer_needs") or [])[:3]],
    }
    return {k: v for k, v in insights.items() if v}


def _brand_context(brand_name: Optional[str], brand_brain: Optional[dict]) -> dict:
    """Voice and language only. The Brand Brain's ideal-customer and journey text
    is deliberately NOT given to the writer: when it is wrong (Lawtorney's
    describes a legal-drafting SaaS; its payments are a board-director
    programme) the model repeats it to the rep as if it were known about this
    lead. The Brand Brain still sets the tone here and the buying window in
    rules.conversion_window."""
    answers = (brand_brain or {}).get("answers") or {}
    return {k: v for k, v in {
        "name": brand_name,
        "voice": answers.get("brandVoice"),
        "language": answers.get("language"),
    }.items() if v}


def _act_within(hours: int) -> str:
    return f"{hours} hours" if hours < 48 else f"{hours // 24} days"


def _first_name(full_name: Optional[str]) -> Optional[str]:
    parts = (full_name or "").strip().split()
    return parts[0].title() if parts else None


async def suggest_next_action(lead_id: str, deps: SuggestDeps, *, refresh: bool = False,
                              brand_brain_id: Optional[str] = None) -> dict:
    cfg = load_framework()
    now = deps.now()

    try:
        uuid.UUID(str(lead_id))
    except ValueError:
        return _unavailable(lead_id, UNAVAILABLE_LEAD_NOT_FOUND, cfg, deps.llm_model, now)

    try:
        ctx = await deps.run_sync(deps.load_lead, lead_id)
    except _data.DataUnavailable:
        return _unavailable(lead_id, UNAVAILABLE_DB_ERROR, cfg, deps.llm_model, now)
    if ctx is None:
        return _unavailable(lead_id, UNAVAILABLE_LEAD_NOT_FOUND, cfg, deps.llm_model, now)

    brand = ctx["brand"]

    # Tiers: the brand's override when it has one, otherwise inferred.
    override = await deps.store.get_tier_override(brand["id"]) if deps.store else None
    if override and override.get("tiers"):
        tiers, tier_source = override["tiers"], "override"
    else:
        payments = await deps.run_sync(deps.brand_payments, brand["id"])
        tiers = infer_tiers(payments, cfg["tiers"])
        tier_source = "inferred" if tiers else "none"

    brand_brain = await deps.load_brand_brain(brand_brain_id or brand.get("brand_brain_id"))
    stats = await deps.run_sync(deps.brand_windows, brand["id"])
    window = _rules.conversion_window(stats, brand_brain, cfg)

    decision = _rules.decide(ctx, cfg=cfg, tiers=tiers, window=window,
                             brand_brain=brand_brain, now=now)

    # The most recent analysed call, if any, makes a follow-up specific.
    insights = None
    analysed = [c for c in ctx["calls"] if c.get("analysis_id")]
    if analysed:
        insights = _call_insights(await deps.load_call_analysis(analysed[-1]["analysis_id"]))

    level = decision["urgency_level"]
    urgency_cfg = cfg["urgency"][level]
    template = cfg["actions"][decision["action_type"]]

    payload = {
        "brand": _brand_context(brand.get("name"), brand_brain),
        "lead": {k: v for k, v in {
            "first_name": _first_name(ctx["lead"].get("full_name")),
            "designation": ctx["lead"].get("designation"),
        }.items() if v},
        "decision": {
            "action_type": decision["action_type"], "channel": template["channel"],
            "urgency": urgency_cfg["label"],
            # "Disqualify within 7 days" is meaningless, so no-action carries no deadline.
            "act_within": (None if decision["action_type"] == "disqualify"
                           else _act_within(urgency_cfg["act_within_hours"])),
            "urgency_basis": decision["urgency_basis"],
            "default_title": template["title"],
            "default_recommendation": template["text"],
        },
        "facts": decision["facts"],
        "evidence": decision["evidence"],
    }
    if insights:
        payload["call_insights"] = insights

    versions = _versions(cfg, deps.llm_model)
    fingerprint = hashlib.sha256(json.dumps(
        [payload, versions], sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")).hexdigest()

    cached_doc = None
    if deps.store and not refresh:
        cached_doc = await deps.store.get_cached(lead_id)
    if cached_doc and cached_doc.get("fingerprint") == fingerprint and cached_doc.get("wording"):
        wording, source, fallback_reason, cached = cached_doc["wording"], "cache", None, True
    else:
        written = await _writer.write(deps.llm_client, deps.llm_model, payload, template)
        wording, source = written["wording"], written["source"]
        fallback_reason, cached = written["fallback_reason"], False
        # Template wording is not cached, so the next request tries the model again.
        if deps.store and source == "llm":
            await deps.store.save(lead_id, fingerprint=fingerprint, wording=wording,
                                  decision=decision, versions=versions)

    return {
        "lead_id": lead_id,
        "brand_id": brand.get("id"),
        "brand_name": brand.get("name"),
        "ai_suggested_next_action": {
            "urgency": {
                "level": level,
                "label": urgency_cfg["label"],
                "act_within_hours": urgency_cfg["act_within_hours"],
                "due_by": decision["due_by"],
                "basis": decision["urgency_basis"],
            },
            "recommendation": {
                "action_type": decision["action_type"],
                "channel": template["channel"],
                "title": wording["title"],
                "text": wording["recommendation"],
            },
            "reason": {
                "headline": wording["reason_headline"],
                "text": wording["reason"],
                "evidence": decision["evidence"],
            },
        },
        "lead_state": decision["lead_state"],
        "tiers": {"source": tier_source, "items": tiers},
        "conversion_window": window,
        "confidence": decision["confidence"],
        "data_flags": decision["flags"],
        "wording": {"source": source, "fallback_reason": fallback_reason,
                    "call_analysis_used": bool(insights)},
        "versions": versions,
        "generated_at": now.isoformat(timespec="seconds"),
        "cached": cached,
        "availability": {"available": True, "reason": None, "message": None},
        "fallback": source == "template",
    }
