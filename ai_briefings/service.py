"""Generation: one brand, one local day -> five stored briefings.

    load brand -> resolve timezone -> load the day (scrumdb, read-only)
    -> score untouched leads -> load call analyses (Mongo)
    -> build sales / ads / whatsapp / leads -> word them (Gemini, in parallel)
    -> build All from the four -> word it -> save all five -> record the run

Owns no clients or connections: everything arrives in BriefingDeps, which is
what lets the tests run the whole pipeline with fakes. Used by both the API
(POST /generate, as a background task) and the daily worker process.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from . import data as _data
from . import hot_leads as _hot
from . import timezones as tzs
from . import writer as _writer
from .config import SECTIONS, load_config
from .sections import BUILDERS, DayContext, overall
from .store import briefing_id

logger = logging.getLogger("ai_briefings")


@dataclass
class BriefingDeps:
    run_sync: Callable[..., Awaitable[Any]]                  # threadpool runner
    store: Any                                               # BriefingStore
    llm_client: Any
    llm_model: str
    load_brand: Callable[[str], Optional[dict]] = _data.load_brand
    load_day: Callable[..., dict] = _data.load_day
    score_hot: Callable[[list, list], dict] = _hot.score
    load_analyses: Optional[Callable[[list], Awaitable[dict]]] = None
    load_brand_brain: Optional[Callable[[Optional[str]], Awaitable[Optional[dict]]]] = None
    price_usage: Optional[Callable[[dict, int], Optional[dict]]] = None
    now: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))


class BrandNotFound(Exception):
    pass


def new_run_id() -> str:
    return uuid.uuid4().hex


def _callout_labels(section: str, result: dict) -> tuple[str, str]:
    if section == "sales" and any(r.get("watch") for r in result.get("reps") or []):
        return "Top", "Coach"
    return "Top", "Watch"


def _document(result: dict, *, brand: dict, day: date, zone: tzs.BrandZone, cfg: dict,
              wording: dict, source: str, fallback_reason: Optional[str], usage: dict,
              model: str, freshness: list, run_id: str, now: datetime,
              cost: Optional[dict]) -> dict:
    section = result["section"]
    labels = cfg["sections"][section]
    top_label, watch_label = _callout_labels(section, result)
    window = tzs.day_window(day, zone)
    bid = briefing_id(brand["id"], day.isoformat(), section)
    doc = {
        "_id": bid, "briefing_id": bid, "brand_id": brand["id"], "brand_name": brand.get("name"),
        "date": day.isoformat(), "date_label": tzs.date_label(day),
        "short_label": tzs.short_label(day),
        "section": section, "section_label": labels["label"],
        "available": result["available"], "reason": result["reason"],
        "message": result["message"],
        "summary": {"label": labels["summary_label"], "text": wording["summary"],
                    "top_label": top_label, "top": wording["top"],
                    "watch_label": watch_label, "watch": wording["watch"]},
        "blocks": [{"key": b["key"], "title": next(
                        (t["title"] for t in result["template"]["blocks"] if t["key"] == b["key"]),
                        b["key"]), "bullets": b["bullets"]} for b in wording["blocks"]],
        "watch": result["watch"], "kpis": result["kpis"], "facts": result["facts"],
        "timezone": {"name": zone.name, "source": zone.source, "raw": brand.get("timezone")},
        "window": {"start": window.start.isoformat(), "end": window.end.isoformat()},
        "data_as_of": {f["platform"]: (f["last_date"].isoformat() if f["last_date"] else None)
                       for f in freshness if f["accounts"]},
        "wording": {"source": source, "fallback_reason": fallback_reason},
        "fallback": source != "llm",
        "versions": {"config_version": cfg["config_version"],
                     "prompt_version": _writer.PROMPT_VERSION, "llm_model": model},
        # Tokens and what they cost at the rates in force NOW. Stored per
        # briefing so a later price change cannot rewrite an old bill.
        "usage": {**usage, "total_tokens": sum(int(v or 0) for v in usage.values())} if usage else {},
        "cost": cost, "cost_usd": (cost or {}).get("usd"),
        "generated_at": now.isoformat(timespec="seconds"), "run_id": run_id,
    }
    for extra in ("score_card", "reps", "public_summary"):
        if extra in result:
            doc[extra] = result[extra]
    return doc


async def _word(section_result: dict, deps: BriefingDeps, cfg: dict, day: date,
                brand_ctx: dict) -> dict:
    if not section_result["available"]:
        return {"wording": _writer.template_wording({"draft": section_result["template"]}),
                "source": "none", "fallback_reason": section_result["reason"],
                "meta": {"attempts": 0, "usage": {}}}
    payload = _writer.build_payload(
        section_result, section_label=cfg["sections"][section_result["section"]]["label"],
        date_label=tzs.date_label(day), brand=brand_ctx)
    forbid = tuple(r["rep"] for r in section_result.get("reps") or []) \
        if section_result["section"] == "sales" else ()
    return await _writer.write(deps.llm_client, deps.llm_model, payload,
                               forbid_in_summary=forbid)


async def generate_day(brand_id: str, deps: BriefingDeps, *, day: Optional[date] = None,
                       trigger: str = "manual", run_id: Optional[str] = None,
                       run_queued: bool = False) -> dict:
    """Generate and store all five briefings for one brand-local day.

    Returns the finished run document. Never raises for expected failures (brand
    missing, scrumdb down): the run is recorded as failed with a reason."""
    cfg = load_config()
    now = deps.now()
    run_id = run_id or new_run_id()
    started = datetime.now(timezone.utc)

    try:
        brand = await deps.run_sync(deps.load_brand, brand_id)
    except _data.DataUnavailable as err:
        brand, load_error = None, f"database_unavailable:{err}"
    else:
        load_error = None if brand else "brand_not_found"
    zone = tzs.resolve((brand or {}).get("timezone"))
    day = day or tzs.yesterday(now, zone)
    day_str = day.isoformat()

    if run_queued:
        await deps.store.mark_running(run_id)
    else:
        await deps.store.start_run(run_id, brand_id=brand_id, day=day_str, trigger=trigger)

    async def fail(reason: str) -> dict:
        logger.warning("briefing run failed [brand_id=%s date=%s reason=%s]",
                       brand_id, day_str, reason)
        await deps.store.finish_run(run_id, status="failed", error=reason)
        return {"run_id": run_id, "brand_id": brand_id, "date": day_str,
                "status": "failed", "error": reason}

    if load_error:
        return await fail(load_error)

    try:
        raw = await deps.run_sync(deps.load_day, brand_id, day, zone, cfg, now)
    except _data.DataUnavailable as err:
        return await fail(f"database_unavailable:{err}")

    hot = await deps.run_sync(deps.score_hot, raw["untouched"], cfg["leads"]["hot_priorities"])
    analysis_ids = sorted({c["analysis_id"] for c in raw["calls"] if c.get("analysis_id")})
    analyses = (await deps.load_analyses(analysis_ids)) if deps.load_analyses and analysis_ids else {}

    ctx = DayContext(day=day, zone=zone, cfg=cfg, currency=brand.get("currency") or "INR",
                     now=now, analyses=analyses, hot=hot)
    results = {}
    for name in SECTIONS:
        try:
            results[name] = BUILDERS[name](raw, ctx)
        except Exception as err:  # noqa: BLE001 - one bad section must not sink the rest
            logger.exception("briefing section failed [brand_id=%s date=%s section=%s]",
                             brand_id, day_str, name)
            from .sections.common import unavailable
            results[name] = unavailable(name, "section_error",
                                        f"This section could not be built ({type(err).__name__}).")
    results["all"] = overall.build(results, ctx)

    brand_brain = (await deps.load_brand_brain(brand.get("brand_brain_id"))
                   if deps.load_brand_brain else None)
    voice = ((brand_brain or {}).get("answers") or {}).get("brandVoice")
    brand_ctx = {k: v for k, v in {"name": brand.get("name"), "voice": voice}.items() if v}

    worded = dict(zip(SECTIONS, await asyncio.gather(
        *(_word(results[s], deps, cfg, day, brand_ctx) for s in SECTIONS))))
    worded["all"] = await _word(results["all"], deps, cfg, day, brand_ctx)

    status_by_section, totals = {}, {}
    cost_usd, cost_inr, calls = 0.0, 0.0, 0
    for name in ("all",) + SECTIONS:
        w = worded[name]
        usage = w["meta"].get("usage") or {}
        attempts = w["meta"].get("attempts", 0)
        for k, v in usage.items():
            totals[k] = totals.get(k, 0) + v
        calls += attempts
        cost = deps.price_usage(usage, attempts) if deps.price_usage else None
        cost_usd += float((cost or {}).get("usd") or 0.0)
        cost_inr += float((cost or {}).get("inr") or 0.0)
        doc = _document(results[name], brand=brand, day=day, zone=zone, cfg=cfg,
                        wording=w["wording"], source=w["source"],
                        fallback_reason=w["fallback_reason"], usage=usage,
                        model=deps.llm_model, freshness=raw["freshness"], run_id=run_id,
                        now=now, cost=cost)
        await deps.store.save(doc)
        status_by_section[name] = {"available": results[name]["available"],
                                   "wording": w["source"],
                                   "fallback_reason": w["fallback_reason"]}

    duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    if totals:
        totals["total_tokens"] = sum(int(v or 0) for v in totals.values())
    cost_summary = {"usd": round(cost_usd, 6), "inr": round(cost_inr, 4), "llm_calls": calls,
                    "estimated": True}
    await deps.store.finish_run(run_id, status="completed", sections=status_by_section,
                                usage=totals, cost=cost_summary, duration_ms=duration_ms,
                                timezone={"name": zone.name, "source": zone.source})
    logger.info("briefing run completed [brand_id=%s date=%s ms=%d]", brand_id, day_str,
                duration_ms)
    return {"run_id": run_id, "brand_id": brand_id, "date": day_str, "status": "completed",
            "sections": status_by_section, "usage": totals, "cost": cost_summary,
            "duration_ms": duration_ms}
