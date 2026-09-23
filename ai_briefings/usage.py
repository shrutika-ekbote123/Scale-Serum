"""What the briefings cost: tokens and estimated spend over a date range.

Pure aggregation of what generation already stored. Every briefing records the
tokens its Gemini call used and the cost at the RATES IN FORCE WHEN IT RAN, so
a later price change never rewrites an old bill - the same rule the Sales Call
Analyzer's billing summary follows.

Everything here is an ESTIMATE from configured rates. Where the model name is
an unconfirmed alias, a rate is unconfirmed, or the USD->INR rate is the team's
fixed one, the response says so in `notes` rather than quietly rounding it away.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

TOKEN_KEYS = ("input_tokens", "output_tokens", "thinking_tokens", "cached_tokens")
UNPRICED = "unpriced_briefings"
# Briefings generated before the cost block existed stored USD only; their INR
# is converted at today's rate, which is not the rate they were priced at.
LEGACY = "legacy_cost_usd_only_inr_converted_at_current_rate"


def _blank() -> dict:
    return {**{k: 0 for k in TOKEN_KEYS}, "total_tokens": 0, "briefings": 0,
            "llm_calls": 0, "cost_usd": 0.0, "cost_inr": 0.0}


def _add(bucket: dict, usage: dict, cost: Optional[dict], attempts: int) -> None:
    bucket["briefings"] += 1
    bucket["llm_calls"] += attempts
    for key in TOKEN_KEYS:
        value = int(usage.get(key) or 0)
        bucket[key] += value
        bucket["total_tokens"] += value
    if cost:
        bucket["cost_usd"] += float(cost.get("usd") or 0.0)
        bucket["cost_inr"] += float(cost.get("inr") or 0.0)


def _finish(bucket: dict) -> dict:
    bucket["cost_usd"] = round(bucket["cost_usd"], 6)
    bucket["cost_inr"] = round(bucket["cost_inr"], 4)
    return bucket


def _cost_of(doc: dict, fx: Optional[float]) -> tuple[Optional[dict], bool]:
    """(cost, legacy). Briefings generated before the cost block existed stored
    only cost_usd; their INR is converted here at the current rate."""
    cost = doc.get("cost")
    if cost:
        return cost, False
    usd = doc.get("cost_usd")
    if usd is not None:
        return {"usd": usd, "inr": round(usd * fx, 4) if fx else None}, True
    return None, False


def summarize(docs: Iterable[dict], *, brand_id: str, start: str, end: str,
              pricing: Optional[dict] = None) -> dict:
    """`docs`: stored briefings in [start, end], each with usage/cost/wording."""
    fx = ((pricing or {}).get("fx") or {}).get("usd_to_inr")
    totals, per_day, per_section, per_model = _blank(), defaultdict(_blank), \
        defaultdict(_blank), defaultdict(_blank)
    notes: set = set()
    unpriced = 0
    wording = defaultdict(int)
    model_rates: dict = {}

    for doc in docs:
        usage = doc.get("usage") or {}
        source = (doc.get("wording") or {}).get("source") or "none"
        wording[source] += 1
        if not usage:
            # A tab with no data, or template wording from an LLM outage: no call,
            # no tokens, no cost. Counted, but it is not a priced briefing.
            continue
        cost, legacy = _cost_of(doc, fx)
        if legacy:
            notes.add(LEGACY)
        attempts = int((cost or {}).get("attempts") or 1)
        if not cost or cost.get("usd") is None:
            unpriced += 1
        for bucket in (totals, per_day[doc["date"]], per_section[doc["section"]]):
            _add(bucket, usage, cost, attempts)
        model = (doc.get("versions") or {}).get("llm_model") or "unknown"
        _add(per_model[model], usage, cost, attempts)
        if cost:
            notes.update(cost.get("notes") or [])
            model_rates.setdefault(model, {k: cost.get(k) for k in (
                "model_priced", "alias_confirmed", "rate_confirmed", "rate_effective_from",
                "rate_input_per_1m_usd", "rate_output_per_1m_usd",
                "rate_cached_input_per_1m_usd", "pricing_version")})
    if unpriced:
        notes.add(UNPRICED)

    days = len(per_day)
    cost_usd = round(totals["cost_usd"], 6)
    averages = {
        "per_briefing_usd": round(cost_usd / totals["briefings"], 6) if totals["briefings"] else None,
        "per_day_usd": round(cost_usd / days, 6) if days else None,
        "per_day_inr": round(totals["cost_inr"] / days, 4) if days else None,
        "projected_30_days_usd": round(cost_usd / days * 30, 4) if days else None,
        "basis": "days with at least one briefing generated in this range",
    }
    fx_cfg = (pricing or {}).get("fx") or {}
    return {
        "brand_id": brand_id, "from": start, "to": end,
        "days_with_briefings": days,
        "totals": _finish(totals),
        "wording": dict(wording),
        "unpriced_briefings": unpriced,
        "per_day": [{"date": d, **_finish(b)} for d, b in sorted(per_day.items(), reverse=True)],
        "per_section": [{"section": s, **_finish(b)}
                        for s, b in sorted(per_section.items(), key=lambda kv: -kv[1]["cost_usd"])],
        "per_model": [{"model": m, **model_rates.get(m, {}), **_finish(b)}
                      for m, b in per_model.items()],
        "averages": averages,
        "pricing": {"version": (pricing or {}).get("pricing_version"),
                    "usd_to_inr": fx_cfg.get("usd_to_inr"),
                    "fx_fixed_rate": fx_cfg.get("fixed_rate"),
                    "fx_set_on": fx_cfg.get("set_on")},
        "estimated": True,
        "notes": sorted(notes),
    }
