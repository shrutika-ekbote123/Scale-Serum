"""The Mongo reads and the cost pricing the service needs, built once from the
caller's collections - app.py and the worker share them, so both processes
produce identical briefings."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("ai_briefings")


def analyses_loader(collection):
    """analysis_id -> the completed report, for the Good/Watch column and scores."""
    async def load(ids: list) -> dict:
        if collection is None or not ids:
            return {}
        try:
            docs = await collection.find(
                {"_id": {"$in": list(ids)}, "status": "completed"},
                {"report.scores": 1, "report.strengths": 1, "report.weaknesses": 1},
            ).to_list(len(ids))
        except Exception:  # noqa: BLE001 - analyses refine the tab, never block it
            logger.warning("could not read sales call analyses for a briefing")
            return {}
        return {d["_id"]: d.get("report") or {} for d in docs}
    return load


def brand_brain_loader(collection):
    async def load(brand_brain_id: Optional[str]) -> Optional[dict]:
        if collection is None or not brand_brain_id:
            return None
        try:
            return await collection.find_one({"_id": brand_brain_id}, {"answers.brandVoice": 1})
        except Exception:  # noqa: BLE001
            return None
    return load


def usage_pricer(model: str):
    """What one section's Gemini call cost, via billing/pricing.json.

    Returns the rates it used and the caveats that apply, not just a number:
    an unconfirmed alias or rate, and the fixed USD->INR rate, are named on
    every briefing. `usd` is None when the model is unpriced - never an
    invented rate. None (no pricer) when billing itself is unavailable."""
    try:
        import billing
        from billing.cost import (NOTE_FX_FIXED_RATE, NOTE_GEMINI_ALIAS_UNCONFIRMED,
                                  NOTE_GEMINI_RATE_UNCONFIRMED, gemini_cost)
        pricing = billing.load_pricing()
    except Exception:  # noqa: BLE001
        return None

    fx_cfg = pricing.get("fx") or {}
    fx = fx_cfg.get("usd_to_inr")

    def price(usage: dict, attempts: int) -> Optional[dict]:
        if not attempts:
            return None
        try:
            cost = gemini_cost(pricing, model_requested=model, attempts=attempts,
                               input_tokens=usage.get("input_tokens"),
                               output_tokens=usage.get("output_tokens"),
                               thinking_tokens=usage.get("thinking_tokens"),
                               cached_tokens=usage.get("cached_tokens"))
        except Exception:  # noqa: BLE001
            return None
        notes = []
        if cost.alias_confirmed is False:
            notes.append(NOTE_GEMINI_ALIAS_UNCONFIRMED)
        if cost.rate_confirmed is False:
            notes.append(NOTE_GEMINI_RATE_UNCONFIRMED)
        if fx and fx_cfg.get("fixed_rate", True):
            notes.append(NOTE_FX_FIXED_RATE)
        return {
            "usd": cost.usd,
            "inr": round(cost.usd * fx, 4) if cost.usd is not None and fx else None,
            "attempts": cost.attempts,
            "model_requested": cost.model_requested, "model_priced": cost.model_priced,
            "alias_confirmed": cost.alias_confirmed, "rate_confirmed": cost.rate_confirmed,
            "rate_effective_from": cost.rate_effective_from,
            "rate_input_per_1m_usd": cost.rate_input_per_1m_usd,
            "rate_output_per_1m_usd": cost.rate_output_per_1m_usd,
            "rate_cached_input_per_1m_usd": cost.rate_cached_input_per_1m_usd,
            "usd_to_inr": fx, "pricing_version": pricing.get("pricing_version"),
            "estimated": True, "reason": cost.reason, "notes": notes,
        }
    return price
