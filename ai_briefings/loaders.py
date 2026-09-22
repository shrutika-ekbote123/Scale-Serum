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
    """Gemini cost in USD from token counts, via billing/pricing.json. None when
    billing is unavailable or the model is unpriced - never an invented rate."""
    try:
        import billing
        pricing = billing.load_pricing()
        from billing.cost import gemini_cost
    except Exception:  # noqa: BLE001
        return None

    def price(usage: dict, attempts: int) -> Optional[float]:
        if not attempts:
            return None
        try:
            return gemini_cost(pricing, model_requested=model, attempts=attempts,
                               input_tokens=usage.get("input_tokens"),
                               output_tokens=usage.get("output_tokens"),
                               thinking_tokens=usage.get("thinking_tokens"),
                               cached_tokens=usage.get("cached_tokens")).usd
        except Exception:  # noqa: BLE001
            return None
    return price
