"""MongoDB storage: cached wording and per-brand tier overrides.

Two collections, both keyed by `_id` so no secondary index is needed:

    ai_suggested_next_actions      _id = lead_id   latest wording + its fingerprint
    ai_suggested_next_action_tiers _id = brand_id  the brand's tier override

Only the WORDING is cached. The decision is recomputed on every request because
it depends on the clock (a lead goes from high to medium urgency as time passes);
when the recomputed decision and evidence match the cached fingerprint, the cached
wording is reused and Gemini is not called.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NextActionStore:
    def __init__(self, cache_collection, tiers_collection):
        self.cache = cache_collection
        self.tiers = tiers_collection

    # ------------------------------------------------------------------ wording
    async def get_cached(self, lead_id: str) -> Optional[dict]:
        try:
            return await self.cache.find_one({"_id": lead_id})
        except Exception:  # noqa: BLE001 - a cache miss must never fail the request
            return None

    async def save(self, lead_id: str, *, fingerprint: str, wording: dict,
                   decision: dict, versions: dict) -> None:
        try:
            await self.cache.update_one({"_id": lead_id}, {"$set": {
                "lead_id": lead_id, "fingerprint": fingerprint, "wording": wording,
                "action_type": decision["action_type"],
                "urgency_level": decision["urgency_level"],
                "versions": versions, "updated_at": _now(),
            }}, upsert=True)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ tier override
    async def get_tier_override(self, brand_id: Optional[str]) -> Optional[dict]:
        if not brand_id:
            return None
        try:
            return await self.tiers.find_one({"_id": brand_id})
        except Exception:  # noqa: BLE001
            return None

    async def set_tier_override(self, brand_id: str, tiers: list[dict]) -> dict:
        doc = {"brand_id": brand_id, "tiers": tiers, "updated_at": _now()}
        await self.tiers.update_one({"_id": brand_id}, {"$set": doc}, upsert=True)
        return {"_id": brand_id, **doc}

    async def delete_tier_override(self, brand_id: str) -> bool:
        result = await self.tiers.delete_one({"_id": brand_id})
        return bool(getattr(result, "deleted_count", 0))
