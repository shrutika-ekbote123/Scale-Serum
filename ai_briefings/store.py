"""MongoDB storage for generated briefings.

    ai_briefings       _id = "{brand_id}:{YYYY-MM-DD}:{section}"
                       one per brand, local day and tab. Regenerating a day
                       replaces it; other days are never touched.
    ai_briefing_runs   _id = run_id
                       one per generation attempt: trigger, status per
                       section, timings, Gemini tokens. The worker reads it to
                       stay idempotent and to cap retries.

Dates are stored as ISO strings, which sort correctly and need no timezone.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

HISTORY_FIELDS = {"_id": 1, "brand_id": 1, "date": 1, "date_label": 1, "short_label": 1,
                  "section": 1, "section_label": 1, "summary": 1, "available": 1,
                  "generated_at": 1, "fallback": 1}


def briefing_id(brand_id: str, day: str, section: str) -> str:
    return f"{brand_id}:{day}:{section}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BriefingStore:
    def __init__(self, briefings, runs):
        self.briefings = briefings
        self.runs = runs

    async def ensure_indexes(self) -> None:
        await self.briefings.create_index([("brand_id", 1), ("section", 1), ("date", -1)])
        await self.runs.create_index([("brand_id", 1), ("date", 1), ("started_at", -1)])

    # ------------------------------------------------------------ briefings
    async def save(self, doc: dict) -> None:
        await self.briefings.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)

    async def get(self, bid: str) -> Optional[dict]:
        return await self.briefings.find_one({"_id": bid})

    async def get_for(self, brand_id: str, day: str, section: str) -> Optional[dict]:
        return await self.briefings.find_one({"_id": briefing_id(brand_id, day, section)})

    async def latest(self, brand_id: str, section: str, before: str) -> Optional[dict]:
        """The newest briefing for this tab dated before `before` (exclusive)."""
        docs = await (self.briefings.find({"brand_id": brand_id, "section": section,
                                           "date": {"$lt": before}})
                      .sort("date", -1).to_list(1))
        return docs[0] if docs else None

    async def history(self, brand_id: str, sections: list[str], before: Optional[str],
                      limit: int) -> list[dict]:
        query: dict = {"brand_id": brand_id, "section": {"$in": sections}}
        if before:
            query["date"] = {"$lt": before}
        return await (self.briefings.find(query, HISTORY_FIELDS)
                      .sort("date", -1).to_list(limit))

    # ----------------------------------------------------------------- runs
    async def start_run(self, run_id: str, *, brand_id: str, day: str, trigger: str) -> dict:
        doc = {"_id": run_id, "run_id": run_id, "brand_id": brand_id, "date": day,
               "trigger": trigger, "status": "running", "sections": {},
               "started_at": _now(), "finished_at": None, "error": None}
        await self.runs.insert_one(doc)
        return doc

    async def queue_run(self, run_id: str, *, brand_id: str, day: str, trigger: str) -> dict:
        doc = {"_id": run_id, "run_id": run_id, "brand_id": brand_id, "date": day,
               "trigger": trigger, "status": "queued", "sections": {},
               "started_at": None, "finished_at": None, "error": None, "queued_at": _now()}
        await self.runs.insert_one(doc)
        return doc

    async def mark_running(self, run_id: str) -> None:
        await self.runs.update_one({"_id": run_id}, {"$set": {"status": "running",
                                                              "started_at": _now()}})

    async def finish_run(self, run_id: str, **fields) -> None:
        await self.runs.update_one({"_id": run_id},
                                   {"$set": {**fields, "finished_at": _now()}})

    async def runs_for(self, brand_id: str, day: str) -> list[dict]:
        return await self.runs.find({"brand_id": brand_id, "date": day}).to_list(50)

    async def recent_runs(self, brand_id: str, limit: int) -> list[dict]:
        return await (self.runs.find({"brand_id": brand_id})
                      .sort("started_at", -1).to_list(limit))
