"""What the coach costs: recording a turn, and reporting a date range.

WHY THIS SERVICE STORES ANYTHING AT ALL
    The conversation belongs to the main backend, in coach_thread. But a usage
    report has to count turns that happened, and asking the backend to give
    those back would make the bill depend on somebody else's schema. So one
    small document per turn is written here, in MongoDB - which this service
    already writes - carrying what it cost and nothing else.

WHAT IS DELIBERATELY NOT STORED
    The question, the answer and the script. They are in coach_thread already,
    and a second copy in a second database is a second thing to secure and to
    delete. This collection holds counters.

PRICED WHEN IT RAN
    Each turn records the cost at the rates in force that day, exactly as the
    Sales Call Analyzer and AI Briefings bills do, so a later price change never
    rewrites an old bill. Everything is an ESTIMATE; `notes` names every caveat.
"""
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable, Optional

logger = logging.getLogger("script_lab_coach")

TOKEN_KEYS = ("input_tokens", "output_tokens", "thinking_tokens", "cached_tokens")

# WHAT COUNTS AS TROUBLE
#
# Thresholds, not opinions: each one is the level at which somebody should look,
# and each is overridable per deployment. They are deliberately tight. A coach
# that invents a figure once in a hundred turns is a coach nobody should trust,
# and a fallback rate creeping up means users are reading the stand-in wording
# while the dashboard still shows green.
THRESHOLDS = {
    # An answer the gate rejected twice. Zero, because the evaluation suite
    # gates groundedness at 100% and production should not hold itself to a
    # lower bar than the test suite: any ungrounded turn is worth a look.
    "ungrounded_rate": float(os.environ.get("COACH_ALERT_UNGROUNDED", 0.0)),
    # No model answer at all: an outage, a timeout, or repeated rejection.
    "fallback_rate": float(os.environ.get("COACH_ALERT_FALLBACK", 0.02)),
    # The panel is a chat. Past this it stops feeling like one.
    "latency_p95_ms": float(os.environ.get("COACH_ALERT_P95_MS", 12_000)),
    # People rarely bother to complain. When they do, it matters.
    "thumbs_down_rate": float(os.environ.get("COACH_ALERT_THUMBS_DOWN", 0.15)),
    # Classification is the expensive route. Mostly-matched traffic is the
    # design working; a collapse means the phrase patterns have stopped fitting
    # what people ask, which is a product signal before it is a cost one.
    "model_routed_rate": float(os.environ.get("COACH_ALERT_MODEL_ROUTED", 0.40)),
}

# Below this many turns, a rate is noise. Two turns, one of them a fallback, is
# not a 50% fallback rate worth waking anyone for.
MIN_TURNS_FOR_ALERTS = int(os.environ.get("COACH_ALERT_MIN_TURNS", 20))


class UsageStore:
    """One document per coach turn. `_id` is the request id, so a retried
    request updates its row instead of billing twice."""

    def __init__(self, collection):
        self.collection = collection
        self._indexed = False

    async def ensure_indexes(self) -> None:
        """Create the index once, lazily, on the first write.

        NOT at startup. Doing it in the app's lifespan opens a MongoDB
        connection the moment the app is constructed, which breaks every test
        that builds a TestClient: the driver binds to that request's event loop
        and the next test finds it closed. Ten sales-call billing tests started
        erroring the moment this ran on boot.

        Lazily is also simply more honest about what it is - housekeeping for a
        collection that does not exist until something is written to it."""
        if self._indexed:
            return
        try:
            await self.collection.create_index([("brand_id", 1), ("day", -1)])
            await self.collection.create_index([("feedback.rating", 1), ("created_at", -1)])
        except Exception:  # noqa: BLE001 - an index is not worth failing a turn for
            logger.warning("could not create coach usage indexes")
        self._indexed = True

    async def record(self, turn: dict, ctx, user_id: Optional[str] = None) -> None:
        await self.ensure_indexes()
        doc = {
            "_id": turn["request_id"],
            "brand_id": ctx.test.get("brand_id"),
            "test_id": ctx.test.get("test_id"),
            "user_id": user_id,
            "day": (turn["created_at"] or "")[:10],
            "created_at": turn["created_at"],
            "intent": turn["intent"],
            "routed_by": turn["routed_by"],
            "brand_brain_tier": (turn.get("brand_brain") or {}).get("tier"),
            "grounded": turn["grounded"],
            "fallback": turn["fallback"],
            "fallback_reason": turn["fallback_reason"],
            "confidence": turn["confidence"],
            "latency_ms": turn["latency_ms"],
            "model": turn["model"],
            "prompt_version": turn["prompt_version"],
            "usage": turn.get("usage") or {},
            "cost": turn.get("cost"),
        }
        await self.collection.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)

    async def between(self, brand_id: str, start: str, end: str) -> list:
        return await self.collection.find(
            {"brand_id": brand_id, "day": {"$gte": start, "$lte": end}}
        ).to_list(length=20_000)

    async def record_feedback(self, request_id: str, *, rating: str,
                              reason: Optional[str] = None,
                              user_id: Optional[str] = None) -> bool:
        """A thumb on one turn. False when the turn is unknown.

        Stored on the turn's own counter row rather than in a table of its own:
        a rating is only meaningful beside the intent, the tier and the latency
        that produced it, and keeping them together is what lets a thumbs-down
        become an evaluation case later without a join."""
        result = await self.collection.update_one(
            {"_id": request_id},
            {"$set": {"feedback": {
                "rating": rating,
                "reason": (reason or "").strip()[:500] or None,
                "user_id": user_id,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }}},
        )
        return bool(result.matched_count)

    async def flagged(self, brand_id: Optional[str] = None, limit: int = 100) -> list:
        """Turns a user marked down - the queue that feeds the golden set."""
        query: dict = {"feedback.rating": "down"}
        if brand_id:
            query["brand_id"] = brand_id
        return await self.collection.find(query).sort("created_at", -1).to_list(length=limit)


def _blank() -> dict:
    return {**{k: 0 for k in TOKEN_KEYS}, "total_tokens": 0, "turns": 0, "llm_calls": 0,
            "fallbacks": 0, "ungrounded": 0, "cost_usd": 0.0, "cost_inr": 0.0}


def _add(bucket: dict, doc: dict) -> None:
    bucket["turns"] += 1
    bucket["fallbacks"] += int(bool(doc.get("fallback")))
    bucket["ungrounded"] += int(doc.get("grounded") is False)
    usage = doc.get("usage") or {}
    if usage:
        bucket["llm_calls"] += 1
    for key in TOKEN_KEYS:
        value = int(usage.get(key) or 0)
        bucket[key] += value
        bucket["total_tokens"] += value
    cost = doc.get("cost") or {}
    if cost.get("usd") is not None:
        bucket["cost_usd"] += float(cost["usd"])
    if cost.get("inr") is not None:
        bucket["cost_inr"] += float(cost["inr"])


def _finish(bucket: dict) -> dict:
    bucket["cost_usd"] = round(bucket["cost_usd"], 6)
    bucket["cost_inr"] = round(bucket["cost_inr"], 4)
    return bucket


def summarize(docs: Iterable[dict], *, brand_id: str, start: str, end: str) -> dict:
    """Tokens, cost and health over a range - what the turns already recorded.

    The health counters travel with the bill on purpose: a month that got
    cheaper because half its turns fell back to the template is not a saving,
    and a report that hides that is a misleading one."""
    totals = _blank()
    per_day: dict = defaultdict(_blank)
    per_intent: dict = defaultdict(_blank)
    per_model: dict = defaultdict(_blank)
    per_route: dict = defaultdict(_blank)
    per_tier: dict = defaultdict(_blank)
    latencies: list = []
    notes: set = set()
    feedback = {"up": 0, "down": 0, "rated": 0, "reasons": []}
    fallback_reasons: dict = defaultdict(int)

    for doc in docs:
        _add(totals, doc)
        _add(per_day[doc.get("day") or "unknown"], doc)
        _add(per_intent[doc.get("intent") or "unknown"], doc)
        _add(per_model[doc.get("model") or "none"], doc)
        _add(per_route[doc.get("routed_by") or "unknown"], doc)
        _add(per_tier[doc.get("brand_brain_tier") or "unknown"], doc)
        if doc.get("fallback") and doc.get("fallback_reason"):
            fallback_reasons[doc["fallback_reason"]] += 1
        rating = (doc.get("feedback") or {}).get("rating")
        if rating in ("up", "down"):
            feedback["rated"] += 1
            feedback[rating] += 1
            reason = (doc.get("feedback") or {}).get("reason")
            if rating == "down" and reason:
                feedback["reasons"].append({"request_id": doc.get("_id"),
                                            "intent": doc.get("intent"),
                                            "reason": reason})
        if doc.get("latency_ms"):
            latencies.append(int(doc["latency_ms"]))
        if (doc.get("usage") or {}) and not (doc.get("cost") or {}):
            notes.add("unpriced_turns")
        for note in (doc.get("cost") or {}).get("notes") or []:
            notes.add(note)

    latencies.sort()

    def pct(p: float):
        """Nearest-rank percentile: the smallest value at or above which p of
        the turns fall. Interpolating would invent a latency nobody measured,
        which is the one thing this whole feature refuses to do."""
        if not latencies:
            return None
        rank = max(1, math.ceil(p * len(latencies)))
        return latencies[min(rank, len(latencies)) - 1]

    turns = totals["turns"]
    health = {
        "turns": turns,
        "fallback_rate": round(totals["fallbacks"] / turns, 4) if turns else 0.0,
        "ungrounded_rate": round(totals["ungrounded"] / turns, 4) if turns else 0.0,
        "latency_p50_ms": pct(0.5), "latency_p95_ms": pct(0.95),
        "thumbs_down_rate": round(feedback["down"] / feedback["rated"], 4)
        if feedback["rated"] else 0.0,
        "model_routed_rate": round(per_route.get("model", _blank())["turns"] / turns, 4)
        if turns else 0.0,
    }
    alerts = raise_alerts(health)

    return {
        "brand_id": brand_id, "from": start, "to": end,
        "totals": _finish(totals),
        "health": health,
        "status": "alert" if any(a["severity"] == "alert" for a in alerts) else (
            "warn" if alerts else "ok"),
        "alerts": alerts,
        "feedback": {"up": feedback["up"], "down": feedback["down"],
                     "rated": feedback["rated"],
                     "unrated": max(0, turns - feedback["rated"]),
                     "down_reasons": feedback["reasons"][:20]},
        "fallback_reasons": dict(sorted(fallback_reasons.items(),
                                        key=lambda kv: -kv[1])),
        "per_day": {k: _finish(v) for k, v in sorted(per_day.items())},
        "per_intent": {k: _finish(v) for k, v in sorted(per_intent.items())},
        "per_model": {k: _finish(v) for k, v in sorted(per_model.items())},
        "per_route": {k: _finish(v) for k, v in sorted(per_route.items())},
        "per_brand_brain_tier": {k: _finish(v) for k, v in sorted(per_tier.items())},
        "notes": sorted(notes),
    }


def raise_alerts(health: dict) -> list:
    """Which signals are past their threshold, and how badly.

    A rate over a handful of turns is noise, so nothing fires below
    MIN_TURNS_FOR_ALERTS - a quiet day with one fallback is not a 100% fallback
    rate, and an alert that cries wolf on a Sunday is an alert people mute."""
    turns = int(health.get("turns") or 0)
    if turns < MIN_TURNS_FOR_ALERTS:
        return []

    out = []
    for signal, note, severity in (
        ("ungrounded_rate",
         "answers were rejected for citing figures nothing supports", "alert"),
        ("fallback_rate",
         "turns produced no AI answer at all - users are reading the stand-in", "alert"),
        ("latency_p95_ms",
         "the slowest turns are too slow for a chat panel", "warn"),
        ("thumbs_down_rate",
         "users are marking answers down - read the reasons", "warn"),
        ("model_routed_rate",
         "most questions now need the classifier: the phrase patterns have "
         "stopped fitting what people ask", "warn"),
    ):
        value = health.get(signal)
        threshold = THRESHOLDS.get(signal)
        if value is None or threshold is None or value <= threshold:
            continue
        out.append({"signal": signal, "value": value, "threshold": threshold,
                    "severity": severity, "note": note})
    return out
