"""Turn thumbs-down turns into candidate evaluation cases.

    python -m evals.from_feedback                 # write evals/candidates.jsonl
    python -m evals.from_feedback --brand <uuid>  # one brand only

THE LOOP THIS CLOSES
    The golden set started with 12 real questions, because that is all the
    history there was. It grows from here the only way a golden set honestly
    can: somebody used the coach, disagreed with an answer, and said why. This
    reads those turns and writes them out in case form.

WHAT IT DOES NOT DO
    Label them. A candidate arrives with `expect_intent: null` and empty
    expectations, and stays out of the suite until a person decides what the
    right answer would have been. A case whose expectations were generated from
    the same system it is meant to test proves nothing, so that judgment is left
    where it belongs.

    It also does not copy the answer that was marked down. The usage store keeps
    counters, not conversations; the text lives in the backend's coach_thread,
    and the `request_id` here is how to find it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "candidates.jsonl")


async def _store():
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO, ".env"))
    from motor.motor_asyncio import AsyncIOMotorClient

    from script_lab_coach.usage import UsageStore

    db = AsyncIOMotorClient(os.environ["MONGODB_URI"])[os.environ["MONGODB_DB"]]
    return UsageStore(db["script_lab_coach_usage"])


def _existing_questions() -> set:
    """So a complaint already covered by the suite is not added twice."""
    seen = set()
    for name in ("production.jsonl", "adversarial.jsonl", "candidates.jsonl"):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    case = json.loads(line)
                    seen.add((case.get("test_id"), (case.get("question") or "").lower()))
    return seen


async def main() -> None:
    parser = argparse.ArgumentParser(description="thumbs-down turns -> candidate cases")
    parser.add_argument("--brand", default=None, help="one brand id")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    store = await _store()
    rows = await store.flagged(args.brand, args.limit)
    seen = _existing_questions()

    written = 0
    with open(OUT, "a", encoding="utf-8") as fh:
        for row in rows:
            feedback = row.get("feedback") or {}
            # The question itself is not in the counter row - only the shape of
            # the turn. The reason the user gave is what a human needs to write
            # the case, and the request_id is how they find the conversation.
            case = {
                "id": f"fb-{str(row.get('_id'))[:8]}",
                "source": "feedback",
                "scenario": "user_marked_down",
                "test_id": row.get("test_id"),
                "brand_id": row.get("brand_id"),
                "question": "",
                "expect_intent": None,
                "must_mention": [],
                "must_refuse": False,
                "labelled": False,
                "from_turn": {
                    "request_id": row.get("_id"),
                    "at": row.get("created_at"),
                    "intent": row.get("intent"),
                    "routed_by": row.get("routed_by"),
                    "brand_brain_tier": row.get("brand_brain_tier"),
                    "grounded": row.get("grounded"),
                    "fallback": row.get("fallback"),
                    "fallback_reason": row.get("fallback_reason"),
                    "confidence": row.get("confidence"),
                    "latency_ms": row.get("latency_ms"),
                },
                "user_said": feedback.get("reason"),
                "note": "UNLABELLED. Copy the question from coach_thread using request_id, "
                        "decide what the right answer was, then move this into "
                        "adversarial.jsonl. The runner skips unlabelled cases.",
            }
            if (case["test_id"], "") in seen:
                continue
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
            written += 1

    print(f"{len(rows)} turns marked down; wrote {written} candidate(s) to {OUT}")
    if written:
        print("Each one needs a question and expectations before it counts as a test.")


if __name__ == "__main__":
    asyncio.run(main())
