"""Frozen contexts, so the suite can run with no database and no API key.

    python -m evals.fixtures            # re-snapshot from live data

WHY
    The full suite calls a model and reads scrumdb. That is the right way to
    measure answer quality, and the wrong way to gate a build: it needs a key,
    a VPN-reachable database and real money, none of which a CI runner should
    have. So the inputs to a handful of real turns are frozen to JSON, and CI
    replays them against the deterministic paths - the fallback wording, the
    starters, the router and every gate.

WHAT IT CANNOT TELL YOU
    Nothing about the model's answers, because no model is called. An offline
    run proves the machinery around the model is sound: that the stand-in
    wording is grounded, that a failed review is never defended, that a brand
    with no Brand Brain still gets a usable turn. Answer quality still needs
    `--adapter coach` against live data, by hand or on a schedule.

Snapshots are inputs, never outputs. Nothing here records what a model said,
so a fixture cannot quietly become the expected answer.
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "fixtures")

# One per shape the coach has to handle. Chosen from real rows, not invented.
SNAPSHOT_TESTS = {
    "di_failing_script": "05cc1f26-3527-48b4-99bc-c748963d44ab",     # 5/100, tier A, stale ad
    "di_never_published": "7f5572ba-e202-4c10-9d44-4c346fdec89c",    # no source_ad_id
    "orphan_no_brand_brain": "57ee67bb-afde-4fa0-8039-bbe72b66add4",  # tier C
    "review_did_not_complete": "b10eb698-d70a-4a56-920f-3d68192f7319",  # ai_fallback
}


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


async def _capture(builder, name: str, test_id: str) -> dict:
    """Every input one turn was built from - and nothing it produced."""
    from script_lab_coach import intents as _intents

    ctx = await builder.build(test_id=test_id,
                              extra_tiers=(_intents.PERFORMANCE, _intents.CORPUS,
                                           _intents.HISTORY))
    return {
        "name": name,
        "test": _json_safe(ctx.test),
        "brand": _json_safe(ctx.brand),
        "brand_brain": {"answers": _json_safe(ctx.brain.answers),
                        "context": _json_safe(ctx.brain.context),
                        "brand_brain_id": ctx.brain.brand_brain_id,
                        "brand_name": ctx.brain.brand_name},
        "performance": _json_safe(ctx.performance),
        "creatives": _json_safe(ctx.creatives),
        "competitors": _json_safe(ctx.competitors),
        "versions": _json_safe(ctx.versions),
        "other_brands": list(ctx.other_brands),
    }


class OfflineBuilder:
    """A context builder backed by frozen JSON instead of two databases.

    Deliberately the same interface as the real one, so the suite, the starters
    and the whole service run against it unchanged - what CI exercises is the
    production code path, not a parallel one written for testing."""

    def __init__(self, snapshots: dict):
        self.snapshots = snapshots
        self.by_test = {s["test"]["test_id"]: s for s in snapshots.values()}

    async def build(self, *, test_id: str, intent_name=None, brand_id=None,
                    extra_tiers=()):
        from script_lab_coach import intents as _intents
        from script_lab_coach.brand_brain import BrandBrain
        from script_lab_coach.context import BrandMismatch, CoachContext, TestNotFound
        from script_lab_coach import facts as _facts

        snap = self.by_test.get(test_id)
        if snap is None:
            raise TestNotFound(test_id)
        test = snap["test"]
        if brand_id and test.get("brand_id") and str(brand_id) != str(test["brand_id"]):
            raise BrandMismatch(test_id)

        intent = _intents.get(intent_name)
        wanted = set(intent.needs) | set(extra_tiers)
        performance = snap["performance"] if _intents.PERFORMANCE in wanted else None
        creatives = snap["creatives"] if _intents.CORPUS in wanted else {}
        competitors = snap["competitors"] if _intents.CORPUS in wanted else []
        versions = snap["versions"] if _intents.HISTORY in wanted else []

        brain = BrandBrain(answers=snap["brand_brain"]["answers"],
                           context=snap["brand_brain"]["context"],
                           brand_brain_id=snap["brand_brain"]["brand_brain_id"],
                           brand_name=snap["brand_brain"]["brand_name"])

        computed = (_facts.from_test(test)
                    + _facts.from_performance(performance)
                    + _facts.from_versions(versions)
                    + _facts.from_creatives(creatives))

        return CoachContext(test=test, brand=snap["brand"], brain=brain, intent=intent,
                            facts=computed, performance=performance, creatives=creatives,
                            competitors=competitors, versions=versions,
                            other_brands=tuple(snap["other_brands"]))


def load() -> OfflineBuilder:
    """The committed snapshots, as a builder."""
    snapshots = {}
    if os.path.isdir(STORE):
        for filename in sorted(os.listdir(STORE)):
            if filename.endswith(".json"):
                with open(os.path.join(STORE, filename), encoding="utf-8") as fh:
                    snap = json.load(fh)
                snapshots[snap["name"]] = snap
    return OfflineBuilder(snapshots)


async def _main() -> None:
    from evals.run import _builder

    os.makedirs(STORE, exist_ok=True)
    builder = await _builder()
    for name, test_id in SNAPSHOT_TESTS.items():
        snap = await _capture(builder, name, test_id)
        path = os.path.join(STORE, f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=1, ensure_ascii=False)
        brain = snap["brand_brain"]["answers"]
        print(f"  {name:24} test={test_id[:8]} brand={snap['brand'].get('name')} "
              f"bb_fields={len(brain)} perf={'yes' if snap['performance'] else 'no'} "
              f"creatives={len(snap['creatives'].get('winners') or [])}")
    print(f"wrote {len(SNAPSHOT_TESTS)} snapshots to {STORE}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
