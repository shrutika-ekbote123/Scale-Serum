"""Run the Creative Coach evaluation suite.

    python -m evals.run --adapter baseline          # score the coach that ships today
    python -m evals.run --adapter coach             # score the new one (phase 3+)
    python -m evals.run --adapter baseline --json report.json

WHY THIS EXISTS BEFORE THE PROMPTS DO
    A prompt change without a measurement is an opinion. The suite is built
    first so that every later change to the coach - a new intent, a reworded
    instruction, a cheaper model - is answerable with a number rather than an
    impression.

    It also gives the rewrite something to beat. `--adapter baseline` scores the
    answers already stored in production, so "better" means better than what
    users have now, not better than nothing.

ADAPTERS
    baseline  replays the answer stored in coach_thread for that question. No
              model call, no cost; it measures what shipped.
    coach     calls the real coach service (phase 3). Not wired yet; the adapter
              raises rather than pretending.

GATES
    A hard check failing fails the run (exit 1), whatever the totals say. Soft
    checks are reported as rates and tracked over time. See metrics.py for which
    is which and why.
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

from evals import metrics  # noqa: E402

# Soft-check thresholds. Deliberately not 100%: these are qualities to improve,
# and a threshold nobody can meet is one everybody learns to ignore.
SOFT_TARGETS = {
    "intent": 0.95,
    "refused": 0.90,
    "covers_required_facts": 0.90,
    "gave_a_rewrite": 0.90,
    "answered_with_ai": 0.95,
}


def load_cases(only_source=None) -> list:
    cases = []
    for name in ("production.jsonl", "adversarial.jsonl"):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    if only_source:
        cases = [c for c in cases if c.get("source") == only_source]
    # An unlabelled production question is a to-do, not a test.
    return [c for c in cases if c.get("expect_intent") or c.get("source") == "adversarial"]


# --------------------------------------------------------------------- context
async def _builder():
    """The real context builder, against live data - the same one the service
    uses. The suite measures answers in the context they were really given."""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO, ".env"))
    from motor.motor_asyncio import AsyncIOMotorClient

    from script_lab_coach import CreativeCoachContextBuilder, brand_brain_loader

    db = AsyncIOMotorClient(os.environ["MONGODB_URI"])[os.environ["MONGODB_DB"]]

    async def run_sync(fn, *a, **kw):
        return await asyncio.to_thread(fn, *a, **kw)

    return CreativeCoachContextBuilder(run_sync=run_sync,
                                       load_brand_brain=brand_brain_loader(db.brand_brains))


# -------------------------------------------------------------------- adapters
async def baseline_adapter(case, ctx):
    """What the coach that ships today answered, replayed from storage."""
    return {"text": case.get("baseline_answer") or "",
            "intent": None, "suggested_rewrite": ""}


_DEPS = None


async def _deps():
    """The real service, in process. Deliberately not over HTTP: the suite is
    measuring the coach's answers, not FastAPI's routing, and an in-process call
    cannot fail for reasons that have nothing to do with the thing under test."""
    global _DEPS
    if _DEPS is not None:
        return _DEPS
    import os

    from google import genai

    from script_lab_coach import service as coach_service

    builder = await _builder()          # loads .env before the key is read
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    _DEPS = coach_service.CoachDeps(
        builder=builder,
        llm_client=client,
        llm_model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
        # Every other brand in the system. A coach that names one of these to
        # this brand's user has leaked, and the gate must catch it.
        # Other COMPANIES. "Director's Institute" is deliberately absent: it is
        # DI's own full name, and listing it made the coach fail for correctly
        # naming the brand it was coaching.
        other_brands=("Lawttorney", "Lawtorney", "LawTorney.ai", "Kaizen CRM",
                      "GlowLab", "Xilana"),
    )
    return _DEPS


async def _fixture_deps(case):
    """Deps whose Brand Brain has been trimmed to a tier the live data lacks.

    No brand in this system sits at tier B - both linked Brand Brains are
    complete - so the partial-context path has no real example and would ship
    untested. The fixture trims a real Brand Brain rather than inventing one, so
    what is exercised is the real document with fields removed, which is exactly
    what a half-finished onboarding produces."""
    from script_lab_coach import CreativeCoachContextBuilder
    from script_lab_coach import service as coach_service
    from script_lab_coach.brand_brain import ANSWER_FIELDS, BrandBrain

    wanted = (case.get("fixture") or {}).get("brand_brain_tier")
    keep = {"A": 9, "B": 4, "C": 0}.get(wanted, 9)
    base = await _deps()
    real_loader = base.builder._load_brain

    async def trimmed(brand_brain_id, brand_name=None):
        full = await real_loader(brand_brain_id, brand_name)
        answers = {f: full.answers[f] for f in ANSWER_FIELDS[:keep] if f in full.answers}
        return BrandBrain(answers=answers, context=full.context,
                          brand_brain_id=full.brand_brain_id, brand_name=brand_name)

    builder = CreativeCoachContextBuilder(run_sync=base.builder._run,
                                          load_brand_brain=trimmed)
    return coach_service.CoachDeps(builder=builder, llm_client=base.llm_client,
                                   llm_model=base.llm_model,
                                   other_brands=base.other_brands)


async def coach_adapter(case, ctx):
    """The real coach, asked the real question."""
    from script_lab_coach import service as coach_service

    deps = await (_fixture_deps(case) if case.get("fixture") else _deps())
    turn = await coach_service.chat(
        deps, test_id=case["test_id"], message=case["question"],
        intent=None,                      # let the router classify: that is under test too
        thread=case.get("thread") or [],
        brand_id=case.get("brand_id"))
    return turn


async def fallback_adapter(case, ctx):
    """The deterministic stand-in, with no model in the loop.

    This is what a user reads during an outage, or whenever the gate rejects an
    answer twice - so it has to clear the same hard gates the model's answer
    does. It costs nothing and needs no key, which is what makes it runnable in
    CI on every push."""
    from script_lab_coach import fallback as fallback_mod
    from script_lab_coach import router

    intent, _ = await router.route(case["question"], thread=case.get("thread") or [])
    ctx.intent = __import__("script_lab_coach.intents", fromlist=["get"]).get(intent)
    answer = fallback_mod.answer(ctx, case["question"])
    return {**answer, "intent": intent, "test_id": ctx.test["test_id"],
            "fallback": True, "fallback_reason": "offline_run"}


ADAPTERS = {"baseline": baseline_adapter, "coach": coach_adapter,
            "fallback": fallback_adapter}


# ---------------------------------------------------------------------- runner
async def run(adapter_name: str, only_source=None, offline: bool = False) -> dict:
    from script_lab_coach import intents as intents_mod
    from script_lab_coach.context import BrandMismatch, TestNotFound

    adapter = ADAPTERS[adapter_name]
    if offline:
        # Frozen contexts: no database, no key, no spend. The production code
        # path is unchanged - only where the context comes from differs.
        from evals import fixtures

        builder = fixtures.load()
    elif adapter_name == "coach":
        builder = (await _deps()).builder
    else:
        builder = await _builder()
    cases = load_cases(only_source)
    if offline:
        known = set(builder.by_test)
        cases = [c for c in cases if c["test_id"] in known]
    results = []
    skipped = []

    for case in cases:
        if adapter_name == "baseline" and not case.get("baseline_answer"):
            # An adversarial case has no stored answer; the shipping coach was
            # never asked it. Counting it as a pass would flatter the baseline.
            skipped.append(case["id"])
            continue
        # A fixture case is generated from a trimmed Brand Brain, so it must be
        # SCORED against that same context - otherwise the report checks a tier B
        # answer against tier A facts.
        case_builder = ((await _fixture_deps(case)).builder
                        if case.get("fixture") and adapter_name == "coach" and not offline
                        else builder)
        try:
            ctx = await case_builder.build(
                test_id=case["test_id"], intent_name=case.get("expect_intent"),
                brand_id=case.get("brand_id"),
                extra_tiers=(intents_mod.PERFORMANCE, intents_mod.HISTORY))
        except (TestNotFound, BrandMismatch) as err:
            print(f"  ! {case['id']}: context unavailable ({type(err).__name__})")
            continue
        answer = await adapter(case, ctx)
        results.append(metrics.score(case, answer, ctx))

    summary = metrics.summarise(results)
    summary["adapter"] = adapter_name
    summary["offline"] = offline
    summary["skipped"] = skipped
    summary["results"] = [r.as_dict() for r in results]
    return summary


def report(summary: dict, strict: bool = False) -> bool:
    """Print the scorecard. True when the run passes.

    Hard gates always decide the outcome. Soft checks decide it only under
    `strict`: they move a few points between runs at this temperature, and a
    build that fails at random teaches people to rerun until green, which
    quietly disables the hard gates travelling with it."""
    mode = " (offline: frozen fixtures, no model)" if summary.get("offline") else ""
    print(f"\n  Creative Coach evaluation — adapter: {summary['adapter']}{mode}")
    print(f"  {summary['passed']}/{summary['cases']} cases fully passed")
    if summary["skipped"]:
        print(f"  {len(summary['skipped'])} skipped (no stored answer to replay)")

    print("\n  HARD GATES")
    hard_ok = True
    for name, bucket in sorted(summary["per_check"].items()):
        if not bucket["hard"]:
            continue
        ok = bucket["passed"] == bucket["total"]
        hard_ok = hard_ok and ok
        mark = "PASS" if ok else "FAIL"
        print(f"    {mark}  {name:22} {bucket['passed']}/{bucket['total']}")

    print("\n  SOFT CHECKS")
    soft_ok = True
    for name, bucket in sorted(summary["per_check"].items()):
        if bucket["hard"]:
            continue
        target = SOFT_TARGETS.get(name)
        ok = target is None or bucket["rate"] >= target
        soft_ok = soft_ok and ok
        goal = f"(target {target:.0%})" if target else ""
        mark = "ok  " if ok else "under"
        print(f"    {mark} {name:22} {bucket['rate']:6.0%}  {goal}")

    failing = [r for r in summary["results"] if not r["passed"]]
    if failing:
        print(f"\n  FAILING CASES ({len(failing)})")
        for r in failing[:12]:
            why = "; ".join(r["notes"]) or ", ".join(r["hard_failures"] + r["soft_failures"])
            print(f"    {r['id']}  {r['question'][:44]:46} {why[:90]}")
        if len(failing) > 12:
            print(f"    ... and {len(failing) - 12} more")

    if not strict and not soft_ok:
        print("\n  (soft checks are under target but do not fail this run; "
              "use --strict to gate on them)")
    print()
    return hard_ok and (soft_ok or not strict)


def compare(summary: dict, baseline_path: str) -> bool:
    """Report what changed against a stored run. False when something regressed.

    Rates move at this temperature, so a rate falling is reported but does not
    fail. What fails is a CASE that used to pass and now does not: that is a
    specific, reproducible claim about a specific question, which is the kind of
    regression worth blocking a build for."""
    try:
        with open(baseline_path, encoding="utf-8") as fh:
            before = json.load(fh)
    except (OSError, ValueError) as err:
        print(f"\n  baseline unreadable ({err}); nothing to compare")
        return True

    was = {r["id"]: r["passed"] for r in before.get("results", [])}
    now = {r["id"]: r["passed"] for r in summary.get("results", [])}

    broke = sorted(i for i, ok in now.items() if ok is False and was.get(i) is True)
    fixed = sorted(i for i, ok in now.items() if ok is True and was.get(i) is False)
    added = sorted(i for i in now if i not in was)
    gone = sorted(i for i in was if i not in now)

    print(f"\n  AGAINST BASELINE ({os.path.basename(baseline_path)}, "
          f"{before.get('adapter')}{' offline' if before.get('offline') else ''})")
    for label, ids in (("regressed", broke), ("fixed", fixed),
                       ("new", added), ("no longer run", gone)):
        if ids:
            print(f"    {label:14} {', '.join(ids)}")
    if not any((broke, fixed, added, gone)):
        print("    unchanged")

    for name, bucket in sorted(summary["per_check"].items()):
        old = (before.get("per_check") or {}).get(name)
        if not old:
            continue
        drift = bucket["rate"] - old["rate"]
        if abs(drift) >= 0.05:
            print(f"    {name:22} {old['rate']:.0%} -> {bucket['rate']:.0%} "
                  f"({drift:+.0%})")
    return not broke


def main() -> None:
    parser = argparse.ArgumentParser(description="Creative Coach evaluation suite")
    parser.add_argument("--adapter", default="baseline", choices=sorted(ADAPTERS))
    parser.add_argument("--source", default=None, choices=["production", "adversarial"],
                        help="run only one half of the suite")
    parser.add_argument("--json", default=None, help="also write the full report here")
    parser.add_argument("--strict", action="store_true",
                        help="also fail when a soft check is under target")
    parser.add_argument("--offline", action="store_true",
                        help="use the frozen fixtures: no database, no API key, no spend")
    parser.add_argument("--baseline", default=None,
                        help="a previous --json report; fails when a passing case regresses")
    args = parser.parse_args()

    summary = asyncio.run(run(args.adapter, args.source, offline=args.offline))
    ok = report(summary, strict=args.strict)
    if args.baseline:
        ok = compare(summary, args.baseline) and ok
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"  full report: {args.json}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
