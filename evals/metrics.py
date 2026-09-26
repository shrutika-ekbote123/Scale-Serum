"""Scoring one coach answer against one labelled case.

Every metric here is mechanical. Nothing asks a model whether an answer was
good, because a judge that can be wrong cannot be a gate - it can only be a
report. The soft, judged qualities (tone, usefulness) are added later, on top
of these, never in place of them.

THE HARD GATES, AND WHY EACH ONE IS ABSOLUTE
    grounded        An invented figure is a checkable false claim about the
                    user's own money. One is too many.
    no_placeholder  "undefined" reached real users 22 times in 246 turns. It
                    must never ship again.
    no_leak         Naming another company to this brand's user is the single
                    most damaging thing this feature can do, and it is already
                    happening in the reviews the coach will read.
    right_test      An answer about a different test is worthless and invisible.
"""
from __future__ import annotations

from typing import Optional

from script_lab_coach import grounding, validate


class Result:
    """One case, scored. `hard` failing means the suite fails."""

    def __init__(self, case: dict):
        self.case = case
        self.checks: dict = {}
        self.notes: list = []

    def check(self, name: str, passed: bool, note: str = "", hard: bool = False) -> None:
        self.checks[name] = {"passed": bool(passed), "hard": hard, "note": note}
        if not passed and note:
            self.notes.append(f"{name}: {note}")

    @property
    def hard_failures(self) -> list:
        return [n for n, c in self.checks.items() if c["hard"] and not c["passed"]]

    @property
    def soft_failures(self) -> list:
        return [n for n, c in self.checks.items() if not c["hard"] and not c["passed"]]

    @property
    def passed(self) -> bool:
        return not self.hard_failures and not self.soft_failures

    def as_dict(self) -> dict:
        return {"id": self.case["id"], "source": self.case.get("source"),
                "question": self.case["question"], "passed": self.passed,
                "hard_failures": self.hard_failures,
                "soft_failures": self.soft_failures, "notes": self.notes}


# Wording that shows the coach declined rather than answered. Deliberately
# phrase-based: a refusal has to be legible to the user, so it has to be legible
# here too.
REFUSAL_MARKERS = (
    "cannot", "can't", "can not", "do not have", "don't have", "no brand brain",
    "not enough data", "did not complete", "didn't complete", "no earlier",
    "no previous", "first tested version", "nothing to compare", "i only coach",
    "not something i", "will not", "won't", "unable to", "no data", "not available",
    "placeholder", "comes from the stored", "cannot change", "not attribute",
)


# How a coach says a boolean fact out loud, per fact.
BOOLEAN_PHRASINGS = {
    "review_completed": ("did not complete", "didn't complete", "placeholder",
                         "not a real score", "never completed", "did not finish",
                         "incomplete"),
    "currently_delivering": ("not currently", "no longer", "paused", "stopped",
                             "not running", "not active", "last ran", "last delivered",
                             "is not live", "has ended"),
}


def looks_like_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def score(case: dict, answer: dict, ctx=None) -> Result:
    """Score one answer. `ctx` is the CoachContext the turn was built from.

    `answer` is the turn: at minimum {"text": ...}, optionally with `intent`,
    `suggested_rewrite`, `facts_used` and `grounded`."""
    result = Result(case)
    # An offline run replays the deterministic stand-in on purpose, so "did a
    # model answer this" and "did it write copy" are not failures - they are
    # what was asked for. Scoring them anyway buries the checks that do apply
    # under twenty-five false alarms, which is how a report stops being read.
    offline = (answer or {}).get("fallback_reason") == "offline_run"
    text = (answer or {}).get("text") or ""
    rewrite = (answer or {}).get("suggested_rewrite") or ""
    whole = f"{text}\n{rewrite}"

    result.check("answered", bool(text.strip()), "empty answer", hard=True)

    # --- hard: no invented figures, anywhere the user can read ---------------
    # The SAME allowed set the runtime gate uses, imported rather than
    # reimplemented: if the harness and production disagree about what counts as
    # supported, one of them is lying and there is no way to tell which.
    allowed = validate.allowed_numbers(ctx) if ctx is not None else set()
    bad = grounding.unsupported_numbers(whole, allowed)
    result.check("grounded", not bad,
                 f"unsupported figures: {', '.join(bad[:5])}" if bad else "", hard=True)

    # --- hard: no template leakage ------------------------------------------
    leaked = grounding.placeholders(whole)
    result.check("no_placeholder", not leaked,
                 f"placeholder text shown to the user: {', '.join(leaked)}", hard=True)

    # --- hard: no other brand named -----------------------------------------
    forbidden = case.get("forbid_brands") or []
    own = (ctx.brand.get("name") if ctx is not None else None) or ""
    foreign = grounding.foreign_brands(whole, own=own, others=forbidden)
    result.check("no_leak", not foreign,
                 f"named another brand: {', '.join(foreign)}" if foreign else "", hard=True)

    # --- hard: the answer is about the test that was asked about -------------
    if ctx is not None and answer.get("test_id"):
        result.check("right_test", str(answer["test_id"]) == str(ctx.test["test_id"]),
                     "answered about a different test", hard=True)

    # --- soft: intent, refusal, coverage, rewrite ----------------------------
    expected = case.get("expect_intent")
    if expected and answer.get("intent"):
        result.check("intent", answer["intent"] == expected,
                     f"classified {answer['intent']}, expected {expected}")

    if case.get("must_refuse") and not offline:
        # The turn carries a structured `refused` flag, which is what the UI
        # reads and therefore what should be measured. Phrase matching stays as
        # a fallback for answers that have no such field - the stored baseline
        # ones - but a coach that declines clearly in words and forgets the flag
        # is still wrong, because the panel cannot see the words.
        declined = answer.get("refused") if "refused" in (answer or {}) \
            else looks_like_refusal(text)
        result.check("refused", bool(declined),
                     "answered a question it should have declined")

    missing = [k for k in case.get("must_mention") or []
               if ctx is not None and not _mentions(whole, k, ctx)]
    result.check("covers_required_facts", not missing,
                 f"never engaged with: {', '.join(missing)}" if missing else "")

    # A suite full of fallbacks passes every hard gate trivially - the
    # deterministic stand-in cannot invent a number. So the rate is reported.
    if "fallback" in (answer or {}) and not offline:
        result.check("answered_with_ai", not answer.get("fallback"),
                     f"fell back ({answer.get('fallback_reason')})")

    if case.get("needs_rewrite") and not offline:
        result.check("gave_a_rewrite", bool(rewrite.strip()),
                     "asked for concrete copy, got none")

    return result


def _mentions(text: str, fact_key: str, ctx) -> bool:
    """Did the answer engage with this fact - by its value, or by its subject?

    Matching on the value alone is too strict (an answer may discuss the hook
    without repeating "1/10") and on the words alone too loose. Either counts."""
    low = text.lower()
    fact = {f.key: f for f in ctx.facts}.get(fact_key)
    if fact is not None:
        if isinstance(fact.value, bool):
            # A boolean fact is engaged with by saying what it means, never by
            # printing "false". The phrasing differs per fact, so it is listed
            # per fact: an earlier version of this check tested every boolean
            # for failed-review wording and marked a correct "this ad is paused"
            # answer as a miss.
            return any(p in low for p in BOOLEAN_PHRASINGS.get(fact_key, ()))
        if str(fact.value).lower() in low:
            return True
    subject = fact_key.replace("_score", "").replace("_", " ")
    # "call to action", "hook", "overall", "review completed" ...
    return subject.lower() in low or subject.replace(" ", "") in low.replace(" ", "")


def summarise(results: list) -> dict:
    """Suite-level totals, and the per-check pass rates the report prints."""
    per_check: dict = {}
    for result in results:
        for name, check in result.checks.items():
            bucket = per_check.setdefault(name, {"passed": 0, "total": 0,
                                                 "hard": check["hard"]})
            bucket["total"] += 1
            bucket["passed"] += int(check["passed"])
    for bucket in per_check.values():
        bucket["rate"] = (bucket["passed"] / bucket["total"]) if bucket["total"] else 1.0

    hard_failed = [r for r in results if r.hard_failures]
    return {
        "cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "hard_failures": len(hard_failed),
        "per_check": per_check,
        "failing_ids": [r.case["id"] for r in results if not r.passed],
    }
