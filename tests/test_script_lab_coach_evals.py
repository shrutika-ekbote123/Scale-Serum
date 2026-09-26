"""The evaluation harness, tested.

A scorer is code, and code that grades other code has to be right or it is
worse than nothing: a gate that cries wolf is a gate somebody switches off.
The first run of this suite produced three false alarms - scale denominators
("34/100"), numbers inside ad names ("1011_ICDP_India"), and "nan" matched
inside "Andromeda" - and each of them is pinned here.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evals import metrics  # noqa: E402
from script_lab_coach import facts as facts_mod  # noqa: E402
from script_lab_coach import grounding  # noqa: E402


class FakeContext:
    """Just enough CoachContext for the scorer."""

    def __init__(self, facts=(), test=None, brand=None):
        self.facts = list(facts)
        self.test = test or {"test_id": "t1", "script_text": "", "section_breakdown": [],
                             "improvements": [], "ai_verdict": "", "ad_name": "", "label": ""}
        self.brand = brand or {"name": "DI"}

    def allowed_numbers(self):
        return facts_mod.allowed_numbers(self.facts)


def a_ctx(**over):
    f = [facts_mod.Fact("overall_score", 34, "/100"),
         facts_mod.Fact("hook_score", 1, "/10"),
         facts_mod.Fact("ctr_pct_30d", 12.29, "%")]
    ctx = FakeContext(facts=f)
    ctx.test.update(over)
    return ctx


def case(**over):
    base = {"id": "c1", "source": "test", "question": "why?", "expect_intent": None,
            "must_mention": [], "must_refuse": False}
    base.update(over)
    return base


# ------------------------------------------------------------------- numbers
def test_an_invented_figure_fails_the_hard_gate():
    r = metrics.score(case(), {"text": "Your CTR is 4.7%."}, a_ctx())
    assert "grounded" in r.hard_failures


def test_a_computed_figure_passes():
    r = metrics.score(case(), {"text": "Your CTR is 12.29% and the hook scored 1."}, a_ctx())
    assert "grounded" not in r.hard_failures


def test_a_scale_denominator_is_not_a_claim():
    """"34/100" quotes the scale, it does not assert a new measurement."""
    r = metrics.score(case(), {"text": "It scores 34/100 and the hook is 1/10."}, a_ctx())
    assert "grounded" not in r.hard_failures


def test_numbers_inside_the_ad_name_are_quotable():
    ctx = a_ctx(ad_name="1011_ICDP_India_UGC based_Cold")
    r = metrics.score(case(), {"text": 'Tested "1011_ICDP_India_UGC based_Cold".'}, ctx)
    assert "grounded" not in r.hard_failures


def test_numbers_quoted_from_the_review_are_allowed():
    ctx = a_ctx(section_breakdown=[{"section": "Hook", "comment": "loses 9 points"}])
    r = metrics.score(case(), {"text": "The review noted it loses 9 points."}, ctx)
    assert "grounded" not in r.hard_failures


def test_rhetorical_small_numbers_are_allowed_without_a_unit():
    r = metrics.score(case(), {"text": "Fix two things in the first 3 seconds."}, a_ctx())
    assert "grounded" not in r.hard_failures


def test_the_same_small_number_with_a_unit_is_a_claim():
    r = metrics.score(case(), {"text": "Your CTR is 3%."}, a_ctx())
    assert "grounded" in r.hard_failures


def test_an_invented_figure_in_a_rewrite_fails_too():
    """A rewrite is ad copy: an invented statistic becomes an ad claim."""
    r = metrics.score(case(), {"text": "Try this.",
                               "suggested_rewrite": "Save 12 hours every week."}, a_ctx())
    assert "grounded" in r.hard_failures


# -------------------------------------------------------------- placeholders
def test_a_template_placeholder_fails_the_hard_gate():
    r = metrics.score(case(), {"text": "Change this first: undefined"}, a_ctx())
    assert "no_placeholder" in r.hard_failures


def test_placeholder_matching_does_not_fire_inside_ordinary_words():
    for word in ("Andromeda", "finance", "nullify", "Nantucket"):
        assert grounding.placeholders(f"The {word} script is fine.") == [], word


# --------------------------------------------------------------- brand leaks
def test_naming_another_brand_fails_the_hard_gate():
    r = metrics.score(case(forbid_brands=["Lawttorney"]),
                      {"text": "Lawttorney's audience is legal professionals."}, a_ctx())
    assert "no_leak" in r.hard_failures


def test_the_brands_own_name_is_not_a_leak():
    ctx = a_ctx()
    ctx.brand = {"name": "Lawttorney"}
    r = metrics.score(case(forbid_brands=["Lawttorney"]),
                      {"text": "Lawttorney's audience is legal professionals."}, ctx)
    assert "no_leak" not in r.hard_failures


def test_very_short_brand_names_are_not_matched():
    """"DI" and "T" are real brand names and would match almost any sentence."""
    assert grounding.foreign_brands("It did not", own="X", others=["DI", "T"]) == []


# -------------------------------------------------------------------- answers
def test_an_answer_about_another_test_fails():
    r = metrics.score(case(), {"text": "fine", "test_id": "other"}, a_ctx())
    assert "right_test" in r.hard_failures


def test_a_refusal_is_recognised():
    r = metrics.score(case(must_refuse=True),
                      {"text": "I cannot attribute revenue to a single ad."}, a_ctx())
    assert "refused" not in r.soft_failures


def test_answering_something_that_needed_a_refusal_is_flagged():
    r = metrics.score(case(must_refuse=True), {"text": "Your ROAS is fine."}, a_ctx())
    assert "refused" in r.soft_failures


def test_required_facts_can_be_met_by_subject_or_by_value():
    by_value = metrics.score(case(must_mention=["hook_score"]),
                             {"text": "It scored 1 there."}, a_ctx())
    by_subject = metrics.score(case(must_mention=["hook_score"]),
                               {"text": "The hook is the problem."}, a_ctx())
    assert "covers_required_facts" not in by_value.soft_failures
    assert "covers_required_facts" not in by_subject.soft_failures


def test_a_missing_required_fact_is_flagged():
    r = metrics.score(case(must_mention=["hook_score"]),
                      {"text": "The pacing is fine."}, a_ctx())
    assert "covers_required_facts" in r.soft_failures


def test_an_empty_answer_fails_hard():
    assert "answered" in metrics.score(case(), {"text": ""}, a_ctx()).hard_failures


# -------------------------------------------------------------------- summary
def test_the_summary_counts_hard_and_soft_separately():
    results = [metrics.score(case(), {"text": "Your CTR is 4.7%."}, a_ctx()),
               metrics.score(case(), {"text": "The hook is weak."}, a_ctx())]
    out = metrics.summarise(results)
    assert out["cases"] == 2 and out["hard_failures"] == 1
    assert out["per_check"]["grounded"]["hard"] is True
    assert out["per_check"]["grounded"]["rate"] == 0.5
