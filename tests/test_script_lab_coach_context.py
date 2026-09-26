"""Creative Coach phase 2: the context builder, the fact set and the starters.

No network and no database. The PostgreSQL loaders are monkeypatched with rows
shaped like the live ones - including the parts of live data that are wrong:
camelCase JSON keys, a review that never completed, an ad id that matches no ad,
and a brand with no Brand Brain.

What these tests are really protecting:
  * a figure the coach may say must be one Python computed;
  * one brand's coaching must never be able to read another brand's data;
  * every missing input must have a defined, non-catastrophic behaviour.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from script_lab_coach import brand_brain as bb  # noqa: E402
from script_lab_coach import context as ctxmod  # noqa: E402
from script_lab_coach import data as data_mod  # noqa: E402
from script_lab_coach import facts as facts_mod  # noqa: E402
from script_lab_coach import intents as intents_mod  # noqa: E402
from script_lab_coach import starters as starters_mod  # noqa: E402
from script_lab_coach.context import BrandMismatch  # noqa: E402
from script_lab_coach.context import CreativeCoachContextBuilder  # noqa: E402
# Imported under another name: pytest tries to collect anything called Test*.
from script_lab_coach.context import TestNotFound as UnknownTest  # noqa: E402

DI = "bee79bff-d5f6-4220-a7d4-7a04bf173e59"
OTHER = "ea8c5046-ae40-41cf-957f-de7b35cbfb17"
TEST_ID = "05cc1f26-3527-48b4-99bc-c748963d44ab"


def run(coro):
    return asyncio.run(coro)


async def _run_sync(fn, *a, **kw):
    return fn(*a, **kw)


# --------------------------------------------------------------------- doubles
def a_test(**over):
    """A stored test, shaped like a real row (already key-normalised)."""
    row = {
        "test_id": TEST_ID, "brand_id": DI, "ad_number": 53, "version": 1, "label": "v1",
        "ad_name": "1411_PGLI_Dubai", "source_ad_id": "120251070761150057",
        "source_platform": "meta", "funnel_stage": "cold", "marketing_angle": "original",
        "script_text": "some copy", "score": 82, "attention": 89, "resonance": 80,
        "creative": 78, "conversion": 74, "marketing_angle_execution": 81,
        "verdict": "iterate", "verdict_band": "Minor tweaks only",
        "ai_verdict": "Solid, with a soft CTA.",
        "emotional_angle": {"label": "Authority", "status": "ANGLE WORKS", "critique": "-"},
        "context_alignment": {"brand_voice_fit": "Strong", "funnel_stage_fit": "Moderate",
                              "marketing_angle_fit": "Weak"},
        "section_breakdown": [{"section": "Hook", "score": 9, "comment": "-"},
                              {"section": "Call to Action", "score": 4, "comment": "-"}],
        "improvements": [{"title": "Sharpen the CTA", "why_it_matters": "-"}],
        "what_worked": [], "what_to_fix": [], "top_recommendation": "-",
        "ai_fallback": False, "created_by_user_id": "u1", "created_at": None,
    }
    row.update(over)
    return row


def a_brain(filled=9):
    answers = {f: "value" for f in bb.ANSWER_FIELDS[:filled]}
    return bb.BrandBrain(answers=answers, context={"brandName": "DI"},
                         brand_brain_id="bb1", brand_name="DI")


def builder(monkeypatch, *, test=None, brand=None, perf=None, creatives=None,
            competitors=None, versions=None, brain=None, fail=(), others=None):
    """A builder wired to doubles. `fail` names loaders that raise."""
    def loader(name, value):
        def load(*a, **kw):
            if name in fail:
                raise data_mod.DataUnavailable("pretend outage")
            return value
        return load

    monkeypatch.setattr(data_mod, "load_test", loader("test", a_test() if test is None else test))
    monkeypatch.setattr(data_mod, "load_brand", loader(
        "brand", {"id": DI, "name": "DI", "timezone": "Asia/Kolkata", "currency": "INR",
                  "brand_brain_id": "bb1"} if brand is None else brand))
    monkeypatch.setattr(data_mod, "load_other_brand_names", loader(
        "brand", ["Lawttorney", "Kaizen CRM"] if others is None else others))
    monkeypatch.setattr(data_mod, "load_ad_performance", loader("performance", perf))
    monkeypatch.setattr(data_mod, "load_brand_creatives", loader(
        "corpus", creatives if creatives is not None else {"winners": [], "losers": []}))
    monkeypatch.setattr(data_mod, "load_competitor_hooks", loader("corpus", competitors or []))
    monkeypatch.setattr(data_mod, "load_versions", loader("history", versions or []))

    async def load_brain(brand_brain_id, brand_name=None):
        return brain if brain is not None else a_brain()

    return CreativeCoachContextBuilder(run_sync=_run_sync, load_brand_brain=load_brain)


def perf_row(**over):
    row = {"ad_id": "120251070761150057", "window": "30d", "window_days": 30,
           "days_with_data": 22, "first_date": "2026-08-26", "last_date": "2026-09-23",
           "stale": False, "spend": 3100.04, "impressions": 8235, "clicks": 1012,
           "leads": 94, "purchases": 0, "ctr_pct": 12.29, "cpm": 376.45, "cpc": 3.06,
           "cpl": 32.98, "effective_status": "ACTIVE"}
    row.update(over)
    return row


# ------------------------------------------------------------------ retrieval
def test_cheap_intents_read_nothing_beyond_the_test(monkeypatch):
    """"Why is my hook weak?" must not pay for ad insights or a corpus scan."""
    calls = []
    b = builder(monkeypatch, perf=perf_row())
    for name in ("load_ad_performance", "load_brand_creatives", "load_versions"):
        original = getattr(data_mod, name)
        monkeypatch.setattr(data_mod, name,
                            lambda *a, _n=name, _o=original, **kw: (calls.append(_n), _o(*a, **kw))[1])

    ctx = run(b.build(test_id=TEST_ID, intent_name="explain", brand_id=DI))
    assert calls == []
    assert ctx.facts  # but it still has the score card


def test_each_intent_loads_only_its_declared_tiers(monkeypatch):
    for intent, expected in (
            ("performance", {"load_ad_performance"}),
            ("improve", {"load_brand_creatives", "load_competitor_hooks"}),
            # `compare` also reads the corpus on purpose: no test in the system
            # has an earlier version, so a compare limited to history could
            # never answer. The corpus gives it the brand's own best creative to
            # compare against instead.
            ("compare", {"load_versions", "load_ad_performance",
                         "load_brand_creatives", "load_competitor_hooks"})):
        calls = set()
        b = builder(monkeypatch, perf=perf_row())
        for name in ("load_ad_performance", "load_brand_creatives",
                     "load_competitor_hooks", "load_versions"):
            original = getattr(data_mod, name)
            monkeypatch.setattr(data_mod, name,
                                lambda *a, _n=name, _o=original, **kw: (calls.add(_n), _o(*a, **kw))[1])
        run(b.build(test_id=TEST_ID, intent_name=intent, brand_id=DI))
        assert calls == expected, intent


def test_extra_tiers_let_starters_see_capabilities(monkeypatch):
    b = builder(monkeypatch, perf=perf_row(), versions=[{"test_id": "t0", "version": 1,
                                                         "label": "v1", "score": 61,
                                                         "ai_fallback": False}])
    ctx = run(b.build(test_id=TEST_ID, brand_id=DI,
                      extra_tiers=(intents_mod.PERFORMANCE, intents_mod.HISTORY)))
    assert ctx.capabilities() == {"performance": True, "compare": True, "brand_fit": True}


def test_second_turn_on_the_same_test_hits_the_cache(monkeypatch):
    hits = []
    b = builder(monkeypatch)
    original = data_mod.load_test
    monkeypatch.setattr(data_mod, "load_test",
                        lambda *a, **kw: (hits.append(1), original(*a, **kw))[1])
    run(b.build(test_id=TEST_ID, intent_name="explain", brand_id=DI))
    run(b.build(test_id=TEST_ID, intent_name="diagnose", brand_id=DI))
    assert len(hits) == 1


# ----------------------------------------------------------------------- gates
def test_another_brand_cannot_read_this_test(monkeypatch):
    """The gate that stops confident, plausible, wrong cross-brand coaching."""
    b = builder(monkeypatch)
    with pytest.raises(BrandMismatch):
        run(b.build(test_id=TEST_ID, intent_name="explain", brand_id=OTHER))


def test_unknown_test_is_not_found(monkeypatch):
    b = builder(monkeypatch, test=None)
    monkeypatch.setattr(data_mod, "load_test", lambda *a, **kw: None)
    with pytest.raises(UnknownTest):
        run(b.build(test_id="nope", intent_name="explain"))


def test_a_tier_outage_degrades_and_is_named(monkeypatch):
    b = builder(monkeypatch, fail=("performance",))
    ctx = run(b.build(test_id=TEST_ID, intent_name="performance", brand_id=DI))
    assert ctx.performance is None
    assert "performance" in ctx.degraded
    assert ctx.capabilities()["performance"] is False


# ----------------------------------------------------------------------- facts
def test_the_fact_set_is_the_only_source_of_numbers(monkeypatch):
    b = builder(monkeypatch, perf=perf_row())
    ctx = run(b.build(test_id=TEST_ID, intent_name="performance", brand_id=DI))
    keys = {f.key for f in ctx.facts}
    assert {"overall_score", "hook_score", "call_to_action_score", "ctr_pct_30d"} <= keys
    allowed = ctx.allowed_numbers()
    assert 82 in allowed and 9 in allowed and round(12.29, 4) in allowed
    # a plausible-looking number nobody measured is not admissible
    assert 4.7 not in allowed


def test_percentages_are_admissible_in_both_forms():
    f = [facts_mod.Fact("ctr_pct_30d", 12.29, "%")]
    allowed = facts_mod.allowed_numbers(f)
    assert round(12.29, 4) in allowed and round(0.1229, 6) in allowed


def test_per_ad_revenue_is_never_a_fact(monkeypatch):
    """Payments carry no ad id, so any per-ad ROAS would be invented."""
    b = builder(monkeypatch, perf=perf_row(revenue=99999))
    ctx = run(b.build(test_id=TEST_ID, intent_name="performance", brand_id=DI))
    keys = " ".join(f.key for f in ctx.facts)
    assert "revenue" not in keys and "roas" not in keys


def test_a_failed_review_contributes_no_scores(monkeypatch):
    """Its stored 50s are placeholders. Quoting them defends a meaningless number."""
    b = builder(monkeypatch, test=a_test(ai_fallback=True, score=50))
    ctx = run(b.build(test_id=TEST_ID, intent_name="explain", brand_id=DI))
    assert ctx.review_failed
    assert [f.key for f in ctx.facts] == ["review_completed"]
    assert 50 not in ctx.allowed_numbers()


def test_a_stale_ad_says_so_rather_than_implying_it_is_live(monkeypatch):
    b = builder(monkeypatch, perf=perf_row(window="lifetime", window_days=None, stale=True,
                                           days_with_data=4, last_date="2026-06-27"))
    ctx = run(b.build(test_id=TEST_ID, intent_name="performance", brand_id=DI))
    keys = {f.key: f.value for f in ctx.facts}
    assert keys["currently_delivering"] is False
    assert keys["last_delivery_date"] == "2026-06-27"
    assert ctx.performance_is_thin


def test_a_failed_previous_version_is_not_a_baseline(monkeypatch):
    b = builder(monkeypatch, versions=[
        {"test_id": "t0", "version": 1, "label": "v1", "score": 50, "ai_fallback": True},
        {"test_id": "t1", "version": 2, "label": "v2", "score": 61, "ai_fallback": False}])
    ctx = run(b.build(test_id=TEST_ID, intent_name="compare", brand_id=DI))
    previous = [f for f in ctx.facts if f.key.startswith("previous_score")]
    assert [f.value for f in previous] == [61]


# --------------------------------------------------------------------- evidence
def test_evidence_is_built_from_what_was_read(monkeypatch):
    b = builder(monkeypatch, perf=perf_row(),
                creatives={"winners": [{"ad_id": "a1", "score": 86.0}], "losers": []},
                competitors=[{"ad_archive_id": "c1", "est_run_days": 134}])
    ctx = run(b.build(test_id=TEST_ID, intent_name="performance", brand_id=DI))
    kinds = {e["type"] for e in ctx.evidence()}
    assert "ad_performance" in kinds and "brand_brain" in kinds
    # the corpus was not requested by this intent, so it cannot be cited
    assert "competitor_ad" not in kinds


# ------------------------------------------------------------------ brand brain
@pytest.mark.parametrize("filled,tier", [(9, "A"), (7, "A"), (6, "B"), (3, "B"),
                                         (2, "C"), (0, "C")])
def test_completeness_is_graded_not_boolean(filled, tier):
    assert a_brain(filled).tier == tier


def test_tier_c_cannot_judge_brand_fit_and_says_what_it_lacks():
    brain = bb.BrandBrain()
    assert brain.tier == "C" and not brain.used and not brain.can_judge_brand_fit
    assert "no Brand Brain on file" in brain.prompt_block()


def test_a_partial_brand_brain_names_its_gaps():
    block = a_brain(4).prompt_block()
    assert "NOT ON FILE" in block and "competitors" in block


def test_compare_without_history_still_has_the_brand_corpus_to_use(monkeypatch):
    """Otherwise `compare` is an intent that can never answer: nothing in the
    system increments `version`, so no test has an earlier one."""
    b = builder(monkeypatch, versions=[],
                creatives={"winners": [{"ad_id": "a1", "score": 86.0}], "losers": []})
    ctx = run(b.build(test_id=TEST_ID, intent_name="compare", brand_id=DI))
    assert ctx.has_history is False
    assert ctx.creatives.get("winners")
    assert any(f.key == "best_brand_creative_score" for f in ctx.facts)


def test_the_context_carries_the_other_brands_for_the_leak_check(monkeypatch):
    """Read per brand rather than configured per deployment: a brand added today
    has to be protected today. An empty list silently disables the check that
    caught the coach advertising another company to this brand's user."""
    b = builder(monkeypatch, others=["Lawttorney", "GlowLab"])
    ctx = run(b.build(test_id=TEST_ID, intent_name="explain", brand_id=DI))
    assert ctx.other_brands == ("Lawttorney", "GlowLab")


def test_a_brand_lookup_outage_does_not_fail_the_turn(monkeypatch):
    b = builder(monkeypatch, fail=("brand",))
    ctx = run(b.build(test_id=TEST_ID, intent_name="explain", brand_id=DI))
    assert ctx.other_brands == () and ctx.brain.tier == "C"


def test_a_brand_with_no_row_is_tier_c_not_an_error(monkeypatch):
    """133 stored tests point at brands that do not exist. That is a supported path."""
    b = builder(monkeypatch, brand=None)
    monkeypatch.setattr(data_mod, "load_brand", lambda *a, **kw: None)
    ctx = run(b.build(test_id=TEST_ID, intent_name="brand_fit", brand_id=DI))
    assert ctx.brain.tier == "C"
    assert ctx.capabilities()["brand_fit"] is False


def test_mongo_outage_degrades_to_craft_mode(monkeypatch):
    """A Brand Brain that cannot be read is tier C, never an error in the panel."""
    async def explode(*a, **kw):
        raise RuntimeError("mongo down")

    b = builder(monkeypatch)
    b._load_brain = explode
    ctx = run(b.build(test_id=TEST_ID, intent_name="explain", brand_id=DI))
    assert ctx.brain.tier == "C" and ctx.facts

    class Boom:
        async def find_one(self, *a, **kw):
            raise RuntimeError("mongo down")

    brain = run(bb.loader(Boom())("bb1", "DI"))
    assert brain.tier == "C"


# ----------------------------------------------------------------- normalising
def test_camelcase_json_columns_are_normalised_on_read():
    raw = {"brandVoiceFit": "Strong", "improvements": [{"whyItMatters": "x",
                                                        "suggestedRewrite": "y"}]}
    out = data_mod._normalise(raw)
    assert out["brand_voice_fit"] == "Strong"
    assert out["improvements"][0]["why_it_matters"] == "x"
    assert out["improvements"][0]["suggested_rewrite"] == "y"


def test_json_columns_survive_being_stored_as_text():
    assert data_mod._json('{"brandVoiceFit": "Weak"}', {}) == {"brand_voice_fit": "Weak"}
    assert data_mod._json("not json", {"d": 1}) == {"d": 1}
    assert data_mod._json(None, []) == []


# -------------------------------------------------------------------- starters
def test_starters_point_at_the_weakest_section(monkeypatch):
    b = builder(monkeypatch)
    ctx = run(b.build(test_id=TEST_ID, brand_id=DI))
    out = starters_mod.build(ctx)
    assert "Call to Action" in out["starters"][0]["label"]
    assert out["starters"][0]["intent"] == "explain"
    assert "82/100" in out["intro"]["text"]
    assert "marketing angle" in out["intro"]["text"]  # the Weak alignment


def test_a_failing_script_is_not_asked_whether_it_will_scale(monkeypatch):
    b = builder(monkeypatch, test=a_test(score=22), perf=perf_row())
    ctx = run(b.build(test_id=TEST_ID, brand_id=DI, extra_tiers=(intents_mod.PERFORMANCE,)))
    intents = [c["intent"] for c in starters_mod.build(ctx)["starters"]]
    assert "scale" not in intents and "rewrite" in intents


def test_a_failed_review_gets_an_honest_intro(monkeypatch):
    b = builder(monkeypatch, test=a_test(ai_fallback=True, score=50))
    ctx = run(b.build(test_id=TEST_ID, brand_id=DI))
    out = starters_mod.build(ctx)
    assert "did not complete" in out["intro"]["text"]
    assert "50" not in out["intro"]["text"]


# --------------------------------------------------------------------- intents
def test_unknown_intent_falls_back_rather_than_crashing_a_thread():
    assert intents_mod.get("make_it_viral").name == intents_mod.DEFAULT.name
    assert intents_mod.get(None).name == intents_mod.DEFAULT.name


def test_every_intent_in_the_contract_exists():
    contract = {"explain", "diagnose", "prioritize", "improve", "rewrite", "compare",
                "performance", "scale", "brand_fit", "out_of_scope"}
    assert contract == set(intents_mod.NAMES)
