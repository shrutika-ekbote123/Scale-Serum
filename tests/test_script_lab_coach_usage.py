"""Creative Coach phase 8: what the turns recorded, and when to worry about it.

No database. The usage summary is a pure function over stored counter rows, so
it is tested as one.

The thing being protected here is that the bill and the health travel together.
A month that got cheaper because half its turns fell back to the template is not
a saving, and a report that shows the saving without the cause is worse than no
report at all.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from script_lab_coach import usage as usage_mod  # noqa: E402


def turn(**over):
    row = {"_id": "r1", "brand_id": "b1", "test_id": "t1", "day": "2026-09-25",
           "created_at": "2026-09-25T09:00:00Z", "intent": "explain",
           "routed_by": "matched", "brand_brain_tier": "A", "grounded": True,
           "fallback": False, "fallback_reason": None, "confidence": "high",
           "latency_ms": 2000, "model": "gemini-flash-latest",
           "prompt_version": "coach_p1",
           "usage": {"input_tokens": 100, "output_tokens": 20},
           "cost": {"usd": 0.001, "inr": 0.088}}
    row.update(over)
    return row


def many(n, **over):
    return [turn(_id=f"r{i}", **over) for i in range(n)]


def summarize(docs):
    return usage_mod.summarize(docs, brand_id="b1", start="2026-09-01", end="2026-09-30")


# ------------------------------------------------------------------- totals
def test_tokens_and_cost_add_up():
    out = summarize(many(3))
    assert out["totals"]["turns"] == 3
    assert out["totals"]["total_tokens"] == 360
    assert out["totals"]["cost_usd"] == pytest.approx(0.003)


def test_a_fallback_turn_costs_nothing_but_is_still_a_turn():
    out = summarize(many(19) + [turn(_id="x", fallback=True, usage={}, cost=None)])
    assert out["totals"]["turns"] == 20
    assert out["totals"]["llm_calls"] == 19
    assert out["health"]["fallback_rate"] == pytest.approx(0.05)


def test_the_breakdowns_split_by_what_a_reader_would_ask():
    out = summarize([turn(intent="explain", routed_by="matched", brand_brain_tier="A"),
                     turn(_id="r2", intent="scale", routed_by="model",
                          brand_brain_tier="C")])
    assert set(out["per_intent"]) == {"explain", "scale"}
    assert set(out["per_route"]) == {"matched", "model"}
    assert set(out["per_brand_brain_tier"]) == {"A", "C"}


def test_fallback_reasons_are_counted_so_the_cause_is_visible():
    docs = many(18) + [turn(_id="a", fallback=True, fallback_reason="foreign_brand"),
                       turn(_id="b", fallback=True, fallback_reason="foreign_brand"),
                       turn(_id="c", fallback=True, fallback_reason="llm_unavailable")]
    out = summarize(docs)
    assert out["fallback_reasons"] == {"foreign_brand": 2, "llm_unavailable": 1}


def test_latency_percentiles_come_from_the_turns():
    out = summarize([turn(_id=f"r{i}", latency_ms=ms)
                     for i, ms in enumerate([1000, 2000, 3000, 30000])])
    assert out["health"]["latency_p50_ms"] == 2000
    assert out["health"]["latency_p95_ms"] == 30000


# ------------------------------------------------------------------ alerts
def test_nothing_fires_on_a_quiet_day():
    """One fallback out of two turns is not a 100% fallback rate worth waking
    anyone for, and an alert that cries wolf is one people mute."""
    out = summarize([turn(), turn(_id="r2", fallback=True)])
    assert out["alerts"] == [] and out["status"] == "ok"


def test_an_ungrounded_answer_raises_an_alert():
    docs = many(99) + [turn(_id="bad", grounded=False)]
    out = summarize(docs)
    signals = {a["signal"]: a for a in out["alerts"]}
    assert "ungrounded_rate" in signals
    assert signals["ungrounded_rate"]["severity"] == "alert"
    assert out["status"] == "alert"


def test_a_healthy_month_is_ok():
    out = summarize(many(50))
    assert out["alerts"] == [] and out["status"] == "ok"


def test_slow_turns_warn_rather_than_alert():
    out = summarize(many(30, latency_ms=20_000))
    signals = {a["signal"]: a["severity"] for a in out["alerts"]}
    assert signals.get("latency_p95_ms") == "warn"
    assert out["status"] == "warn"


def test_the_classifier_taking_over_is_a_product_signal():
    """Mostly-matched traffic is the design working. A collapse means the phrase
    patterns have stopped fitting what people actually ask."""
    out = summarize(many(30, routed_by="model"))
    assert "model_routed_rate" in {a["signal"] for a in out["alerts"]}


def test_every_alert_says_what_it_means():
    out = summarize(many(99) + [turn(_id="bad", grounded=False)])
    for alert in out["alerts"]:
        assert alert["note"] and alert["threshold"] is not None
        assert alert["severity"] in ("warn", "alert")


# ---------------------------------------------------------------- feedback
def test_thumbs_are_counted_and_reasons_kept():
    docs = many(18)
    docs.append(turn(_id="u", feedback={"rating": "up"}))
    docs.append(turn(_id="d", intent="scale",
                     feedback={"rating": "down", "reason": "it ignored my question"}))
    out = summarize(docs)
    assert out["feedback"]["up"] == 1 and out["feedback"]["down"] == 1
    assert out["feedback"]["rated"] == 2 and out["feedback"]["unrated"] == 18
    assert out["feedback"]["down_reasons"][0]["reason"] == "it ignored my question"
    assert out["feedback"]["down_reasons"][0]["intent"] == "scale"


def test_the_thumbs_down_rate_is_of_rated_turns_not_all_turns():
    """Most turns are never rated. Dividing by all of them would hide a problem
    the people who did bother to answer are telling you about."""
    docs = many(38)
    docs += [turn(_id="d1", feedback={"rating": "down"}),
             turn(_id="d2", feedback={"rating": "down"})]
    out = summarize(docs)
    assert out["health"]["thumbs_down_rate"] == 1.0
    assert "thumbs_down_rate" in {a["signal"] for a in out["alerts"]}


def test_unrated_turns_do_not_count_against_the_coach():
    out = summarize(many(40))
    assert out["health"]["thumbs_down_rate"] == 0.0
    assert "thumbs_down_rate" not in {a["signal"] for a in out["alerts"]}


# --------------------------------------------------------------- thresholds
def test_thresholds_are_configurable_and_documented():
    for signal in ("ungrounded_rate", "fallback_rate", "latency_p95_ms",
                   "thumbs_down_rate", "model_routed_rate"):
        assert signal in usage_mod.THRESHOLDS


def test_an_empty_range_is_zeroes_not_an_error():
    out = summarize([])
    assert out["totals"]["turns"] == 0 and out["status"] == "ok"
    assert out["health"]["fallback_rate"] == 0.0
