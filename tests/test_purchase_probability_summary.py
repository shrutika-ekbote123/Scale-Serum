"""Tests for the narrative summary (`explain.summary_block`).

Data policy, same as the rest of the suite: real leads, read-only Postgres,
nothing written anywhere. The pure-function tests need no database at all.

What these assertions are actually protecting:

  * The summary explains `purchase_probability`. It must never quote the
    `lead_priority` score, which is a different number on a different scale.
  * It must introduce no number of its own. Every figure in the prose has to be
    one the response already carries.
  * It must stay readable by a non-technical user: no log-odds, coefficients,
    contributions, percentiles, deciles or internal field names, and no claim
    that a lead will or will not buy.
  * It must keep its shape when there is nothing to say, so the UI branches on
    `available` and never on whether a key exists.
"""
from __future__ import annotations

import os
import re
import sys
import uuid

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO, ".env"))

import purchase_probability_model as ppm  # noqa: E402
from purchase_probability_model import explain as ex  # noqa: E402

# Causal or certainty claims a logistic regression on observational data cannot
# support, plus the machine-learning vocabulary this block exists to avoid.
FORBIDDEN_CLAIMS = ("caused", "causes", "because of x", "guarantees",
                    "will buy", "will not buy", "ensures", "certain to")
FORBIDDEN_JARGON = ("log-odds", "log odds", "coefficient", "contribution",
                    "percentile", "decile", "feature", "model", "calibrat",
                    "top_factors", "lead_priority", "affects", "logistic")

SUMMARY_KEYS = {"available", "headline", "text", "sentences", "positive",
                "negative", "standing", "counts", "note", "basis"}
COUNT_KEYS = {"total", "positive", "negative"}

# ~3am, so f_hour_time_pattern resolves to the "night" window.
FEATURES = {"f_hour_sin": 0.707107, "f_hour_cos": 0.707107}
LEAD_PRIORITY = {"score": 96, "priority": "High", "probability_percent": 2.23}


def f(feature, contribution, value=None, affects="purchase_probability"):
    return {"feature": feature, "contribution": contribution, "value": value,
            "affects": affects}


def why(percent, base=1.09):
    return {"starting_point": {"percent": base}, "result": {"percent": percent}}


MODEL_POS = f("f_locale", 0.500586, "en-US")
MODEL_POS_2 = f("f_seniority", 0.421617, "founder_c_level")
MODEL_NEG = f("f_hour_time_pattern", -0.164086)
MODEL_NEG_2 = f("f_company_len", -0.09, 8)
PRIOR_POS = f("e_high_intent_pages", 0.35, affects="lead_priority")
PRIOR_POS_2 = f("e_recency", 0.53, affects="lead_priority")
PRIOR_NEG = f("b_icp_seniority_fit", -0.30, affects="lead_priority")
PRIOR_NEG_2 = f("b_business_type_email_fit", -0.16, affects="lead_priority")


def summary(factors, w=None, **kw):
    kw.setdefault("lead_priority", LEAD_PRIORITY)
    return ex.summary_block(factors, w or why(1.28), FEATURES, **kw)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def conn():
    psycopg = pytest.importorskip("psycopg")
    try:
        c = psycopg.connect(
            host=os.environ["DB_HOST"], port=int(os.environ.get("DB_PORT", 5432)),
            dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"], connect_timeout=10)
        c.read_only = True
    except Exception as exc:                                   # pragma: no cover
        pytest.skip(f"database unavailable: {exc}")
    yield c
    c.close()


@pytest.fixture(scope="module")
def scorable_lead_id(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.id FROM leads l
            WHERE EXISTS (
                SELECT 1 FROM touchpoint_events te
                WHERE te.lead_id = l.id AND te.type = 'form_submit'
                  AND te.created_at <= l.created_at + INTERVAL '1 hour'
                  AND te.created_at - te.occurred_at <= INTERVAL '1 hour')
            ORDER BY l.created_at LIMIT 1""")
        row = cur.fetchone()
    if not row:
        pytest.skip("no scorable lead found")
    return str(row[0])


# ----------------------------------------------------------- 1: the five cases
def test_mixed_signals_reads_as_reasons_for_and_against():
    s = summary([PRIOR_POS_2, MODEL_POS, PRIOR_POS, PRIOR_NEG, PRIOR_NEG_2,
                 MODEL_NEG_2], why(1.19))
    assert s["text"].startswith("This lead has a 1.19% purchase probability because")
    assert "However," in s["text"]
    assert "baseline" in s["text"]
    assert s["counts"] == {"total": 6, "positive": 3, "negative": 3}


def test_all_positive_has_no_however_clause():
    s = summary([MODEL_POS, MODEL_POS_2, PRIOR_POS], why(4.8))
    assert "However" not in s["text"]
    assert "well above the typical 1.09% baseline" in s["text"]
    assert s["negative"] == []


def test_all_negative_omits_the_because_clause():
    """With nothing in its favour there is no 'because' to write - the sentence
    states the number and moves to what is holding it back."""
    s = summary([PRIOR_NEG, MODEL_NEG, MODEL_NEG_2], why(0.6))
    assert s["text"].startswith("This lead has a 0.6% purchase probability.")
    assert "because" not in s["text"]
    assert "below the typical 1.09% baseline" in s["text"]
    assert s["positive"] == []


def test_above_baseline_negatives_do_not_contradict_the_standing():
    """'However X, which keeps the probability above the baseline' contradicts
    itself. A lead above the baseline is held back TO a number still above it."""
    s = summary([MODEL_POS, MODEL_POS_2, MODEL_NEG], why(3.0))
    assert "still leaving it well above the typical 1.09% baseline" in s["text"]
    assert "which keeps it well above" not in s["text"]


def test_below_baseline_says_so():
    s = summary([MODEL_POS, MODEL_NEG, MODEL_NEG_2], why(0.4))
    assert "well below the typical 1.09% baseline" in s["text"]
    assert s["standing"] == "well below average"


def test_standing_tracks_the_ratio_not_the_difference():
    """At a 1.09% base rate, half a point higher is a near-doubling."""
    def standing(percent):
        return summary([MODEL_POS], why(percent))["standing"]

    assert standing(1.10) == "around average"
    assert standing(1.60) == "above average"
    assert standing(3.00) == "well above average"
    assert standing(0.70) == "below average"
    assert standing(0.20) == "well below average"


# ------------------------------------------------ 2: it explains the right number
def test_the_lead_priority_score_never_appears():
    """The load-bearing test. `lead_priority.score` is a different number on a
    different scale, and quoting it here would misreport the probability."""
    s = summary([MODEL_POS, PRIOR_POS, PRIOR_NEG], why(1.28),
                lead_priority={"score": 96, "priority": "High"})
    assert "96" not in s["text"]
    assert "out of 100" not in s["text"]
    assert "/100" not in s["text"]
    assert "ranking" not in s["text"].lower()


def test_every_number_in_the_text_appears_in_the_response():
    """The summary restates; it does not compute."""
    s = summary([MODEL_POS, MODEL_NEG, PRIOR_POS, PRIOR_NEG], why(1.28))
    allowed = {"1.28", "1.09"}
    found = set(re.findall(r"\d+(?:\.\d+)?", s["text"]))
    assert found <= allowed, f"unexplained numbers in summary: {found - allowed}"


def test_the_probability_is_quoted_verbatim():
    for percent in (0.52, 1.28, 4.8, 12.0):
        s = summary([MODEL_POS], why(percent))
        assert f"{percent}% purchase probability" in s["text"]
        assert f"{percent}%" in s["headline"]


def test_no_contribution_values_are_exposed():
    """Factor contributions are model signals, not percentages to show a user."""
    s = summary([MODEL_POS, MODEL_NEG, PRIOR_NEG], why(1.28))
    for factor in (MODEL_POS, MODEL_NEG, PRIOR_NEG):
        assert str(abs(factor["contribution"])) not in s["text"]


# --------------------------------------------------------------- 3: the language
def test_no_machine_learning_jargon_reaches_the_user():
    s = summary([MODEL_POS, MODEL_POS_2, MODEL_NEG, PRIOR_POS, PRIOR_NEG],
                why(1.28))
    blob = " ".join([s["text"], s["headline"]] + s["sentences"]).lower()
    for word in FORBIDDEN_JARGON:
        assert word not in blob, f"jargon leaked into the summary: {word!r}"


def test_no_certainty_or_causal_claims():
    s = summary([MODEL_POS, MODEL_NEG, PRIOR_NEG], why(1.28))
    blob = " ".join([s["text"], s["headline"]] + s["sentences"]).lower()
    for word in FORBIDDEN_CLAIMS:
        assert word not in blob, f"overclaiming language in summary: {word!r}"


def test_summary_is_two_or_three_sentences():
    s = summary([MODEL_POS, MODEL_POS_2, PRIOR_POS, PRIOR_NEG, PRIOR_NEG_2,
                 MODEL_NEG], why(1.28))
    assert 1 <= len(s["sentences"]) <= 3
    assert len(s["text"]) < 460, f"summary got long: {len(s['text'])} chars"


def test_at_most_three_reasons_on_each_side():
    """Many factors, short summary. Strongest first - top_factors arrives sorted."""
    s = summary([MODEL_POS, MODEL_POS_2, PRIOR_POS, PRIOR_POS_2,
                 PRIOR_NEG, PRIOR_NEG_2, MODEL_NEG, MODEL_NEG_2], why(1.28))
    assert len(s["positive"]) <= 3
    assert len(s["negative"]) <= 3
    # The strongest positive arrived first, so it must be quoted.
    assert s["positive"][0] == ex._clause(MODEL_POS, FEATURES, ex.load_language())


def test_repeated_reasons_are_not_said_twice():
    """Two factors can share a clause; saying it twice reads as a bug."""
    twin = f("e_high_intent_pages", 0.20, affects="lead_priority")
    s = summary([PRIOR_POS, twin, MODEL_POS], why(1.28))
    assert len(s["positive"]) == len(set(s["positive"]))


def test_clauses_carry_no_internal_commas():
    """Several clauses are joined into one sentence; internal punctuation makes
    the list unreadable."""
    lang = ex.load_language()
    for feature, spec in (lang.get("features") or {}).items():
        for entry in (spec.get("values") or {}).values():
            if entry.get("clause"):
                assert "," not in entry["clause"], f"{feature}: {entry['clause']}"
        for key in ("clause_positive", "clause_negative", "clause_zero"):
            if spec.get(key):
                assert "," not in spec[key], f"{feature}.{key}"
        for clause in (spec.get("clause_windows") or {}).values():
            assert "," not in clause, f"{feature}.clause_windows"
    for feature, spec in (lang.get("signals") or {}).items():
        for key in ("clause_positive", "clause_negative"):
            if spec.get(key):
                assert "," not in spec[key], f"{feature}.{key}"


def test_every_factor_the_response_can_emit_has_a_clause():
    """A factor with no clause is silently dropped from the summary, which is how
    a lead ends up with an explanation that omits its strongest reason."""
    lang = ex.load_language()
    for feature in (lang.get("signals") or {}):
        spec = lang["signals"][feature]
        assert spec.get("clause_positive"), f"{feature} has no positive clause"
        assert spec.get("clause_negative"), f"{feature} has no negative clause"
    for feature, spec in (lang.get("features") or {}).items():
        if spec.get("values"):
            for value, entry in spec["values"].items():
                assert entry.get("clause"), f"{feature}.{value} has no clause"


# ------------------------------------------------------------ 4: shape stability
def test_empty_summary_matches_the_populated_shape():
    populated = summary([MODEL_POS], why(1.28))
    empty = ex.empty_summary()
    assert set(populated) == set(empty) == SUMMARY_KEYS
    assert set(populated["counts"]) == set(empty["counts"]) == COUNT_KEYS
    assert empty["available"] is False and empty["text"] is None


def test_no_factors_still_names_the_probability():
    s = summary([], why(1.28))
    assert set(s) == SUMMARY_KEYS
    assert "1.28% purchase probability" in s["text"]
    assert s["positive"] == [] and s["negative"] == []


def test_no_probability_reads_as_unscorable():
    s = ex.summary_block([MODEL_POS], {"starting_point": {"percent": 1.09}})
    assert set(s) == SUMMARY_KEYS
    assert s["available"] is False
    assert "not enough" in s["text"]


def test_missing_why_does_not_raise():
    s = ex.summary_block([MODEL_POS], None, FEATURES)
    assert set(s) == SUMMARY_KEYS
    assert s["available"] is False


def test_note_appears_only_when_ranking_signals_are_present():
    with_priors = summary([MODEL_POS, PRIOR_POS], why(1.28))
    without = summary([MODEL_POS, MODEL_NEG], why(1.28))
    assert with_priors["note"]
    assert without["note"] is None


# ---------------------------------------------------------------- 5: live wiring
def test_scored_lead_carries_a_summary(scorable_lead_id):
    r = ppm.predict_for_lead(scorable_lead_id)
    if not r["availability"]["available"]:
        pytest.skip("lead not scorable")
    s = r["summary"]
    assert set(s) == SUMMARY_KEYS
    assert s["available"] is True
    assert f'{r["purchase_probability"]}% purchase probability' in s["text"]
    # The number explained is the probability, never the ranking score.
    assert str(r["lead_priority"]["score"]) not in s["text"]
    blob = s["text"].lower()
    for word in FORBIDDEN_JARGON:
        assert word not in blob


def test_unscorable_lead_has_an_empty_summary_of_the_same_shape():
    r = ppm.predict_for_lead(str(uuid.uuid4()))
    s = r["summary"]
    assert set(s) == SUMMARY_KEYS
    assert s["available"] is False
    assert s["text"] is None
    assert s["counts"]["total"] == 0
