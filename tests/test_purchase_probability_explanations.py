"""Tests for the plain-language factor layer (`explain.py`).

Data policy, same as the rest of the suite: real leads, read-only Postgres,
nothing written anywhere. The pure-function tests need no database at all.

What these assertions are actually protecting:

  * `affects` must survive. `top_factors` deliberately mixes the calibrated
    base-model factors with unfitted engagement and brand-fit priors. That is
    only honest while every row says which number it moves.
  * The copy must stay associational. A logistic regression on observational
    data cannot support "caused", "because" or "drives", however much better
    those read on a lead card.
  * The presentation layer must not touch a number. It renames; it never
    rescores.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO, ".env"))

import purchase_probability_model as ppm  # noqa: E402
from purchase_probability_model import explain as ex  # noqa: E402

# Words that assert causation, plus the softer ones that imply it. A coefficient
# fitted on observational data supports none of them.
FORBIDDEN = ("caused", "causes", "because", "due to", "drives", "makes them",
             "guarantees", "will buy", "ensures")


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


@pytest.fixture(scope="module")
def engaged_lead_id(conn):
    """A real lead with admissible, non-backdated behavioural history."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT te.lead_id FROM touchpoint_events te
            WHERE te.type <> 'payment'
              AND te.created_at - te.occurred_at <= INTERVAL '1 hour'
            GROUP BY te.lead_id HAVING count(*) > 1
            ORDER BY count(*) DESC LIMIT 1""")
        row = cur.fetchone()
    if not row:
        pytest.skip("no lead with admissible behavioural history")
    return str(row[0])


def _scored(lead_id, conn):
    r = ppm.predict_for_lead(lead_id, conn=conn)
    if not r["availability"]["available"]:
        pytest.skip("fixture lead is not scorable")
    return r


# --------------------------------------------------------------------------- 1
def test_every_factor_is_renderable(conn, scorable_lead_id):
    """The lead card must never have to fall back to a raw feature key."""
    r = _scored(scorable_lead_id, conn)
    assert r["top_factors"], "expected at least one factor"
    for f in r["top_factors"]:
        for key in ("label", "title", "impact", "affects", "source",
                    "contribution", "direction", "odds_multiplier"):
            assert key in f, f"{f['feature']} is missing {key}"
        assert f["title"] and not f["title"].startswith(("f_", "e_", "b_")), (
            f"raw feature key leaked into the title: {f['title']}")
        assert f["impact"] != "No effect", "zero-contribution rows must be dropped"


# --------------------------------------------------------------------------- 1b
def test_label_carries_the_reason_not_the_field_name(conn, engaged_lead_id):
    """The point of the whole layer.

    `label` is the only text the existing lead card renders, so it has to answer
    'why did this move the score' on its own. The old values named the model
    input instead - 'Lead locale' told a salesperson nothing - and those exact
    strings must never come back.
    """
    r = _scored(engaged_lead_id, conn)
    dead = {"Lead locale", "Email address type", "Senior decision-maker role",
            "Company information detail", "Submission time pattern",
            "Company information quality", "Recency of last activity",
            "Returning sessions", "Activity per active day"}
    for f in r["top_factors"]:
        assert f["label"] not in dead, (
            f"{f['feature']} regressed to the old field-name label: {f['label']}")
        # '<what was observed> - <why that matters>'.
        assert " - " in f["label"], f"{f['feature']} label states no reason: {f['label']}"
        head, _, reason = f["label"].partition(" - ")
        assert head.strip() and reason.strip()
        # A stutter means the two halves were built from the same source.
        assert head.strip().lower() not in reason.strip().lower()


# --------------------------------------------------------------------------- 2
def test_language_stays_associational(conn, scorable_lead_id, engaged_lead_id):
    """No causal claims anywhere a user can read."""
    for lead_id in (scorable_lead_id, engaged_lead_id):
        r = ppm.predict_for_lead(lead_id, conn=conn)
        for f in r["top_factors"]:
            text = (f"{f.get('label', '')} {f.get('title', '')} "
                    f"{f.get('detail', '')}").lower()
            for word in FORBIDDEN:
                assert word not in text, f"causal language in {f['feature']}: {word}"


# --------------------------------------------------------------------------- 3
def test_affects_tag_is_correct(conn, engaged_lead_id):
    """The load-bearing field. Base-model rows move the probability; the
    heuristic layers move the ranking score, and must never claim otherwise."""
    r = _scored(engaged_lead_id, conn)
    for f in r["top_factors"]:
        if f["feature"].startswith("f_"):
            assert f["affects"] == "purchase_probability"
            assert f["source"] == "Form submission"
        else:
            assert f["affects"] == "lead_priority"
            assert f["source"] in ("Touchpoint history", "Brand profile")


# --------------------------------------------------------------------------- 4
def test_touchpoint_factors_reach_the_card(conn, engaged_lead_id):
    """The point of the merge: touchpoint_events evidence is visible on the card."""
    r = _scored(engaged_lead_id, conn)
    if not r["engagement"]["available"]:
        pytest.skip("engagement layer found nothing admissible for this lead")
    tp = [f for f in r["top_factors"] if f["source"] == "Touchpoint history"]
    assert tp, "engagement factors did not reach top_factors"
    for f in tp:
        assert f["feature"].startswith("e_")
        assert f.get("detail"), "a touchpoint factor with no sentence is unreadable"


# --------------------------------------------------------------------------- 5
def test_presentation_never_changes_a_number(conn, scorable_lead_id):
    """`top_factors` renames; it must not add to or edit the calibrated set."""
    r = _scored(scorable_lead_id, conn)
    merged = {f["feature"]: f["contribution"] for f in r["top_factors"]
              if f["affects"] == "purchase_probability"}
    base = {f["feature"]: f["contribution"] for f in r["model_factors"]
            if abs(f["contribution"]) > 1e-12}
    assert merged == base


# --------------------------------------------------------------------------- 6
def test_odds_multiplier_restates_the_contribution(conn, scorable_lead_id):
    """It is exp() of the log-odds contribution - the same quantity, readable.
    If these ever diverge, the card is showing an invented number."""
    r = _scored(scorable_lead_id, conn)
    for f in r["top_factors"]:
        assert f["odds_multiplier"] == pytest.approx(
            math.exp(f["contribution"]), rel=1e-3)
        assert (f["odds_multiplier"] > 1.0) == (f["direction"] == "positive")


# --------------------------------------------------------------------------- 7
def test_why_block_anchors_on_the_real_base_rate(conn, scorable_lead_id):
    """The starting point is the measured training base rate, not a design
    constant. The prototype's '+45' was invented; this must not be."""
    r = _scored(scorable_lead_id, conn)
    why = r["why"]
    meta = ppm.load_artefacts()["metadata"]["training"]
    expected = round(meta["dataset_positives"] / meta["dataset_rows"] * 100.0, 2)

    assert why["starting_point"]["percent"] == expected
    assert why["result"]["percent"] == r["purchase_probability"]
    # The caveat that stops a reader summing the rows must be present.
    assert "do not add up" in why["reading_note"]


# --------------------------------------------------------------------------- 8
def test_unscorable_lead_has_the_same_shape(conn):
    """A missing probability means no factors and no `why` - never a fabricated
    explanation of a number that does not exist."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.id FROM leads l
            WHERE NOT EXISTS (
                SELECT 1 FROM touchpoint_events te
                WHERE te.lead_id = l.id AND te.type = 'form_submit'
                  AND te.created_at <= l.created_at + INTERVAL '1 hour'
                  AND te.created_at - te.occurred_at <= INTERVAL '1 hour')
            LIMIT 1""")
        row = cur.fetchone()
    if not row:
        pytest.skip("no unscorable lead found")
    r = ppm.predict_for_lead(str(row[0]), conn=conn)
    assert r["availability"]["available"] is False
    assert r["top_factors"] == [] and r["model_factors"] == []
    assert r["why"] is None


# --------------------------------------------------------------------------- 9
def test_hour_recovery_round_trips():
    """Pure unit test - no database. The submission hour is not stored; it is
    recovered from the sin/cos pair, and the card prints it to the user."""
    for hour in range(24):
        radians = 2 * math.pi * hour / 24
        got = ex._hour_from_cyclic({
            "f_hour_sin": round(math.sin(radians), 6),
            "f_hour_cos": round(math.cos(radians), 6),
        })
        assert got == pytest.approx(hour, abs=0.01), f"hour {hour} did not survive"


# --------------------------------------------------------------------------- 10
def test_impact_bands_are_ordered():
    """Pure unit test. A bigger contribution must never read as a smaller word."""
    order = {"Slight": 0, "Moderate": 1, "Strong": 2}
    lang = ex.load_language()
    words = [ex._impact_word(c, lang).split()[0] for c in (0.05, 0.3, 0.9)]
    assert [order[w] for w in words] == [0, 1, 2]
    assert ex._impact_word(0.0, lang) == "No effect"
    assert ex._impact_word(-0.9, lang).endswith("negative")


# --------------------------------------------------------------------------- 11
def test_recency_phrasing_is_rounded():
    """Pure unit test. False precision on a recency figure invites arguments
    about minutes that do not matter to a salesperson."""
    assert ex._ago(0.4) == "less than an hour ago"
    assert ex._ago(1.2) == "1 hour ago"
    assert ex._ago(22.4) == "22 hours ago"
    assert ex._ago(72.0) == "3 days ago"
