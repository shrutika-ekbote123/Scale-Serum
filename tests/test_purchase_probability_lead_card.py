"""Tests for the CRM lead card (`lead_summary` and its flat mirrors).

Data policy, same as the rest of the suite: real leads, read-only PostgreSQL,
no writes anywhere. The pure-function tests below build card dicts by hand -
they are inputs to `_lead_summary` / `_lifetime_value`, not database rows.

The load-bearing assertions here are the ones about what the card must NOT do:
it must not let a post-conversion column reach the model, it must not present an
estimate as recorded money, and it must not go blank just because the model had
no probability to give.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO, ".env"))

import purchase_probability_model as ppm  # noqa: E402
from purchase_probability_model import inference as inf  # noqa: E402

CARD_KEYS = {
    "available", "reason", "message", "lead_score", "lead_score_max", "temperature",
    "status", "stage", "source", "created_at", "touchpoint_count", "total_revenue",
    "currency", "payment_count", "converted", "converted_at", "days_to_convert",
    "lifetime_value", "basis",
}
FLAT_KEYS = ("lead_score", "temperature", "total_revenue", "days_to_convert",
             "lifetime_value")

T0 = datetime(2026, 9, 1, 11, 5, 45, tzinfo=timezone.utc)
ORDER_STATS = {"order_count": 1137, "average_order_value": 49441.2,
               "median_order_value": 299.0, "currency": "INR"}


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
def scored_lead_id(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.id FROM leads l
            JOIN touchpoint_events te ON te.lead_id = l.id
             AND te.type = 'form_submit'
             AND te.created_at <= l.created_at + INTERVAL '1 hour'
             AND te.created_at - te.occurred_at <= INTERVAL '1 hour'
            WHERE l.score IS NOT NULL
            LIMIT 1""")
        row = cur.fetchone()
    if not row:
        pytest.skip("no scorable lead with a CRM score available")
    return str(row[0])


@pytest.fixture(scope="module")
def unscorable_lead_id(conn):
    """A real lead the model cannot score but the CRM still knows things about."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.id FROM leads l
            WHERE l.score IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM touchpoint_events te
                              WHERE te.lead_id = l.id AND te.type = 'form_submit')
            LIMIT 1""")
        row = cur.fetchone()
    if not row:
        pytest.skip("no unscorable lead with a CRM score available")
    return str(row[0])


def _card(**over) -> dict:
    base = {"created_at": T0, "score": 25, "temperature": "cold", "status": "new",
            "stage": None, "source": "ICDP Webinar Registration New", "revenue": None,
            "converted_at": None, "time_to_convert_days": None, "touchpoint_count": 1}
    base.update(over)
    return base


# ----------------------------------------------------------------------- 1: separation
def test_card_columns_never_reach_the_model():
    """The whole point of the second query: the CRM columns are mutated at payment
    time, so they must be unreachable from the model's own lead row."""
    for mutable in ("score", "temperature", "status", "revenue", "converted_at",
                    "time_to_convert_days", "touchpoint_count", "last_activity_at"):
        assert mutable not in inf._SQL_LEAD
    # ...and the card query must not pretend to be a feature source.
    assert "form_submit" not in inf._SQL_LEAD_CARD
    assert "payload" not in inf._SQL_LEAD_CARD


def test_card_query_columns_match_the_names_they_are_zipped_to():
    """_CARD_COLUMNS is positional against _SQL_LEAD_CARD; a silent drift here
    would mis-label every value on the card."""
    select = inf._SQL_LEAD_CARD.split("FROM")[0]
    cols = [c.strip().split("::")[0].strip() for c in
            select.replace("SELECT", "").split(",")]
    assert tuple(cols) == inf._CARD_COLUMNS


# ------------------------------------------------------------------- 2: live responses
def test_scored_lead_has_a_populated_card(scored_lead_id):
    r = ppm.predict_for_lead(scored_lead_id)
    card = r["lead_summary"]
    assert set(card) == CARD_KEYS
    assert card["available"] is True
    assert card["lead_score_max"] == 100
    assert isinstance(card["lead_score"], int)
    for k in FLAT_KEYS:
        assert k in r, f"{k} should be mirrored at the top level"
    assert r["lead_score"] == card["lead_score"]
    assert r["lifetime_value"] == card["lifetime_value"]["amount"]


def test_lead_score_is_not_the_probability(scored_lead_id):
    """The confusion this endpoint exists to remove: a CRM score of 25 is not a
    25% chance of purchase."""
    r = ppm.predict_for_lead(scored_lead_id)
    if not r["availability"]["available"]:
        pytest.skip("lead not scorable")
    assert r["lead_score"] != r["purchase_probability"]
    assert r["purchase_probability"] < 25.0, (
        "the calibrated base rate is ~1.1%; a score-shaped number here means the "
        "lead score has leaked into the probability")


def test_unscorable_lead_still_shows_its_card(unscorable_lead_id):
    """The user's requirement: nulls for the probability, real values for the rest."""
    r = ppm.predict_for_lead(unscorable_lead_id)
    assert r["availability"]["available"] is False
    assert r["purchase_probability"] is None
    card = r["lead_summary"]
    assert card["available"] is True
    assert card["lead_score"] is not None
    assert card["temperature"] is not None
    # No probability means no expected value - and no invented one either.
    assert card["lifetime_value"]["amount"] is None
    assert card["lifetime_value"]["reason"] == inf.UNAVAILABLE_NO_FORM_PAYLOAD


def test_missing_lead_has_an_empty_card_of_the_same_shape():
    r = ppm.predict_for_lead(str(uuid.uuid4()))
    card = r["lead_summary"]
    assert set(card) == CARD_KEYS
    assert card["available"] is False
    assert card["reason"] == inf.UNAVAILABLE_LEAD_NOT_FOUND
    assert all(r[k] is None for k in FLAT_KEYS)


# ------------------------------------------------------------------ 3: lifetime value
def test_expected_value_is_probability_times_median_order():
    ltv = inf._lifetime_value(0.0139649, ORDER_STATS, "INR")
    assert ltv["available"] is True
    assert ltv["estimated"] is True
    assert ltv["amount"] == pytest.approx(0.0139649 * 299.0, abs=0.01)
    assert ltv["order_value_used"] == 299.0
    # The mean is reported but not used - it is dragged by long-tail orders.
    assert ltv["average_order_value"] == 49441.2


def test_realised_revenue_wins_over_the_estimate():
    """A lead that has paid is not a forecast."""
    ltv = inf._lifetime_value(0.0119, ORDER_STATS, "INR", realised=48000000.0)
    assert ltv["estimated"] is False
    assert ltv["amount"] == 48000000.0
    assert "recorded" in ltv["basis"]
    # ...and the estimate is kept alongside, not thrown away.
    assert ltv["expected_amount"] == pytest.approx(0.0119 * 299.0, abs=0.01)


def test_no_probability_yields_null_not_a_substitute():
    ltv = inf._lifetime_value(None, ORDER_STATS, "INR")
    assert ltv["amount"] is None
    assert ltv["available"] is False
    assert ltv["reason"] == inf.LTV_NO_PROBABILITY
    # Undiscounted deal size does not depend on the model, so it survives.
    assert ltv["potential_amount"] == 299.0


def test_no_order_history_yields_null():
    ltv = inf._lifetime_value(0.02, {}, None)
    assert ltv["amount"] is None
    assert ltv["reason"] == inf.LTV_NO_ORDER_HISTORY
    assert ltv["potential_amount"] is None


# ------------------------------------------------------------------------- 4: pure card
def test_zero_revenue_is_not_null_revenue():
    """0 means 'nothing yet' and null means 'not known'. The card must not collapse
    them, even though it renders both as a dash."""
    card = inf._lead_summary(_card(revenue=0), [], ORDER_STATS, 0.01)
    assert card["total_revenue"] == 0.0
    assert card["converted"] is False
    assert card["payment_count"] == 0


def test_days_to_convert_is_null_when_not_converted():
    """Zero days to convert is a same-day purchase - a very different fact."""
    card = inf._lead_summary(_card(), [], ORDER_STATS, 0.01)
    assert card["days_to_convert"] is None
    assert card["converted_at"] is None


def test_revenue_falls_back_to_payment_touchpoints():
    tps = [{"type": "form_submit", "occurred_at": None},
           {"type": "payment", "occurred_at": None, "value": 30000.0, "currency": "INR"},
           {"type": "payment", "occurred_at": None, "value": 300.0, "currency": "INR"}]
    card = inf._lead_summary(_card(revenue=None), tps, ORDER_STATS, 0.01)
    assert card["total_revenue"] == 30300.0
    assert card["payment_count"] == 2
    assert card["currency"] == "INR"
    assert card["converted"] is True


def test_crm_touchpoint_count_is_reported_separately():
    """leads.touchpoint_count and the displayed history can legitimately differ;
    the card shows the CRM's counter."""
    card = inf._lead_summary(_card(touchpoint_count=7), [], ORDER_STATS, 0.01)
    assert card["touchpoint_count"] == 7


def test_empty_card_and_populated_card_have_the_same_shape():
    populated = inf._lead_summary(_card(), [], ORDER_STATS, 0.01)
    empty = inf._empty_lead_summary(inf.UNAVAILABLE_DB_ERROR)
    assert set(populated) == set(empty) == CARD_KEYS
    assert set(populated["lifetime_value"]) == set(empty["lifetime_value"])
