"""Tests for the engagement (click + timeline) and Brand Brain layers.

Data policy, same as the baseline suite:
  * Layer tests run against REAL leads read from PostgreSQL, read-only.
  * The Brand Brain fixtures are literal dicts shaped like the real MongoDB
    documents. They are inputs to pure functions and never touch the database.
  * Nothing here writes anywhere.

The load-bearing assertions in this file are the ones about what the layers must
NOT do: they must not move the calibrated probability, they must not score a
missing signal as a negative one, and they must not admit backdated rows.
"""
from __future__ import annotations

import math
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO, ".env"))

import purchase_probability_model as ppm  # noqa: E402
from purchase_probability_model import behavioural as beh  # noqa: E402
from purchase_probability_model import blend as bl  # noqa: E402
from purchase_probability_model import brand_fit as bf  # noqa: E402
from purchase_probability_model import inference as inf  # noqa: E402

# Shaped exactly like the real `brand_brains` document.
BRAND_BRAIN = {
    "_id": "test_brand_brain_id",
    "answers": {
        "businessType": "Education / Coaching / Consulting",
        "idealCustomer": "40-60 cxo ceos and founders of mid-market companies",
        "language": "English only",
        "trafficChannels": ["Google Search", "Meta (Facebook + Instagram)", "YouTube"],
        "salesCycle": "Same day",
        "marketingGoal": "Lead quality (higher intent)",
        "journey": "Executives discover us through search and book a session.",
    },
    "context": {"industry": "Education / Coaching", "brandName": "DI",
                "audienceShort": "CXO, CEO", "website": "https://example.com"},
}

FIXED_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def cfg():
    config = ppm.load_artefacts().get("signal_config")
    if not config:
        pytest.skip("signal_config.json not present")
    return config


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
    """A real lead carrying more than one admissible, non-payment touchpoint."""
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


# ===========================================================================  1
def test_sales_cycle_parsing(cfg):
    assert bf.parse_sales_cycle_days("Same day", cfg) == 1.0
    assert bf.parse_sales_cycle_days("1-4 weeks", cfg) == 28.0     # upper bound
    assert bf.parse_sales_cycle_days("1-3 months", cfg) == 90.0
    assert bf.parse_sales_cycle_days("6+ months", cfg) == 180.0
    assert bf.parse_sales_cycle_days("2 weeks", cfg) == 14.0
    assert bf.parse_sales_cycle_days("Less than a week", cfg) == 7.0
    # Non-committal or absent answers must yield None, never a guessed number:
    # the factor is then skipped rather than applied against an invented cycle.
    for empty in ("varies", "not sure", "", None, "whenever they're ready"):
        assert bf.parse_sales_cycle_days(empty, cfg) is None


# ===========================================================================  2
def test_channel_normalisation(cfg):
    aliases = cfg["channel_aliases"]
    assert beh.normalise_channel("Meta (Facebook + Instagram)", aliases) == "meta"
    assert beh.normalise_channel("Google Search", aliases) == "google"
    assert beh.normalise_channel("  LinkedIn ", aliases) == "linkedin"
    assert beh.normalise_channel("youtube", aliases) == "youtube"
    assert beh.normalise_channel("carrier pigeon", aliases) is None
    assert beh.normalise_channel(None, aliases) is None
    assert beh.normalise_channel("", aliases) is None


# ===========================================================================  3
def test_brand_profile_from_real_document_shape(cfg):
    p = bf.parse_brand_profile(BRAND_BRAIN, cfg)
    assert p["available"] is True
    assert p["target_seniority"] == "founder_c_level"
    assert set(p["declared_channels"]) == {"google", "meta", "youtube"}
    assert p["business_model"] == "b2b"
    assert p["language_prefix"] == "en"
    assert p["sales_cycle_days"] == 1.0


# ===========================================================================  4
def test_absent_signal_is_never_a_penalty(cfg):
    """The rule this whole design rests on: no data must contribute exactly 0.

    A lead we know nothing about must not sink below a lead we know something
    mildly bad about.
    """
    factors, _ = bl.engagement_factors(
        {"available": False, "reason": beh.NO_BEHAVIOURAL_DATA,
         "message": "nothing recorded"}, {}, cfg)
    assert factors, "every factor must still be reported, not omitted"
    for f in factors:
        assert f["status"] == "no_data"
        assert f["contribution"] == 0.0
        assert f["direction"] == "neutral"

    brand = bf.brand_fit_factors({"available": False, "reason": bf.NO_BRAND_BRAIN},
                                 {}, {}, {"created_at": FIXED_NOW}, cfg, FIXED_NOW)
    assert brand
    for f in brand:
        assert f["status"] == "no_data"
        assert f["contribution"] == 0.0

    # And an empty brand brain must not be conjured into a profile.
    assert bf.parse_brand_profile(None, cfg)["available"] is False


# ===========================================================================  5
def test_brand_brain_drives_the_recency_half_life(cfg):
    """The timeline decays at the pace the brand said it sells at."""
    fast = bl.recency_half_life_hours({"sales_cycle_days": 1.0}, cfg)
    slow = bl.recency_half_life_hours({"sales_cycle_days": 180.0}, cfg)
    unknown = bl.recency_half_life_hours({}, cfg)

    assert fast["derived_from"] == "brand_sales_cycle"
    assert slow["derived_from"] == "brand_sales_cycle"
    assert unknown["derived_from"] == "default"
    assert fast["hours"] < unknown["hours"] < slow["hours"]
    bounds = cfg["engagement"]["recency_half_life_hours"]
    for h in (fast, slow, unknown):
        assert bounds["min"] <= h["hours"] <= bounds["max"]


# ===========================================================================  6
def test_layer_totals_are_clamped(cfg):
    """A prior that is not fitted must never be able to dominate the model."""
    lo, hi = cfg["engagement"]["max_total_adjustment"]
    huge = [{"contribution": 99.0, "status": "observed"}] * 5
    tiny = [{"contribution": -99.0, "status": "observed"}] * 5
    up = bl._apply_layer(huge, [lo, hi])
    down = bl._apply_layer(tiny, [lo, hi])
    assert up["total_applied"] == hi and up["clamped"] is True
    assert down["total_applied"] == lo and down["clamped"] is True
    assert up["total_raw"] == pytest.approx(495.0)


# ===========================================================================  7
def test_blend_arithmetic_is_self_consistent(cfg):
    eng, _ = bl.engagement_factors(
        {"available": True, "observed": {
            "clicks": 4, "page_views": 6, "high_intent_hits": 2, "sessions": 3,
            "active_days": 3, "events_total": 12, "returning": True,
            "ad_click_arrival": True, "recency_hours": 2.0,
            "events_per_active_day": 4.0}},
        {"sales_cycle_days": 7.0}, cfg)
    brand = bf.brand_fit_factors(
        bf.parse_brand_profile(BRAND_BRAIN, cfg),
        {"f_seniority": "founder_c_level", "f_email_class": "corporate",
         "f_locale": "en-IN"},
        {"channel": "google"},
        {"created_at": FIXED_NOW - timedelta(hours=3)}, cfg, FIXED_NOW)

    combo = bl.combine_layers(0.02, eng, brand, cfg)
    total = (combo["base"]["log_odds"] + combo["engagement"]["total_applied"]
             + combo["brand_brain"]["total_applied"])
    # Reported values are rounded for transport (log-odds 6dp, probability 8dp,
    # the same rounding the base probability gets), so compare at that grain.
    assert combo["adjusted"]["log_odds"] == pytest.approx(total, abs=1e-6)
    assert combo["adjusted"]["probability"] == pytest.approx(bl.sigmoid(total), abs=1e-6)
    assert combo["base"]["calibrated"] is True
    assert combo["adjusted"]["calibrated"] is False
    # An engaged lead matching its brand's ICP should rank above the raw base.
    assert combo["adjusted"]["probability"] > 0.02


# ===========================================================================  8
def test_behavioural_sql_cannot_read_the_receipt():
    """The anti-backfill clauses are correctness, not style. Assert them literally.

    scrumdb backdates attribution rows at payment time; without both clauses this
    layer would be scoring the outcome it is meant to predict.
    """
    sql = beh._SQL_ADMISSIBLE_BEHAVIOUR
    assert "te.type <> 'payment'" in sql, "payment is the label and must be excluded"
    assert "te.created_at - te.occurred_at <= %(tolerance)s" in sql
    assert "te.created_at <= %(window_end)s" in sql
    for mutable in ("converted_at", "revenue", "score", "temperature",
                    "touchpoint_count", "status", "last_activity_at"):
        assert mutable not in sql

    # The window closes at the payment WRITE time, which cannot be backdated.
    assert "MIN(te.created_at)" in beh._SQL_PAYMENT_WRITE_CUTOFF

    # client_ts is client-controlled, so it must never gate admissibility.
    tracking = beh._SQL_TRACKING_EVENTS
    assert "te.received_at >= %(window_start)s" in tracking
    assert "te.received_at <= %(window_end)s" in tracking
    assert "client_ts" not in tracking


# ===========================================================================  9
def test_layers_never_move_the_calibrated_probability(conn, scorable_lead_id):
    """The contract: brand context changes the RANKING, never the probability."""
    without = ppm.predict_for_lead(scorable_lead_id, conn=conn, now=FIXED_NOW)
    with_bb = ppm.predict_for_lead(scorable_lead_id, conn=conn,
                                   brand_brain=BRAND_BRAIN, now=FIXED_NOW)
    if not without["availability"]["available"]:
        pytest.skip("fixture lead is not scorable")

    assert without["probability"] == with_bb["probability"]
    assert without["purchase_probability"] == with_bb["purchase_probability"]
    assert without["percentile"] == with_bb["percentile"]
    # `model_factors` is the calibrated explanation and must be untouched by
    # brand context. `top_factors` is the merged lead-card list and is EXPECTED
    # to grow a brand-fit row here - that is the feature, not a regression.
    assert without["model_factors"] == with_bb["model_factors"]
    assert without["model_features"] == with_bb["model_features"]

    # ...but the brand layer did engage, and the ranking signal reflects it.
    assert without["brand_brain"]["available"] is False
    assert with_bb["brand_brain"]["available"] is True
    assert with_bb["lead_priority"]["calibrated"] is False
    assert with_bb["lead_priority"]["probability"] != without["probability"]


# ==========================================================================  10
def test_top_factors_declare_what_they_move(conn, scorable_lead_id):
    """`top_factors` is the merged lead-card list: base-model factors alongside
    the touchpoint-derived and brand-fit ones.

    Mixing them is safe ONLY because every row says which number it moves. If
    `affects` is ever dropped, a reader summing the rows would expect them to
    reach the displayed probability, and they do not. That is what this guards.
    """
    r = ppm.predict_for_lead(scorable_lead_id, conn=conn, brand_brain=BRAND_BRAIN,
                             now=FIXED_NOW)
    if not r["availability"]["available"]:
        pytest.skip("fixture lead is not scorable")

    for f in r["top_factors"]:
        assert f["affects"] in ("purchase_probability", "lead_priority")
        if f["feature"].startswith("f_"):
            assert f["affects"] == "purchase_probability"
        if f["feature"].startswith(("e_", "b_")):
            assert f["affects"] == "lead_priority", (
                "unfitted priors must never claim to move the calibrated score")

    # The calibrated explanation is still available on its own, undiluted.
    assert r["model_factors"], "expected base-model factors"
    for f in r["model_factors"]:
        assert f["feature"].startswith("f_")
        assert not f["feature"].startswith(("e_", "b_"))

    # The purchase_probability rows of the merged list ARE the base-model
    # factors - the merge renames, it never adds to or edits the calibrated set.
    merged = {f["feature"]: f["contribution"] for f in r["top_factors"]
              if f["affects"] == "purchase_probability"}
    base = {f["feature"]: f["contribution"] for f in r["model_factors"]
            if abs(f["contribution"]) > 1e-12}
    assert merged == base

    # The merged list is where the layers surface, tagged with their provenance.
    layers = {f["layer"] for f in r["ranking_factors"]}
    assert "model" in layers and "brand_brain" in layers
    bases = {f["basis"] for f in r["ranking_factors"]}
    assert bases <= {"calibrated_model", "heuristic"}
    mags = [abs(f["contribution"]) for f in r["ranking_factors"]]
    assert mags == sorted(mags, reverse=True)


# ==========================================================================  11
def test_engagement_ignores_a_lone_form_submission(conn, scorable_lead_id):
    """The admitting form submission is already scored by the base model.

    Counting it again as 'engagement' would double-count the one input we have.
    """
    r = ppm.predict_for_lead(scorable_lead_id, conn=conn, now=FIXED_NOW)
    eng = r["engagement"]
    if eng["available"]:
        obs = eng["observed"]
        assert obs["events_total"] > obs["form_submits"] or obs["form_submits"] > 1
    else:
        assert eng["reason"] in (beh.ONLY_FORM_SUBMISSION, beh.NO_BEHAVIOURAL_DATA,
                                 inf.SIGNAL_CONFIG_MISSING)
        assert all(f["contribution"] == 0.0 for f in eng["factors"])


# ==========================================================================  12
def test_engaged_lead_reports_a_real_timeline(conn, engaged_lead_id):
    r = ppm.predict_for_lead(engaged_lead_id, conn=conn, brand_brain=BRAND_BRAIN,
                             now=FIXED_NOW)
    eng = r["engagement"]
    assert eng["available"] is True
    obs = eng["observed"]
    assert obs["events_total"] >= 2
    assert obs["active_days"] >= 1
    assert obs["recency_hours"] is not None and obs["recency_hours"] >= 0
    stamps = [e["at"] for e in eng["timeline"]]
    assert stamps == sorted(stamps), "timeline must be chronological"
    assert eng["window"]["end_reason"] in ("now", "first_payment_write")
    # Nothing in the timeline may be a payment: that is the label.
    assert all(e.get("kind") != "payment" for e in eng["timeline"])


# ==========================================================================  13
def test_deterministic_given_a_fixed_clock(conn, engaged_lead_id):
    a = ppm.predict_for_lead(engaged_lead_id, conn=conn, brand_brain=BRAND_BRAIN,
                             now=FIXED_NOW)
    b = ppm.predict_for_lead(engaged_lead_id, conn=conn, brand_brain=BRAND_BRAIN,
                             now=FIXED_NOW)
    assert a["lead_priority"] == b["lead_priority"]
    assert a["engagement"]["factors"] == b["engagement"]["factors"]
    assert a["brand_brain"]["factors"] == b["brand_brain"]["factors"]


# ==========================================================================  14
def test_lead_ages_out_of_its_brands_sales_cycle(cfg):
    """A same-day brand should treat a month-old lead as cold, and say why."""
    profile = bf.parse_brand_profile(BRAND_BRAIN, cfg)  # salesCycle: "Same day"
    fresh = bf.brand_fit_factors(profile, {}, {},
                                 {"created_at": FIXED_NOW - timedelta(hours=1)},
                                 cfg, FIXED_NOW)
    stale = bf.brand_fit_factors(profile, {}, {},
                                 {"created_at": FIXED_NOW - timedelta(days=45)},
                                 cfg, FIXED_NOW)
    f = next(x for x in fresh if x["feature"] == "b_sales_cycle_timing")
    s = next(x for x in stale if x["feature"] == "b_sales_cycle_timing")
    assert f["contribution"] > 0 > s["contribution"]
    assert abs(s["contribution"]) <= cfg["brand_brain"]["weights"]["sales_cycle_timing"]
    assert s["value"]["brand_sales_cycle_days"] == 1.0

    # A brand that never declared a cycle gets no verdict on lead age at all.
    silent = bf.parse_brand_profile({"answers": {}, "context": {}}, cfg)
    q = next(x for x in bf.brand_fit_factors(
        silent, {}, {}, {"created_at": FIXED_NOW - timedelta(days=45)}, cfg, FIXED_NOW)
        if x["feature"] == "b_sales_cycle_timing")
    assert q["status"] == "no_data" and q["contribution"] == 0.0


# ==========================================================================  15
def test_response_shape_is_complete_including_fallbacks(conn):
    """Every response carries every layer key, so the UI branches on `available`
    rather than on whether a key exists."""
    layer_keys = ("engagement", "brand_brain", "lead_priority",
                  "ranking_factors", "signal_version")
    missing = ppm.predict_for_lead(str(uuid.UUID(int=1)), conn=conn, now=FIXED_NOW)
    assert missing["fallback"] is True
    for key in layer_keys:
        assert key in missing
    assert missing["lead_priority"]["priority"] == "Unavailable"
    assert missing["lead_priority"]["probability"] is None
    assert missing["engagement"]["available"] is False
    assert missing["brand_brain"]["available"] is False
    assert missing["ranking_factors"] == []


# ==========================================================================  15b
def test_layer_reasons_describe_the_layer_not_the_base_model(conn):
    """A layer must not report a base-model failure as its own reason.

    `no_admissible_form_payload` is a fact about the lead's form history. Echoing
    it into the brand_brain block claims the brand has no Brand Brain, which may be
    flatly untrue - the brand can have a perfectly good one that simply was not
    consulted because scoring stopped earlier.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.id FROM leads l
            WHERE NOT EXISTS (SELECT 1 FROM touchpoint_events te WHERE te.lead_id = l.id)
            LIMIT 1""")
        row = cur.fetchone()
    if not row:
        pytest.skip("no lead without touchpoints found")

    r = ppm.predict_for_lead(str(row[0]), conn=conn, brand_brain=BRAND_BRAIN,
                             now=FIXED_NOW)
    assert r["availability"]["reason"] == inf.UNAVAILABLE_NO_FORM_PAYLOAD
    assert r["brand_brain"]["reason"] == ppm.LAYERS_NOT_RUN
    assert r["brand_brain"]["reason"] != r["availability"]["reason"]
    assert "base model could not score" in r["brand_brain"]["message"]

    # Engagement ran independently, so it keeps its own, more specific finding.
    assert r["engagement"]["reason"] in (beh.NO_BEHAVIOURAL_DATA,
                                         beh.ONLY_FORM_SUBMISSION,
                                         ppm.LAYERS_NOT_RUN)


# ==========================================================================  16
def test_brand_brain_ref_resolves_without_raising(conn, scorable_lead_id):
    ref = ppm.resolve_brand_brain_ref(scorable_lead_id, conn=conn)
    assert set(ref) == {"lead_id", "brand_id", "brand_name", "brand_brain_id"}
    assert ref["lead_id"] == scorable_lead_id
    # A brand that has not finished onboarding has no id, and that is not an error.
    assert ref["brand_brain_id"] is None or isinstance(ref["brand_brain_id"], str)

    unknown = ppm.resolve_brand_brain_ref(str(uuid.UUID(int=1)), conn=conn)
    assert unknown["brand_id"] is None and unknown["brand_brain_id"] is None


# ==========================================================================  17
def test_probability_bounds_hold_across_the_layers(conn, engaged_lead_id):
    r = ppm.predict_for_lead(engaged_lead_id, conn=conn, brand_brain=BRAND_BRAIN,
                             now=FIXED_NOW)
    if not r["availability"]["available"]:
        pytest.skip("fixture lead is not scorable")
    lp = r["lead_priority"]
    assert 0.0 < lp["probability"] < 1.0
    assert math.isfinite(lp["probability"])
    assert 0 <= lp["percentile"] <= 100
    assert 1 <= lp["decile"] <= 10
    # The adjustment is bounded by construction, so the ranking probability can
    # never run away from the calibrated one.
    eb = r["engagement"]["bounds"]
    bb = r["brand_brain"]["bounds"]
    assert eb[0] <= lp["log_odds"]["engagement"] <= eb[1]
    assert bb[0] <= lp["log_odds"]["brand_brain"] <= bb[1]


# ==========================================================================  18
def test_endpoint_returns_the_layers(scorable_lead_id):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import app as app_module

    headers = {}
    if os.environ.get("API_KEY"):
        headers["X-API-Key"] = os.environ["API_KEY"]
    with TestClient(app_module.app) as client:
        resp = client.get(f"/api/purchase-probability/{scorable_lead_id}",
                          headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    for key in ("purchase_probability", "engagement", "brand_brain",
                "lead_priority", "ranking_factors"):
        assert key in body
    assert body["brand_brain"]["brand_brain_store"] in ("configured", "not_configured")
    assert "resolved_brand_brain_id" in body["brand_brain"]
