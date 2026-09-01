"""Tests for the Purchase Probability baseline-MVP endpoint and inference package.

Data policy for this file:
  * Scoring tests run against REAL leads read from PostgreSQL, read-only.
  * The only synthetic objects are malformed-payload fixtures used for pure unit
    testing of build_features(). They never touch training or evaluation data.
  * Nothing here writes to the database, and nothing reads the TEST split.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import uuid

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO, ".env"))

import purchase_probability_model as ppm  # noqa: E402
from purchase_probability_model import inference as inf  # noqa: E402

V1_FEATURES = ["f_seniority", "f_email_class", "f_locale", "f_company_len",
               "f_company_is_placeholder", "f_hour_sin", "f_hour_cos"]
V2_FORBIDDEN = ["f_jt_token_count", "f_jt_is_generic_title", "f_co_has_entity_suffix"]


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
    """A real lead that has an admissible form_submit."""
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
def no_form_lead_id(conn):
    """A real lead with no touchpoints at all (e.g. the Kaizen CRM cohort)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.id FROM leads l
            WHERE NOT EXISTS (SELECT 1 FROM touchpoint_events te WHERE te.lead_id = l.id)
            LIMIT 1""")
        row = cur.fetchone()
    if not row:
        pytest.skip("no lead without touchpoints found")
    return str(row[0])


@pytest.fixture(scope="module")
def multi_touchpoint_lead_id(conn):
    """A real lead with several touchpoints including a post-purchase payment."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.id FROM leads l
            WHERE EXISTS (SELECT 1 FROM touchpoint_events te
                          WHERE te.lead_id = l.id AND te.type = 'payment')
              AND EXISTS (SELECT 1 FROM touchpoint_events te
                          WHERE te.lead_id = l.id AND te.type = 'form_submit')
            LIMIT 1""")
        row = cur.fetchone()
    if not row:
        pytest.skip("no lead with payment + form_submit found")
    return str(row[0])


# --------------------------------------------------------------------------- 1
def test_known_valid_lead_scores(conn, scorable_lead_id):
    r = ppm.predict_for_lead(scorable_lead_id, conn=conn)
    assert r["availability"]["available"] is True
    assert r["fallback"] is False
    assert r["lead_id"] == scorable_lead_id
    assert r["probability"] is not None
    assert r["priority"] in {"Low", "Medium", "High"}
    assert r["model"]["name"] == "purchase_probability"
    assert r["model"]["status"] == "baseline_mvp"


# --------------------------------------------------------------------------- 2
def test_missing_lead_returns_unavailable(conn):
    r = ppm.predict_for_lead(str(uuid.UUID(int=0)), conn=conn)
    assert r["availability"]["available"] is False
    assert r["reason"] == ppm.UNAVAILABLE_LEAD_NOT_FOUND
    assert r["purchase_probability"] is None
    assert r["priority"] == "Unavailable"


# --------------------------------------------------------------------------- 3
def test_missing_admissible_form_payload(conn, no_form_lead_id):
    r = ppm.predict_for_lead(no_form_lead_id, conn=conn)
    assert r["availability"]["available"] is False
    assert r["reason"] == ppm.UNAVAILABLE_NO_FORM_PAYLOAD
    # Unavailable is NOT zero.
    assert r["purchase_probability"] is None
    assert r["purchase_probability_percent"] is None
    assert r["percentile"] is None and r["decile"] is None


# --------------------------------------------------------------------------- 4
def test_unknown_brand_never_fabricates(conn):
    """Leads from brands absent from training must score or degrade - never invent."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.id FROM leads l JOIN brands b ON b.id = l.brand_id
            WHERE b.name NOT IN ('Lawtorney') LIMIT 5""")
        ids = [str(r[0]) for r in cur.fetchall()]
    if not ids:
        pytest.skip("no non-primary-brand leads")
    for lid in ids:
        r = ppm.predict_for_lead(lid, conn=conn)
        if r["availability"]["available"]:
            assert 0.0 <= r["probability"] <= 1.0
        else:
            assert r["purchase_probability"] is None
            assert r["priority"] == "Unavailable"


# --------------------------------------------------------------------------- 5
def test_missing_model_artefacts(monkeypatch, tmp_path, conn, scorable_lead_id):
    monkeypatch.setattr(inf, "PKG_DIR", str(tmp_path))
    monkeypatch.setattr(inf, "_ARTEFACTS", None)
    try:
        r = inf.predict_for_lead(scorable_lead_id, conn=conn)
        assert r["availability"]["available"] is False
        assert r["reason"] == ppm.UNAVAILABLE_MODEL_MISSING
        assert r["purchase_probability"] is None
    finally:
        monkeypatch.undo()
        inf._ARTEFACTS = None
        inf.load_artefacts(force=True)


# --------------------------------------------------------------------------- 6
@pytest.mark.parametrize("payload", [
    {},                                                   # empty
    {"data": None},                                       # null data
    {"data": {"contact": None}},                          # null contact
    {"data": {"contact": {}}},                            # no fields
    {"data": {"contact": {"jobTitle": "", "company": "", "email": "", "locale": ""}}},
    {"data": {"contact": {"jobTitle": 12345, "company": ["x"], "locale": {"a": 1}}}},
])
def test_malformed_payload_still_builds_features(payload):
    """Synthetic fixtures only - isolated from training/evaluation data."""
    import datetime as dt
    schema = ppm.load_artefacts()["schema"]
    lead = {"created_at": dt.datetime(2026, 6, 15, 9, 0, tzinfo=dt.timezone.utc),
            "email": None}
    f = ppm.build_features(lead, payload, schema)
    assert set(f) == set(V1_FEATURES)
    assert isinstance(f["f_company_len"], float)
    assert f["f_company_is_placeholder"] in (0.0, 1.0)
    assert -1.0 <= f["f_hour_sin"] <= 1.0 and -1.0 <= f["f_hour_cos"] <= 1.0


# --------------------------------------------------------------------------- 7
def test_probability_range(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.id FROM leads l
            WHERE EXISTS (SELECT 1 FROM touchpoint_events te
                          WHERE te.lead_id = l.id AND te.type='form_submit')
            LIMIT 40""")
        ids = [str(r[0]) for r in cur.fetchall()]
    import math
    scored = 0
    for lid in ids:
        r = ppm.predict_for_lead(lid, conn=conn)
        if not r["availability"]["available"]:
            continue
        scored += 1
        assert 0.0 <= r["probability"] <= 1.0
        assert 0 <= r["purchase_probability_percent"] <= 100
        assert 0.0 <= r["purchase_probability"] <= 100.0
        assert math.isfinite(r["probability"])
        assert math.isfinite(r["purchase_probability"])
        assert 1 <= r["decile"] <= 10
        assert 0 <= r["percentile"] <= 100
    assert scored > 0


# --------------------------------------------------------------------------- 8
def test_probability_is_not_inflated(conn):
    """purchase_probability must be the calibrated probability as a percentage -
    not a rescaled marketing score."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT l.id FROM leads l
            WHERE EXISTS (SELECT 1 FROM touchpoint_events te
                          WHERE te.lead_id = l.id AND te.type='form_submit')
            LIMIT 60""")
        ids = [str(r[0]) for r in cur.fetchall()]
    seen = 0
    for lid in ids:
        r = ppm.predict_for_lead(lid, conn=conn)
        if not r["availability"]["available"]:
            continue
        seen += 1
        assert r["purchase_probability"] == pytest.approx(r["probability"] * 100.0, abs=5e-3)
        assert r["purchase_probability_percent"] == round(r["probability"] * 100.0)
        # Base rate is ~1.1%; a calibrated baseline cannot legitimately exceed ~20%.
        assert r["probability"] < 0.20, "probability looks rescaled/inflated"
    assert seen > 0


# --------------------------------------------------------------------------- 9
def test_deterministic_inference(conn, scorable_lead_id):
    a = ppm.predict_for_lead(scorable_lead_id, conn=conn)
    b = ppm.predict_for_lead(scorable_lead_id, conn=conn)
    assert a["probability"] == b["probability"]
    assert a["percentile"] == b["percentile"] and a["decile"] == b["decile"]
    assert [f["feature"] for f in a["top_factors"]] == [f["feature"] for f in b["top_factors"]]


# --------------------------------------------------------------------------- 10
def test_top_factor_ordering_and_language(conn, scorable_lead_id):
    r = ppm.predict_for_lead(scorable_lead_id, conn=conn)
    factors = r["top_factors"]
    assert factors, "expected at least one contributing factor"
    mags = [abs(f["contribution"]) for f in factors]
    assert mags == sorted(mags, reverse=True), "factors must be ranked by |contribution|"
    for f in factors:
        assert f["direction"] in ("positive", "negative")
        assert (f["contribution"] > 0) == (f["direction"] == "positive")
        assert "Contributed" in f["explanation"]
        # No causal claims.
        assert "caused" not in f["explanation"].lower()


# --------------------------------------------------------------------------- 11
def test_no_forbidden_v2_features(conn, scorable_lead_id):
    schema = ppm.load_artefacts()["schema"]
    assert schema["features"] == V1_FEATURES
    assert schema["feature_version"] == "V1"
    for bad in V2_FORBIDDEN:
        assert bad not in schema["features"]
        assert bad not in schema["design_columns"]
        assert bad not in json.dumps(schema)
    r = ppm.predict_for_lead(scorable_lead_id, conn=conn)
    assert sorted(r["model_features"]["values"]) == sorted(V1_FEATURES)


# --------------------------------------------------------------------------- 12
def test_test_split_was_never_used():
    meta = ppm.load_artefacts()["metadata"]
    assert meta["test_set_usage"].startswith("NONE")
    assert meta["calibration"]["test_used"] is False
    ref = json.load(open(os.path.join(inf.PKG_DIR, "percentile_reference.json"),
                         encoding="utf-8"))
    assert ref["test_used"] is False
    # The percentile reference is the pooled OOF population, not TEST.
    assert ref["n"] == meta["training"]["oof_rows"] == 8445
    assert ref["positives"] == meta["training"]["oof_positives"] == 100
    assert meta["training"]["test_rows_excluded"] == 2939


# --------------------------------------------------------------------------- 13
def test_no_future_data_in_model_features():
    """The feature query must enforce both admissibility clauses and use only
    form_submit. Payment / ad_click / call must never be feature sources."""
    sql = inf._SQL_ADMISSIBLE_FORM
    assert "type = 'form_submit'" in sql
    assert "te.created_at <= %s + INTERVAL '1 hour'" in sql
    assert "te.created_at - te.occurred_at <= INTERVAL '1 hour'" in sql
    for forbidden in ("'payment'", "'ad_click'", "'call'"):
        assert forbidden not in sql
    for mutable in ("last_activity_at", "converted_at", "revenue", "score",
                    "temperature", "touchpoint_count", "status"):
        assert mutable not in sql
        assert mutable not in inf._SQL_LEAD


# --------------------------------------------------------------------------- 14
def test_touchpoints_are_display_only(conn, multi_touchpoint_lead_id):
    r = ppm.predict_for_lead(multi_touchpoint_lead_id, conn=conn)
    types = {t["type"] for t in r["touchpoints"]}
    assert "payment" in types, "display history should include the real payment event"
    assert r["touchpoint_count"] == len(r["touchpoints"])
    if r["availability"]["available"]:
        # ...but the model saw only the admissible form submission.
        assert sorted(r["model_features"]["values"]) == sorted(V1_FEATURES)
        assert "form_submit" in r["model_features"]["source"]
        assert "payment" not in r["model_features"]["source"]


# --------------------------------------------------------------------------- 15
def test_touchpoints_ordered(conn, multi_touchpoint_lead_id):
    r = ppm.predict_for_lead(multi_touchpoint_lead_id, conn=conn)
    stamps = [t["occurred_at"] for t in r["touchpoints"] if t.get("occurred_at")]
    assert stamps == sorted(stamps), "touchpoints must be chronological"


# --------------------------------------------------------------------------- 16
def test_fallback_shape_is_complete(conn):
    r = ppm.predict_for_lead(str(uuid.UUID(int=1)), conn=conn)
    for key in ("lead_id", "purchase_probability", "purchase_probability_percent",
                "percentile", "decile", "priority", "top_factors", "touchpoints",
                "touchpoint_count", "model", "availability", "fallback", "reason"):
        assert key in r
    assert r["fallback"] is True
    assert r["top_factors"] == [] and r["touchpoints"] == []
    assert r["availability"]["message"]


# --------------------------------------------------------------------------- 17
def test_authentication(scorable_lead_id):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import app as app_module

    if not app_module.API_KEY:
        pytest.skip("API_KEY not configured; auth is intentionally open in dev")

    client = TestClient(app_module.app)
    url = f"/api/purchase-probability/{scorable_lead_id}"

    # The service's existing convention for a bad/missing key is 401 (app.py:103).
    assert client.get(url).status_code == 401                       # no key
    assert client.get(url, headers={"X-API-Key": "wrong"}).status_code == 401

    ok = client.get(url, headers={"X-API-Key": app_module.API_KEY})
    assert ok.status_code == 200
    body = ok.json()
    assert body["lead_id"] == scorable_lead_id
    assert body["model"]["name"] == "purchase_probability"
    if body["availability"]["available"]:
        assert 0 <= body["purchase_probability_percent"] <= 100


def test_health_still_open():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import app as app_module
    assert TestClient(app_module.app).get("/health").status_code == 200
