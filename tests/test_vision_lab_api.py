"""The Vision Lab endpoints, driven through the real app.

app.py is imported with a dummy GEMINI_API_KEY and no MONGODB_URI, then the
Vision Lab store is swapped for fake collections. No network, no database, no
keys, no ffmpeg, no model weights.

Also asserts the existing endpoints are untouched - Vision Lab must not be able
to break onboarding, Script Lab, purchase probability or sales calls.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Set before the import so the module-level checks see them.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.pop("MONGODB_URI", None)
os.environ["API_KEY"] = "test-api-key"
os.environ["VL_ENABLED"] = "true"
os.environ["VL_ALLOWED_URL_HOSTS"] = "bucket.s3.amazonaws.com"

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
import vision_lab as vl  # noqa: E402
from vision_lab import pipeline as pl  # noqa: E402
from vision_lab import store as st  # noqa: E402

from test_vision_lab_pipeline import (  # noqa: E402
    FakeCollection,
    make_deps,
)

HEADERS = {"X-API-Key": "test-api-key"}

CREATIVE_URL = "https://bucket.s3.amazonaws.com/test/di_board_seat_v4_final.mp4?X-Amz-Signature=abc"

BODY = {
    "creative_id": "cre_8f21",
    "division": "directors_institute",
    "ad_number": "042",
    "creative": {
        "url": CREATIVE_URL,
        "kind": "video",
        "mime_type": "video/mp4",
        "duration_seconds": 24,
        "width": 1920,
        "height": 1080,
    },
    "brand_assets": {"brand_names": ["Director's Institute"]},
}


@pytest.fixture
def wired(monkeypatch):
    """A configured Vision Lab with fake storage. Returns the store so a test
    can drive the worker by hand."""
    analyses = FakeCollection()
    measurements = FakeCollection()
    store = st.AnalysisStore(analyses, measurements_collection=measurements)

    monkeypatch.setattr(app_module, "vision_lab_analyses", analyses)
    monkeypatch.setattr(app_module, "vision_lab_measurements", measurements)
    monkeypatch.setattr(app_module, "vision_lab_store", store)
    monkeypatch.setattr(app_module, "VISION_LAB_AVAILABLE", True)
    monkeypatch.setattr(app_module, "VL_ENABLED", True)
    monkeypatch.setattr(app_module, "VL_ALLOW_ANY_URL", False)
    monkeypatch.setattr(app_module, "VL_ALLOWED_URL_HOSTS", ["bucket.s3.amazonaws.com"])
    return store


@pytest.fixture
def client():
    return TestClient(app_module.app)


def work_the_queue(store):
    """Do what vision-worker would do, synchronously, so a test can get from
    `queued` to a finished report without a second process."""
    async def scenario():
        doc = await store.claim_next("test-worker")
        if doc:
            await pl.run_analysis(doc, make_deps(store))
    asyncio.run(scenario())


# =========================================================================== #
# Submit
# =========================================================================== #
def test_analyze_accepts_and_queues(wired, client):
    response = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS)
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == vl.STATUS_QUEUED
    assert payload["creative_id"] == "cre_8f21"
    assert payload["analysis_id"]
    assert payload["poll_url"] == f"/api/vision-lab/analysis/{payload['analysis_id']}"
    assert payload["suggested_poll_interval_seconds"] >= 1
    assert payload["idempotent_hit"] is False


def test_the_api_key_is_required(wired, client):
    assert client.post("/api/vision-lab/analyze", json=BODY).status_code == 401
    assert client.get("/api/vision-lab/history").status_code == 401


def test_a_url_off_the_allowlist_is_refused_as_a_stated_failure(wired, client):
    """Fetching an arbitrary URL server-side is SSRF. The refusal is a recorded
    analysis with a reason code, not a 4xx, so it matches every other failure."""
    body = dict(BODY, creative={"url": "https://evil.example.com/x.mp4",
                                "kind": "video"})
    response = client.post("/api/vision-lab/analyze", json=body, headers=HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == vl.STATUS_FAILED
    assert payload["reason"] == vl.URL_NOT_ALLOWED
    assert payload["availability"]["available"] is False


def test_a_plain_http_url_is_refused(wired, client):
    body = dict(BODY, creative={"url": "http://bucket.s3.amazonaws.com/x.mp4",
                                "kind": "video"})
    payload = client.post("/api/vision-lab/analyze", json=body,
                          headers=HEADERS).json()
    assert payload["reason"] == vl.URL_NOT_ALLOWED


def test_a_missing_url_is_refused(wired, client):
    body = dict(BODY, creative={})
    payload = client.post("/api/vision-lab/analyze", json=body,
                          headers=HEADERS).json()
    assert payload["reason"] == vl.NO_CREATIVE


def test_a_resubmit_is_idempotent_and_costs_nothing(wired, client):
    first = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS).json()
    work_the_queue(wired)

    second = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS).json()

    assert second["idempotent_hit"] is True
    assert second["analysis_id"] == first["analysis_id"]
    assert len(wired.collection.docs) == 1


def test_force_reanalysis_overrides_idempotency(wired, client):
    first = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS).json()
    work_the_queue(wired)

    body = dict(BODY, options={"force_reanalysis": True})
    second = client.post("/api/vision-lab/analyze", json=body, headers=HEADERS).json()

    assert second["idempotent_hit"] is False
    assert second["analysis_id"] != first["analysis_id"]
    assert len(wired.collection.docs) == 2


def test_a_presigned_url_signed_twice_is_still_the_same_creative(wired, client):
    """The signature changes on every presign. If it were part of the identity,
    idempotency would never hit and every poll would start a new job."""
    client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS)
    work_the_queue(wired)

    resigned = dict(BODY, creative=dict(BODY["creative"],
                                        url=CREATIVE_URL.replace("abc", "zzz")))
    second = client.post("/api/vision-lab/analyze", json=resigned,
                         headers=HEADERS).json()

    assert second["idempotent_hit"] is True


# =========================================================================== #
# Poll
# =========================================================================== #
def test_while_queued_the_envelope_has_no_scores(wired, client):
    submitted = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS).json()

    payload = client.get(submitted["poll_url"], headers=HEADERS).json()

    assert payload["status"] == vl.STATUS_QUEUED
    assert payload["scores"] is None
    assert payload["availability"]["available"] is True
    assert payload["suggested_poll_interval_seconds"] >= 1


def test_a_completed_analysis_returns_the_full_report(wired, client):
    submitted = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS).json()
    work_the_queue(wired)

    report = client.get(submitted["poll_url"], headers=HEADERS).json()

    assert report["status"] == vl.STATUS_COMPLETED
    assert report["stub"] is False         # every block is real or states why not
    assert set(report["scores"]) == {
        "attention", "focus", "cognitive_demand", "clarity",
        "brand_memory", "engagement"}
    assert report["media"]["frames_analyzed"] == 48
    assert report["media"]["sample_fps"] == 2.0
    assert report["config_disclosure"]["weighting"] == "equal_unweighted_placeholder"
    assert report["versions"]["framework_version"] == "vision_v1"
    assert report["versions"]["psychology_version"] == "triggers_v1"
    # The credential never reaches a response.
    assert "creative_url" not in report


def test_a_failure_reports_its_reason_in_both_places(wired, client):
    """`availability.reason` is canonical, but the POST response also carries
    `reason` at the top level. If the GET did not, the frontend would have to
    look in different places depending on which call it made."""
    body = dict(BODY, creative={"url": CREATIVE_URL, "mime_type": "application/pdf"})
    submitted = client.post("/api/vision-lab/analyze", json=body, headers=HEADERS).json()
    work_the_queue(wired)

    payload = client.get(submitted["poll_url"], headers=HEADERS).json()

    assert payload["status"] == vl.STATUS_FAILED
    assert payload["reason"] == vl.UNSUPPORTED_FORMAT
    assert payload["availability"]["reason"] == vl.UNSUPPORTED_FORMAT
    assert payload["message"] == payload["availability"]["message"]
    assert payload["scores"] is None          # a failure never invents a scorecard
    assert payload["fallback"] is True


def test_an_unknown_analysis_id_is_a_404(wired, client):
    assert client.get("/api/vision-lab/analysis/nope", headers=HEADERS).status_code == 404


def test_by_creative_returns_the_latest_analysis(wired, client):
    submitted = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS).json()
    work_the_queue(wired)

    payload = client.get("/api/vision-lab/analysis/by-creative/cre_8f21",
                         headers=HEADERS).json()
    assert payload["analysis_id"] == submitted["analysis_id"]

    assert client.get("/api/vision-lab/analysis/by-creative/unknown",
                      headers=HEADERS).status_code == 404


# =========================================================================== #
# History
# =========================================================================== #
def test_history_filters_by_ad_number(wired, client):
    client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS)
    work_the_queue(wired)
    other = dict(BODY, creative_id="cre_9", ad_number="043")
    client.post("/api/vision-lab/analyze", json=other, headers=HEADERS)
    work_the_queue(wired)

    everything = client.get("/api/vision-lab/history", headers=HEADERS).json()
    just_042 = client.get("/api/vision-lab/history?ad_number=042",
                          headers=HEADERS).json()

    assert everything["count"] == 2
    assert just_042["count"] == 1
    assert just_042["items"][0]["ad_number"] == "042"
    # Null because the stubbed vision layer measures nothing, and a metric with
    # no inputs reports null rather than a zero that would be indistinguishable
    # from a genuinely terrible creative.
    assert just_042["items"][0]["overall_score"] is None
    assert just_042["items"][0]["status"] == vl.STATUS_COMPLETED


# =========================================================================== #
# Rescore and delete
# =========================================================================== #
def test_rescore_requires_a_completed_analysis(wired, client):
    submitted = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS).json()
    response = client.post(
        f"/api/vision-lab/analysis/{submitted['analysis_id']}/rescore", headers=HEADERS)
    assert response.status_code == 409


def test_rescore_refreshes_config_without_reprocessing(wired, client):
    submitted = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS).json()
    work_the_queue(wired)

    response = client.post(
        f"/api/vision-lab/analysis/{submitted['analysis_id']}/rescore", headers=HEADERS)

    assert response.status_code == 200
    report = response.json()
    assert report["status"] == vl.STATUS_COMPLETED
    assert report["config_disclosure"]["weighting"] == "equal_unweighted_placeholder"
    # Still exactly one job document - nothing was re-queued or re-processed.
    assert len(wired.collection.docs) == 1


def test_rescore_refuses_when_the_measurements_have_expired(wired, client):
    submitted = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS).json()
    work_the_queue(wired)
    wired.measurements.docs.clear()          # TTL expiry, simulated

    response = client.post(
        f"/api/vision-lab/analysis/{submitted['analysis_id']}/rescore", headers=HEADERS)
    assert response.status_code == 409


def test_delete_removes_the_analysis_and_its_measurements(wired, client):
    submitted = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS).json()
    work_the_queue(wired)
    analysis_id = submitted["analysis_id"]

    response = client.delete(f"/api/vision-lab/analysis/{analysis_id}",
                             headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert wired.collection.docs == {}
    assert wired.measurements.docs == {}
    assert client.get(f"/api/vision-lab/analysis/{analysis_id}",
                      headers=HEADERS).status_code == 404


# =========================================================================== #
# Degradation
# =========================================================================== #
def test_a_disabled_feature_returns_503_not_a_broken_endpoint(wired, client, monkeypatch):
    monkeypatch.setattr(app_module, "VL_ENABLED", False)
    response = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS)
    assert response.status_code == 503
    assert "not enabled" in response.json()["detail"].lower()


def test_unconfigured_storage_returns_503(client, monkeypatch):
    monkeypatch.setattr(app_module, "VISION_LAB_AVAILABLE", True)
    monkeypatch.setattr(app_module, "VL_ENABLED", True)
    monkeypatch.setattr(app_module, "vision_lab_store", None)
    response = client.post("/api/vision-lab/analyze", json=BODY, headers=HEADERS)
    assert response.status_code == 503
    assert "MONGODB_URI" in response.json()["detail"]


def test_health_reports_vision_lab_without_leaking_anything(wired, client):
    payload = client.get("/health").json()

    assert payload["ok"] is True
    assert payload["vision_lab"]["available"] is True
    assert payload["vision_lab"]["storage"] == "configured"
    assert "worker" in payload["vision_lab"]
    # Booleans and names only - never a key, a URI or a URL.
    body = client.get("/health").text
    assert "mongodb" not in body.lower()
    assert "amazonaws" not in body.lower()


# =========================================================================== #
# The rest of the service
# =========================================================================== #
def test_vision_lab_did_not_disturb_the_existing_endpoints(client):
    paths = {route.path for route in app_module.app.routes}
    for existing in ("/health",
                     "/api/brand-brain/rewrite-persona",
                     "/api/brand-brain/suggest-funnel",
                     "/api/brand-brain/analyze-gaps",
                     "/api/script-lab/test-script",
                     "/api/sales-calls/analyze"):
        assert existing in paths, existing


def test_all_five_vision_lab_routes_are_registered(client):
    paths = {route.path for route in app_module.app.routes}
    for expected in ("/api/vision-lab/analyze",
                     "/api/vision-lab/analysis/{analysis_id}",
                     "/api/vision-lab/analysis/by-creative/{creative_id}",
                     "/api/vision-lab/history",
                     "/api/vision-lab/analysis/{analysis_id}/rescore"):
        assert expected in paths, expected
