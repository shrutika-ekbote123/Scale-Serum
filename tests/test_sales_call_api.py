"""The FastAPI endpoints, driven through the real app.

app.py is imported with a dummy GEMINI_API_KEY and no MONGODB_URI, then the
sales-call store is swapped for a fake collection and the pipeline's providers
are stubbed. No network, no database, no keys.

Also asserts the existing endpoints are untouched.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from datetime import timedelta

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Import app.py in a configured-but-offline state. Set before the import so the
# module-level checks see them.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.pop("MONGODB_URI", None)
os.environ["API_KEY"] = "test-api-key"

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import framework as fw  # noqa: E402
from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import store as st  # noqa: E402
from sales_call_analyzer.deepgram_client import DeepgramError  # noqa: E402
from test_sales_call_pipeline import (  # noqa: E402
    DEEPGRAM_RESPONSE,
    FakeCollection,
    FakeResponse,
    llm_payload,
    FakeClient,
)

HEADERS = {"X-API-Key": "test-api-key"}


@pytest.fixture(scope="module")
def cfg():
    return fw.load_framework()


@pytest.fixture
def wired(cfg, monkeypatch):
    """A configured analyzer with fake storage and fake providers."""
    collection = FakeCollection()
    store = st.AnalysisStore(collection, raw_collection=FakeCollection())
    monkeypatch.setattr(app_module, "sales_call_store", store)
    monkeypatch.setattr(app_module, "SALES_CALL_ANALYZER_AVAILABLE", True)
    monkeypatch.setattr(app_module, "DEEPGRAM_API_KEY", "test-deepgram-key")

    state = {"transcribe_calls": 0,
             "llm": FakeClient([FakeResponse(llm_payload(cfg))] * 20)}

    async def fake_transcribe(url, **kwargs):
        state["transcribe_calls"] += 1
        return DEEPGRAM_RESPONSE

    async def fake_brand_brain(_id):
        return {"_id": "bb1", "answers": {"businessType": "Education"},
                "context": {"brandName": "EdTech Pro"}}

    async def fake_brand_ref(_lead_id):
        return {"brand_id": "brand_1", "brand_name": "EdTech Pro", "brand_brain_id": "bb1"}

    def deps():
        return pl.PipelineDeps(
            store=store, llm_client=state["llm"], llm_model="gemini-test",
            transcribe=fake_transcribe, load_brand_brain=fake_brand_brain,
            resolve_brand_ref=fake_brand_ref, transcription_model="nova-test")

    monkeypatch.setattr(app_module, "_sales_call_deps", deps)
    state["store"] = store
    state["collection"] = collection
    return state


@pytest.fixture
def client(wired):
    with TestClient(app_module.app) as test_client:
        yield test_client


def payload(**kwargs):
    body = {
        "call_id": "call_1",
        "lead_id": "lead_1",
        "audio": {"url": "https://rec.example.com/a.mp3", "mime_type": "audio/mpeg"},
        "call_metadata": {"direction": "outbound", "disposition": "SQL",
                          "duration_seconds": 45, "remarks": "testing"},
        "rep": {"name": "Rajan Kumar"},
        "customer": {"name": "Meera Patel", "email": "meera@example.com",
                     "phone": "+919876543210", "region": "Maharashtra"},
        "product": {"name": "Career Accelerator", "price": 200000, "currency": "INR"},
    }
    body.update(kwargs)
    return body


def submit(client, **kwargs):
    return client.post("/api/sales-calls/analyze", json=payload(**kwargs), headers=HEADERS)


# =========================================================================== #
# Existing behaviour is untouched
# =========================================================================== #
def test_health_still_reports_ok_for_the_deploy_check(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["model"]


def test_health_reports_subsystems_without_leaking_the_key(client):
    body = client.get("/health").json()
    block = body["sales_call_analyzer"]
    assert block["available"] is True
    assert block["transcription"] == "configured"
    assert "test-deepgram-key" not in json.dumps(body)


def test_the_existing_endpoints_are_still_registered(client):
    paths = {route.path for route in app_module.app.routes}
    for existing in ("/api/brand-brain/rewrite-persona", "/api/brand-brain/suggest-funnel",
                     "/api/brand-brain/analyze-gaps", "/api/brand-brain/save",
                     "/api/script-lab/test-script",
                     "/api/purchase-probability/{lead_id}"):
        assert existing in paths


# =========================================================================== #
# Auth
# =========================================================================== #
def test_analyze_requires_the_api_key(client):
    assert client.post("/api/sales-calls/analyze", json=payload()).status_code == 401
    assert client.get("/api/sales-calls/analysis/whatever").status_code == 401


def test_a_wrong_key_is_rejected(client):
    response = client.post("/api/sales-calls/analyze", json=payload(),
                           headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


# =========================================================================== #
# Submit and poll
# =========================================================================== #
def test_submit_returns_an_id_immediately_then_the_report(client, wired):
    response = submit(client)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in (sca.STATUS_QUEUED, sca.STATUS_COMPLETED)
    assert body["idempotent_hit"] is False
    assert body["poll_url"] == f"/api/sales-calls/analysis/{body['analysis_id']}"
    assert body["suggested_poll_interval_seconds"] >= 1

    # TestClient runs background tasks on response close, so by now it is done.
    report = client.get(body["poll_url"], headers=HEADERS).json()
    assert report["status"] == sca.STATUS_COMPLETED
    assert report["scores"]["overall"] is not None
    assert len(report["scores"]["stages"]) == 6
    assert report["call"]["disposition"] == "SQL"
    assert report["transcript"]["speaker_count"] == 2


def test_the_report_carries_everything_the_ui_needs(client):
    body = submit(client).json()
    report = client.get(body["poll_url"], headers=HEADERS).json()

    for key in ("analysis_id", "call_id", "scores", "stage_evaluations", "summary",
                "highlights", "strengths", "weaknesses", "recommendations",
                "customer_needs", "objections", "buying_signals", "transcript",
                "context_used", "analysis_quality", "processing"):
        assert key in report, key

    assert report["call"]["customer"]["email"] == "meera@example.com"   # UI needs it
    assert report["call"]["rep"]["name"] == "Rajan Kumar"
    stage = report["stage_evaluations"][0]
    assert stage["objective"] and stage["kpis"] and stage["criteria"]
    criterion = stage["criteria"][0]
    assert criterion["name"] and criterion["rating"] and criterion["evidence"]


def test_the_response_admits_its_scoring_is_unconfigured(client):
    body = submit(client).json()
    report = client.get(body["poll_url"], headers=HEADERS).json()
    scores = report["scores"]
    assert scores["band"] is None
    assert scores["band_reason"] == "thresholds_not_configured"
    assert scores["weighting"] == "equal_unweighted_placeholder"
    assert scores["rating_scale_mode"] == "uniform_placeholder"
    assert scores["stage_weights_confirmed"] is False


def test_disposition_is_echoed_as_rep_reported_never_replaced(client):
    body = submit(client).json()
    report = client.get(body["poll_url"], headers=HEADERS).json()
    assert report["call"]["disposition"] == "SQL"
    assert report["call"]["disposition_source"] == "rep_reported"
    assert "suggested_disposition" not in report


def test_unknown_analysis_id_is_a_404(client):
    assert client.get("/api/sales-calls/analysis/nope", headers=HEADERS).status_code == 404


def test_lookup_by_call_id(client):
    submit(client)
    response = client.get("/api/sales-calls/analysis/by-call/call_1", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["call_id"] == "call_1"
    assert client.get("/api/sales-calls/analysis/by-call/absent",
                      headers=HEADERS).status_code == 404


# =========================================================================== #
# Idempotency
# =========================================================================== #
def test_an_identical_resubmission_costs_nothing(client, wired):
    first = submit(client).json()
    assert wired["transcribe_calls"] == 1

    second = submit(client).json()
    assert second["idempotent_hit"] is True
    assert second["analysis_id"] == first["analysis_id"]
    assert wired["transcribe_calls"] == 1          # no second transcription


def test_force_reanalysis_runs_again(client, wired):
    first = submit(client).json()
    again = submit(client, options={"force_reanalysis": True}).json()
    assert again["idempotent_hit"] is False
    assert again["analysis_id"] != first["analysis_id"]
    # Still no re-transcription: the stored transcript for this call is reused.
    assert wired["transcribe_calls"] == 1


def test_a_different_call_is_analysed_separately(client, wired):
    submit(client)
    other = submit(client, call_id="call_2").json()
    assert other["idempotent_hit"] is False
    assert wired["transcribe_calls"] == 2


# =========================================================================== #
# Failures
# =========================================================================== #
def test_no_audio_and_no_transcript_is_a_stated_failure_not_a_422(client, wired):
    response = submit(client, audio=None, transcript=None)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == sca.STATUS_FAILED
    assert body["reason"] == sca.NO_AUDIO_OR_TRANSCRIPT
    assert wired["transcribe_calls"] == 0

    report = client.get(body["poll_url"], headers=HEADERS).json()
    assert report["scores"] is None
    assert report["availability"]["available"] is False


def test_a_transcription_failure_reaches_the_caller_with_its_reason(client, wired,
                                                                   monkeypatch):
    async def failing(url, **kwargs):
        raise DeepgramError(sca.TRANSCRIPTION_RATE_LIMITED, "rate limited", retryable=True)

    original = app_module._sales_call_deps

    def deps():
        d = original()
        d.transcribe = failing
        return d

    monkeypatch.setattr(app_module, "_sales_call_deps", deps)
    body = submit(client).json()
    report = client.get(body["poll_url"], headers=HEADERS).json()
    assert report["status"] == sca.STATUS_FAILED
    assert report["availability"]["reason"] == sca.TRANSCRIPTION_RATE_LIMITED
    assert report["scores"] is None


def test_a_stale_job_is_reaped_on_read_rather_than_reported_as_running(client, wired):
    store = wired["store"]
    run = asyncio.get_event_loop_policy().new_event_loop().run_until_complete
    doc_id = "stuck_analysis"
    wired["collection"].docs[doc_id] = {
        "_id": doc_id, "call_id": "call_stuck", "status": sca.STATUS_ANALYZING,
        "heartbeat_at": st.now_utc() - timedelta(seconds=st.STALE_AFTER_SECONDS + 120),
        "created_at": st.now_utc() - timedelta(hours=2), "attempts": 1,
    }
    body = client.get(f"/api/sales-calls/analysis/{doc_id}", headers=HEADERS).json()
    assert body["status"] == sca.STATUS_FAILED
    assert body["availability"]["reason"] == sca.PROCESSING_INTERRUPTED
    assert body["scores"] is None


# =========================================================================== #
# Storage not configured
# =========================================================================== #
def test_missing_storage_is_a_503_not_a_crash(monkeypatch):
    monkeypatch.setattr(app_module, "sales_call_store", None)
    with TestClient(app_module.app) as unconfigured:
        response = unconfigured.post("/api/sales-calls/analyze", json=payload(),
                                     headers=HEADERS)
        assert response.status_code == 503
        assert "MONGODB_URI" in response.json()["detail"]


# =========================================================================== #
# Rescore
# =========================================================================== #
def test_rescoring_applies_a_new_rating_scale_without_calling_a_provider(
        client, wired, cfg, monkeypatch):
    body = submit(client).json()
    before = client.get(body["poll_url"], headers=HEADERS).json()["scores"]["overall"]
    calls_before = wired["transcribe_calls"]
    llm_calls_before = wired["llm"].models.calls

    updated = copy.deepcopy(cfg)
    updated["rating_scale"] = {"absent": 0.0, "weak": 0.2, "adequate": 0.5, "strong": 1.0}
    updated["framework_version"] = "sales_v2"
    monkeypatch.setattr(app_module._sca_framework, "load_framework",
                        lambda *a, **k: updated)

    rescored = client.post(
        f"/api/sales-calls/analysis/{body['analysis_id']}/rescore", headers=HEADERS)
    assert rescored.status_code == 200
    scores = rescored.json()["scores"]
    assert scores["overall"] == 5.0
    assert scores["overall"] != before
    assert scores["framework_version"] == "sales_v2"
    assert scores["rating_scale_mode"] == "configured"

    assert wired["transcribe_calls"] == calls_before        # no Deepgram
    assert wired["llm"].models.calls == llm_calls_before    # no Gemini


def test_an_incomplete_analysis_cannot_be_rescored(client, wired):
    body = submit(client, audio=None, transcript=None).json()
    response = client.post(
        f"/api/sales-calls/analysis/{body['analysis_id']}/rescore", headers=HEADERS)
    assert response.status_code == 409


# =========================================================================== #
# Swagger
# =========================================================================== #
def test_the_endpoints_are_documented(client):
    schema = client.get("/openapi.json").json()
    assert "/api/sales-calls/analyze" in schema["paths"]
    assert "/api/sales-calls/analysis/{analysis_id}" in schema["paths"]
    assert "/api/sales-calls/analysis/{analysis_id}/rescore" in schema["paths"]
