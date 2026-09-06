"""Persistence, idempotency, and the pipeline end to end.

No network and no MongoDB: a fake async collection stands in for motor, and the
Deepgram and Gemini calls are injected. Everything the pipeline does with a
document is exercised exactly as it will run in production.

The load-bearing assertions: a repeat request costs nothing, a transcript is
never paid for twice, and a failure never produces a scorecard.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import framework as fw  # noqa: E402
from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import store as st  # noqa: E402
from sales_call_analyzer.deepgram_client import DeepgramError  # noqa: E402
from sales_call_analyzer.models import (  # noqa: E402
    AnalyzeOptions,
    AnalyzeRequest,
    AudioRef,
    CallMetadata,
    CustomerInfo,
    ProductInfo,
    RepInfo,
    SuppliedSegment,
    SuppliedTranscript,
)


# =========================================================================== #
# Fakes
# =========================================================================== #
class FakeCursor:
    def __init__(self, docs):
        self.docs = docs

    def sort(self, key, direction=1):
        self.docs = sorted(self.docs, key=lambda d: d.get(key) or datetime.min.replace(
            tzinfo=timezone.utc), reverse=direction < 0)
        return self

    async def to_list(self, length=None):
        return self.docs[:length] if length else self.docs


class FakeCollection:
    """Enough of motor's surface for the store, with real matching semantics."""

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.indexes: list = []

    def _matches(self, doc, query):
        for key, expected in query.items():
            value = doc.get(key)
            if isinstance(expected, dict):
                if "$ne" in expected and value == expected["$ne"]:
                    return False
                if "$nin" in expected and value in expected["$nin"]:
                    return False
            elif value != expected:
                return False
        return True

    async def find_one(self, query):
        for doc in self.docs.values():
            if self._matches(doc, query):
                return copy.deepcopy(doc)
        return None

    def find(self, query):
        return FakeCursor([copy.deepcopy(d) for d in self.docs.values()
                           if self._matches(d, query)])

    async def insert_one(self, doc):
        self.docs[doc["_id"]] = copy.deepcopy(doc)

    async def update_one(self, query, update, upsert=False):
        target = None
        for doc in self.docs.values():
            if self._matches(doc, query):
                target = doc
                break
        if target is None:
            if not upsert:
                return
            target = {"_id": query.get("_id")}
            self.docs[target["_id"]] = target
        target.update(update.get("$set") or {})
        for key, amount in (update.get("$inc") or {}).items():
            target[key] = (target.get(key) or 0) + amount

    async def create_index(self, *args, **kwargs):
        self.indexes.append((args, kwargs))


class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = None


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate_content(self, *, model, contents, config):
        self.calls += 1
        item = self.responses.pop(0) if self.responses else self.responses
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)
        self.aio = self


DEEPGRAM_RESPONSE = {
    "metadata": {"duration": 45.0},
    "results": {"utterances": [
        {"speaker": 0, "start": 0.0, "end": 6.0, "confidence": 0.95,
         "transcript": "Hi Meera, this is Rajan Kumar from EdTech Pro. Is now a good time?"},
        {"speaker": 1, "start": 6.4, "end": 12.0, "confidence": 0.93,
         "transcript": "Yes. My main problem is I don't have a recognised certification."},
        {"speaker": 0, "start": 12.2, "end": 18.0, "confidence": 0.9,
         "transcript": "Which certification were you looking at specifically?"},
    ]},
}


def llm_payload(cfg, rating="adequate"):
    criteria = [{"criterion_id": c["id"], "applicable": True, "rating": rating,
                 "confidence": "high", "observation": "Observed.",
                 "evidence": [{"segment_index": 0, "speaker_id": "speaker_0",
                               "quote": "this is Rajan Kumar"}]}
                for s in cfg["stages"] for c in s["criteria"]]
    return json.dumps({
        "summary": "A consultative discovery call.",
        "criteria": criteria,
        "stages": [{"stage_id": s["id"], "assessment": "Handled.", "confidence": "high"}
                   for s in cfg["stages"]],
        "highlights": [{"text": "Opened with a clear self-introduction", "type": "positive",
                        "evidence": [{"segment_index": 0, "quote": "this is Rajan Kumar"}]}],
        "objections": [{"summary": "Lacks a recognised certification",
                        "category": "fit",
                        "evidence": [{"segment_index": 1, "quote": "recognised certification"}]}],
        "weaknesses": [{"text": "Never asked about budget", "evidence": []}],
    })


@pytest.fixture(scope="module")
def cfg():
    return fw.load_framework()


@pytest.fixture
def collection():
    return FakeCollection()


@pytest.fixture
def store(collection):
    return st.AnalysisStore(collection, raw_collection=FakeCollection())


def make_request(**kwargs):
    base = dict(
        call_id="call_1", lead_id="lead_1",
        audio=AudioRef(url="https://rec.example.com/a.mp3", mime_type="audio/mpeg"),
        rep=RepInfo(name="Rajan Kumar"),
        customer=CustomerInfo(name="Meera Patel", region="Maharashtra"),
        product=ProductInfo(name="Career Accelerator", price=200000, currency="INR"),
        call_metadata=CallMetadata(direction="outbound", disposition="SQL",
                                   duration_seconds=45),
    )
    base.update(kwargs)
    return AnalyzeRequest(**base)


def make_deps(store, cfg, *, llm_responses=None, transcribe=None):
    calls = {"transcribe": 0}

    async def default_transcribe(url, **kwargs):
        calls["transcribe"] += 1
        return DEEPGRAM_RESPONSE

    async def load_brand_brain(_id):
        return {"_id": "bb1", "answers": {"businessType": "Education"},
                "context": {"brandName": "EdTech Pro"}}

    async def resolve_brand_ref(_lead_id):
        return {"brand_id": "brand_1", "brand_name": "EdTech Pro", "brand_brain_id": "bb1"}

    deps = pl.PipelineDeps(
        store=store,
        llm_client=FakeClient(llm_responses or [FakeResponse(llm_payload(cfg))]),
        llm_model="gemini-test",
        transcribe=transcribe or default_transcribe,
        load_brand_brain=load_brand_brain,
        resolve_brand_ref=resolve_brand_ref,
        transcription_model="nova-test",
    )
    deps.extra["calls"] = calls
    return deps


def run(coro):
    return asyncio.run(coro)


async def create_and_run(store, cfg, request, deps=None):
    deps = deps or make_deps(store, cfg)
    analysis_id = st.new_analysis_id()
    fingerprint = st.compute_fingerprint(
        request, framework_version=cfg["framework_version"], prompt_version="p1",
        llm_model="gemini-test", transcription_model="nova-test")
    await store.create(analysis_id=analysis_id, request=request,
                       fingerprint=fingerprint, versions={})
    report = await pl.run_analysis(analysis_id, request, deps)
    return analysis_id, report, deps


# =========================================================================== #
# Fingerprint / idempotency
# =========================================================================== #
def test_identical_requests_share_a_fingerprint(cfg):
    kwargs = dict(framework_version="sales_v1", prompt_version="p1",
                  llm_model="m", transcription_model="t")
    assert (st.compute_fingerprint(make_request(), **kwargs)
            == st.compute_fingerprint(make_request(), **kwargs))


def test_a_different_recording_is_a_different_analysis(cfg):
    kwargs = dict(framework_version="sales_v1", prompt_version="p1",
                  llm_model="m", transcription_model="t")
    other = make_request(audio=AudioRef(url="https://rec.example.com/b.mp3"))
    assert st.compute_fingerprint(make_request(), **kwargs) != st.compute_fingerprint(other, **kwargs)


def test_a_new_framework_version_is_a_different_analysis():
    base = dict(prompt_version="p1", llm_model="m", transcription_model="t")
    assert (st.compute_fingerprint(make_request(), framework_version="sales_v1", **base)
            != st.compute_fingerprint(make_request(), framework_version="sales_v2", **base))


def test_a_completed_analysis_is_reusable_but_a_failed_one_is_not():
    assert st.reusable({"status": sca.STATUS_COMPLETED}, force=False) is True
    assert st.reusable({"status": sca.STATUS_SKIPPED}, force=False) is True
    assert st.reusable({"status": sca.STATUS_FAILED}, force=False) is False
    assert st.reusable(None, force=False) is False


def test_force_reanalysis_overrides_reuse():
    assert st.reusable({"status": sca.STATUS_COMPLETED}, force=True) is False


def test_an_in_flight_job_is_reused_so_a_second_one_never_starts():
    fresh = {"status": sca.STATUS_ANALYZING, "heartbeat_at": st.now_utc()}
    assert st.reusable(fresh, force=False) is True


def test_a_stale_in_flight_job_is_not_reused():
    dead = {"status": sca.STATUS_ANALYZING,
            "heartbeat_at": st.now_utc() - timedelta(seconds=st.STALE_AFTER_SECONDS + 60)}
    assert st.is_stale(dead) is True
    assert st.reusable(dead, force=False) is False


def test_a_completed_job_is_never_stale():
    assert st.is_stale({"status": sca.STATUS_COMPLETED,
                        "heartbeat_at": st.now_utc() - timedelta(days=30)}) is False


# =========================================================================== #
# Store behaviour
# =========================================================================== #
def test_the_signed_url_is_never_persisted(store, cfg):
    request = make_request()
    doc = run(store.create(analysis_id="a1", request=request,
                           fingerprint="fp", versions={}))
    serialised = json.dumps(doc, default=str)
    assert "rec.example.com" not in serialised
    assert doc["request_snapshot"]["has_audio"] is True


def test_raw_transcript_is_not_stored_by_default(store):
    kept = run(store.save_raw_transcript("a1", DEEPGRAM_RESPONSE, enabled=None))
    assert kept is st.STORE_RAW_TRANSCRIPT is False
    assert store.raw_collection.docs == {}


def test_raw_transcript_is_stored_when_explicitly_enabled(store):
    assert run(store.save_raw_transcript("a1", DEEPGRAM_RESPONSE, enabled=True)) is True
    assert "expires_at" in store.raw_collection.docs["a1"]


def test_interrupted_jobs_are_marked_not_left_spinning(store, cfg):
    run(store.create(analysis_id="a1", request=make_request(), fingerprint="fp", versions={}))
    run(store.set_status("a1", sca.STATUS_ANALYZING))
    run(store.mark_interrupted("a1"))
    doc = run(store.get("a1"))
    assert doc["status"] == sca.STATUS_FAILED
    assert doc["reason"] == sca.PROCESSING_INTERRUPTED


def test_a_completed_job_is_not_clobbered_by_interruption_reaping(store, cfg):
    run(store.create(analysis_id="a1", request=make_request(), fingerprint="fp", versions={}))
    run(store.complete("a1", report={"ok": True}, analysis={}, processing={}))
    run(store.mark_interrupted("a1"))
    assert run(store.get("a1"))["status"] == sca.STATUS_COMPLETED


# =========================================================================== #
# The pipeline, happy path
# =========================================================================== #
def test_full_run_produces_a_scored_persisted_report(store, cfg):
    analysis_id, report, deps = run(create_and_run(store, cfg, make_request()))

    assert report.status == sca.STATUS_COMPLETED
    assert report.scores.overall is not None
    assert report.scores.overall_100 is not None
    assert len(report.scores.stages) == 6
    assert report.summary == "A consultative discovery call."
    assert report.transcript.speaker_count == 2
    assert report.call.disposition == "SQL"
    assert report.call.disposition_source == "rep_reported"
    assert report.context_used.brand_brain["available"] is True

    doc = run(store.get(analysis_id))
    assert doc["status"] == sca.STATUS_COMPLETED
    assert doc["report"]["scores"]["overall"] == report.scores.overall
    assert doc["analysis"]["criteria"]          # ratings kept for rescoring
    assert doc["attempts"] == 1


def test_speaker_roles_are_resolved_in_the_run(store, cfg):
    _id, report, _ = run(create_and_run(store, cfg, make_request()))
    roles = {s.speaker_id: s.role for s in report.transcript.speakers}
    assert roles["speaker_0"] == sca.ROLE_SALES_REP
    assert roles["speaker_1"] == sca.ROLE_CUSTOMER


def test_evidence_survives_into_the_report_with_real_timings(store, cfg):
    _id, report, _ = run(create_and_run(store, cfg, make_request()))
    opening = report.stage_evaluations[0]
    evidence = opening.criteria[0].evidence[0]
    assert evidence.verified is True
    assert evidence.start == 0.0 and evidence.end == 6.0
    assert report.objections[0].evidence[0].segment_index == 1


def test_processing_metadata_is_recorded(store, cfg):
    _id, report, _ = run(create_and_run(store, cfg, make_request()))
    p = report.processing
    assert p.transcription_ms is not None and p.total_ms is not None
    assert p.llm_model == "gemini-test"
    assert p.transcription_model == "nova-test"
    assert p.framework_version == cfg["framework_version"]
    assert p.audio_seconds_submitted == 45.0


# =========================================================================== #
# The three input cases
# =========================================================================== #
def test_transcript_only_never_calls_deepgram(store, cfg):
    request = make_request(audio=None, transcript=SuppliedTranscript(
        text="Rajan: Hi Meera, this is Rajan Kumar.\nMeera: I need a certification."))
    _id, report, deps = run(create_and_run(store, cfg, request))
    assert deps.extra["calls"]["transcribe"] == 0
    assert report.transcript.source == "supplied_text"
    assert report.analysis_quality.degraded is True     # no timings, no tone


def test_structured_transcript_with_audio_still_skips_deepgram(store, cfg):
    request = make_request(transcript=SuppliedTranscript(segments=[
        SuppliedSegment(speaker_id="speaker_0", start=0.0, end=4.0,
                        text="Hi Meera, this is Rajan Kumar."),
        SuppliedSegment(speaker_id="speaker_1", start=4.2, end=8.0,
                        text="I need a recognised certification."),
    ]))
    _id, report, deps = run(create_and_run(store, cfg, request))
    assert deps.extra["calls"]["transcribe"] == 0
    assert report.transcript.source == "supplied_structured"


def test_plain_text_plus_audio_uses_the_audio(store, cfg):
    request = make_request(transcript=SuppliedTranscript(text="we talked about the programme"))
    _id, report, deps = run(create_and_run(store, cfg, request))
    assert deps.extra["calls"]["transcribe"] == 1
    assert report.transcript.source == "deepgram"


# =========================================================================== #
# Failures
# =========================================================================== #
def test_no_input_fails_before_spending_anything(store, cfg):
    deps = make_deps(store, cfg)
    request = make_request(audio=None, transcript=None)
    analysis_id, report, _ = run(create_and_run(store, cfg, request, deps))

    assert report.status == sca.STATUS_FAILED
    assert report.availability.reason == sca.NO_AUDIO_OR_TRANSCRIPT
    assert report.scores is None
    assert deps.extra["calls"]["transcribe"] == 0
    assert deps.llm_client.models.calls == 0


def test_deepgram_failure_is_reported_with_its_reason(store, cfg):
    async def failing(url, **kwargs):
        raise DeepgramError(sca.TRANSCRIPTION_RATE_LIMITED, "rate limited", retryable=True)

    deps = make_deps(store, cfg, transcribe=failing)
    analysis_id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.status == sca.STATUS_FAILED
    assert report.availability.reason == sca.TRANSCRIPTION_RATE_LIMITED
    assert report.scores is None
    assert run(store.get(analysis_id))["reason"] == sca.TRANSCRIPTION_RATE_LIMITED


def test_deepgram_failure_degrades_to_a_supplied_text_transcript(store, cfg):
    async def failing(url, **kwargs):
        raise DeepgramError(sca.TRANSCRIPTION_PROVIDER_ERROR, "down", retryable=True)

    request = make_request(transcript=SuppliedTranscript(
        text="Rajan: Hi Meera, this is Rajan Kumar.\nMeera: I need a certification."))
    deps = make_deps(store, cfg, transcribe=failing)
    _id, report, _ = run(create_and_run(store, cfg, request, deps))

    assert report.status == sca.STATUS_COMPLETED     # the call was not lost
    assert report.transcript.source == "supplied_text"
    assert report.analysis_quality.degraded is True


def test_an_empty_transcript_is_not_a_zero_score(store, cfg):
    async def silent(url, **kwargs):
        return {"metadata": {"duration": 3.0}, "results": {"channels": []}}

    deps = make_deps(store, cfg, transcribe=silent)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.status == sca.STATUS_FAILED
    assert report.availability.reason == sca.TRANSCRIPT_EMPTY
    assert report.scores is None


def test_llm_failure_keeps_the_transcript_so_a_retry_is_cheap(store, cfg):
    deps = make_deps(store, cfg, llm_responses=[FakeResponse("garbage"),
                                                FakeResponse("still garbage")])
    analysis_id, report, _ = run(create_and_run(store, cfg, make_request(), deps))

    assert report.status == sca.STATUS_FAILED
    assert report.availability.reason == sca.ANALYSIS_FAILED_AFTER_TRANSCRIPTION
    assert report.scores is None
    assert report.transcript is not None and report.transcript.segment_count == 3

    doc = run(store.get(analysis_id))
    assert doc["transcript"] is not None      # kept, so the retry skips Deepgram


def test_a_retry_after_an_llm_failure_does_not_re_transcribe(store, cfg):
    failing = make_deps(store, cfg, llm_responses=[FakeResponse("x"), FakeResponse("y")])
    run(create_and_run(store, cfg, make_request(), failing))
    assert failing.extra["calls"]["transcribe"] == 1

    retry = make_deps(store, cfg)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), retry))
    assert retry.extra["calls"]["transcribe"] == 0     # reused the stored transcript
    assert report.status == sca.STATUS_COMPLETED


def test_a_crash_is_recorded_rather_than_leaving_the_job_spinning(store, cfg):
    async def exploding(url, **kwargs):
        raise RuntimeError("something unexpected")

    deps = make_deps(store, cfg, transcribe=exploding)
    analysis_id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    doc = run(store.get(analysis_id))
    assert doc["status"] == sca.STATUS_FAILED
    assert doc["status"] not in sca.ACTIVE_STATUSES
    assert report.scores is None


def test_a_failed_report_still_shows_the_call_card(store, cfg):
    deps = make_deps(store, cfg, llm_responses=[FakeResponse("x"), FakeResponse("y")])
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.call is not None
    assert report.call.customer.name == "Meera Patel"
    assert report.call.disposition == "SQL"


# =========================================================================== #
# Missing context is not fatal
# =========================================================================== #
def test_a_missing_brand_brain_does_not_stop_the_analysis(store, cfg):
    deps = make_deps(store, cfg)

    async def no_brand_brain(_id):
        return None

    deps.load_brand_brain = no_brand_brain
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.status == sca.STATUS_COMPLETED
    assert report.context_used.brand_brain["available"] is False
    assert sca.NO_BRAND_BRAIN in report.context_used.missing


def test_a_brand_lookup_error_is_survivable(store, cfg):
    deps = make_deps(store, cfg)

    async def broken(_lead_id):
        raise RuntimeError("postgres down")

    deps.resolve_brand_ref = broken
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.status == sca.STATUS_COMPLETED


# =========================================================================== #
# Disposition policy
# =========================================================================== #
def test_nothing_is_skipped_under_the_shipped_configuration(store, cfg):
    request = make_request(call_metadata=CallMetadata(disposition="No Answer",
                                                      duration_seconds=3))
    _id, report, _ = run(create_and_run(store, cfg, request))
    assert report.status == sca.STATUS_COMPLETED     # no business rule configured


def test_a_configured_disposition_skip_produces_a_transcript_but_no_score(store, cfg, monkeypatch):
    configured = copy.deepcopy(cfg)
    configured["disposition_policy"]["skip_full_evaluation"] = ["No Answer"]
    configured["disposition_policy"]["confirmed"] = True
    monkeypatch.setattr(fw, "load_framework", lambda *a, **k: configured)

    request = make_request(call_metadata=CallMetadata(disposition="No Answer"))
    deps = make_deps(store, cfg)
    analysis_id, report, _ = run(create_and_run(store, cfg, request, deps))

    assert report.status == sca.STATUS_SKIPPED
    assert report.availability.reason == sca.DISPOSITION_EXCLUDED
    assert report.scores is None
    assert report.transcript.segment_count == 3      # transcript still produced
    assert deps.llm_client.models.calls == 0         # and no LLM spend
    assert run(store.get(analysis_id))["status"] == sca.STATUS_SKIPPED


# =========================================================================== #
# Concurrency
# =========================================================================== #
def test_the_concurrency_gate_bounds_simultaneous_jobs(store, cfg, monkeypatch):
    monkeypatch.setattr(pl, "_GATE", asyncio.Semaphore(1))
    live = {"now": 0, "peak": 0}

    async def slow_transcribe(url, **kwargs):
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        await asyncio.sleep(0)
        live["now"] -= 1
        return DEEPGRAM_RESPONSE

    async def two_at_once():
        await asyncio.gather(*[
            create_and_run(store, cfg,
                           make_request(call_id=f"call_{i}"),
                           make_deps(store, cfg, transcribe=slow_transcribe))
            for i in range(2)])

    run(two_at_once())
    assert live["peak"] == 1


# =========================================================================== #
# Transcript reuse must not freeze in an earlier run's role resolution
# =========================================================================== #
def test_reusing_a_transcript_re_resolves_roles_with_the_new_names(store, cfg):
    """Transcription is expensive and worth reusing. Role resolution is cheap and
    depends on the CRM names in THIS request - reusing the previous run's answer
    would ignore corrected names."""
    wrong = make_request(rep=RepInfo(name="Wrong Person"),
                         customer=CustomerInfo(name="Also Wrong"))
    _id, first, first_deps = run(create_and_run(store, cfg, wrong))
    assert first_deps.extra["calls"]["transcribe"] == 1
    # The brand name in the transcript still identifies the rep's voice, so the
    # wrong CRM name gets attached to it. That is what must be corrigible.
    first_rep = next(s for s in first.transcript.speakers if s.role == sca.ROLE_SALES_REP)
    assert first_rep.name == "Wrong Person"

    corrected = make_request(rep=RepInfo(name="Rajan Kumar"),
                             customer=CustomerInfo(name="Meera Patel"))
    _id2, second, second_deps = run(create_and_run(store, cfg, corrected))
    assert second_deps.extra["calls"]["transcribe"] == 0      # transcript reused
    rep = next(s for s in second.transcript.speakers if s.role == sca.ROLE_SALES_REP)
    assert rep.name == "Rajan Kumar"
    assert rep.role_basis == sca.ROLE_BASIS_REP_SELF_INTRO
    # Never mislabelled as something the caller asserted.
    assert all(s.role_basis != sca.ROLE_BASIS_SUPPLIED
               for s in second.transcript.speakers)
