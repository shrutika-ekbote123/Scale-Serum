"""Channel-aware provider cost: pricing config, per-run cost, the pipeline's
cost block, and the date-range bill endpoint.

No network, no database. Named test_sales_call_* so CI's `-k sales_call` runs it.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from datetime import datetime

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import billing  # noqa: E402
from billing import pricing as pr  # noqa: E402
from sales_call_analyzer import analyzer as an  # noqa: E402
from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from sales_call_analyzer import report as report_mod  # noqa: E402
from sales_call_analyzer.models import (  # noqa: E402
    NormalizedTranscript,
    ProcessingInfo,
    SuppliedTranscript,
)
from test_sales_call_pipeline import (  # noqa: E402
    DEEPGRAM_RESPONSE,
    FakeResponse,
    create_and_run,
    make_deps,
    make_request,
    run,
)

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import framework as fw  # noqa: E402
from sales_call_analyzer import store as st  # noqa: E402
from test_sales_call_pipeline import FakeClient, FakeCollection  # noqa: E402


@pytest.fixture(scope="module")
def pricing():
    return billing.load_pricing()


@pytest.fixture(scope="module")
def cfg():
    return fw.load_framework()


@pytest.fixture
def store():
    return st.AnalysisStore(FakeCollection(), raw_collection=FakeCollection())


SEPT = "2026-09-15T10:00:00+00:00"


# =========================================================================== #
# Pricing config
# =========================================================================== #
def test_shipped_pricing_config_is_valid(pricing):
    assert pricing["pricing_version"]
    assert pricing["fx"]["usd_to_inr"] == 88


def test_nova2_is_a_flagged_placeholder(pricing):
    period, _ = billing.resolve_rate(pricing, "deepgram", "nova-2", SEPT)
    assert period["rates"]["monolingual"] == 0.31
    assert period["confirmed"] is False


def test_merged_channel_rule_is_one_and_unconfirmed(pricing):
    rule = pricing["deepgram"]["merged_audio_billed_channels"]
    assert rule == {**rule, "value": 1, "confirmed": False}


def _minimal():
    return copy.deepcopy(billing.load_pricing())


def test_validator_rejects_overlapping_periods():
    cfg = _minimal()
    periods = cfg["gemini"]["models"]["gemini-3.8-flash"]["periods"]
    periods[0]["effective_to"] = "2027-02-01"
    with pytest.raises(billing.PricingConfigError, match="overlap"):
        billing.validate_pricing(cfg)


def test_validator_rejects_a_null_rate_marked_confirmed():
    cfg = _minimal()
    cfg["deepgram"]["models"]["nova-3"]["periods"][0]["rates"]["monolingual"] = None
    with pytest.raises(billing.PricingConfigError, match="cannot be confirmed"):
        billing.validate_pricing(cfg)


def test_validator_rejects_an_alias_to_an_unknown_model():
    cfg = _minimal()
    cfg["gemini"]["aliases"]["gemini-flash-latest"][0]["resolves_to"] = "gemini-9"
    with pytest.raises(billing.PricingConfigError, match="not a configured model"):
        billing.validate_pricing(cfg)


def test_missing_pricing_file_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("BILLING_PRICING_PATH", str(tmp_path / "nope.json"))
    with pytest.raises(billing.PricingConfigError, match="missing"):
        billing.load_pricing(force=True)


# =========================================================================== #
# Deepgram: audio x channels x rate
# =========================================================================== #
def test_ten_minutes_mono_nova3(pricing):
    cost = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model="nova-3",
                                 audio_seconds=600, channels_processed=1, at=SEPT)
    assert cost.billed_seconds == 600
    assert cost.usd == pytest.approx(0.043333, abs=1e-6)


def test_multichannel_bills_every_processed_channel(pricing):
    merged = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model="nova-3",
                                   audio_seconds=600, channels_processed=1, at=SEPT)
    split = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model="nova-3",
                                  audio_seconds=600, channels_processed=2,
                                  multichannel_requested=True, at=SEPT)
    assert merged.billed_channels == 1 and merged.billed_seconds == 600
    assert split.billed_channels == 2 and split.billed_seconds == 1200
    assert split.billed_channels_basis == billing.BASIS_DEEPGRAM_METADATA
    assert split.usd == pytest.approx(merged.usd * 2, abs=1e-5)   # both rounded to 6 dp


def test_no_channel_count_falls_back_to_the_config_rule(pricing):
    cost = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model="nova-3",
                                 audio_seconds=600, at=SEPT)
    assert cost.billed_channels == 1
    assert cost.billed_channels_basis == billing.BASIS_CONFIG_RULE
    assert cost.billed_channels_confirmed is False


def test_multichannel_without_a_count_is_unpriced(pricing):
    cost = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model="nova-3",
                                 audio_seconds=600, multichannel_requested=True, at=SEPT)
    assert cost.billed_channels is None and cost.usd is None


def test_language_multi_uses_the_multilingual_rate(pricing):
    cost = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model="nova-3",
                                 audio_seconds=3600, language_sent="multi", at=SEPT)
    assert cost.language_variant == "multilingual"
    assert cost.usd == pytest.approx(0.31)


def test_unknown_deepgram_model_is_null_not_zero(pricing):
    cost = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model="whisper-x",
                                 audio_seconds=600, at=SEPT)
    assert cost.charged is True and cost.usd is None
    assert cost.reason == billing.REASON_RATE_NOT_CONFIGURED


def test_not_charged_is_zero_with_a_reason(pricing):
    cost = billing.deepgram_cost(pricing, outcome=billing.NOT_CHARGED,
                                 reason=billing.REASON_REUSED_TRANSCRIPT,
                                 model="nova-2", audio_seconds=None, at=SEPT)
    assert cost.usd == 0.0 and cost.charged is False
    assert cost.reason == billing.REASON_REUSED_TRANSCRIPT


# =========================================================================== #
# Gemini: tokens x date-effective rate
# =========================================================================== #
def _gemini(pricing, at, **kw):
    args = dict(model_requested="gemini-flash-latest", attempts=1,
                input_tokens=1_000_000, output_tokens=0, thinking_tokens=0,
                cached_tokens=0, at=at)
    args.update(kw)
    return billing.gemini_cost(pricing, **args)


def test_rate_changes_exactly_on_2027_01_01(pricing):
    before = _gemini(pricing, "2026-12-31T23:59:00+00:00")
    after = _gemini(pricing, "2027-01-01T00:00:00+00:00")
    assert before.usd == pytest.approx(0.75)
    assert after.usd == pytest.approx(1.50)


def test_thinking_is_billed_at_the_output_rate(pricing):
    cost = _gemini(pricing, SEPT, input_tokens=0, output_tokens=500_000,
                   thinking_tokens=500_000)
    assert cost.usd == pytest.approx(3.75)


def test_cached_tokens_are_subtracted_from_input(pricing):
    cost = _gemini(pricing, SEPT, input_tokens=1_000_000, cached_tokens=400_000)
    assert cost.usd == pytest.approx(0.6 * 0.75 + 0.4 * 0.075)


def test_alias_is_resolved_and_flagged(pricing):
    cost = _gemini(pricing, SEPT)
    assert cost.model_priced == "gemini-3.8-flash"
    assert cost.alias_confirmed is False


def test_provider_reported_version_wins_over_the_alias(pricing):
    cost = _gemini(pricing, SEPT, model_version="gemini-3.8-flash")
    assert cost.model_priced == "gemini-3.8-flash"
    assert cost.alias_confirmed is None


def test_unknown_gemini_model_nulls_total_but_keeps_partial(pricing):
    dg = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model="nova-3",
                               audio_seconds=3600, at=SEPT)
    gm = _gemini(pricing, SEPT, model_requested="gemini-9")
    total = billing.combine(pricing, dg, gm, at=SEPT)
    assert gm.usd is None
    assert total.total_usd is None and total.total_inr is None
    assert total.priced_usd_partial == pytest.approx(0.26)
    assert total.confirmed is False


def test_inr_uses_the_fixed_rate(pricing):
    dg = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model="nova-3",
                               audio_seconds=3600, at=SEPT)
    gm = billing.gemini_cost(pricing, model_requested=None, at=SEPT)
    total = billing.combine(pricing, dg, gm, at=SEPT)
    assert total.total_usd == pytest.approx(0.26)
    assert total.total_inr == pytest.approx(0.26 * 88, abs=0.01)
    assert billing.NOTE_FX_FIXED_RATE in total.notes
    assert billing.NOTE_BILLED_CHANNELS_UNCONFIRMED in total.notes


# =========================================================================== #
# Transcript + analyzer capture what billing needs
# =========================================================================== #
def test_channels_processed_is_read_from_deepgram_metadata():
    stereo = copy.deepcopy(DEEPGRAM_RESPONSE)
    stereo["metadata"]["channels"] = 2
    assert tr.from_deepgram(stereo).channels_processed == 2
    no_meta = {"metadata": {"duration": 1.0},
               "results": {"channels": [{}, {}], "utterances": []}}
    assert tr.from_deepgram(no_meta).channels_processed == 2
    assert tr.from_deepgram({"metadata": {}, "results": {"utterances": []}}).channels_processed is None


def test_old_stored_transcripts_still_load():
    stored = tr.from_deepgram(DEEPGRAM_RESPONSE).model_dump(mode="json")
    stored.pop("channels_processed")
    assert NormalizedTranscript(**stored).channels_processed is None


class _Usage:
    def __init__(self, i, o, t):
        self.prompt_token_count = i
        self.candidates_token_count = o
        self.thoughts_token_count = t
        self.total_token_count = i + o + t
        self.cached_content_token_count = None


def _usage_response(text, i, o, t, version="gemini-3.8-flash-001"):
    response = FakeResponse(text)
    response.usage_metadata = _Usage(i, o, t)
    response.model_version = version
    return response


def _analyze(cfg, responses):
    from test_sales_call_analyzer import make_context, make_transcript
    client = FakeClient(responses)
    return run(an.analyze(client, "gemini-flash-latest", make_context(),
                          make_transcript(), cfg, fw.load_signals()))


def test_rejected_attempt_tokens_are_summed(cfg):
    from test_sales_call_pipeline import llm_payload
    outcome = _analyze(cfg, [_usage_response("not json", 100, 10, 5),
                             _usage_response(llm_payload(cfg), 200, 20, 7)])
    meta = outcome["meta"]
    assert meta["llm_attempts"] == 2
    assert (meta["llm_input_tokens"], meta["llm_output_tokens"],
            meta["llm_thinking_tokens"]) == (300, 30, 12)
    assert meta["llm_model_version"] == "gemini-3.8-flash-001"


def test_a_failed_analysis_still_reports_its_tokens(cfg):
    with pytest.raises(an.AnalyzerError) as caught:
        _analyze(cfg, [_usage_response("bad", 100, 10, 5),
                       _usage_response("bad", 100, 10, 5)])
    assert caught.value.meta["llm_input_tokens"] == 200
    assert caught.value.meta["llm_attempts"] == 2


# =========================================================================== #
# Pipeline: every exit carries a cost block
# =========================================================================== #
def test_completed_run_bills_the_channels_deepgram_processed(store, cfg):
    _id, report, _ = run(create_and_run(store, cfg, make_request()))
    p = report.processing
    assert p.channels_processed == 1 and p.billed_channels == 1
    assert p.multichannel_requested is False
    assert p.cost.deepgram.billed_seconds == 45.0
    assert p.cost.deepgram.billed_channels_basis == billing.BASIS_DEEPGRAM_METADATA
    assert p.transcript_strategy == tr.STRATEGY_TRANSCRIBE_AUDIO
    doc = run(store.get(_id))
    assert doc["processing"]["cost"]["deepgram"]["billed_channels"] == 1


def test_multichannel_doubles_billed_seconds(store, cfg, monkeypatch):
    monkeypatch.setattr(pl._dg_client, "MULTICHANNEL", True)
    split = copy.deepcopy(DEEPGRAM_RESPONSE)
    split["metadata"]["channels"] = 2

    async def transcribe(url, **kwargs):
        return split

    deps = make_deps(store, cfg, transcribe=transcribe)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.processing.multichannel_requested is True
    assert report.processing.cost.deepgram.billed_seconds == 90.0


def test_nova_test_model_is_unpriced_not_zero(store, cfg):
    _id, report, _ = run(create_and_run(store, cfg, make_request()))
    cost = report.processing.cost
    assert cost.deepgram.usd is None and cost.total_usd is None


def test_reused_transcript_costs_nothing_on_deepgram(store, cfg):
    run(create_and_run(store, cfg, make_request()))
    _id, report, deps = run(create_and_run(store, cfg, make_request()))
    assert deps.extra["calls"]["transcribe"] == 0
    assert report.processing.cost.deepgram.usd == 0.0
    assert report.processing.cost.deepgram.reason == billing.REASON_REUSED_TRANSCRIPT


def test_supplied_transcript_costs_nothing_on_deepgram(store, cfg):
    request = make_request(audio=None, transcript=SuppliedTranscript(
        text="Rajan: Hi Meera, this is Rajan Kumar.\nMeera: I need a certification."))
    _id, report, _ = run(create_and_run(store, cfg, request))
    assert report.processing.cost.deepgram.usd == 0.0
    assert report.processing.cost.deepgram.reason == billing.REASON_SUPPLIED_TRANSCRIPT


def test_fallback_after_deepgram_failure_is_unknown_not_zero(store, cfg):
    async def failing(url, **kwargs):
        raise pl.DeepgramError(sca.TRANSCRIPTION_TIMEOUT, "timed out")

    request = make_request(transcript=SuppliedTranscript(
        text="Rajan: Hi Meera, this is Rajan Kumar.\nMeera: I need a certification."))
    deps = make_deps(store, cfg, transcribe=failing)
    _id, report, _ = run(create_and_run(store, cfg, request, deps))
    assert report.processing.cost.deepgram.charged is True
    assert report.processing.cost.deepgram.usd is None


def test_failed_analysis_carries_a_cost_block(store, cfg):
    deps = make_deps(store, cfg, llm_responses=[FakeResponse("bad"), FakeResponse("bad")])
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.status == sca.STATUS_FAILED
    assert report.processing.cost is not None
    assert report.processing.cost.gemini.attempts == 2
    assert run(store.get(_id))["processing"]["cost"]


def test_a_pricing_error_never_fails_the_analysis(store, cfg, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("pricing broke")

    monkeypatch.setattr(billing, "load_pricing", boom)
    _id, report, _ = run(create_and_run(store, cfg, make_request()))
    assert report.status == sca.STATUS_COMPLETED
    assert report.processing.cost is None


def test_a_rerun_keeps_the_earlier_runs_cost(store, cfg):
    analysis_id, _report, deps = run(create_and_run(store, cfg, make_request()))
    report = run(pl.run_analysis(analysis_id, make_request(), make_deps(store, cfg)))
    assert len(report.processing.cost_prior_runs) == 1
    assert report.processing.cost_prior_runs[0].deepgram.billed_seconds == 45.0


# =========================================================================== #
# Summary
# =========================================================================== #
def _doc(status, created, **processing):
    return {"_id": f"a_{status}_{created.isoformat()}", "status": status,
            "created_at": created, "processing": processing}


def _priced_processing(pricing, audio=600, model="nova-3", channels=1):
    dg = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model=model,
                               audio_seconds=audio, channels_processed=channels, at=SEPT)
    gm = _gemini(pricing, SEPT, input_tokens=10_000, output_tokens=1_000, thinking_tokens=500)
    return {"cost": billing.combine(pricing, dg, gm, at=SEPT).model_dump(mode="json")}


def test_summary_totals_and_groups_by_channels(pricing):
    docs = [_doc("completed", datetime(2026, 9, 2), **_priced_processing(pricing, channels=2)),
            _doc("failed", datetime(2026, 9, 3), **_priced_processing(pricing, channels=1)),
            _doc("analyzing", datetime(2026, 9, 4))]
    out = billing.summarize(docs, pricing)
    assert out["counts"]["analyses"] == 2 and out["counts"]["in_progress"] == 1
    gemini_run = (10_000 * 0.75 + 1_500 * 3.75) / 1e6
    expected = 1800 / 3600 * 0.26 + 2 * gemini_run          # 1200 s + 600 s billed
    assert out["totals"]["usd"] == pytest.approx(expected, abs=1e-5)
    assert out["totals"]["inr"] == pytest.approx(expected * 88, abs=0.02)
    rows = {r["billed_channels"]: r for r in out["by_billed_channels"]}
    assert rows[2]["billed_seconds"] == 1200 and rows[1]["billed_seconds"] == 600
    assert rows[2]["basis"] == billing.BASIS_DEEPGRAM_METADATA
    assert any("usage export" in d for d in out["disclaimers"])


def test_summary_backfills_analyses_without_a_cost_block(pricing):
    old = _doc("completed", datetime(2026, 9, 2), audio_seconds_submitted=3600.0,
               transcription_model="nova-2", llm_model="gemini-flash-latest",
               llm_attempts=1, llm_input_tokens=0, llm_output_tokens=0,
               started_at="2026-09-02T00:00:00+00:00")
    out = billing.summarize([old], pricing)
    assert out["counts"]["backfilled"] == 1
    assert out["totals"]["usd"] == pytest.approx(0.31)
    assert any("placeholder" in d for d in out["disclaimers"])


def test_summary_counts_prior_runs(pricing):
    processing = _priced_processing(pricing)
    processing["cost_prior_runs"] = [processing["cost"]]
    out = billing.summarize([_doc("completed", datetime(2026, 9, 2), **processing)], pricing)
    assert out["counts"]["runs_priced"] == 2


def test_summary_null_total_when_anything_is_unpriced(pricing):
    docs = [_doc("completed", datetime(2026, 9, 2),
                 **_priced_processing(pricing, model="unknown-model"))]
    out = billing.summarize(docs, pricing)
    assert out["totals"]["usd"] is None
    assert out["totals"]["priced_usd_partial"] > 0
    assert out["totals"]["complete"] is False


# =========================================================================== #
# The endpoint
# =========================================================================== #
@pytest.fixture
def api(monkeypatch, pricing):
    os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
    os.environ.pop("MONGODB_URI", None)
    os.environ["API_KEY"] = "test-api-key"
    from fastapi.testclient import TestClient
    import app as app_module

    collection = FakeCollection()
    store = st.AnalysisStore(collection)
    monkeypatch.setattr(app_module, "sales_call_store", store)
    monkeypatch.setattr(app_module, "SALES_CALL_ANALYZER_AVAILABLE", True)
    monkeypatch.setattr(app_module, "API_KEY", "test-api-key", raising=False)

    def add(status, created, **processing):
        doc = _doc(status, created, **(processing or _priced_processing(pricing)))
        collection.docs[doc["_id"]] = doc

    with TestClient(app_module.app) as client:
        yield client, add, app_module


HEADERS = {"X-API-Key": "test-api-key"}
URL = "/api/sales-calls/billing/summary"


def test_bill_requires_the_api_key(api):
    client, _add, _ = api
    assert client.get(URL, params={"from": "2026-09-01"}).status_code == 401


def test_bill_range_is_half_open_in_ist(api):
    client, add, _ = api
    # 2026-09-01 00:00 IST == 2026-08-31 18:30 UTC
    add("completed", datetime(2026, 8, 31, 18, 30))       # included: exactly `from`
    add("completed", datetime(2026, 8, 31, 18, 29))       # excluded: before
    add("completed", datetime(2026, 9, 30, 18, 30))       # excluded: exactly `to`
    body = client.get(URL, params={"from": "2026-09-01", "to": "2026-10-01"},
                      headers=HEADERS).json()
    assert body["counts"]["analyses"] == 1
    assert body["range"]["from_utc"] == "2026-08-31T18:30:00+00:00"


def test_bill_tz_utc_moves_the_boundary(api):
    client, add, _ = api
    add("completed", datetime(2026, 8, 31, 20, 0))        # Sept in IST, August in UTC
    ist = client.get(URL, params={"from": "2026-09-01"}, headers=HEADERS).json()
    utc = client.get(URL, params={"from": "2026-09-01", "tz": "utc"}, headers=HEADERS).json()
    assert ist["counts"]["analyses"] == 1 and utc["counts"]["analyses"] == 0
    assert utc["range"]["to"] == "2026-10-01T00:00:00+00:00"   # default: one month


def test_bill_status_filter(api):
    client, add, _ = api
    add("completed", datetime(2026, 9, 5))
    add("failed", datetime(2026, 9, 6))
    body = client.get(URL, params={"from": "2026-09-01", "status": "failed"},
                      headers=HEADERS).json()
    assert body["counts"]["analyses"] == 1 and body["counts"]["failed"] == 1


@pytest.mark.parametrize("params", [
    {"from": "2026-09-10", "to": "2026-09-01"},
    {"from": "2025-01-01", "to": "2026-09-01"},
    {"from": "not-a-date"},
    {"from": "2026-09-01", "tz": "pst"},
    {"from": "2026-09-01", "status": "queued"},
])
def test_bill_rejects_bad_ranges(api, params):
    client, _add, _ = api
    assert client.get(URL, params=params, headers=HEADERS).status_code == 422


def test_bill_returns_no_customer_data(api):
    client, add, _ = api
    add("completed", datetime(2026, 9, 5))
    text = json.dumps(client.get(URL, params={"from": "2026-09-01"}, headers=HEADERS).json())
    assert "transcript" not in text and "customer" not in text


def test_rescore_keeps_the_cost_block(api, cfg):
    client, _add, app_module = api
    store = app_module.sales_call_store
    request = make_request()
    analysis_id, report, _ = run(create_and_run(store, cfg, request))
    before = run(store.get(analysis_id))["processing"]["cost"]
    response = client.post(f"/api/sales-calls/analysis/{analysis_id}/rescore", headers=HEADERS)
    assert response.status_code == 200
    assert run(store.get(analysis_id))["processing"]["cost"] == before


# =========================================================================== #
# The usage block - tokens, audio and money in one place
# =========================================================================== #
def _used(pricing, **overrides):
    fields = dict(llm_model="gemini-flash-latest", llm_attempts=1,
                  llm_input_tokens=7688, llm_output_tokens=9665,
                  llm_thinking_tokens=1814, llm_total_tokens=19167,
                  audio_seconds_submitted=593.64, transcription_model="nova-3",
                  transcription_language_sent="multi", channels_processed=1,
                  started_at=datetime(2026, 9, 17))
    fields.update(overrides)
    processing = ProcessingInfo(**fields)
    dg = billing.deepgram_cost(
        pricing, outcome=billing.CHARGED, model=processing.transcription_model,
        audio_seconds=processing.audio_seconds_submitted,
        channels_processed=processing.channels_processed,
        language_sent=processing.transcription_language_sent, at=SEPT)
    gm = billing.gemini_cost(
        pricing, model_requested=processing.llm_model, attempts=1,
        input_tokens=processing.llm_input_tokens,
        output_tokens=processing.llm_output_tokens,
        thinking_tokens=processing.llm_thinking_tokens, at=SEPT)
    processing.cost = billing.combine(pricing, dg, gm, at=SEPT)
    processing.billed_channels = dg.billed_channels
    return processing


def test_usage_counts_every_llm_step_including_language_identification(pricing):
    """Leaving the identification call out of "total tokens" would be the same
    mistake as ignoring thinking tokens - it is a real call with a real bill."""
    usage = report_mod.build_usage(_used(
        pricing, language_id_input_tokens=3235, language_id_output_tokens=76,
        language_id_thinking_tokens=751))
    assert usage.tokens["analysis"].total == 19167
    assert usage.tokens["language_id"].total == 4062
    assert usage.total_tokens == 23229


def test_usage_omits_language_identification_when_it_did_not_run(pricing):
    usage = report_mod.build_usage(_used(pricing))
    assert "language_id" not in usage.tokens
    assert usage.total_tokens == 19167


def test_usage_reports_audio_in_the_units_deepgram_bills_in(pricing):
    usage = report_mod.build_usage(_used(pricing))
    assert usage.audio.seconds == 593.64
    assert usage.audio.minutes == 9.89
    assert usage.audio.billed_minutes == 9.89      # 1 channel, so the same
    assert usage.audio.billed_hours == 0.1649
    assert usage.audio.billed_channels == 1


def test_usage_repeats_the_money_without_recomputing_it(pricing):
    processing = _used(pricing)
    usage = report_mod.build_usage(processing)
    assert usage.cost.total_usd == processing.cost.total_usd
    assert usage.cost.deepgram_usd == processing.cost.deepgram.usd
    assert usage.cost.total_inr == pytest.approx(processing.cost.total_usd * 88, abs=0.01)
    assert usage.cost.estimated is True


def test_usage_never_invents_a_zero(pricing):
    """Nothing reported is not the same as nothing used."""
    usage = report_mod.build_usage(ProcessingInfo())
    assert usage.total_tokens is None
    assert usage.tokens["analysis"].total is None
    assert usage.audio.minutes is None
    assert usage.cost.total_usd is None


def test_a_failed_analysis_still_reports_what_it_used(store, cfg):
    deps = make_deps(store, cfg, llm_responses=[FakeResponse("bad"), FakeResponse("bad")])
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.status == sca.STATUS_FAILED
    assert report.usage is not None
    assert report.usage.audio.billed_minutes is not None


def test_the_completed_report_carries_the_usage_block(store, cfg):
    _id, report, _ = run(create_and_run(store, cfg, make_request()))
    assert report.usage.audio.seconds == 45.0
    assert report.usage.cost.total_usd == report.processing.cost.total_usd


def test_the_bill_totals_tokens_and_averages_per_analysis(pricing):
    docs = [_doc("completed", datetime(2026, 9, 2), **_priced_processing(pricing)),
            _doc("completed", datetime(2026, 9, 3), **_priced_processing(pricing))]
    out = billing.summarize(docs, pricing)
    assert out["tokens"]["input"] == 20_000
    assert out["tokens"]["total"] == 20_000 + 2_000 + 1_000     # cached is inside input
    assert out["per_analysis"]["analyses"] == 2
    assert out["per_analysis"]["tokens"] == pytest.approx(11_500)
    assert out["per_analysis"]["usd"] == pytest.approx(out["totals"]["usd"] / 2, abs=1e-6)
    assert out["by_provider"][0]["billed_minutes"] == 20.0      # 2 x 600 s
