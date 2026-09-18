"""Choosing the language a call is transcribed in.

The problem being tested: Deepgram picks ONE language per file, our reps open in
English and do most of the talking, so its own detector answers "en" and the
customer's Hindi or Marathi is lost. The language therefore comes from the
caller, or from a model that listens to the audio - never from Deepgram.

No network, no ffmpeg, no Gemini: identification is injected, exactly like
transcription.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import framework as fw  # noqa: E402
from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import store as st  # noqa: E402
from sales_call_analyzer.models import AnalyzeOptions, SuppliedTranscript  # noqa: E402
from transcription import audio_slice  # noqa: E402
from transcription import deepgram_client as dg  # noqa: E402
from transcription import language_id as lid  # noqa: E402
from test_sales_call_pipeline import (  # noqa: E402
    DEEPGRAM_RESPONSE,
    FakeCollection,
    create_and_run,
    make_deps,
    make_request,
    run,
)


@pytest.fixture(scope="module")
def cfg():
    return fw.load_framework()


@pytest.fixture
def store():
    return st.AnalysisStore(FakeCollection(), raw_collection=FakeCollection())


# =========================================================================== #
# What the model is asked, and what comes back
# =========================================================================== #
class FakeResponse:
    def __init__(self, text, usage=None, model_version=None):
        self.text = text
        self.usage_metadata = usage
        self.model_version = model_version


class FakeUsage:
    prompt_token_count = 552
    candidates_token_count = 51
    thoughts_token_count = 294
    cached_content_token_count = None


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeGemini:
    def __init__(self, response):
        self.aio = type("Aio", (), {"models": FakeModels(response)})()


MARATHI = '{"languages": [{"code": "en", "share_percent": 52}, ' \
          '{"code": "mr", "share_percent": 48}], "dominant_non_english": "mr"}'
ENGLISH = '{"languages": [{"code": "en", "share_percent": 100}], ' \
          '"dominant_non_english": null}'


def identify(text_or_error, usage=FakeUsage(), model_version=None):
    response = (text_or_error if isinstance(text_or_error, Exception)
                else FakeResponse(text_or_error, usage, model_version))
    client = FakeGemini(response)
    decision = run(lid.identify(client, "gemini-test", b"audio-bytes"))
    return decision, client


def test_a_mixed_call_reports_both_languages_and_the_dominant_one():
    decision, _ = identify(MARATHI)
    assert decision.ok is True
    assert decision.dominant_non_english == "mr"
    assert decision.english_share == 52 and decision.dominant_share == 48


def test_an_english_call_has_no_dominant_other_language():
    decision, _ = identify(ENGLISH)
    assert decision.ok is True and decision.dominant_non_english is None


def test_english_is_never_returned_as_the_dominant_non_english_language():
    decision, _ = identify('{"languages": [], "dominant_non_english": "EN"}')
    assert decision.dominant_non_english is None


def test_usage_and_model_version_are_captured_for_billing():
    decision, _ = identify(MARATHI, model_version="gemini-3.8-flash")
    assert decision.input_tokens == 552 and decision.thinking_tokens == 294
    assert decision.model_version == "gemini-3.8-flash"
    assert decision.ms is not None


def test_a_provider_error_is_a_decision_not_an_exception():
    decision, _ = identify(RuntimeError("503 from the provider"))
    assert decision.ok is False
    assert decision.reason == lid.REASON_PROVIDER_ERROR


def test_an_unreadable_answer_is_not_usable():
    decision, _ = identify("sorry, I cannot tell")
    assert decision.ok is False and decision.reason == lid.REASON_UNREADABLE


def test_nothing_is_asked_when_there_is_no_audio_or_no_client():
    assert run(lid.identify(FakeGemini(FakeResponse(MARATHI)), "m", b"")).reason == lid.REASON_NO_AUDIO
    assert run(lid.identify(None, "m", b"x")).reason == lid.REASON_NOT_CONFIGURED


def test_the_prompt_rules_out_english_loanwords():
    """Counting 'demo' and 'payment' as English is what made the naive
    detectors answer 'en' for a Marathi call."""
    _decision, client = identify(MARATHI)
    prompt = client.aio.models.calls[0]["contents"][1]
    assert "loanword" in prompt.lower()
    assert "share" in prompt.lower()


def test_the_meta_carries_no_transcript_text():
    decision, _ = identify(MARATHI)
    meta = decision.as_meta()
    assert meta["language_detected"] == "mr"
    assert set(meta) == {
        "language_detected", "language_detection_ok", "language_detection_reason",
        "language_detection_shares", "language_detection_ms",
        "language_id_input_tokens", "language_id_output_tokens",
        "language_id_thinking_tokens", "language_id_cached_tokens"}


# =========================================================================== #
# Turning a detected language into what Deepgram is asked for
# =========================================================================== #
@pytest.mark.parametrize("detected,expected,why", [
    ("hi", None, dg.LANGUAGE_COVERED_BY_MULTI),    # multi already handles Hindi
    ("en", None, dg.LANGUAGE_COVERED_BY_MULTI),
    (None, None, dg.LANGUAGE_COVERED_BY_MULTI),
    ("mr", "mr", dg.LANGUAGE_REGIONAL),            # multi has no Marathi
    ("TA", "ta", dg.LANGUAGE_REGIONAL),
    ("ml", None, dg.LANGUAGE_UNSUPPORTED),         # Deepgram has no Malayalam
    ("xx", None, dg.LANGUAGE_UNKNOWN),
])
def test_detected_language_maps_to_a_request_deepgram_accepts(detected, expected, why):
    assert dg.language_for(detected) == (expected, why)


# =========================================================================== #
# The audio window
# =========================================================================== #
def test_the_window_starts_after_the_greeting():
    """A window from 0:00 hears the English opening and answers 'English' for a
    call that is mostly Marathi - the exact failure this avoids."""
    assert audio_slice.WINDOW_START_SECONDS > 0


def test_no_ffmpeg_means_no_window_rather_than_an_error(monkeypatch):
    monkeypatch.setattr(audio_slice, "ffmpeg_path", lambda: None)
    assert run(audio_slice.window(b"audio")) is None


def test_empty_audio_is_not_sent_to_ffmpeg():
    assert run(audio_slice.window(b"")) is None


# =========================================================================== #
# The pipeline
# =========================================================================== #
def lang_deps(store, cfg, *, decision=None, mode=pl.LANGUAGE_ID_ON, fetch_error=None,
              window=b"window-bytes"):
    """Deps whose language identification is a stub, recording what it saw."""
    seen = {"identify": 0, "language": "unset", "sample": None, "fetched": 0}

    async def transcribe(url, **kwargs):
        seen["language"] = kwargs.get("language")
        return DEEPGRAM_RESPONSE

    async def fetch_audio(url):
        seen["fetched"] += 1
        if fetch_error:
            raise fetch_error
        return b"whole-file-bytes", "audio/mpeg"

    async def identify_language(audio, *, mime_type="audio/mpeg"):
        seen["identify"] += 1
        seen["sample"] = audio
        return decision

    deps = make_deps(store, cfg, transcribe=transcribe)
    deps.fetch_audio = fetch_audio
    deps.identify_language = identify_language
    deps.language_id_mode = mode
    deps.extra["seen"] = seen
    return deps


def decision_for(code, ok=True):
    return lid.LanguageDecision(
        ok=ok, dominant_non_english=code, ms=3400,
        languages=[{"code": "en", "share_percent": 52}, {"code": code, "share_percent": 48}],
        input_tokens=552, output_tokens=51, thinking_tokens=294)


class _Slicer:
    """Stands in for the audio_slice module, so patching the pipeline never
    reaches into the real module and disables its own tests."""

    def __init__(self, result=b"window-bytes"):
        self.result = result

    async def window(self, audio, **kwargs):
        return self.result


@pytest.fixture(autouse=True)
def _window(monkeypatch):
    """ffmpeg is not required to run these tests."""
    monkeypatch.setattr(pl, "_audio_slice", _Slicer())


def test_off_by_default_so_nothing_costs_money_until_it_is_switched_on(store, cfg):
    deps = lang_deps(store, cfg, decision=decision_for("mr"), mode=pl.LANGUAGE_ID_OFF)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["seen"]["identify"] == 0
    assert deps.extra["seen"]["fetched"] == 0
    assert report.processing.language_basis == sca.LANGUAGE_BASIS_DEFAULT


def test_a_detected_regional_language_is_what_deepgram_is_asked_for(store, cfg):
    deps = lang_deps(store, cfg, decision=decision_for("mr"))
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["seen"]["language"] == "mr"
    p = report.processing
    assert p.language_basis == sca.LANGUAGE_BASIS_DETECTED
    assert p.language_detected == "mr"
    assert p.language_decision == dg.LANGUAGE_REGIONAL
    assert p.transcription_language_sent == "mr"


def test_a_hindi_call_changes_nothing_because_multi_already_covers_it(store, cfg):
    deps = lang_deps(store, cfg, decision=decision_for("hi"))
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["seen"]["language"] is None          # server default: multi
    assert report.processing.language_detected == "hi"
    assert report.processing.language_decision == dg.LANGUAGE_COVERED_BY_MULTI
    assert report.processing.language_basis == sca.LANGUAGE_BASIS_DEFAULT


def test_shadow_mode_records_the_choice_without_acting_on_it(store, cfg):
    deps = lang_deps(store, cfg, decision=decision_for("mr"), mode=pl.LANGUAGE_ID_SHADOW)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["seen"]["identify"] == 1
    assert deps.extra["seen"]["language"] is None          # nothing was changed
    assert report.processing.language_shadow_choice == "mr"
    assert report.processing.language_basis == sca.LANGUAGE_BASIS_DEFAULT


def test_a_caller_supplied_hint_wins_and_costs_no_detection(store, cfg):
    """The rep who just had the conversation beats any detector."""
    deps = lang_deps(store, cfg, decision=decision_for("mr"))
    request = make_request(options=AnalyzeOptions(language_hint="gu"))
    _id, report, _ = run(create_and_run(store, cfg, request, deps))
    assert deps.extra["seen"]["language"] == "gu"
    assert deps.extra["seen"]["identify"] == 0
    assert report.processing.language_basis == sca.LANGUAGE_BASIS_SUPPLIED


def test_a_failed_identification_leaves_the_default_and_completes(store, cfg):
    deps = lang_deps(store, cfg, decision=lid.LanguageDecision(
        ok=False, reason=lid.REASON_PROVIDER_ERROR, ms=900))
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.status == sca.STATUS_COMPLETED
    assert deps.extra["seen"]["language"] is None
    assert report.processing.language_detection_reason == lid.REASON_PROVIDER_ERROR


def test_audio_that_cannot_be_fetched_never_fails_the_analysis(store, cfg):
    deps = lang_deps(store, cfg, decision=decision_for("mr"),
                     fetch_error=RuntimeError("the signed URL expired"))
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.status == sca.STATUS_COMPLETED
    assert deps.extra["seen"]["language"] is None
    assert (report.processing.language_detection_reason
            == pl.REASON_LANGUAGE_ID_AUDIO_UNAVAILABLE)


def test_a_window_is_sent_rather_than_the_whole_recording(store, cfg):
    deps = lang_deps(store, cfg, decision=decision_for("mr"))
    run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["seen"]["sample"] == b"window-bytes"


def test_without_ffmpeg_a_short_file_is_sent_whole(store, cfg, monkeypatch):
    monkeypatch.setattr(pl, "_audio_slice", _Slicer(None))
    deps = lang_deps(store, cfg, decision=decision_for("mr"))
    run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["seen"]["sample"] == b"whole-file-bytes"


def test_without_ffmpeg_a_long_file_is_left_alone(store, cfg, monkeypatch):
    """Gemini charges by audio length, so identifying an hour-long call from the
    whole file is not worth it."""
    monkeypatch.setattr(pl, "_audio_slice", _Slicer(None))
    monkeypatch.setattr(pl, "LANGUAGE_ID_MAX_WHOLE_BYTES", 4)
    deps = lang_deps(store, cfg, decision=decision_for("mr"))
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["seen"]["identify"] == 0
    assert (report.processing.language_detection_reason
            == pl.REASON_LANGUAGE_ID_FILE_TOO_LONG)


def test_a_supplied_transcript_never_triggers_detection(store, cfg):
    deps = lang_deps(store, cfg, decision=decision_for("mr"))
    request = make_request(audio=None, transcript=SuppliedTranscript(
        text="Rajan: Hi Meera, this is Rajan Kumar.\nMeera: I need a certification."))
    run(create_and_run(store, cfg, request, deps))
    assert deps.extra["seen"]["identify"] == 0


def test_identification_tokens_are_billed_not_swallowed(store, cfg):
    with_id = lang_deps(store, cfg, decision=decision_for("mr"))
    _id, detected, _ = run(create_and_run(store, cfg, make_request(), with_id))

    other = st.AnalysisStore(FakeCollection(), raw_collection=FakeCollection())
    without = lang_deps(other, cfg, decision=decision_for("mr"), mode=pl.LANGUAGE_ID_OFF)
    _id2, plain, _ = run(create_and_run(other, cfg, make_request(), without))

    assert detected.processing.language_id_input_tokens == 552
    assert plain.processing.cost.gemini.input_tokens is None      # nothing to bill
    assert detected.processing.cost.gemini.input_tokens == 552    # identification is
    assert detected.processing.cost.gemini.thinking_tokens == 294
    assert detected.processing.cost.gemini.attempts == plain.processing.cost.gemini.attempts + 1


def test_the_language_hint_is_part_of_the_idempotency_key():
    """Re-submitting a Marathi call with a hint must not return the earlier
    English-only result."""
    versions = dict(framework_version="sales_v1", prompt_version="p1",
                    llm_model="m", transcription_model="t")
    plain = st.compute_fingerprint(make_request(), **versions)
    hinted = st.compute_fingerprint(
        make_request(options=AnalyzeOptions(language_hint="mr")), **versions)
    assert plain != hinted
