"""The segment pass - the customer's turns re-transcribed in their own language.

Measured before it was built (scripts/sca_eval/segment_pass_experiment.py): on
the customer's turns of 8 English + regional synthetic calls, `multi` scored
57% WER; Gemini on the same clips 23%, near the best-of-three ceiling. With the
pass in the pipeline, customer WER on those calls fell from 31% to 16% overall.

No network: Gemini is a fake that returns what each test tells it to.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import framework as fw  # noqa: E402
from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import store as st  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from sales_call_analyzer.models import AnalyzeOptions, CustomerInfo, RepInfo  # noqa: E402
from transcription import segment_transcribe as seg  # noqa: E402
from test_sales_call_pipeline import (  # noqa: E402
    FakeCollection,
    create_and_run,
    make_deps,
    make_request,
    run,
)
from test_sales_call_transcript import deepgram_response  # noqa: E402

RATE = 16000


# =========================================================================== #
# The Gemini call
# =========================================================================== #
class _Usage:
    prompt_token_count = 1000
    candidates_token_count = 200
    thoughts_token_count = 300
    cached_content_token_count = None


class _Response:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = _Usage()


class FakeGemini:
    """Answers each request from `answer(n_clips)`; records what it was sent."""

    def __init__(self, answer):
        self.answer = answer
        self.requests = []
        outer = self

        class _Models:
            async def generate_content(self, model, contents, config):
                clips = [c for c in contents if not isinstance(c, str)]
                outer.requests.append({"model": model, "clips": len(clips), "contents": contents})
                result = outer.answer(len(clips))
                if isinstance(result, Exception):
                    raise result
                return _Response(result)

        class _Aio:
            models = _Models()

        self.aio = _Aio()


def _clips(*seconds):
    return [np.full(int(s * RATE), 0.1, np.float32) for s in seconds]


def _answer(texts):
    return lambda n: json.dumps({"clips": [{"n": i + 1, "text": t} for i, t in enumerate(texts[:n])]})


def test_each_clip_gets_its_own_text_and_tokens_are_counted():
    gemini = FakeGemini(_answer(["ಹೌದು", "ಸರಿ, but time ಇಲ್ಲ"]))
    result = run(seg.transcribe_clips(gemini, "m", _clips(1, 3), "kn"))
    assert result.ok and result.texts == ["ಹೌದು", "ಸರಿ, but time ಇಲ್ಲ"]
    assert (result.input_tokens, result.output_tokens, result.thinking_tokens) == (1000, 200, 300)


def test_each_clip_is_labelled_with_its_duration():
    """v2 prompt: durations stop Gemini running one clip's words into the next."""
    gemini = FakeGemini(_answer(["a", "b"]))
    run(seg.transcribe_clips(gemini, "m", _clips(1.2, 3.4), "kn"))
    labels = [c for c in gemini.requests[0]["contents"] if isinstance(c, str) and c.startswith("Clip")]
    assert labels == ["Clip 1 (1.2 s):", "Clip 2 (3.4 s):"]


def test_long_calls_are_split_into_bounded_requests():
    gemini = FakeGemini(lambda n: json.dumps({"clips": [{"n": i + 1, "text": "ಸರಿ"} for i in range(n)]}))
    result = run(seg.transcribe_clips(gemini, "m", _clips(*([2] * 95)), "kn"))
    assert result.requests == 3                                  # 40 + 40 + 15
    assert all(r["clips"] <= seg.MAX_CLIPS_PER_REQUEST for r in gemini.requests)
    assert all(t == "ಸರಿ" for t in result.texts)


def test_text_longer_than_anyone_speaks_is_refused():
    """An invented sentence on a one-second clip keeps Deepgram's words."""
    invented = "ಇದು ತುಂಬಾ ಉದ್ದವಾದ ವಾಕ್ಯ " * 10
    gemini = FakeGemini(_answer([invented, "ಸರಿ"]))
    result = run(seg.transcribe_clips(gemini, "m", _clips(1, 1), "kn"))
    assert result.texts == [None, "ಸರಿ"]
    assert result.rejected == {0: seg.REJECT_TOO_LONG}


def test_an_empty_clip_is_refused_not_blanked():
    gemini = FakeGemini(_answer(["", "ಸರಿ"]))
    result = run(seg.transcribe_clips(gemini, "m", _clips(1, 1), "kn"))
    assert result.texts[0] is None and result.rejected == {0: seg.REJECT_EMPTY}


@pytest.mark.parametrize("answer,reason", [
    (RuntimeError("503"), seg.REASON_PROVIDER_ERROR),
    ("not json", seg.REASON_UNREADABLE),
])
def test_a_failed_request_never_raises(answer, reason):
    gemini = FakeGemini(lambda n: answer)
    result = run(seg.transcribe_clips(gemini, "m", _clips(1), "kn"))
    assert result.texts == [None] and result.reason == reason


def test_an_unconfigured_client_or_language_is_a_reason():
    assert run(seg.transcribe_clips(None, "m", _clips(1), "kn")).reason == seg.REASON_NOT_CONFIGURED
    assert run(seg.transcribe_clips(object(), "m", _clips(1), "xx")).reason == seg.REASON_UNSUPPORTED_LANGUAGE


# =========================================================================== #
# Replacing the words
# =========================================================================== #
def test_new_words_keep_the_turn_and_the_original_text():
    t = tr.from_deepgram(deepgram_response([
        {"speaker": 0, "start": 0.0, "end": 4.0, "confidence": 0.9, "transcript": "Hello this is Rohan"},
        {"speaker": 1, "start": 4.5, "end": 7.0, "confidence": 0.9, "transcript": "How do?"},
    ]))
    tr.replace_segment_texts(t, {1: "ಹೌದು ಹೇಳಿ"}, "gemini_segment_pass")
    seg1 = t.segments[1]
    assert seg1.text == "ಹೌದು ಹೇಳಿ" and seg1.text_original == "How do?"
    assert seg1.text_source == "gemini_segment_pass"
    assert (seg1.start, seg1.end, seg1.speaker_id) == (4.5, 7.0, "speaker_1")
    assert t.segments[0].text_source is None
    assert t.word_count == 4 + 2


# =========================================================================== #
# The pipeline
# =========================================================================== #
@pytest.fixture(scope="module")
def cfg():
    return fw.load_framework()


@pytest.fixture
def store():
    return st.AnalysisStore(FakeCollection(), raw_collection=FakeCollection())


CALL = deepgram_response([
    {"speaker": 0, "start": 0.0, "end": 5.0, "confidence": 0.95,
     "transcript": "Hi Meera, this is Rajan Kumar calling from EdTech Pro."},
    {"speaker": 1, "start": 5.5, "end": 8.0, "confidence": 0.9, "transcript": "How do? Hailey."},
    {"speaker": 0, "start": 8.5, "end": 14.0, "confidence": 0.95,
     "transcript": "Great. What is driving your interest right now?"},
    {"speaker": 1, "start": 14.5, "end": 18.0, "confidence": 0.9, "transcript": "Time is tight money bag."},
], duration=20.0)


class _Audio:
    async def channel_count(self, audio):
        return 1

    async def decode(self, audio, channels=1):
        return np.zeros((20 * RATE, 1), np.float32)

    async def window(self, audio, **kwargs):
        return None


def pass_deps(store, cfg, monkeypatch, *, mode=pl.SEGMENT_PASS_ON, language="kn", share=30,
              answer=None, sent=None, scope=pl.SEGMENT_SCOPE_CUSTOMER):
    seen = {"clips": None, "language": None, "calls": 0}

    async def transcribe(url, **kwargs):
        return CALL

    async def fetch_audio(url):
        return b"audio", "audio/mpeg"

    async def identify_language(audio, *, mime_type="audio/mpeg"):
        from transcription import language_id as lid
        return lid.LanguageDecision(ok=True, dominant_non_english=language,
                                    english_share=100 - share, dominant_share=share,
                                    languages=[{"code": "en", "share_percent": 100 - share},
                                               {"code": language, "share_percent": share}])

    async def segment_transcribe(clips, lang, keyterms=None):
        seen["calls"] += 1
        seen["clips"], seen["language"] = len(clips), lang
        seen["keyterms"] = keyterms
        seen["clip_seconds"] = [len(c) / RATE for c in clips]
        texts = answer(len(clips)) if answer else [f"ಉತ್ತರ {i}" for i in range(len(clips))]
        return seg.ClipResult(ok=True, texts=texts, requests=1, input_tokens=500,
                              output_tokens=50, thinking_tokens=100)

    monkeypatch.setattr(pl, "_audio_slice", _Audio())
    deps = make_deps(store, cfg, transcribe=transcribe)
    deps.fetch_audio = fetch_audio
    deps.identify_language = identify_language
    deps.language_id_mode = pl.LANGUAGE_ID_ON
    deps.segment_transcribe = segment_transcribe
    deps.segment_pass_mode = mode
    deps.segment_pass_model = "gemini-test"
    deps.segment_pass_scope = scope
    deps.extra["seen"] = seen
    return deps


def test_only_the_customers_turns_are_re_transcribed(store, cfg, monkeypatch):
    deps = pass_deps(store, cfg, monkeypatch)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    t = report.transcript
    assert deps.extra["seen"]["clips"] == 2 and deps.extra["seen"]["language"] == "kn"
    rep_turns = [s for s in t.segments if s.speaker_id == "speaker_0"]
    cust_turns = [s for s in t.segments if s.speaker_id == "speaker_1"]
    assert all(s.text_source is None for s in rep_turns)          # the rep's English is kept
    assert [s.text for s in cust_turns] == ["ಉತ್ತರ 0", "ಉತ್ತರ 1"]
    assert cust_turns[0].text_original == "How do? Hailey."
    info = report.processing.segment_pass
    assert info["applied"] is True and info["replaced"] == 2 and info["rep_excluded"] is True


def test_all_scope_re_transcribes_the_reps_turns_too(store, cfg, monkeypatch):
    """testaudio/mycall4.mp3: the rep switched into Marathi as well, and his turns
    stayed garbled under the customer-only design."""
    deps = pass_deps(store, cfg, monkeypatch, scope=pl.SEGMENT_SCOPE_ALL)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["seen"]["clips"] == 4
    assert all(s.text_source for s in report.transcript.segments)
    assert report.processing.segment_pass["scope"] == "all"
    assert report.processing.segment_pass["rep_excluded"] is False


def test_mixed_scope_adds_only_the_reps_code_switched_turns(store, cfg, monkeypatch):
    deps = pass_deps(store, cfg, monkeypatch, scope=pl.SEGMENT_SCOPE_MIXED)
    mixed = json.loads(json.dumps(CALL))
    mixed["results"]["utterances"][2]["transcript"] = "Great. अपना अपना business growth site की platform"

    async def transcribe(url, **kwargs):
        return mixed

    deps.transcribe = transcribe
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["seen"]["clips"] == 3            # 2 customer + the rep's Hindi-script turn
    rep_turns = [s for s in report.transcript.segments if s.speaker_id == "speaker_0"]
    assert [bool(s.text_source) for s in rep_turns] == [False, True]


def test_the_names_go_to_gemini_too(store, cfg, monkeypatch):
    """Gemini wrote "ScaleCRM" for the brand Deepgram and Sarvam heard as ScaleSerum."""
    deps = pass_deps(store, cfg, monkeypatch)
    run(create_and_run(store, cfg, make_request(), deps))
    assert "EdTech Pro" in deps.extra["seen"]["keyterms"]


def test_a_clip_reaches_into_the_following_silence_but_not_the_next_turn(store, cfg, monkeypatch):
    """A turn's last words can fall in the gap before the next turn."""
    deps = pass_deps(store, cfg, monkeypatch, scope=pl.SEGMENT_SCOPE_ALL)
    run(create_and_run(store, cfg, make_request(), deps))
    # turn 1: 5.5-8.0 -> 5.35 (small pad) to 8.25 (0.25 s before the next turn
    # at 8.5, whose first syllable Deepgram times late): not 9.0
    # turn 3 (last): 14.5-18.0 -> 14.35 to 19.0 (a full second of silence)
    seconds = deps.extra["seen"]["clip_seconds"]
    assert seconds[1] == pytest.approx(8.25 - 5.35, abs=0.01)
    assert seconds[3] == pytest.approx(19.0 - 14.35, abs=0.01)


def test_its_tokens_are_on_the_bill(store, cfg, monkeypatch):
    deps = pass_deps(store, cfg, monkeypatch)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.processing.segment_pass_input_tokens == 500
    assert report.usage.tokens["segment_pass"].total == 650
    assert report.usage.total_tokens >= 650 + (report.usage.tokens["analysis"].total or 0)


def test_shadow_re_transcribes_and_changes_nothing(store, cfg, monkeypatch):
    deps = pass_deps(store, cfg, monkeypatch, mode=pl.SEGMENT_PASS_SHADOW)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    info = report.processing.segment_pass
    assert info["replaced"] == 2 and info["applied"] is False
    assert all(s.text_source is None for s in report.transcript.segments)


def test_off_by_default(store, cfg, monkeypatch):
    deps = pass_deps(store, cfg, monkeypatch, mode=None)
    monkeypatch.setattr(pl, "SEGMENT_PASS_MODE", pl.SEGMENT_PASS_OFF)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.processing.segment_pass is None and deps.extra["seen"]["calls"] == 0


@pytest.mark.parametrize("language,share,reason", [
    ("hi", 40, pl.REASON_SEGMENT_NO_REGIONAL),     # multi already handles Hindi
    ("kn", 5, pl.REASON_SEGMENT_NO_REGIONAL),      # too little to matter
])
def test_it_runs_only_for_a_regional_language(store, cfg, monkeypatch, language, share, reason):
    deps = pass_deps(store, cfg, monkeypatch, language=language, share=share)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.processing.segment_pass["reason"] == reason
    assert deps.extra["seen"]["calls"] == 0


def test_a_call_already_sent_in_the_regional_language_is_left_alone(store, cfg, monkeypatch):
    deps = pass_deps(store, cfg, monkeypatch, share=90)          # >= 80%: whole call sent as kn
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.processing.transcription_language_sent == "kn"
    assert report.processing.segment_pass["reason"] == pl.REASON_SEGMENT_WHOLE_CALL


def test_a_supplied_regional_hint_is_enough(store, cfg, monkeypatch):
    """The caller says Marathi; detection is skipped but the pass still knows."""
    deps = pass_deps(store, cfg, monkeypatch)
    request = make_request(options=AnalyzeOptions(language_hint="mr"))
    _id, report, _ = run(create_and_run(store, cfg, request, deps))
    assert report.processing.segment_pass["reason"] == pl.REASON_SEGMENT_WHOLE_CALL


def test_with_no_rep_identified_every_turn_is_re_transcribed(store, cfg, monkeypatch):
    """The real Marathi test call: the CRM names were wrong, nobody was
    identified, and every turn was re-transcribed - the rep's English came back
    correct too."""
    deps = pass_deps(store, cfg, monkeypatch)

    async def no_brand(_lead_id):
        return {}

    deps.resolve_brand_ref = no_brand
    request = make_request(rep=RepInfo(name="Nobody Named"),
                           customer=CustomerInfo(name="Someone Else"))
    _id, report, _ = run(create_and_run(store, cfg, request, deps))
    assert report.processing.segment_pass["rep_excluded"] is False
    assert deps.extra["seen"]["clips"] == 4


def test_a_refused_clip_keeps_deepgrams_words(store, cfg, monkeypatch):
    deps = pass_deps(store, cfg, monkeypatch, answer=lambda n: [None, "ಸರಿ"])
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    cust = [s for s in report.transcript.segments if s.speaker_id == "speaker_1"]
    assert cust[0].text == "How do? Hailey." and cust[0].text_source is None
    assert cust[1].text == "ಸರಿ"


def test_a_reused_transcript_is_not_re_transcribed_again(store, cfg, monkeypatch):
    deps = pass_deps(store, cfg, monkeypatch)
    run(create_and_run(store, cfg, make_request(), deps))
    request = make_request(options=AnalyzeOptions(force_reanalysis=True))
    _id, report, _ = run(create_and_run(store, cfg, request, deps))
    assert deps.extra["seen"]["calls"] == 1                      # only the first run
    assert any(s.text_source for s in report.transcript.segments)  # its words are kept
