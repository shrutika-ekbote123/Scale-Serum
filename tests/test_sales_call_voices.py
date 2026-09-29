"""Speaker refinement, rep voiceprints and voice activity - Phase 1.

No model, no network: a "voice" here is a constant sample value, and the fake
embedding maps each value to a fixed direction. That makes every similarity
exact, so each test states precisely what refinement must and must not do with
the three failures measured on real diarization output:

  * two voices collapsed into one speaker
  * one voice split into two speakers after a language switch
  * a customer's one-word reply attached to the rep
"""
from __future__ import annotations

import asyncio
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import diarization as diar  # noqa: E402
from sales_call_analyzer import framework as fw  # noqa: E402
from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import speakers as spk  # noqa: E402
from sales_call_analyzer import store as st  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from sales_call_analyzer import voiceprints as vp  # noqa: E402
from sales_call_analyzer.models import AnalyzeOptions, CustomerInfo, RepInfo  # noqa: E402
from test_sales_call_pipeline import (  # noqa: E402
    FakeCollection,
    create_and_run,
    make_deps,
    make_request,
    run,
)

RATE = 16000
REP, CUSTOMER, THIRD = 0.1, 0.2, 0.3            # sample values = voices
DIRECTIONS = {REP: np.array([1.0, 0, 0]), CUSTOMER: np.array([0, 1.0, 0]),
              THIRD: np.array([0, 0, 1.0])}


def fake_embed(samples):
    """The embedding of a slice is the mix of the voices in it."""
    if len(samples) < int(0.3 * RATE):
        return None
    v = np.zeros(3)
    for value, direction in DIRECTIONS.items():
        v += np.isclose(samples, value, atol=0.01).mean() * direction
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n else None


REP_PRINT = DIRECTIONS[REP].astype(np.float32)


def build(turns):
    """turns: [(voice, deepgram_speaker, start, [word, ...])], one word per 0.5 s.
    Returns (raw Deepgram response, samples)."""
    words, end = [], 0.0
    for voice, speaker, start, texts in turns:
        for i, text in enumerate(texts):
            s = start + i * 0.5
            words.append({"word": text.lower(), "punctuated_word": text, "start": s,
                          "end": s + 0.45, "confidence": 0.99, "speaker": speaker,
                          "_voice": voice})
            end = max(end, s + 0.5)
    samples = np.zeros(int((end + 1) * RATE), np.float32)
    for w in words:
        samples[int(w["start"] * RATE):int((w["start"] + 0.5) * RATE)] = w.pop("_voice")
    raw = {"metadata": {"duration": end + 1},
           "results": {"channels": [{"alternatives": [{"transcript": "", "words": words}]}],
                       "utterances": []}}
    return raw, samples


def speakers_by_voice(raw, samples):
    """{voice: set of speaker labels its words ended up with}."""
    out = {}
    for w in raw["results"]["channels"][0]["alternatives"][0]["words"]:
        voice = float(samples[int((w["start"] + 0.1) * RATE)])
        out.setdefault(round(voice, 1), set()).add(w["speaker"])
    return out


REP_LINE = ["Hello", "this", "is", "Rohan", "from", "ScaleSerum", "calling", "about", "your", "enquiry"]
CUST_LINE = ["Yes", "I", "remember", "please", "tell", "me", "about", "the", "price", "now"]


# =========================================================================== #
# With the rep's voiceprint
# =========================================================================== #
def test_two_voices_collapsed_into_one_speaker_are_separated():
    """Deepgram returned one speaker for a same-gender pair on 4 of 16 calls."""
    raw, samples = build([(REP, 0, 0, REP_LINE), (CUSTOMER, 0, 6, CUST_LINE),
                          (REP, 0, 12, REP_LINE), (CUSTOMER, 0, 18, CUST_LINE)])
    refined, report = diar.refine(raw, samples, fake_embed, REP_PRINT, "fake")
    by_voice = speakers_by_voice(refined, samples)
    assert report.applied and report.speakers_after == 2
    assert by_voice[REP] == {0}
    assert len(by_voice[CUSTOMER]) == 1 and by_voice[CUSTOMER] != {0}
    assert report.rep_speaker_id == "speaker_0"


def test_a_rep_split_by_a_language_switch_becomes_one_speaker():
    raw, samples = build([(REP, 0, 0, REP_LINE), (CUSTOMER, 1, 6, CUST_LINE),
                          (REP, 2, 12, REP_LINE), (CUSTOMER, 1, 18, CUST_LINE)])
    refined, report = diar.refine(raw, samples, fake_embed, REP_PRINT, "fake")
    by_voice = speakers_by_voice(refined, samples)
    assert len(by_voice[REP]) == 1
    assert by_voice[CUSTOMER] == {1}
    assert report.merged


def test_a_one_word_reply_stuck_to_the_front_of_a_rep_stretch_goes_to_the_customer():
    """Deepgram's timings leave no gap between 'Okay.' and the rep's next line."""
    raw, samples = build([(REP, 0, 0, REP_LINE), (CUSTOMER, 1, 6, CUST_LINE),
                          (CUSTOMER, 0, 12, ["Okay."]), (REP, 0, 12.5, REP_LINE)])
    refined, _ = diar.refine(raw, samples, fake_embed, REP_PRINT, "fake")
    okay = next(w for w in refined["results"]["channels"][0]["alternatives"][0]["words"]
                if w["punctuated_word"] == "Okay.")
    assert okay["speaker"] == 1


def test_an_absent_rep_changes_nothing():
    """The enrolled rep is not on this call: no voice is declared the rep."""
    raw, samples = build([(CUSTOMER, 0, 0, CUST_LINE), (THIRD, 1, 6, CUST_LINE)])
    refined, report = diar.refine(raw, samples, fake_embed, REP_PRINT, "fake")
    assert report.rep_speaker_id is None
    assert speakers_by_voice(refined, samples) == {CUSTOMER: {0}, THIRD: {1}}


def test_refinement_never_touches_words_timings_or_order():
    raw, samples = build([(REP, 0, 0, REP_LINE), (CUSTOMER, 0, 6, CUST_LINE)])
    refined, _ = diar.refine(raw, samples, fake_embed, REP_PRINT, "fake")
    before = [(w["punctuated_word"], w["start"], w["end"])
              for w in raw["results"]["channels"][0]["alternatives"][0]["words"]]
    after = [(w["punctuated_word"], w["start"], w["end"])
             for w in refined["results"]["channels"][0]["alternatives"][0]["words"]]
    assert before == after


def test_refined_output_is_a_deepgram_response_the_transcript_reads():
    raw, samples = build([(REP, 0, 0, REP_LINE), (CUSTOMER, 0, 6, CUST_LINE)])
    refined, _ = diar.refine(raw, samples, fake_embed, REP_PRINT, "fake")
    t = tr.from_deepgram(refined)
    assert t.speaker_count == 2
    assert t.segments[0].text.startswith("Hello this is Rohan")


# =========================================================================== #
# Without a voiceprint - conservative
# =========================================================================== #
def test_without_a_voiceprint_distinct_voices_are_left_alone():
    raw, samples = build([(REP, 0, 0, REP_LINE), (CUSTOMER, 1, 6, CUST_LINE),
                          (REP, 0, 12, REP_LINE)])
    refined, report = diar.refine(raw, samples, fake_embed, None, "fake")
    assert speakers_by_voice(refined, samples) == {REP: {0}, CUSTOMER: {1}}
    assert report.rep_speaker_id is None


def test_without_a_voiceprint_an_unmistakable_duplicate_is_merged():
    raw, samples = build([(REP, 0, 0, REP_LINE), (CUSTOMER, 1, 6, CUST_LINE),
                          (REP, 2, 12, REP_LINE)])
    refined, report = diar.refine(raw, samples, fake_embed, None, "fake")
    assert len(speakers_by_voice(refined, samples)[REP]) == 1
    assert report.merged


# =========================================================================== #
# Never fails an analysis
# =========================================================================== #
def test_a_failing_model_returns_the_input_with_a_reason():
    raw, samples = build([(REP, 0, 0, REP_LINE)])

    def broken(_):
        raise RuntimeError("model crashed")

    out, report = diar.refine(raw, samples, broken, REP_PRINT, "fake")
    assert out is raw
    assert report.applied is False and report.reason == diar.REASON_ERROR


def test_no_word_timings_is_a_stated_reason():
    raw = {"results": {"channels": [{"alternatives": [{"words": []}]}]}}
    out, report = diar.refine(raw, np.zeros(RATE, np.float32), fake_embed, None, "fake")
    assert out is raw and report.reason == diar.REASON_NO_WORDS


# =========================================================================== #
# Dual-channel recordings
# =========================================================================== #
def test_the_same_mix_on_both_channels_is_not_speaker_split():
    """Every 'stereo' test recording was this: L/R correlation 0.92-1.0."""
    rng = np.random.default_rng(0)
    mono = rng.normal(0, 0.1, RATE * 10).astype(np.float32)
    assert diar.stereo_profile(np.stack([mono, mono], axis=1))["speaker_split"] is False


def test_one_speaker_per_channel_is_detected_and_attributes_each_word():
    rng = np.random.default_rng(1)
    left = np.zeros(RATE * 10, np.float32)
    right = np.zeros(RATE * 10, np.float32)
    left[: RATE * 5] = rng.normal(0, 0.2, RATE * 5)
    right[RATE * 5:] = rng.normal(0, 0.2, RATE * 5)
    stereo = np.stack([left, right], axis=1)
    assert diar.stereo_profile(stereo)["speaker_split"] is True
    raw = {"results": {"channels": [{"alternatives": [{"words": [
        {"punctuated_word": "hello", "start": 1.0, "end": 1.5, "speaker": 0},
        {"punctuated_word": "yes", "start": 6.0, "end": 6.5, "speaker": 0}]}]}]}}
    out, report = diar.attribute_by_channel(raw, stereo)
    labels = [w["speaker"] for w in out["results"]["channels"][0]["alternatives"][0]["words"]]
    assert labels == [0, 1] and report.words_relabelled == 1


# =========================================================================== #
# Enrolment
# =========================================================================== #
def _voice_audio(voices_seconds):
    return np.concatenate([np.full(int(sec * RATE), v, np.float32) for v, sec in voices_seconds])


def test_a_clean_sample_enrols_with_stats():
    samples = _voice_audio([(REP, 40)])
    vector, stats = vp.build(samples, [(0, 40)], fake_embed)
    assert np.allclose(vector, REP_PRINT, atol=1e-5)
    assert stats["speech_seconds"] >= 30 and stats["consistency"] > 0.99


def test_a_short_sample_is_refused():
    with pytest.raises(vp.EnrolmentRejected) as err:
        vp.build(_voice_audio([(REP, 10)]), [(0, 10)], fake_embed)
    assert err.value.reason == vp.REJECT_TOO_SHORT


def test_a_sample_with_two_voices_is_refused():
    """A voiceprint of the wrong voice would mislabel every future call."""
    samples = _voice_audio([(REP, 20), (CUSTOMER, 20)])
    with pytest.raises(vp.EnrolmentRejected) as err:
        vp.build(samples, [(0, 40)], fake_embed)
    assert err.value.reason == vp.REJECT_TWO_VOICES


class _AsyncCollection:
    def __init__(self):
        self.docs = {}

    async def update_one(self, flt, update, upsert=False):
        doc = self.docs.setdefault(flt["_id"], {})
        doc.update(update.get("$setOnInsert", {}) if not doc else {})
        doc.update(update["$set"])

    async def find_one(self, flt):
        return self.docs.get(flt["_id"])

    async def delete_one(self, flt):
        class R:
            deleted_count = 1 if self.docs.pop(flt["_id"], None) else 0
        return R()


def test_the_store_never_returns_the_vector_and_ignores_another_models_print():
    store = vp.VoiceprintStore(_AsyncCollection())
    status = run(store.save(rep_id="rep_1", vector=REP_PRINT, model="m1", stats={},
                            consent={"confirmed": True, "recorded_by": "admin"}, source={}))
    assert status["enrolled"] and "vector" not in status
    assert run(store.vector_for("rep_1", "m1")) is not None
    assert run(store.vector_for("rep_1", "m2")) is None        # not comparable
    assert vp.status(run(store.get("rep_1")), "m2")["usable"] is False
    assert run(store.delete("rep_1")) is True
    assert run(store.vector_for("rep_1", "m1")) is None


def test_samples_combine_weighted_by_speech():
    """A rep enrolled from English and Hindi calls: the embedding shifts with
    language, so one-language voiceprints under-recognise the other language."""
    a, b = np.array([1.0, 0, 0], np.float32), np.array([0, 1.0, 0], np.float32)
    mixed = vp.combine(a, 30, b, 10)
    assert np.isclose(np.linalg.norm(mixed), 1.0)
    assert mixed[0] > mixed[1] > 0


def test_adding_a_sample_folds_it_into_the_existing_voiceprint():
    store = vp.VoiceprintStore(_AsyncCollection())
    consent = {"confirmed": True, "recorded_by": "admin"}
    run(store.save(rep_id="r", vector=DIRECTIONS[REP].astype(np.float32), model="m",
                   stats={"speech_seconds": 30}, consent=consent, source={"kind": "a"}))
    status = run(store.save(rep_id="r", vector=DIRECTIONS[CUSTOMER].astype(np.float32),
                            model="m", stats={"speech_seconds": 30}, consent=consent,
                            source={"kind": "b"}, add_to_existing=True))
    assert status["stats"]["samples"] == 2 and status["stats"]["speech_seconds"] == 60
    assert [s["kind"] for s in status["sources"]] == ["a", "b"]
    v = run(store.vector_for("r", "m"))
    assert np.isclose(v[0], v[1], atol=1e-4)                  # equal weights


def test_replacing_is_the_default():
    store = vp.VoiceprintStore(_AsyncCollection())
    consent = {"confirmed": True}
    run(store.save(rep_id="r", vector=DIRECTIONS[REP].astype(np.float32), model="m",
                   stats={"speech_seconds": 30}, consent=consent, source={}))
    run(store.save(rep_id="r", vector=DIRECTIONS[CUSTOMER].astype(np.float32), model="m",
                   stats={"speech_seconds": 30}, consent=consent, source={}))
    assert np.allclose(run(store.vector_for("r", "m")), DIRECTIONS[CUSTOMER])


# =========================================================================== #
# Roles from the voice
# =========================================================================== #
def test_a_voice_match_names_the_rep_even_when_the_crm_name_is_wrong():
    """On the real test calls the CRM rep name was often not the person speaking."""
    raw, _ = build([(REP, 0, 0, ["Hi", "Aniket", "here"]), (CUSTOMER, 1, 2, ["Yes?"])])
    t = spk.resolve_roles(tr.from_deepgram(raw), rep=RepInfo(name="Aaditya WDC"),
                          customer=CustomerInfo(name="Vikram Desai"),
                          rep_voice={"speaker_id": "speaker_0", "score": 0.8})
    roles = {s.speaker_id: s for s in t.speakers}
    assert roles["speaker_0"].role == sca.ROLE_SALES_REP
    assert roles["speaker_0"].role_basis == sca.ROLE_BASIS_VOICE_MATCH
    assert roles["speaker_0"].role_confidence == "high"
    assert roles["speaker_1"].role == sca.ROLE_CUSTOMER        # elimination


def test_a_weaker_voice_match_is_medium_confidence():
    raw, _ = build([(REP, 0, 0, ["Hello"]), (CUSTOMER, 1, 1, ["Yes"])])
    t = spk.resolve_roles(tr.from_deepgram(raw),
                          rep_voice={"speaker_id": "speaker_0", "score": 0.5})
    assert next(s for s in t.speakers if s.speaker_id == "speaker_0").role_confidence == "medium"


# =========================================================================== #
# Voice activity
# =========================================================================== #
def test_speech_kept_is_measured_against_detected_speech_not_recording_length():
    """Hold music and silence made the recording-length measure warn wrongly."""
    raw, _ = build([(REP, 0, 0, REP_LINE), (CUSTOMER, 1, 40, CUST_LINE)])
    raw["metadata"]["duration"] = 120.0
    t = tr.from_deepgram(raw)
    assert tr.WARN_LOW_COVERAGE in t.quality.warnings           # 10 s of 120 s
    t = tr.apply_vad(t, [(0, 5), (40, 45)])                    # but that IS all the speech
    assert t.quality.speech_kept > 0.9
    assert tr.WARN_LOW_COVERAGE not in t.quality.warnings


def test_speech_the_transcript_dropped_is_flagged():
    raw, _ = build([(REP, 0, 0, REP_LINE)])
    raw["metadata"]["duration"] = 60.0
    t = tr.apply_vad(tr.from_deepgram(raw), [(0, 5), (10, 50)])
    assert t.quality.speech_kept < 0.2
    assert tr.WARN_LOW_COVERAGE in t.quality.warnings


# =========================================================================== #
# The pipeline
# =========================================================================== #
@pytest.fixture(scope="module")
def cfg():
    return fw.load_framework()


@pytest.fixture
def store():
    return st.AnalysisStore(FakeCollection(), raw_collection=FakeCollection())


class _Audio:
    """Stands in for audio_slice: no ffmpeg needed."""

    def __init__(self, samples):
        self.samples = samples

    async def channel_count(self, audio):
        return 1

    async def decode(self, audio, channels=1):
        return self.samples.reshape(-1, 1)

    async def window(self, audio, **kwargs):
        return None


def voice_deps(store, cfg, monkeypatch, *, mode, voiceprint=REP_PRINT):
    raw, samples = build([(REP, 0, 0, REP_LINE), (CUSTOMER, 0, 6, CUST_LINE),
                          (REP, 0, 12, REP_LINE), (CUSTOMER, 0, 18, CUST_LINE)])

    async def transcribe(url, **kwargs):
        return raw

    async def fetch_audio(url):
        return b"audio", "audio/mpeg"

    async def load_voiceprint(rep_id):
        return voiceprint

    monkeypatch.setattr(pl, "_audio_slice", _Audio(samples))
    deps = make_deps(store, cfg, transcribe=transcribe)
    deps.fetch_audio = fetch_audio
    deps.speaker_refine_mode = mode
    deps.load_voiceprint = load_voiceprint
    deps.embed_voice = fake_embed
    deps.speech_regions = lambda s: [(0, 24)]
    deps.voice_model = "fake"
    return deps


def test_refinement_is_off_by_default(store, cfg, monkeypatch):
    deps = voice_deps(store, cfg, monkeypatch, mode=pl.SPEAKER_REFINE_OFF)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.processing.speaker_refinement is None


def test_shadow_measures_and_changes_nothing(store, cfg, monkeypatch):
    deps = voice_deps(store, cfg, monkeypatch, mode=pl.SPEAKER_REFINE_SHADOW)
    request = make_request(rep=RepInfo(id="rep_1", name="Rajan Kumar"))
    _id, report, _ = run(create_and_run(store, cfg, request, deps))
    info = report.processing.speaker_refinement
    assert info["would_apply"] is True and info["applied"] is False
    assert info["speakers_after"] == 2 and info["voiceprint_used"] is True
    assert report.transcript.speaker_count == 1                  # untouched


def test_on_separates_the_voices_and_names_the_rep_by_voice(store, cfg, monkeypatch):
    deps = voice_deps(store, cfg, monkeypatch, mode=pl.SPEAKER_REFINE_ON)
    request = make_request(rep=RepInfo(id="rep_1", name="Someone Else"))
    _id, report, _ = run(create_and_run(store, cfg, request, deps))
    assert report.processing.speaker_refinement["applied"] is True
    assert report.transcript.speaker_count == 2
    rep = next(s for s in report.transcript.speakers if s.role == sca.ROLE_SALES_REP)
    assert rep.role_basis == sca.ROLE_BASIS_VOICE_MATCH
    assert report.transcript.rep_voice["rep_id"] == "rep_1"
    assert report.transcript.quality.vad_speech_seconds == 24


def test_no_rep_id_means_no_voiceprint_and_no_guessing(store, cfg, monkeypatch):
    """Without the rep's voice, two voices under one label are NOT split: voices
    compared only with each other cannot tell a language switch from a second
    person."""
    deps = voice_deps(store, cfg, monkeypatch, mode=pl.SPEAKER_REFINE_ON)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    info = report.processing.speaker_refinement
    assert info["voiceprint_used"] is False
    assert report.transcript.rep_voice is None


def test_voice_evidence_is_not_reused_for_a_different_rep(store, cfg, monkeypatch):
    deps = voice_deps(store, cfg, monkeypatch, mode=pl.SPEAKER_REFINE_ON)
    first = make_request(rep=RepInfo(id="rep_1", name="Rajan Kumar"))
    run(create_and_run(store, cfg, first, deps))
    again = make_request(rep=RepInfo(id="rep_2", name="Rajan Kumar"),
                         options=AnalyzeOptions(force_reanalysis=True))
    _id, report, _ = run(create_and_run(store, cfg, again, deps))
    assert all(s.role_basis != sca.ROLE_BASIS_VOICE_MATCH for s in report.transcript.speakers)


def test_without_the_voice_library_the_analysis_completes_with_a_reason(store, cfg, monkeypatch):
    deps = voice_deps(store, cfg, monkeypatch, mode=pl.SPEAKER_REFINE_ON)
    deps.embed_voice = None
    monkeypatch.setattr(pl._voice, "availability", lambda: (False, "voice_model_not_installed"))
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.status == sca.STATUS_COMPLETED
    assert report.processing.speaker_refinement["reason"] == "voice_model_not_installed"
