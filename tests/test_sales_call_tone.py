"""Tone - how it was said, from the audio (Phase 3).

Measured before it was built (scripts/sca_eval/tone_eval.py), on sentences
whose WORDS carry no tone:
  * text only (what the analysis did before): exactly chance on both sets
  * CREMA-D, real actors, 5 emotions: Gemini listening 48% (chance 20%); the
    dataset's own human listeners, audio only, are reported at about 41%

No network and no model here: the acoustic measures run on generated signals,
and Gemini is a fake.
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

from sales_call_analyzer import analyzer  # noqa: E402
from sales_call_analyzer import framework as fw  # noqa: E402
from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import prosody as pr  # noqa: E402
from sales_call_analyzer import store as st  # noqa: E402
from sales_call_analyzer import tone as tn  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from test_sales_call_pipeline import (  # noqa: E402
    FakeCollection,
    create_and_run,
    make_deps,
    make_request,
    run,
)
from test_sales_call_transcript import deepgram_response  # noqa: E402

RATE = 16000


def tone_signal(seconds, f0=150.0, level=0.2, gaps=0.0):
    """A voiced-like harmonic signal at f0, with `gaps` share of silence."""
    t = np.arange(int(seconds * RATE)) / RATE
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    x = (x / np.abs(x).max() * level).astype(np.float32)
    if gaps:
        period = int(0.5 * RATE)
        for start in range(0, len(x), period):
            x[start:start + int(period * gaps)] = 0
    return x


# =========================================================================== #
# Measures
# =========================================================================== #
def test_pitch_is_measured():
    m = pr.measure_segment(tone_signal(2, f0=200), 0, 2, "one two three four")
    assert m["pitch_hz"] == pytest.approx(200, rel=0.08)


def test_louder_is_louder_and_pauses_are_pauses():
    quiet = pr.measure_segment(tone_signal(2, level=0.05), 0, 2, "a b c")
    loud = pr.measure_segment(tone_signal(2, level=0.5), 0, 2, "a b c")
    halting = pr.measure_segment(tone_signal(2, level=0.2, gaps=0.5), 0, 2, "a b c")
    assert loud["loudness_db"] > quiet["loudness_db"] + 15
    assert halting["pause_ratio"] > quiet["pause_ratio"] + 0.3


def test_fillers_and_pace_come_from_the_words():
    m = pr.measure_segment(tone_signal(2), 0, 2.0, "um I think uh maybe matlab later")
    assert m["fillers"] == 3 and m["rate_wps"] == pytest.approx(3.5)


def test_measures_are_relative_to_the_speaker_and_interruptions_found():
    samples = np.concatenate([tone_signal(3, level=0.1), tone_signal(3, level=0.1),
                              tone_signal(3, level=0.1), tone_signal(3, level=0.6)])
    t = tr.from_deepgram(deepgram_response([
        {"speaker": 0, "start": 0.0, "end": 3.0, "confidence": 0.9, "transcript": "Let me explain the plan"},
        {"speaker": 0, "start": 3.1, "end": 6.0, "confidence": 0.9, "transcript": "It covers three modules and"},
        {"speaker": 1, "start": 6.05, "end": 9.0, "confidence": 0.9, "transcript": "Wait what is the price."},
        {"speaker": 0, "start": 9.1, "end": 12.0, "confidence": 0.9, "transcript": "It is twenty thousand!"},
    ], duration=12.0), merge_gap_seconds=0)
    measures, speakers = pr.analyse(t, samples)
    by_index = {m.index: m for m in measures}
    assert by_index[3].z["loudness_db"] > 1          # louder than THIS speaker usually is
    assert by_index[2].interruption is True           # cut in after an unfinished sentence
    rep = next(s for s in speakers if s.speaker_id == "speaker_0")
    assert rep.talk_share == pytest.approx(8.8 / 11.75, rel=0.01)


# =========================================================================== #
# Which turns to listen to
# =========================================================================== #
def _call(turns):
    return tr.from_deepgram(deepgram_response(
        [{"speaker": s, "start": a, "end": b, "confidence": 0.9, "transcript": text}
         for s, a, b, text in turns], duration=60.0), merge_gap_seconds=0)


def test_price_and_objection_moments_and_the_customers_last_turns_come_first():
    t = _call([(0, 0, 4, "Hello this is Rohan from ScaleSerum"),
               (1, 5, 9, "Okay tell me what it is"),
               (0, 10, 14, "It is a leadership programme for directors"),
               (1, 15, 19, "What is the price for this programme"),
               (0, 20, 24, "It is two and a half lakhs all inclusive"),
               (1, 25, 26, "Yeah."),
               (1, 27, 31, "I will think about it and call back")])
    roles = {"speaker_0": "sales_rep", "speaker_1": "customer"}
    chosen = tn.select_moments(t, [], roles, limit=3)
    assert chosen == sorted(chosen)
    assert 3 in chosen and 6 in chosen                # price question, last customer turn
    assert 5 not in chosen                            # "Yeah." is too little to judge


def test_the_number_of_moments_is_bounded():
    t = _call([(i % 2, i * 3, i * 3 + 2.5, "what is the price of this plan") for i in range(40)])
    assert len(tn.select_moments(t, [], {}, limit=12)) == 12


# =========================================================================== #
# Listening
# =========================================================================== #
class _Usage:
    prompt_token_count, candidates_token_count, thoughts_token_count = 800, 120, 200


class FakeGemini:
    def __init__(self, answer):
        self.answer, self.sent = answer, []
        outer = self

        class _Models:
            async def generate_content(self, model, contents, config):
                outer.sent.append(contents)
                a = outer.answer(sum(1 for c in contents if not isinstance(c, str)))
                if isinstance(a, Exception):
                    raise a

                class R:
                    text = a
                    usage_metadata = _Usage()
                return R()

        class _Aio:
            models = _Models()
        self.aio = _Aio()


def _labels(*tones):
    return lambda n: json.dumps({"clips": [
        {"n": i + 1, "tone": t, "secondary": None, "intensity": "high", "confidence": "high",
         "cue": "raised pitch, clipped"} for i, t in enumerate(tones[:n])]})


def test_each_clip_is_labelled_with_role_and_words_given():
    gemini = FakeGemini(_labels("frustrated", "interested"))
    clips = [tn.ToneClip(tone_signal(2), "the customer", "I don't think this works for me"),
             tn.ToneClip(tone_signal(2), "the sales rep", "Let me show you")]
    result = run(tn.classify_clips(gemini, "m", clips))
    assert [lab["tone"] for lab in result.labels] == ["frustrated", "interested"]
    header = gemini.sent[0][0]
    assert "the customer said" in header and "I don't think this works for me" in header
    assert (result.input_tokens, result.thinking_tokens) == (800, 200)


def test_an_unknown_tone_is_dropped_not_guessed():
    gemini = FakeGemini(lambda n: json.dumps({"clips": [
        {"n": 1, "tone": "sarcastic", "intensity": "high", "confidence": "high", "cue": "x"}]}))
    result = run(tn.classify_clips(gemini, "m", [tn.ToneClip(tone_signal(2), "c", "ok")]))
    assert result.labels == [None]


@pytest.mark.parametrize("answer,reason", [(RuntimeError("503"), tn.REASON_PROVIDER_ERROR),
                                           ("{", tn.REASON_UNREADABLE)])
def test_a_failed_listen_never_raises(answer, reason):
    result = run(tn.classify_clips(FakeGemini(lambda n: answer), "m",
                                   [tn.ToneClip(tone_signal(2), "c", "ok")]))
    assert result.ok is False and result.reason == reason


@pytest.mark.parametrize("tone,z,expected", [
    ("frustrated", {"loudness_db": 1.4, "pitch_range_st": 0.9}, "supports"),
    ("frustrated", {"loudness_db": -1.2, "pitch_range_st": -0.8}, "contradicts"),
    ("hesitant", {"pause_ratio": 1.1, "rate_wps": -0.9}, "supports"),
    ("confused", {"loudness_db": 2.0}, "neutral"),          # no acoustic expectation
    ("frustrated", {"loudness_db": 0.2}, "neutral"),         # no clear difference
])
def test_whether_the_measured_voice_agrees(tone, z, expected):
    assert tn.acoustic_support(tone, z) == expected


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
    {"speaker": 1, "start": 5.5, "end": 9.0, "confidence": 0.9,
     "transcript": "What is the price for this programme?"},
    {"speaker": 0, "start": 9.5, "end": 14.0, "confidence": 0.95,
     "transcript": "It is two lakhs, and it pays for itself quickly."},
    {"speaker": 1, "start": 14.5, "end": 18.0, "confidence": 0.9,
     "transcript": "I don't think this works for me right now."},
], duration=20.0)


class _Audio:
    async def channel_count(self, audio):
        return 1

    async def decode(self, audio, channels=1):
        return tone_signal(20).reshape(-1, 1)

    async def window(self, audio, **kwargs):
        return None


def tone_deps(store, cfg, monkeypatch, *, mode=pl.TONE_ON, answer=None):
    seen = {"clips": 0, "calls": 0}

    async def transcribe(url, **kwargs):
        return CALL

    async def fetch_audio(url):
        return b"audio", "audio/mpeg"

    async def classify(clips):
        seen["calls"] += 1
        seen["clips"] = len(clips)
        labels = (answer or (lambda n: [{"tone": "frustrated", "secondary": None, "intensity": "high",
                                         "confidence": "high", "cue": "tense, raised"}] * n))(len(clips))
        return tn.ToneResult(ok=True, labels=labels, requests=1, input_tokens=900,
                             output_tokens=100, thinking_tokens=300)

    monkeypatch.setattr(pl, "_audio_slice", _Audio())
    deps = make_deps(store, cfg, transcribe=transcribe)
    deps.fetch_audio = fetch_audio
    deps.classify_tone = classify
    deps.tone_mode = mode
    deps.tone_model = "gemini-test"
    deps.extra["seen"] = seen
    return deps


def test_on_puts_how_it_was_said_in_the_report(store, cfg, monkeypatch):
    deps = tone_deps(store, cfg, monkeypatch)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    voice = report.voice
    assert voice is not None and voice.moments
    m = voice.moments[0]
    assert m.tone == "frustrated" and m.quote and m.start is not None
    assert {s.role for s in voice.speakers} == {"sales_rep", "customer"}
    assert report.transcript.voice == voice
    assert voice.tone_counts


def test_the_analysis_is_told_how_it_was_said(store, cfg, monkeypatch):
    deps = tone_deps(store, cfg, monkeypatch)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    prompt = "\n".join(analyzer.render_voice(report.transcript))
    assert "HOW IT WAS SAID" in prompt and "frustrated" in prompt
    assert "instead of guessing them from the words" in prompt


def test_without_voice_the_prompt_is_unchanged(store, cfg, monkeypatch):
    deps = tone_deps(store, cfg, monkeypatch, mode=pl.TONE_OFF)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.voice is None and analyzer.render_voice(report.transcript) == []
    assert deps.extra["seen"]["calls"] == 0 and report.processing.tone is None


def test_shadow_listens_and_changes_nothing(store, cfg, monkeypatch):
    deps = tone_deps(store, cfg, monkeypatch, mode=pl.TONE_SHADOW)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.voice is None
    assert report.processing.tone["tone_counts"] and report.processing.tone["applied"] is False


def test_its_tokens_are_on_the_bill(store, cfg, monkeypatch):
    deps = tone_deps(store, cfg, monkeypatch)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.usage.tokens["tone"].total == 1300
    assert report.processing.tone_input_tokens == 900


def test_a_contradicted_label_is_downgraded_to_low_confidence(store, cfg, monkeypatch):
    """Measured voice pointing the other way: Gemini was right 23% of the time."""
    deps = tone_deps(store, cfg, monkeypatch)
    monkeypatch.setattr(pl.tone_mod, "acoustic_support", lambda tone, z: "contradicts")
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert all(m.confidence == "low" and "disagrees" in m.cue for m in report.voice.moments)


def test_a_failed_listen_leaves_the_analysis_complete(store, cfg, monkeypatch):
    deps = tone_deps(store, cfg, monkeypatch)

    async def broken(clips):
        raise RuntimeError("down")

    deps.classify_tone = broken
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert report.status == "completed" and report.voice is None
    assert report.processing.tone["reason"] == tn.REASON_PROVIDER_ERROR


def test_a_tone_judgement_is_linked_to_the_moment_it_names():
    """Real call: opening_tone was judged from 'segment 9 sounded hesitant' but
    the evidence quote came from segment 6. The link makes it traceable."""
    from sales_call_analyzer import report as rp
    from sales_call_analyzer.models import (CriterionEvaluation, StageEvaluation,
                                            VoiceAnalysis, VoiceMoment)
    t = _call([(0, 0, 4, "Hello this is Aniket from ScaleSerum")])
    t.voice = VoiceAnalysis(moments=[VoiceMoment(segment_index=9, speaker_id="speaker_0",
                                                 tone="hesitant")])
    crit = CriterionEvaluation(criterion_id="opening_tone",
                               observation="In segment 9, Aniket sounded hesitant; segment 4 was fine.")
    rp.link_voice_moments([StageEvaluation(stage_id="opening", criteria=[crit])], t)
    assert crit.voice_moments == [9]              # 4 was not a moment heard: not linked
