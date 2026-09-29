"""Transcription and speaker accuracy, Phase 0.

Four changes, each measured on real or synthetic calls before it was made
(scripts/sca_eval):

  * English-only calls are sent as `en`, not `multi`.
  * Keyterms (brand, product, rep, customer) go to Deepgram.
  * Dropped speech is reported: speech coverage, words per minute and a
    `low_speech_coverage` warning, because confidence cannot see it.
  * Role resolution survives other scripts and misspellings: a romanised,
    fuzzy second pass, one confidence step lower than an exact match.

Pure unit tests - no network, no keys.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import framework as fw  # noqa: E402
from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import speakers as spk  # noqa: E402
from sales_call_analyzer import store as st  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from sales_call_analyzer.models import CustomerInfo, ProductInfo, RepInfo  # noqa: E402
from transcription import deepgram_client as dg  # noqa: E402
from transcription import language_id as lid  # noqa: E402
from transcription.romanize import indic_share, romanize  # noqa: E402
from test_sales_call_pipeline import (  # noqa: E402
    DEEPGRAM_RESPONSE,
    FakeCollection,
    create_and_run,
    make_deps,
    make_request,
    run,
)
from test_sales_call_transcript import deepgram_response  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return fw.load_framework()


@pytest.fixture
def store():
    return st.AnalysisStore(FakeCollection(), raw_collection=FakeCollection())


# =========================================================================== #
# Romanisation - a matching key, never display text
# =========================================================================== #
@pytest.mark.parametrize("text,expected", [
    ("मेरा नाम राजन है", "mera nam rajan hai"),           # final inherent vowel dropped
    ("मैं स्केल सीरम से बोल रहा हूँ", "main skel siram se bol raha hun"),
    ("ಹೇಳಿ ಮಂಜುನಾಥ್", "heli manjunath"),                  # Kannada
    ("ਗੁਰਪ੍ਰੀਤ", "guraprit"),                              # Gurmukhi
    ("Hello Rajan, डेमो कब है?", "hello rajan demo kab hai"),
    ("", ""),
])
def test_romanize_gives_one_latin_key_for_any_script(text, expected):
    assert romanize(text) == expected


def test_tamil_is_romanised_without_the_broken_library_mapping():
    """sanscript turns வணக்கம் into 'vaṇaghghaṃ'; the generic path keeps the k."""
    assert romanize("வணக்கம்").startswith("va")
    assert "k" in romanize("வணக்கம்")


def test_indic_share_measures_script_not_language():
    assert indic_share("hello") == 0.0
    assert indic_share("नमस्ते") == 1.0
    assert 0 < indic_share("hello नमस्ते") < 1


# =========================================================================== #
# The language-identification window on a short call
# =========================================================================== #
@pytest.mark.skipif(not __import__("transcription.audio_slice", fromlist=["x"]).available(),
                    reason="needs ffmpeg")
def test_a_call_shorter_than_the_window_start_gives_no_window():
    """ffmpeg writes a 44-byte header past the end of the audio. Treated as a
    sample, it made Gemini fail on every call under 20 s."""
    import subprocess

    from transcription import audio_slice
    tone = subprocess.run([audio_slice.ffmpeg_path(), "-v", "error", "-f", "lavfi", "-i",
                           "sine=frequency=300:duration=8", "-f", "wav", "pipe:1"],
                          capture_output=True).stdout
    assert run(audio_slice.window(tone)) is None                  # starts at 20 s
    assert len(run(audio_slice.window(tone, start=0))) > audio_slice.MIN_WINDOW_BYTES


# =========================================================================== #
# English-only routing
# =========================================================================== #
@pytest.mark.parametrize("detected,english,expected,why", [
    (None, 98, "en", dg.LANGUAGE_ENGLISH_ONLY),       # all English -> en, not multi
    (None, 95, "en", dg.LANGUAGE_ENGLISH_ONLY),
    ("mr", 97, "en", dg.LANGUAGE_ENGLISH_ONLY),       # 3% Marathi is lost by multi too
    ("hi", 97, None, dg.LANGUAGE_COVERED_BY_MULTI),   # multi keeps the Hindi; en would not
    (None, 80, None, dg.LANGUAGE_COVERED_BY_MULTI),
    ("mr", 60, "mr", dg.LANGUAGE_REGIONAL),
    ("mr", None, "mr", dg.LANGUAGE_REGIONAL),         # no share reported: old behaviour
])
def test_a_clearly_english_call_is_sent_as_english(detected, english, expected, why):
    assert dg.language_for(detected, english) == (expected, why)


@pytest.mark.parametrize("share,expected,why", [
    (100, "mr", dg.LANGUAGE_REGIONAL),         # FLEURS: kn WER 122% -> 28% when routed
    (85, "mr", dg.LANGUAGE_REGIONAL),
    (65, None, dg.LANGUAGE_REGIONAL_MINORITY),  # real call: mr kept 252 words, multi 385
    (25, None, dg.LANGUAGE_REGIONAL_MINORITY),  # synthetic: routing took rep WER 9% -> 35%
    (None, "mr", dg.LANGUAGE_REGIONAL),         # no share reported: old routing
])
def test_a_mixed_regional_call_stays_on_multi(share, expected, why):
    """A one-language regional model cannot write English, so on a code-switched
    call it destroys the rep's half - and the self-introduction roles need."""
    english = None if share is None else 100 - share
    assert dg.language_for("mr", english, share) == (expected, why)


def test_a_mixed_regional_call_in_shadow_mode_records_the_minority(store, cfg, monkeypatch):
    seen = {}

    async def transcribe(url, **kwargs):
        seen.update(kwargs)
        return DEEPGRAM_RESPONSE

    async def fetch_audio(url):
        return b"audio", "audio/mpeg"

    async def identify_language(audio, *, mime_type="audio/mpeg"):
        return lid.LanguageDecision(ok=True, dominant_non_english="kn", english_share=70,
                                    dominant_share=30,
                                    languages=[{"code": "en", "share_percent": 70},
                                               {"code": "kn", "share_percent": 30}])

    class _Slicer:
        async def window(self, audio, **kwargs):
            return b"window"

    monkeypatch.setattr(pl, "_audio_slice", _Slicer())
    deps = make_deps(store, cfg, transcribe=transcribe)
    deps.fetch_audio, deps.identify_language = fetch_audio, identify_language
    deps.language_id_mode = pl.LANGUAGE_ID_ON
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert seen["language"] is None                          # multi
    assert report.processing.language_decision == dg.LANGUAGE_REGIONAL_MINORITY
    assert report.processing.language_detected == "kn"      # still recorded for Phase 2


def test_the_english_threshold_is_configurable(monkeypatch):
    monkeypatch.setattr(dg, "ENGLISH_ONLY_MIN_SHARE", 99)
    assert dg.language_for(None, 97) == (None, dg.LANGUAGE_COVERED_BY_MULTI)


# =========================================================================== #
# Keyterms
# =========================================================================== #
def test_keyterms_are_sent_as_a_repeated_parameter():
    params = dg.build_params(None, ["ScaleSerum", "Rohan Mehta"])
    assert params["keyterm"] == ["ScaleSerum", "Rohan Mehta"]
    assert params["language"] == "multi"


def test_no_keyterms_means_no_parameter():
    assert "keyterm" not in dg.build_params()
    assert "keyterm" not in dg.build_params(None, [])


def test_keyterms_are_cleaned_deduplicated_and_capped(monkeypatch):
    monkeypatch.setattr(dg, "MAX_KEYTERMS", 3)
    cleaned = dg.clean_keyterms(["  Scale   Serum ", "scale serum", None, "x", "A" * 200,
                                 "Rohan", "Meera"])
    assert cleaned[0] == "Scale Serum"
    assert len(cleaned) == 3
    assert len(cleaned[1]) == dg.MAX_KEYTERM_CHARS


def test_keyterms_are_never_sent_to_a_model_that_rejects_them(monkeypatch):
    """keyterm is nova-3 only; nova-2 fails the whole request."""
    monkeypatch.setattr(dg, "DEEPGRAM_MODEL", "nova-2")
    assert "keyterm" not in dg.build_params(None, ["ScaleSerum"])


def test_keyterms_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(dg, "DEEPGRAM_KEYTERMS", False)
    assert "keyterm" not in dg.build_params(None, ["ScaleSerum"])


def test_filler_words_are_off_unless_asked_for(monkeypatch):
    assert "filler_words" not in dg.build_params()
    monkeypatch.setattr(dg, "DEEPGRAM_FILLER_WORDS", True)
    assert dg.build_params()["filler_words"] == "true"


def test_request_keyterms_lead_with_brand_and_include_first_names():
    request = make_request(rep=RepInfo(name="Rohan Mehta"),
                           customer=CustomerInfo(name="Meera Joshi"),
                           product=ProductInfo(name="Career Accelerator"))
    terms = pl.transcription_keyterms(request, "ScaleSerum")
    assert terms[0] == "ScaleSerum"
    assert {"Career Accelerator", "Rohan Mehta", "Rohan", "Meera Joshi", "Meera"} <= set(terms)


def test_the_pipeline_sends_keyterms_including_the_brand(store, cfg):
    """The brand is looked up BEFORE transcription now, so its name can be a keyterm."""
    seen = {}

    async def transcribe(url, **kwargs):
        seen.update(kwargs)
        return DEEPGRAM_RESPONSE

    deps = make_deps(store, cfg, transcribe=transcribe)
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert "EdTech Pro" in seen["keyterms"]
    assert "Rajan Kumar" in seen["keyterms"]
    assert report.processing.transcription_keyterm_count == len(seen["keyterms"])


def test_the_pipeline_routes_an_english_call_to_en(store, cfg, monkeypatch):
    seen = {}

    async def transcribe(url, **kwargs):
        seen.update(kwargs)
        return DEEPGRAM_RESPONSE

    async def fetch_audio(url):
        return b"audio", "audio/mpeg"

    async def identify_language(audio, *, mime_type="audio/mpeg"):
        return lid.LanguageDecision(ok=True, dominant_non_english=None, english_share=99,
                                    languages=[{"code": "en", "share_percent": 99}])

    class _Slicer:
        async def window(self, audio, **kwargs):
            return b"window"

    monkeypatch.setattr(pl, "_audio_slice", _Slicer())
    deps = make_deps(store, cfg, transcribe=transcribe)
    deps.fetch_audio, deps.identify_language = fetch_audio, identify_language
    deps.language_id_mode = pl.LANGUAGE_ID_ON
    _id, report, _ = run(create_and_run(store, cfg, make_request(), deps))
    assert seen["language"] == "en"
    assert report.processing.language_decision == dg.LANGUAGE_ENGLISH_ONLY
    assert report.processing.transcription_language_sent == "en"


# =========================================================================== #
# Dropped speech
# =========================================================================== #
def _call(turns, duration):
    return tr.from_deepgram(deepgram_response(
        [{"speaker": s, "start": a, "end": b, "confidence": 0.99, "transcript": text}
         for s, a, b, text in turns], duration=duration))


def test_coverage_and_words_per_minute_are_measured():
    t = _call([(0, 0.0, 30.0, "one two three"), (1, 30.5, 60.0, "four five six")], 100.0)
    assert t.quality.speech_coverage == pytest.approx(0.595)
    assert t.quality.words_per_minute == pytest.approx(3.6)


def test_overlapping_turns_are_counted_once():
    t = _call([(0, 0.0, 30.0, "a"), (1, 20.0, 40.0, "b")], 100.0)
    assert t.quality.speech_coverage == pytest.approx(0.4)


def test_dropped_speech_is_reported_even_at_high_confidence():
    """mycall3.mp3: 39 s of 101 s kept at 0.99 confidence. The warning is the
    only thing that says so."""
    t = _call([(0, 0.0, 20.0, "hello good morning"), (1, 21.0, 39.0, "yes tell me")], 101.0)
    assert t.quality.mean_confidence > 0.9
    assert tr.WARN_LOW_COVERAGE in t.quality.warnings


def test_a_normal_call_has_no_coverage_warning():
    t = _call([(0, 0.0, 40.0, "a b c"), (1, 41.0, 80.0, "d e f")], 101.0)
    assert tr.WARN_LOW_COVERAGE not in t.quality.warnings


def test_a_very_short_call_is_not_judged_on_coverage():
    t = _call([(0, 0.0, 3.0, "hello")], 20.0)
    assert tr.WARN_LOW_COVERAGE not in t.quality.warnings


def test_pasted_text_has_no_coverage_to_measure():
    from sales_call_analyzer.models import SuppliedTranscript
    t = tr.from_supplied_text(SuppliedTranscript(text="Rep: hello\nCustomer: hi"))
    assert t.quality.speech_coverage is None
    assert tr.WARN_LOW_COVERAGE not in t.quality.warnings


# =========================================================================== #
# Roles across scripts and misspellings
# =========================================================================== #
def _roles(turns, rep="Rajan Kumar", customer="Meera Patel", brand="ScaleSerum"):
    t = spk.resolve_roles(_call(turns, 60.0), rep=RepInfo(name=rep),
                          customer=CustomerInfo(name=customer), brand_name=brand)
    return {s.speaker_id: s for s in t.speakers}


def test_a_devanagari_self_introduction_identifies_the_rep():
    """Deepgram writes Hindi in Devanagari; the CRM name is Latin. This left the
    rep unidentified on every Hindi call."""
    roles = _roles([(0, 0.0, 5.0, "नमस्ते, मेरा नाम राजन है।"),
                    (1, 5.5, 8.0, "हाँ जी बोलिए")])
    assert roles["speaker_0"].role == sca.ROLE_SALES_REP
    assert roles["speaker_0"].role_confidence == "medium"      # loose: one step down
    assert roles["speaker_1"].role == sca.ROLE_CUSTOMER        # elimination


def test_a_split_brand_name_still_identifies_the_rep():
    """'Scale Serum' for 'ScaleSerum' - the misspelling keyterms usually prevent."""
    roles = _roles([(0, 0.0, 5.0, "Hello, I am calling from Scale Serum."),
                    (1, 5.5, 8.0, "Okay, go on.")], rep="Unknown Person")
    assert roles["speaker_0"].role == sca.ROLE_SALES_REP
    assert roles["speaker_0"].role_confidence == "low"         # medium, loosened


def test_a_hindi_brand_introduction_identifies_the_rep():
    roles = _roles([(0, 0.0, 5.0, "जी मैं स्केल सीरम से बोल रहा हूँ"),
                    (1, 5.5, 8.0, "हाँ")], rep="Unknown Person")
    assert roles["speaker_0"].role == sca.ROLE_SALES_REP


def test_a_misheard_customer_name_is_still_the_customer_addressed():
    roles = _roles([(0, 0.0, 5.0, "Hi Rajan here from ScaleSerum."),
                    (1, 5.5, 8.0, "Hello?"),
                    (2, 8.5, 10.0, "Is that you?"),
                    (0, 10.5, 14.0, "Good afternoon mister Manjunaf, how are you?"),
                    (2, 14.5, 18.0, "I am fine, tell me.")],
                   customer="Manjunath Gowda")
    assert roles["speaker_2"].role == sca.ROLE_CUSTOMER
    assert roles["speaker_2"].role_confidence == "low"         # medium, loosened
    assert roles["speaker_1"].role == sca.ROLE_PARTICIPANT


def test_an_exact_match_is_unchanged_and_high_confidence():
    roles = _roles([(0, 0.0, 5.0, "Hi, this is Rajan from ScaleSerum."),
                    (1, 5.5, 8.0, "Yes?")])
    assert roles["speaker_0"].role == sca.ROLE_SALES_REP
    assert roles["speaker_0"].role_confidence == "high"


def test_an_exact_match_on_a_later_speaker_beats_a_loose_one_on_an_earlier():
    roles = _roles([(0, 0.0, 5.0, "Main Rajana ko jaanta hoon"),      # loose "main rajan"
                    (1, 5.5, 9.0, "Hello, this is Rajan from ScaleSerum.")])
    assert roles["speaker_1"].role == sca.ROLE_SALES_REP
    assert roles["speaker_1"].role_confidence == "high"


def test_a_similar_but_different_name_is_not_the_name():
    """rajni is not rajan."""
    roles = _roles([(0, 0.0, 5.0, "This is Rajni speaking."),
                    (1, 5.5, 8.0, "Okay.")], brand=None)
    assert all(s.role == sca.ROLE_UNKNOWN for s in roles.values())


def test_short_names_are_never_fuzzy_matched():
    roles = _roles([(0, 0.0, 5.0, "This is Ravi speaking."),
                    (1, 5.5, 8.0, "Okay.")], rep="Ram", brand=None)
    assert all(s.role == sca.ROLE_UNKNOWN for s in roles.values())
