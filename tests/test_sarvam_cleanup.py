"""Cleaning Sarvam AI transcripts - standalone modules, not yet in the analyzer.

transcription/sarvam_cleanup.py (free, deterministic), sarvam_recheck.py
(Gemini listens to flagged turns) and sarvam_client.py (the Batch API call).
Every case below is one seen on the real test calls (2026-09-30/10-01).

No network: Sarvam and Gemini are fakes.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sales_call_analyzer import transcript as tr  # noqa: E402
from transcription import sarvam_cleanup as sc  # noqa: E402
from transcription import sarvam_client  # noqa: E402
from transcription import sarvam_recheck as rc  # noqa: E402

RATE = 16000
LAW = ["Lawtorney", "Lawtorney AI"]


def response(*entries, language="mr-IN"):
    """A Sarvam batch response from (speaker, start, end, text) tuples."""
    return {"language_code": language, "diarized_transcript": {"entries": [
        {"speaker_id": str(s), "start_time_seconds": a, "end_time_seconds": b, "transcript": t}
        for s, a, b, t in entries]}}


def text_of(entries):
    return [e.text for e in entries]


# =========================================================================== #
# 1. Duplicated fragments
# =========================================================================== #
def test_echo_of_the_other_speakers_words_at_the_same_moment_is_removed():
    # mycall8: the rep's "तुम्ही सांगू शकता की कधी" also came back under the customer.
    r = sc.clean(response(
        (0, 301.2, 304.8, "हो तर तुम्ही सांगू शकता की कधी तुम्ही कधी तुम्ही available आहात"),
        (1, 301.9, 303.0, "तुम्ही सांगू शकता की कधी")), [])
    assert text_of(r.entries) == ["हो तर तुम्ही सांगू शकता की कधी तुम्ही कधी तुम्ही available आहात"]
    assert r.duplicates_removed[0]["echo_of"] == 0


def test_backchannel_whose_word_is_elsewhere_in_the_turn_is_kept():
    # "हो" is in the rep's turn, but at its start - the customer's "हो" 9 s
    # later is a real backchannel.
    r = sc.clean(response(
        (0, 87.7, 104.2, "हो so basically हा आपला tool जे आहे ते specifically advocate साठी "
                         "त्याच्यामध्ये तुम्ही drafting legal notice चा reply written statement"),
        (1, 97.1, 97.3, "हो")), [])
    assert len(r.entries) == 2 and not r.duplicates_removed


def test_one_word_greeting_next_to_the_other_turn_is_kept():
    # mycall8: customer "Hello" ends just before the rep's "Hello हां आवाज येतीये आता".
    r = sc.clean(response((1, 69.3, 69.8, "Hello"), (0, 69.9, 71.6, "Hello हां आवाज येतीये आता")), [])
    assert len(r.entries) == 2


def test_two_short_turns_saying_the_same_thing_are_both_kept():
    r = sc.clean(response((0, 10.5, 11.0, "Hello"), (1, 11.0, 11.4, "Hello")), [])
    assert len(r.entries) == 2


def test_long_turns_are_never_treated_as_echoes():
    r = sc.clean(response(
        (0, 0.0, 10.0, "one two three four five six seven eight nine ten"),
        (1, 0.0, 10.0, "one two three four five six seven eight nine ten eleven")), [])
    assert len(r.entries) == 2


def test_echo_removal_can_be_switched_off():
    r = sc.clean(response((0, 301.2, 304.8, "हो तर तुम्ही सांगू शकता की कधी तुम्ही कधी"),
                          (1, 301.9, 303.0, "तुम्ही सांगू शकता की कधी")), [], remove_echoes=False)
    assert len(r.entries) == 2


# =========================================================================== #
# 2. Wrong-script stray words
# =========================================================================== #
def test_odia_and_gujarati_words_are_written_in_the_calls_script():
    r = sc.clean(response((1, 1.0, 1.4, "ହଁ"), (0, 2.0, 3.0, "હા, sorry."), language="mr-IN"), [])
    assert text_of(r.entries) == ["हँ", "हा, sorry."]
    assert len(r.script_fixes) == 2


def test_an_english_call_takes_devanagari_for_stray_words():
    r = sc.clean(response((1, 1.0, 1.4, "ହଁ okay"), language="en-IN"), [])
    assert text_of(r.entries) == ["हँ okay"]


def test_the_calls_own_script_and_hindi_are_left_alone():
    ml = "ഹലോ, ഞാൻ ABC Educational Institute-ൽ നിന്ന് വിളിക്കുകയാണ്"
    r = sc.clean(response((0, 0.0, 3.0, ml), (1, 3.0, 4.0, "हाँ ठीक है"), language="ml-IN"), [])
    assert text_of(r.entries) == [ml, "हाँ ठीक है"] and not r.script_fixes


def test_danda_ending_an_all_english_turn_becomes_a_full_stop():
    r = sc.clean(response((1, 1.0, 2.0, "Good Afternoon।"), (0, 2.0, 3.0, "हाँ जी।")), [])
    assert text_of(r.entries) == ["Good Afternoon.", "हाँ जी।"]


def test_main_script_falls_back_to_the_letters_when_the_language_is_unknown():
    entries = sc.entries_from_sarvam(response((0, 0, 5, "ఇది ఒక పరీక్ష వాక్యం మాత్రమే ఇంకా కొంచెం"),
                                              language=None))
    assert sc.main_script(entries, None) == "telugu"


# =========================================================================== #
# 3. Brand and product names
# =========================================================================== #
def test_brand_said_after_from_is_corrected():
    # mycall7
    r = sc.clean(response((0, 6.2, 9.0, "Yes, you done with your lunch Shruti from Lot Earning?")), LAW)
    assert text_of(r.entries) == ["Yes, you done with your lunch Shruti from Lawtorney?"]
    assert r.term_fixes[0]["tier"] == "context"


def test_brand_before_ai_or_dot_ai_is_corrected():
    r = sc.clean(response((0, 6.2, 9.5, "हा रुपेश सर मी श्रुती बोलतीये lot आणि AI मधून"),
                          (0, 20.0, 22.0, "I am calling from lotany.ai."),
                          (0, 30.0, 32.0, "visit lawterney.ai, please")), LAW)
    assert text_of(r.entries) == ["हा रुपेश सर मी श्रुती बोलतीये Lawtorney AI मधून",
                                  "I am calling from Lawtorney.ai.",
                                  "visit Lawtorney.ai, please"]


def test_close_spelling_is_corrected_without_context():
    r = sc.clean(response((0, 26.6, 28.6, "मी Law Attorney AI मधून बोलतीये."),
                          (0, 4.4, 9.0, "this is Rahul calling from Skyl Serum. आपण")),
                 LAW + ["ScaleSerum"])
    assert text_of(r.entries) == ["this is Rahul calling from ScaleSerum. आपण",
                                  "मी Lawtorney AI मधून बोलतीये."]


def test_product_name_is_not_written_unless_its_last_word_is_heard():
    # "Lot Earning" must become "Lawtorney", never "Lawtorney AI".
    r = sc.clean(response((0, 0, 2, "Shruti from Lot Earning?")), LAW)
    assert "AI" not in r.entries[0].text


def test_sound_alike_without_context_is_flagged_not_changed():
    text = "कुठे आहे office आपलं law training इथे Nagpur आहे की Mumbai ला आहे?"
    r = sc.clean(response((1, 81.5, 86.0, text)), LAW)
    assert r.entries[0].text == text
    assert r.suspects[0]["heard"] == "law training" and r.suspect_indices == [0]


def test_ordinary_words_are_left_alone():
    for text, terms in (("our sales team can help", ["ScaleSerum"]),
                        ("literally less than a lot of later on", LAW),
                        ("from the attorney general", LAW)):
        r = sc.clean(response((0, 0, 3, text)), terms)
        assert r.entries[0].text == text and not r.term_fixes, text


def test_brand_already_right_is_not_rewritten():
    r = sc.clean(response((0, 0, 3, "So Scale Serum helps you"), (0, 3, 6, "from Director's Institute.")),
                 ["ScaleSerum", "Directors Institute"])
    assert text_of(r.entries) == ["So Scale Serum helps you", "from Director's Institute."]
    assert not r.term_fixes and not r.suspects


def test_short_terms_are_matched_exactly_only():
    r = sc.clean(response((0, 0, 3, "I am calling from WDC and VDC")), ["WDC"])
    assert r.entries[0].text == "I am calling from WDC and VDC" and not r.suspects


def test_marker_attached_with_a_hyphen_counts_as_context():
    r = sc.clean(response((0, 0, 3, "मी Arjun बोलतो आहे Skyl Serum-मधून")), ["ScaleSerum"])
    assert r.entries[0].text == "मी Arjun बोलतो आहे ScaleSerum-मधून"


# =========================================================================== #
# 4. Product-term flags
# =========================================================================== #
VOCAB = ["ChatGPT", "trial version", "refund", "prompt book", "our tool"]


def flagged(text):
    return [f["heard"] for f in sc.flag_vocabulary(sc.entries_from_sarvam(response((0, 0, 5, text))), VOCAB)]


def test_misheard_product_terms_are_flagged():
    assert "free fund" in flagged("free fund के लिए है")
    assert "JTPT" in flagged("सेम JTPT और सारे के सारे")
    assert "Child version" in flagged("Child version नाहीये.")
    assert "Chromebook" in flagged("secret AI Chromebook and AI for Law book")
    assert "Gartol" in flagged("He is also using Gartol")


def test_terms_said_correctly_are_not_flagged():
    for text in ("copy this prompt", "these prompts help", "trial version नाही", "refund policy",
                 "काही version नाही", "prompt को copy करो", "हो किंवा trial trial वगैरे"):
        assert flagged(text) == [], text


# =========================================================================== #
# Shape for the rest of the pipeline
# =========================================================================== #
def test_cleaned_entries_read_as_a_deepgram_response():
    r = sc.clean(response((0, 0.0, 2.0, "Hello from Lot Earning"), (1, 2.5, 3.0, "हो")), LAW)
    t = tr.from_deepgram(sc.to_deepgram_shape(r.entries))
    assert [s.text for s in t.segments] == ["Hello from Lawtorney", "हो"]
    assert {s.speaker_id for s in t.speakers} == {"speaker_0", "speaker_1"}


def test_report_counts_every_change():
    r = sc.clean(response((0, 0, 3, "from Lot Earning"), (1, 4, 5, "ହଁ"),
                          (1, 6, 8, "law training इथे")), LAW)
    assert r.report()["counts"] == {"duplicates_removed": 0, "script_fixes": 1,
                                    "term_fixes": 1, "suspect_entries": 1}


# =========================================================================== #
# Gemini re-check
# =========================================================================== #
class _Usage:
    prompt_token_count = 1000
    candidates_token_count = 100
    thoughts_token_count = 50


class _Response:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = _Usage()


class FakeGemini:
    def __init__(self, texts=None, error=None):
        self.texts, self.error, self.requests = texts or [], error, []
        outer = self

        class _Models:
            async def generate_content(self, model, contents, config):
                outer.requests.append(contents)
                if outer.error:
                    raise outer.error
                return _Response(json.dumps({"clips": [{"n": i + 1, "text": t}
                                                       for i, t in enumerate(outer.texts)]}))

        class _Aio:
            models = _Models()

        self.aio = _Aio()


def _call(*texts):
    r = sc.clean(response(*[(0, i * 5.0, i * 5.0 + 4.0, t) for i, t in enumerate(texts)]), LAW,
                 ["refund"])
    return r, np.zeros(RATE * 5 * len(texts), np.float32)


def test_recheck_applies_a_correction_and_writes_the_brand_one_way():
    r, audio = _call("free fund के लिए है", "कुठे आहे office आपलं law training इथे")
    fake = FakeGemini(["refund के लिए है", "कुठे आहे office आपलं Law Attorney इथे"])
    out = asyncio.run(rc.recheck(fake, "m", audio, r.entries, r.suspect_indices, LAW + ["refund"],
                                 brand_terms=LAW))
    assert out.ok and text_of(r.entries) == ["refund के लिए है", "कुठे आहे office आपलं Lawtorney इथे"]
    assert len(fake.requests) == 1 and out.input_tokens == 1000 and out.thinking_tokens == 50


def test_recheck_sends_only_the_flagged_turns_with_their_text():
    r, audio = _call("all good here", "free fund के लिए है")
    fake = FakeGemini(["refund के लिए है"])
    asyncio.run(rc.recheck(fake, "m", audio, r.entries, r.suspect_indices, ["refund"]))
    sent = [c for c in fake.requests[0] if isinstance(c, str)]
    assert any("free fund के लिए है" in c for c in sent) and not any("all good here" in c for c in sent)
    assert r.entries[0].text == "all good here"


def test_recheck_refuses_a_rewrite_instead_of_a_correction():
    r, audio = _call("free fund के लिए है")
    asyncio.run(rc.recheck(FakeGemini(["something completely different was said here"]), "m",
                           audio, r.entries, r.suspect_indices, ["refund"]))
    assert r.entries[0].text == "free fund के लिए है"


def test_recheck_refuses_empty_text():
    r, audio = _call("free fund के लिए है")
    out = asyncio.run(rc.recheck(FakeGemini([""]), "m", audio, r.entries, r.suspect_indices, ["refund"]))
    assert r.entries[0].text == "free fund के लिए है" and out.rejected == {0: "empty"}


def test_recheck_provider_error_keeps_the_text():
    r, audio = _call("free fund के लिए है")
    out = asyncio.run(rc.recheck(FakeGemini(error=RuntimeError("boom")), "m", audio, r.entries,
                                 r.suspect_indices, ["refund"]))
    assert not out.ok and out.reason == rc.REASON_PROVIDER_ERROR
    assert r.entries[0].text == "free fund के लिए है"


def test_recheck_without_a_client_or_flags_does_nothing():
    r, audio = _call("free fund के लिए है")
    assert asyncio.run(rc.recheck(None, "m", audio, r.entries, [0], [])).reason == rc.REASON_NOT_CONFIGURED
    fake = FakeGemini(["x"])
    assert asyncio.run(rc.recheck(fake, "m", audio, r.entries, [], [])).ok and not fake.requests


def test_changed_share_ignores_script():
    assert rc.changed_share("हो so basically", "हो so basically") == 0.0
    assert rc.changed_share("free fund के लिए", "refund के लिए") == 0.5


# =========================================================================== #
# Sarvam client
# =========================================================================== #
def test_client_without_a_key_is_not_configured(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    out = asyncio.run(sarvam_client.transcribe(b"x"))
    assert not out.ok and out.reason == sarvam_client.REASON_NOT_CONFIGURED


def test_client_returns_the_job_output(monkeypatch):
    monkeypatch.setattr(sarvam_client, "_run_job", lambda *a: response((0, 0, 1, "Hello")))
    out = asyncio.run(sarvam_client.transcribe(b"x", api_key="k"))
    assert out.ok and out.response["diarized_transcript"]["entries"][0]["transcript"] == "Hello"


def test_client_failure_is_a_reason_not_an_exception(monkeypatch):
    def boom(*a):
        raise ConnectionError("down")
    monkeypatch.setattr(sarvam_client, "_run_job", boom)
    out = asyncio.run(sarvam_client.transcribe(b"x", api_key="k"))
    assert not out.ok and out.reason == sarvam_client.REASON_PROVIDER_ERROR and out.error == "ConnectionError"


def test_client_names_an_empty_account(monkeypatch):
    class ApiError(Exception):
        status_code = 402
    def no_credits(*a):
        raise ApiError("No credits available.")
    monkeypatch.setattr(sarvam_client, "_run_job", no_credits)
    out = asyncio.run(sarvam_client.transcribe(b"x", api_key="k"))
    assert out.reason == sarvam_client.REASON_NO_CREDITS and out.error == "ApiError 402"


def test_cost_estimate():
    assert sarvam_client.estimated_cost_inr(600) == 7.5
    assert sarvam_client.estimated_cost_inr(600, diarization=False) == 5.0
