"""Transcript normalisation, input-strategy selection and speaker roles.

Pure unit tests - no network, no keys. Deepgram is represented by fixtures
shaped like a real pre-recorded response.

The load-bearing assertions are about what must NOT happen: roles must not be
forced onto a three-way call, timestamps must not be invented for pasted text,
and turn merging must never move words across a speaker change.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import speakers as spk  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from sales_call_analyzer.models import (  # noqa: E402
    AudioRef,
    CustomerInfo,
    RepInfo,
    SuppliedSegment,
    SuppliedTranscript,
)


# --------------------------------------------------------------------------- fixtures
def deepgram_response(utterances, duration=120.0, detected="en"):
    return {
        "metadata": {"duration": duration, "request_id": "req_test"},
        "results": {
            "channels": [{"detected_language": detected,
                          "alternatives": [{"transcript": " ".join(u["transcript"] for u in utterances),
                                            "confidence": 0.9, "words": []}]}],
            "utterances": utterances,
        },
    }


TWO_SPEAKER = deepgram_response([
    {"speaker": 0, "start": 0.0, "end": 6.5, "confidence": 0.95,
     "transcript": "Hi Meera, this is Rajan Kumar calling from EdTech Pro. Is this a good time?"},
    {"speaker": 1, "start": 6.9, "end": 10.2, "confidence": 0.93,
     "transcript": "Yes, that's fine. I registered for the masterclass last month."},
    {"speaker": 0, "start": 10.4, "end": 15.0, "confidence": 0.91,
     "transcript": "Great. What is driving your interest right now?"},
    {"speaker": 1, "start": 15.3, "end": 22.0, "confidence": 0.9,
     "transcript": "My main problem is I don't have a recognised certification."},
])

THREE_SPEAKER = deepgram_response([
    {"speaker": 0, "start": 0.0, "end": 5.0, "confidence": 0.94,
     "transcript": "Good morning, this is Rajan Kumar from EdTech Pro."},
    {"speaker": 1, "start": 5.2, "end": 9.0, "confidence": 0.92,
     "transcript": "Hello, my husband is also on the call."},
    {"speaker": 2, "start": 9.2, "end": 14.0, "confidence": 0.88,
     "transcript": "Yes, I wanted to understand the fee structure before we decide."},
    {"speaker": 1, "start": 14.4, "end": 18.0, "confidence": 0.9,
     "transcript": "We will decide together after this call."},
])


# =========================================================================== #
# Deepgram normalisation
# =========================================================================== #
def test_utterances_become_indexed_segments():
    t = tr.from_deepgram(TWO_SPEAKER)
    assert t.source == tr.SOURCE_DEEPGRAM
    assert t.segment_count == 4
    assert [s.index for s in t.segments] == [0, 1, 2, 3]
    assert t.segments[0].speaker_id == "speaker_0"
    assert t.segments[1].speaker_id == "speaker_1"
    assert t.timestamps_available is True
    assert t.diarization_available is True
    assert t.speaker_count == 2
    assert t.duration_seconds == 120.0
    assert t.language_detected == "en"


def test_speaker_statistics_are_measured_not_guessed():
    t = tr.from_deepgram(TWO_SPEAKER)
    rep = next(s for s in t.speakers if s.speaker_id == "speaker_0")
    assert rep.turn_count == 2
    assert rep.talk_time_seconds == pytest.approx(6.5 + 4.6, abs=0.01)
    assert rep.word_count > 0


def test_three_speakers_are_all_preserved():
    t = tr.from_deepgram(THREE_SPEAKER)
    assert t.speaker_count == 3
    assert {s.speaker_id for s in t.speakers} == {"speaker_0", "speaker_1", "speaker_2"}


def test_consecutive_same_speaker_turns_merge_within_the_gap():
    response = deepgram_response([
        {"speaker": 0, "start": 0.0, "end": 2.0, "confidence": 0.9, "transcript": "Hello there."},
        {"speaker": 0, "start": 2.3, "end": 4.0, "confidence": 0.9, "transcript": "How are you?"},
        {"speaker": 1, "start": 4.5, "end": 6.0, "confidence": 0.9, "transcript": "I am well."},
    ])
    t = tr.from_deepgram(response, merge_gap_seconds=1.0)
    assert t.segment_count == 2
    assert t.segments[0].text == "Hello there. How are you?"
    assert t.segments[0].end == 4.0
    assert t.segments[1].speaker_id == "speaker_1"


def test_merging_never_crosses_a_speaker_change():
    response = deepgram_response([
        {"speaker": 0, "start": 0.0, "end": 2.0, "confidence": 0.9, "transcript": "Mine."},
        {"speaker": 1, "start": 2.1, "end": 3.0, "confidence": 0.9, "transcript": "Theirs."},
    ])
    t = tr.from_deepgram(response, merge_gap_seconds=10.0)
    assert t.segment_count == 2
    assert t.segments[0].text == "Mine."
    assert t.segments[1].text == "Theirs."


def test_falls_back_to_words_when_no_utterances():
    response = {
        "metadata": {"duration": 6.0},
        "results": {"channels": [{"alternatives": [{
            "transcript": "hello there hi",
            "words": [
                {"punctuated_word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.9, "speaker": 0},
                {"punctuated_word": "there.", "start": 0.6, "end": 1.0, "confidence": 0.8, "speaker": 0},
                {"punctuated_word": "Hi.", "start": 1.4, "end": 1.9, "confidence": 0.7, "speaker": 1},
            ]}]}]},
    }
    t = tr.from_deepgram(response)
    assert t.segment_count == 2
    assert t.segments[0].text == "Hello there."
    assert t.segments[0].confidence == pytest.approx(0.85, abs=0.001)
    assert t.segments[1].speaker_id == "speaker_1"


def test_empty_response_is_empty_not_an_exception():
    t = tr.from_deepgram({"metadata": {}, "results": {"channels": []}})
    assert t.segment_count == 0
    assert t.quality.usable is False
    assert tr.WARN_EMPTY in t.quality.warnings


def test_low_confidence_is_reported_not_corrected():
    response = deepgram_response([
        {"speaker": 0, "start": 0.0, "end": 2.0, "confidence": 0.31, "transcript": "mumble mumble"},
        {"speaker": 1, "start": 2.5, "end": 4.0, "confidence": 0.28, "transcript": "hard to hear"},
    ])
    t = tr.from_deepgram(response)
    assert t.quality.mean_confidence < 0.4
    assert tr.WARN_LOW_CONFIDENCE in t.quality.warnings
    assert t.quality.usable is True   # poor audio is still analysable, just flagged


# =========================================================================== #
# Supplied transcripts
# =========================================================================== #
def test_structured_supplied_transcript_keeps_speakers_and_times():
    supplied = SuppliedTranscript(format="structured", segments=[
        SuppliedSegment(speaker_id="speaker_0", role="sales_rep", name="Rajan Kumar",
                        start=0.0, end=4.0, text="Good morning."),
        SuppliedSegment(speaker_id="speaker_1", role="customer", name="Meera Patel",
                        start=4.2, end=8.0, text="Good morning to you."),
    ])
    t = tr.from_supplied_structured(supplied)
    assert t.source == tr.SOURCE_SUPPLIED_STRUCTURED
    assert t.speaker_count == 2
    assert t.timestamps_available is True
    assert {s.role for s in t.speakers} == {"sales_rep", "customer"}


def test_arbitrary_speaker_labels_map_to_stable_ids():
    supplied = SuppliedTranscript(segments=[
        SuppliedSegment(speaker_label="Agent", text="Hello."),
        SuppliedSegment(speaker_label="Caller", text="Hi."),
        SuppliedSegment(speaker_label="Agent", text="How can I help?"),
    ])
    t = tr.from_supplied_structured(supplied)
    assert t.speaker_count == 2
    assert t.segments[0].speaker_id == t.segments[2].speaker_id
    assert t.segments[0].speaker_id != t.segments[1].speaker_id


def test_labelled_plain_text_recovers_speakers_but_not_timestamps():
    supplied = SuppliedTranscript(text=(
        "Rajan: Hi Meera, thanks for taking the time today.\n"
        "Meera: Happy to talk.\n"
        "Rajan: Let me tell you about the programme.\n"))
    t = tr.from_supplied_text(supplied)
    assert t.source == tr.SOURCE_SUPPLIED_TEXT
    assert t.speaker_count == 2
    assert t.timestamps_available is False
    assert tr.WARN_NO_TIMESTAMPS in t.quality.warnings
    assert all(s.start is None for s in t.segments)


def test_unlabelled_prose_is_segmented_but_not_attributed():
    supplied = SuppliedTranscript(text=(
        "We discussed the certification programme in detail.\n\n"
        "The customer asked about instalment options.\n\n"
        "A follow up was agreed for next week."))
    t = tr.from_supplied_text(supplied)
    assert t.segment_count == 3
    assert t.diarization_available is False
    assert t.speaker_count == 0
    assert tr.WARN_NO_SPEAKER_ATTRIBUTION in t.quality.warnings
    # It must NOT pretend one known person said everything.
    assert all(s.speaker_id == tr.UNATTRIBUTED_SPEAKER_ID for s in t.segments)


# =========================================================================== #
# Input strategy (the three supported cases)
# =========================================================================== #
def test_case_a_audio_only_transcribes():
    strategy, why = tr.select_input_strategy(AudioRef(url="https://x/rec.mp3"), None)
    assert strategy == tr.STRATEGY_TRANSCRIBE_AUDIO
    assert why


def test_case_b_transcript_only_skips_deepgram():
    supplied = SuppliedTranscript(text="Rajan: hello.\nMeera: hi.")
    strategy, _ = tr.select_input_strategy(None, supplied)
    assert strategy == tr.STRATEGY_USE_SUPPLIED_TEXT

    structured = SuppliedTranscript(segments=[
        SuppliedSegment(speaker_id="speaker_0", start=0.0, end=1.0, text="hello")])
    strategy, _ = tr.select_input_strategy(None, structured)
    assert strategy == tr.STRATEGY_USE_SUPPLIED_STRUCTURED


def test_case_c_structured_transcript_beats_audio():
    """Already has what Deepgram would produce, so transcribing is duplicate cost."""
    structured = SuppliedTranscript(segments=[
        SuppliedSegment(speaker_id="speaker_0", start=0.0, end=1.0, text="hello"),
        SuppliedSegment(speaker_id="speaker_1", start=1.2, end=2.0, text="hi"),
    ])
    strategy, _ = tr.select_input_strategy(AudioRef(url="https://x/rec.mp3"), structured)
    assert strategy == tr.STRATEGY_USE_SUPPLIED_STRUCTURED


def test_case_c_plain_text_plus_audio_uses_the_audio():
    """Plain text has no speakers or timings, so the audio is the better source."""
    plain = SuppliedTranscript(text="we talked about the programme and the price")
    strategy, why = tr.select_input_strategy(AudioRef(url="https://x/rec.mp3"), plain)
    assert strategy == tr.STRATEGY_TRANSCRIBE_AUDIO
    assert "fallback" in why


def test_a_format_label_cannot_fake_structure():
    """A caller labelling plain text as structured must not disable diarization."""
    lying = SuppliedTranscript(format="structured", segments=[
        SuppliedSegment(text="no speaker here"), SuppliedSegment(text="none here either")])
    assert tr.is_structured(lying) is False
    strategy, _ = tr.select_input_strategy(AudioRef(url="https://x/rec.mp3"), lying)
    assert strategy == tr.STRATEGY_TRANSCRIBE_AUDIO


def test_no_input_is_reported():
    strategy, _ = tr.select_input_strategy(None, None)
    assert strategy == tr.STRATEGY_NONE


# =========================================================================== #
# Speaker roles
# =========================================================================== #
def test_two_speaker_call_resolves_both_roles():
    """The rep introduces themselves and greets the customer by name, so both
    roles resolve on direct evidence rather than by elimination."""
    t = spk.resolve_roles(tr.from_deepgram(TWO_SPEAKER),
                          rep=RepInfo(name="Rajan Kumar"),
                          customer=CustomerInfo(name="Meera Patel"),
                          brand_name="EdTech Pro")
    rep = next(s for s in t.speakers if s.role == sca.ROLE_SALES_REP)
    cust = next(s for s in t.speakers if s.role == sca.ROLE_CUSTOMER)
    assert rep.speaker_id == "speaker_0"
    assert rep.name == "Rajan Kumar"
    assert rep.role_basis == sca.ROLE_BASIS_REP_SELF_INTRO
    assert cust.speaker_id == "speaker_1"
    assert cust.role_basis == sca.ROLE_BASIS_CUSTOMER_ADDRESSED
    assert cust.name == "Meera Patel"


def test_elimination_still_covers_a_two_speaker_call_with_no_name_evidence():
    """Nobody is greeted by name here, so the second voice is the customer only
    because there is exactly one other voice."""
    response = deepgram_response([
        {"speaker": 0, "start": 0.0, "end": 5.0, "confidence": 0.9,
         "transcript": "Good afternoon, this is Rajan Kumar from EdTech Pro."},
        {"speaker": 1, "start": 5.2, "end": 9.0, "confidence": 0.9,
         "transcript": "Go ahead, I have a few minutes."},
    ])
    t = spk.resolve_roles(tr.from_deepgram(response),
                          rep=RepInfo(name="Rajan Kumar"),
                          customer=CustomerInfo(name="Meera Patel"))
    cust = next(s for s in t.speakers if s.role == sca.ROLE_CUSTOMER)
    assert cust.speaker_id == "speaker_1"
    assert cust.role_basis == sca.ROLE_BASIS_ELIMINATION


def test_customer_addressed_with_an_honorific_on_a_three_way_call():
    """The real-world shape: a receptionist answers, the rep greets the customer
    as 'mister Sanjay', and the customer replies. Elimination cannot help with
    three voices - only the address evidence can."""
    response = deepgram_response([
        {"speaker": 0, "start": 0.0, "end": 5.0, "confidence": 0.9,
         "transcript": "Hi. If you record your name and reason for calling, I will see if they are available."},
        {"speaker": 1, "start": 5.2, "end": 9.0, "confidence": 0.9,
         "transcript": "Hi. My name is Devansh Sharma. I am speaking from Directors Institute."},
        {"speaker": 2, "start": 9.2, "end": 11.0, "confidence": 0.9,
         "transcript": "Hello? Hello?"},
        {"speaker": 1, "start": 11.5, "end": 18.0, "confidence": 0.9,
         "transcript": "Yes? Hi, mister Sanjay. Good afternoon. Devansh here from Directors Institute."},
        {"speaker": 2, "start": 18.5, "end": 26.0, "confidence": 0.9,
         "transcript": "Hey, Devansh. I had a couple of questions about the programme."},
    ])
    t = spk.resolve_roles(tr.from_deepgram(response),
                          rep=RepInfo(name="Devansh Sharma"),
                          customer=CustomerInfo(name="Sanjay"))
    roles = {s.speaker_id: s for s in t.speakers}
    assert roles["speaker_1"].role == sca.ROLE_SALES_REP
    assert roles["speaker_2"].role == sca.ROLE_CUSTOMER
    assert roles["speaker_2"].role_basis == sca.ROLE_BASIS_CUSTOMER_ADDRESSED
    assert roles["speaker_2"].name == "Sanjay"
    # The receptionist is neither, and is not guessed at.
    assert roles["speaker_0"].role == sca.ROLE_PARTICIPANT
    assert roles["speaker_0"].name is None


def test_an_address_with_no_reply_does_not_identify_anyone():
    """Being named is not enough - somebody has to answer to it."""
    response = deepgram_response([
        {"speaker": 0, "start": 0.0, "end": 5.0, "confidence": 0.9,
         "transcript": "Hello, this is Rajan Kumar. Hi Meera, are you there?"},
        {"speaker": 0, "start": 5.2, "end": 9.0, "confidence": 0.9,
         "transcript": "Hello? I think the line has dropped."},
    ])
    t = spk.resolve_roles(tr.from_deepgram(response),
                          rep=RepInfo(name="Rajan Kumar"),
                          customer=CustomerInfo(name="Meera Patel"))
    assert all(s.role != sca.ROLE_CUSTOMER for s in t.speakers)


def test_a_reused_transcript_is_re_resolved_not_frozen():
    """A transcript reused from an earlier run arrives carrying that run's
    derived roles. They must be re-derived, not mistaken for roles the caller
    supplied - otherwise corrected CRM names could never take effect."""
    first = spk.resolve_roles(tr.from_deepgram(TWO_SPEAKER),
                              rep=RepInfo(name="Wrong Person"),
                              customer=CustomerInfo(name="Also Wrong"))
    assert all(s.role == sca.ROLE_UNKNOWN for s in first.speakers)

    # Same transcript object, now with the real names.
    second = spk.resolve_roles(first,
                               rep=RepInfo(name="Rajan Kumar"),
                               customer=CustomerInfo(name="Meera Patel"))
    rep = next(s for s in second.speakers if s.role == sca.ROLE_SALES_REP)
    assert rep.speaker_id == "speaker_0"
    assert rep.role_basis == sca.ROLE_BASIS_REP_SELF_INTRO
    assert rep.role_basis != sca.ROLE_BASIS_SUPPLIED


def test_derived_roles_are_cleared_when_the_evidence_goes_away():
    """Re-resolving with names that match nobody must reset to unknown, not keep
    the previous run's answer."""
    resolved = spk.resolve_roles(tr.from_deepgram(TWO_SPEAKER),
                                 rep=RepInfo(name="Rajan Kumar"),
                                 customer=CustomerInfo(name="Meera Patel"))
    assert any(s.role == sca.ROLE_SALES_REP for s in resolved.speakers)

    again = spk.resolve_roles(resolved, rep=RepInfo(), customer=CustomerInfo())
    assert all(s.role == sca.ROLE_UNKNOWN for s in again.speakers)
    assert all(s.name is None for s in again.speakers)
    assert all(s.role_basis == sca.ROLE_BASIS_UNRESOLVED for s in again.speakers)


def test_three_speaker_call_never_guesses_the_customer_by_elimination():
    t = spk.resolve_roles(tr.from_deepgram(THREE_SPEAKER),
                          rep=RepInfo(name="Rajan Kumar"),
                          customer=CustomerInfo(name="Meera Patel"),
                          brand_name="EdTech Pro")
    roles = {s.speaker_id: s.role for s in t.speakers}
    assert roles["speaker_0"] == sca.ROLE_SALES_REP
    # Nobody introduced themselves as Meera, so neither other voice may be
    # promoted to "customer" just because they are not the rep.
    assert roles["speaker_1"] == sca.ROLE_PARTICIPANT
    assert roles["speaker_2"] == sca.ROLE_PARTICIPANT
    assert sca.ROLE_CUSTOMER not in roles.values()


def test_customer_self_introduction_is_honoured_on_a_three_way_call():
    response = deepgram_response([
        {"speaker": 0, "start": 0.0, "end": 4.0, "confidence": 0.9,
         "transcript": "Hello, this is Rajan Kumar from EdTech Pro."},
        {"speaker": 1, "start": 4.2, "end": 8.0, "confidence": 0.9,
         "transcript": "Hi, my name is Meera Patel and my husband has joined too."},
        {"speaker": 2, "start": 8.2, "end": 12.0, "confidence": 0.9,
         "transcript": "Yes, I had a question about the fees."},
    ])
    t = spk.resolve_roles(tr.from_deepgram(response),
                          rep=RepInfo(name="Rajan Kumar"),
                          customer=CustomerInfo(name="Meera Patel"))
    roles = {s.speaker_id: s.role for s in t.speakers}
    assert roles["speaker_0"] == sca.ROLE_SALES_REP
    assert roles["speaker_1"] == sca.ROLE_CUSTOMER
    assert roles["speaker_2"] == sca.ROLE_PARTICIPANT


def test_no_crm_names_leaves_every_role_unknown():
    t = spk.resolve_roles(tr.from_deepgram(TWO_SPEAKER), rep=RepInfo(), customer=CustomerInfo())
    assert all(s.role == sca.ROLE_UNKNOWN for s in t.speakers)
    assert all(s.name is None for s in t.speakers)
    assert all(s.role_basis == sca.ROLE_BASIS_UNRESOLVED for s in t.speakers)


def test_roles_are_never_inferred_from_turn_order_or_talk_time():
    """The first and longest speaker here is the customer, not the rep. With no
    identifying evidence the system must decline to guess."""
    response = deepgram_response([
        {"speaker": 0, "start": 0.0, "end": 40.0, "confidence": 0.9,
         "transcript": "I have been thinking about this for a long time and I have many questions."},
        {"speaker": 1, "start": 40.5, "end": 44.0, "confidence": 0.9,
         "transcript": "Certainly, let me help."},
    ])
    t = spk.resolve_roles(tr.from_deepgram(response),
                          rep=RepInfo(name="Rajan Kumar"),
                          customer=CustomerInfo(name="Meera Patel"))
    assert all(s.role == sca.ROLE_UNKNOWN for s in t.speakers)


def test_names_are_never_invented():
    """A name may only come from the CRM record. An unresolved speaker has none."""
    t = spk.resolve_roles(tr.from_deepgram(THREE_SPEAKER),
                          rep=RepInfo(name="Rajan Kumar"),
                          customer=CustomerInfo(name="Meera Patel"))
    for speaker in t.speakers:
        if speaker.role != sca.ROLE_SALES_REP:
            assert speaker.name is None


def test_supplied_roles_are_trusted_and_recorded_as_supplied():
    supplied = SuppliedTranscript(segments=[
        SuppliedSegment(speaker_id="speaker_0", role="sales_rep", start=0.0, end=2.0, text="Hello."),
        SuppliedSegment(speaker_id="speaker_1", role="customer", start=2.2, end=4.0, text="Hi."),
    ])
    t = spk.resolve_roles(tr.from_supplied_structured(supplied),
                          rep=RepInfo(name="Rajan Kumar"), customer=CustomerInfo(name="Meera Patel"))
    rep = next(s for s in t.speakers if s.role == sca.ROLE_SALES_REP)
    assert rep.role_basis == sca.ROLE_BASIS_SUPPLIED
    assert rep.name == "Rajan Kumar"


def test_role_summary_reports_what_was_not_resolved():
    t = spk.resolve_roles(tr.from_deepgram(THREE_SPEAKER), rep=RepInfo(name="Rajan Kumar"))
    summary = spk.role_summary(t)
    assert summary["sales_rep_identified"] is True
    assert summary["customer_identified"] is False
    assert len(summary["unresolved"]) == 2


# =========================================================================== #
# Prompt rendering
# =========================================================================== #
def test_prompt_transcript_carries_the_evidence_anchors():
    t = spk.resolve_roles(tr.from_deepgram(TWO_SPEAKER), rep=RepInfo(name="Rajan Kumar"),
                          customer=CustomerInfo(name="Meera Patel"))
    text = tr.render_for_prompt(t)
    assert text.startswith("[0] speaker_0 (sales_rep)")
    assert "[3] speaker_1 (customer)" in text
    assert "certification" in text


def test_prompt_speaker_table_admits_unresolved_roles():
    t = spk.resolve_roles(tr.from_deepgram(THREE_SPEAKER), rep=RepInfo(name="Rajan Kumar"))
    text = spk.render_for_prompt(t)
    assert "speaker_2" in text
    assert "participant" in text
    assert "do not assume who that person is" in text
