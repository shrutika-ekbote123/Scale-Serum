"""
Milestone D - the interpretation layer: transcript, triggers, the LLM pass and
the verification that decides what a client is allowed to be shown.

WHAT THESE TESTS ARE ACTUALLY DEFENDING
    Every other test in this suite checks that we computed the right number.
    These check something different: that nothing the MODEL said reaches a
    report unless the measurement record backs it. A quote that was never said,
    a timestamp past the end of the ad, a number nobody measured and a defect
    the model nominated itself are each dropped here - and dropped LOUDLY, into
    an audit a caller can read.

    None of them are hypothetical. A model asked to write about an ad will
    happily produce "the 34-second mark" for a 20-second creative.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_lab import analyzer as an  # noqa: E402
from vision_lab import evidence as ev  # noqa: E402
from vision_lab import psychology as psy  # noqa: E402
from vision_lab import scoring as sc  # noqa: E402
from vision_lab import transcript as tx  # noqa: E402
from vision_lab import framework as fw  # noqa: E402


# --------------------------------------------------------------------------- fixtures
def transcript(*lines) -> dict:
    return {"segments": [{"t": t, "end": t + 3.0, "text": text}
                         for t, text in lines],
            "text": " ".join(text for _, text in lines), "language": "en"}


SPOKEN = transcript(
    (0.0, "Most operations directors think the bottleneck is headcount."),
    (4.0, "Four thousand two hundred directors have already been certified."),
    (9.0, "Applications close on the thirtieth."),
)


def defect(defect_id="dead_zone", t_start=12.0, t_end=16.0, **measured):
    return {"defect_id": defect_id, "severity": "high",
            "t_start": t_start, "t_end": t_end,
            "measured": measured or {"seconds": 4.0},
            "scores_impacted": ["engagement"],
            "note": "attention falls away and does not recover"}


SUMMARY = {"duration_seconds": 20.0, "shot_count": 5, "max_words_on_screen": 14,
           "cut_rate_per_minute": 15.0, "sample_fps": 2.0, "brand": {},
           "overloaded_shots": []}
TIMELINE = {"points": [{"t": 0.0, "attention": 70}, {"t": 12.0, "attention": 22}],
            "weak_zones": [{"t_start": 12.0, "t_end": 16.0, "seconds": 4.0}]}


# =========================================================================== #
# Step 19 - evidence verification
# =========================================================================== #
def test_a_timestamp_that_does_not_exist_in_the_creative_is_rejected():
    """The Step 19 acceptance test, hand-corrupted exactly as specified.

    A 20 s creative, and the model cites 34 s. That is not a rounding error and
    not a judgement call - there is no such moment - so the evidence goes and
    the trigger is reported `unsupported` rather than rated off it.
    """
    corrupted = {
        "summary": "",
        "triggers": [{"id": "loss_aversion", "rating": "strong",
                      "evidence": [{"t": 34.0, "quote": "Applications close on the thirtieth."}]}],
        "recommendations": [],
    }

    verified = ev.verify(corrupted, defects=[defect()], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    trigger = verified["triggers"][0]
    assert trigger["evidence"] == []
    assert trigger["status"] == "unsupported"
    audit = verified["evidence_audit"]
    assert audit["dropped"] == 1
    assert audit["unsupported"][0]["reason"] == ev.UNSUPPORTED_TIME
    # And the failure names WHERE it happened, so a caller can act on it.
    assert "loss_aversion" in audit["unsupported"][0]["where"]


def test_a_quote_that_was_never_said_is_dropped():
    fabricated = {
        "summary": "",
        "triggers": [{"id": "bandwagon", "rating": "strong",
                      "evidence": [
                          {"t": 4.0, "quote": "Four thousand two hundred directors have already been certified."},
                          {"t": 4.0, "quote": "Rated the number one programme in Asia for six years running."},
                      ]}],
        "recommendations": [],
    }

    verified = ev.verify(fabricated, defects=[], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    kept = verified["triggers"][0]["evidence"]
    # One citation, not two: the fabricated line points at a moment already
    # cited, and a repair that duplicates existing evidence is a drop.
    assert len(kept) == 1
    assert "Four thousand two hundred" in kept[0]["quote"]
    # One real quote survives, so the rating stands - partial evidence is still
    # evidence, and discarding the whole finding would be its own distortion.
    assert verified["triggers"][0].get("status") != "unsupported"
    assert verified["evidence_audit"]["unsupported"][0]["reason"] == ev.UNSUPPORTED_QUOTE


def test_a_number_nobody_measured_takes_the_whole_recommendation_with_it():
    """The most dangerous failure mode: invented prose that READS like a
    measurement. '7 seconds of dead air' next to a real defect is indistinguishable
    from a finding unless something checks it."""
    invented = {
        "summary": "",
        "recommendations": [{
            "defect_id": "dead_zone",
            "title": "Cut the dead air",
            "why": "There are 7 seconds where nothing holds the eye.",
            "fix": "Bring the proof point forward.",
        }],
    }

    verified = ev.verify(invented, defects=[defect(seconds=4.0)], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    assert verified["recommendations"] == []
    assert verified["evidence_audit"]["unsupported"][0]["reason"] == ev.UNSUPPORTED_NUMBER


def test_a_target_in_the_fix_is_not_treated_as_an_invented_measurement():
    """REGRESSION, found on a real ad: EVERY recommendation was dropped.

    Both were sound arguments about real defects, rejected because their advice
    was specific - "hold the logo for 3 seconds", "cut this to 16 words". Those
    numbers describe a version of the ad that does not exist yet, so of course
    nothing measured them. The claim is checked; the prescription is not.
    """
    advice = {
        "summary": "",
        "recommendations": [{
            "defect_id": "dead_zone",
            "title": "Cut the dead air",
            "why": "Attention falls away for 4.0 seconds and does not recover.",
            "fix": "Hold the logo for 3 seconds from 12s and cut the copy to 16 words.",
        }],
    }

    verified = ev.verify(advice, defects=[defect(seconds=4.0)], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    assert len(verified["recommendations"]) == 1
    assert verified["evidence_audit"]["dropped"] == 0
    # ...but the same invented number stated as a FINDING still costs the item.
    claimed = {"summary": "", "recommendations": [{
        "defect_id": "dead_zone", "title": "Cut the dead air",
        "why": "There are 7 seconds of dead air.", "fix": "Tighten the middle."}]}
    assert ev.verify(claimed, defects=[defect(seconds=4.0)], summary=SUMMARY,
                     timeline=TIMELINE, transcript=SPOKEN,
                     duration=20.0)["recommendations"] == []


def test_a_number_the_model_was_shown_may_be_quoted_back():
    """REGRESSION. The allowed set was assembled by naming keys by hand, so it
    drifted from the context actually sent: a timeline point at 34.8s - a
    timestamp WE gave the model - came back in the prose and was rejected as
    invented, taking a sound recommendation with it. What we showed it, it may
    quote."""
    context = {"attention_timeline": [{"t": 34.8, "attention": 31}]}
    quoting = {"summary": "", "recommendations": [{
        "defect_id": "dead_zone", "title": "The dip at 34.8s",
        "why": "Attention is still falling 34.8 seconds in.", "fix": "Tighten it."}]}

    assert ev.verify(quoting, defects=[defect()], summary=SUMMARY,
                     timeline=TIMELINE, transcript=SPOKEN, duration=40.0,
                     context=context)["recommendations"]
    # Without having been shown it, the same number is not quotable.
    quoting["recommendations"][0]["verified"] = None
    assert ev.verify({"summary": "", "recommendations": [
        {"defect_id": "dead_zone", "title": "The dip at 34.8s",
         "why": "Attention is still falling 34.8 seconds in.",
         "fix": "Tighten it."}],
    }, defects=[defect()], summary=SUMMARY, timeline={"points": []},
        transcript=SPOKEN, duration=40.0)["recommendations"] == []


def test_a_hinglish_quote_is_matched_despite_its_punctuation():
    """REGRESSION, found on a real Hinglish ad. Punctuation normalises to a
    space, leaving "है, सर" as two spaces where a word-by-word comparison gives
    one - so a quote the presenter really had said was rejected and the trigger
    reported unsupported. Half of ScaleSerum's creatives are Hinglish."""
    hinglish = {"segments": [
        {"t": 11.4, "end": 16.0,
         "text": "और दूसरा client बोलता है, सर draft अभी चाहिए."}]}
    payload = {"summary": "", "recommendations": [], "triggers": [
        {"id": "framing_effect", "rating": "strong", "evidence": [
            {"t": 11.4, "quote": "और दूसरा client बोलता है, सर draft अभी चाहिए."}]}]}

    verified = ev.verify(payload, defects=[], summary=SUMMARY,
                         timeline=TIMELINE, transcript=hinglish, duration=20.0)

    assert len(verified["triggers"][0]["evidence"]) == 1
    assert verified["evidence_audit"]["dropped"] == 0


def test_a_mistyped_quote_with_a_real_timestamp_is_repaired_from_the_transcript():
    """REGRESSION, seen twice on the same Hinglish ad: the model reproduced a
    line as "दहले डेरे लषआ इसकव" that it had copied correctly the run before.
    Devanagari does not survive being retyped by a model reliably.

    The citation is a POINTER. When the timestamp lands on a real line, that
    line's own text becomes the quote - stricter than trusting the model's
    version, because the published words are now always ours - and the report
    says the model's wording was rejected.
    """
    hinglish = {"segments": [
        {"t": 11.4, "end": 16.0,
         "text": "पहले मेरे लिए इसका मतलब था रात भर research करना."}]}
    garbled = {"summary": "", "recommendations": [], "triggers": [
        {"id": "framing_effect", "rating": "strong", "evidence": [
            {"t": 11.4, "quote": "दहले डेरे लषआ इसकव डतलड थव रवत दीर research करनव."}]}]}

    verified = ev.verify(garbled, defects=[], summary=SUMMARY,
                         timeline=TIMELINE, transcript=hinglish, duration=20.0)

    piece = verified["triggers"][0]["evidence"][0]
    assert piece["quote"] == "पहले मेरे लिए इसका मतलब था रात भर research करना."
    assert piece["model_quote_rejected"] is True
    assert verified["triggers"][0].get("status") != "unsupported"


def test_a_mistyped_quote_pointing_at_no_line_is_still_dropped():
    """The repair is not a way in. Without a timestamp that resolves to a real
    line, a quote that is not in the transcript goes, as before."""
    payload = {"summary": "", "recommendations": [], "triggers": [
        {"id": "framing_effect", "rating": "strong", "evidence": [
            {"t": 18.5, "quote": "Rated number one in Asia for six years."}]}]}

    verified = ev.verify(payload, defects=[], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    assert verified["triggers"][0]["evidence"] == []
    assert verified["triggers"][0]["status"] == "unsupported"


def test_a_measured_number_restated_with_rounding_survives():
    """13.9 quoted as '14 seconds' is the same finding, not a fabrication. A
    verifier that cannot tell those apart would reject every honest report."""
    honest = {
        "summary": "",
        "recommendations": [{
            "defect_id": "dead_zone",
            "title": "Cut the dead air",
            "why": "There are 14 seconds where nothing holds the eye.",
            "fix": "Bring the proof point forward.",
        }],
    }

    verified = ev.verify(honest, defects=[defect(seconds=13.9)], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    assert len(verified["recommendations"]) == 1
    assert verified["evidence_audit"]["dropped"] == 0


def test_a_verified_recommendation_carries_the_defect_it_was_written_about():
    written = {"summary": "", "recommendations": [
        {"defect_id": "dead_zone", "title": "Cut the dead air",
         "why": "Attention falls away.", "fix": "Bring the proof point forward."}]}

    verified = ev.verify(written, defects=[defect()], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    item = verified["recommendations"][0]
    assert item["anchor"]["t_start"] == 12.0
    assert item["anchor"]["measured"]["seconds"] == 4.0
    assert item["severity"] == "high"
    assert item["scores_impacted"] == ["engagement"]
    assert item["rank"] == 1


def test_judgement_is_not_treated_as_a_claim_to_verify():
    """Interpretation is what we asked the model for. Only quotes, times and
    numbers are checked - prose about what the creative is DOING is not
    something a transcript can confirm or deny, and rejecting it would leave
    the report with nothing but numbers."""
    judged = {"summary": "", "recommendations": [
        {"defect_id": "dead_zone",
         "title": "The middle asks the viewer to wait",
         "why": "The leap from outcome to enrolment is where scepticism enters, "
                "and nothing on screen answers it.",
         "fix": "Answer the objection while attention is still high."}]}

    verified = ev.verify(judged, defects=[defect()], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    assert len(verified["recommendations"]) == 1
    assert verified["evidence_audit"]["dropped"] == 0


def test_a_rating_with_no_evidence_at_all_is_recorded_as_its_own_failure():
    """REGRESSION. A trigger rated with nothing cited became `unsupported`
    while the audit still read `dropped: 0` - which invites a reader to conclude
    the verifier had not run. Nothing cited and everything rejected are
    different failures and the audit now names which one happened."""
    payload = {"summary": "", "recommendations": [], "triggers": [
        {"id": "ikea_effect", "rating": "strong", "evidence": []}]}

    verified = ev.verify(payload, defects=[], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    assert verified["triggers"][0]["status"] == "unsupported"
    assert verified["triggers"][0]["reason"] == "no_evidence_cited"
    assert verified["evidence_audit"]["dropped"] == 1
    assert verified["evidence_audit"]["unsupported"][0]["reason"] == "no_evidence_cited"


def test_an_absent_rating_needs_no_evidence():
    """REGRESSION. There is nothing to quote for a trigger the creative did not
    use. Demanding evidence for an absence marked two honest "absent" verdicts
    on a real ad as `unsupported` - which reads as the verifier catching the
    model out, when the model had answered correctly."""
    payload = {"summary": "", "recommendations": [], "triggers": [
        {"id": "ikea_effect", "rating": "absent", "evidence": []}]}

    verified = ev.verify(payload, defects=[], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    assert verified["triggers"][0].get("status") != "unsupported"
    assert verified["evidence_audit"]["dropped"] == 0


def test_verification_works_without_a_transcript():
    """A silent creative must not fail every quote by default - there is nothing
    to check against, which is different from checking and finding nothing."""
    payload = {"summary": "", "recommendations": [], "triggers": [
        {"id": "loss_aversion", "rating": "weak",
         "evidence": [{"t": 8.0, "quote": "OFFER ENDS FRIDAY"}]}]}

    verified = ev.verify(payload, defects=[], summary=SUMMARY,
                         timeline=TIMELINE, transcript=None, duration=20.0)

    assert verified["triggers"][0]["evidence"][0]["quote"] == "OFFER ENDS FRIDAY"
    assert verified["evidence_audit"]["dropped"] == 0


# =========================================================================== #
# Step 18 - what the model is not allowed to return
# =========================================================================== #
def test_a_numeric_rating_is_refused_outright():
    """The rule the whole design rests on: the model rates, Python scores. A
    response carrying a number where a rating belongs is not repaired - it is
    rejected, because repairing it means guessing what it meant."""
    with pytest.raises(an.AnalyzerError) as raised:
        an.validate({"summary": "", "clarity": {"rating": "72"},
                     "recommendations": []}, {"defects": []})
    assert raised.value.reason == "analysis_invalid_output"


def test_a_defect_the_model_invented_is_dropped_not_published():
    parsed = an.validate({
        "summary": "",
        "recommendations": [
            {"defect_id": "dead_zone", "title": "Real", "why": "", "fix": ""},
            {"defect_id": "pacing_feels_slow", "title": "Invented", "why": "", "fix": ""},
        ]}, {"defects": [defect()]})

    assert [r["defect_id"] for r in parsed["recommendations"]] == ["dead_zone"]
    assert parsed["dropped_invented_recommendations"] == ["pacing_feels_slow"]


def test_a_score_smuggled_into_prose_is_stripped():
    parsed = an.validate({
        "summary": "",
        "recommendations": [{"defect_id": "dead_zone", "title": "Fix the dip",
                             "why": "This scores 42 out of 100 for engagement.",
                             "fix": ""}]}, {"defects": [defect()]})

    why = parsed["recommendations"][0]["why"]
    assert "42" not in why
    assert "[score removed]" in why


def test_the_context_sent_to_the_model_carries_no_score_to_copy():
    """The model is shown measurements and defects. It is NOT shown the six
    scores - if it were, the easiest thing it could do is restate them, and the
    separation between measuring and interpreting would exist only on paper."""
    context = an.build_context(
        media={"kind": "video", "duration_seconds": 20.0},
        summary=SUMMARY, timeline=TIMELINE, defects=[defect()],
        transcript=SPOKEN, triggers={"triggers": []})

    assert "scores" not in context
    assert "overall" not in context
    assert json.dumps(context, default=str)


# =========================================================================== #
# Step 17 - folding the ratings back onto the triggers
# =========================================================================== #
def measured_block():
    return psy.detect([], SUMMARY, TIMELINE, SPOKEN, fw.load_psychology(), None)


def test_the_triggers_run_against_the_real_measurement_record_shape():
    """REGRESSION. `text_boxes` in a stored record is a COUNT - measure.py
    collapses the boxes so a record is not 120 frames of OCR geometry - and
    psychology.py was calling len() on it. Nothing caught it because the stub
    record omitted the key entirely, so `or []` quietly supplied a list.

    The fix is in both places: the detector reads the integer, and the stub now
    carries every key measure.py emits, with the same type.
    """
    from vision_lab import stubs

    record = stubs.measure_frames({"frames": [{"index": 0, "t": 0.0}]})[0]
    assert isinstance(record["regions"]["text_boxes"], int)

    real_shaped = [{**record, "regions": {**record["regions"], "text_boxes": 4,
                                          "text": "APPLY NOW", "word_count": 2}}]
    block = psy.detect(real_shaped, SUMMARY, TIMELINE, SPOKEN,
                       fw.load_psychology(), None)

    compromise = next(t for t in block["triggers"] if t["id"] == "compromise_effect")
    assert compromise["measured"]["max_text_blocks_on_a_frame"] == 4


def test_a_closing_line_still_being_spoken_counts_as_the_close():
    """REGRESSION, found on a real 82.5 s ad. Its closing CTA begins at 76.3 s
    and runs to about 80.5 s, so it is what the viewer hears over the close -
    but the window matched on START time, and two triggers reported "nothing is
    said in the final seconds" about an ad that ends on its call to action."""
    late = {"segments": [
        {"t": 0.0, "end": 3.0, "text": "Aspiring directors, board seats are opening."},
        {"t": 76.3, "end": 80.5,
         "text": "Reserve your board seat before registrations fill up."}]}
    summary = {**SUMMARY, "duration_seconds": 82.5}

    block = psy.detect([], summary, TIMELINE, late, fw.load_psychology(), None)
    recency = next(t for t in block["triggers"] if t["id"] == "recency")

    assert recency["note"] != "nothing is said in the final seconds"
    assert recency["evidence"][0]["quote"].startswith("Reserve your board seat")


def test_a_single_digit_ocr_fragment_is_not_a_price():
    """REGRESSION, found on a real ad. OCR read "$8" on five frames of a webinar
    ad that shows no price at all - "$" is a routine misread of "S" - and that
    was enough to make Anchoring applicable and invite the model to judge a
    price nobody had shown. A one-digit price is noise; 499 and 1,299 are not.
    """
    from vision_lab import regions as reg

    assert reg.find_prices("Only $8 today") == []
    assert reg.find_prices("₹499 per seat") == ["₹499"]
    assert reg.find_prices("Rs 25,000 for the programme") == ["Rs 25,000"]


def test_the_rating_says_how_strongly_a_trigger_shows_not_whether_it_is_good():
    """REGRESSION. Choice Overload is the one trigger where showing it strongly
    is a DEFECT. It was returning rating "weak" alongside status "absent" - an
    ad that asked for exactly one thing, reported as if it had done badly."""
    block = measured_block()
    overload = next(t for t in block["triggers"] if t["id"] == "choice_overload")

    assert overload["status"] == "absent"
    assert overload["rating"] == "absent"
    # And the report says which direction is good news for this one.
    assert overload["polarity"] == "defect"
    assert next(t for t in block["triggers"]
                if t["id"] == "bandwagon")["polarity"] == "positive"


def test_a_measured_trigger_is_never_overwritten_by_the_model():
    """Bandwagon was answered by counting a numeral in the transcript. If the
    model disagrees, the count wins - measuring it was the whole point."""
    block = measured_block()
    before = next(t for t in block["triggers"] if t["id"] == "bandwagon")

    after_block = psy.apply_interpretation(block, {
        "available": True,
        "triggers": [{"id": "bandwagon", "rating": "absent",
                      "note": "no social proof at all"}]})
    after = next(t for t in after_block["triggers"] if t["id"] == "bandwagon")

    assert after["status"] == before["status"]
    assert after.get("judged_by") != "llm"


def test_a_not_applicable_trigger_stays_not_applicable():
    """Anchoring on a creative with no price on screen is not something the
    model gets to rate. Letting it would put a score on an opportunity the ad
    never had."""
    block = measured_block()
    anchoring = next(t for t in block["triggers"] if t["id"] == "anchoring")
    assert anchoring["status"] == "not_applicable"

    after_block = psy.apply_interpretation(block, {
        "available": True,
        "triggers": [{"id": "anchoring", "rating": "strong"}]})

    after = next(t for t in after_block["triggers"] if t["id"] == "anchoring")
    assert after["status"] == "not_applicable"
    assert after["rating"] is None


def test_a_judged_trigger_takes_its_rating_and_its_evidence():
    block = measured_block()
    framing = next(t for t in block["triggers"] if t["id"] == "framing_effect")
    assert framing["reason"] == "awaiting_interpretation"

    after_block = psy.apply_interpretation(block, {
        "available": True,
        "triggers": [{"id": "framing_effect", "rating": "strong",
                      "note": "the cost of standing still is named first",
                      "evidence": [{"t": 0.0, "quote": "Most operations directors think the bottleneck is headcount."}]}]})

    after = next(t for t in after_block["triggers"] if t["id"] == "framing_effect")
    assert after["status"] == "present"
    assert after["rating"] == "strong"
    assert after["judged_by"] == "llm"
    assert after["reason"] is None


def test_a_trigger_whose_evidence_failed_is_unsupported_not_absent():
    """'We could not stand behind this' and 'the creative did not do this' are
    different findings, and a marketer acting on the second when we mean the
    first would rewrite something that was working."""
    block = measured_block()

    after_block = psy.apply_interpretation(block, {
        "available": True,
        "triggers": [{"id": "framing_effect", "status": "unsupported",
                      "reason": "evidence_failed_verification"}]})

    after = next(t for t in after_block["triggers"] if t["id"] == "framing_effect")
    assert after["status"] == "unsupported"
    assert after["rating"] is None
    assert after_block["unsupported"] == 1


def test_an_unavailable_interpretation_leaves_the_measured_triggers_intact():
    block = measured_block()
    unchanged = psy.apply_interpretation(block, an.unavailable("analysis_provider_error"))
    assert unchanged["triggers"] == block["triggers"]


# =========================================================================== #
# The two metrics the interpretation completes
# =========================================================================== #
def frame(t, text_mass=0.25, text_area=0.1):
    return {"index": int(t * 2), "t": t, "shot": 0, "seconds_into_shot": 1.0,
            "frame_contrast": 0.3, "motion_energy": 0.05, "gaze_stability": 0.8,
            "saliency": {"concentration": 0.3, "peak_count": 2,
                         "peaks": [{"box": [0.3, 0.3, 0.6, 0.6], "share": 0.3, "rank": 1}]},
            "regions": {"word_count": 6, "text_area_share": text_area,
                        "ocr_available": True, "faces": 0, "brand": None,
                        "cta": None, "prices": []},
            "mass": {"text": text_mass, "faces": 0.0, "brand": None, "cta": None}}


def test_focus_measures_where_the_key_message_is_not_the_whole_creative():
    """Without a named key message, Focus averages over every word OCR read -
    a legal line and the headline counted alike. With one, it is measured where
    the claim is actually on screen, and says so."""
    cfg = fw.load_framework()
    # Gaze is on the copy at 8 s and nowhere near it for the rest of the ad.
    frames = ([frame(t, text_mass=0.05) for t in (0.0, 2.0, 4.0)]
              + [frame(t, text_mass=0.45) for t in (7.5, 8.0, 8.5)]
              + [frame(t, text_mass=0.05) for t in (12.0, 14.0)])

    whole = sc.focus(frames, cfg)
    at_key = sc.focus(frames, cfg, {"element": "headline", "t": 8.0})

    assert at_key["score"] > whole["score"]
    assert at_key["basis"] == "measured"
    assert at_key["reason"] is None
    assert at_key["signals"]["frames_at_key_message"] == 3
    # Without one it is honest about being partial rather than presenting the
    # average as the finished measure.
    assert whole["basis"] == "measured_partial"
    assert whole["reason"] == sc.AWAITING_LLM


def test_a_key_message_pointing_nowhere_falls_back_rather_than_measuring_one_frame():
    cfg = fw.load_framework()
    frames = [frame(t) for t in (0.0, 2.0, 4.0)]

    result = sc.focus(frames, cfg, {"element": "end card", "t": 45.0})

    assert result["basis"] == "measured_partial"
    assert result["reason"] == sc.KEY_MESSAGE_OUT_OF_RANGE
    assert "frames_at_key_message" not in result["signals"]


def test_a_spoken_key_message_does_not_score_focus_zero():
    """REGRESSION, found on a real ad. The model correctly named the central
    claim at 70.8 s - where the VOICEOVER states it, with no copy on screen.
    Focus then measured gaze against text that was not there and scored 0 for an
    ad whose copy was in fact being read.

    Gaze cannot land on a sentence nobody drew. The window is used only where
    the frames actually carry text, and Focus says why it fell back.
    """
    cfg = fw.load_framework()
    # Copy on screen early; the claim is spoken over a text-free shot at 40 s.
    frames = ([frame(t, text_mass=0.35, text_area=0.1) for t in (0.0, 2.0, 4.0)]
              + [frame(t, text_mass=0.0, text_area=0.0) for t in (39.0, 40.0, 41.0)])

    spoken = sc.focus(frames, cfg,
                      {"element": "voiceover claim", "t": 40.0,
                       "carrier": "voiceover"})

    assert spoken["reason"] == sc.KEY_MESSAGE_NOT_ON_SCREEN
    assert spoken["basis"] == "measured_partial"
    # And it is the whole-creative figure, not a zero.
    assert spoken["score"] == sc.focus(frames, cfg)["score"]
    assert spoken["score"] > 0


def test_a_spoken_claim_is_not_measured_against_whatever_text_happens_to_be_there():
    """THE SECOND HALF OF THE SAME REGRESSION. Checking only 'is there text on
    screen' was not enough: on the real ad the voiceover carried the claim while
    unrelated copy sat on screen, so the window was used and Focus measured gaze
    against text that was not the message. The model's `carrier` settles it, and
    the frame check remains for when the model claims on_screen_text wrongly."""
    cfg = fw.load_framework()
    frames = ([frame(t, text_mass=0.35, text_area=0.1) for t in (0.0, 2.0, 4.0)]
              # Plenty of text at 40 s - just not the claim.
              + [frame(t, text_mass=0.02, text_area=0.12) for t in (39.0, 40.0, 41.0)])

    spoken = sc.focus(frames, cfg, {"element": "the voiceover", "t": 40.0,
                                    "carrier": "voiceover"})
    written = sc.focus(frames, cfg, {"element": "headline", "t": 40.0,
                                     "carrier": "on_screen_text"})

    assert spoken["reason"] == sc.KEY_MESSAGE_NOT_ON_SCREEN
    assert "frames_at_key_message" not in spoken["signals"]
    # The same moment, claimed as written copy, IS measured there.
    assert written["basis"] == "measured"
    assert written["signals"]["frames_at_key_message"] == 3


def test_clarity_blends_the_rating_but_the_measurement_still_moves_it():
    cfg = fw.load_framework()
    frames = [{**frame(t), "regions": {**frame(t)["regions"],
                                       "cta": {"text": "Apply now",
                                               "box": [0.3, 0.8, 0.7, 0.88]}}}
              for t in (0.0, 0.5, 1.0, 1.5)]
    summary = {**SUMMARY, "cta": {"detected": True, "weak": False}}

    unrated = sc.clarity(frames, summary, cfg)
    strong = sc.clarity(frames, summary, cfg, "strong")
    absent = sc.clarity(frames, summary, cfg, "absent")

    assert unrated["reason"] == sc.AWAITING_LLM
    assert unrated["basis"] == "measured_partial"
    assert strong["basis"] == "hybrid"
    assert strong["score"] > absent["score"]
    # The rating is blended, never substituted: an "absent" rating cannot drive
    # a clearly legible, long-held CTA to zero.
    assert absent["score"] > 0


# =========================================================================== #
# Step 16 - the transcript join
# =========================================================================== #
def test_each_line_takes_the_attention_measured_over_its_own_span():
    """The per-line number is a JOIN over two things already computed, not a
    second prediction. A line overlapping a weak zone is what draws the warning
    icon in the transcript panel."""
    lines = transcript((0.0, "Opening claim."), (12.0, "The middle."))
    timeline = {"points": [{"t": 0.0, "attention": 80}, {"t": 2.0, "attention": 70},
                           {"t": 12.0, "attention": 20}, {"t": 14.0, "attention": 24}],
                "weak_zones": [{"t_start": 11.0, "t_end": 16.0, "seconds": 5.0}]}

    joined = tx.join_to_timeline(lines, timeline)

    assert joined["segments"][0]["attention"] == 75
    assert joined["segments"][0]["in_weak_zone"] is False
    assert joined["segments"][1]["attention"] == 22
    assert joined["segments"][1]["in_weak_zone"] is True


def test_a_creative_with_no_timeline_still_returns_its_transcript():
    lines = transcript((0.0, "A still image has no timeline."))
    assert tx.join_to_timeline(lines, None)["segments"][0]["text"]


def test_the_transcript_summary_reports_what_the_triggers_will_use():
    joined = tx.join_to_timeline(
        transcript((0.0, "One two three four five."), (4.0, "Six seven eight.")),
        {"points": [{"t": 0.0, "attention": 60}], "weak_zones": []})
    stats = tx.summarise(joined)

    assert stats["lines"] == 2
    assert stats["words"] == 8
    assert stats["seconds_of_speech"] == 6.0


# =========================================================================== #
# The published contract
# =========================================================================== #
def test_the_report_version_moves_when_the_report_shape_does():
    """REGRESSION, caught by submitting a real ad to the running service.

    PROMPT_VERSION is part of the idempotency fingerprint. Milestone D added the
    transcript, the 15 triggers and the recommendations - and because the
    version had not moved, a re-submitted creative kept returning its old
    Milestone-A report. Correct by the fingerprint, and useless to the caller.

    This test cannot know what the next shape is; it pins the CURRENT one to the
    CURRENT version, so changing the payload without changing the version fails
    here rather than in an integrator's cache.
    """
    import vision_lab as vl

    assert vl.PROMPT_VERSION == "vl_2026_09_d", (
        "The report shape and PROMPT_VERSION have to move together. If you "
        "changed the payload, bump the version and update this test; if you "
        "did not, put it back.")


def test_the_report_still_matches_the_example_handed_to_the_frontend_team():
    """`tests/fixtures/vision_lab/example_report.json` is a REAL report from a
    real ad, and it is item 5 of the submission pack - the frontend team builds
    against it. A field quietly disappearing from the payload is a broken UI
    they find out about in integration, so the shape is pinned here.

    Add a field freely; this only fails when one goes missing or is renamed.
    """
    from vision_lab import report as rp
    from vision_lab import stubs

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", "vision_lab", "example_report.json")
    with open(path, encoding="utf-8") as handle:
        example = json.load(handle)

    built = rp.build_report(
        {"_id": "vl_x", "creative_id": "c", "ad_number": "1", "division": "d"},
        media={"kind": "video", "duration_seconds": 20.0, "frames": []},
        summary=SUMMARY, heatmaps={"hero": None, "strip": []}, kind="video",
        sample_fps=2.0, notes=[], vision=stubs.saliency_info(),
        timeline=TIMELINE, key_moments=[], scored=None, defects=[],
        transcript=SPOKEN, triggers=None, interpretation=None)

    missing = set(example) - set(built)
    assert not missing, f"the report no longer carries: {sorted(missing)}"

    for block in ("scores", "overall", "media", "heatmap", "interpretation"):
        gone = set(example[block]) - set(built[block])
        assert not gone, f"{block} no longer carries: {sorted(gone)}"

    # The example must be a finished report, not a placeholder someone saved.
    assert example["stub"] is False
    assert example["interpretation"]["available"] is True
    assert example["transcript"]["available"] is True
    assert example["recommendations"]
    assert len(example["psychology"]["triggers"]) == 15

    # THE ATTENTION REPORT MUST BE DEMONSTRATED, not just declared. The first
    # version of this file was generated with S3 upload switched off, so the one
    # screen the frontend most needs a reference for - the heatmap with its
    # numbered markers - was empty in their reference file.
    heatmap = example["heatmap"]
    assert heatmap["object_key"], "the example has no rendered heatmap"
    assert heatmap["frame_time"] is not None
    assert heatmap["peaks"], "the example has no attention peaks to place markers from"
    assert example["thumbnails"], "the example has no key-moment strip"
    for peak in heatmap["peaks"]:
        assert set(peak) >= {"rank", "box", "share", "element", "element_text"}
        assert len(peak["box"]) == 4
        assert all(0.0 <= v <= 1.0 for v in peak["box"]), "boxes are FRACTIONAL"
        assert 0.0 <= peak["share"] <= 1.0
    # At least one peak is named, or the labelling has silently stopped working.
    assert any(p["element"] for p in heatmap["peaks"])

    # A signed URL is a credential and this file goes to another team.
    body = json.dumps(example)
    assert "X-Amz-Signature=SIGNED_AT_READ_TIME" in body
    assert not re.search(r"X-Amz-Signature=[0-9a-f]{16}", body),         "a live presigned signature is committed in the example report"


# =========================================================================== #
# End to end, with a model that behaves badly
# =========================================================================== #
class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def generate_content(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return FakeResponse(json.dumps(self.payload))


class FakeClient:
    def __init__(self, payload):
        self.aio = type("Aio", (), {"models": FakeModels(payload)})()


def test_interpret_verifies_before_returning_anything():
    """interpret() is the only door into the model's output, and it does not
    open without validation. A caller cannot get the unverified version even by
    mistake."""
    client = FakeClient({
        "summary": "The hook works; the middle does not.",
        "clarity": {"rating": "adequate"},
        "key_message": {"element": "headline", "t": 8.0,
                        "quote": "Applications close on the thirtieth."},
        "recommendations": [
            {"defect_id": "dead_zone", "title": "Fill the dip",
             "why": "Attention falls away.", "fix": "Move the proof point up."},
            {"defect_id": "the_music_is_wrong", "title": "Change the track",
             "why": "", "fix": ""}],
        "triggers": [{"id": "framing_effect", "rating": "strong",
                      "evidence": [{"t": 99.0, "quote": "Never said."}]}],
    })

    context = an.build_context(media={"kind": "video", "duration_seconds": 20.0},
                               summary=SUMMARY, timeline=TIMELINE,
                               defects=[defect()], transcript=SPOKEN,
                               triggers={"triggers": []})
    parsed = asyncio.run(an.interpret(context, client=client, model="fake"))
    verified = ev.verify(parsed, defects=[defect()], summary=SUMMARY,
                         timeline=TIMELINE, transcript=SPOKEN, duration=20.0)

    # The invented defect never made it out of validate().
    assert parsed["dropped_invented_recommendations"] == ["the_music_is_wrong"]
    # The real one survives and is anchored.
    assert len(verified["recommendations"]) == 1
    assert verified["recommendations"][0]["anchor"]["t_start"] == 12.0
    # The fabricated quote at 99 s in a 20 s ad does not.
    assert verified["triggers"][0]["status"] == "unsupported"
    assert verified["evidence_audit"]["dropped"] == 1


def test_a_measured_rate_is_not_mistaken_for_a_score():
    """REGRESSION, found in a real client-facing recommendation.

    The stripper matched bare "rate", so "at a standard reading rate of 4.0
    words per second" was published as "at a standard reading [score removed].0
    words per second" - a sentence correctly quoting a MEASURED value, mangled
    by the guard meant to catch invented ones. A rate is a frequency, not a
    verdict: reading rate, cut rate, click rate, frame rate.
    """
    kept = an.validate({
        "summary": "",
        "recommendations": [{
            "defect_id": "dead_zone", "title": "Trim the slide",
            "why": "At a standard reading rate of 4.0 words per second, 24 words "
                   "demand 6.0s of screen time.",
            "fix": "Cut it to 14 words."}]}, {"defects": [defect()]})

    why = kept["recommendations"][0]["why"]
    assert "[score removed]" not in why
    assert "reading rate of 4.0 words per second" in why
    # The orphaned decimal was the other half of the same bug.
    assert ".0 words" in why


def test_the_judgements_that_ARE_scores_are_still_stripped():
    """The guard must not be loosened into uselessness - these are the phrasings
    a model actually reaches for when it decides to score something itself."""
    for prose in ("This scores 42 out of 100 for engagement.",
                  "I would score it 7/10 overall.",
                  "The creative is rated 8 out of 10.",
                  "Clarity rating: 65 for this ad.",
                  "It scored 55 on brand memory."):
        cleaned = an.validate({
            "summary": "",
            "recommendations": [{"defect_id": "dead_zone", "title": "t",
                                 "why": prose, "fix": ""}]},
            {"defects": [defect()]})["recommendations"][0]["why"]
        assert "[score removed]" in cleaned, prose
        assert not re.search(r"\d", cleaned.replace("[score removed]", "")), prose


def test_an_ordinal_rating_in_prose_is_left_alone():
    """The model is ASKED for ordinal ratings. "rated strong" is the correct
    answer, not a smuggled score, and must survive."""
    kept = an.validate({
        "summary": "",
        "recommendations": [{"defect_id": "dead_zone", "title": "t",
                             "why": "Attention is rated strong through the hook.",
                             "fix": ""}]}, {"defects": [defect()]})
    assert kept["recommendations"][0]["why"] == \
        "Attention is rated strong through the hook."
