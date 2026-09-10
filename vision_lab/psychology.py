"""
Step 17 - the 15 psychological triggers.

WHAT STOPS THIS BEING A HOROSCOPE
    Eight of the fifteen are MEASURED, not judged. Serial Position and Recency
    fall out of the transcript and the timeline; Mere Exposure shares its signal
    with Brand Memory; Bandwagon and Choice Overload are counted from the
    transcript and the OCR text; Anchoring needs a price to be on screen at all.

    A trigger detected by counting something carries the count. A trigger the
    LLM rates carries the evidence it cited, and evidence.py checks that
    evidence against the transcript before any of it reaches a client.

NOT APPLICABLE IS NOT ABSENT, AND NEITHER IS A ZERO
    Anchoring on a creative that shows no price is `not_applicable` with a
    stated reason. Marking a brand down for failing to anchor a price it was
    never going to show is an artefact of the framework, not a finding about the
    ad - and it is exactly the kind of thing that makes a scorecard untrusted.

TRIGGER 15 IS DELIBERATELY UNSCORED
    Blind-Spot Bias is a property of the MARKETER - the assumption that the
    viewer already shares their framing. It is not an artefact present in the
    frames. Scoring it would mean inventing a measurement, so it is returned as
    a report-level observation with no rating and no weight.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from . import (
    TRIGGER_ABSENT,
    TRIGGER_NOT_APPLICABLE,
    TRIGGER_PRESENT,
    TRIGGER_UNSUPPORTED,
    TRIGGER_WEAK,
)
from . import framework as fw

logger = logging.getLogger("vision_lab.psychology")

# How close to the start and end counts as "first" and "last" for the Serial
# Position and Recency effects. 3 s is the platform convention for a counted
# view and is already `hook_seconds` in config.
CLOSE_SECONDS = 3.0

# Similarity above which the close is judged to restate the opening.
RESTATEMENT_THRESHOLD = 0.25


def _finding(trigger: dict, status: str, *, rating: Optional[str] = None,
             evidence: Optional[list] = None, reason: Optional[str] = None,
             note: str = "", measured: Optional[dict] = None) -> dict:
    return {
        "id": trigger["id"],
        "number": trigger.get("number"),
        "name": trigger["name"],
        "subtitle": trigger.get("subtitle"),
        "status": status,
        "rating": rating,
        # Which direction is good news. Choice Overload is the one trigger where
        # exhibiting it strongly is a DEFECT, and a scorecard that does not say
        # so reads its 'present' as a win.
        "polarity": trigger.get("polarity", "positive"),
        "detection": trigger.get("detection"),
        "scored": trigger.get("scored", True),
        "reason": reason,
        "evidence": evidence or [],
        "measured": measured or {},
        "note": note,
    }


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9ऀ-ॿ]+", (text or "").lower())
            if len(w) > 2}


def _overlap(a: str, b: str) -> float:
    """Jaccard similarity on content words. Crude, and deliberately so - it is
    checking whether the close reuses the opening's language, not whether it
    means the same thing, which is a judgement left to the LLM."""
    first, second = _words(a), _words(b)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _spoken(transcript: dict) -> str:
    return " ".join(s.get("text", "") for s in (transcript or {}).get("segments") or [])


def _on_screen(measurements: list[dict]) -> str:
    return " ".join((m.get("regions") or {}).get("text", "") for m in measurements)


def _matches(text: str, phrases: list[str]) -> list[str]:
    lowered = (text or "").lower()
    return [p for p in (phrases or []) if p.lower() in lowered]


def _opening_and_closing(segments: list[dict], duration: float) -> tuple:
    """The words spoken in the first and last few seconds.

    A LINE COUNTS AS THE CLOSE IF IT IS STILL BEING SPOKEN THERE.
        Matching on start time alone, as this first did, reported "nothing is
        said in the final seconds" for a real ad whose closing CTA begins at
        76.3 s and runs to about 80.5 s of an 82.5 s creative. The viewer hears
        it over the close; that it began a moment earlier is not a finding.
    """
    boundary = max(0.0, (duration or 0) - CLOSE_SECONDS)
    opening = " ".join(s["text"] for s in segments if s["t"] <= CLOSE_SECONDS)
    closing = " ".join(s["text"] for s in segments
                       if (s.get("end") if s.get("end") is not None
                           else s["t"] + 2.0) >= boundary)
    return opening, closing


# --------------------------------------------------------------------------- measured
def _serial_position(trigger, ctx) -> dict:
    """Does the core claim occupy BOTH the opening and the close?"""
    segments = ctx["transcript"].get("segments") or []
    if not segments and not ctx["ocr_text"]:
        return _finding(trigger, TRIGGER_NOT_APPLICABLE,
                        reason="no_transcript_or_on_screen_text")

    duration = ctx["duration"] or 0
    opening, closing = _opening_and_closing(segments, duration)

    if not opening or not closing:
        return _finding(trigger, TRIGGER_ABSENT,
                        note="the ad does not speak in both the opening and the close",
                        measured={"opening_words": len(opening.split()),
                                  "closing_words": len(closing.split())})

    similarity = _overlap(opening, closing)
    status = (TRIGGER_PRESENT if similarity >= RESTATEMENT_THRESHOLD
              else TRIGGER_WEAK if similarity > 0.1 else TRIGGER_ABSENT)
    return _finding(
        trigger, status,
        rating="strong" if status == TRIGGER_PRESENT else "weak",
        evidence=[{"t": 0.0, "quote": opening[:160], "source": "transcript"},
                  {"t": round(max(0, duration - CLOSE_SECONDS), 1),
                   "quote": closing[:160], "source": "transcript"}],
        measured={"opening_close_similarity": round(similarity, 3),
                  "threshold": RESTATEMENT_THRESHOLD},
        note=(f"the close reuses {similarity * 100:.0f}% of the opening's "
              f"language"))


def _recency(trigger, ctx) -> dict:
    """Is the promise restated at the close, or is the viewer left holding a CTA
    with no reason attached?"""
    segments = ctx["transcript"].get("segments") or []
    if not segments:
        return _finding(trigger, TRIGGER_NOT_APPLICABLE,
                        reason="no_transcript_or_on_screen_text")

    duration = ctx["duration"] or 0
    opening, closing = _opening_and_closing(segments, duration)
    similarity = _overlap(opening, closing)

    if not closing:
        return _finding(trigger, TRIGGER_ABSENT,
                        note="nothing is said in the final seconds")
    status = TRIGGER_PRESENT if similarity >= RESTATEMENT_THRESHOLD else TRIGGER_ABSENT
    return _finding(
        trigger, status, rating="strong" if status == TRIGGER_PRESENT else "absent",
        evidence=[{"t": round(max(0, duration - CLOSE_SECONDS), 1),
                   "quote": closing[:160], "source": "transcript"}],
        measured={"opening_close_similarity": round(similarity, 3)},
        note=("the close restates the opening promise" if status == TRIGGER_PRESENT
              else "the close does not restate the promise the ad opened with"))


def _mere_exposure(trigger, ctx) -> dict:
    """Repeated brand exposure. Shares its signal with Brand Memory."""
    brand = ctx["summary"].get("brand") or {}
    if not brand.get("detected"):
        return _finding(trigger, TRIGGER_NOT_APPLICABLE,
                        reason=trigger.get("not_applicable_reason")
                        or "no_brand_detected")

    seconds = brand.get("exposure_seconds") or 0.0
    duration = ctx["duration"] or 1
    share = seconds / duration
    status = (TRIGGER_PRESENT if share >= 0.25 else
              TRIGGER_WEAK if share >= 0.05 else TRIGGER_ABSENT)
    return _finding(
        trigger, status,
        rating="strong" if share >= 0.5 else "adequate" if share >= 0.25 else "weak",
        measured={"exposure_seconds": round(seconds, 2),
                  "share_of_runtime": round(share, 3),
                  "first_appearance_seconds": brand.get("first_appearance_seconds")},
        note=(f"the brand is on screen for {seconds:.1f}s, "
              f"{share * 100:.0f}% of the runtime"))


def _bandwagon(trigger, ctx) -> dict:
    """Evidence others have already acted. A specific number beats a vague claim."""
    text = f"{ctx['spoken']} {ctx['ocr_text']}"
    if not text.strip():
        return _finding(trigger, TRIGGER_NOT_APPLICABLE,
                        reason="no_transcript_or_on_screen_text")

    phrases = _matches(text, fw.lexicon("social_proof_phrase"))
    # A bare adoption numeral is the strongest form: "4,200 directors certified"
    # persuades where "trusted by many" does not.
    numerals = re.findall(r"\b\d[\d,]{2,}\b", text)
    spelled = re.findall(
        r"\b(?:hundred|thousand|lakh|crore|million)\b", text.lower())

    # A number said aloud is still a number. "Four thousand two hundred
    # directors" persuades exactly as "4,200 directors" does, and a
    # digits-only test would report the strongest possible form of this trigger
    # as the weakest.
    quantified = bool(numerals or spelled)
    status = (TRIGGER_PRESENT if quantified and phrases else
              TRIGGER_WEAK if phrases or quantified else TRIGGER_ABSENT)
    return _finding(
        trigger, status,
        rating=("strong" if quantified and phrases else
                "adequate" if phrases else "weak"),
        evidence=_quote_evidence(ctx, (numerals[:2] + spelled[:1] + phrases[:2])),
        measured={"adoption_numerals": numerals[:5],
                  "spelled_quantities": spelled[:5],
                  "social_proof_phrases": phrases[:5]},
        note=("a specific adoption number is stated" if quantified
              else "social proof is claimed without a number" if phrases
              else "no evidence that others have already acted"))


def _choice_overload(trigger, ctx) -> dict:
    """Too many asks. Shares its competing-peaks signal with Cognitive Demand."""
    ctas = {(m.get("regions") or {}).get("cta", {}).get("text")
            for m in ctx["measurements"] if (m.get("regions") or {}).get("cta")}
    ctas.discard(None)
    peaks = [(m.get("saliency") or {}).get("peak_count")
             for m in ctx["measurements"]]
    usable = [p for p in peaks if p is not None]
    mean_peaks = sum(usable) / len(usable) if usable else 0

    # The trigger fires when the ad asks for too much, so PRESENT here is a
    # defect, not a virtue - the report has to say which way round that is.
    status = (TRIGGER_PRESENT if len(ctas) > 2 or mean_peaks > 4 else
              TRIGGER_WEAK if len(ctas) == 2 or mean_peaks > 3 else TRIGGER_ABSENT)
    # The rating says how strongly the trigger is EXHIBITED, never whether that
    # is good news - `polarity` answers that. Returning "weak" for an absent
    # overload, as this first did, reads as a poor result for an ad that in fact
    # asked for exactly one thing.
    return _finding(
        trigger, status,
        rating={TRIGGER_PRESENT: "strong", TRIGGER_WEAK: "weak"}.get(status, "absent"),
        measured={"distinct_ctas": sorted(c for c in ctas if c),
                  "mean_competing_peaks": round(mean_peaks, 2)},
        note=(f"{len(ctas)} distinct calls to action and an average of "
              f"{mean_peaks:.1f} competing attention centres - overload is a "
              f"defect, so PRESENT here is bad news"))


def _anchoring(trigger, ctx) -> dict:
    """Not applicable without a price. That is a fact about the creative, not a
    failure by it."""
    prices = [p for m in ctx["measurements"]
              for p in ((m.get("regions") or {}).get("prices") or [])]
    if not prices:
        return _finding(trigger, TRIGGER_NOT_APPLICABLE,
                        reason=trigger.get("not_applicable_reason")
                        or "no_price_shown",
                        note="no price appears on screen, so there is nothing to anchor")
    return _finding(trigger, TRIGGER_PRESENT, rating="adequate",
                    measured={"prices_detected": prices[:5]},
                    evidence=_quote_evidence(ctx, prices[:2]),
                    note=("a price is shown - whether an anchor precedes it is "
                          "for the interpretation pass to judge"))


def _compromise(trigger, ctx) -> dict:
    """Exactly three options is the effect; two or seven is not."""
    # `text_boxes` in the measurement record is a COUNT, not the boxes - see
    # measure.py, which collapses them so a stored record is not 120 frames of
    # OCR geometry. Calling len() on it is how this first ran on a real ad.
    counts = [(m.get("regions") or {}).get("text_boxes") or 0
              for m in ctx["measurements"]]
    # Option blocks are not reliably separable from any other text with OCR
    # alone, so this reports what it can and defers rather than guessing.
    return _finding(trigger, TRIGGER_NOT_APPLICABLE,
                    reason=trigger.get("not_applicable_reason")
                    or "no_options_presented",
                    measured={"max_text_blocks_on_a_frame": max(counts) if counts else 0},
                    note=("distinct option blocks cannot be separated from other "
                          "on-screen text by OCR alone - deferred to the "
                          "interpretation pass"))


def _loss_aversion(trigger, ctx) -> dict:
    text = f"{ctx['spoken']} {ctx['ocr_text']}"
    deadlines = _matches(text, fw.lexicon("deadline_phrase"))
    scarcity = _matches(text, fw.lexicon("scarcity_numeral"))
    status = (TRIGGER_PRESENT if deadlines else
              TRIGGER_WEAK if scarcity else TRIGGER_ABSENT)
    return _finding(
        trigger, status, rating="strong" if deadlines else "weak",
        evidence=_quote_evidence(ctx, (deadlines[:2] + scarcity[:2])),
        measured={"deadline_phrases": deadlines[:5],
                  "scarcity_phrases": scarcity[:5]},
        note=("a deadline or closing date is stated" if deadlines
              else "scarcity language without a deadline" if scarcity
              else "nothing is framed as something to lose"))


def _peltzman(trigger, ctx) -> dict:
    text = f"{ctx['spoken']} {ctx['ocr_text']}"
    found = _matches(text, fw.lexicon("risk_reversal_phrase"))
    status = TRIGGER_PRESENT if found else TRIGGER_ABSENT
    return _finding(
        trigger, status, rating="strong" if found else "absent",
        evidence=_quote_evidence(ctx, found[:2]),
        measured={"risk_reversal_phrases": found[:5]},
        note=("perceived risk is lowered explicitly" if found
              else "no guarantee, trial or reversible commitment is offered"))


def _halo(trigger, ctx) -> dict:
    """The opening sets the frame. Measured on attention; the LLM judges the
    quality of the impression itself."""
    points = [p for p in (ctx["timeline"].get("points") or [])
              if p.get("attention") is not None]
    if not points:
        return _finding(trigger, TRIGGER_NOT_APPLICABLE, reason="no_timeline")

    hook = [p["attention"] for p in points if p["t"] <= CLOSE_SECONDS]
    rest = [p["attention"] for p in points if p["t"] > CLOSE_SECONDS]
    if not hook:
        return _finding(trigger, TRIGGER_NOT_APPLICABLE, reason="no_timeline")

    hook_mean = sum(hook) / len(hook)
    baseline = sum(rest) / len(rest) if rest else hook_mean
    status = (TRIGGER_PRESENT if hook_mean > baseline + 5 else
              TRIGGER_WEAK if hook_mean >= baseline - 5 else TRIGGER_ABSENT)
    return _finding(
        trigger, status,
        rating="strong" if status == TRIGGER_PRESENT else "weak",
        measured={"hook_attention": round(hook_mean, 1),
                  "rest_of_ad_attention": round(baseline, 1),
                  "hook_seconds": CLOSE_SECONDS},
        note=(f"the first {CLOSE_SECONDS:g}s score {hook_mean:.0f} against "
              f"{baseline:.0f} for the rest"))


def _quote_evidence(ctx, needles: list[str]) -> list[dict]:
    """Find where each matched phrase was actually said, so evidence.py has a
    timestamp to verify rather than a bare assertion."""
    out = []
    for needle in needles:
        if not needle:
            continue
        for segment in ctx["transcript"].get("segments") or []:
            if str(needle).lower() in segment.get("text", "").lower():
                out.append({"t": segment["t"], "quote": segment["text"][:160],
                            "source": "transcript", "matched": str(needle)})
                break
        else:
            out.append({"t": None, "quote": str(needle), "source": "on_screen_text",
                        "matched": str(needle)})
    return out


MEASURED = {
    "halo_effect": _halo,
    "serial_position": _serial_position,
    "recency": _recency,
    "mere_exposure": _mere_exposure,
    "loss_aversion": _loss_aversion,
    "compromise_effect": _compromise,
    "anchoring": _anchoring,
    "choice_overload": _choice_overload,
    "peltzman_effect": _peltzman,
    "bandwagon": _bandwagon,
}


# --------------------------------------------------------------------------- entry
def detect(measurements: list[dict], summary: dict, timeline: Optional[dict],
           transcript: Optional[dict], cfg: Optional[dict] = None,
           brand_brain: Optional[dict] = None) -> dict:
    """Every trigger. Measured ones answered here; judged ones left for the LLM.

    A judged trigger comes back `status="absent"` with `awaiting_interpretation`
    rather than a verdict, so a caller can tell "we looked and found nothing"
    from "nothing has looked yet".
    """
    cfg = cfg or fw.load_psychology()
    triggers = cfg.get("triggers") or []
    timeline = timeline or {"points": []}
    transcript = transcript or {"segments": []}

    ctx = {
        "measurements": measurements,
        "summary": summary,
        "timeline": timeline,
        "transcript": transcript,
        "spoken": _spoken(transcript),
        "ocr_text": _on_screen(measurements),
        "duration": summary.get("duration_seconds") or 0,
        "brand_brain": brand_brain,
    }

    findings = []
    for trigger in triggers:
        detector = MEASURED.get(trigger["id"])
        if detector:
            findings.append(detector(trigger, ctx))
            continue

        # Trigger 15 is a property of the marketer, not of the frames.
        if trigger.get("scored") is False:
            findings.append(_finding(
                trigger, TRIGGER_NOT_APPLICABLE,
                reason="report_level_observation_only",
                note=trigger.get("scored_note", "")[:220]))
            continue

        if "brand_brain" in (trigger.get("applies_when") or {}).get("requires", []) \
                and not brand_brain:
            findings.append(_finding(
                trigger, TRIGGER_NOT_APPLICABLE,
                reason=trigger.get("not_applicable_reason") or "no_brand_brain",
                note="needs the brand's persona to judge against"))
            continue

        findings.append(_finding(
            trigger, TRIGGER_ABSENT, reason="awaiting_interpretation",
            note="judged by the interpretation pass, which has not run"))

    return {"triggers": findings, "coverage": coverage(findings),
            "version": cfg.get("psychology_version"),
            "affects_overall_score": bool(cfg.get("affects_overall_score"))}


RATING_STATUS = {"strong": TRIGGER_PRESENT, "adequate": TRIGGER_PRESENT,
                 "weak": TRIGGER_WEAK, "absent": TRIGGER_ABSENT}


def apply_interpretation(block: dict, interpretation: Optional[dict]) -> dict:
    """Fold the model's verified ratings onto the triggers it was asked to judge.

    THREE RULES, AND THEY ARE WHY THIS IS A SEPARATE FUNCTION RATHER THAN A LOOP
    IN THE PIPELINE:

      1. A MEASURED TRIGGER IS NEVER OVERWRITTEN. Bandwagon was answered by
         counting a numeral in the transcript. If the model disagrees, the count
         wins - the whole point of measuring it was to not have to ask.
      2. A TRIGGER MARKED not_applicable STAYS THAT WAY. Anchoring on a creative
         with no price on screen is not a thing the model gets to rate.
      3. EVIDENCE MUST HAVE SURVIVED evidence.py. A rating whose quotes were all
         dropped is reported `unsupported`, not `absent` - we do not know how
         the creative did, which is different from knowing it did badly.
    """
    interpretation = interpretation or {}
    if not interpretation.get("available"):
        return block

    judged = {t.get("id"): t for t in interpretation.get("triggers") or []}
    findings = []
    for finding in block.get("triggers") or []:
        verdict = judged.get(finding["id"])
        judgeable = (finding.get("reason") == "awaiting_interpretation"
                     and finding["status"] != TRIGGER_NOT_APPLICABLE)
        if not verdict or not judgeable:
            findings.append(finding)
            continue

        if verdict.get("status") == TRIGGER_UNSUPPORTED:
            reason = verdict.get("reason") or "evidence_failed_verification"
            finding.update({
                "status": TRIGGER_UNSUPPORTED, "rating": None, "reason": reason,
                "note": ("the model rated this but cited nothing to support it"
                         if reason == "no_evidence_cited" else
                         "the model rated this but the evidence it cited is not "
                         "in the transcript or not in this creative")})
            findings.append(finding)
            continue

        rating = (verdict.get("rating") or "").lower()
        finding.update({
            "status": RATING_STATUS.get(rating, TRIGGER_ABSENT),
            "rating": rating or None,
            "reason": None,
            "evidence": verdict.get("evidence") or [],
            "note": (verdict.get("note") or finding.get("note") or "")[:300],
            "judged_by": "llm",
        })
        findings.append(finding)

    return {**block, "triggers": findings, "coverage": coverage(findings),
            "unsupported": sum(1 for f in findings
                               if f["status"] == TRIGGER_UNSUPPORTED)}


def coverage(findings: list[dict]) -> dict:
    counts = {"present": 0, "weak": 0, "absent": 0, "not_applicable": 0}
    for finding in findings:
        if finding["status"] in counts:
            counts[finding["status"]] += 1
    counts["measured"] = sum(1 for f in findings if f["detection"] == "measured")
    return counts
