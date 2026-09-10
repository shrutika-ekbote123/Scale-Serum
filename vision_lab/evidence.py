"""
Step 19 - verify every claim against the measurement record.

WHY THIS FILE DECIDES WHETHER THE REPORT CAN BE TRUSTED
    analyzer.py already refuses a response that scores, or that invents a
    defect. This is the finer check: of the claims that survived, which are
    actually supported by something we measured?

    A quote is only a quote if it appears in the transcript. A timestamp is only
    valid if it falls inside the creative. A number in the prose is only valid
    if it matches a number we measured. Anything else is dropped and reported as
    unsupported - not silently corrected, because a caller needs to know the
    model produced it.

UNSUPPORTED IS NOT FALSE
    A dropped claim may well be true. We simply cannot stand behind it, and a
    report that presents unverifiable prose beside measured findings teaches the
    reader to trust neither. Same rule as sales_call_analyzer/evidence.py.

WHAT IS DELIBERATELY NOT CHECKED
    Judgement. "The leap from outcome to enrolment is where scepticism enters"
    is not verifiable and is not meant to be - it is the interpretation we asked
    for. Only factual claims are checked: quotes, times and numbers.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("vision_lab.evidence")

# Numbers in prose that are worth checking. Bare small integers are skipped:
# "three questions", "two lines" are ordinary English, not measurements, and
# flagging them would bury the ones that matter.
NUMBER_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(s\b|sec|second|words?|%)", re.I)

# How far a quoted timestamp may sit from the defect it belongs to. A model
# rounding 34.1 to 34 is not an error; a model citing 12s for a 34s defect is.
TIME_TOLERANCE_SECONDS = 2.0

UNSUPPORTED_QUOTE = "quote_not_in_transcript"
UNSUPPORTED_TIME = "timestamp_outside_creative"
UNSUPPORTED_NUMBER = "number_not_measured"
UNSUPPORTED_DEFECT = "no_matching_defect"


def _normalise(text: str) -> str:
    """Lower-cased words and spaces only, with runs of space collapsed.

    THE COLLAPSE IS NOT COSMETIC. Punctuation becomes a space, so "है, सर"
    normalises to two spaces while the same words compared word-by-word give
    one. On a Hinglish ad that mismatch rejected a quote the presenter really
    had said, word for word, and reported the trigger as unsupported.

    The Devanagari range is here for the same reason: half of ScaleSerum's
    creatives are Hinglish, and stripping the script would leave every such
    quote as an empty needle that matches anything.
    """
    return re.sub(r"\s+", " ",
                  re.sub(r"[^a-z0-9ऀ-ॿ ]+", " ", (text or "").lower())).strip()


def _transcript_text(transcript: Optional[dict]) -> str:
    segments = (transcript or {}).get("segments") or []
    return _normalise(" ".join(s.get("text", "") for s in segments))


def line_at(transcript: Optional[dict], when) -> Optional[dict]:
    """The transcript line the cited timestamp points at, if any.

    A CITATION IS A POINTER, NOT A TRANSCRIPTION.
        On a Hinglish ad the model reproduced a line as "दहले डेरे लषआ इसकव" -
        the same sentence it had copied correctly on the previous run. Devanagari
        does not survive being retyped by a model reliably, and a verifier that
        only accepts a perfect copy throws away real citations of real lines.

        So the timestamp is what we check. When it lands on a line, that line's
        OWN text becomes the quote - which is stricter than trusting the model's
        version, not looser: the published quote is now always ours.
    """
    try:
        when = float(when)
    except (TypeError, ValueError):
        return None

    best, distance = None, None
    for segment in (transcript or {}).get("segments") or []:
        start = segment.get("t")
        if start is None:
            continue
        end = segment.get("end") if segment.get("end") is not None else start + 2.0
        if start - 0.5 <= when <= end + 0.5:
            return segment
        gap = min(abs(when - start), abs(when - end))
        if distance is None or gap < distance:
            best, distance = segment, gap
    return best if distance is not None and distance <= TIME_TOLERANCE_SECONDS else None


def quote_is_real(quote: str, haystack: str) -> bool:
    """Is this quote actually in the transcript?

    Matched on normalised text so punctuation and casing do not cause a false
    rejection, and on a prefix so a model trimming a long line is not punished
    for it. A quote that shares no run of words with the transcript is invented.
    """
    needle = _normalise(quote)
    if len(needle) < 8:
        return True          # too short to verify either way; not worth a flag
    if needle in haystack:
        return True
    # Allow a trimmed or slightly reworded tail: the first eight words must match.
    words = needle.split()
    return " ".join(words[:8]) in haystack if len(words) >= 8 else False


def collect_measured_numbers(*sources) -> set:
    """Every number the report is entitled to state.

    WHAT WE SHOWED THE MODEL IS WHAT IT MAY QUOTE BACK.
        This walks the measurement structures whole rather than naming the keys
        it expects. Listing keys by hand meant the set drifted from the context
        the model was actually sent: a timeline point at 34.833 s - a timestamp
        we handed it - came back in the prose and was rejected as invented,
        which took a sound recommendation with it.

        Any number reachable in what was sent is quotable. Anything else the
        model produced, it produced on its own.
    """
    numbers: set = set()

    def walk(node, depth=0):
        if depth > 8:
            return
        if isinstance(node, bool):
            return                      # True is not the number 1
        if isinstance(node, (int, float)):
            numbers.add(round(float(node), 1))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value, depth + 1)

    for source in sources:
        walk(source)
    return numbers


def verify(interpretation: dict, *, defects: list[dict], summary: dict,
           timeline: dict, transcript: Optional[dict],
           duration: Optional[float] = None,
           context: Optional[dict] = None) -> dict:
    """Check every factual claim. Returns the interpretation plus an audit.

    `context` is what analyzer.py actually sent the model. When it is supplied,
    every number in it is quotable - which is the honest rule, since the model
    read those numbers from us.
    """
    haystack = _transcript_text(transcript)
    measured = collect_measured_numbers(defects, summary, timeline, context)
    by_id = {d["defect_id"]: d for d in defects or []}
    duration = duration or summary.get("duration_seconds") or 0

    audit = {"checked": 0, "dropped": 0, "unsupported": []}

    def flag(where: str, reason: str, detail: str):
        audit["dropped"] += 1
        audit["unsupported"].append({"where": where, "reason": reason,
                                     "detail": detail[:160]})

    # ---- recommendations -------------------------------------------------
    kept = []
    for item in interpretation.get("recommendations") or []:
        audit["checked"] += 1
        defect = by_id.get(item.get("defect_id"))
        if not defect:
            flag(f"recommendation[{item.get('defect_id')}]", UNSUPPORTED_DEFECT,
                 item.get("title", ""))
            continue

        # ONLY THE CLAIM IS CHECKED, NOT THE PRESCRIPTION.
        #
        # `title` and `why` say what the creative IS - a number there is a
        # measurement, and an unmeasured one reads exactly like a real finding.
        # `fix` says what to CHANGE - "hold the logo for 3 seconds", "cut this
        # to 16 words" - and those numbers are targets the editor has not hit
        # yet, so of course we never measured them.
        #
        # Checking `fix` too, as this first did, dropped every recommendation on
        # a real ad: both were sound arguments about real defects, rejected for
        # the crime of being specific about the remedy.
        claim = " ".join(str(item.get(f) or "") for f in ("title", "why"))
        bad_number = _first_unmeasured_number(claim, measured)
        if bad_number is not None:
            flag(f"recommendation[{item['defect_id']}]", UNSUPPORTED_NUMBER,
                 f"{bad_number} is stated as a finding but was never measured")
            continue

        item["anchor"] = {
            "defect_id": defect["defect_id"],
            "t_start": defect.get("t_start"),
            "t_end": defect.get("t_end"),
            "measured": defect.get("measured", {}),
        }
        item["scores_impacted"] = defect.get("scores_impacted", [])
        item["severity"] = defect.get("severity")
        item["verified"] = True
        kept.append(item)

    for rank, item in enumerate(kept, start=1):
        item["rank"] = rank
    interpretation["recommendations"] = kept

    # ---- trigger evidence -------------------------------------------------
    for trigger in interpretation.get("triggers") or []:
        verified = []
        cited_lines: set = set()
        for piece in trigger.get("evidence") or []:
            audit["checked"] += 1
            quote = piece.get("quote") or ""
            when = piece.get("t")

            if when is not None and duration and not (-1 <= float(when) <= duration + 1):
                flag(f"trigger[{trigger.get('id')}]", UNSUPPORTED_TIME, str(when))
                continue
            if quote and haystack and not quote_is_real(quote, haystack):
                # The wording did not survive, but the citation might still
                # point at a real line - in which case that line's own text is
                # the quote, and we say we replaced it.
                line = line_at(transcript, when)
                # A REPAIR MUST NOT MANUFACTURE EVIDENCE. A fabricated quote
                # with a plausible timestamp would otherwise come back as a
                # second copy of a line already cited - the same line counted
                # twice, which reads as more support than the model gave.
                if not line or line.get("t") in cited_lines:
                    flag(f"trigger[{trigger.get('id')}]", UNSUPPORTED_QUOTE, quote)
                    continue
                piece = {**piece, "quote": line.get("text"),
                         "t": line.get("t"), "quote_source": "transcript_line",
                         "model_quote_rejected": True}

            resolved = line_at(transcript, piece.get("t"))
            if resolved is not None:
                cited_lines.add(resolved.get("t"))
            verified.append(piece)

        cited = len(trigger.get("evidence") or [])
        trigger["evidence"] = verified
        # A judged trigger whose evidence all failed is reported as unsupported,
        # NOT as absent. We do not know how the creative did; that is different
        # from knowing it did badly.
        #
        # ONLY A POSITIVE RATING NEEDS EVIDENCE. There is nothing to quote for a
        # trigger the creative did not use - demanding it marked two honest
        # "absent" verdicts on a real ad as unsupported, which reads as a
        # verification failure when it was the model answering correctly.
        claims_presence = (trigger.get("rating") or "").lower() in (
            "weak", "adequate", "strong")
        if claims_presence and not verified and trigger.get("evidence_required", True):
            trigger["status"] = "unsupported"
            # Two different failures, and the audit has to tell them apart: the
            # model cited nothing at all, or everything it cited was rejected.
            # An audit reading "dropped: 0" beside an unsupported trigger is how
            # this looked before the distinction was recorded.
            trigger["reason"] = ("evidence_failed_verification" if cited
                                 else "no_evidence_cited")
            if not cited:
                flag(f"trigger[{trigger.get('id')}]", "no_evidence_cited",
                     f"rated {trigger.get('rating')} with nothing cited")

    # ---- key message ------------------------------------------------------
    key = interpretation.get("key_message") or {}
    if key.get("quote") and haystack and not quote_is_real(key["quote"], haystack):
        line = line_at(transcript, key.get("t"))
        if line:
            key["quote"] = line.get("text")
            key["model_quote_rejected"] = True
            key["verified"] = True
        else:
            flag("key_message", UNSUPPORTED_QUOTE, key["quote"])
            key["quote"] = None
            key["verified"] = False
    elif key:
        key["verified"] = True

    interpretation["evidence_audit"] = audit
    if audit["dropped"]:
        logger.info("evidence verification dropped %d claim(s) of %d checked",
                    audit["dropped"], audit["checked"])
    return interpretation


def _first_unmeasured_number(prose: str, measured: set) -> Optional[str]:
    """The first quantity in the prose that we never measured, if any."""
    for match in NUMBER_PATTERN.finditer(prose or ""):
        try:
            value = round(float(match.group(1)), 1)
        except ValueError:
            continue
        # Allow a rounded restatement: 13.9 quoted as 14 is the same finding.
        if any(abs(value - m) <= 0.6 for m in measured):
            continue
        return match.group(0)
    return None
