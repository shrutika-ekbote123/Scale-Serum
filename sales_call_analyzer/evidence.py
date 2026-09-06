"""
Evidence verification - the gate that makes fabricated findings impossible to
persist.

HOW IT WORKS
    The model cites a segment_index and copies a quote. This module resolves
    that index against the real transcript and checks the quote is genuinely in
    that turn, spoken by the speaker claimed. Timestamps are copied FROM the
    segment, never taken from the model, so a timestamp cannot be invented.

WHAT SURVIVES AND WHAT DOES NOT
    A claim that something HAPPENED needs verified evidence: a strength, an
    objection, a stated need, a buying signal, a persuasion technique. Without
    it the claim is dropped.

    A claim about ABSENCE cannot be quoted - there is no line where the
    representative failed to ask about budget. So the lowest rating level,
    weaknesses and recommendations are allowed to stand without an anchor, and
    are marked `evidence_backed: false` so the UI can show the difference.
    Nothing unsupported passes silently; it either dies or is labelled.

WHY THIS IS CODE AND NOT PROMPT WORDING
    A prompt asking for honest citations is a request. This is a check. It also
    limits what prompt injection can achieve: text in the transcript cannot
    manufacture an anchor into a segment that does not contain it.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .models import Evidence, NormalizedTranscript

# Normalisation for quote matching: casefold, unify quote characters, drop
# punctuation, collapse whitespace. Tolerates the model tidying a trailing
# comma; does not tolerate it inventing words.
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-"})
# Apostrophes are DELETED rather than replaced with a space, so a model writing
# "dont" still matches a transcript's "don't". Every other punctuation mark
# becomes a space, which is what makes "Kumar." match "Kumar".
_APOSTROPHE = re.compile(r"['`]")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")

# Below this, a "quote" is too short to be evidence of anything.
MIN_QUOTE_CHARS = 3

# Lists whose items assert that something was said. No verified anchor, no item.
POSITIVE_CLAIM_LISTS = ("strengths", "customer_needs", "objections",
                        "buying_signals", "customer_signals", "rep_techniques")
# Lists whose items may legitimately describe an absence.
ABSENCE_ALLOWED_LISTS = ("weaknesses", "recommendations")


def normalise(text: str) -> str:
    folded = _APOSTROPHE.sub("", (text or "").translate(_QUOTES).casefold())
    return _SPACES.sub(" ", _NON_WORD.sub(" ", folded)).strip()


class VerificationStats:
    def __init__(self) -> None:
        self.anchors_total = 0
        self.anchors_kept = 0
        self.items_dropped = 0
        self.criteria_unsupported = 0

    @property
    def anchors_dropped(self) -> int:
        return self.anchors_total - self.anchors_kept

    def as_dict(self) -> dict:
        return {"evidence_anchors_total": self.anchors_total,
                "evidence_anchors_kept": self.anchors_kept,
                "evidence_anchors_dropped": self.anchors_dropped,
                "items_dropped": self.items_dropped,
                "criteria_unsupported": self.criteria_unsupported}


def verify_anchor(anchor: dict, segments_by_index: dict[int, Any],
                  stats: Optional[VerificationStats] = None) -> Optional[Evidence]:
    """Resolve one anchor, or return None if it does not hold up."""
    if stats:
        stats.anchors_total += 1

    segment = segments_by_index.get(anchor.get("segment_index"))
    if segment is None:
        return None                                  # cites a turn that does not exist

    claimed_speaker = (anchor.get("speaker_id") or "").strip()
    if claimed_speaker and claimed_speaker != segment.speaker_id:
        return None                                  # attributed to the wrong person

    quote = (anchor.get("quote") or "").strip()
    if len(quote) < MIN_QUOTE_CHARS:
        return None

    needle, haystack = normalise(quote), normalise(segment.text)
    if not needle or needle not in haystack:
        return None                                  # not what was said in that turn

    if stats:
        stats.anchors_kept += 1
    return Evidence(
        segment_index=segment.index,
        speaker_id=segment.speaker_id,               # from the transcript, not the model
        start=segment.start,
        end=segment.end,
        quote=quote,
        verified=True,
    )


def verify_many(anchors: Any, segments_by_index: dict[int, Any],
                stats: VerificationStats) -> list[Evidence]:
    out: list[Evidence] = []
    seen: set[tuple[int, str]] = set()
    for anchor in anchors or []:
        if not isinstance(anchor, dict):
            continue
        verified = verify_anchor(anchor, segments_by_index, stats)
        if verified is None:
            continue
        key = (verified.segment_index, normalise(verified.quote))
        if key in seen:
            continue
        seen.add(key)
        out.append(verified)
    return out


def verify_analysis(analysis: dict, transcript: NormalizedTranscript,
                    cfg: dict) -> tuple[dict, dict]:
    """Verify every anchor in the analysis. Returns (analysis, stats).

    The input dict is not mutated; a verified copy is returned so the raw model
    output stays available for debugging and for rescoring later.
    """
    segments_by_index = {s.index: s for s in transcript.segments}
    stats = VerificationStats()
    lowest_rating = cfg["rating_levels"][0]          # the "did not happen" level
    out: dict = dict(analysis)

    # ---- criteria ---------------------------------------------------------
    criteria: list[dict] = []
    for item in analysis.get("criteria") or []:
        verified = verify_many(item.get("evidence"), segments_by_index, stats)
        entry = dict(item)
        entry["evidence"] = [e.model_dump() for e in verified]
        entry["evidence_backed"] = bool(verified)

        rating = item.get("rating")
        applicable = bool(item.get("applicable", True))
        # An "absent" rating is a claim about what did not happen, and cannot be
        # quoted. Requiring an anchor for it would silently delete every
        # negative finding in the report.
        absence_claim = (not applicable) or rating == lowest_rating or rating is None

        if applicable and rating is not None and not verified and not absence_claim:
            entry["status"] = "unsupported"
            stats.criteria_unsupported += 1
        criteria.append(entry)
    out["criteria"] = criteria

    # ---- lists that assert something happened ------------------------------
    for key in POSITIVE_CLAIM_LISTS:
        kept: list[dict] = []
        for item in analysis.get(key) or []:
            verified = verify_many(item.get("evidence"), segments_by_index, stats)
            if not verified:
                stats.items_dropped += 1
                continue
            entry = dict(item)
            entry["evidence"] = [e.model_dump() for e in verified]
            entry["evidence_backed"] = True
            kept.append(entry)
        out[key] = kept

    # ---- lists that may describe an absence --------------------------------
    for key in ABSENCE_ALLOWED_LISTS:
        kept = []
        for item in analysis.get(key) or []:
            verified = verify_many(item.get("evidence"), segments_by_index, stats)
            entry = dict(item)
            entry["evidence"] = [e.model_dump() for e in verified]
            entry["evidence_backed"] = bool(verified)
            kept.append(entry)
        out[key] = kept

    # ---- highlights: positive ones must be evidenced -----------------------
    highlights: list[dict] = []
    for item in analysis.get("highlights") or []:
        verified = verify_many(item.get("evidence"), segments_by_index, stats)
        if item.get("type") == "positive" and not verified:
            stats.items_dropped += 1
            continue
        entry = dict(item)
        entry["evidence"] = [e.model_dump() for e in verified]
        entry["evidence_backed"] = bool(verified)
        highlights.append(entry)
    out["highlights"] = highlights

    # ---- single objects ----------------------------------------------------
    for key in ("pitch_structure", "context_adaptation"):
        block = analysis.get(key)
        if not isinstance(block, dict):
            continue
        verified = verify_many(block.get("evidence"), segments_by_index, stats)
        entry = dict(block)
        entry["evidence"] = [e.model_dump() for e in verified]
        entry["evidence_backed"] = bool(verified)
        out[key] = entry

    return out, stats.as_dict()
