"""
Transcript normalisation - one internal shape, whatever the source.

WHAT THIS MODULE OWNS
    Turning a Deepgram response, a caller-supplied structured transcript, or a
    block of pasted text into the same NormalizedTranscript: indexed speaker
    turns with stable speaker ids, timestamps where they exist, and an honest
    account of what is missing.

RULES
    * `speaker_id` is the primary identity. Diarization ids are preserved, never
      renamed. Roles are decided later, in speakers.py, and only on evidence.
    * `index` is the anchor every piece of evidence cites. It is assigned here
      and never changes afterwards, which is what lets evidence.py verify a
      quote against the exact turn the model claims it came from.
    * Nothing is inferred that was not measured. A pasted text transcript has no
      timestamps and no diarization, and says so, rather than being dressed up
      with fabricated ones.
    * Two or more speakers is the normal case, not the special one. Nothing here
      assumes a call has exactly a rep and a customer.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from . import ROLE_BASIS_SUPPLIED
from .models import (
    AudioRef,
    NormalizedTranscript,
    SuppliedTranscript,
    TranscriptQuality,
    TranscriptSegment,
    TranscriptSpeaker,
)

# Sources, reported on the transcript so a reader knows what the analysis had.
SOURCE_DEEPGRAM = "deepgram"
SOURCE_SUPPLIED_STRUCTURED = "supplied_structured"
SOURCE_SUPPLIED_TEXT = "supplied_text"

# Input strategies chosen by select_input_strategy().
STRATEGY_TRANSCRIBE_AUDIO = "transcribe_audio"
STRATEGY_USE_SUPPLIED_STRUCTURED = "use_supplied_structured"
STRATEGY_USE_SUPPLIED_TEXT = "use_supplied_text"
STRATEGY_NONE = "no_input"

# Used when pasted text carries no speaker attribution at all. Deliberately not
# "speaker_0": pretending we know one person said everything would be a lie the
# rest of the pipeline could not detect.
UNATTRIBUTED_SPEAKER_ID = "speaker_unattributed"

# Warning codes on transcript.quality.warnings. Stable - the UI may branch.
WARN_NO_SPEAKER_ATTRIBUTION = "no_speaker_attribution"
WARN_NO_TIMESTAMPS = "no_timestamps"
WARN_SINGLE_SPEAKER = "single_speaker_detected"
WARN_LOW_CONFIDENCE = "low_transcription_confidence"
WARN_EMPTY = "empty_transcript"

# Tunables. Env-driven so they can be adjusted without a code change, in the
# style of the existing GEMINI_MODEL / PP_ORDER_STATS_TTL_SECONDS settings.
MERGE_GAP_SECONDS = float(os.environ.get("SCA_TURN_MERGE_GAP_SECONDS", 1.0))
LOW_CONFIDENCE_AT = float(os.environ.get("SCA_LOW_CONFIDENCE_AT", 0.6))
STRUCTURED_MIN_RATIO = float(os.environ.get("SCA_STRUCTURED_MIN_RATIO", 0.8))

# "REP — RAJAN: text", "Meera Patel: text", "Speaker 1: text"
_LABELLED_LINE = re.compile(r"^\s*([^:\n]{1,60}?)\s*:\s*(\S.*)$")


# =========================================================================== #
# Input strategy - which of the three supported cases are we in?
# =========================================================================== #
def is_structured(supplied: Optional[SuppliedTranscript]) -> bool:
    """Does the caller's transcript already carry reliable speaker attribution?

    Reliable means most segments have both text and a speaker identifier. A
    caller labelling plain text as `format: "structured"` does not make it so -
    we check the payload, because trusting the label would silently disable
    diarization on a call that has audio.
    """
    if not supplied or not supplied.segments:
        return False
    segments = [s for s in supplied.segments if (s.text or "").strip()]
    if not segments:
        return False
    with_speaker = sum(
        1 for s in segments
        if (s.speaker_id is not None and str(s.speaker_id).strip())
        or (s.speaker_label or "").strip()
        or (s.role or "").strip()
    )
    return (with_speaker / len(segments)) >= STRUCTURED_MIN_RATIO


def has_audio(audio: Optional[AudioRef]) -> bool:
    return bool(audio and (audio.url or "").strip())


def has_text(supplied: Optional[SuppliedTranscript]) -> bool:
    if not supplied:
        return False
    if (supplied.text or "").strip():
        return True
    return any((s.text or "").strip() for s in supplied.segments)


def select_input_strategy(audio: Optional[AudioRef],
                          supplied: Optional[SuppliedTranscript]) -> tuple[str, str]:
    """Decide where the transcript comes from. Returns (strategy, why).

    The ordering is deliberate and is the cost/quality decision for this
    feature:

      1. A structured supplied transcript wins outright, even when audio is
         present. It already has what Deepgram would produce, so transcribing
         would be a pure duplicate cost.
      2. Otherwise audio wins. A pasted plain-text transcript has no speakers,
         no timestamps and no tone, so trusting it over available audio would
         degrade every piece of evidence in the report to save one API call.
      3. Plain text alone is analysed in degraded mode.
    """
    if is_structured(supplied):
        return (STRATEGY_USE_SUPPLIED_STRUCTURED,
                "A structured transcript with speaker attribution was supplied, so "
                "transcription was skipped.")
    if has_audio(audio):
        why = "Audio was supplied, so it was transcribed and diarized."
        if has_text(supplied):
            why = ("Audio and a plain-text transcript were both supplied. The audio was "
                   "transcribed because the supplied text carries no speaker or timing "
                   "information; the supplied text is kept only as a fallback.")
        return STRATEGY_TRANSCRIBE_AUDIO, why
    if has_text(supplied):
        return (STRATEGY_USE_SUPPLIED_TEXT,
                "Only a plain-text transcript was supplied, so the analysis runs without "
                "speaker attribution, timings or tone.")
    return STRATEGY_NONE, "Neither a recording nor a transcript was supplied."


# =========================================================================== #
# Deepgram -> internal
# =========================================================================== #
def _speaker_id(raw: Any) -> str:
    """Deepgram reports speakers as integers. Preserve them as speaker_<n>."""
    if raw is None:
        return UNATTRIBUTED_SPEAKER_ID
    if isinstance(raw, bool):
        return UNATTRIBUTED_SPEAKER_ID
    if isinstance(raw, int):
        return f"speaker_{raw}"
    text = str(raw).strip()
    if not text:
        return UNATTRIBUTED_SPEAKER_ID
    if text.isdigit():
        return f"speaker_{int(text)}"
    return text if text.startswith("speaker_") else f"speaker_{text}"


def _unique_speaker_id(label: str, taken: dict[str, str]) -> str:
    """Map a caller's speaker label to an internal id, preserving an id the
    caller already expressed in our own form, without letting two different
    labels collide onto one id."""
    if label.lower().startswith("speaker") or label.isdigit():
        candidate = _speaker_id(label)
    else:
        candidate = f"speaker_{len(taken)}"
    used = set(taken.values())
    if candidate not in used:
        return candidate
    suffix = len(taken)
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    return f"{candidate}_{suffix}"


def _first_alternative(response: dict) -> dict:
    channels = ((response.get("results") or {}).get("channels") or [])
    if not channels:
        return {}
    alternatives = channels[0].get("alternatives") or []
    return alternatives[0] if alternatives else {}


def _raw_turns_from_deepgram(response: dict) -> list[dict]:
    """Pull speaker turns out of whichever shape Deepgram returned.

    Preference order: utterances (what `utterances=true` gives us, already one
    entry per speaker turn), then diarized paragraphs, then words grouped by
    speaker, then the flat transcript as a single unattributed turn.
    """
    results = response.get("results") or {}

    utterances = results.get("utterances") or []
    if utterances:
        return [{"speaker": u.get("speaker"), "start": u.get("start"), "end": u.get("end"),
                 "text": (u.get("transcript") or "").strip(),
                 "confidence": u.get("confidence")}
                for u in utterances if (u.get("transcript") or "").strip()]

    alt = _first_alternative(response)

    paragraphs = ((alt.get("paragraphs") or {}).get("paragraphs") or [])
    if paragraphs:
        turns = []
        for para in paragraphs:
            sentences = para.get("sentences") or []
            text = " ".join((s.get("text") or "").strip() for s in sentences).strip()
            if not text:
                continue
            starts = [s.get("start") for s in sentences if s.get("start") is not None]
            ends = [s.get("end") for s in sentences if s.get("end") is not None]
            turns.append({"speaker": para.get("speaker"),
                          "start": min(starts) if starts else para.get("start"),
                          "end": max(ends) if ends else para.get("end"),
                          "text": text, "confidence": None})
        if turns:
            return turns

    words = alt.get("words") or []
    if words:
        turns: list[dict] = []
        current: Optional[dict] = None
        for word in words:
            token = word.get("punctuated_word") or word.get("word") or ""
            if not token:
                continue
            speaker = word.get("speaker")
            if current is None or current["speaker"] != speaker:
                if current:
                    turns.append(current)
                current = {"speaker": speaker, "start": word.get("start"),
                           "end": word.get("end"), "text": token,
                           "confidences": [word.get("confidence")]}
            else:
                current["text"] += f" {token}"
                current["end"] = word.get("end")
                current["confidences"].append(word.get("confidence"))
        if current:
            turns.append(current)
        for turn in turns:
            scores = [c for c in turn.pop("confidences", []) if isinstance(c, (int, float))]
            turn["confidence"] = round(sum(scores) / len(scores), 4) if scores else None
        return [t for t in turns if t["text"].strip()]

    flat = (alt.get("transcript") or "").strip()
    if flat:
        return [{"speaker": None, "start": None, "end": None, "text": flat,
                 "confidence": alt.get("confidence")}]
    return []


def _detected_language(response: dict) -> Optional[str]:
    """Deepgram reports detected language in more than one place depending on
    the model and options. Read defensively rather than assuming one shape."""
    results = response.get("results") or {}
    channels = results.get("channels") or []
    if channels:
        for key in ("detected_language", "language"):
            value = channels[0].get(key)
            if value:
                return str(value)
    alt = _first_alternative(response)
    for key in ("detected_language", "language"):
        if alt.get(key):
            return str(alt[key])
    metadata = response.get("metadata") or {}
    for key in ("detected_language", "language"):
        if metadata.get(key):
            return str(metadata[key])
    return None


def from_deepgram(response: dict, *, language_hint: Optional[str] = None,
                  merge_gap_seconds: Optional[float] = None) -> NormalizedTranscript:
    """Normalise a Deepgram pre-recorded response."""
    turns = _raw_turns_from_deepgram(response)
    segments = _build_segments(
        [{"speaker_id": _speaker_id(t.get("speaker")), "start": t.get("start"),
          "end": t.get("end"), "text": t.get("text") or "",
          "confidence": t.get("confidence")} for t in turns],
        merge_gap_seconds=merge_gap_seconds,
    )

    metadata = response.get("metadata") or {}
    duration = metadata.get("duration")
    detected = _detected_language(response)

    transcript = _assemble(
        segments,
        source=SOURCE_DEEPGRAM,
        language=language_hint or detected,
        language_detected=detected,
        duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
    )
    transcript.diarization_available = any(
        s.speaker_id != UNATTRIBUTED_SPEAKER_ID for s in transcript.segments)
    if not transcript.diarization_available:
        transcript.quality.warnings.append(WARN_NO_SPEAKER_ATTRIBUTION)
    return transcript


# =========================================================================== #
# Supplied -> internal
# =========================================================================== #
def from_supplied_structured(supplied: SuppliedTranscript,
                             merge_gap_seconds: Optional[float] = None) -> NormalizedTranscript:
    """Normalise a caller-supplied transcript that already has speakers.

    Supplied speaker labels are mapped to speaker_0..n in order of first
    appearance, and any role/name the caller attached is preserved so
    speakers.py can record it as caller-supplied rather than re-deriving it.
    """
    label_to_id: dict[str, str] = {}
    supplied_roles: dict[str, dict] = {}
    raw: list[dict] = []

    for seg in supplied.segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        label = (str(seg.speaker_id).strip() if seg.speaker_id is not None else "") \
            or (seg.speaker_label or "").strip() \
            or (seg.role or "").strip()
        if label:
            if label not in label_to_id:
                label_to_id[label] = _unique_speaker_id(label, label_to_id)
            speaker_id = label_to_id[label]
        else:
            speaker_id = UNATTRIBUTED_SPEAKER_ID

        if speaker_id not in supplied_roles and ((seg.role or "").strip() or (seg.name or "").strip()):
            supplied_roles[speaker_id] = {"role": (seg.role or "").strip() or None,
                                          "name": (seg.name or "").strip() or None,
                                          "label": label or None}
        raw.append({"speaker_id": speaker_id, "start": seg.start, "end": seg.end,
                    "text": text, "confidence": seg.confidence})

    segments = _build_segments(raw, merge_gap_seconds=merge_gap_seconds)
    transcript = _assemble(segments, source=SOURCE_SUPPLIED_STRUCTURED,
                           language=supplied.language, language_detected=None,
                           duration_seconds=None)
    transcript.diarization_available = any(
        s.speaker_id != UNATTRIBUTED_SPEAKER_ID for s in transcript.segments)

    # Carry the caller's own role/name annotations through for speakers.py, and
    # STAMP THE BASIS. Without that marker a role on a transcript is ambiguous:
    # speakers.py cannot tell a role the caller asserted from one an earlier run
    # derived, and a reused transcript would have its derived roles frozen in as
    # though the caller had supplied them.
    for speaker in transcript.speakers:
        hint = supplied_roles.get(speaker.speaker_id)
        if hint:
            if hint["role"]:
                speaker.role = hint["role"]
                speaker.role_basis = ROLE_BASIS_SUPPLIED
                speaker.role_confidence = "high"
            speaker.name = hint["name"]
    return transcript


def from_supplied_text(supplied: SuppliedTranscript) -> NormalizedTranscript:
    """Normalise a pasted plain-text transcript.

    Two shapes are handled: lines labelled "Someone: text", which give us
    speaker attribution but no timings, and unlabelled prose, which gives us
    neither. Both are segmented so evidence can still anchor to a specific piece
    of text - that is the whole point of the index.
    """
    text = (supplied.text or "").strip()
    if not text:
        joined = "\n".join((s.text or "").strip() for s in supplied.segments)
        text = joined.strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    matches = [(_LABELLED_LINE.match(ln), ln) for ln in lines]
    labelled = [m for m, _ in matches if m]

    raw: list[dict] = []
    if lines and (len(labelled) / len(lines)) >= STRUCTURED_MIN_RATIO:
        label_to_id: dict[str, str] = {}
        last_speaker_id: Optional[str] = None
        for match, line in matches:
            if match:
                label = match.group(1).strip()
                body = match.group(2).strip()
                if not body:
                    continue
                if label not in label_to_id:
                    label_to_id[label] = _unique_speaker_id(label, label_to_id)
                speaker_id = label_to_id[label]
                last_speaker_id = speaker_id
            else:
                # An unlabelled line continues the previous speaker's turn.
                body = line
                if not body:
                    continue
                speaker_id = last_speaker_id or UNATTRIBUTED_SPEAKER_ID
            raw.append({"speaker_id": speaker_id, "start": None, "end": None,
                        "text": body, "confidence": None})
    else:
        for chunk in _split_prose(text):
            raw.append({"speaker_id": UNATTRIBUTED_SPEAKER_ID, "start": None, "end": None,
                        "text": chunk, "confidence": None})

    # Merging by gap is meaningless without timestamps, so it is disabled here.
    segments = _build_segments(raw, merge_gap_seconds=None)
    transcript = _assemble(segments, source=SOURCE_SUPPLIED_TEXT,
                           language=supplied.language, language_detected=None,
                           duration_seconds=None)
    transcript.diarization_available = any(
        s.speaker_id != UNATTRIBUTED_SPEAKER_ID for s in transcript.segments)
    if not transcript.diarization_available:
        transcript.quality.warnings.append(WARN_NO_SPEAKER_ATTRIBUTION)
    return transcript


def _split_prose(text: str) -> list[str]:
    """Break unlabelled prose into anchorable chunks: paragraphs first, then
    lines, and only then the whole thing as one segment."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) > 1:
        return parts
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines if len(lines) > 1 else ([text] if text else [])


# =========================================================================== #
# Shared assembly
# =========================================================================== #
def _build_segments(raw: list[dict],
                    merge_gap_seconds: Optional[float] = None) -> list[TranscriptSegment]:
    """Merge consecutive same-speaker turns, then assign the stable indices.

    Merging exists because a diarizer emits breath-groups, not conversational
    turns; the model reads turns far better. Merging never crosses a speaker
    change, so it cannot move words from one person to another.
    """
    gap = MERGE_GAP_SECONDS if merge_gap_seconds is None else merge_gap_seconds
    merged: list[dict] = []

    for turn in raw:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        previous = merged[-1] if merged else None
        can_merge = (
            previous is not None
            and gap is not None
            and previous["speaker_id"] == turn["speaker_id"]
            and previous.get("end") is not None
            and turn.get("start") is not None
            and (float(turn["start"]) - float(previous["end"])) <= gap
        )
        if can_merge:
            previous["text"] = f"{previous['text']} {text}".strip()
            previous["end"] = turn.get("end", previous["end"])
            previous["confidences"].append(turn.get("confidence"))
        else:
            merged.append({"speaker_id": turn["speaker_id"], "start": turn.get("start"),
                           "end": turn.get("end"), "text": text,
                           "confidences": [turn.get("confidence")]})

    segments: list[TranscriptSegment] = []
    for index, item in enumerate(merged):
        scores = [c for c in item["confidences"] if isinstance(c, (int, float))]
        segments.append(TranscriptSegment(
            index=index,
            speaker_id=item["speaker_id"],
            start=float(item["start"]) if item.get("start") is not None else None,
            end=float(item["end"]) if item.get("end") is not None else None,
            text=item["text"],
            confidence=round(sum(scores) / len(scores), 4) if scores else None,
        ))
    return segments


def _assemble(segments: list[TranscriptSegment], *, source: str,
              language: Optional[str], language_detected: Optional[str],
              duration_seconds: Optional[float]) -> NormalizedTranscript:
    """Speaker statistics, quality metrics and the assembled transcript."""
    stats: dict[str, dict] = {}
    for seg in segments:
        entry = stats.setdefault(seg.speaker_id, {"talk": 0.0, "turns": 0, "words": 0})
        entry["turns"] += 1
        entry["words"] += len(seg.text.split())
        if seg.start is not None and seg.end is not None and seg.end >= seg.start:
            entry["talk"] += float(seg.end) - float(seg.start)

    speakers = [
        TranscriptSpeaker(
            speaker_id=speaker_id,
            talk_time_seconds=round(entry["talk"], 3),
            turn_count=entry["turns"],
            word_count=entry["words"],
        )
        # Sort by first appearance so speaker_0 is the first voice on the call.
        for speaker_id, entry in sorted(
            stats.items(),
            key=lambda kv: next(s.index for s in segments if s.speaker_id == kv[0]))
    ]

    confidences = [s.confidence for s in segments if s.confidence is not None]
    mean_confidence = round(sum(confidences) / len(confidences), 4) if confidences else None
    low_ratio = (round(sum(1 for c in confidences if c < LOW_CONFIDENCE_AT) / len(confidences), 4)
                 if confidences else None)

    warnings: list[str] = []
    if not segments:
        warnings.append(WARN_EMPTY)
    if not any(s.start is not None for s in segments):
        warnings.append(WARN_NO_TIMESTAMPS)
    real_speakers = [s for s in speakers if s.speaker_id != UNATTRIBUTED_SPEAKER_ID]
    if len(real_speakers) == 1:
        warnings.append(WARN_SINGLE_SPEAKER)
    if mean_confidence is not None and mean_confidence < LOW_CONFIDENCE_AT:
        warnings.append(WARN_LOW_CONFIDENCE)

    if duration_seconds is None:
        ends = [s.end for s in segments if s.end is not None]
        duration_seconds = round(max(ends), 3) if ends else None

    return NormalizedTranscript(
        source=source,
        language=language,
        language_detected=language_detected,
        multilingual=bool(language_detected and str(language_detected).lower() in ("multi", "multilingual")),
        diarization_available=len(real_speakers) > 0,
        timestamps_available=any(s.start is not None for s in segments),
        speaker_count=len(real_speakers),
        segment_count=len(segments),
        duration_seconds=duration_seconds,
        word_count=sum(len(s.text.split()) for s in segments),
        speakers=speakers,
        segments=segments,
        quality=TranscriptQuality(
            mean_confidence=mean_confidence,
            low_confidence_ratio=low_ratio,
            usable=bool(segments),
            warnings=warnings,
        ),
    )


def clear_role_annotations(transcript: NormalizedTranscript) -> NormalizedTranscript:
    """Strip every role annotation, keeping the speaker ids, text and timings.

    Used when a transcript is reused from an earlier analysis. Transcription is
    the expensive half and is worth reusing; ROLE RESOLUTION IS NOT, and must not
    be. The request context that drives it - the CRM rep and customer names -
    can differ between runs, and reusing a previous run's answer would freeze in
    a conclusion drawn from names that have since been corrected.
    """
    for speaker in transcript.speakers:
        speaker.role = "unknown"
        speaker.name = None
        speaker.role_basis = "unresolved"
        speaker.role_confidence = "none"
    return transcript


def render_for_prompt(transcript: NormalizedTranscript, max_chars: Optional[int] = None) -> str:
    """The transcript as the model sees it: one line per turn, prefixed with the
    index it must cite as evidence and the speaker id it must attribute to.

    Roles are shown alongside the speaker id, never instead of it, so a wrong
    role annotation cannot corrupt an evidence anchor.
    """
    lines: list[str] = []
    roles = {s.speaker_id: s.role for s in transcript.speakers}
    for seg in transcript.segments:
        role = roles.get(seg.speaker_id, "unknown")
        stamp = f" [{seg.start:.1f}-{seg.end:.1f}s]" if seg.start is not None and seg.end is not None else ""
        lines.append(f"[{seg.index}] {seg.speaker_id} ({role}){stamp}: {seg.text}")
    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text
