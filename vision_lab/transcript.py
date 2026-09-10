"""
Step 16 - the transcript, and joining it to the attention timeline.

WHY DEEPGRAM AND NOT IN-BROWSER TRANSCRIPTION
    The Vision Lab prototype transcribed in the browser. Server-side Deepgram is
    more accurate, gives word-level timestamps we can join to the timeline, and
    keeps the API key off the client. It is also already configured and paid
    for by the Sales Call Analyzer, which is why the client now lives in a
    shared `transcription/` package rather than inside either feature.

BYTES, NOT A URL
    Sales calls hand Deepgram a signed URL and let it fetch. Vision Lab has
    already downloaded the creative and extracted a 16 kHz mono WAV with ffmpeg,
    so it posts those bytes directly. That avoids a second 150 MB fetch of the
    same video, and avoids depending on Deepgram being able to reach our bucket.

THE PER-LINE ATTENTION NUMBERS ARE A JOIN, NOT A SECOND MODEL
    Each transcript line takes the mean of the attention index over its own time
    span. The warning icon in the prototype's transcript panel is simply a line
    that overlaps a detected weak zone. Nothing here predicts anything.

NO AUDIO IS NOT A FAILURE
    A silent creative reports `no_audio_stream` and the report omits the
    transcript block. Half the ads a marketing team tests are silent cutdowns.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from . import (
    NO_AUDIO_STREAM,
    TRANSCRIPTION_NOT_CONFIGURED,
    TRANSCRIPTION_PROVIDER_ERROR,
    TRANSCRIPT_EMPTY,
)

logger = logging.getLogger("vision_lab.transcript")

# Deepgram bills by audio minute, so a long creative is cheap - but a 200 MB
# WAV is not something to post over a flaky link either.
MAX_AUDIO_BYTES = int(os.environ.get("VL_MAX_AUDIO_BYTES", 60 * 1024 * 1024))


def is_configured() -> bool:
    try:
        from transcription import deepgram_client as dg
    except ImportError:
        return False
    return dg.is_configured()


async def transcribe_file(path: str, *, language: Optional[str] = None) -> dict:
    """Post a local audio file to Deepgram. Used by scripts and by tests."""
    with open(path, "rb") as handle:
        return await transcribe_audio(handle.read(), language=language)


async def transcribe_audio(audio: bytes, *, language: Optional[str] = None) -> dict:
    """Post extracted WAV bytes to Deepgram and return the raw response.

    This is the VisionDeps callable. It takes BYTES because media.py already
    pulled the track out of the video before deleting the download - see the
    module docstring for why we do not hand Deepgram a URL and let it fetch.

    Raises DeepgramError, which the pipeline turns into a stated reason. The
    transcript is never load-bearing for the six scores, so a failure here
    degrades the report rather than failing the analysis.
    """
    from transcription import deepgram_client as dg

    size = len(audio or b"")
    if not size:
        # Same code the client already uses for an empty recording, so both
        # features report an empty track the same way.
        raise dg.DeepgramError(dg.AUDIO_UNREACHABLE,
                               "The extracted audio was empty.", retryable=False)
    if size > MAX_AUDIO_BYTES:
        raise dg.DeepgramError(
            dg.AUDIO_TOO_LARGE,
            f"The extracted audio is {size / 1048576:.0f} MB.", retryable=False)

    client = await dg.get_client()
    params = dg.build_params(language)
    headers = {"Authorization": f"Token {dg._api_key()}",
               "Content-Type": "audio/wav"}
    return await dg._post(client, params=params, headers=headers, content=audio)


# --------------------------------------------------------------------------- shape
def normalise(raw: dict) -> dict:
    """Deepgram's response to the shape the report publishes.

    Interpreting the provider payload happens HERE and nowhere else, so a change
    in Deepgram's response shape has exactly one place to be handled.
    """
    results = (raw or {}).get("results") or {}
    channels = results.get("channels") or []
    if not channels:
        return {"segments": [], "text": "", "language": None}

    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        return {"segments": [], "text": "", "language": None}

    best = alternatives[0]
    segments = []

    # Prefer Deepgram's own utterance/paragraph split - it breaks on natural
    # pauses, which is what a viewer hears as a line.
    for utterance in (results.get("utterances") or []):
        text = (utterance.get("transcript") or "").strip()
        if text:
            segments.append({"t": round(float(utterance.get("start") or 0), 3),
                             "end": round(float(utterance.get("end") or 0), 3),
                             "text": text,
                             "confidence": utterance.get("confidence")})

    if not segments:
        paragraphs = ((best.get("paragraphs") or {}).get("paragraphs") or [])
        for paragraph in paragraphs:
            for sentence in paragraph.get("sentences") or []:
                text = (sentence.get("text") or "").strip()
                if text:
                    segments.append({"t": round(float(sentence.get("start") or 0), 3),
                                     "end": round(float(sentence.get("end") or 0), 3),
                                     "text": text, "confidence": None})

    # Last resort: group words into ~8-second lines so the panel is readable.
    if not segments and best.get("words"):
        segments = _group_words(best["words"])

    return {
        "segments": segments,
        "text": (best.get("transcript") or "").strip(),
        "language": (channels[0].get("detected_language")
                     or results.get("language")),
    }


def _group_words(words: list[dict], seconds: float = 8.0) -> list[dict]:
    lines, current, start = [], [], None
    for word in words:
        if start is None:
            start = float(word.get("start") or 0)
        current.append(word.get("punctuated_word") or word.get("word") or "")
        end = float(word.get("end") or 0)
        if end - start >= seconds:
            lines.append({"t": round(start, 3), "end": round(end, 3),
                          "text": " ".join(current).strip(), "confidence": None})
            current, start = [], None
    if current and start is not None:
        lines.append({"t": round(start, 3),
                      "end": round(float(words[-1].get("end") or 0), 3),
                      "text": " ".join(current).strip(), "confidence": None})
    return lines


# --------------------------------------------------------------------------- join
def join_to_timeline(transcript: dict, timeline: Optional[dict]) -> dict:
    """Attach the mean attention over each line's own time span.

    This is arithmetic over two things we already have, not a second prediction.
    A line that overlaps a weak zone is flagged, which is the warning icon in
    the prototype's transcript panel.
    """
    if not timeline:
        return transcript

    points = [p for p in (timeline.get("points") or [])
              if p.get("attention") is not None]
    zones = timeline.get("weak_zones") or []

    for segment in transcript.get("segments") or []:
        start, end = segment.get("t"), segment.get("end")
        if start is None:
            continue
        end = end if end is not None else start + 2.0

        covering = [p["attention"] for p in points if start <= p["t"] <= end]
        if not covering and points:
            # A short line can fall between two samples; the nearest frame is a
            # better answer than none.
            nearest = min(points, key=lambda p: abs(p["t"] - start))
            covering = [nearest["attention"]]

        segment["attention"] = (round(sum(covering) / len(covering))
                                if covering else None)
        segment["in_weak_zone"] = any(
            not (end < z["t_start"] or start > z["t_end"]) for z in zones)

    return transcript


def summarise(transcript: dict) -> dict:
    """Facts the triggers and the LLM need, without re-deriving them."""
    segments = transcript.get("segments") or []
    if not segments:
        return {"lines": 0, "words": 0, "seconds_of_speech": 0.0}

    words = sum(len(s["text"].split()) for s in segments)
    speech = sum(max(0.0, (s.get("end") or s["t"]) - s["t"]) for s in segments)
    attended = [s["attention"] for s in segments if s.get("attention") is not None]

    return {
        "lines": len(segments),
        "words": words,
        "seconds_of_speech": round(speech, 2),
        "words_per_second": round(words / speech, 2) if speech else None,
        "mean_line_attention": round(sum(attended) / len(attended), 1)
        if attended else None,
        "lines_in_weak_zones": sum(1 for s in segments if s.get("in_weak_zone")),
    }


def unavailable(reason: str, message: str = "") -> dict:
    """A transcript that could not be produced, stated rather than omitted."""
    return {"segments": [], "text": "", "language": None,
            "available": False, "reason": reason, "message": message}


REASONS = {
    NO_AUDIO_STREAM: "The creative has no audio track.",
    TRANSCRIPTION_NOT_CONFIGURED: "Transcription is not configured on this server.",
    TRANSCRIPTION_PROVIDER_ERROR: "The transcription provider could not be reached.",
    TRANSCRIPT_EMPTY: "No speech was found in the audio.",
}
