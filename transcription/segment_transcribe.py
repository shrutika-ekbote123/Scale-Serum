"""
Re-transcribing chosen stretches of a call with Gemini - the words only.

WHY
    Deepgram's `multi` covers Hindi only among Indian languages, and a regional
    model cannot write English. On a call where the rep speaks English and the
    customer speaks Kannada, Marathi, Punjabi, Tamil or Telugu, neither setting
    works for the whole call. Measured 2026-09-28 (scripts/sca_eval/
    segment_pass_experiment.py) on the customer's turns of 8 such calls:

                                      WER    CER    words kept
        today (nova-3 multi)          57.0%  47.3%  0.70
        Deepgram regional model       62.6%  37.8%  0.80
        Gemini, one clip per turn     23.4%  17.4%  0.89
        best of the three per turn    22.6%  17.2%  0.89   (the ceiling)

    Checked on a real Marathi call too: multi wrote Hindi-sounding nonsense,
    Deepgram's `mr` model kept the Marathi but garbled the English and dropped
    a whole exchange, Gemini's text was correct Marathi with the English intact.

WHAT GEMINI IS AND IS NOT TRUSTED WITH
    language_id.py records that Gemini invented timestamps. Here it is never
    asked for one: Deepgram's turn boundaries cut the clips, each clip is a
    separate numbered audio part, and Gemini returns only the words of each.
    Timing and speakers stay Deepgram's (and the speaker check's).

GUARDED
    A clip's text is refused - and the original kept - when it is empty, or
    longer than anyone speaks (MAX_CHARS_PER_SECOND of romanised text), which is
    what an invented sentence looks like. Every failure returns ok=False with a
    reason; the caller keeps Deepgram's text. Never raises.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .romanize import romanize

logger = logging.getLogger("transcription.segment_transcribe")

RATE = 16000
TIMEOUT_SECONDS = 120.0
# One request carries at most this much: long enough for a whole customer side
# of most calls, short enough to stay fast and bounded.
MAX_CLIPS_PER_REQUEST = 40
MAX_SECONDS_PER_REQUEST = 300.0
# Brisk Indian-English and Hindi speech is ~15-20 romanised characters a second.
MAX_CHARS_PER_SECOND = 30.0
# Verbatim transcription needs no reasoning, and thinking tokens are billed as
# output. None leaves the model's default. See the README of scripts/sca_eval
# for what 0 did to accuracy and cost.
_budget = os.environ.get("SCA_SEGMENT_PASS_THINKING_BUDGET", "").strip()
THINKING_BUDGET = int(_budget) if _budget.lstrip("-").isdigit() else None

LANGUAGES = {
    "hi": ("Hindi", "Devanagari"), "mr": ("Marathi", "Devanagari"),
    "kn": ("Kannada", "Kannada script"), "pa": ("Punjabi", "Gurmukhi"),
    "ta": ("Tamil", "Tamil script"), "te": ("Telugu", "Telugu script"),
    "bn": ("Bengali", "Bengali script"), "gu": ("Gujarati", "Gujarati script"),
    "ur": ("Urdu", "Urdu script"), "ml": ("Malayalam", "Malayalam script"),
}

# v2 (2026-09-28). v1 let two things through, both seen on a Telugu call:
#   * English words written in the regional script ("కాస్ట్" for "cost") -
#     the words right, but unreadable to a manager and not quotable as English
#   * words moved between clips - clip 2's words returned under clip 1, which
#     would put them on the wrong turn and the wrong speaker
# Each clip's duration is now given and both are ruled out by example.
PROMPT_VERSION = "segment-pass-v2"
PROMPT = """You are given {n} numbered audio clips, cut in order from one phone sales call in India.
The speaker mostly uses {lang}, mixed with English words.

Transcribe each clip VERBATIM, on its own.
- Each clip's words belong to that clip only. Never move words from one clip into another,
  even if a sentence seems to continue. A clip may hold a single word such as "okay".
- Write {lang} words in {script}. Write every English word in Latin letters, exactly as
  English is spelled - for example "cost", "details", "weekends", never in {script}.
- Do not translate, summarise, correct grammar or add words that are not spoken.
- If a clip has no intelligible speech, return an empty string for it.
Return JSON: {{"clips": [{{"n": 1, "text": "..."}}, ...]}} with one entry per clip, in order."""

SCHEMA = {"type": "object", "properties": {"clips": {"type": "array", "items": {
    "type": "object", "properties": {"n": {"type": "integer"}, "text": {"type": "string"}},
    "required": ["n", "text"]}}}, "required": ["clips"]}

REASON_NOT_CONFIGURED = "segment_pass_not_configured"
REASON_UNSUPPORTED_LANGUAGE = "segment_pass_unsupported_language"
REASON_PROVIDER_ERROR = "segment_pass_provider_error"
REASON_UNREADABLE = "segment_pass_unreadable_response"

REJECT_EMPTY = "empty"
REJECT_TOO_LONG = "longer_than_speech_allows"


@dataclass
class ClipResult:
    """Per clip: the new text, or None where the original must stand."""
    ok: bool = False
    texts: list[Optional[str]] = field(default_factory=list)
    rejected: dict[int, str] = field(default_factory=dict)     # clip index -> why
    reason: Optional[str] = None
    requests: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    ms: Optional[int] = None


def _wav(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _batches(clips: list[np.ndarray]) -> list[list[int]]:
    out, cur, seconds = [], [], 0.0
    for i, clip in enumerate(clips):
        length = len(clip) / RATE
        if cur and (len(cur) >= MAX_CLIPS_PER_REQUEST or seconds + length > MAX_SECONDS_PER_REQUEST):
            out.append(cur)
            cur, seconds = [], 0.0
        cur.append(i)
        seconds += length
    if cur:
        out.append(cur)
    return out


def _add(total: Optional[int], value: Optional[int]) -> Optional[int]:
    if value is None:
        return total
    return (total or 0) + value


def check(text: Optional[str], seconds: float) -> Optional[str]:
    """Why this clip's text must not be used, or None if it may."""
    if not (text or "").strip():
        return REJECT_EMPTY
    if len(romanize(text).replace(" ", "")) > MAX_CHARS_PER_SECOND * max(seconds, 0.5):
        return REJECT_TOO_LONG
    return None


KEYTERMS_LINE = ("Names and terms that may be spoken on this call: {terms}. If you hear one, "
                 "spell it exactly like that. Never write one that is not actually spoken.")


async def transcribe_clips(client: Any, model: str, clips: list[np.ndarray],
                           language: str, *, keyterms: Optional[list[str]] = None,
                           timeout: float = TIMEOUT_SECONDS) -> ClipResult:
    """Words for each 16 kHz clip, in the given dominant language. Never raises."""
    result = ClipResult(texts=[None] * len(clips))
    if client is None or not model:
        result.reason = REASON_NOT_CONFIGURED
        return result
    if language not in LANGUAGES:
        result.reason = REASON_UNSUPPORTED_LANGUAGE
        return result
    if not clips:
        result.ok = True
        return result

    from google.genai import types

    name, script = LANGUAGES[language]
    config = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json", response_schema=SCHEMA,
        http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        thinking_config=(types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                         if THINKING_BUDGET is not None else None))
    started = time.monotonic()
    for batch in _batches(clips):
        parts: list[Any] = []
        for n, i in enumerate(batch, 1):
            parts += [f"Clip {n} ({len(clips[i]) / RATE:.1f} s):",
                      types.Part.from_bytes(data=_wav(clips[i]), mime_type="audio/wav")]
        prompt = PROMPT.format(n=len(batch), lang=name, script=script)
        if keyterms:
            prompt += "\n" + KEYTERMS_LINE.format(terms=", ".join(keyterms[:20]))
        parts.append(prompt)
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(model=model, contents=parts, config=config),
                timeout=timeout + 5)
        except Exception as err:  # noqa: BLE001 - opaque provider errors; this step is optional
            logger.warning("segment re-transcription failed: %s", type(err).__name__)
            result.reason = REASON_PROVIDER_ERROR
            break
        result.requests += 1
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            result.input_tokens = _add(result.input_tokens, getattr(usage, "prompt_token_count", None))
            result.output_tokens = _add(result.output_tokens, getattr(usage, "candidates_token_count", None))
            result.thinking_tokens = _add(result.thinking_tokens, getattr(usage, "thoughts_token_count", None))
            result.cached_tokens = _add(result.cached_tokens, getattr(usage, "cached_content_token_count", None))
        try:
            got = {int(c["n"]): c.get("text") for c in json.loads(response.text)["clips"]}
        except Exception:  # noqa: BLE001 - a malformed answer is simply unusable
            logger.warning("segment re-transcription returned an unreadable response")
            result.reason = REASON_UNREADABLE
            break
        for n, i in enumerate(batch, 1):
            text = (got.get(n) or "").strip()
            why = check(text, len(clips[i]) / RATE)
            if why:
                result.rejected[i] = why
            else:
                result.texts[i] = text
    result.ms = int((time.monotonic() - started) * 1000)
    # Partial success is still success for the clips that came back.
    result.ok = result.requests > 0
    return result
