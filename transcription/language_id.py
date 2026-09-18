"""
Which language is actually being spoken - asked of Gemini, not of Deepgram.

WHY NOT DEEPGRAM'S OWN DETECTOR
    Measured 2026-09-17 on real calls. Deepgram picks ONE language for a file.
    Our reps open in English and do about 85% of the talking, so it answered
    "en" at 98.5% confidence for a call whose customer spoke Marathi - and it
    still answered "en" when handed only that customer's audio. Acting on that
    costs us the customer's half of the conversation.

    Gemini identified all four test calls correctly, including a 20-second
    customer-only clip, and it reports the SHARE of each language rather than a
    single winner. That share is what separates English-with-loanwords from a
    call that switches language outright.

WHAT THIS DOES NOT DO
    It does not transcribe. Gemini was tested for transcription and invented
    timestamps, so Deepgram keeps that job. This module answers one question -
    which languages, in what proportion - and returns a decision the caller is
    free to ignore.

    Every failure returns a decision with `ok=False` and a reason. Identification
    is an optimisation; an analysis must never fail because it could not run.

COST
    About 25 tokens per second of audio, so roughly $0.002 for a three-minute
    window at Gemini Flash rates. Callers should send a window rather than a
    whole call - audio_slice.window() exists for that.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from google.genai import types

logger = logging.getLogger("transcription.language_id")

TIMEOUT_SECONDS = 60.0

# English loanwords are everywhere in Indian sales calls - demo, payment, EMI.
# Counting them as English is what made the naive detectors answer "en", so the
# instruction rules them out explicitly.
PROMPT = (
    "You are given audio from a sales call between a sales representative and a "
    "customer.\n\n"
    "Identify which languages are SPOKEN in it.\n\n"
    "Rules:\n"
    "- Ignore individual English loanwords borrowed into another language, such "
    "as demo, payment, EMI or platform. A sentence whose grammar is Hindi or "
    "Marathi is Hindi or Marathi, not English.\n"
    "- Report the approximate share of speech for each language, as percentages "
    "that add up to 100.\n"
    "- Use ISO 639-1 codes: en, hi, mr, ta, te, kn, bn, gu, pa, ur, ml.\n"
    "- dominant_non_english is the code of the most spoken language that is NOT "
    "English, or null when English is the only language spoken.\n"
    "- Report only what you actually hear. Do not guess from names, accents or "
    "the topic of the call."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "share_percent": {"type": "number"},
                },
                "required": ["code", "share_percent"],
            },
        },
        "dominant_non_english": {"type": "string", "nullable": True},
    },
    "required": ["languages", "dominant_non_english"],
}

# Why a decision could not be used.
REASON_NOT_CONFIGURED = "language_id_not_configured"
REASON_NO_AUDIO = "language_id_no_audio"
REASON_PROVIDER_ERROR = "language_id_provider_error"
REASON_UNREADABLE = "language_id_unreadable_response"


@dataclass
class LanguageDecision:
    """What was heard. `ok` false means the caller should carry on without it."""
    ok: bool = False
    dominant_non_english: Optional[str] = None
    languages: list[dict] = field(default_factory=list)
    english_share: Optional[float] = None
    dominant_share: Optional[float] = None
    reason: Optional[str] = None
    model: Optional[str] = None
    model_version: Optional[str] = None
    ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

    def as_meta(self) -> dict:
        """Flat, storable, and free of anything said on the call."""
        return {
            "language_detected": self.dominant_non_english,
            "language_detection_ok": self.ok,
            "language_detection_reason": self.reason,
            "language_detection_shares": self.languages,
            "language_detection_ms": self.ms,
            "language_id_input_tokens": self.input_tokens,
            "language_id_output_tokens": self.output_tokens,
            "language_id_thinking_tokens": self.thinking_tokens,
            "language_id_cached_tokens": self.cached_tokens,
        }


def _share(languages: list[dict], code: str) -> Optional[float]:
    for entry in languages:
        if (entry.get("code") or "").strip().lower() == code:
            try:
                return float(entry.get("share_percent"))
            except (TypeError, ValueError):
                return None
    return None


def _parse(text: str, decision: LanguageDecision) -> LanguageDecision:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("the response was not a JSON object")
    raw = payload.get("languages")
    languages: list[dict] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        code = (entry.get("code") or "").strip().lower()
        if not code:
            continue
        try:
            share = float(entry.get("share_percent"))
        except (TypeError, ValueError):
            share = None
        languages.append({"code": code, "share_percent": share})

    dominant = (payload.get("dominant_non_english") or "").strip().lower() or None
    if dominant == "en":        # "dominant NON-english" cannot be English
        dominant = None

    decision.languages = languages
    decision.dominant_non_english = dominant
    decision.english_share = _share(languages, "en")
    decision.dominant_share = _share(languages, dominant) if dominant else None
    decision.ok = True
    return decision


async def identify(client: Any, model: str, audio: bytes, *,
                   mime_type: str = "audio/mpeg",
                   timeout: float = TIMEOUT_SECONDS) -> LanguageDecision:
    """Ask which languages are spoken in this audio. Never raises."""
    decision = LanguageDecision(model=model)
    if client is None or not model:
        decision.reason = REASON_NOT_CONFIGURED
        return decision
    if not audio:
        decision.reason = REASON_NO_AUDIO
        return decision

    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        http_options=types.HttpOptions(timeout=int(timeout * 1000)),
    )
    started = time.monotonic()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=audio, mime_type=mime_type), PROMPT],
                config=config),
            timeout=timeout + 5)
    except Exception as err:  # noqa: BLE001 - provider errors are opaque, and this is optional
        decision.ms = int((time.monotonic() - started) * 1000)
        decision.reason = REASON_PROVIDER_ERROR
        logger.warning("language identification failed: %s", type(err).__name__)
        return decision

    decision.ms = int((time.monotonic() - started) * 1000)
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        decision.input_tokens = getattr(usage, "prompt_token_count", None)
        decision.output_tokens = getattr(usage, "candidates_token_count", None)
        decision.thinking_tokens = getattr(usage, "thoughts_token_count", None)
        decision.cached_tokens = getattr(usage, "cached_content_token_count", None)
    version = getattr(response, "model_version", None)
    decision.model_version = version if isinstance(version, str) and version else None

    try:
        return _parse(getattr(response, "text", "") or "", decision)
    except Exception:  # noqa: BLE001 - a malformed answer is simply unusable
        decision.reason = REASON_UNREADABLE
        logger.warning("language identification returned an unreadable response")
        return decision
