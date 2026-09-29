"""
Tone - how something was said, heard from the audio.

WHY
    The analysis only ever read the transcript. The same words - "I don't
    think this works for me" - are curious, hesitant or angry depending on the
    voice, and the criteria that ask about tone were rated without it. Here
    Gemini LISTENS to chosen turns of the call and says how each was delivered.

WHAT IT RETURNS, per turn listened to
    tone         one of TONES (the primary impression)
    secondary    optionally a second one
    intensity    low | medium | high
    confidence   low | medium | high
    cue          what in the voice showed it ("raised pitch, clipped words")
    Plus the acoustic measures from prosody.py, so a manager can see whether
    the voice itself agrees: `acoustic_support` says whether the measured
    delivery is consistent with the tone heard.

WHICH TURNS
    Not the whole call - the moments that matter and the ones that sound
    different: turns with objections, price or commitment language, the
    customer's last turns, and the turns furthest from that speaker's own usual
    pitch, loudness and pace. MAX_MOMENTS bounds cost.

NEVER FAILS AN ANALYSIS. Every failure is a reason code and no tone.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("sales_call_analyzer.tone")

RATE = 16000
TONES = ("neutral", "interested", "confident", "hesitant", "confused",
         "frustrated", "urgent", "disengaged")
LEVELS = ("low", "medium", "high")

MAX_MOMENTS = int(os.environ.get("SCA_TONE_MAX_MOMENTS", 12))
MIN_SECONDS = 0.8
# Fewer words than this ("Yeah. The") says too little to be worth a listen: on a
# real call two of twelve moments were such fillers, chosen only for sounding odd.
MIN_WORDS = 4
MAX_CLIP_SECONDS = 20.0
TIMEOUT_SECONDS = 120.0

REASON_NOT_CONFIGURED = "tone_not_configured"
REASON_PROVIDER_ERROR = "tone_provider_error"
REASON_UNREADABLE = "tone_unreadable_response"
REASON_NO_MOMENTS = "tone_no_turns_to_listen_to"

# v2 (2026-09-28). Measured on CREMA-D (real actors, same sentence, different
# emotion): v1 heard HOW INTENSE a voice was but not whether it was positive or
# negative - anger came back "urgent" (13/40) or "confident" (9/40), fear
# "urgent". v2 narrows "urgent" to time pressure and "confident" to calm, puts
# a tense or raised voice under "frustrated" and nervousness under "hesitant".
# 44% -> 48% on CREMA-D, the same on both halves of the actors. Adding each
# clip's measured delivery to the prompt did not help (48.5% without) and is
# off in the pipeline; ToneClip.measured is kept for experiments.
PROMPT_VERSION = "tone-v2"
PROMPT = """You are listening to {n} numbered audio clips from a phone sales call in India.
For each clip you are told who is speaking, the words they said, and - where it could be
measured - how the delivery differed from that speaker's own usual voice on this call.

Judge HOW each clip is said - the voice, not the words. The same sentence can be
interested, hesitant, frustrated or confident depending on pitch, loudness, pace,
pauses and energy. First decide how much energy the voice has, then whether that energy
is positive (warm, pleased) or negative (tense, irritated, anxious) - both matter.

For each clip give:
- tone: the single best description of the delivery, one of {tones}
- secondary: a second one if clearly also present, else null
- intensity: low, medium or high
- confidence: how sure you are from the audio - low, medium or high. Use low when
  the voice itself gives little away.
- cue: a few words on what you HEARD that shows it (e.g. "raised pitch, clipped,
  tense"), never a paraphrase of the words.
Definitions:
- neutral: plain, even delivery with no strong colour
- interested: warm, pleased, lively, engaged - positive energy
- confident: calm, steady, assured, unhurried. A loud or forceful voice is NOT confident.
- hesitant: unsure, halting, anxious, nervous, shaky, trailing off
- confused: puzzled, questioning intonation, as if not understanding
- frustrated: irritated, impatient, annoyed, angry, tense or raised voice - negative energy
- urgent: pressing about TIME - hurrying to get something done soon. High energy alone is
  not urgency; a tense raised voice is frustrated.
- disengaged: flat, low energy, tired, down, minimal effort, uninterested
Return JSON: {{"clips": [{{"n": 1, "tone": "...", "secondary": null, "intensity": "...",
"confidence": "...", "cue": "..."}}, ...]}} with one entry per clip, in order."""

SCHEMA = {"type": "object", "properties": {"clips": {"type": "array", "items": {
    "type": "object", "properties": {
        "n": {"type": "integer"},
        "tone": {"type": "string", "enum": list(TONES)},
        "secondary": {"type": "string", "enum": list(TONES), "nullable": True},
        "intensity": {"type": "string", "enum": list(LEVELS)},
        "confidence": {"type": "string", "enum": list(LEVELS)},
        "cue": {"type": "string"}},
    "required": ["n", "tone", "intensity", "confidence", "cue"]}}}, "required": ["clips"]}

# Words that mark the moments tone matters most in a sales call.
_KEY_MOMENT = re.compile(
    r"\b(price|pricing|cost|expensive|budget|discount|fee|fees|lakh|rupees|think about|"
    r"not sure|don't think|already using|busy|later|call back|demo|start|sign|buy|"
    r"interested|problem|issue|why|how much|कीमत|महंगा|सोच|बाद में|पैसे|बजट)\b", re.IGNORECASE)


@dataclass
class ToneClip:
    samples: np.ndarray
    role: str
    text: str
    measured: str = ""      # describe(z): delivery against the speaker's usual


_MEASURE_WORDS = {
    "loudness_db": ("louder", "quieter"), "pitch_range_st": ("wider pitch", "flatter pitch"),
    "rate_wps": ("faster", "slower"), "pause_ratio": ("more pauses", "fewer pauses"),
    "pitch_hz": ("higher pitch", "lower pitch"),
}


def describe(measured: dict, threshold: float = 1.0) -> str:
    """z-scores against the speaker's usual, as words - only clear differences."""
    words = [(_MEASURE_WORDS[k][0] if v > 0 else _MEASURE_WORDS[k][1])
             for k, v in (measured or {}).items()
             if k in _MEASURE_WORDS and v is not None and abs(v) >= threshold]
    return ", ".join(words) + " than usual" if words else "close to their usual delivery"


@dataclass
class ToneResult:
    ok: bool = False
    labels: list[Optional[dict]] = field(default_factory=list)
    reason: Optional[str] = None
    requests: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
    ms: Optional[int] = None


def _wav(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _add(total, value):
    return total if value is None else (total or 0) + value


async def classify_clips(client: Any, model: str, clips: list[ToneClip], *,
                         timeout: float = TIMEOUT_SECONDS) -> ToneResult:
    """How each clip was said. Never raises."""
    result = ToneResult(labels=[None] * len(clips))
    if client is None or not model:
        result.reason = REASON_NOT_CONFIGURED
        return result
    if not clips:
        result.reason = REASON_NO_MOMENTS
        return result
    from google.genai import types

    parts: list[Any] = []
    for n, clip in enumerate(clips, 1):
        audio = clip.samples[: int(MAX_CLIP_SECONDS * RATE)]
        label = f'Clip {n} ({len(audio) / RATE:.1f} s) - {clip.role} said: "{clip.text[:400]}"'
        if clip.measured:
            label += f" Measured: {clip.measured}."
        parts += [label,
                  types.Part.from_bytes(data=_wav(audio), mime_type="audio/wav")]
    parts.append(PROMPT.format(n=len(clips), tones=", ".join(TONES)))
    config = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json", response_schema=SCHEMA,
        http_options=types.HttpOptions(timeout=int(timeout * 1000)))
    started = time.monotonic()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(model=model, contents=parts, config=config),
            timeout=timeout + 5)
    except Exception as err:  # noqa: BLE001 - optional step
        logger.warning("tone listening failed: %s", type(err).__name__)
        result.reason = REASON_PROVIDER_ERROR
        return result
    result.ms = int((time.monotonic() - started) * 1000)
    result.requests = 1
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        result.input_tokens = _add(None, getattr(usage, "prompt_token_count", None))
        result.output_tokens = _add(None, getattr(usage, "candidates_token_count", None))
        result.thinking_tokens = _add(None, getattr(usage, "thoughts_token_count", None))
    try:
        got = {int(c["n"]): c for c in json.loads(response.text)["clips"]}
    except Exception:  # noqa: BLE001
        result.reason = REASON_UNREADABLE
        return result
    for n in range(1, len(clips) + 1):
        c = got.get(n)
        if not c or c.get("tone") not in TONES:
            continue
        secondary = c.get("secondary") if c.get("secondary") in TONES else None
        result.labels[n - 1] = {
            "tone": c["tone"],
            "secondary": secondary if secondary != c["tone"] else None,
            "intensity": c.get("intensity") if c.get("intensity") in LEVELS else "medium",
            "confidence": c.get("confidence") if c.get("confidence") in LEVELS else "low",
            "cue": (c.get("cue") or "")[:160]}
    result.ok = True
    return result


# =========================================================================== #
# Which turns to listen to
# =========================================================================== #
def select_moments(transcript, measures: list, roles: dict[str, str],
                   limit: int = MAX_MOMENTS) -> list[int]:
    """Segment indices worth hearing, most informative first, in call order.

    Priority: the customer's key moments (price, objections, commitment) and
    last turns; then the turns that sound least like the speaker usually does;
    then the rep's key moments. A turn shorter than MIN_SECONDS carries too
    little voice to judge.
    """
    segs = {s.index: s for s in transcript.segments
            if s.start is not None and s.end is not None and s.end - s.start >= MIN_SECONDS
            and len((s.text or "").split()) >= MIN_WORDS}
    if not segs:
        return []
    deviation = {m.index: max((abs(v) for v in m.z.values()), default=0.0) for m in measures}

    def score(i: int) -> float:
        s = segs[i]
        customer = roles.get(s.speaker_id) != "sales_rep"
        key = bool(_KEY_MOMENT.search(s.text or ""))
        return (3.0 if customer and key else 0) + (1.5 if key else 0) \
            + min(deviation.get(i, 0.0), 3.0) + (0.5 if customer else 0)

    ranked = sorted(segs, key=score, reverse=True)
    customer_turns = [i for i in segs if roles.get(segs[i].speaker_id) != "sales_rep"]
    chosen = list(dict.fromkeys(customer_turns[-2:] + ranked))[:limit]
    return sorted(chosen)


# =========================================================================== #
# Does the voice agree?
# =========================================================================== #
# What each tone should look like relative to the speaker's own average, as
# z-scores (loudness, pitch range, pace, pauses). "supports" means the measured
# delivery points the same way; "contradicts" means it clearly points the other.
_EXPECT = {
    "frustrated": {"loudness_db": +1, "pitch_range_st": +1},
    "urgent": {"rate_wps": +1, "pause_ratio": -1},
    "hesitant": {"pause_ratio": +1, "rate_wps": -1},
    "disengaged": {"loudness_db": -1, "pitch_range_st": -1},
    "interested": {"pitch_range_st": +1},
    "confident": {"pause_ratio": -1},
}


def acoustic_support(tone: str, z: dict) -> str:
    """supports | contradicts | neutral - never used to drop a label, only to
    show a reader whether the measured voice agrees with what was heard."""
    expect = _EXPECT.get(tone)
    if not expect or not z:
        return "neutral"
    votes = [np.sign(z[k]) * sign for k, sign in expect.items() if k in z and abs(z[k]) >= 0.5]
    if not votes:
        return "neutral"
    total = sum(votes)
    return "supports" if total > 0 else "contradicts" if total < 0 else "neutral"
