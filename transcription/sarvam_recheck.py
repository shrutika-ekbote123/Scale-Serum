"""
Letting Gemini listen again to the few turns of a Sarvam transcript that
probably hold a misheard brand or product term - and correct only those words.

STATUS: STANDALONE. Nothing in sales_call_analyzer imports this yet.

WHY
    sarvam_cleanup.py fixes what text alone can fix safely. "free fund" for
    refund, "JTPT" for ChatGPT, "Child version" for trial version, "law
    training" for Lawtorney cannot be decided from the text: the same letters
    are a real phrase elsewhere. The audio decides. sarvam_cleanup flags those
    turns; this sends just their clips to Gemini.

HOW IT DIFFERS FROM segment_transcribe.py
    That module re-transcribes from scratch (verbatim, regional languages only).
    This one hands Gemini Sarvam's text and asks for the misheard words to be
    corrected and everything else kept - so it works on English calls too, and
    a good transcript cannot be made worse by a fresh mistake elsewhere in the
    turn. Clip audio and the length check are shared with segment_transcribe.

GUARDED
    A correction is refused, and Sarvam's text kept, when it is empty, longer
    than anyone speaks (segment_transcribe.check), rewrites more than half of
    the turn's words - a correction, not a new transcript - or brings in a
    script the turn did not have (2026-10-01, mycall7: "We are not for law
    training" came back as "क्या आप part of Lawtorney"). Never raises.

THINKING
    Capped by default (THINKING_BUDGET). Correcting a few words needs little
    reasoning, and thinking is billed as output: on one run mycall7's three
    clips used 10,556 thinking tokens (Rs 3.59) against 467 on another.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from rapidfuzz.distance import Levenshtein

from .romanize import romanize
from .sarvam_cleanup import Entry, apply_terms, script_of
from .segment_transcribe import RATE, SCHEMA, _batches, _wav, check

logger = logging.getLogger("transcription.sarvam_recheck")

TIMEOUT_SECONDS = 120.0
CLIP_PAD_SECONDS = 0.15
MAX_CHANGED_WORD_SHARE = 0.5
MAX_TERMS_IN_PROMPT = 30
_budget = os.environ.get("SCA_SARVAM_RECHECK_THINKING_BUDGET", "512").strip()
THINKING_BUDGET = int(_budget) if _budget.lstrip("-").isdigit() else None

PROMPT_VERSION = "sarvam-recheck-v1"
PROMPT = """You are checking {n} numbered audio clips, cut in order from one phone sales call in India.
For each clip you get its audio and the text a speech recogniser wrote for it. That text is mostly right.

Listen, and correct ONLY the words that were misheard.
These names and terms are spoken on this call. Where you hear one, write it exactly like this: {terms}
- Keep every correctly heard word exactly as written, in the same script: words of an Indian
  language in their own script, English words in Latin letters.
- Never translate, rephrase, summarise or add words that are not spoken. Never write one of the
  terms above unless it is actually spoken in that clip.
- If the text is already right, return it unchanged.
Return JSON: {{"clips": [{{"n": 1, "text": "..."}}, ...]}} with one entry per clip, in order."""

REASON_NOT_CONFIGURED = "recheck_not_configured"
REASON_PROVIDER_ERROR = "recheck_provider_error"
REASON_UNREADABLE = "recheck_unreadable_response"
REJECT_TOO_MUCH_CHANGED = "rewrote_more_than_a_correction"
REJECT_NEW_SCRIPT = "brought_in_a_new_script"


@dataclass
class RecheckResult:
    ok: bool = False
    changes: list[dict] = field(default_factory=list)       # index, before, after
    unchanged: list[int] = field(default_factory=list)
    rejected: dict[int, str] = field(default_factory=dict)   # entry index -> why
    reason: Optional[str] = None
    requests: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
    ms: Optional[int] = None

    def report(self) -> dict:
        return {"ok": self.ok, "prompt_version": PROMPT_VERSION, "reason": self.reason,
                "changes": self.changes, "unchanged": self.unchanged,
                "rejected": {str(k): v for k, v in self.rejected.items()},
                "requests": self.requests, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "thinking_tokens": self.thinking_tokens,
                "ms": self.ms}


def _add(total: Optional[int], value: Optional[int]) -> Optional[int]:
    if value is None:
        return total
    return (total or 0) + value


def changed_share(before: str, after: str) -> float:
    """Share of the turn's words a correction rewrote (romanised, so a script
    change of the same word does not count)."""
    a, b = romanize(before).split(), romanize(after).split()
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def scripts_in(text: str) -> set[str]:
    """The writing systems used: "latin" and/or Indic script names."""
    found = set()
    for ch in text or "":
        if ch.isalpha():
            found.add(script_of(ch) or "latin")
    return found


def judge(before: str, after: Optional[str], seconds: float,
          terms: Optional[list[str]] = None) -> Optional[str]:
    """Why this correction must not be used, or None if it may."""
    why = check(after, seconds)
    if why:
        return why
    if changed_share(before, after) > MAX_CHANGED_WORD_SHARE:
        return REJECT_TOO_MUCH_CHANGED
    allowed = scripts_in(before) | set().union(*(scripts_in(t) for t in terms or [""]))
    if not scripts_in(after) <= allowed:
        return REJECT_NEW_SCRIPT
    return None


def clip_of(samples: np.ndarray, entry: Entry) -> np.ndarray:
    a = max(0, int((entry.start - CLIP_PAD_SECONDS) * RATE))
    b = min(len(samples), int((entry.end + CLIP_PAD_SECONDS) * RATE))
    return samples[a:b]


async def recheck(client: Any, model: str, samples: np.ndarray, entries: list[Entry],
                  indices: list[int], terms: list[str], *, brand_terms: Optional[list[str]] = None,
                  timeout: float = TIMEOUT_SECONDS) -> RecheckResult:
    """Correct the chosen entries (by Entry.index) in place. `samples` is the
    whole call, mono float32 at 16 kHz. `terms` go into the prompt; accepted
    text is passed through sarvam_cleanup.apply_terms with `brand_terms` so the
    brand is written one way. Never raises."""
    result = RecheckResult()
    if client is None or not model:
        result.reason = REASON_NOT_CONFIGURED
        return result
    chosen = [e for e in entries if e.index in set(indices) and e.end > e.start]
    if not chosen:
        result.ok = True
        return result

    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json", response_schema=SCHEMA,
        http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        thinking_config=(types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                         if THINKING_BUDGET is not None else None))
    clips = [clip_of(samples, e) for e in chosen]
    listed = ", ".join(dict.fromkeys(t for t in terms if t))[:2000] or "(none given)"
    started = time.monotonic()
    for batch in _batches(clips):
        parts: list[Any] = []
        for n, i in enumerate(batch, 1):
            parts += [f"Clip {n} ({len(clips[i]) / RATE:.1f} s). Recognised text: «{chosen[i].text}»",
                      types.Part.from_bytes(data=_wav(clips[i]), mime_type="audio/wav")]
        parts.append(PROMPT.format(n=len(batch), terms=listed))
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(model=model, contents=parts, config=config),
                timeout=timeout + 5)
        except Exception as err:  # noqa: BLE001 - opaque provider errors; this step is optional
            logger.warning("Sarvam recheck failed: %s", type(err).__name__)
            result.reason = REASON_PROVIDER_ERROR
            break
        result.requests += 1
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            result.input_tokens = _add(result.input_tokens, getattr(usage, "prompt_token_count", None))
            result.output_tokens = _add(result.output_tokens, getattr(usage, "candidates_token_count", None))
            result.thinking_tokens = _add(result.thinking_tokens, getattr(usage, "thoughts_token_count", None))
        try:
            got = {int(c["n"]): c.get("text") for c in json.loads(response.text)["clips"]}
        except Exception:  # noqa: BLE001 - a malformed answer is simply unusable
            logger.warning("Sarvam recheck returned an unreadable response")
            result.reason = REASON_UNREADABLE
            break
        for n, i in enumerate(batch, 1):
            entry = chosen[i]
            text = (got.get(n) or "").strip()
            why = judge(entry.text, text, len(clips[i]) / RATE, terms)
            if why:
                result.rejected[entry.index] = why
                continue
            if brand_terms:
                text, _, _ = apply_terms(text, brand_terms)
            if text == entry.text:
                result.unchanged.append(entry.index)
                continue
            result.changes.append({"index": entry.index, "start": entry.start,
                                   "before": entry.text, "after": text})
            entry.text = text
    result.ms = int((time.monotonic() - started) * 1000)
    result.ok = result.requests > 0
    return result
