"""Words the decided action for the card (Gemini), with a template fallback.

The model receives the decision (action + urgency, already fixed by rules.py) and
the evidence, and returns four strings. It is never asked what to do or how
urgent it is.

Output is rejected - one repair retry, then the template - when a field is
missing or too long, or when it contains a number that does not appear anywhere
in the facts it was given. That last check is what stops an invented amount,
date or count from reaching a sales rep.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

from google.genai import types

from prompts import NEXT_ACTION_REPAIR_INSTRUCTION, NEXT_ACTION_SYSTEM_INSTRUCTION

PROMPT_VERSION = "na_p1"
LLM_TIMEOUT_MS = int(os.environ.get("NA_LLM_TIMEOUT_MS", 15_000))
TEMPERATURE = float(os.environ.get("NA_LLM_TEMPERATURE", 0.2))

LIMITS = {"title": 90, "recommendation": 400, "reason_headline": 90, "reason": 400}

RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "title": types.Schema(type=types.Type.STRING),
        "recommendation": types.Schema(type=types.Type.STRING),
        "reason_headline": types.Schema(type=types.Type.STRING),
        "reason": types.Schema(type=types.Type.STRING),
    },
    required=list(LIMITS),
)

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_GENDERED = re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", re.IGNORECASE)


def _numbers(text: str) -> set[str]:
    out = set()
    for token in _NUMBER.findall(text):
        token = token.replace(",", "").rstrip(".")
        if token.endswith(".0"):
            token = token[:-2]
        if token:
            out.add(token)
    return out


class _SafeDict(dict):
    def __missing__(self, key):
        return "not recorded"


def template_wording(template: dict, facts: dict) -> dict:
    fill = _SafeDict({k: v for k, v in facts.items() if not isinstance(v, (list, dict))})
    return {
        "title": template["title"].format_map(fill),
        "recommendation": template["text"].format_map(fill),
        "reason_headline": template["reason_headline"].format_map(fill),
        "reason": template["reason_text"].format_map(fill),
    }


def validate(wording: dict, payload: dict) -> Optional[str]:
    """None when acceptable, otherwise a short reason code."""
    for field, limit in LIMITS.items():
        value = wording.get(field)
        if not isinstance(value, str) or not value.strip():
            return f"missing_{field}"
        if len(value) > limit:
            return f"too_long_{field}"
    text = " ".join(wording[f] for f in LIMITS)
    allowed = _numbers(json.dumps(payload, ensure_ascii=False, default=str))
    if _numbers(text) - allowed:
        return "unsupported_number"
    # The CRM records no gender; a pronoun here is the model guessing from a name.
    if _GENDERED.search(text):
        return "gendered_pronoun"
    return None


async def write(client, model: str, payload: dict, template: dict,
                timeout_ms: int = LLM_TIMEOUT_MS) -> dict:
    """Returns {"wording", "source": "llm"|"template", "fallback_reason", "meta"}."""
    config = types.GenerateContentConfig(
        system_instruction=NEXT_ACTION_SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )
    prompt = json.dumps(payload, ensure_ascii=False, default=str, indent=1)
    started = time.monotonic()
    reason = "llm_unavailable"
    attempts = 0
    for attempt in range(2):
        attempts += 1
        contents = prompt if attempt == 0 else f"{prompt}\n\n{NEXT_ACTION_REPAIR_INSTRUCTION}"
        try:
            response = await client.aio.models.generate_content(
                model=model, contents=contents, config=config)
            wording = json.loads(response.text or "")
        except json.JSONDecodeError:
            reason = "unparseable_output"
            continue
        except Exception:  # noqa: BLE001 - provider errors are opaque
            reason = "llm_error"
            break          # a timeout or outage will not fix itself on retry
        if not isinstance(wording, dict):
            reason = "unparseable_output"
            continue
        wording = {k: str(wording.get(k) or "").strip() for k in LIMITS}
        reason = validate(wording, payload)
        if reason is None:
            return {"wording": wording, "source": "llm", "fallback_reason": None,
                    "meta": {"attempts": attempts,
                             "latency_ms": int((time.monotonic() - started) * 1000)}}
    return {"wording": template_wording(template, payload["facts"]), "source": "template",
            "fallback_reason": reason,
            "meta": {"attempts": attempts,
                     "latency_ms": int((time.monotonic() - started) * 1000)}}
