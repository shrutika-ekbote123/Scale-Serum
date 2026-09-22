"""Words one briefing section (Gemini), with the template as the fallback.

The model receives the section's facts, its watch items and the deterministic
draft, and returns {summary, top, watch, blocks}. It never chooses what
matters and never produces a number of its own.

Output is rejected - one repair retry, then the draft - when a field is missing
or too long, a block key is missing or extra, the model adds a top performer or
a watch item the draft does not have, a rep is named in the sales summary (that
summary is shown to reps without team view), or any number in the text does not
appear in the JSON it was given.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

from google.genai import types

from prompts import BRIEFING_REPAIR_INSTRUCTION, BRIEFING_SYSTEM_INSTRUCTION

PROMPT_VERSION = "briefing_p1"
LLM_TIMEOUT_MS = int(os.environ.get("BRIEFING_LLM_TIMEOUT_MS", 30_000))
TEMPERATURE = float(os.environ.get("BRIEFING_LLM_TEMPERATURE", 0.2))

LIMITS = {"summary": 420, "top": 160, "watch": 320, "bullet": 260, "bullets": 5}

RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "summary": types.Schema(type=types.Type.STRING),
        "top": types.Schema(type=types.Type.STRING),
        "watch": types.Schema(type=types.Type.STRING),
        "blocks": types.Schema(type=types.Type.ARRAY, items=types.Schema(
            type=types.Type.OBJECT,
            properties={"key": types.Schema(type=types.Type.STRING),
                        "bullets": types.Schema(type=types.Type.ARRAY,
                                                items=types.Schema(type=types.Type.STRING))},
            required=["key", "bullets"])),
    },
    required=["summary", "top", "watch", "blocks"],
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


def build_payload(section: dict, *, section_label: str, date_label: str,
                  brand: dict) -> dict:
    return {
        "section": section["section"],
        "section_label": section_label,
        "date": date_label,
        "brand": brand,
        "facts": section["facts"],
        "watch_items": [w["text"] for w in section["watch"]],
        "draft": section["template"],
    }


def _text(wording: dict) -> str:
    parts = [wording["summary"], wording["top"], wording["watch"]]
    for block in wording["blocks"]:
        parts.extend(block["bullets"])
    return " ".join(parts)


def validate(wording: dict, payload: dict, forbid_in_summary: tuple = ()) -> Optional[str]:
    """None when acceptable, otherwise a short reason code."""
    draft = payload["draft"]
    if not wording["summary"]:
        return "missing_summary"
    for field in ("summary", "top", "watch"):
        if len(wording[field]) > LIMITS[field]:
            return f"too_long_{field}"
    if wording["top"] and not draft.get("top"):
        return "invented_top"
    if wording["watch"] and not (draft.get("watch") or payload["watch_items"]):
        return "invented_watch"
    expected = [b["key"] for b in draft.get("blocks") or []]
    if [b["key"] for b in wording["blocks"]] != expected:
        return "block_keys_mismatch"
    for block in wording["blocks"]:
        if not block["bullets"] or len(block["bullets"]) > LIMITS["bullets"]:
            return "bad_bullet_count"
        if any(not b or len(b) > LIMITS["bullet"] for b in block["bullets"]):
            return "bad_bullet"
    summary = wording["summary"].lower()
    if any(name and name.lower() in summary for name in forbid_in_summary):
        return "rep_named_in_summary"
    text = _text(wording)
    allowed = _numbers(json.dumps(payload, ensure_ascii=False, default=str))
    if _numbers(text) - allowed:
        return "unsupported_number"
    if _GENDERED.search(text):
        return "gendered_pronoun"
    return None


def _clean(raw: dict) -> dict:
    blocks = []
    for b in raw.get("blocks") or []:
        if isinstance(b, dict):
            blocks.append({"key": str(b.get("key") or ""),
                           "bullets": [str(x).strip() for x in (b.get("bullets") or [])
                                       if str(x).strip()]})
    return {"summary": str(raw.get("summary") or "").strip(),
            "top": str(raw.get("top") or "").strip(),
            "watch": str(raw.get("watch") or "").strip(),
            "blocks": blocks}


def _usage(response) -> dict:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    out = {}
    for key, attr in (("input_tokens", "prompt_token_count"),
                      ("output_tokens", "candidates_token_count"),
                      ("thinking_tokens", "thoughts_token_count"),
                      ("cached_tokens", "cached_content_token_count")):
        value = getattr(meta, attr, None)
        if value is not None:
            out[key] = int(value)
    return out


def template_wording(payload: dict) -> dict:
    draft = payload["draft"]
    return {"summary": draft["summary"], "top": draft.get("top") or "",
            "watch": draft.get("watch") or "",
            "blocks": [{"key": b["key"], "bullets": list(b["bullets"])}
                       for b in draft.get("blocks") or []]}


async def write(client, model: str, payload: dict, *, forbid_in_summary: tuple = (),
                timeout_ms: int = LLM_TIMEOUT_MS) -> dict:
    """{"wording", "source": "llm"|"template", "fallback_reason", "meta": {attempts,
    latency_ms, usage}}."""
    started = time.monotonic()
    usage: dict = {}
    reason, attempts = "llm_unavailable", 0
    if client is not None:
        config = types.GenerateContentConfig(
            system_instruction=BRIEFING_SYSTEM_INSTRUCTION,
            temperature=TEMPERATURE,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        prompt = json.dumps(payload, ensure_ascii=False, default=str, indent=1)
        for attempt in range(2):
            attempts += 1
            contents = prompt if attempt == 0 else f"{prompt}\n\n{BRIEFING_REPAIR_INSTRUCTION}"
            try:
                response = await client.aio.models.generate_content(
                    model=model, contents=contents, config=config)
                for key, value in _usage(response).items():
                    usage[key] = usage.get(key, 0) + value
                raw = json.loads(response.text or "")
            except json.JSONDecodeError:
                reason = "unparseable_output"
                continue
            except Exception:  # noqa: BLE001 - provider errors are opaque
                reason = "llm_error"
                break          # a timeout or outage will not fix itself on retry
            if not isinstance(raw, dict):
                reason = "unparseable_output"
                continue
            wording = _clean(raw)
            reason = validate(wording, payload, forbid_in_summary)
            if reason is None:
                # The model may drop a callout the draft has; the tab would then
                # show a bare "Coach:" label. Keep the draft's wording for it.
                for callout in ("top", "watch"):
                    if not wording[callout] and payload["draft"].get(callout):
                        wording[callout] = payload["draft"][callout]
                return {"wording": wording, "source": "llm", "fallback_reason": None,
                        "meta": {"attempts": attempts, "usage": usage,
                                 "latency_ms": int((time.monotonic() - started) * 1000)}}
    return {"wording": template_wording(payload), "source": "template",
            "fallback_reason": reason,
            "meta": {"attempts": attempts, "usage": usage,
                     "latency_ms": int((time.monotonic() - started) * 1000)}}
