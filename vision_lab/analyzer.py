"""
Step 18 - the interpretation pass. Gemini reads what Python measured.

THE DIVISION THIS FILE ENFORCES
    The model receives measurements and a DEFECT LIST. It returns prose, ordinal
    ratings and evidence. It never returns a number that is a score, and it never
    nominates a defect of its own.

    Both rules are enforced in code, not merely asked for in the prompt. A
    response containing a numeric score is rejected outright; a recommendation
    whose `defect_id` is not in the list we sent is dropped. A prompt is a
    request, and a request is not a guarantee.

WHAT IT IS ASKED FOR
    * `key_message`   which element carries the central claim, and when. This is
                      what completes the Focus metric - until now Focus averaged
                      over ALL on-screen text because nothing had named the one
                      that matters.
    * `clarity`       an ordinal rating, which scoring.py converts. Never a score.
    * trigger ratings for the seven the counters cannot settle.
    * `recommendations` - prose for the defects it was handed.

FAILURE IS NEVER FATAL
    Gemini being unavailable, slow or malformed costs the report its
    interpretation layer and nothing else. The six scores, the timeline and the
    defect list are already computed and stand on their own.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from . import ANALYSIS_INVALID_OUTPUT, ANALYSIS_PROVIDER_ERROR

logger = logging.getLogger("vision_lab.analyzer")

PROMPT_VERSION = "vl_interpret_2026_09"

# How much of the measurement record the model sees. It does not need 120 frames
# of per-frame data to write four paragraphs, and sending them would cost tokens
# and invite the model to start doing arithmetic of its own.
MAX_TRANSCRIPT_LINES = 60
MAX_DEFECTS = 8

# Any of these in a rating field means the model produced a score. Ordinal
# ratings only - the conversion to a number happens in scoring.py.
RATING_LEVELS = ("absent", "weak", "adequate", "strong")


class AnalyzerError(Exception):
    def __init__(self, reason: str, message: str = ""):
        self.reason = reason
        super().__init__(message or reason)


# --------------------------------------------------------------------------- input
def build_context(*, media: dict, summary: dict, timeline: dict,
                  defects: list[dict], transcript: Optional[dict],
                  triggers: dict, brand_brain: Optional[dict] = None) -> dict:
    """What the model is shown. Deliberately a summary, not the raw record."""
    lines = []
    for segment in (transcript or {}).get("segments", [])[:MAX_TRANSCRIPT_LINES]:
        lines.append({"t": segment.get("t"), "text": segment.get("text"),
                      "attention": segment.get("attention"),
                      "in_weak_zone": segment.get("in_weak_zone")})

    points = timeline.get("points") or []
    # A sampled curve, not 120 points: enough to see the shape, not enough to
    # tempt the model into recomputing anything from it.
    step = max(1, len(points) // 24)

    return {
        "media": {k: media.get(k) for k in
                  ("kind", "duration_seconds", "width", "height",
                   "frames_analyzed", "shot_count", "has_audio")},
        "measured": {k: summary.get(k) for k in
                     ("shot_count", "cut_rate_per_minute", "max_words_on_screen",
                      "reading_speed_words_per_second", "overloaded_shots",
                      "brand", "cta", "frames_with_text_fraction")},
        "attention_timeline": [
            {"t": p.get("t"), "attention": p.get("attention")}
            for p in points[::step]],
        "weak_zones": timeline.get("weak_zones") or [],
        "defects": [
            {"defect_id": d["defect_id"], "severity": d["severity"],
             "t_start": d["t_start"], "t_end": d["t_end"],
             "measured": d["measured"], "note": d["note"]}
            for d in defects[:MAX_DEFECTS]],
        "transcript": lines,
        "triggers_already_measured": [
            {"id": t["id"], "name": t["name"], "status": t["status"],
             "note": t["note"]}
            for t in triggers.get("triggers", [])
            if t.get("detection") == "measured"],
        "triggers_needing_judgement": [
            {"id": t["id"], "name": t["name"], "subtitle": t.get("subtitle")}
            for t in triggers.get("triggers", [])
            if t.get("reason") == "awaiting_interpretation"],
        "brand": brand_brain or {},
    }


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_message": {
            "type": "object",
            "properties": {
                "element": {"type": "string"},
                "t": {"type": "number"},
                "quote": {"type": "string"},
                # WHICH CHANNEL CARRIES IT. Focus measures where GAZE lands, so
                # a claim delivered by the voiceover has nothing on screen to
                # measure against. Asked for explicitly, and then checked
                # against the frames in scoring.py rather than believed.
                "carrier": {"type": "string",
                            "enum": ["on_screen_text", "voiceover", "both",
                                     "visual"]},
            },
            "required": ["element"],
        },
        "clarity": {
            "type": "object",
            "properties": {
                "rating": {"type": "string", "enum": list(RATING_LEVELS)},
                "why": {"type": "string"},
            },
            "required": ["rating"],
        },
        "triggers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "rating": {"type": "string", "enum": list(RATING_LEVELS)},
                    "status": {"type": "string"},
                    "note": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "t": {"type": "number"},
                                "quote": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["id", "rating"],
            },
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "defect_id": {"type": "string"},
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["defect_id", "title", "why", "fix"],
            },
        },
        "observations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "recommendations"],
}


# --------------------------------------------------------------------------- call
async def interpret(context: dict, *, client: Any, model: str) -> dict:
    """One Gemini call. Raises AnalyzerError; the pipeline decides what that means."""
    from prompts import VISION_LAB_SYSTEM_INSTRUCTION

    if client is None:
        raise AnalyzerError(ANALYSIS_PROVIDER_ERROR, "No LLM client configured.")

    try:
        from google.genai import types
        response = await client.aio.models.generate_content(
            model=model,
            contents=json.dumps(context, default=str),
            config=types.GenerateContentConfig(
                system_instruction=VISION_LAB_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.4,
            ),
        )
    except Exception as err:  # noqa: BLE001
        raise AnalyzerError(ANALYSIS_PROVIDER_ERROR, str(err)[:200]) from err

    text = getattr(response, "text", None)
    if not text:
        raise AnalyzerError(ANALYSIS_INVALID_OUTPUT, "empty response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        raise AnalyzerError(ANALYSIS_INVALID_OUTPUT, str(err)[:200]) from err

    return validate(parsed, context)


# --------------------------------------------------------------------------- guard
# Verb forms included deliberately: "scores 42 out of 100" is the phrasing a
# model actually reaches for, and a pattern that only caught the noun would have
# let the one sentence this guard exists to stop straight through.
#
# BARE "rate" AND "rates" ARE DELIBERATELY NOT MATCHED. They were, and on a real
# ad this fired on "at a standard reading rate of 4.0 words per second" and
# published "at a standard reading [score removed].0 words per second" to a
# client - mangling a sentence that was correctly quoting a MEASURED value.
# A rate is a frequency, not a verdict: reading rate, cut rate, click rate,
# frame rate. `rated` and `rating` carry the judgement and are still caught.
#
# The decimal group matters for the same reason - without it the pattern ate
# "4" and left an orphaned ".0" behind.
SCORE_PATTERN = re.compile(
    r"\b(?:scor(?:e|es|ed|ing)|rat(?:ed|ing))\b\s*(?:of|at|it|this|:|=|is)?\s*"
    r"\d+(?:\.\d+)?(?:\s*(?:/|out of)\s*\d+)?",
    re.IGNORECASE)


def validate(parsed: dict, context: dict) -> dict:
    """Enforce in code what the prompt asked for.

    A prompt is a request. These are the two rules the design rests on, so they
    are checked rather than trusted.
    """
    if not isinstance(parsed, dict):
        raise AnalyzerError(ANALYSIS_INVALID_OUTPUT, "response was not an object")

    # RULE 1: the model does not produce scores.
    rating = ((parsed.get("clarity") or {}).get("rating") or "").lower()
    if rating and rating not in RATING_LEVELS:
        raise AnalyzerError(
            ANALYSIS_INVALID_OUTPUT,
            f"clarity rating {rating!r} is not one of {RATING_LEVELS} - the "
            f"model may only produce ordinal ratings, never numbers")

    for trigger in parsed.get("triggers") or []:
        level = (trigger.get("rating") or "").lower()
        if level and level not in RATING_LEVELS:
            raise AnalyzerError(
                ANALYSIS_INVALID_OUTPUT,
                f"trigger {trigger.get('id')} was rated {level!r} - ordinal "
                f"ratings only")

    # RULE 2: the model writes about defects Python found, never its own.
    known = {d["defect_id"] for d in context.get("defects") or []}
    kept, dropped = [], []
    for item in parsed.get("recommendations") or []:
        if item.get("defect_id") in known:
            kept.append(item)
        else:
            dropped.append(item.get("defect_id") or item.get("title", "?"))
    if dropped:
        logger.warning("dropped %d invented recommendation(s): %s",
                       len(dropped), dropped[:4])
    parsed["recommendations"] = kept
    parsed["dropped_invented_recommendations"] = dropped

    # A score smuggled into prose is still a score.
    for item in kept:
        for field in ("title", "why", "fix"):
            value = item.get(field) or ""
            if SCORE_PATTERN.search(value):
                item[field] = SCORE_PATTERN.sub("[score removed]", value)
                logger.warning("stripped an invented score from %s.%s",
                               item.get("defect_id"), field)

    parsed["prompt_version"] = PROMPT_VERSION
    return parsed


def unavailable(reason: str, message: str = "") -> dict:
    """An interpretation that did not happen, stated rather than faked."""
    return {"available": False, "reason": reason, "message": message,
            "summary": "", "recommendations": [], "triggers": [],
            "prompt_version": PROMPT_VERSION}
