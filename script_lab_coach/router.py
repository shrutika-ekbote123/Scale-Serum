"""Which question is this? Answered without a model wherever possible.

THREE ROUTES, CHEAPEST FIRST
  1. The caller said so. A chip click carries its intent, so most turns are
     routed for nothing. This is why the UI sends `intent` with a chip.
  2. Phrase match. The real questions people ask are short and repetitive -
     "why this verdict?", "will this scale?", "how do I improve the hook?"
     account for most of the traffic that exists. Matching them is a dictionary
     lookup, not an inference problem.
  3. The model, once, cheaply - only when 1 and 2 both fail.

WHY NOT JUST ASK THE MODEL EVERY TIME
    A classification call on every turn doubles the request count and adds a
    second of latency to questions a regex settles with certainty. It is also
    less predictable: the same question could route differently on two days,
    which makes the evaluation suite meaningless.

GREETINGS ARE NOT QUESTIONS
    "hello", "hi", "h9i]" - four of the twelve questions real users have asked
    are greetings or mistypes. The coach that ships today answers them with a
    full score summary. They route to `out_of_scope`, which replies in one line
    and invites a real question.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from . import intents as _intents

logger = logging.getLogger("script_lab_coach")

# The provider refuses a deadline under 10 seconds with a 400.
CLASSIFY_TIMEOUT_MS = int(os.environ.get("COACH_CLASSIFY_TIMEOUT_MS", 10_000))

# Ordered: the first pattern that matches wins, so put the specific before the
# general. Written against the questions really asked, not invented ones.
_PATTERNS = (
    # Off topic, checked FIRST. "Write me an email to my landlord" matches the
    # rewrite pattern on its verb alone, so subject beats verb here.
    ("out_of_scope", r"\b(landlord|deposit refund|capital of|weather|recipe|homework"
                     r"|who made you|which model (are|do)"
                     r"|stock price|translate this (?!script|ad|copy))\b"),
    # Asking to SEE the instructions is off topic. Merely saying "ignore your
    # instructions" is an injection attempt wrapped around a real question about
    # the script, and the user still deserves the real answer.
    ("out_of_scope", r"\b(repeat|show|print|reveal|what are) (me )?(the |your |full )*"
                     r"(system )?(prompt|instructions)\b"),
    ("out_of_scope", r"^\s*(hi+|hey|hello|yo|sup|test(ing)?|ok(ay)?|thanks|thank you|ty"
                     r"|hmm+|\W+)\s*[!.?]*\s*$"),
    # Four characters or fewer and not a word anyone meant: "h9i]", "asdf".
    ("out_of_scope", r"^\s*(?=\S*\d)(?=\S*[a-z])\S{1,5}\s*$"),
    # Questions about the coach itself, and pushing back on its answers.
    ("explain", r"\b(what do you mean|speak to me|are you (a )?(bot|real|ai)"
                r"|can you actually|i disagree|raise (it|the score)|change the score"
                r"|that.s (too )?(vague|generic))\b"),
    ("performance", r"\b(roas|revenue|ctr|cpm|cpc|cpl|impressions?|clicks?|spend"
                    r"|how (is|are|did) (it|this|the ad)|performing|performance)\b"),
    ("scale", r"\b(scale|scaling|more budget|increase budget|push (more )?spend"
              r"|should i (run|spend|boost))\b"),
    ("compare", r"\b(compare|better than|versus|vs\.?|last version|previous version"
                r"|improved since|any better)\b"),
    ("rewrite", r"\b(rewrite|re-write|write (me|it|a)|give me (a|the|some)"
                r"|actual (change|copy|words|line)|caption ideas?|variations?"
                r"|kaise likh|likh do)\b"),
    # After `rewrite`, deliberately: "rewrite the opening for our ideal customer"
    # is a rewrite that happens to mention the audience, and the verb is what
    # the user wants acted on.
    ("brand_fit", r"\b(brand ?voice|sound like us|on ?brand|our voice|our tone"
                  r"|our (ideal )?(customer|audience)|fit our"
                  r"|who is the (target )?audience|which audience|target persona)\b"),
    ("prioritize", r"\b(change first|fix first|priority|prioriti[sz]e|what (should|do) i"
                   r" (change|fix|do) first|biggest (issue|problem|lift)|most important)\b"),
    ("improve", r"\b(improve|make (it|this) (better|stronger)|stronger|sharpen|punch"
                r"|better hook|fix the hook|improve karun|behtar)\b"),
    ("explain", r"\b(why|explain|what does .* mean|how come|reason|justify"
                r"|break ?(it |them |my )?down|breakdown|verdict)\b"),
    ("diagnose", r"\b(what.s wrong|whats wrong|problem|issues?|weak|bad|how was"
                 r"|review|assess|diagnos)\b"),
)

_COMPILED = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _PATTERNS)

# Asked of the model only when nothing above matched. Kept to one line of
# output: the classification is a label, not an essay.
CLASSIFY_INSTRUCTION = (
    "You label a marketer's question about ONE ad script they just had reviewed.\n"
    "Reply with exactly one of these labels and nothing else:\n"
    + ", ".join(_intents.NAMES) + "\n"
    "Use out_of_scope when the question is not about this script, its score, its "
    "copy, its brand fit or its ad performance."
)


# A continuation carries no subject of its own: "ok and then?", "go on", "more".
# It means "keep going with what we were just doing", so it inherits the intent
# of the last coach turn. Routing it afresh restarts the conversation, which is
# precisely the complaint users had about the coach that ships today.
_CONTINUATION = re.compile(
    r"^\s*(ok(ay)?|and)?[\s,]*(and )?(then|next|what else|go on|continue|more"
    r"|carry on|after that)\s*\??\s*$", re.IGNORECASE)


def previous_intent(thread: Optional[list]) -> Optional[str]:
    for turn in reversed(thread or []):
        if turn.get("role") == "coach" and turn.get("intent") in _intents.BY_NAME:
            return turn["intent"]
    return None


def match(message: str) -> Optional[str]:
    """The intent a phrase match is certain of, or None."""
    text = (message or "").strip()
    if not text:
        return "out_of_scope"
    for name, pattern in _COMPILED:
        if pattern.search(text):
            return name
    return None


async def route(message: str, *, supplied: Optional[str] = None,
                client=None, model: Optional[str] = None,
                thread: Optional[list] = None) -> tuple:
    """(intent_name, how) where `how` is "supplied" | "matched" | "continued" |
    "model" | "default".

    `how` is recorded on the turn so the monitoring can show what fraction of
    traffic is being settled for free."""
    if supplied and supplied in _intents.BY_NAME:
        return supplied, "supplied"

    if _CONTINUATION.match(message or ""):
        carried = previous_intent(thread)
        if carried:
            return carried, "continued"

    matched = match(message)
    if matched:
        return matched, "matched"

    if client is None or not model:
        return _intents.DEFAULT.name, "default"

    try:
        from google.genai import types

        settings = dict(
            system_instruction=CLASSIFY_INSTRUCTION,
            temperature=0,
            # The budget covers thinking as well as the label. Eight tokens left
            # nothing for the word itself once the model had thought, so every
            # classification came back empty and every free-text question was
            # filed as the default.
            max_output_tokens=256,
            # The provider rejects anything under 10s outright: a shorter
            # deadline is a 400, not a fast failure.
            http_options=types.HttpOptions(timeout=CLASSIFY_TIMEOUT_MS),
        )
        try:
            settings["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:  # noqa: BLE001 - models and SDKs without thinking
            pass

        response = await client.aio.models.generate_content(
            model=model, contents=f"Question: {message.strip()[:400]}",
            config=types.GenerateContentConfig(**settings))
        words = (response.text or "").strip().lower().replace("\n", " ").split()
        for word in words:
            label = word.strip(".,'\"`*")
            if label in _intents.BY_NAME:
                return label, "model"
    except Exception as err:  # noqa: BLE001 - classification must never fail a turn
        logger.warning("coach intent classification failed: %s", err)
    return _intents.DEFAULT.name, "default"
