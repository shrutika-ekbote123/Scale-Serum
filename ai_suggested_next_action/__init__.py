"""AI Suggested Next Action - what a sales rep should do next with one lead.

The card on the Lead Journey page has three parts, and so does the response:

    urgency         - how soon to act (HIGH / MEDIUM / LOW URGENCY)
    recommendation  - what the sales team should do
    reason          - why, citing the lead's own touchpoints

DIVISION OF LABOUR (a correctness rule, not a style preference)
  * rules.py DECIDES the action and the urgency, deterministically, from the
    lead's touchpoints, payments, calls and WhatsApp state. Same inputs, same
    decision, every time.
  * writer.py only WORDS that decision (Gemini). It may not change the action or
    the urgency, and any number it writes must appear in the facts it was given;
    otherwise the fixed template wording from next_action_framework.json is used.

Postgres is read-only here. Cached wording and per-brand tier overrides live in
this service's own MongoDB collections (store.py).
"""
from .framework import load_framework  # noqa: F401
from .rules import decide  # noqa: F401
from .service import (  # noqa: F401
    UNAVAILABLE_DB_ERROR,
    UNAVAILABLE_LEAD_NOT_FOUND,
    SuggestDeps,
    suggest_next_action,
)
from .store import NextActionStore  # noqa: F401
from .tiers import classify_amount, infer_tiers, validate_override  # noqa: F401
from .writer import PROMPT_VERSION  # noqa: F401

__all__ = [
    "load_framework", "decide", "suggest_next_action", "SuggestDeps",
    "NextActionStore", "infer_tiers", "classify_amount", "validate_override",
    "PROMPT_VERSION", "UNAVAILABLE_LEAD_NOT_FOUND", "UNAVAILABLE_DB_ERROR",
]
