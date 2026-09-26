"""What the coach can be asked, and what each question needs loaded.

THE POINT OF THIS FILE IS COST AND LATENCY, NOT TAXONOMY.
"Why is my hook weak?" is answerable from the stored score card alone - no
PostgreSQL read, no corpus, no history. "How is this ad doing?" needs ad-level
insights. Loading everything for every turn would make the cheap questions -
which are most of them - as slow and as expensive as the dear ones.

Each intent therefore declares the retrieval tiers it needs, and the context
builder loads nothing else. See context.py for what a tier costs.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

# Tier names, used as keys throughout. CORE (the test row) and BRAND (the Brand
# Brain) are not listed because they are always loaded.
PERFORMANCE = "performance"   # ad-level meta_insights for source_ad_id
CORPUS = "corpus"             # this brand's past creatives + competitor hooks
HISTORY = "history"           # earlier tested versions of the same ad

TIERS = (PERFORMANCE, CORPUS, HISTORY)


class Intent(NamedTuple):
    name: str
    needs: tuple           # which retrieval tiers to load
    wants_rewrite: bool    # whether a suggested_rewrite is expected in the turn
    question: str          # the user question this answers, for the prompt header


_INTENTS = (
    Intent("explain", (), False, "Why did I get this score?"),
    Intent("diagnose", (), False, "What is wrong with it?"),
    Intent("prioritize", (), False, "What should I change first?"),
    Intent("improve", (CORPUS,), True, "How do I make this better?"),
    Intent("rewrite", (CORPUS,), True, "Rewrite this part for me"),
    # CORPUS as well as HISTORY, deliberately. No test in the system has an
    # earlier version - `version` is always 1 and every test takes a fresh
    # ad_number - so a compare that only reads history is an intent that can
    # never answer. With the corpus loaded it can say what this script does
    # differently from the brand's own best performer, which is the question
    # behind the question.
    Intent("compare", (HISTORY, PERFORMANCE, CORPUS), False,
           "Is this better than the last version?"),
    Intent("performance", (PERFORMANCE,), False, "How is this ad actually doing?"),
    Intent("scale", (PERFORMANCE,), False, "Should I put more budget behind it?"),
    Intent("brand_fit", (), False, "Does this sound like us?"),
    Intent("out_of_scope", (), False, "(not about this script)"),
)

BY_NAME = {i.name: i for i in _INTENTS}
NAMES = tuple(i.name for i in _INTENTS)

# `intro` is produced by starters.py, never by the chat endpoint. It is listed
# here only so a stored thread containing it can be read back without error.
THREAD_ONLY = ("intro",)

DEFAULT = BY_NAME["diagnose"]


def get(name: Optional[str] = None) -> Intent:
    """The intent by name, or the default when it is unknown or absent.

    Unknown names are rejected at the API boundary (422); falling back here as
    well means an intent stored by an older version of the service can still be
    replayed rather than crashing a thread."""
    return BY_NAME.get(name or "", DEFAULT)


def needs(name: str, tier: str) -> bool:
    return tier in get(name).needs
