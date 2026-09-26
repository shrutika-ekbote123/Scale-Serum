"""Creative Coach - the chat panel beside the Script Lab score card.

WHAT THIS PACKAGE IS
    The context layer for the coach: it decides what the coach is allowed to
    know about a turn, computes every figure it is allowed to say, and says what
    it cannot answer. The prompt layer, validation and the HTTP routes are built
    on top of it (phases 3-5); nothing here calls a model.

DIVISION OF LABOUR (a correctness rule, not a style preference)
  * facts.py computes every number, from stored data, in Python.
  * the prompt layer only words them. A figure in an answer that is not in the
    fact set means the model invented it, and the turn is repaired or replaced.

  This is the same rule AI Briefings follows, for the same reason: it is the
  only thing that makes "the coach does not invent metrics" a guarantee rather
  than a hope.

STORAGE
    Nothing here writes. PostgreSQL is read-only for this service, and the
    conversation itself lives in sl_script_lab_tests.coach_thread, written by
    the main backend. The coach is stateless between turns: the thread is passed
    in and the new turn is handed back.

GRADED CONTEXT, NOT ASSUMED CONTEXT
    Two thirds of stored tests belong to brands with no Brand Brain, a third
    have a review that never completed, and some carry ad ids that match no ad.
    Each of those is a supported path with defined behaviour - see
    SCRIPT_LAB_COACH_API.md section 5 - rather than an edge case that produces a
    confident wrong answer.
"""
from .brand_brain import BrandBrain  # noqa: F401
from .brand_brain import loader as brand_brain_loader  # noqa: F401
from .context import (  # noqa: F401
    BrandMismatch,
    CoachContext,
    CreativeCoachContextBuilder,
    TestNotFound,
)
from .data import DataUnavailable  # noqa: F401
from .intents import NAMES as INTENT_NAMES  # noqa: F401
from .starters import build as build_starters  # noqa: F401

__all__ = ["BrandBrain", "brand_brain_loader", "BrandMismatch", "CoachContext",
           "CreativeCoachContextBuilder", "TestNotFound", "DataUnavailable",
           "INTENT_NAMES", "build_starters"]
