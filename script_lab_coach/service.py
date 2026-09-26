"""One coach turn, end to end.

    route the intent  ->  build the context  ->  word it  ->  gate it  ->  price it

Nothing here decides what is true; context.py and facts.py did that. This module
decides what happens when a step fails, and the answer is always the same: the
panel gets a usable turn, marked for what it is.

    fallback: true    no model answer at all (outage, timeout, gate failed twice)
    grounded: false   a model answer was produced and rejected; you are reading
                      the deterministic stand-in instead
    confidence: low   the answer rests on thin, stale or missing data

STATELESS ON PURPOSE
    The thread arrives in the request and the new turn is handed back. The
    conversation lives in sl_script_lab_tests.coach_thread, written by the main
    backend, because this service does not write PostgreSQL. The only thing it
    records is what the turn cost, in Mongo, so the usage report has something
    to count.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from . import fallback as _fallback
from . import intents as _intents
from . import router as _router
from . import starters as _starters
from . import validate as _validate
from . import writer as _writer

logger = logging.getLogger("script_lab_coach")

MAX_QUESTION = 1000
MAX_THREAD_TURNS = 40


class ThreadTooLong(Exception):
    """409. The conversation has outgrown what one context can carry."""


class CoachDeps:
    """What a turn needs from the process around it.

    Injected rather than imported so the service can be tested without a Gemini
    client, a database or a running app."""

    def __init__(self, *, builder, llm_client=None, llm_model: Optional[str] = None,
                 price_usage=None, usage_store=None, other_brands=()):
        self.builder = builder
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.price_usage = price_usage
        self.usage_store = usage_store
        # Brand names that must never appear in this brand's coaching.
        self.other_brands = tuple(other_brands)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


async def starters(deps: CoachDeps, *, test_id: str, brand_id: Optional[str] = None) -> dict:
    """The opening turn and the chips. No model call, so nothing to fall back to."""
    ctx = await deps.builder.build(test_id=test_id, brand_id=brand_id,
                                   extra_tiers=(_intents.PERFORMANCE, _intents.HISTORY))
    return _starters.build(ctx)


async def chat(deps: CoachDeps, *, test_id: str, message: str,
               intent: Optional[str] = None, thread: Optional[list] = None,
               brand_id: Optional[str] = None, user_id: Optional[str] = None,
               request_id: Optional[str] = None) -> dict:
    """One turn, ready to be appended to `coach_thread` and returned to the panel."""
    thread = thread or []
    if len([t for t in thread if t.get("role") == "user"]) >= MAX_THREAD_TURNS:
        raise ThreadTooLong(test_id)

    request_id = request_id or str(uuid.uuid4())
    question = (message or "").strip()[:MAX_QUESTION]

    intent_name, routed_by = await _router.route(
        question, supplied=intent, client=deps.llm_client, model=deps.llm_model,
        thread=thread)

    ctx = await deps.builder.build(test_id=test_id, intent_name=intent_name,
                                   brand_id=brand_id)

    def gate(answer, context):
        # The brands the context loaded from the database, plus anything the
        # deployment added. Reading them per brand means a brand added today is
        # protected today, without a redeploy.
        others = tuple(context.other_brands) + tuple(deps.other_brands)
        return _validate.check(answer, context, other_brands=others)

    written = await _writer.write(deps.llm_client, deps.llm_model, ctx, question,
                                  thread=thread, validate=gate)

    answer = written["answer"]
    grounded = True
    if answer is None:
        # Either the model could not be reached, or it produced something the
        # gate refused twice. Both mean the user gets the deterministic answer,
        # but they are different failures and are reported as such: `grounded`
        # is false only when a model answer existed and was rejected.
        grounded = written["reason"] not in _validate.REASONS
        answer = _fallback.answer(ctx, question)
        logger.info("coach fallback (%s) test=%s intent=%s", written["reason"],
                    test_id, intent_name)

    usage = written["meta"]["usage"]
    cost = deps.price_usage(usage, written["meta"]["attempts"]) if deps.price_usage else None

    turn = {
        # The three keys the stored thread already uses. Everything else is
        # additive, so an older reader still understands this turn.
        "role": "coach",
        "text": answer["text"],
        "intent": intent_name,

        "test_id": ctx.test["test_id"],
        "facts_used": ctx.facts_used(),
        "evidence": ctx.evidence(),
        "suggested_rewrite": answer.get("suggested_rewrite"),
        "follow_ups": answer.get("follow_ups") or [],
        "confidence": answer.get("confidence") or "medium",
        "refused": bool(answer.get("refused")),
        "brand_brain": ctx.brain.as_dict(),
        "capabilities": ctx.capabilities(),
        "grounded": grounded,
        "fallback": written["source"] != "llm",
        "fallback_reason": written["reason"],
        "routed_by": routed_by,
        "usage": usage,
        "cost": cost,
        "model": deps.llm_model if written["source"] == "llm" else None,
        "prompt_version": _writer.PROMPT_VERSION,
        "latency_ms": written["meta"]["latency_ms"],
        "request_id": request_id,
        "created_at": _now(),
    }

    if deps.usage_store is not None:
        try:
            await deps.usage_store.record(turn, ctx, user_id=user_id)
        except Exception:  # noqa: BLE001 - accounting must never fail a turn
            logger.warning("could not record coach usage for %s", test_id)

    return turn
