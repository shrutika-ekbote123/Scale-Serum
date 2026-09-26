"""Build the production half of the golden set from real coach threads.

    python -m evals.extract            # rewrites evals/production.jsonl

WHAT THIS IS DRAWN FROM
    sl_script_lab_tests.coach_thread already holds real conversations with the
    coach that ships today: 50 user turns across 196 tests. That is the only
    honest source of "what do people actually ask the Creative Coach", and it
    costs nothing to mine.

    Those 50 turns contain 12 DISTINCT questions. Volume is not variety, so the
    generated file is deduplicated - one case per distinct question, carrying
    how often it was really asked - and the hand-written adversarial.jsonl
    covers what users have not asked yet but will.

LABELLING
    Intent labels are assigned here, by hand, in QUESTIONS below. The legacy
    intents stored alongside the answers (verdict, fix, hook, summary) are NOT
    trusted as ground truth: they come from the very system under evaluation,
    and at least one of them is plainly wrong - "can you give me caption ideas"
    is filed as `summary`.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "production.jsonl")

# The twelve real questions, hand-labelled. `must` names fact keys the answer
# has to engage with; `refuse` marks questions the coach must decline rather
# than answer with a generic summary.
QUESTIONS = {
    "how do i improve the hook?": {
        "intent": "improve", "must": ["hook_score"], "refuse": False},
    "why this verdict?": {
        "intent": "explain", "must": ["overall_score", "verdict_band"], "refuse": False},
    "will this scale?": {
        "intent": "scale", "must": ["overall_score"], "refuse": False},
    "what should i change first?": {
        "intent": "prioritize", "must": [], "refuse": False},
    "how was the script": {
        "intent": "diagnose", "must": ["overall_score"], "refuse": False},
    "tell me the actual change to make for the hook": {
        # The user is pushing back on advice that was too vague to act on. A
        # correct answer contains a concrete rewrite, not more principles.
        "intent": "rewrite", "must": ["hook_score"], "refuse": False,
        "needs_rewrite": True},
    "can you give me caption ideas": {
        "intent": "rewrite", "must": [], "refuse": False, "needs_rewrite": True},
    "what do you mean when you say undefined": {
        # Asked because the coach printed a template placeholder at them. The
        # answer must not contain one itself.
        "intent": "explain", "must": [], "refuse": False},
    "can you actually speak to me?": {
        "intent": "explain", "must": [], "refuse": False},
    "hello": {"intent": "out_of_scope", "must": [], "refuse": True},
    "hi": {"intent": "out_of_scope", "must": [], "refuse": True},
    "h9i]": {"intent": "out_of_scope", "must": [], "refuse": True},
}


def _connect():
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO, ".env"))
    import psycopg

    return psycopg.connect(
        host=os.environ["DB_HOST"], port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"], connect_timeout=10)


def harvest() -> list:
    """One case per distinct question, attached to a real test it was asked on."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT id, brand_id, coach_thread
                       FROM sl_script_lab_tests
                       WHERE coach_thread IS NOT NULL""")
        rows = cur.fetchall()

    seen: dict = {}
    counts: Counter = Counter()
    for test_id, brand_id, thread in rows:
        for position, turn in enumerate(thread or []):
            if turn.get("role") != "user":
                continue
            question = (turn.get("text") or "").strip()
            key = question.lower()
            if not question:
                continue
            counts[key] += 1
            # Keep the FIRST test a question was asked on, with the thread that
            # preceded it: a replayed case should carry its real conversation.
            if key not in seen:
                seen[key] = {"test_id": str(test_id), "brand_id": str(brand_id),
                             "question": question,
                             "thread": (thread or [])[:position],
                             "baseline_answer": ((thread or [])[position + 1] or {}).get("text")
                             if position + 1 < len(thread or []) else None}

    cases = []
    for index, (key, found) in enumerate(sorted(seen.items()), 1):
        label = QUESTIONS.get(key)
        if label is None:
            # A question asked in production that nobody has labelled yet. It is
            # emitted, flagged, and skipped by the runner rather than silently
            # dropped - an unlabelled real question is a to-do, not a non-event.
            label = {"intent": None, "must": [], "refuse": False}
        cases.append({
            "id": f"prod-{index:03d}",
            "source": "production",
            "asked_times": counts[key],
            "test_id": found["test_id"],
            "brand_id": found["brand_id"],
            "question": found["question"],
            "thread": found["thread"],
            "expect_intent": label["intent"],
            "must_mention": label["must"],
            "must_refuse": label["refuse"],
            "needs_rewrite": label.get("needs_rewrite", False),
            "labelled": label["intent"] is not None,
            "baseline_answer": found["baseline_answer"],
        })
    return cases


def main() -> None:
    cases = harvest()
    with open(OUT, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    labelled = sum(1 for c in cases if c["labelled"])
    asked = sum(c["asked_times"] for c in cases)
    print(f"wrote {OUT}")
    print(f"  {len(cases)} distinct questions from {asked} real user turns "
          f"({labelled} labelled, {len(cases) - labelled} awaiting a label)")


if __name__ == "__main__":
    main()
