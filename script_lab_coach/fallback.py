"""What the coach says when the model cannot be used.

An outage, a timeout, or an answer that failed the gate twice. The panel still
has to say something useful, so these are written from the score card in Python:
no call, no cost, no possibility of an invented figure.

They are deliberately plain. A fallback that tries to sound like the real coach
invites the reader to trust it as much; one that states what is known and stops
is honest about being a stand-in.
"""
from __future__ import annotations

from typing import Optional


def _weakest(ctx) -> Optional[dict]:
    sections = [s for s in (ctx.test.get("section_breakdown") or [])
                if isinstance(s.get("score"), (int, float))]
    return min(sections, key=lambda s: s["score"]) if sections else None


def _top_improvement(ctx) -> Optional[dict]:
    items = ctx.test.get("improvements") or []
    return items[0] if items else None


def _score_line(ctx) -> str:
    test = ctx.test
    band = (test.get("verdict_band") or "").strip()
    line = f"This script scored {test.get('score')}/100"
    return f"{line} - {band.lower()}." if band else f"{line}."


def answer(ctx, question: str = "") -> dict:
    """A deterministic turn. Same shape the model would have returned."""
    if ctx.review_failed:
        return {
            "text": ("The automated review did not complete for this script, so there is "
                     "no real result to explain - the score stored against it is a "
                     "placeholder. Re-run the test and I can break down what it found."),
            "suggested_rewrite": None,
            "follow_ups": ["Re-run the test"],
            "confidence": "low",
            "refused": True,
        }

    intent = ctx.intent.name

    if intent == "out_of_scope":
        return {
            "text": ("I only coach the ad script you have open. Ask me why it scored the "
                     "way it did, what to change first, or for a rewrite."),
            "suggested_rewrite": None,
            "follow_ups": ["What should I change first?", "Why did I get this score?"],
            "confidence": "high",
            "refused": True,
        }

    if intent == "performance" and ctx.has_performance:
        # The stand-in has the delivery figures too - they are facts, computed in
        # Python - so it can answer the question rather than deflecting to the
        # score card. It says plainly when the ad stopped running, because a
        # reader who thinks a paused ad is live misreads every figure after it.
        perf = ctx.performance
        window = "over its lifetime" if perf.get("stale") else "in the last 30 days"
        bits = [f"{perf['impressions']} impressions" if perf.get("impressions") else "",
                f"{perf['clicks']} clicks" if perf.get("clicks") else "",
                f"a CTR of {perf['ctr_pct']}%" if perf.get("ctr_pct") is not None else "",
                f"{perf['leads']} leads" if perf.get("leads") else ""]
        line = ", ".join(b for b in bits if b) or "no recorded delivery"
        text = f"This ad took {line} {window}."
        if perf.get("stale"):
            text += (f" It is not currently delivering - it last ran on "
                     f"{perf.get('last_date')}, so treat these as history rather than "
                     "a live read.")
        text += (" " + _score_line(ctx)
                 + " I couldn't reach the AI coach just now, so this is straight from "
                   "the stored figures.")
        return {"text": text, "suggested_rewrite": None,
                "follow_ups": ["What should I change first?", "Will this scale?"],
                "confidence": "low", "refused": False}

    if intent in ("performance", "scale") and not ctx.has_performance:
        return {
            "text": ("I don't have delivery data for this ad, so I can't tell you how it "
                     "is performing - that arrives once the ad has run. " + _score_line(ctx)
                     + " I can still coach the script itself."),
            "suggested_rewrite": None,
            "follow_ups": ["What should I change first?", "How do I make this stronger?"],
            "confidence": "low",
            "refused": True,
        }

    if intent == "compare" and not ctx.has_history:
        best = ((ctx.creatives or {}).get("winners") or [None])[0]
        text = ("This is the first tested version of this ad, so there is nothing to "
                "compare it against yet. " + _score_line(ctx))
        if best and best.get("score") is not None:
            # Not the comparison that was asked for, but a real one: the brand's
            # own best performer is a better yardstick than nothing.
            text += (f" For a yardstick, this brand's best analysed creative scored "
                     f"{round(best['score'])}/100.")
        text += " Test a revision and I can show you exactly what moved."
        return {
            "text": text,
            "suggested_rewrite": None,
            "follow_ups": ["What should I change first?"],
            "confidence": "low",
            "refused": True,
        }

    if intent == "brand_fit" and not ctx.brain.can_judge_brand_fit:
        return {
            "text": ("No Brand Brain is connected for this brand, so I can't judge whether "
                     "this sounds like you - I'd only be guessing at your voice. "
                     + _score_line(ctx)
                     + " Connect the Brand Brain and I can judge voice and persona too."),
            "suggested_rewrite": None,
            "follow_ups": ["What should I change first?"],
            "confidence": "low",
            "refused": True,
        }

    # The general case: the score card, stated plainly.
    parts = [_score_line(ctx)]
    weakest = _weakest(ctx)
    if weakest:
        comment = str(weakest.get("comment") or "").strip()
        parts.append(f"The weakest section is {weakest['section']} at "
                     f"{int(weakest['score'])}/10."
                     + (f" The review's note: {comment}" if comment else ""))
    top = _top_improvement(ctx)
    if top and top.get("title"):
        parts.append(f"The review's first fix: {top['title']}."
                     + (f" {top.get('why_it_matters')}" if top.get("why_it_matters") else ""))
    parts.append("I couldn't reach the AI coach just now, so this is straight from the "
                 "stored review. Ask again in a moment for the full reasoning.")

    return {
        "text": " ".join(p for p in parts if p),
        "suggested_rewrite": (top or {}).get("suggested_rewrite") or None,
        "follow_ups": ["What should I change first?", "How do I make this stronger?"],
        "confidence": "low",
        "refused": False,
    }
