"""The opening turn and the three chips - computed, not generated.

WHY NO MODEL CALL
    The panel opens on every test. Generating an intro would cost a request, a
    second or two of latency and a slice of the budget to say something the
    score card already states exactly. Worse, a generated intro is one more
    place a wrong number can enter. Everything here is derived from the stored
    review, so it is instant, free, and cannot be wrong in a way the review is
    not already wrong.

    The chips are chosen the same way: the weakest section, the sharpest
    alignment gap, and whether the ad is live. A user who opens a failing script
    is offered "what do I fix first?", not "will this scale?".
"""
from __future__ import annotations

from typing import Optional

SCALE_READY = 70          # at or above this, scaling is a sensible question
REWRITE_BELOW = 50        # below this, the useful question is what to fix


def _weakest_section(test: dict) -> Optional[dict]:
    sections = [s for s in (test.get("section_breakdown") or [])
                if isinstance(s.get("score"), (int, float))]
    return min(sections, key=lambda s: s["score"]) if sections else None


def _weakest_alignment(test: dict) -> Optional[str]:
    """The first alignment dimension that came back Weak, if any."""
    alignment = test.get("context_alignment") or {}
    labels = {"brand_voice_fit": "brand voice", "funnel_stage_fit": "funnel stage",
              "marketing_angle_fit": "marketing angle"}
    for key, label in labels.items():
        if str(alignment.get(key) or "").strip().lower() == "weak":
            return label
    return None


def _ad_label(test: dict) -> str:
    name = (test.get("ad_name") or "").strip()
    if name:
        return f'"{name}"'
    label = (test.get("label") or "").strip()
    return f"version {label}" if label else "this script"


def intro_text(ctx) -> str:
    """One or two sentences: what was tested, where it stands, what is weakest."""
    test = ctx.test
    if ctx.review_failed:
        return (f"The AI review did not complete for {_ad_label(test)}, so the score you "
                "see is a placeholder rather than a real result. Re-run the test and I "
                "will break down what it found.")

    score = test.get("score")
    band = (test.get("verdict_band") or "").strip()
    head = f"Tested {_ad_label(test)} - score {score}/100"
    head += f", {band.lower()}." if band else "."

    weakest = _weakest_section(test)
    if weakest:
        head += (f" The weakest part is {weakest['section'].strip()} at "
                 f"{int(weakest['score'])}/10.")
    alignment = _weakest_alignment(test)
    if alignment:
        head += f" It also drifts from your {alignment}."

    return head + " Ask me why, or how to push the score higher."


def chips(ctx) -> list:
    """Three starters, chosen for this result rather than fixed copy."""
    test = ctx.test
    if ctx.review_failed:
        return [{"label": "Why did the review fail?", "intent": "explain"},
                {"label": "What should I check in the script?", "intent": "diagnose"}]

    out = []
    weakest = _weakest_section(test)
    if weakest:
        out.append({"label": f"Why is the {weakest['section'].strip()} only "
                             f"{int(weakest['score'])}/10?", "intent": "explain"})
    else:
        out.append({"label": "Why did I get this score?", "intent": "explain"})

    out.append({"label": "What should I change first?", "intent": "prioritize"})

    score = test.get("score") or 0
    if score < REWRITE_BELOW:
        # Nothing about scaling is worth asking yet; a rewrite is.
        out.append({"label": "Rewrite the hook for me", "intent": "rewrite"})
    elif ctx.has_performance:
        out.append({"label": "Will this scale?", "intent": "scale"})
    elif ctx.brain.can_judge_brand_fit:
        out.append({"label": "Does this sound like us?", "intent": "brand_fit"})
    else:
        out.append({"label": "How do I make this stronger?", "intent": "improve"})
    return out


def build(ctx) -> dict:
    """The `/coach/starters` response."""
    return {
        "intro": {"role": "coach", "intent": "intro", "text": intro_text(ctx),
                  "fallback": False},
        "starters": chips(ctx),
        "brand_brain": ctx.brain.as_dict(),
        "capabilities": ctx.capabilities(),
    }
