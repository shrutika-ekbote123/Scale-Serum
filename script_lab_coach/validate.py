"""The gate a turn must pass before a user can read it.

Same checks the evaluation harness scores with - both call grounding.py - so
what is measured in evals and what is enforced in production cannot drift.

REJECTION IS CHEAP, A WRONG ANSWER IS NOT
    A rejected turn costs one repair call, and a repaired one costs nothing more
    than latency. An invented CTR costs a marketer a budget decision made on a
    number nobody measured. The gate is therefore strict, and everything it
    cannot verify it refuses to let through.
"""
from __future__ import annotations

import re
from typing import Optional

from . import grounding

MAX_TEXT = 4000

# Every reason this module can return. The caller distinguishes "the gate
# rejected an answer" (the answer was wrong) from "there was no answer" (the
# model was unreachable), and reports them differently.
REASONS = ("empty", "too_long", "placeholder", "unsupported_number", "foreign_brand",
           "defended_failed_review", "unbacked_performance", "unbacked_comparison",
           "unbacked_brand_claim")

# CLAIMS MADE IN WORDS
#
# The number gate catches an invented CTR. It does not catch "this ad is doing
# well", said about an ad whose delivery was never loaded - a claim with no
# figure in it, and no data behind it either. These patterns catch the sentence
# shapes that assert something the turn has no basis for.
#
# Each is written to need a VERB of assertion, not just a topic word: the coach
# must stay free to say "I can't tell you how it is performing", which mentions
# performance while claiming nothing.
_PERFORMANCE_CLAIM = re.compile(
    r"\b(?:it|this ad|the ad|your ad|performance|delivery)\s+(?:is|was|has been|looks|"
    r"seems|appears)\s+(?:currently\s+)?(?:really\s+|very\s+|quite\s+)?"
    r"(?:performing|delivering|doing|running)\b"
    r"|\b(?:is|was)\s+(?:performing|delivering|converting)\s+(?:well|badly|poorly|"
    r"strongly|above|below)\b"
    r"|\byour (?:ctr|cpm|cpc|cpl|spend|impressions|clicks) (?:is|was|are|were)\b",
    re.IGNORECASE)

_COMPARISON_CLAIM = re.compile(
    r"\b(?:better|worse|stronger|weaker|higher|lower|improved|up|down)\s+than\s+"
    r"(?:your|the|it)?\s*(?:last|previous|earlier|first)\s+(?:version|test|script|round)\b"
    r"|\b(?:this|the new) version (?:is|scores?)\s+(?:better|worse|higher|lower)\b",
    re.IGNORECASE)

_BRAND_CLAIM = re.compile(
    r"\byour brand (?:voice|tone|persona) (?:is|sounds|tends)\b"
    r"|\b(?:this|it) (?:matches|fits|is on)\s+(?:your |the )?brand\b"
    r"|\byour (?:ideal customer|audience|persona) (?:is|are)\b",
    re.IGNORECASE)

# A sentence that denies, hedges or asks is not an assertion. Without this the
# gate fires on the coach doing exactly what it is supposed to do - "I can't
# tell you how it is performing" contains "it is performing" - and the first
# version of these checks rejected the service's own fallback wording.
_NOT_ASSERTING = re.compile(
    r"\b(?:can't|cannot|can not|don't|do not|doesn't|does not|won't|will not|no|not|"
    r"never|unable|without|unknown|yet to|too early|if|once|when|would|could|should)\b"
    r"|\?",
    re.IGNORECASE)

_SENTENCE = re.compile(r"[^.!?\n]+[.!?\n]?")


def _asserts(pattern: re.Pattern, text: str) -> bool:
    """True when some sentence makes this claim outright.

    Per sentence, because a turn can legitimately say "I can't see how it is
    performing. The script itself is weak." - the first sentence denies, the
    second asserts something else entirely."""
    for sentence in _SENTENCE.findall(text or ""):
        if pattern.search(sentence) and not _NOT_ASSERTING.search(sentence):
            return True
    return False


def allowed_numbers(ctx) -> set:
    """Every figure this answer may contain.

    The computed facts, plus numbers already visible in what the model was
    shown: the ad name, the script, the review's own comments and improvements.
    Quoting the material back is not inventing."""
    allowed = {str(n) for n in ctx.allowed_numbers()}
    allowed |= {u.lstrip("/") for u in {f.unit for f in ctx.facts} if u.startswith("/")}
    # Facts whose VALUE is text still carry figures - "2026-06-27" is three of
    # them. Without this, a coach correctly saying when an ad last ran is
    # rejected for citing the date it was given. Three of the first six live
    # rejections were exactly that.
    for fact in ctx.facts:
        if isinstance(fact.value, str):
            allowed |= grounding.numbers(fact.value)
    test = ctx.test
    allowed |= grounding.numbers(str(test.get("ad_name") or ""))
    allowed |= grounding.numbers(str(test.get("label") or ""))
    allowed |= grounding.numbers(test.get("script_text") or "")
    allowed |= grounding.numbers(str(test.get("ai_verdict") or ""))
    for section in test.get("section_breakdown") or []:
        allowed |= grounding.numbers(str(section.get("comment") or ""))
    for improvement in test.get("improvements") or []:
        allowed |= grounding.numbers(" ".join(str(v) for v in improvement.values()))
    return allowed


def check(answer: dict, ctx, *, other_brands=()) -> Optional[str]:
    """None when the turn may be shown, otherwise a short reason code.

    The reason is recorded on the turn and in monitoring, so a rise in any one
    of these is visible before users complain about it."""
    text = (answer or {}).get("text") or ""
    rewrite = (answer or {}).get("suggested_rewrite") or ""
    whole = f"{text}\n{rewrite}"

    if not text.strip():
        return "empty"
    if len(text) > MAX_TEXT:
        return "too_long"

    if grounding.placeholders(whole):
        return "placeholder"

    # A rewrite is ad copy a human may paste into a live campaign, so it is held
    # to the same standard as the prose around it.
    if grounding.unsupported_numbers(whole, allowed_numbers(ctx)):
        return "unsupported_number"

    if grounding.foreign_brands(whole, own=ctx.brand.get("name"), others=other_brands):
        return "foreign_brand"

    # --- claims made in words, with no data behind them ---------------------
    # Checked after the number gate, because a claim with an invented figure in
    # it is better reported as an invented figure.
    if not ctx.has_performance and _asserts(_PERFORMANCE_CLAIM, whole):
        return "unbacked_performance"
    if not ctx.has_history and _asserts(_COMPARISON_CLAIM, whole):
        return "unbacked_comparison"
    if not ctx.brain.can_judge_brand_fit and _asserts(_BRAND_CLAIM, whole):
        return "unbacked_brand_claim"

    # A review that never completed has no scores to discuss. An answer that
    # talks about them anyway is defending a placeholder.
    if ctx.review_failed and not answer.get("refused"):
        low = text.lower()
        if not any(p in low for p in ("did not complete", "didn't complete", "placeholder",
                                      "not a real score", "re-run", "rerun", "run it again")):
            return "defended_failed_review"

    return None
