"""Resolving a brand's Brand Brain, and grading how much of it there is.

WHY THIS IS NOT A BOOLEAN
    On live data the Brand Brain is not "present or absent". One brand has all
    nine answers, another has one, and fourteen of twenty brands have none at
    all. A coach that treats all three the same either invents a brand voice it
    was never told, or refuses to help a user whose only sin is incomplete
    onboarding. So completeness is graded, and the grade changes what the coach
    is allowed to say:

      tier A  >= 7 of 9 answers   full brand-grounded coaching
      tier B  3-6 answers         coach on what is known, NAME what is missing
      tier C  0-2 or no document  craft mode: copywriting and performance only

RESOLUTION IS BY ID, NEVER BY NAME
    Several brands share a display name in live data ("Lawttorney" appears four
    times, only one of which has a Brand Brain). The only correct path is
    brands.brand_brain_id -> brand_brains._id. Matching on name would coach a
    brand against a different brand's voice.

A NOTE ON TRUST
    A resolved Brand Brain is not necessarily a correct one - at least one in
    production describes a different company's customers entirely. Tiering
    measures how COMPLETE a document is, not how TRUE it is. Where the Brand
    Brain and real performance data disagree, performance wins; that rule lives
    in the prompt layer, not here.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("script_lab_coach")

# The nine questionnaire answers, in the order the onboarding asks them.
ANSWER_FIELDS = ("businessType", "idealCustomer", "brandVoice", "language",
                 "trafficChannels", "salesCycle", "competitors", "marketingGoal",
                 "journey")

# Business context is supporting detail, not part of the grade: a brand can be
# perfectly coachable without an ad budget on file.
CONTEXT_FIELDS = ("businessType", "industry", "brandName", "website",
                  "audienceShort", "channels", "adBudget")

TIER_A_MIN = 7
TIER_B_MIN = 3

# What the coach may lean on at each tier. `brand_fit` at tier C is answerable
# only as a refusal, which is itself a useful answer.
FULL_BRAND_FIT = "A"


class BrandBrain:
    """A resolved Brand Brain and how much of it there is.

    `used` is False at tier C: there is nothing to ground a voice claim in, and
    the prompt layer switches to craft mode."""

    def __init__(self, *, answers: Optional[dict] = None, context: Optional[dict] = None,
                 brand_brain_id: Optional[str] = None, brand_name: Optional[str] = None):
        self.brand_brain_id = brand_brain_id
        self.brand_name = brand_name
        self.answers = {k: v for k, v in (answers or {}).items() if _filled(v)}
        self.context = {k: v for k, v in (context or {}).items() if _filled(v)}
        self.missing = [f for f in ANSWER_FIELDS if f not in self.answers]
        self.filled = len(ANSWER_FIELDS) - len(self.missing)

    @property
    def tier(self) -> str:
        if self.filled >= TIER_A_MIN:
            return "A"
        if self.filled >= TIER_B_MIN:
            return "B"
        return "C"

    @property
    def used(self) -> bool:
        return self.tier != "C"

    @property
    def can_judge_brand_fit(self) -> bool:
        """Voice, persona and positioning claims need most of the document.

        Tier B is allowed to judge fit but must say what it could not check;
        tier C may not judge it at all."""
        return self.tier in ("A", "B")

    def as_dict(self) -> dict:
        """The `brand_brain` block of the API response."""
        return {"tier": self.tier, "used": self.used, "missing": list(self.missing)}

    def prompt_block(self) -> str:
        """The Brand Brain as the model sees it: filled answers only, plus an
        explicit list of what is unknown.

        Naming the gaps is deliberate. A model shown nine fields with three
        blanks will cheerfully fill them in; a model told "you do not know the
        competitors" will say so instead."""
        if not self.answers and not self.context:
            return ("(no Brand Brain on file for this brand - you know nothing about "
                    "its voice, persona, offer or funnel)")
        lines = [f"- {k}: {_render(v)}" for k, v in self.answers.items()]
        lines += [f"- {k}: {_render(v)}" for k, v in self.context.items()
                  if k not in self.answers]
        if self.missing:
            lines.append(f"- NOT ON FILE (do not guess at these): {', '.join(self.missing)}")
        return "\n".join(lines)


def _filled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _render(value) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value)


EMPTY = BrandBrain()


def loader(collection):
    """brand_brain_id -> BrandBrain, from the Mongo collection app.py holds.

    A read failure degrades to tier C rather than failing the turn: a coach that
    cannot reach Mongo can still coach on craft, and saying "no Brand Brain
    connected" is a better outcome than an error in the panel."""
    async def load(brand_brain_id: Optional[str], brand_name: Optional[str] = None) -> BrandBrain:
        if collection is None or not brand_brain_id:
            return BrandBrain(brand_name=brand_name)
        try:
            doc = await collection.find_one({"_id": brand_brain_id})
        except Exception:  # noqa: BLE001 - never block coaching on Mongo
            logger.warning("could not read brand_brain %s", brand_brain_id)
            return BrandBrain(brand_name=brand_name)
        if not doc:
            return BrandBrain(brand_name=brand_name)
        return BrandBrain(answers=doc.get("answers") or {},
                          context=doc.get("context") or {},
                          brand_brain_id=brand_brain_id,
                          brand_name=brand_name)
    return load
