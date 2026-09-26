"""Every number the coach is allowed to utter, computed in Python.

THE RULE THIS FILE EXISTS TO ENFORCE
    The model writes sentences. It does not produce figures. Each fact below is
    computed here from stored data, handed to the prompt as a closed list, and
    checked against the answer afterwards. A figure in the answer that is not in
    this list means the model invented it, and the turn is repaired or replaced.

    This is the same division of labour that makes AI Briefings trustworthy, and
    it is the only reason the coach cannot quietly invent a CTR that nobody
    measured.

WHAT IS DELIBERATELY NOT A FACT
    Per-ad revenue and ROAS. Payments in scrumdb are written by an attribution
    backfill and carry no ad id, so any per-ad return figure would be fiction.
    The coach can speak about delivery - clicks, cost, leads - and must decline
    revenue attribution. Leaving it out of the fact set is what makes that
    refusal automatic rather than a matter of prompt wording.
"""
from __future__ import annotations

from typing import Any, Iterable, NamedTuple, Optional

# Section names the reviewer emits, in the order it scores them.
SECTION_ORDER = ("Hook", "Problem / Tension", "Solution / Offer",
                 "Social Proof / Credibility", "Call to Action", "Pacing & Tightness")


class Fact(NamedTuple):
    key: str
    value: Any
    unit: str = ""
    label: str = ""

    def render(self) -> str:
        value = self.value
        if isinstance(value, float):
            value = f"{value:g}"
        return f"- {self.label or self.key}: {value}{self.unit}"

    def as_dict(self) -> dict:
        out = {"key": self.key, "value": self.value}
        if self.unit:
            out["unit"] = self.unit
        return out


def _num(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slug(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in name.strip()]
    return "".join(keep).strip("_").replace("__", "_")


def from_test(test: dict) -> list:
    """The score card as facts.

    A test that fell back (ai_fallback) contributes NO scores. Its stored 50s
    are placeholders from a review that never completed, and a coach that quotes
    them is defending a number that means nothing. The fallback flag itself is
    the fact, and the prompt layer turns it into an honest answer."""
    if test.get("ai_fallback"):
        return [Fact("review_completed", False, label="Did the AI review complete")]

    facts = [Fact("overall_score", test.get("score"), "/100", "Overall score")]
    for key, label in (("attention", "Attention"), ("resonance", "Resonance"),
                       ("conversion", "Conversion"), ("creative", "Creative"),
                       ("marketing_angle_execution", "Marketing angle execution")):
        if _num(test.get(key)) is not None:
            facts.append(Fact(key, test[key], "/100", label))

    for section in test.get("section_breakdown") or []:
        name = str(section.get("section") or "").strip()
        score = _num(section.get("score"))
        if name and score is not None:
            facts.append(Fact(f"{_slug(name)}_score", int(score), "/10", f"{name} section"))

    alignment = test.get("context_alignment") or {}
    for key, label in (("brand_voice_fit", "Brand voice fit"),
                       ("funnel_stage_fit", "Funnel stage fit"),
                       ("marketing_angle_fit", "Marketing angle fit")):
        if alignment.get(key):
            facts.append(Fact(key, alignment[key], "", label))

    angle = test.get("emotional_angle") or {}
    if angle.get("status"):
        facts.append(Fact("emotional_angle_status", angle["status"], "", "Angle verdict"))

    for key, label in (("verdict_band", "Verdict band"),
                       ("marketing_angle", "Chosen marketing angle"),
                       ("funnel_stage", "Chosen funnel stage")):
        if test.get(key):
            facts.append(Fact(key, test[key], "", label))

    facts.append(Fact("improvement_count", len(test.get("improvements") or []),
                      "", "Improvements the review listed"))
    return facts


def from_performance(perf: Optional[dict]) -> list:
    """Delivery for the published ad. Empty when the ad could not be verified."""
    if not perf:
        return []
    window = perf.get("window") or "30d"
    out = [Fact("days_with_data", perf["days_with_data"], "", "Days of delivery data")]
    if perf.get("last_date"):
        out.append(Fact("last_delivery_date", perf["last_date"], "",
                        "Last day this ad delivered"))
    if perf.get("stale"):
        # Named as a fact so the coach states it rather than implying the ad is
        # still running.
        out.append(Fact("currently_delivering", False, "", "Delivering in the last 30 days"))
    for key, unit, label in (
        ("impressions", "", "Impressions"),
        ("clicks", "", "Clicks"),
        ("leads", "", "Leads"),
        ("spend", "", "Spend"),
        ("ctr_pct", "%", "CTR"),
        ("cpm", "", "CPM"),
        ("cpc", "", "CPC"),
        ("cpl", "", "Cost per lead"),
    ):
        value = perf.get(key)
        if value is not None:
            scope = "lifetime" if window == "lifetime" else f"last {window}"
            out.append(Fact(f"{key}_{window}", value, unit, f"{label} ({scope})"))
    if perf.get("effective_status"):
        out.append(Fact("ad_status", perf["effective_status"], "", "Ad status on Meta"))
    return out


def from_versions(versions: Iterable[dict]) -> list:
    """Earlier scores for the same ad, so `compare` has a real baseline."""
    out = []
    for v in versions or []:
        if v.get("ai_fallback") or _num(v.get("score")) is None:
            continue  # a failed review is not a baseline
        label = v.get("label") or f"v{v.get('version')}"
        out.append(Fact(f"previous_score_{_slug(str(label))}", v["score"], "/100",
                        f"Previous version {label} score"))
    return out


def from_creatives(creatives: Optional[dict]) -> list:
    """The brand's own best and worst analysed creatives, as comparables."""
    out = []
    for item in (creatives or {}).get("winners", [])[:1]:
        if _num(item.get("score")) is not None:
            out.append(Fact("best_brand_creative_score", round(item["score"]), "/100",
                            "Best analysed creative for this brand"))
    for item in (creatives or {}).get("losers", [])[:1]:
        if _num(item.get("score")) is not None:
            out.append(Fact("worst_brand_creative_score", round(item["score"]), "/100",
                            "Worst analysed creative for this brand"))
    return out


def render(facts: Iterable[Fact]) -> str:
    """The fact block the prompt receives."""
    lines = [f.render() for f in facts]
    return "\n".join(lines) if lines else "(no figures are available for this test)"


def allowed_numbers(facts: Iterable[Fact]) -> set:
    """Every numeric value the answer may contain.

    Percentages are admitted both as written and as their fractional form, since
    1.41% and 0.0141 are the same measurement and either may legitimately appear.
    Validation in a later phase compares extracted numbers against this set; it
    lives here so the allowed set can never drift from the facts that produced it."""
    allowed = set()
    for fact in facts:
        value = _num(fact.value)
        if value is None:
            continue
        allowed.add(round(value, 4))
        if fact.unit == "%":
            allowed.add(round(value / 100, 6))
        if float(value).is_integer():
            allowed.add(int(value))
    return allowed


def index(facts: Iterable[Fact]) -> dict:
    """key -> Fact, for the turn's `facts_used` block."""
    return {f.key: f for f in facts}
