"""
Brand Brain layer - how well this lead fits the brand that owns it.

The Brand Brain lives in MongoDB (collection `brand_brains`), keyed by the
`brand_brain_id` stored on the Postgres `brands` row. It is what the brand told
us about itself during onboarding: who its ideal customer is, which channels it
buys, what language it sells in, and how long its sales cycle runs.

WHAT THIS LAYER IS
    A declared-intent prior. It reads the brand's own answers and asks whether
    THIS lead looks like the customer the brand said it wants, and whether the
    lead is still inside the window the brand said it closes in.

WHAT THIS LAYER IS NOT
    Trained. Nothing here is fitted to outcomes - the Brand Brain is free text a
    human typed, and only one of nineteen brands currently has one at all. Every
    weight is a documented prior in signal_config.json, applied as a bounded
    adjustment and reported separately from the calibrated model output.

    It is also deliberately conservative about double counting: f_seniority,
    f_email_class and f_locale are already inputs to the base model. This layer
    re-reads them only to test brand-specific fit, and never fires a factor when
    the brand has not actually declared the thing being tested. A brand that left
    a question blank produces `no_data`, which contributes exactly zero - not a
    penalty.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Optional

NO_BRAND_BRAIN = "no_brand_brain"

_LANGUAGE_PREFIX = {
    "english": "en", "hindi": "hi", "spanish": "es", "french": "fr",
    "german": "de", "portuguese": "pt", "arabic": "ar", "marathi": "mr",
    "tamil": "ta", "telugu": "te", "bengali": "bn", "gujarati": "gu",
}


def _text(*values: Any) -> str:
    """Flatten assorted answer fields into one lowercase haystack."""
    parts = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            parts.extend(str(x) for x in v if x)
        else:
            parts.append(str(v))
    return " \n ".join(parts).lower()


# --------------------------------------------------------------------------- parsing
def parse_sales_cycle_days(raw: Optional[str], cfg: dict) -> Optional[float]:
    """Turn a free-text sales cycle ('Same day', '1-4 weeks') into days.

    Returns None when the brand did not answer or answered non-committally -
    the caller must then skip the factor rather than assume a number.
    """
    if not raw:
        return None
    text = re.sub(r"\s+", " ", str(raw).strip().lower())
    if not text:
        return None

    for phrase, days in cfg["sales_cycle_phrases"].items():
        if phrase in text:
            return None if days is None else float(days)

    units = cfg["sales_cycle_units"]
    unit_alt = "|".join(sorted(units, key=len, reverse=True))
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:[-–—]|to|and)?\s*(\d+(?:\.\d+)?)?\s*\+?\s*"
        r"(" + unit_alt + r")s?",
        text)
    if not m:
        # No digits: catch the written forms brands actually type, e.g.
        # "less than a week", "about a month", "one quarter".
        m2 = re.search(r"\b(?:a|an|one)\s+(" + unit_alt + r")s?\b", text)
        return float(units[m2.group(1)]) if m2 else None
    low, high, unit = m.group(1), m.group(2), m.group(3)
    # A range is represented by its upper bound: the pessimistic end of the window
    # a brand quotes is the one a lead is actually judged against.
    value = float(high) if high else float(low)
    days = value * float(units[unit])
    return days if days > 0 else None


def parse_brand_profile(brand_brain: Optional[dict], cfg: dict) -> dict:
    """Reduce a Brand Brain document to the facts this layer can act on."""
    if not brand_brain:
        return {"available": False, "reason": NO_BRAND_BRAIN,
                "message": ("No Brand Brain is linked to this lead's brand, so no "
                            "brand-fit context could be applied."),
                "brand_brain_id": None}

    answers = brand_brain.get("answers") or {}
    context = brand_brain.get("context") or {}

    # --- target seniority, from whatever the brand wrote about its customer ---
    icp_text = _text(answers.get("idealCustomer"), context.get("audienceShort"),
                     answers.get("journey"), context.get("industry"))
    ranks = cfg["icp_seniority_rank"]
    matched = [level for level, pattern in cfg["icp_seniority_keywords"].items()
               if re.search(pattern, icp_text)]
    target_seniority = max(matched, key=lambda lv: ranks.get(lv, 0)) if matched else None

    # --- channels the brand says it buys ---
    from .behavioural import normalise_channel
    declared = []
    for value in (answers.get("trafficChannels") or []) + (context.get("channels") or []):
        canonical = normalise_channel(value, cfg["channel_aliases"])
        if canonical:
            declared.append(canonical)
    channels = sorted(set(declared))

    # --- business model ---
    model_text = _text(answers.get("businessType"), context.get("businessType"),
                       context.get("industry"))
    if re.search(cfg["business_type_b2b"], model_text):
        business_model = "b2b"
    elif re.search(cfg["business_type_b2c"], model_text):
        business_model = "b2c"
    else:
        business_model = None

    # --- selling language ---
    language_raw = answers.get("language")
    language_prefix = None
    if language_raw:
        low = str(language_raw).lower()
        for name, prefix in _LANGUAGE_PREFIX.items():
            if name in low:
                language_prefix = prefix
                break

    return {
        "available": True,
        "reason": None,
        "message": None,
        "brand_brain_id": brand_brain.get("_id") or brand_brain.get("brand_brain_id"),
        "brand_name": context.get("brandName"),
        "business_model": business_model,
        "business_type_raw": answers.get("businessType") or context.get("businessType"),
        "industry": context.get("industry"),
        "target_seniority": target_seniority,
        "target_seniority_matches": sorted(matched),
        "declared_channels": channels,
        "language": language_raw,
        "language_prefix": language_prefix,
        "sales_cycle_raw": answers.get("salesCycle"),
        "sales_cycle_days": parse_sales_cycle_days(answers.get("salesCycle"), cfg),
        "marketing_goal": answers.get("marketingGoal"),
    }


# --------------------------------------------------------------------------- fit
def _factor(key: str, cfg: dict, contribution: float, status: str,
            value: Any = None, detail: Optional[str] = None) -> dict:
    return {
        "feature": key,
        "label": cfg["labels"].get(key, key),
        "layer": "brand_brain",
        "basis": "heuristic",
        "status": status,
        "value": value,
        "contribution": round(float(contribution), 6),
        "direction": ("positive" if contribution > 0
                      else "negative" if contribution < 0 else "neutral"),
        "explanation": detail or (
            "Contributed positively to this prediction." if contribution > 0 else
            "Contributed negatively to this prediction." if contribution < 0 else
            "No data for this factor, so it contributed nothing."),
    }


def brand_fit_factors(profile: dict, features: dict, behaviour: dict,
                      lead: dict, cfg: dict, now: Optional[datetime] = None) -> list:
    """Score brand fit. Every factor is reported, including the ones with no data."""
    now = now or datetime.now(timezone.utc)
    w = cfg["brand_brain"]["weights"]
    mismatch = float(cfg["brand_brain"]["mismatch_scale"])
    ranks = cfg["icp_seniority_rank"]
    out = []

    if not profile.get("available"):
        for key in ("b_icp_seniority_fit", "b_channel_fit",
                    "b_business_type_email_fit", "b_language_locale_fit",
                    "b_sales_cycle_timing"):
            out.append(_factor(key, cfg, 0.0, "no_data", None,
                               "No Brand Brain is linked to this brand, so this "
                               "factor contributed nothing."))
        return out

    # ---- 1. ideal-customer seniority fit ------------------------------------
    target = profile.get("target_seniority")
    lead_seniority = features.get("f_seniority")
    if not target:
        out.append(_factor("b_icp_seniority_fit", cfg, 0.0, "no_data", lead_seniority,
                           "The brand has not described a target seniority, so this "
                           "factor contributed nothing."))
    else:
        gap = ranks.get(lead_seniority, 0) - ranks.get(target, 0)
        if gap >= 0:
            score = 1.0
        elif gap == -1:
            score = 0.0
        else:
            score = -mismatch
        out.append(_factor(
            "b_icp_seniority_fit", cfg, w["icp_seniority_fit"] * score, "observed",
            {"lead": lead_seniority, "brand_target": target},
            ("Contributed positively: the lead's seniority meets the brand's stated "
             "target." if score > 0 else
             "Contributed negatively: the lead's seniority sits below the brand's "
             "stated target." if score < 0 else
             "Contributed nothing: the lead sits just below the brand's stated target.")))

    # ---- 2. acquisition channel fit -----------------------------------------
    declared = profile.get("declared_channels") or []
    lead_channel = (behaviour or {}).get("channel")
    if not declared or not lead_channel:
        out.append(_factor("b_channel_fit", cfg, 0.0, "no_data",
                           {"lead": lead_channel, "brand_channels": declared},
                           "The brand's channels or the lead's channel are unknown, so "
                           "this factor contributed nothing."))
    else:
        hit = lead_channel in declared
        out.append(_factor(
            "b_channel_fit", cfg,
            w["channel_fit"] * (1.0 if hit else -mismatch), "observed",
            {"lead": lead_channel, "brand_channels": declared},
            ("Contributed positively: the lead arrived through a channel the brand "
             "invests in." if hit else
             "Contributed negatively: the lead arrived outside the channels the brand "
             "says it invests in.")))

    # ---- 3. business model vs contact type ----------------------------------
    model = profile.get("business_model")
    email_class = features.get("f_email_class")
    if model != "b2b":
        out.append(_factor(
            "b_business_type_email_fit", cfg, 0.0,
            "not_applicable" if model == "b2c" else "no_data", email_class,
            ("The brand sells to consumers, where a personal email address carries no "
             "signal, so this factor contributed nothing." if model == "b2c" else
             "The brand's business model is not stated, so this factor contributed "
             "nothing.")))
    else:
        corporate = email_class == "corporate"
        out.append(_factor(
            "b_business_type_email_fit", cfg,
            w["business_type_email_fit"] * (1.0 if corporate else -mismatch),
            "observed", email_class,
            ("Contributed positively: a business email address on a lead for a "
             "business-to-business brand." if corporate else
             "Contributed negatively: a personal email address on a lead for a "
             "business-to-business brand.")))

    # ---- 4. language alignment ----------------------------------------------
    prefix = profile.get("language_prefix")
    locale = features.get("f_locale")
    if not prefix or not locale or locale == "other":
        out.append(_factor("b_language_locale_fit", cfg, 0.0, "no_data",
                           {"lead_locale": locale, "brand_language": profile.get("language")},
                           "The brand's selling language or the lead's locale is "
                           "unknown, so this factor contributed nothing."))
    else:
        hit = str(locale).lower().startswith(prefix)
        out.append(_factor(
            "b_language_locale_fit", cfg,
            w["language_locale_fit"] * (1.0 if hit else -mismatch), "observed",
            {"lead_locale": locale, "brand_language": profile.get("language")},
            ("Contributed positively: the lead's locale matches the language the brand "
             "sells in." if hit else
             "Contributed negatively: the lead's locale differs from the language the "
             "brand sells in.")))

    # ---- 5. lead age against the brand's own sales cycle --------------------
    # This is the timeline factor the Brand Brain drives directly: a lead that has
    # outlived the window the brand says it closes in is a colder lead, and how
    # long that window is comes from the brand, not from a constant.
    cycle_days = profile.get("sales_cycle_days")
    t0 = lead.get("created_at")
    if not cycle_days or t0 is None:
        out.append(_factor("b_sales_cycle_timing", cfg, 0.0, "no_data",
                           profile.get("sales_cycle_raw"),
                           "The brand has not declared a sales cycle, so lead age could "
                           "not be judged and this factor contributed nothing."))
    else:
        age_hours = max((now - t0).total_seconds() / 3600.0, 0.0)
        cycle_hours = float(cycle_days) * 24.0
        ratio = max(age_hours / cycle_hours, 0.03) if cycle_hours > 0 else 1.0
        softness = float(cfg["brand_brain"]["sales_cycle_timing_softness"])
        score = -math.tanh(math.log2(ratio) / softness)
        out.append(_factor(
            "b_sales_cycle_timing", cfg, w["sales_cycle_timing"] * score, "observed",
            {"lead_age_hours": round(age_hours, 2),
             "brand_sales_cycle_days": cycle_days,
             "sales_cycle_raw": profile.get("sales_cycle_raw"),
             "age_vs_cycle": round(age_hours / cycle_hours, 4) if cycle_hours else None},
            ("Contributed positively: the lead is still well inside the brand's stated "
             "sales cycle." if score > 0 else
             "Contributed negatively: the lead has outlived the brand's stated sales "
             "cycle." if score < 0 else
             "Contributed nothing: the lead sits exactly at the brand's stated sales "
             "cycle length.")))

    return out
