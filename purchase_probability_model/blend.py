"""
Blending - turning behavioural and brand-fit evidence into a prioritisation score.

THE RULE THIS FILE EXISTS TO ENFORCE
    The calibrated base probability is never modified. It is a real, measured,
    out-of-fold-calibrated number and it stays exactly what the model produced.
    Everything here is a *separate, bounded, clearly-labelled* adjustment layered
    on top of it, reported as its own quantity so a reader can always see which
    part is measured and which part is prior.

WHY NOT JUST FOLD IT IN
    The adjustments are hand-set priors, not fitted coefficients - there is no
    behavioural data in this database to fit them on, and the ad-click rows that
    do exist are backdated to conversion time. Folding an unfitted prior into a
    calibrated probability would destroy the calibration and quietly turn a
    measured 2.3% into a number that means nothing.

    So: `probability` stays calibrated. `lead_priority` is the ranking signal the
    product should sort on, and it says of itself that it is not calibrated.

ABSENT SIGNAL IS NOT NEGATIVE SIGNAL
    A factor with no data contributes exactly 0.0 and is reported with
    status="no_data". A lead we know nothing about must not be pushed down the
    list as though we knew something bad about it.
"""
from __future__ import annotations

import math
from typing import Optional

_EPS = 1e-12


def logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _saturate(value: float, knee: float) -> float:
    """Diminishing returns in [0, 1]: the knee is where the factor is ~fully paid.

    Ten page views should not count ten times one page view. log1p gives the
    first visit most of the weight and flattens after that.
    """
    if value is None or value <= 0 or knee <= 0:
        return 0.0
    return min(math.log1p(float(value)) / math.log1p(float(knee)), 1.0)


def _factor(key: str, cfg: dict, contribution: float, status: str,
            value=None, detail: Optional[str] = None) -> dict:
    return {
        "feature": key,
        "label": cfg["labels"].get(key, key),
        "layer": "engagement",
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


# --------------------------------------------------------------------------- engagement
def recency_half_life_hours(profile: dict, cfg: dict) -> dict:
    """How fast interest goes stale - taken from the brand's own sales cycle.

    A brand that closes same-day should treat a two-day-old click as cold; a brand
    with a six-month cycle should not. This is the Brand Brain driving the
    behavioural timeline directly.
    """
    spec = cfg["engagement"]["recency_half_life_hours"]
    cycle_days = (profile or {}).get("sales_cycle_days")
    if not cycle_days:
        return {"hours": float(spec["default_when_cycle_unknown"]),
                "derived_from": "default", "sales_cycle_days": None}
    hours = float(cycle_days) * 24.0 * float(spec["fraction_of_sales_cycle"])
    hours = min(max(hours, float(spec["min"])), float(spec["max"]))
    return {"hours": round(hours, 3), "derived_from": "brand_sales_cycle",
            "sales_cycle_days": cycle_days}


def engagement_factors(behaviour: dict, profile: dict, cfg: dict) -> tuple:
    """Score observed click + timeline behaviour. Returns (factors, half_life)."""
    w = cfg["engagement"]["weights"]
    sat = cfg["engagement"]["saturation"]
    half_life = recency_half_life_hours(profile, cfg)
    keys = ("e_ad_click_arrival", "e_click_volume", "e_high_intent_pages",
            "e_page_view_volume", "e_session_return", "e_recency",
            "e_engagement_velocity")

    if not behaviour or not behaviour.get("available"):
        detail = ((behaviour or {}).get("message")
                  or "No behavioural data, so this factor contributed nothing.")
        return [_factor(k, cfg, 0.0, "no_data", None, detail) for k in keys], half_life

    o = behaviour["observed"]
    out = []

    arrived = bool(o.get("ad_click_arrival"))
    out.append(_factor(
        "e_ad_click_arrival", cfg, w["ad_click_arrival"] if arrived else 0.0,
        "observed", arrived,
        "Contributed positively: the lead arrived through a paid ad click." if arrived
        else "No ad click id is attached to this lead, so this factor contributed nothing."))

    for key, weight_key, obs_key, knee_key, phrase in (
            ("e_click_volume", "click_volume", "clicks", "click_volume_at",
             "recorded click activity"),
            ("e_high_intent_pages", "high_intent_pages", "high_intent_hits",
             "high_intent_pages_at", "visits to high-intent pages"),
            ("e_page_view_volume", "page_view_volume", "page_views",
             "page_view_volume_at", "pages viewed"),
            ("e_engagement_velocity", "engagement_velocity", "events_per_active_day",
             "engagement_velocity_at", "activity per active day")):
        raw = o.get(obs_key) or 0
        score = _saturate(raw, sat[knee_key])
        out.append(_factor(
            key, cfg, w[weight_key] * score, "observed" if raw else "no_data", raw,
            ("Contributed positively: " + phrase + " on this lead.") if raw else
            ("No " + phrase + " on record, so this factor contributed nothing.")))

    returns = max(int(o.get("sessions") or 0), int(o.get("active_days") or 0)) - 1
    out.append(_factor(
        "e_session_return", cfg, w["session_return"] * _saturate(returns, sat["session_return_at"]),
        "observed" if returns > 0 else "no_data", max(returns, 0),
        "Contributed positively: the lead came back after the first visit." if returns > 0
        else "No return visit is on record, so this factor contributed nothing."))

    # The only engagement factor allowed to go negative: we have seen this lead
    # act, so silence since then is genuine evidence rather than missing data.
    recency = o.get("recency_hours")
    if recency is None:
        out.append(_factor("e_recency", cfg, 0.0, "no_data", None,
                           "No timestamped activity, so this factor contributed nothing."))
    else:
        freshness = 2.0 ** (-max(float(recency), 0.0) / half_life["hours"])
        score = 2.0 * freshness - 1.0
        out.append(_factor(
            "e_recency", cfg, w["recency"] * score, "observed",
            {"recency_hours": recency, "half_life_hours": half_life["hours"],
             "half_life_from": half_life["derived_from"]},
            ("Contributed positively: the lead was active recently against this brand's "
             "pace." if score > 0 else
             "Contributed negatively: the lead has been quiet for longer than this "
             "brand's pace allows." if score < 0 else
             "Contributed nothing: the lead sits exactly at this brand's half-life.")))

    return out, half_life


# --------------------------------------------------------------------------- combining
def _apply_layer(factors: list, bounds: list) -> dict:
    """Sum a layer and clamp it. The clamp is what keeps a prior from taking over."""
    raw = sum(float(f["contribution"]) for f in factors)
    lo, hi = float(bounds[0]), float(bounds[1])
    applied = min(max(raw, lo), hi)
    return {
        "total_raw": round(raw, 6),
        "total_applied": round(applied, 6),
        "clamped": abs(applied - raw) > 1e-9,
        "bounds": [lo, hi],
        "contributing_factors": sum(1 for f in factors if abs(f["contribution"]) > 1e-9),
        "factors_with_data": sum(1 for f in factors if f["status"] == "observed"),
    }


def combine_layers(base_probability: float, eng_factors: list, brand_factors: list,
                   cfg: dict) -> dict:
    """Base log-odds + two bounded adjustments -> the prioritisation probability."""
    base_lo = logit(base_probability)
    eng = _apply_layer(eng_factors, cfg["engagement"]["max_total_adjustment"])
    brand = _apply_layer(brand_factors, cfg["brand_brain"]["max_total_adjustment"])
    adjusted_lo = base_lo + eng["total_applied"] + brand["total_applied"]
    adjusted_p = sigmoid(adjusted_lo)
    return {
        "base": {"probability": round(float(base_probability), 8),
                 "log_odds": round(base_lo, 6),
                 "calibrated": True},
        "engagement": eng,
        "brand_brain": brand,
        "adjusted": {"log_odds": round(adjusted_lo, 6),
                     "probability": round(adjusted_p, 8),
                     "calibrated": False},
        "total_adjustment": round(eng["total_applied"] + brand["total_applied"], 6),
        "signal_version": cfg["signal_version"],
    }
