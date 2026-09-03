"""
Plain-language rendering of factors. PRESENTATION ONLY.

WHAT THIS DOES
    Takes the factors the model and the ranking layers already produced and adds
    the fields a human can read: a title, a sentence, an impact word, and where
    the evidence came from. It renames; it never rescores.

WHAT THIS MUST NOT DO
    Change a probability, a coefficient, a contribution, an ordering rule, or
    which factors exist. Every number here is passed through untouched -
    `contribution` is copied verbatim, and `odds_multiplier` is exp() of it,
    which is a restatement of the same quantity, not a new one.

WHY `affects` EXISTS
    `top_factors` now carries factors from three places, and they do not all move
    the same number:

        affects = "purchase_probability"  - base-model factors. These sum, in
                                            log-odds, to the calibrated score.
        affects = "lead_priority"         - engagement and brand-fit factors.
                                            These are bounded, UNFITTED priors.
                                            They move the ranking score only.

    A reader who sums every row and expects the displayed probability will be
    wrong, so each row says what it moves. Do not drop this field from the UI.

LANGUAGE RULE
    Associations, never causes. "Founders convert more often" is a fact about the
    training data. "Being a founder causes conversion" is a claim we cannot make
    from a logistic regression on observational data. A test enforces this.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

_LANG: Optional[dict] = None
_LANG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "factor_language.json")

# Explanations the underlying layers emit when they have nothing specific to say.
# When we see one of these we substitute the copy from factor_language.json;
# anything else is a computed sentence and is better than our generic text.
_GENERIC_PREFIXES = ("Contributed positively", "Contributed negatively",
                     "No data for this factor")


def load_language() -> dict:
    """Cached read of factor_language.json. Missing file is not fatal: the
    factors keep their original labels and simply stay technical."""
    global _LANG
    if _LANG is None:
        try:
            with open(_LANG_PATH, "r", encoding="utf-8") as fh:
                _LANG = json.load(fh)
        except Exception:
            _LANG = {}
    return _LANG


def _impact_word(contribution: float, lang: dict) -> str:
    """Size band + direction, e.g. 'Strong positive'. Bands live in the config."""
    c = float(contribution or 0.0)
    if abs(c) < 1e-12:
        return "No effect"
    bands = lang.get("impact_bands") or [{"min_abs": 0.0, "word": "Slight"}]
    size = bands[-1].get("word", "Slight")
    for band in bands:
        if abs(c) >= float(band.get("min_abs", 0.0)):
            size = band.get("word", size)
            break
    return f"{size} {'positive' if c > 0 else 'negative'}"


def _odds_multiplier(contribution: float) -> Optional[float]:
    """exp(log-odds contribution). 0.5 -> 1.65, i.e. 'about 1.65x the odds'.

    This is the same number as `contribution`, stated in a unit people read.
    Guarded because exp() overflows on absurd input.
    """
    try:
        return round(math.exp(float(contribution)), 3)
    except (OverflowError, TypeError, ValueError):
        return None


def _hour_from_cyclic(features: dict) -> Optional[float]:
    """Recover the submission hour from the sin/cos pair.

    The hour itself is not stored as a feature - it is encoded as a point on a
    circle so that 23:00 and 00:00 sit next to each other. atan2 inverts that.
    """
    try:
        s = float(features["f_hour_sin"])
        c = float(features["f_hour_cos"])
    except (KeyError, TypeError, ValueError):
        return None
    if abs(s) < 1e-9 and abs(c) < 1e-9:
        return None
    return (math.degrees(math.atan2(s, c)) / 360.0 * 24.0) % 24.0


def _hour_window(hour: float) -> str:
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _format_hour(hour: float) -> str:
    h = int(hour) % 24
    suffix = "am" if h < 12 else "pm"
    display = h % 12 or 12
    return f"{display}{suffix}"


def _join(title: Optional[str], reason: Optional[str]) -> Optional[str]:
    """'<what was observed> - <why that matters>', the one line the card prints."""
    if title and reason:
        return f"{title} - {reason}"
    return title or reason


def _model_copy(feature: str, value: Any, contribution: float, features: dict,
                lang: dict) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(label, title, detail) for one base-model feature.

    `label` is the line the lead card prints and must stand on its own. Returns
    (None, None, None) when the config has nothing, so the caller can fall back
    to the technical label rather than printing an empty row.
    """
    spec = ((lang.get("features") or {}).get(feature)) or {}
    if not spec:
        return None, None, None

    # Categorical: the copy depends on which level the lead landed on.
    values = spec.get("values")
    if values:
        entry = values.get(str(value))
        if entry:
            return (_join(entry.get("title"), entry.get("reason")),
                    entry.get("title"), entry.get("detail"))
        return spec.get("title"), spec.get("title"), None

    if feature == "f_hour_time_pattern":
        hour = _hour_from_cyclic(features)
        if hour is None:
            return spec.get("title"), spec.get("title"), None
        window = _hour_window(hour)
        shown = _format_hour(hour)
        label = (spec.get("label") or "").format(
            hour=shown,
            window_reason=(spec.get("window_reasons") or {}).get(window, "")).strip()
        detail = (spec.get("detail") or "").format(
            hour=shown, window_note=(spec.get("windows") or {}).get(window, "")).strip()
        return label or spec.get("title"), spec.get("title"), detail

    if feature == "f_company_len":
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            return spec.get("title"), spec.get("title"), None
        if n <= 0:
            return (spec.get("label_zero"), spec.get("title"), spec.get("detail_zero"))
        suffix = "positive" if contribution > 0 else "negative"
        return ((spec.get(f"label_{suffix}") or "").format(value=n) or spec.get("title"),
                spec.get("title"),
                (spec.get(f"detail_{suffix}") or "").format(value=n))

    suffix = "positive" if contribution > 0 else "negative"
    return (spec.get(f"label_{suffix}") or spec.get("title"),
            spec.get("title"),
            spec.get(f"detail_{suffix}") or spec.get("detail"))


def _ago(hours: float) -> str:
    """'22 hours ago' / '3 days ago'. Rounded, because false precision on a
    recency figure invites arguments about minutes that do not matter."""
    if hours < 1:
        return "less than an hour ago"
    if hours < 48:
        n = int(round(hours))
        return f"{n} hour{'s' if n != 1 else ''} ago"
    n = int(round(hours / 24.0))
    return f"{n} day{'s' if n != 1 else ''} ago"


def _layer_reason(explanation: Optional[str]) -> Optional[str]:
    """Pull the specific clause out of a layer's own explanation.

    The layers write 'Contributed positively: <the actual reason>.' That trailing
    clause is computed per lead and is better copy than anything static, so the
    label uses it when present. The generic form carries no clause and yields
    None, letting the configured wording take over.
    """
    text = (explanation or "").strip()
    for prefix in ("Contributed positively:", "Contributed negatively:"):
        if text.startswith(prefix):
            clause = text[len(prefix):].strip().rstrip(".")
            return clause or None
    return None


def _signal_headline(feature: str, factor: dict, observed: Optional[dict]) -> Optional[str]:
    """The short measured phrase for a label - one line, no trailing sentence."""
    obs = observed or {}
    value = factor.get("value")

    def count(key):
        n = obs.get(key)
        return int(n) if isinstance(n, (int, float)) else None

    if feature == "e_ad_click_arrival":
        return "Arrived from a paid ad click" if value else None
    if feature == "e_click_volume":
        n = count("clicks")
        return f"{n} click{'s' if n != 1 else ''} on site" if n else None
    if feature == "e_high_intent_pages":
        n = count("high_intent_hits")
        return (f"{n} visit{'s' if n != 1 else ''} to pricing or checkout pages"
                if n else None)
    if feature == "e_page_view_volume":
        n = count("page_views")
        return f"{n} page{'s' if n != 1 else ''} viewed" if n else None
    if feature == "e_session_return":
        days = count("active_days")
        return (f"Came back across {days} separate days" if days
                else "Came back after the first visit")
    if feature == "e_engagement_velocity":
        try:
            return f"About {float(value):.1f} events per active day"
        except (TypeError, ValueError):
            return None
    if feature == "e_recency":
        if isinstance(value, dict) and value.get("recency_hours") is not None:
            try:
                return f"Last active {_ago(float(value['recency_hours']))}"
            except (TypeError, ValueError):
                return None
    return None


def _signal_fact(feature: str, factor: dict, observed: Optional[dict]) -> Optional[str]:
    """The measured statement for a touchpoint-derived factor.

    Returns what was actually counted for this lead, so the row reads as an
    observation rather than as a category. Everything here comes from the
    engagement layer's own `observed` block - nothing is recomputed or inferred.
    """
    obs = observed or {}
    value = factor.get("value")

    def count(key):
        n = obs.get(key)
        return int(n) if isinstance(n, (int, float)) else None

    if feature == "e_ad_click_arrival":
        return "Arrived through a paid ad click." if value else None
    if feature == "e_click_volume":
        n = count("clicks")
        return f"{n} click{'s' if n != 1 else ''} recorded." if n else None
    if feature == "e_high_intent_pages":
        n = count("high_intent_hits")
        return f"{n} visit{'s' if n != 1 else ''} to high-intent pages." if n else None
    if feature == "e_page_view_volume":
        n = count("page_views")
        return f"{n} page{'s' if n != 1 else ''} viewed." if n else None
    if feature == "e_session_return":
        days = count("active_days")
        return (f"Active again after the first visit, across {days} separate days."
                if days else "Active again after the first visit.")
    if feature == "e_engagement_velocity":
        try:
            return f"About {float(value):.1f} events on each active day."
        except (TypeError, ValueError):
            return None
    if feature == "e_recency":
        if isinstance(value, dict) and value.get("recency_hours") is not None:
            try:
                phrase = _ago(float(value["recency_hours"]))
            except (TypeError, ValueError):
                return None
            half = value.get("half_life_hours")
            if half:
                return (f"Last active {phrase}. This brand's interest half-life is "
                        f"about {int(round(float(half) / 24.0))} days.")
            return f"Last active {phrase}."
    return None


def humanise(factor: dict, layer: str, features: Optional[dict],
             lang: Optional[dict] = None,
             observed: Optional[dict] = None) -> dict:
    """Return a copy of `factor` with the human-readable fields added.

    The original keys - feature, label, contribution, direction, explanation -
    are preserved untouched so existing consumers keep working.
    """
    lang = lang if lang is not None else load_language()
    out = dict(factor)
    feature = str(factor.get("feature") or "")
    contribution = float(factor.get("contribution") or 0.0)

    if layer == "model":
        label, title, detail = _model_copy(feature, factor.get("value"),
                                           contribution, features or {}, lang)
    else:
        spec = (lang.get("signals") or {}).get(feature) or {}
        title, detail = spec.get("title"), spec.get("detail")
        # What was measured, then why it matters. The measured half is dropped
        # when nothing was counted, leaving the general sentence on its own.
        fact = _signal_fact(feature, factor, observed)
        if fact:
            detail = f"{fact} {detail}".strip() if detail else fact
        elif not detail:
            existing = str(factor.get("explanation") or "")
            if existing and not existing.startswith(_GENERIC_PREFIXES):
                detail = existing
        # When we already state what was measured, the configured reason adds the
        # 'why'. When we do not - the brand-fit factors - the layer's own computed
        # clause is the specific half, and repeating it would read as a stutter.
        headline = _signal_headline(feature, factor, observed)
        if headline:
            label = _join(headline, spec.get("reason"))
        else:
            label = _join(title, _layer_reason(factor.get("explanation"))
                          or spec.get("reason"))

    # `label` is the one line the card prints, so it must never fall back to a
    # bare technical name silently - the original label is the last resort.
    out["label"] = label or title or factor.get("label") or feature
    out["title"] = title or factor.get("label") or feature
    if detail:
        out["detail"] = detail
    out["impact"] = _impact_word(contribution, lang)
    out["odds_multiplier"] = _odds_multiplier(contribution)
    out["source"] = (lang.get("sources") or {}).get(layer, layer)
    out["affects"] = (lang.get("affects") or {}).get(layer, "lead_priority")
    out.setdefault("layer", layer)
    return out


def merge_top_factors(model_factors: list, engagement_factors: Optional[list],
                      brand_factors: Optional[list],
                      features: Optional[dict],
                      observed: Optional[dict] = None) -> list[dict]:
    """The single ranked list the lead card renders.

    Base-model factors first-class, then the touchpoint-derived engagement
    factors and the brand-fit factors, every row tagged with what it moves.
    Ordering is by absolute contribution, unchanged from before - a factor that
    moved the score most is still shown first.

    Factors that contributed nothing are dropped: a row reading "No effect"
    is noise on a lead card, and an absent signal is reported by its layer's
    `available: false`, not as a zero.
    """
    lang = load_language()
    merged = [humanise(f, "model", features, lang) for f in (model_factors or [])]
    merged += [humanise(f, "engagement", features, lang, observed)
               for f in (engagement_factors or [])]
    merged += [humanise(f, "brand_brain", features, lang, observed)
               for f in (brand_factors or [])]
    merged = [f for f in merged if abs(float(f.get("contribution") or 0.0)) > 1e-12]
    merged.sort(key=lambda d: abs(float(d.get("contribution") or 0.0)), reverse=True)
    return merged


def baseline_block(metadata: dict, constants: Optional[dict],
                   probability: float) -> dict:
    """The 'start here, end there' header the lead card shows above the factors.

    The base rate is the observed conversion rate of the training snapshot. It is
    a real measured number, not a presentation constant.
    """
    training = (metadata or {}).get("training") or {}
    rows = training.get("dataset_rows")
    positives = training.get("dataset_positives")
    base_rate = None
    if rows and positives is not None:
        try:
            base_rate = round(float(positives) / float(rows) * 100.0, 2)
        except (TypeError, ZeroDivisionError, ValueError):
            base_rate = None

    horizon = (constants or {}).get("horizon_days")
    window = f" within {horizon} days" if horizon else ""
    return {
        "starting_point": {
            "label": "Base rate - all leads",
            "percent": base_rate,
            "detail": (f"{base_rate}% of leads purchase{window}."
                       if base_rate is not None else None),
        },
        "result": {
            "label": "Purchase probability",
            "percent": round(float(probability) * 100.0, 2),
        },
        "reading_note": (
            "Factors marked purchase_probability sum, in log-odds, to the result. "
            "Factors marked lead_priority are bounded priors that move the ranking "
            "score instead, so the rows do not add up to the percentage on screen."
        ),
    }
