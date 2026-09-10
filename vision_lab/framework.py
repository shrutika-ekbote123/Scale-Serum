"""
Loads and validates vision_framework.json and psychology_framework.json.

WHY A FILE AND NOT CONSTANTS
    Same rule as sales_call_analyzer/framework.py: every business value lives in
    JSON so management can change a weight or a threshold and redeploy without a
    Python change. The code here only reads, validates and reports what was
    still unconfigured.

NULL IS A VALUE
    A null weight means "no decision has been made", not "worth 1". Every reader
    below returns both the value it will use AND the mode it is operating in, so
    a placeholder can never be reported as a decision. config_disclosure() is
    what the API puts in every response.
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_PATH = os.path.join(HERE, "vision_framework.json")
PSYCHOLOGY_PATH = os.path.join(HERE, "psychology_framework.json")

# Weighting modes, reported in every response.
WEIGHTING_EQUAL = "equal_unweighted_placeholder"
WEIGHTING_CONFIGURED = "configured"

RATING_SCALE_UNIFORM = "uniform_placeholder"
RATING_SCALE_CONFIGURED = "configured"

CALIBRATION_PROVISIONAL = "provisional_absolute"
CALIBRATION_PERCENTILE = "percentile"

# The six metrics, in display order. Kept here as well as in the JSON so a
# malformed config is caught at import rather than at request time.
METRIC_IDS = ("attention", "focus", "cognitive_demand", "clarity",
              "brand_memory", "engagement")


class FrameworkConfigError(RuntimeError):
    """The config file is missing, unreadable, or missing something required."""


def _read(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as err:
        raise FrameworkConfigError(f"config file not found: {path}") from err
    except json.JSONDecodeError as err:
        raise FrameworkConfigError(f"config file is not valid JSON: {path} - {err}") from err


@functools.lru_cache(maxsize=1)
def load_framework() -> dict:
    """The metric framework. Cached - it is read once per process."""
    cfg = _read(FRAMEWORK_PATH)

    if not cfg.get("framework_version"):
        raise FrameworkConfigError("vision_framework.json has no framework_version")

    metrics = cfg.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise FrameworkConfigError("vision_framework.json has no metrics")

    found = tuple(m.get("id") for m in metrics)
    missing = [m for m in METRIC_IDS if m not in found]
    if missing:
        raise FrameworkConfigError(
            f"vision_framework.json is missing metric(s): {', '.join(missing)}")

    for metric in metrics:
        if metric.get("direction") not in ("higher_better", "lower_better"):
            raise FrameworkConfigError(
                f"metric '{metric.get('id')}' has no valid direction")

    timeline = cfg.get("timeline") or {}
    if not timeline.get("index_terms"):
        raise FrameworkConfigError("vision_framework.json has no timeline.index_terms")

    return cfg


@functools.lru_cache(maxsize=1)
def load_psychology() -> dict:
    """The 15 triggers. Cached."""
    cfg = _read(PSYCHOLOGY_PATH)

    if not cfg.get("psychology_version"):
        raise FrameworkConfigError("psychology_framework.json has no psychology_version")

    triggers = cfg.get("triggers")
    if not isinstance(triggers, list) or len(triggers) != 15:
        raise FrameworkConfigError(
            f"psychology_framework.json must define exactly 15 triggers, "
            f"found {len(triggers) if isinstance(triggers, list) else 'none'}")

    seen = set()
    for trigger in triggers:
        tid = trigger.get("id")
        if not tid:
            raise FrameworkConfigError("a trigger has no id")
        if tid in seen:
            raise FrameworkConfigError(f"duplicate trigger id: {tid}")
        seen.add(tid)
        if trigger.get("detection") not in ("measured", "judged", "hybrid"):
            raise FrameworkConfigError(
                f"trigger '{tid}' has no valid detection mode")

    return cfg


# --------------------------------------------------------------------------- readers
def metric_index(cfg: Optional[dict] = None) -> dict:
    """metric id -> its definition."""
    cfg = cfg or load_framework()
    return {m["id"]: m for m in cfg["metrics"]}


def trigger_index(cfg: Optional[dict] = None) -> dict:
    """trigger id -> its definition."""
    cfg = cfg or load_psychology()
    return {t["id"]: t for t in cfg["triggers"]}


def weighting_mode(cfg: Optional[dict] = None) -> str:
    cfg = cfg or load_framework()
    confirmed = (cfg.get("weights") or {}).get("metric_weights_confirmed")
    return WEIGHTING_CONFIGURED if confirmed else WEIGHTING_EQUAL


def rating_scale(cfg: Optional[dict] = None) -> tuple[dict, str]:
    """Rating level -> fraction of score_max, and which mode produced it.

    With no configured scale the levels are spaced uniformly over [0, 1]. That
    is a mechanical default, not a judgement about what 'adequate' is worth.
    """
    cfg = cfg or load_framework()
    configured = cfg.get("rating_scale")
    levels = cfg.get("rating_levels") or []
    if isinstance(configured, dict) and configured:
        return configured, RATING_SCALE_CONFIGURED
    if len(levels) < 2:
        return {}, RATING_SCALE_UNIFORM
    step = 1.0 / (len(levels) - 1)
    return {level: round(i * step, 6) for i, level in enumerate(levels)}, RATING_SCALE_UNIFORM


def bands(cfg: Optional[dict] = None) -> tuple[Optional[list], Optional[str]]:
    """Score bands, or None plus the reason there are none."""
    cfg = cfg or load_framework()
    configured = cfg.get("bands")
    if configured:
        return configured, None
    return None, "thresholds_not_configured"


def reading_speed(cfg: Optional[dict] = None) -> float:
    """Words per second used to decide whether on-screen copy can be read in the
    time it is held. Quoted verbatim in fix recommendations, so it is config."""
    cfg = cfg or load_framework()
    return float(cfg.get("reading_speed_words_per_second") or 4)


def sampling(cfg: Optional[dict] = None) -> dict:
    cfg = cfg or load_framework()
    return dict(cfg.get("sampling") or {})


def weak_zone_rule(cfg: Optional[dict] = None) -> tuple[float, float, bool]:
    """(threshold, minimum seconds, whether both were actually configured)."""
    cfg = cfg or load_framework()
    tl = cfg.get("timeline") or {}
    threshold = tl.get("weak_threshold")
    seconds = tl.get("min_weak_seconds")
    configured = threshold is not None and seconds is not None
    return (
        float(threshold if threshold is not None else tl.get("weak_threshold_placeholder", 35)),
        float(seconds if seconds is not None else tl.get("min_weak_seconds_placeholder", 1.0)),
        configured,
    )


def hook_seconds(cfg: Optional[dict] = None) -> float:
    cfg = cfg or load_framework()
    return float((cfg.get("timeline") or {}).get("hook_seconds") or 3.0)


def saliency_model(cfg: Optional[dict] = None) -> dict:
    cfg = cfg or load_framework()
    return dict(cfg.get("saliency_model") or {})


def triggers_affect_score(cfg: Optional[dict] = None) -> bool:
    cfg = cfg or load_psychology()
    return bool(cfg.get("affects_overall_score"))


def lexicon(name: str, cfg: Optional[dict] = None) -> Any:
    cfg = cfg or load_psychology()
    return (cfg.get("lexicons") or {}).get(name)


# --------------------------------------------------------------------------- disclosure
def config_disclosure(framework: Optional[dict] = None,
                      psychology: Optional[dict] = None) -> dict:
    """What every response says about which values were still placeholders.

    This is the mechanism that stops a neutral default being mistaken for a
    business decision. It is cheap, and it goes in every payload.
    """
    framework = framework or load_framework()
    psychology = psychology or load_psychology()

    _, band_reason = bands(framework)
    _, rating_mode = rating_scale(framework)
    _, _, weak_configured = weak_zone_rule(framework)
    timeline = framework.get("timeline") or {}
    calibration = framework.get("calibration") or {}

    unconfirmed = list(framework.get("unconfirmed_business_values") or [])
    unconfirmed += list(psychology.get("unconfirmed_business_values") or [])

    return {
        "weighting": weighting_mode(framework),
        "rating_scale": rating_mode,
        "bands": "configured" if band_reason is None else band_reason,
        "weak_zone_rule": "configured" if weak_configured else "placeholder",
        "timeline_coefficients": ("configured"
                                  if timeline.get("coefficients_confirmed")
                                  else "uniform_placeholder"),
        "calibration": calibration.get("mode") or CALIBRATION_PROVISIONAL,
        "triggers_affect_overall_score": triggers_affect_score(psychology),
        "reading_speed_words_per_second": reading_speed(framework),
        "unconfirmed": unconfirmed,
    }


def versions() -> dict:
    """The version stamp that goes in every report and every fingerprint."""
    return {
        "framework_version": load_framework()["framework_version"],
        "psychology_version": load_psychology()["psychology_version"],
    }
