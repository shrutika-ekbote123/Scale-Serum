"""
Step 13 - the attention timeline, weak zones and key moments.

WHAT THIS NUMBER IS, AND WHAT IT IS NOT
    A saliency map is a spatial probability distribution: every frame's map sums
    to 1. A frame of grey mush and a frame with a gripping close-up both sum to
    1. So the model CANNOT tell us whether a frame held the viewer - only where
    they would look within it.

    The per-second "attention" figure is therefore a DERIVED COMPOSITE INDEX
    over six measured signals, not a model output. It is reported as
    basis="derived_composite" everywhere it appears and must never be described
    as measured attention.

WHY THE INDEX IS ABSOLUTE, NOT PER-CREATIVE
    Standardising each signal within one ad would force every creative to
    average the same score, which makes "attention fell to 26 here" meaningless
    across ads and impossible to compare in the History tab. So each signal is
    mapped through a documented absolute anchor.

    Those anchors are PLACEHOLDERS. They come from the observed range across our
    own corpus, not from a fitted model, and every response says
    calibration="provisional_absolute". Phase 5 replaces them with a percentile
    reference and, eventually, coefficients fitted against real platform
    retention curves.

WEAK ZONES AND KEY MOMENTS ARE RULES, NOT OPINIONS
    A weak zone is the index below a threshold for a minimum duration. PEAK is
    the global maximum. HERO is the strongest moment in the second half. Both
    thresholds are business decisions sitting null in config, so the placeholders
    are used and disclosed.
"""
from __future__ import annotations

import math
from typing import Optional

from . import (
    MOMENT_HERO,
    MOMENT_KEY,
    MOMENT_PEAK,
    MOMENT_WEAK,
)
from . import framework as fw

# Absolute anchors: the raw value that maps to 0 and to 1 for each signal.
#
# Taken from the observed spread across our own annotated corpus - they are
# calibration placeholders, not fitted parameters, and they are the first thing
# Phase 5 replaces. Documented here rather than buried so the numbers can be
# argued with.
ANCHORS = {
    "concentration": (0.08, 0.45),    # share of mass in the top 5% of pixels
    "motion_energy": (0.00, 0.12),    # mean absolute luma change between samples
    "face_presence": (0.00, 0.35),    # saliency mass landing on a detected face
    "gaze_stability": (0.30, 0.95),   # 1 - total variation between maps
    "text_load": (0.00, 2.00),        # words / (reading speed x shot seconds)
    "novelty": (0.00, 4.00),          # seconds since the last cut (inverted)
}

# Which way each term pushes. text_load and novelty are negative: more reading
# to do, and a shot that has been held a long time, both reduce attention.
DIRECTION = {
    "concentration": 1.0, "motion_energy": 1.0, "face_presence": 1.0,
    "gaze_stability": 1.0, "text_load": -1.0, "novelty": -1.0,
}

# The engagement line is the SAME function with novelty and motion up-weighted
# and concentration down-weighted - one function, two coefficient sets, so the
# two lines cannot drift apart in meaning.
ENGAGEMENT_BIAS = {
    "concentration": 0.6, "motion_energy": 1.6, "face_presence": 1.0,
    "gaze_stability": 0.8, "text_load": 1.0, "novelty": 1.6,
}


def _scale(value: Optional[float], anchor: tuple[float, float]) -> Optional[float]:
    """Raw signal to 0-1 against its absolute anchor. None stays None - a
    missing measurement is not a zero one."""
    if value is None:
        return None
    low, high = anchor
    if high <= low:
        return None
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def signals_for(record: dict, reading_speed: float,
                shot_seconds: Optional[float] = None) -> dict:
    """The six index terms for one frame, each scaled to 0-1 or None."""
    saliency = record.get("saliency") or {}
    regions = record.get("regions") or {}
    mass = record.get("mass") or {}

    # None, not 0.0, when OCR could not read the frame.
    #
    # "We could not read the text" and "there is no text" are different
    # findings. Collapsing them to zero fabricates a measurable signal out of a
    # missing one - and because the index only needs ONE available term, a
    # frame with nothing measured at all would still produce a confident
    # attention number built entirely on that fabrication.
    words = regions.get("word_count")
    seconds = shot_seconds or 1.0
    text_load = None if words is None else (words / (reading_speed * seconds))

    since_cut = record.get("seconds_into_shot")

    return {
        "concentration": _scale(saliency.get("concentration"),
                                ANCHORS["concentration"]),
        "motion_energy": _scale(record.get("motion_energy"),
                                ANCHORS["motion_energy"]),
        "face_presence": _scale(mass.get("faces"), ANCHORS["face_presence"]),
        "gaze_stability": _scale(record.get("gaze_stability"),
                                 ANCHORS["gaze_stability"]),
        "text_load": _scale(text_load, ANCHORS["text_load"]),
        "novelty": _scale(since_cut, ANCHORS["novelty"]),
    }


def _index(signals: dict, weights: dict) -> Optional[float]:
    """Weighted sum of the available terms, through a sigmoid, to 0-100.

    Terms that could not be measured are EXCLUDED from the denominator rather
    than treated as zero. A frame where OCR failed is a frame we know less
    about, not a frame with no text on it.
    """
    total = 0.0
    used = 0.0
    for name, value in signals.items():
        if value is None:
            continue
        weight = weights.get(name, 1.0)
        total += DIRECTION[name] * weight * value
        used += weight
    if used <= 0:
        return None

    # Centred so that an average frame lands near the middle of the range
    # rather than at an arbitrary point.
    mean = total / used
    return round(100.0 / (1.0 + math.exp(-4.0 * (mean - 0.1))), 1)


def build(measurements: list[dict], media: dict,
          cfg: Optional[dict] = None) -> dict:
    """The full timeline: two lines, weak zones, and the coefficient disclosure."""
    cfg = cfg or fw.load_framework()
    reading_speed = fw.reading_speed(cfg)
    timeline_cfg = cfg.get("timeline") or {}
    configured = {term["id"]: term.get("weight")
                  for term in timeline_cfg.get("index_terms") or []}
    coefficients_set = any(v is not None for v in configured.values())

    # A null weight means no decision has been made, not a weight of zero.
    attention_weights = {k: (configured.get(k) if configured.get(k) is not None
                             else 1.0) for k in ANCHORS}
    engagement_weights = {k: attention_weights[k] * ENGAGEMENT_BIAS[k]
                          for k in ANCHORS}

    sample_fps = media.get("sample_fps") or 2.0
    shot_lengths = _shot_lengths(measurements, sample_fps)

    points = []
    for record in measurements:
        shot_seconds = shot_lengths.get(record.get("shot"), 1.0)
        signals = signals_for(record, reading_speed, shot_seconds)
        points.append({
            "t": record.get("t"),
            "attention": _index(signals, attention_weights),
            "engagement": _index(signals, engagement_weights),
            "signals": {k: (round(v, 4) if v is not None else None)
                        for k, v in signals.items()},
        })

    threshold, min_seconds, rule_configured = fw.weak_zone_rule(cfg)
    zones = weak_zones(points, threshold, min_seconds, sample_fps)

    return {
        "unit": "index_0_100",
        "basis": "derived_composite",
        "basis_note": ("A composite over six measured signals, not a model "
                       "output. Saliency maps sum to 1 on every frame and "
                       "cannot say whether a frame held the viewer."),
        "points": [{k: v for k, v in p.items() if k != "signals"} for p in points],
        "signals": [p["signals"] for p in points],
        "weak_zones": zones,
        "markers": [_marker(z) for z in zones],
        "coefficients": ("configured" if coefficients_set
                         else "uniform_placeholder"),
        "calibration": (cfg.get("calibration") or {}).get(
            "mode", "provisional_absolute"),
        "weak_zone_rule": {
            "threshold": threshold,
            "min_seconds": min_seconds,
            "configured": rule_configured,
        },
    }


def _shot_lengths(measurements: list[dict], sample_fps: float) -> dict:
    counts: dict = {}
    for record in measurements:
        counts[record.get("shot")] = counts.get(record.get("shot"), 0) + 1
    return {shot: frames / sample_fps for shot, frames in counts.items()}


def weak_zones(points: list[dict], threshold: float, min_seconds: float,
               sample_fps: float) -> list[dict]:
    """Runs where the index stays below the threshold for long enough.

    A single dip is not a dead zone - a viewer does not disengage because one
    sampled frame was flat. The duration requirement is what separates a dip
    from a hole.
    """
    zones = []
    run: list[dict] = []
    for point in points + [{"attention": None, "t": None}]:
        value = point.get("attention")
        if value is not None and value < threshold:
            run.append(point)
            continue
        if run:
            seconds = len(run) / sample_fps
            if seconds >= min_seconds:
                values = [p["attention"] for p in run]
                zones.append({
                    "t_start": run[0]["t"],
                    "t_end": round(run[-1]["t"] + 1 / sample_fps, 3),
                    "seconds": round(seconds, 2),
                    "min_attention": min(values),
                    "mean_attention": round(sum(values) / len(values), 1),
                })
            run = []
    return zones


def _marker(zone: dict) -> dict:
    return {
        "t": zone["t_start"],
        "severity": ("high" if zone["seconds"] >= 2.0 else
                     "medium" if zone["seconds"] >= 1.0 else "low"),
        "text": (f"Attention falls to {zone['min_attention']:.0f} for "
                 f"{zone['seconds']:.1f}s"),
    }


def key_moments(points: list[dict], measurements: list[dict],
                zones: list[dict]) -> list[dict]:
    """PEAK / HERO / KEY / WEAK - rules over the index, not model output.

    Degenerate frames are excluded from PEAK and HERO. A near-black end card
    drives the saliency map onto one artefact and reads as maximum
    concentration, so ranking naively would make an empty slate the headline
    moment of every report.
    """
    from .measure import is_degenerate

    usable = [(p, m) for p, m in zip(points, measurements)
              if p.get("attention") is not None and not is_degenerate(m)]
    if not usable:
        return []

    moments = []
    peak = max(usable, key=lambda pair: pair[0]["attention"])
    moments.append(_moment(peak[0], MOMENT_PEAK, "strongest moment in the ad"))

    half = usable[len(usable) // 2:]
    if half:
        hero = max(half, key=lambda pair: pair[0]["attention"])
        if hero[0]["t"] != peak[0]["t"]:
            moments.append(_moment(hero[0], MOMENT_HERO,
                                   "strongest moment after the midpoint"))

    # KEY: the first frame where on-screen text carries most of the attention.
    for point, record in usable:
        mass = record.get("mass") or {}
        text = mass.get("text")
        if text is not None and text > 0.4:
            moments.append(_moment(point, MOMENT_KEY,
                                   "on-screen copy takes the majority of gaze"))
            break

    for zone in zones:
        moments.append({
            "time": zone["t_start"],
            "label": MOMENT_WEAK,
            "attention": zone["min_attention"],
            "note": (f"attention below the weak threshold for "
                     f"{zone['seconds']:.1f}s"),
        })

    return sorted(moments, key=lambda m: m["time"])


def _moment(point: dict, label: str, note: str) -> dict:
    return {"time": point["t"], "label": label,
            "attention": point["attention"], "note": note}
