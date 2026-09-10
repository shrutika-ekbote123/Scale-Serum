"""Milestone C - the timeline and the six scores.

Synthetic measurement records in, exact numbers out. No ffmpeg, no model, no
network: this is arithmetic, and arithmetic is the thing to pin.

The rule under test throughout: a measurement that is MISSING must produce a
null score with a stated reason, never a zero. "We could not measure this" and
"this creative did badly" are different findings, and a report that conflates
them is worse than one that says nothing.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

from vision_lab import framework as fw  # noqa: E402
from vision_lab import scoring as sc  # noqa: E402
from vision_lab import timeline as tl  # noqa: E402


def record(t=0.0, shot=0, concentration=0.25, peaks=2, motion=0.05,
           stability=0.8, words=6, faces_mass=0.1, brand=None, cta=None,
           text_mass=0.25, text_area=0.1, contrast=0.3, into_shot=1.0):
    return {
        "index": int(t * 2), "t": t, "shot": shot,
        "seconds_into_shot": into_shot,
        "frame_contrast": contrast,
        "motion_energy": motion,
        "gaze_stability": stability,
        "saliency": {"concentration": concentration, "peak_count": peaks,
                     "peaks": [{"box": [0.3, 0.3, 0.6, 0.6], "share": 0.3,
                                "rank": 1}]},
        "regions": {"word_count": words, "text_area_share": text_area,
                    "ocr_available": True, "faces": 1,
                    "brand": brand, "cta": cta, "prices": []},
        "mass": {"text": text_mass, "faces": faces_mass, "brand": None,
                 "cta": None, "largest_text_box": text_mass * 0.4},
    }


def summary(**overrides):
    base = {
        "frames": 10, "duration_seconds": 5.0, "shot_count": 2,
        "cut_rate_per_minute": 8.0, "sample_fps": 2.0,
        "ocr_operational_fraction": 1.0, "frames_with_text_fraction": 1.0,
        "mean_concentration": 0.25, "max_words_on_screen": 6,
        "overloaded_shots": [], "reading_speed_words_per_second": 4.0,
        "brand": {"detected": False, "first_appearance_seconds": None,
                  "first_appearance_fraction": None, "exposure_frames": 0,
                  "exposure_seconds": 0.0},
        "cta": {"detected": False, "weak": False},
        "prices_detected": False, "faces_detected": True,
    }
    base.update(overrides)
    return base


def media(**overrides):
    base = {"duration_seconds": 5.0, "sample_fps": 2.0, "kind": "video"}
    base.update(overrides)
    return base


# =========================================================================== #
# Timeline
# =========================================================================== #
def test_a_missing_signal_is_excluded_not_treated_as_zero():
    """The index only needs ONE available term, so a fabricated zero would let a
    frame with nothing measured produce a confident number built entirely on
    that fabrication. This is what made a stub run score 43 for engagement."""
    blank = {"index": 0, "t": 0.0, "shot": 0, "saliency": {}, "regions": {},
             "mass": {}, "motion_energy": None, "gaze_stability": None,
             "seconds_into_shot": None}
    signals = tl.signals_for(blank, reading_speed=4.0)

    assert all(v is None for v in signals.values()), signals
    assert tl._index(signals, {k: 1.0 for k in signals}) is None


def test_ocr_failure_leaves_text_load_unknown_rather_than_none_on_screen():
    unread = record(words=None)
    assert tl.signals_for(unread, 4.0)["text_load"] is None

    read = record(words=8)
    assert tl.signals_for(read, 4.0)["text_load"] is not None


def test_the_index_is_absolute_so_two_ads_are_comparable():
    """Standardising within one creative would force every ad to average the
    same score, making 'attention fell to 26' meaningless across ads."""
    dull = [record(t=i / 2, concentration=0.09, motion=0.001, stability=0.35,
                   faces_mass=0.0) for i in range(6)]
    lively = [record(t=i / 2, concentration=0.42, motion=0.10, stability=0.9,
                     faces_mass=0.3) for i in range(6)]

    dull_mean = tl.build(dull, media())["points"][2]["attention"]
    lively_mean = tl.build(lively, media())["points"][2]["attention"]
    assert lively_mean > dull_mean + 10


def test_a_single_dip_is_not_a_weak_zone():
    """A viewer does not disengage because one sampled frame was flat. The
    duration requirement is what separates a dip from a hole."""
    points = [{"t": 0.0, "attention": 70}, {"t": 0.5, "attention": 20},
              {"t": 1.0, "attention": 70}]
    assert tl.weak_zones(points, threshold=35, min_seconds=1.0,
                         sample_fps=2.0) == []

    sustained = [{"t": 0.0, "attention": 70}] + \
                [{"t": 0.5 + i * 0.5, "attention": 20} for i in range(4)] + \
                [{"t": 3.0, "attention": 70}]
    zones = tl.weak_zones(sustained, 35, 1.0, 2.0)
    assert len(zones) == 1
    assert zones[0]["seconds"] == pytest.approx(2.0)
    assert zones[0]["min_attention"] == 20


def test_an_empty_frame_cannot_become_the_hero_moment():
    """A near-black end card drives the saliency map onto one artefact and reads
    as maximum concentration - the highest possible score from a frame with
    nothing in it."""
    frames = [record(t=0.0, concentration=0.2),
              record(t=0.5, concentration=0.3),
              record(t=1.0, concentration=1.0, contrast=0.01)]   # blank end card
    points = tl.build(frames, media())["points"]

    moments = tl.key_moments(points, frames, [])
    peak = next(m for m in moments if m["label"] == "peak")
    assert peak["time"] != 1.0, "the blank frame must not win"


def test_the_timeline_discloses_that_its_coefficients_are_placeholders():
    built = tl.build([record(t=i / 2) for i in range(4)], media())
    assert built["basis"] == "derived_composite"
    assert built["coefficients"] == "uniform_placeholder"
    assert built["calibration"] == "provisional_absolute"


# =========================================================================== #
# Scores
# =========================================================================== #
def test_every_metric_reports_null_with_a_reason_when_nothing_was_measured():
    blank = [{"index": 0, "t": 0.0, "shot": 0, "saliency": {}, "regions": {},
              "mass": {}}]
    result = sc.score_all(blank, summary(frames_with_text_fraction=0.0),
                          {"points": []})

    for metric_id, entry in result["scores"].items():
        assert entry["score"] is None, metric_id
        assert entry["reason"], f"{metric_id}: null with no stated reason"
    assert result["overall"]["score"] is None
    assert result["overall"]["metrics_scored"] == 0


def test_brand_memory_is_null_not_zero_when_no_brand_was_detected():
    """Zero would read as 'this brand is unmemorable'. The truth is that we did
    not find it - possibly because no wordmark was supplied."""
    result = sc.brand_memory([record()], summary(), {"points": []},
                             fw.load_framework())
    assert result["score"] is None
    assert result["reason"] == sc.NO_BRAND_DETECTED


def test_brand_memory_rewards_appearing_during_high_attention():
    """The signal that distinguishes a logo that is SEEN from one that is merely
    present. A logo held through a dead zone is exposure without memory."""
    cfg = fw.load_framework()
    brand = {"box": [0.7, 0.05, 0.95, 0.12], "basis": "wordmark_template_match"}
    frames = [record(t=i / 2, brand=brand if i < 4 else None) for i in range(8)]

    attended = {"points": [{"t": i / 2, "attention": 90 if i < 4 else 20}
                           for i in range(8)]}
    ignored = {"points": [{"t": i / 2, "attention": 20 if i < 4 else 90}
                          for i in range(8)]}

    facts = summary(brand={"detected": True, "first_appearance_seconds": 0.0,
                           "first_appearance_fraction": 0.0,
                           "exposure_frames": 4, "exposure_seconds": 2.0})

    high = sc.brand_memory(frames, facts, attended, cfg)["score"]
    low = sc.brand_memory(frames, facts, ignored, cfg)["score"]
    assert high > low, "a logo shown during a dead zone must score lower"


def test_cognitive_demand_is_driven_by_unreadable_slides():
    """63 words held for 2 seconds cannot be read at 4 words/second. That is the
    measurement the dead-zone finding is built on."""
    cfg = fw.load_framework()
    calm = sc.cognitive_demand([record()], summary(), cfg)["score"]
    overloaded = sc.cognitive_demand([record()], summary(
        overloaded_shots=[{"shot": 0, "words": 63, "seconds_available": 2.0,
                           "seconds_needed": 15.75}]), cfg)["score"]

    assert overloaded > calm + 20
    # Lower is better here, and the metric definition must say so.
    assert fw.metric_index(cfg)["cognitive_demand"]["direction"] == "lower_better"


def test_focus_is_measured_against_chance_not_as_a_raw_share():
    """A raw share cannot distinguish 'the eye is drawn to the headline' from
    'the headline is simply large'."""
    cfg = fw.load_framework()
    drawn = [record(text_mass=0.30, text_area=0.10)]     # 3.0x chance
    ignored = [record(text_mass=0.10, text_area=0.10)]   # exactly chance

    assert sc.focus(drawn, cfg)["score"] > sc.focus(ignored, cfg)["score"]
    assert sc.focus(ignored, cfg)["score"] == 0
    assert sc.focus(drawn, cfg)["signals"]["copy_attention_vs_chance"] == 3.0


def test_a_weak_cta_costs_clarity():
    cfg = fw.load_framework()
    cta = {"text": "apply", "strength": "strong", "box": [0.3, 0.8, 0.7, 0.87]}
    weak = {"text": "learn more", "strength": "weak", "box": [0.3, 0.8, 0.7, 0.87]}
    frames = [record(t=i / 2, cta=cta) for i in range(6)]
    weak_frames = [record(t=i / 2, cta=weak) for i in range(6)]

    strong_score = sc.clarity(frames, summary(
        cta={"detected": True, "weak": False}), cfg)["score"]
    weak_score = sc.clarity(weak_frames, summary(
        cta={"detected": True, "weak": True}), cfg)["score"]
    assert strong_score > weak_score


def test_lower_is_better_metrics_are_inverted_before_the_overall():
    cfg = fw.load_framework()
    scores = {
        "attention": {"score": 80}, "focus": {"score": 80},
        "cognitive_demand": {"score": 100},      # worst possible demand
        "clarity": {"score": 80}, "brand_memory": {"score": 80},
        "engagement": {"score": 80},
    }
    worst = sc.overall(scores, cfg)["score"]
    scores["cognitive_demand"] = {"score": 0}    # best possible demand
    best = sc.overall(scores, cfg)["score"]
    assert best > worst, "high cognitive demand must LOWER the overall"


def test_an_unscored_metric_is_excluded_from_the_overall_not_zeroed():
    cfg = fw.load_framework()
    full = {k: {"score": 60} for k in fw.METRIC_IDS}
    partial = dict(full, brand_memory={"score": None})

    assert sc.overall(full, cfg)["metrics_scored"] == 6
    result = sc.overall(partial, cfg)
    assert result["metrics_scored"] == 5
    assert result["metrics_missing"] == ["brand_memory"]

    # Excluded, not counted as zero. Dropping a metric shifts the mean a little
    # (the remaining five no longer average quite the same), but scoring it zero
    # would drag the overall down by ten points and report a creative as poor on
    # the strength of a measurement that was never taken.
    zeroed = dict(full, brand_memory={"score": 0})
    assert abs(result["score"] - sc.overall(full, cfg)["score"]) <= 2
    assert result["score"] > sc.overall(zeroed, cfg)["score"] + 5


def test_no_bands_are_invented_and_the_two_reasons_stay_separate():
    cfg = fw.load_framework()
    result = sc.overall({k: {"score": 70} for k in fw.METRIC_IDS}, cfg)
    assert result["band"] is None
    assert result["band_reason"] == "thresholds_not_configured"
    assert result["weighting"] == fw.WEIGHTING_EQUAL

    # Why there is no score and why there is no band are different questions.
    empty = sc.overall({k: {"score": None} for k in fw.METRIC_IDS}, cfg)
    assert empty["band_reason"] == "thresholds_not_configured"
    assert empty["score_reason"] == "no metric could be scored"


def test_the_same_measurements_always_produce_the_same_scores():
    """The point of scoring in Python rather than in a model: reproducible, and
    re-derivable under new weights without re-processing anything."""
    cfg = fw.load_framework()
    frames = [record(t=i / 2) for i in range(8)]
    line = tl.build(frames, media())

    first = sc.score_all(frames, summary(), line, cfg)
    second = sc.score_all(frames, summary(), line, cfg)
    assert first["scores"] == second["scores"]
    assert first["overall"] == second["overall"]


# =========================================================================== #
# Defects - the measured half of the fix recommendations
# =========================================================================== #
from vision_lab import defects as df  # noqa: E402


def test_every_defect_carries_the_numbers_that_produced_it():
    """Milestone D writes prose ABOUT these. A defect with no measured values
    is one the model would have to invent detail for, and evidence.py would then
    have nothing to verify the prose against."""
    frames = [record(t=i / 2, brand=None, cta=None) for i in range(8)]
    line = tl.build(frames, media())
    found = df.detect(frames, summary(), line, fw.load_framework())

    assert found, "a creative with no brand and no CTA has defects"
    for defect in found:
        assert defect["defect_id"]
        assert defect["severity"] in ("high", "medium", "low")
        assert defect["scores_impacted"], defect["defect_id"]
        assert defect["note"], defect["defect_id"]


def test_a_missing_brand_and_a_late_brand_are_different_findings():
    cfg = fw.load_framework()
    frames = [record(t=i / 2) for i in range(8)]
    line = tl.build(frames, media())

    absent = df.detect(frames, summary(), line, cfg)
    assert any(d["defect_id"] == df.BRAND_ABSENT for d in absent)

    late = df.detect(frames, summary(
        duration_seconds=10.0,
        brand={"detected": True, "first_appearance_seconds": 7.0,
               "first_appearance_fraction": 0.7, "exposure_frames": 4,
               "exposure_seconds": 2.0}), line, cfg)
    ids = {d["defect_id"] for d in late}
    assert df.LATE_BRAND in ids
    assert df.BRAND_ABSENT not in ids


def test_an_overloaded_slide_reports_the_arithmetic_a_client_can_check():
    """'34 words on screen for 2.5 s' is checkable. 'The pacing feels rushed'
    is not. That difference is the point of the whole defect list."""
    cfg = fw.load_framework()
    frames = [record(t=i / 2) for i in range(4)]
    found = df.detect(frames, summary(overloaded_shots=[
        {"shot": 0, "words": 34, "seconds_available": 2.5,
         "seconds_needed": 8.5}]), tl.build(frames, media()), cfg)

    slide = next(d for d in found if d["defect_id"] == df.OVERLOADED_SLIDE)
    assert slide["measured"]["words"] == 34
    assert slide["measured"]["seconds_available"] == 2.5
    assert slide["measured"]["seconds_needed"] == 8.5
    assert slide["measured"]["reading_speed_words_per_second"] == 4.0
    assert slide["severity"] == "high"       # 3.4x overrun
    assert "cognitive_demand" in slide["scores_impacted"]


def test_defects_are_ranked_worst_first():
    cfg = fw.load_framework()
    frames = [record(t=i / 2) for i in range(6)]
    found = df.detect(frames, summary(overloaded_shots=[
        {"shot": 0, "words": 40, "seconds_available": 2.0,
         "seconds_needed": 10.0}]), tl.build(frames, media()), cfg)

    order = {"high": 0, "medium": 1, "low": 2}
    severities = [order[d["severity"]] for d in found]
    assert severities == sorted(severities), "worst must come first"


def test_a_brand_on_screen_but_never_attended_is_its_own_defect():
    """Exposure without memory - the distinction Brand Memory exists to make."""
    cfg = fw.load_framework()
    brand = {"box": [0.7, 0.05, 0.95, 0.12], "basis": "wordmark_template_match"}
    frames = [record(t=i / 2, brand=brand, concentration=0.1 if i < 6 else 0.4)
              for i in range(8)]
    line = tl.build(frames, media(duration_seconds=4.0))

    found = df.detect(frames, summary(
        duration_seconds=4.0,
        brand={"detected": True, "first_appearance_seconds": 0.0,
               "first_appearance_fraction": 0.0, "exposure_frames": 8,
               "exposure_seconds": 4.0}), line, cfg)

    unseen = [d for d in found if d["defect_id"] == df.BRAND_UNSEEN]
    if unseen:
        assert unseen[0]["measured"]["attention_peak_overlap"] < 0.15
        assert unseen[0]["scores_impacted"] == ["brand_memory"]
