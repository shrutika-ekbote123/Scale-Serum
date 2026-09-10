"""
Step 11 - the measurement record. Where saliency meets what is on screen.

THIS IS THE DURABLE ARTEFACT
    Everything after this point - the timeline, the six scores, the triggers,
    the fix recommendations - is arithmetic over what this file produces. It is
    stored in its own MongoDB collection, which is what makes /rescore free:
    when management sets real weights, historical creatives are re-derived from
    these records with no ffmpeg, no model and no provider call.

    So it stores measurements, never conclusions. No score is computed here.

WHAT ONE RECORD HOLDS
    Per frame: the saliency statistics, the detected regions, the mass of gaze
    landing inside each of those regions, and the motion energy since the
    previous frame. Pixels and full saliency maps are deliberately NOT kept -
    48 float32 maps at 640x360 is ~35 MB per ad with no consumer beyond
    debugging, and VL_STORE_RAW_MAPS exists for the rare case where that is
    wanted.

MOTION IS MEASURED HERE, NOT IN THE MODEL
    Image-mode saliency runs per frame and knows nothing about time. Motion
    energy - the mean absolute luma difference between consecutive samples - is
    the term that carries movement into the attention index, and it is a
    straightforward measurement rather than anything learned.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

from . import framework as fw
from . import saliency as sal

logger = logging.getLogger("vision_lab.measure")

STORE_RAW_MAPS = os.environ.get("VL_STORE_RAW_MAPS", "false").lower() == "true"
OCR_ON_SHOT_CHANGE_ONLY = os.environ.get(
    "VL_OCR_ON_SHOT_CHANGE_ONLY", "true").lower() == "true"


def frame_contrast(pixels: Optional[np.ndarray]) -> Optional[float]:
    """Standard deviation of luma, 0-1. How much is actually IN the frame.

    A near-black end card or a solid colour slate has almost none. It matters
    because a flat frame drives the saliency map degenerate: with no real
    structure to find, the whole map collapses onto one small artefact and
    concentration reads 1.0 - the highest possible "the eye is locked here"
    score, produced by a frame with nothing in it. Anything ranking frames by
    concentration has to know which ones are empty.
    """
    if pixels is None:
        return None
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return round(float(gray.std() / 255.0), 6)


def frame_detail(pixels: Optional[np.ndarray]) -> Optional[float]:
    """Mean absolute Laplacian, 0-1. How much EDGE the frame carries.

    WHY CONTRAST IS NOT ENOUGH
        frame_contrast asks "is this frame flat?" - and a fade-to-white between
        two shots is NOT flat. With letterbox bars top and bottom it is black,
        white and a headline, so its luma standard deviation is high and it sails
        through is_degenerate. It still contains nothing: no presenter, no
        product, no room. Picked as the hero frame it produced an Attention
        Report that was a near-blank page with one marker on it.

        Edges are what separates "a frame of the ad" from "a frame between two
        of them". A fade has almost none; a real scene has thousands.
    """
    if pixels is None:
        return None
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    return round(float(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)).mean())
                 / 255.0, 6)


def is_degenerate(record: dict, min_contrast: float = 0.04) -> bool:
    """A frame whose measurements cannot be trusted as attention."""
    contrast = record.get("frame_contrast")
    return contrast is not None and contrast < min_contrast


# A frame with less edge than this is a transition, not a scene.
#
# CALIBRATED, NOT GUESSED. Measured across two real 80 s creatives:
#
#   white flash frame that was being picked as the hero   0.029
#   its immediate neighbours (presenter to camera)        0.063, 0.078
#   median frame, both ads                                0.060, 0.079
#   fade to black                                         0.005
#
# 0.04 sits in the gap. Nothing above it in either ad was a transition, and the
# flash frame is excluded by a clear margin. Should a minimal creative fall
# entirely below it, hero selection falls back to every frame rather than
# rendering nothing - see _render_heatmaps.
MIN_HERO_DETAIL = 0.04


def is_representative(record: dict) -> bool:
    """Is this frame worth showing a client as THE picture of their ad?

    Deliberately separate from is_degenerate, which the timeline also uses -
    tightening that would move scores, and this is only about which single frame
    gets rendered. A frame can be perfectly sound to measure and still be the
    wrong one to put at the top of a report.
    """
    if is_degenerate(record):
        return False
    detail = record.get("frame_detail")
    return detail is None or detail >= MIN_HERO_DETAIL


def motion_energy(current: np.ndarray, previous: Optional[np.ndarray]) -> Optional[float]:
    """Mean absolute luma change since the previous sampled frame, 0-1.

    None on the first frame - there is nothing to compare it against, and
    reporting 0 would claim a still opening that we did not observe.
    """
    if previous is None:
        return None
    a = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return round(float(np.abs(a - b).mean() / 255.0), 6)


def gaze_stability(current: Optional[np.ndarray],
                   previous: Optional[np.ndarray]) -> Optional[float]:
    """How much the attention map moved between frames, as 1 - total variation.

    A settled gaze holds; a map that jumps every frame is a viewer being pulled
    around the screen. Total variation distance rather than the earth mover's
    distance the design names - EMD over a 640x360 map is far too slow to run 48
    times per job, and TV distance answers the same question well enough at a
    fraction of the cost. Recorded as `gaze_stability_metric` so a later change
    is visible rather than silent.
    """
    if current is None or previous is None:
        return None
    return round(float(1.0 - 0.5 * np.abs(current - previous).sum()), 6)


def seconds_since_shot(t: float, shots: list[float]) -> Optional[float]:
    """How long the current shot has been running. Feeds the novelty term."""
    if not shots:
        return None
    earlier = [s for s in shots if s <= t]
    return round(t - earlier[-1], 3) if earlier else round(t, 3)


def shot_index(t: float, shots: list[float]) -> int:
    return sum(1 for s in shots if s <= t)


def measure_frames(media: dict, *, wordmark: Optional[np.ndarray] = None,
                   brand_names: Optional[list[str]] = None,
                   detect_regions=None, predict_saliency=None,
                   heartbeat=None) -> list[dict]:
    """Build the per-frame record for one creative.

    `detect_regions` and `predict_saliency` are injected rather than imported so
    the whole of this file stays testable without ffmpeg, a model or an OCR
    binary present.
    """
    predict_saliency = predict_saliency or sal.predict
    if detect_regions is None:
        from . import regions as reg
        detect_regions = reg.detect

    shots = media.get("shots") or []
    frames = media.get("frames") or []
    records: list[dict] = []
    previous_pixels: Optional[np.ndarray] = None
    previous_map: Optional[np.ndarray] = None
    last_shot_seen = -1
    carried_regions: Optional[dict] = None

    for index, frame in enumerate(frames):
        pixels = frame.get("pixels")
        t = float(frame.get("t") or 0.0)
        current_shot = shot_index(t, shots)

        saliency = predict_saliency(frame)
        smap = saliency.get("map")

        # Within one continuous shot the on-screen text does not change, so OCR
        # runs on the first frame of each shot and the result is carried. This is
        # the single biggest speed lever in the pipeline.
        new_shot = current_shot != last_shot_seen
        run_ocr = new_shot or not OCR_ON_SHOT_CHANGE_ONLY or carried_regions is None
        if run_ocr:
            regions = detect_regions(frame, wordmark=wordmark,
                                     brand_names=brand_names, run_ocr=True)
            carried_regions = regions
        else:
            regions = dict(carried_regions)
            regions["ocr_reused_from_shot"] = current_shot
        last_shot_seen = current_shot

        record = {
            "index": index,
            "t": t,
            "shot": current_shot,
            "seconds_into_shot": seconds_since_shot(t, shots),
            "saliency": {
                "concentration": saliency.get("concentration"),
                "peak_count": saliency.get("peak_count"),
                "peaks": saliency.get("peaks") or [],
                "map_sum": saliency.get("map_sum"),
            },
            "motion_energy": motion_energy(pixels, previous_pixels)
            if pixels is not None else None,
            "frame_contrast": frame_contrast(pixels),
            "frame_detail": frame_detail(pixels),
            "gaze_stability": gaze_stability(smap, previous_map),
            "gaze_stability_metric": "one_minus_total_variation",
            "regions": {
                "word_count": regions.get("word_count"),
                "text_area_share": regions.get("text_area_share"),
                "ocr_available": regions.get("ocr_available"),
                "text_boxes": len(regions.get("text_boxes") or []),
                "text": regions.get("text", "")[:400],
                "faces": len(regions.get("faces") or []),
                "brand": regions.get("brand"),
                "cta": regions.get("cta"),
                "prices": regions.get("prices") or [],
            },
            # The join. Every marketing conclusion downstream is built on these.
            "mass": _mass_by_region(smap, regions),
        }
        if STORE_RAW_MAPS and smap is not None:
            record["saliency"]["map"] = smap.astype(np.float16).tolist()

        records.append(record)
        previous_pixels = pixels
        previous_map = smap

        if heartbeat and index % 10 == 0:
            heartbeat(index)

    return records


def _mass_by_region(smap: Optional[np.ndarray], regions: dict) -> dict:
    """How much predicted gaze lands on each kind of thing."""
    if smap is None:
        return {}

    text_boxes = regions.get("text_boxes") or []
    faces = regions.get("faces") or []
    brand = regions.get("brand")
    cta = regions.get("cta")

    return {
        "text": round(sum(sal.region_mass(smap, b["box"]) for b in text_boxes), 6),
        "faces": round(sum(sal.region_mass(smap, f["box"]) for f in faces), 6),
        "brand": (sal.region_mass(smap, brand["box"])
                  if brand and brand.get("box") else None),
        "cta": (sal.region_mass(smap, cta["box"])
                if cta and cta.get("box") else None),
        "largest_text_box": max(
            (sal.region_mass(smap, b["box"]) for b in text_boxes), default=0.0),
    }


def summarise(records: list[dict], media: dict) -> dict:
    """Whole-creative facts the scorer needs but should not have to recompute.

    Still measurements, not conclusions - counts, times and shares. The judgement
    about what a late brand appearance means belongs in scoring.py.
    """
    duration = media.get("duration_seconds") or 0.0
    reading_speed = fw.reading_speed()

    brand_frames = [r for r in records if (r["regions"].get("brand") or None)]
    first_brand = brand_frames[0]["t"] if brand_frames else None
    ocr_frames = [r for r in records if r["regions"].get("ocr_available")]

    words = [r["regions"]["word_count"] for r in records
             if r["regions"].get("word_count") is not None]
    concentrations = [r["saliency"]["concentration"] for r in records
                      if r["saliency"].get("concentration") is not None]

    # Reading load per shot: the words held in that shot against the time a
    # viewer actually had to read them. Above 1.0 the slide cannot be finished.
    shots: dict[int, dict] = {}
    for record in records:
        entry = shots.setdefault(record["shot"],
                                 {"words": 0, "frames": 0,
                                  "t_start": record["t"], "t_end": record["t"]})
        entry["words"] = max(entry["words"], record["regions"].get("word_count") or 0)
        entry["frames"] += 1
        # Timestamps, so a defect can point at the moment. A finding a viewer
        # cannot be shown is a finding nobody can act on.
        entry["t_start"] = min(entry["t_start"], record["t"])
        entry["t_end"] = max(entry["t_end"], record["t"])
    sample_fps = media.get("sample_fps") or 2.0
    overloaded = []
    for shot, entry in shots.items():
        seconds = entry["frames"] / sample_fps
        if entry["words"] and seconds > 0:
            needed = entry["words"] / reading_speed
            if needed > seconds:
                overloaded.append({"shot": shot, "words": entry["words"],
                                   "seconds_available": round(seconds, 2),
                                   "seconds_needed": round(needed, 2),
                                   "t_start": round(entry["t_start"], 2),
                                   "t_end": round(entry["t_end"], 2)})

    return {
        "frames": len(records),
        "duration_seconds": duration,
        "shot_count": len(shots),
        "cut_rate_per_minute": round(len(shots) / (duration / 60), 2) if duration else None,
        # Two different things, and conflating them hides a real failure.
        # "OCR ran" is not "text was found": on a Hinglish creative with the
        # wrong page-segmentation mode, OCR ran happily on every frame and
        # returned nothing, which a single `ocr_coverage: 1.0` would have
        # reported as complete success.
        "ocr_operational_fraction": round(len(ocr_frames) / len(records), 3)
        if records else None,
        "frames_with_text_fraction": round(
            sum(1 for r in records if (r["regions"].get("word_count") or 0) > 0)
            / len(records), 3) if records else None,
        "mean_concentration": round(float(np.mean(concentrations)), 6)
        if concentrations else None,
        "max_words_on_screen": max(words) if words else None,
        "overloaded_shots": overloaded,
        "reading_speed_words_per_second": reading_speed,
        "brand": {
            "detected": bool(brand_frames),
            "first_appearance_seconds": first_brand,
            "first_appearance_fraction": round(first_brand / duration, 4)
            if first_brand is not None and duration else None,
            "exposure_frames": len(brand_frames),
            "exposure_seconds": round(len(brand_frames) / sample_fps, 2)
            if brand_frames else 0.0,
        },
        "cta": {
            "detected": any(r["regions"].get("cta") for r in records),
            "weak": any((r["regions"].get("cta") or {}).get("strength") == "weak"
                        for r in records),
        },
        "prices_detected": any(r["regions"].get("prices") for r in records),
        "faces_detected": any(r["regions"].get("faces") for r in records),
    }
