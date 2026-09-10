"""
Turn hand-marked regions into the ground-truth attention map a model is scored
against.

WHAT A "GROUND TRUTH" IS HERE, AND WHAT IT IS NOT
    MIT1003 ships real fixation points - where human eyes actually landed. We
    have no eye-tracker, so ours is a different thing: where the ad's designer
    INTENDED attention to go. Those are related but not identical, and the
    distinction matters when reading the scores.

    It is still the right target. A saliency model that predicts gaze onto the
    background while the headline goes unread is failing at the job Vision Lab
    needs done, however well it scores on photographs of natural scenes.

HOW A MAP IS BUILT
    Each region becomes a Gaussian blob rather than a hard rectangle. Human
    fixation maps are smooth - the eye lands near a target, not exactly on its
    bounding box - so scoring against sharp edges would punish a model for being
    approximately right, which is the only thing any of them can be.

    Primary regions weigh double secondary ones. The map is then normalised so
    every frame contributes equally regardless of how many regions it has.

WHAT GETS EXCLUDED, AND WHY IT IS SAID OUT LOUD
    Two filters, both reported in the output rather than applied silently:

    * Junk OCR tokens the reviewer left in place ("i", ">", "Rn"). These are
      artefacts of detection, not decisions about the creative.
    * Regions covering more than `max_area` of the frame. A box over 60% of the
      image says "look anywhere", which cannot separate a good model from a bad
      one - every candidate scores well against it, so it adds noise and no
      signal.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import cv2
import numpy as np

PRIMARY_WEIGHT = 2.0
SECONDARY_WEIGHT = 1.0

# A blob's spread, as a fraction of the region's own size. Roughly a degree of
# visual angle at normal viewing distance - the scale over which fixations
# scatter around a target.
SIGMA_FRACTION = 0.28
MIN_SIGMA_PX = 6.0


def is_junk(region: dict) -> bool:
    """A detector artefact rather than a judgement about the creative.

    Alphanumerics are counted, NOT just letters: "9600" is the entire point of
    the frame it appears on, and a letters-only test would throw it away as
    noise.
    """
    if region.get("marked_by") != "detector":
        return False
    note = region.get("note") or ""
    if note.startswith("face conf"):
        return False
    meaningful = re.sub(r"[^0-9A-Za-zऀ-ॿ]", "", note)
    return len(meaningful) < 3


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def usable_regions(frame: dict, *, trust: str = "all",
                   max_area: float = 0.6) -> tuple[list[dict], dict]:
    """The regions that count, plus why the others did not."""
    dropped = {"junk": 0, "too_large": 0, "untrusted": 0}
    kept = []
    for region in frame.get("regions") or []:
        if trust == "human" and region.get("marked_by") != "human":
            dropped["untrusted"] += 1
            continue
        if is_junk(region):
            dropped["junk"] += 1
            continue
        if area(region["box"]) > max_area:
            dropped["too_large"] += 1
            continue
        kept.append(region)
    return kept, dropped


def build_map(regions: list[dict], height: int, width: int) -> Optional[np.ndarray]:
    """A smooth importance map, normalised to sum 1 - the same convention every
    predicted saliency map uses, so the two are directly comparable."""
    if not regions:
        return None
    canvas = np.zeros((height, width), dtype=np.float32)

    for region in regions:
        x0, y0, x1, y1 = region["box"]
        cx, cy = (x0 + x1) / 2 * width, (y0 + y1) / 2 * height
        rw, rh = max(1.0, (x1 - x0) * width), max(1.0, (y1 - y0) * height)
        sx = max(MIN_SIGMA_PX, rw * SIGMA_FRACTION)
        sy = max(MIN_SIGMA_PX, rh * SIGMA_FRACTION)
        weight = (PRIMARY_WEIGHT if region.get("importance") == "primary"
                  else SECONDARY_WEIGHT)

        ys = np.arange(height, dtype=np.float32)[:, None]
        xs = np.arange(width, dtype=np.float32)[None, :]
        blob = np.exp(-(((xs - cx) ** 2) / (2 * sx ** 2)
                        + ((ys - cy) ** 2) / (2 * sy ** 2)))
        canvas += weight * blob

    total = float(canvas.sum())
    return canvas / total if total > 0 else None


def fixation_points(regions: list[dict], height: int, width: int) -> np.ndarray:
    """A binary map of region centres, for the metrics that need discrete
    fixation locations (NSS, AUC) rather than a continuous distribution."""
    points = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        x0, y0, x1, y1 = region["box"]
        cx = int(np.clip((x0 + x1) / 2 * width, 0, width - 1))
        cy = int(np.clip((y0 + y1) / 2 * height, 0, height - 1))
        points[cy, cx] = 1
        # Primary regions get a second point, so they carry more weight in NSS
        # exactly as they do in the continuous map.
        if region.get("importance") == "primary":
            offset_y = int(np.clip(cy + (y1 - y0) * height * 0.12, 0, height - 1))
            points[offset_y, cx] = 1
    return points


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarise(annotations: dict, *, trust: str = "all",
              max_area: float = 0.6) -> dict:
    frames, dropped = [], {"junk": 0, "too_large": 0, "untrusted": 0}
    for frame in annotations.get("frames", []):
        kept, drops = usable_regions(frame, trust=trust, max_area=max_area)
        for key, value in drops.items():
            dropped[key] += value
        if kept:
            frames.append({**frame, "regions": kept})
    return {
        "frames": frames,
        "usable": len(frames),
        "total": len(annotations.get("frames", [])),
        "regions": sum(len(f["regions"]) for f in frames),
        "reviewed": sum(1 for f in frames if f.get("reviewed_by_human")),
        "dropped": dropped,
    }
