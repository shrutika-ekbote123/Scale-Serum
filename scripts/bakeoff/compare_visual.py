"""
Side-by-side heatmaps, because a number is not a picture.

    python scripts/bakeoff/compare_visual.py
    python scripts/bakeoff/compare_visual.py --frames 8 --models unisal,msinet

WHY THIS EXISTS ALONGSIDE THE METRICS
    NSS says one model matches our annotations better than another. It does not
    say WHERE they differ, and "where" is what decides whether a model is right
    for ad analysis. A model can win on average while systematically ignoring
    headline text - the single thing this product is built to notice.

    Four panels per frame: the annotated ground truth, then each model's map
    over the same pixels. Looking at eight frames tells you things the summary
    statistics cannot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))

import groundtruth as gt  # noqa: E402
import model_adapters as adapters  # noqa: E402

WORKSPACE = os.environ.get("VL_BAKEOFF_DIR", os.path.join(HERE, "workspace"))
FRAMES_DIR = os.path.join(WORKSPACE, "ad_frames")
OUT_DIR = os.path.join(FRAMES_DIR, "_preview")

TILE_H = 300
ALPHA = 0.55


def overlay(pixels: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    """Heat over the frame. Normalised per frame by its own maximum, so a map
    that spreads its mass is not rendered dimmer than one that concentrates -
    both sum to 1, and brightness here means 'relatively hottest', not 'more
    total attention'."""
    h, w = pixels.shape[:2]
    resized = cv2.resize(saliency.astype(np.float32), (w, h),
                         interpolation=cv2.INTER_CUBIC)
    peak = float(resized.max())
    scaled = (resized / peak * 255).astype(np.uint8) if peak > 0 \
        else np.zeros((h, w), np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), 1 - ALPHA,
                           heat, ALPHA, 0)


def with_boxes(pixels: np.ndarray, regions: list[dict]) -> np.ndarray:
    colours = {"headline": (74, 102, 240), "face": (127, 201, 78),
               "brand": (255, 149, 127), "product": (74, 176, 232),
               "proof": (216, 125, 199), "cta": (207, 182, 63)}
    image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR).copy()
    h, w = image.shape[:2]
    for region in regions:
        x0, y0, x1, y1 = [int(v * (w if i % 2 == 0 else h))
                          for i, v in enumerate(region["box"])]
        cv2.rectangle(image, (x0, y0), (x1, y1),
                      colours.get(region["kind"], (255, 255, 255)),
                      4 if region["importance"] == "primary" else 2)
    return image


def label(tile: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (18, 18, 18), -1)
    cv2.putText(tile, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--models", default="spectral (production),unisal,msinet")
    parser.add_argument("--out", default="compare.png")
    args = parser.parse_args()

    annotations = gt.load(os.path.join(FRAMES_DIR, "annotations.json"))
    usable = [f for f in annotations["frames"]
              if f.get("reviewed_by_human") and f.get("regions")]

    # Spread across the creatives rather than taking the first N, which would
    # all come from one ad and one visual style.
    by_source: dict[str, list] = {}
    for frame in usable:
        by_source.setdefault(frame["source"], []).append(frame)
    chosen = []
    while len(chosen) < args.frames and any(by_source.values()):
        for source in list(by_source):
            if by_source[source] and len(chosen) < args.frames:
                step = max(1, len(by_source[source]) // 3)
                chosen.append(by_source[source].pop(
                    min(step, len(by_source[source]) - 1)))

    available = adapters.available()
    wanted = [m.strip() for m in args.models.split(",")]
    models = [(n, available[n]) for n in wanted if n in available]
    if not models:
        sys.exit(f"None of {wanted} are available. Have: {list(available)}")

    print(f"\n{len(chosen)} frames x {len(models)} models\n")
    rows = []
    for frame in chosen:
        path = os.path.join(FRAMES_DIR, frame["file"])
        image = cv2.imread(path)
        if image is None:
            continue
        pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        regions, _ = gt.usable_regions(frame)

        tiles = [label(with_boxes(pixels, regions), "marked by you")]
        for name, predict in models:
            saliency = predict(pixels)
            if saliency is None:
                continue
            tiles.append(label(overlay(pixels, saliency), name))

        scale = TILE_H / tiles[0].shape[0]
        tiles = [cv2.resize(t, (int(t.shape[1] * scale), TILE_H)) for t in tiles]
        rows.append((frame["file"], np.hstack(tiles)))
        print(f"  {frame['file']}")

    width = max(r.shape[1] for _, r in rows)
    padded = []
    for name, row in rows:
        if row.shape[1] < width:
            row = np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)))
        padded.append(row)
    sheet = np.vstack([np.pad(r, ((0, 8), (0, 0), (0, 0))) for r in padded])

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, args.out)
    cv2.imwrite(out, sheet)
    print(f"\n{os.path.relpath(out, REPO)}  {sheet.shape[1]}x{sheet.shape[0]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
