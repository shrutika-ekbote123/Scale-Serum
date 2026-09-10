"""
Propose annotation regions from detection, for a human to correct.

    python scripts/bakeoff/propose_regions.py
    python scripts/bakeoff/propose_regions.py --preview      (contact sheets)

WHY DETECTION RATHER THAN EYEBALLING
    Marking 100 frames by hand means estimating box coordinates by eye, which is
    approximate at best. OCR already knows exactly where the text is, to the
    pixel, and YuNet knows exactly where the faces are. Those are measurements,
    not opinions - so the machine should place the boxes and the human should
    judge what they MEAN.

    That split is the point. Coordinates come from detection; `importance` -
    which of these the ad is actually built around - is the judgement, and it is
    the part that has to be reviewed.

IS THIS CIRCULAR?
    A fair question, since the ground truth would then be partly built by our
    own OCR. It is not, for one specific reason: OCR answers "where is there
    text", which is an objective fact about the pixels, like measuring with a
    ruler. It does not answer "where should a viewer look" - that is what the
    saliency model is being tested on, and no candidate model gets to see the
    OCR output.

    The one real exposure is that text OCR misses is text the ground truth
    misses too. That penalises every candidate equally, so it cannot favour one
    model over another - but it does mean the absolute scores are a floor rather
    than an exact figure.

WHAT IT WILL GET WRONG
    Grouping. OCR returns words; a headline is a block of them. Words are
    merged into lines by vertical overlap and horizontal proximity, which
    handles subtitles well and multi-column layouts badly. Review the preview
    sheets rather than trusting the JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))
os.environ.setdefault("VL_MODEL_DIR", os.path.join(REPO, ".models"))

from vision_lab import regions as reg  # noqa: E402

WORKSPACE = os.environ.get("VL_BAKEOFF_DIR", os.path.join(HERE, "workspace"))
FRAMES_DIR = os.path.join(WORKSPACE, "ad_frames")
PREVIEW_DIR = os.path.join(FRAMES_DIR, "_preview")

BRAND_NAMES = ["Lawttorney", "Directors' Institute", "Directors Institute",
               "World Development Corporation", "BoardSearch"]

COLOURS = {"headline": (74, 102, 240), "face": (127, 201, 78),
           "brand": (255, 149, 127), "product": (74, 176, 232),
           "cta": (207, 182, 63), "proof": (216, 125, 199)}


# --------------------------------------------------------------------------- grouping
def merge_lines(boxes: list[dict], gap: float = 0.09) -> list[list[dict]]:
    """Group OCR words into lines.

    Two words belong to the same line when they overlap vertically by most of
    their height and sit close horizontally. Subtitles - one or two centred
    lines - are what these creatives use, and this handles those well.
    """
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: (b["box"][1], b["box"][0]))
    lines: list[list[dict]] = []
    for word in ordered:
        x0, y0, x1, y1 = word["box"]
        placed = False
        for line in lines:
            ly0 = min(w["box"][1] for w in line)
            ly1 = max(w["box"][3] for w in line)
            lx1 = max(w["box"][2] for w in line)
            lx0 = min(w["box"][0] for w in line)
            overlap = min(y1, ly1) - max(y0, ly0)
            height = min(y1 - y0, ly1 - ly0)
            # 0.3 rather than 0.5: Devanagari glyphs carry matras above and
            # below the baseline, so a Hindi word's box is taller than the Latin
            # word beside it on the SAME line. A strict overlap test splits
            # "AI aapki Practice ko" into three regions.
            if height > 0 and overlap / height > 0.3 and (x0 - lx1) < gap and (lx0 - x1) < gap:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])
    return lines


def line_box(line: list[dict]) -> list[float]:
    return [round(min(w["box"][0] for w in line), 4),
            round(min(w["box"][1] for w in line), 4),
            round(max(w["box"][2] for w in line), 4),
            round(max(w["box"][3] for w in line), 4)]


def looks_like_brand(text: str) -> bool:
    lowered = text.lower()
    return any(name.lower().split()[0] in lowered for name in BRAND_NAMES)


# --------------------------------------------------------------------------- propose
def propose(path: str) -> list[dict]:
    pixels = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    out: list[dict] = []

    text = reg.read_text(pixels)

    # Tiny type is not the message. A laptop screen inside the shot is full of
    # UI chrome that OCR reads perfectly and that no viewer is being asked to
    # read; so is a book spine on a shelf. Height is the cheapest signal that
    # separates "the designer set this to be read" from "this happens to be
    # legible", and it is the designer's own decision rather than our guess.
    readable = [b for b in (text.get("text_boxes") or [])
                if b.get("height_share", 0) >= 0.012]

    lines = []
    for line in merge_lines(readable):
        words = " ".join(w["text"] for w in line)
        box = line_box(line)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < 0.004:
            continue                       # a stray token, not a message
        lines.append({"box": box, "area": area, "words": words,
                      "height": max(w["height_share"] for w in line)})

    # Rank by area and treat only the largest few as the claim. A frame has one
    # or two things it wants read; anything past that is set dressing, and
    # marking eight regions "primary" would flatten the ground truth into "look
    # everywhere", which scores every model the same.
    lines.sort(key=lambda item: item["area"], reverse=True)
    for rank, line in enumerate(lines):
        kind = "brand" if looks_like_brand(line["words"]) else "headline"
        primary = (kind != "brand" and rank < 3 and line["height"] >= 0.018)
        out.append({"box": line["box"], "kind": kind,
                    "importance": "primary" if primary else "secondary",
                    "marked_by": "detector", "note": line["words"][:70]})

    faces = [f for f in (reg.find_faces(pixels).get("faces") or [])
             if f.get("confidence", 0) >= 0.6]

    # Drop "text" sitting on top of a face. OCR hallucinates tokens out of hair,
    # skin texture and shirt folds, and those land as small confident boxes in
    # the middle of a person. Real on-screen copy is never laid over the
    # presenter's face - the designer would not do that - so an overlap is
    # evidence of a false read rather than of a caption.
    out = [r for r in out
           if not any(_overlaps(r["box"], f["box"], 0.35) for f in faces)]

    for face in faces:
        out.append({"box": [round(v, 4) for v in face["box"]], "kind": "face",
                    "importance": "primary", "marked_by": "detector",
                    "note": f"face conf {face['confidence']:.2f}"})

    # A frame asks a viewer to look at a handful of things, not a dozen. Keeping
    # the largest few stops a busy background - a bookshelf, a courtroom emblem,
    # a screen behind the subject - from burying the actual message under
    # equally-weighted clutter.
    out.sort(key=lambda r: (r["importance"] != "primary", -_area(r["box"])))
    return out[:6]


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlaps(a: list[float], b: list[float], threshold: float) -> bool:
    """Does `a` sit mostly inside `b`? Intersection over a's own area, not IoU -
    a small false token inside a large face box has a tiny IoU but is entirely
    contained, which is exactly the case being caught."""
    inter = (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
             * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))
    own = _area(a)
    return own > 0 and (inter / own) > threshold


# --------------------------------------------------------------------------- preview
def contact_sheet(frames: list[dict], annotations: dict, out_path: str,
                  cols: int = 3, width: int = 300) -> None:
    tiles = []
    for f in frames:
        im = cv2.imread(os.path.join(FRAMES_DIR, f["file"]))
        h, w = im.shape[:2]
        for r in annotations.get(f["file"], []):
            x0, y0, x1, y1 = [int(v * (w if i % 2 == 0 else h))
                              for i, v in enumerate(r["box"])]
            colour = COLOURS.get(r["kind"], (255, 255, 255))
            cv2.rectangle(im, (x0, y0), (x1, y1), colour,
                          5 if r["importance"] == "primary" else 2)
            tag = r["kind"][:4]
            cv2.rectangle(im, (x0, max(0, y0 - 30)), (x0 + len(tag) * 20 + 12, y0),
                          colour, -1)
            cv2.putText(im, tag, (x0 + 6, max(22, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        label = f["file"].replace(".png", "")
        cv2.rectangle(im, (0, 0), (w, 34), (20, 20, 20), -1)
        cv2.putText(im, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (width, int(width * h / w))))

    rows = (len(tiles) + cols - 1) // cols
    cw = max(t.shape[1] for t in tiles)
    ch = max(t.shape[0] for t in tiles)
    pad = 6
    sheet = np.full(((ch + pad) * rows + pad, (cw + pad) * cols + pad, 3), 24, np.uint8)
    for index, tile in enumerate(tiles):
        r_, c_ = divmod(index, cols)
        y, x = pad + r_ * (ch + pad), pad + c_ * (cw + pad)
        sheet[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    cv2.imwrite(out_path, sheet)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="also write contact sheets for review")
    parser.add_argument("--batch", type=int, default=9,
                        help="frames per contact sheet")
    args = parser.parse_args()

    manifest = json.load(open(os.path.join(FRAMES_DIR, "manifest.json"),
                              encoding="utf-8"))
    frames = manifest["frames"]

    existing_path = os.path.join(FRAMES_DIR, "annotations.json")
    existing = {}
    if os.path.isfile(existing_path):
        loaded = json.load(open(existing_path, encoding="utf-8"))
        existing = {f["file"]: f for f in loaded.get("frames", [])}

    print(f"\nOCR: {reg.available_languages()}  |  "
          f"faces: {'YuNet' if reg.face_detector() else 'UNAVAILABLE'}\n")

    proposals: dict[str, list[dict]] = {}
    for index, f in enumerate(frames, 1):
        prior = existing.get(f["file"], {})
        # Never overwrite work that was done or checked by a person.
        if prior.get("regions") and any(
                r.get("marked_by") in ("human", "assistant") for r in prior["regions"]):
            proposals[f["file"]] = prior["regions"]
            continue
        proposals[f["file"]] = propose(os.path.join(FRAMES_DIR, f["file"]))
        if index % 10 == 0:
            print(f"  {index}/{len(frames)} frames")

    out = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "note": ("Regions a viewer SHOULD look at. Boxes fractional (0-1). "
                 "Primary weighs double secondary. `marked_by` records the "
                 "source of each box: detector = OCR/YuNet coordinates with a "
                 "guessed importance, assistant = drawn by eye, human = drawn "
                 "or corrected by a person. evaluate.py reports human-reviewed "
                 "frames separately."),
        "frames": [{
            "file": f["file"], "source": f["source"], "t": f["t"],
            "width": f["width"], "height": f["height"],
            "regions": proposals.get(f["file"], []),
            "skipped": False,
            "reviewed_by_human": bool(existing.get(f["file"], {})
                                      .get("reviewed_by_human")),
            "annotated": bool(proposals.get(f["file"])),
        } for f in frames],
    }
    with open(existing_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, ensure_ascii=False)

    total = sum(len(f["regions"]) for f in out["frames"])
    done = sum(1 for f in out["frames"] if f["regions"])
    print(f"\n{done}/{len(frames)} frames have regions, {total} in total")
    print(f"  -> {os.path.relpath(existing_path, REPO)}")

    if args.preview:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        sheets = 0
        for start in range(0, len(frames), args.batch):
            chunk = frames[start:start + args.batch]
            path = os.path.join(PREVIEW_DIR, f"sheet_{start // args.batch + 1:02d}.png")
            contact_sheet(chunk, proposals, path, cols=3)
            sheets += 1
        print(f"  {sheets} contact sheets in {os.path.relpath(PREVIEW_DIR, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
