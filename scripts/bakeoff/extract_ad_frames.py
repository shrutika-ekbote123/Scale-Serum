"""
Step 8, part 2 - pull frames out of our own ads for annotation.

    python scripts/bakeoff/extract_ad_frames.py
    python scripts/bakeoff/extract_ad_frames.py --per-ad 30
    python scripts/bakeoff/extract_ad_frames.py --serve      (open the annotator)

WHY OUR OWN ADS AND NOT JUST MIT1003
    MIT1003 and CAT2000 are photographs. Our creatives are DESIGNED images -
    headlines, product shots, logos, buttons. A model that predicts fixations on
    natural scenes beautifully can still ignore a headline block, and no
    academic benchmark would show us that.

    Mean NSS against these hand-marked frames is the DECISION metric. The public
    benchmarks are the sanity check that a model is not simply broken.

    It is also the only part of the bake-off nobody else can do for us: what
    *should* draw the eye in a ScaleSerum ad is a judgement about what the ad is
    trying to achieve, which lives with whoever briefed it.

HOW FRAMES ARE CHOSEN
    One per shot, taken 40% of the way in so it misses the transition, then
    evenly spaced fill to reach the target. Near-black frames and end cards are
    skipped: they carry no design decision to mark, and they are exactly the
    frames that drive a saliency map degenerate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

# OpenCV, numpy and ffmpeg are imported lazily, inside the extraction path only.
# `--serve` just runs a static web server: making it fail on a missing image
# library sends whoever is annotating off to debug an install they do not need.

WORKSPACE = os.environ.get("VL_BAKEOFF_DIR", os.path.join(HERE, "workspace"))
FRAMES_DIR = os.path.join(WORKSPACE, "ad_frames")
SOURCE_DIR = os.path.join(REPO, "Ads_Video")

# Wide enough to read 20 px type when marking a headline; small enough that a
# hundred PNGs stay manageable.
ANNOTATION_WIDTH = int(os.environ.get("VL_ANNOTATION_WIDTH", 960))
MIN_CONTRAST = 0.05


def pick_times(duration: float, shots: list[float], target: int) -> list[float]:
    """Timestamps to grab: one per shot, then evenly spaced fill."""
    boundaries = sorted(set([0.0] + [s for s in shots if 0 < s < duration]))
    chosen: list[float] = []

    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else duration
        if end - start < 0.3:                    # a flash, not a shot
            continue
        chosen.append(round(start + (end - start) * 0.4, 3))

    if len(chosen) < target:
        step = duration / (target + 1)
        for index in range(1, target + 1):
            candidate = round(step * index, 3)
            if all(abs(candidate - existing) > 0.75 for existing in chosen):
                chosen.append(candidate)

    return sorted(chosen)[:target]


def _vision_tools():
    """Import the heavy dependencies, or explain exactly how to get them."""
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
        from vision_lab import media as md
        from vision_lab.measure import frame_contrast
        return cv2, np, md, frame_contrast
    except ImportError as err:
        sys.exit("\n".join([
            f"Missing a dependency: {err}",
            "",
            "Frame extraction needs OpenCV, which is installed in this repo's",
            "virtualenv rather than system-wide. Use the venv's interpreter:",
            "",
            r"    .\.venv\Scripts\python.exe scripts/bakeoff/extract_ad_frames.py",
            "",
            "If that still fails, install the dependencies into it:",
            "",
            r"    .\.venv\Scripts\python.exe -m pip install -r requirements.txt",
        ]))


def grab(path: str, times: list[float], width: int, height: int) -> list[dict]:
    """Decode exactly the timestamps we want, one ffmpeg call each.

    Seeking per frame rather than decoding the whole video: we want 25 frames
    out of ~2000, and decoding the other 1975 to throw them away is minutes of
    CPU for nothing.
    """
    _cv2, np, md, _ = _vision_tools()
    out_w, out_h = md._work_size(width, height)
    scale = ANNOTATION_WIDTH / out_w if out_w else 1
    out_w, out_h = ANNOTATION_WIDTH, int(out_h * scale) - (int(out_h * scale) % 2)

    frames = []
    for t in times:
        args = [md._binary("ffmpeg"), "-v", "error", "-ss", f"{t:.3f}", "-i", path,
                "-frames:v", "1", "-vf", f"scale={out_w}:{out_h}",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        result = md._run(args, timeout=120)
        expected = out_w * out_h * 3
        if result.returncode != 0 or len(result.stdout) < expected:
            continue
        pixels = np.frombuffer(result.stdout[:expected], dtype=np.uint8)
        frames.append({"t": t, "pixels": pixels.reshape(out_h, out_w, 3)})
    return frames


def extract(video: str, target: int) -> list[dict]:
    cv2, _np, md, frame_contrast = _vision_tools()
    name = os.path.splitext(os.path.basename(video))[0]
    info = md.probe(video)
    duration = info["duration_seconds"] or 0
    shots = md.detect_shots(video, duration)
    times = pick_times(duration, shots, target)

    print(f"  {name:<10} {duration:6.1f}s  {len(shots):3d} shots  "
          f"-> {len(times)} candidate frames", end="", flush=True)

    saved = []
    skipped = 0
    for frame in grab(video, times, info["width"], info["height"]):
        contrast = frame_contrast(frame["pixels"])
        if contrast is not None and contrast < MIN_CONTRAST:
            skipped += 1          # an end card carries no design decision to mark
            continue
        filename = f"{name}_t{frame['t']:07.2f}.png"
        cv2.imwrite(os.path.join(FRAMES_DIR, filename),
                    cv2.cvtColor(frame["pixels"], cv2.COLOR_RGB2BGR))
        saved.append({
            "file": filename,
            "source": os.path.basename(video),
            "t": frame["t"],
            "width": int(frame["pixels"].shape[1]),
            "height": int(frame["pixels"].shape[0]),
            "contrast": contrast,
        })
    print(f"  -> {len(saved)} saved" + (f", {skipped} blank skipped" if skipped else ""))
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-ad", type=int, default=25,
                        help="frames per creative (default 25 - about 100 across four)")
    parser.add_argument("--source", default=SOURCE_DIR)
    parser.add_argument("--serve", action="store_true",
                        help="start a local server and open the annotator")
    args = parser.parse_args()

    if args.serve:
        return serve()

    _cv2, _np, md, _ = _vision_tools()
    if not md.available():
        sys.exit("ffmpeg is not available. Install it, or set VL_FFMPEG_DIR.")
    if not os.path.isdir(args.source):
        sys.exit(f"No such directory: {args.source}")

    videos = sorted(os.path.join(args.source, f) for f in os.listdir(args.source)
                    if f.lower().endswith((".mp4", ".mov", ".webm", ".m4v")))
    if not videos:
        sys.exit(f"No videos in {args.source}")

    os.makedirs(FRAMES_DIR, exist_ok=True)
    print(f"\nExtracting up to {args.per_ad} frames from each of "
          f"{len(videos)} creative(s)\n")

    manifest = []
    for video in videos:
        manifest.extend(extract(video, args.per_ad))

    path = os.path.join(FRAMES_DIR, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"version": 1,
                   "created": datetime.now(timezone.utc).isoformat(),
                   "annotation_width": ANNOTATION_WIDTH,
                   "frames": manifest}, handle, indent=2)

    write_annotator(manifest)
    print(f"\n{len(manifest)} frames in {os.path.relpath(FRAMES_DIR, REPO)}")
    print("\nNow mark them up:")
    print("    python scripts/bakeoff/extract_ad_frames.py --serve\n")
    return 0


def serve() -> int:
    """Serve the frames directory so the annotator can read its own manifest.

    Opened straight off the filesystem, the browser blocks the page from
    fetching manifest.json - a file:// page cannot read sibling files. A local
    server sidesteps that, and costs one command.
    """
    import http.server
    import socketserver
    import threading
    import webbrowser

    if not os.path.isfile(os.path.join(FRAMES_DIR, "annotate.html")):
        sys.exit("No frames yet. Run without --serve first.")

    os.chdir(FRAMES_DIR)
    port = int(os.environ.get("VL_ANNOTATE_PORT", 8765))
    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("127.0.0.1", port), handler) as server:
        url = f"http://127.0.0.1:{port}/annotate.html"
        print(f"\nAnnotator: {url}")
        print("Mark every frame, click Download, and save the JSON next to the "
              "frames.\nCtrl+C when you are done.\n")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


def write_annotator(manifest: list[dict]) -> None:
    template = os.path.join(HERE, "annotate_template.html")
    with open(template, "r", encoding="utf-8") as handle:
        html = handle.read()
    with open(os.path.join(FRAMES_DIR, "annotate.html"), "w", encoding="utf-8") as out:
        out.write(html)


if __name__ == "__main__":
    sys.exit(main())
