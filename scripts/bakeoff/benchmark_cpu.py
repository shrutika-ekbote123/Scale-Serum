"""
Step 8, part 5 - how fast is each candidate on the machine that will run it?

    python scripts/bakeoff/benchmark_cpu.py
    python scripts/bakeoff/benchmark_cpu.py --frames 20 --threads 1

WHY CPU, AND WHY THIS IS NOT OPTIONAL
    The bake-off's accuracy numbers come off a GPU because that makes the
    comparison quick. Production has no GPU: one CPU VPS, one core to spare,
    shared with onboarding, Script Lab, purchase probability and the Sales Call
    Analyzer.

    So a model's GPU speed is irrelevant to whether we can ship it. UNISAL at
    112 ms/frame on an A1000 says nothing about what it costs on the box that
    matters, and the answer decides as much as accuracy does.

THE BUDGET
    A 24 s ad at 2 fps is 48 frames; a 84 s ad thinned to fit the cap is 120.
    Saliency is one of several stages - ffmpeg decode, OCR, heatmap render and
    two network calls all want the same core. Roughly:

        50 ms/frame  ->   6 s over 120 frames   comfortable
       200 ms/frame  ->  24 s                   acceptable
       500 ms/frame  ->  60 s                   doubles the job; needs thought
      1000 ms/frame  -> 120 s                   too slow at concurrency 1

    Those are engineering judgements, not business rules, and the deploy budget
    in VISION_LAB_PLAN.md is what they should be checked against.

SINGLE-THREADED BY DEFAULT
    The worker runs at concurrency 1 beside other services. Measuring with every
    core available would flatter a heavy model and mislead about what happens
    when onboarding is busy at the same time.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

WORKSPACE = os.environ.get("VL_BAKEOFF_DIR", os.path.join(HERE, "workspace"))
FRAMES_DIR = os.path.join(WORKSPACE, "ad_frames")
REPORT = os.path.join(WORKSPACE, "cpu_benchmark.json")

# What a job actually costs, at the two ends of the range we see.
SHORT_AD_FRAMES = 48
LONG_AD_FRAMES = 120


def verdict(ms: float) -> str:
    if ms < 60:
        return "comfortable"
    if ms < 200:
        return "acceptable"
    if ms < 500:
        return "slow - would dominate the job"
    return "too slow for this box"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=15,
                        help="how many real frames to time over")
    parser.add_argument("--threads", type=int, default=1,
                        help="0 leaves the library default (all cores)")
    parser.add_argument("--only", help="comma-separated model ids")
    args = parser.parse_args()

    # Pin threads BEFORE the libraries load - they read these at import.
    if args.threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(args.threads)
    # CPU only, whatever hardware is present - this is the point of the file.
    #
    # "-1", not "". An EMPTY CUDA_VISIBLE_DEVICES leaves torch.cuda.is_available()
    # returning True while device_count() is 0, so a library that picks its
    # device from is_available() selects cuda:0 and then dies loading onto it.
    # "-1" makes both agree.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import model_adapters as adapters

    manifest = json.load(open(os.path.join(FRAMES_DIR, "manifest.json"),
                              encoding="utf-8"))
    chosen = manifest["frames"][::max(1, len(manifest["frames"]) // args.frames)]
    chosen = chosen[:args.frames]
    frames = []
    for entry in chosen:
        image = cv2.imread(os.path.join(FRAMES_DIR, entry["file"]))
        if image is not None:
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not frames:
        sys.exit("No frames to benchmark.")

    available = adapters.available()
    if args.only:
        wanted = {v.strip() for v in args.only.split(",")}
        available = {k: v for k, v in available.items() if k in wanted}

    print(f"\n{platform.processor() or platform.machine()}")
    print(f"threads: {args.threads or 'default (all cores)'}  |  "
          f"{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    print("\nCPU ONLY - this is the number that decides deployability.\n")
    print(f"{'model':<24}{'ms/frame':>10}{'48 frames':>12}{'120 frames':>12}   verdict")
    print("-" * 82)

    results = []
    for name, predict in available.items():
        predict(frames[0])                     # warm up: first call loads weights
        timings = []
        for frame in frames:
            start = time.perf_counter()
            result = predict(frame)
            if result is None:
                break
            timings.append((time.perf_counter() - start) * 1000)
        if not timings:
            continue

        ms = float(np.median(timings))         # median, not mean: a stray GC
                                               # pause should not define the number
        row = {
            "model": name,
            "ms_per_frame_median": round(ms, 1),
            "ms_per_frame_p90": round(float(np.percentile(timings, 90)), 1),
            "short_ad_seconds": round(ms * SHORT_AD_FRAMES / 1000, 1),
            "long_ad_seconds": round(ms * LONG_AD_FRAMES / 1000, 1),
            "verdict": verdict(ms),
        }
        results.append(row)
        print(f"{name:<24}{row['ms_per_frame_median']:>10.1f}"
              f"{row['short_ad_seconds']:>11.1f}s{row['long_ad_seconds']:>11.1f}s"
              f"   {row['verdict']}")

    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "machine": platform.processor() or platform.machine(),
            "threads": args.threads,
            "frames": len(frames),
            "note": ("Measured on the development laptop's CPU with CUDA "
                     "disabled. The production VPS is a different chip - "
                     "re-run there before the final call, and treat these as "
                     "relative rather than absolute."),
            "results": results,
        }, handle, indent=2)

    print(f"\n{os.path.relpath(REPORT, REPO)}")
    print("\nThis laptop is not the VPS. Re-run on the server before deciding - "
          "the ranking should hold, the absolute numbers will not.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
