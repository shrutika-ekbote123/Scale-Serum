"""
Stand-ins for the vision layer, used until Milestone B replaces them.

WHY THESE EXIST
    They make the whole service real - endpoints, statuses, storage, worker,
    idempotency, failure envelopes - weeks before ffmpeg, ONNX and OCR arrive.
    Everything above the vision layer can then be built and tested against a
    running system instead of against a schema.

    They are also what CI uses. A GitHub runner has no ffmpeg binary and no
    model weights; with these injected, every test above the CV layer runs on a
    bare runner.

WHAT THEY DELIBERATELY DO NOT DO
    They do not invent plausible-looking numbers. Saliency mass is zero,
    regions are empty, no heatmap is written. A stub that produced convincing
    output would eventually be shown to someone as a result, and the point of a
    stub is to be obviously unfinished.

REPLACED BY
    sample_frames    -> media.py     (Step 7)
    predict_saliency -> saliency.py  (Step 9)
    detect_regions   -> regions.py   (Step 10)
    render_heatmap   -> heatmap.py   (Step 12)
"""
from __future__ import annotations

import os

from . import KIND_IMAGE

# A stubbed video is reported as this long, so a stub run produces the same
# frame count as the reference 24 s creative and the shape of the payload is
# realistic. Overridable so a test can ask for a different length.
STUB_DURATION_SECONDS = float(os.environ.get("VL_STUB_DURATION_SECONDS", 24.0))
STUB_WIDTH = 1920
STUB_HEIGHT = 1080


def sample_frames(url: str, *, kind: str, sample_fps: float = 2.0,
                  max_frames: int = 120,
                  max_duration_seconds=None) -> dict:
    """Stand-in for media.py. Produces frame descriptors and no pixels.

    A real implementation downloads the creative, runs ffprobe, samples with
    ffmpeg and returns decoded frames. This returns only the timing skeleton,
    which is all the pipeline needs in order to be exercised.
    """
    if kind == KIND_IMAGE:
        return {
            "duration_seconds": None,
            "width": STUB_WIDTH,
            "height": STUB_HEIGHT,
            "has_audio": False,
            "frames": [{"t": 0.0, "index": 0, "pixels": None}],
        }

    duration = STUB_DURATION_SECONDS
    count = min(int(duration * sample_fps), max_frames)
    frames = [{"t": round(i / sample_fps, 3), "index": i, "pixels": None}
              for i in range(count)]
    return {
        "duration_seconds": duration,
        "width": STUB_WIDTH,
        "height": STUB_HEIGHT,
        "has_audio": True,
        "frames": frames,
    }


def predict_saliency(frame: dict) -> dict:
    """Stand-in for saliency.py.

    The real one returns a map that sums to 1 plus per-region mass. This
    returns the same keys with nothing in them, so a caller that forgets to
    handle an empty result fails in a test rather than in production.
    """
    return {
        "concentration": None,
        "peak_count": None,
        "peaks": [],
        "region_mass": {},
        "map_sum": None,
    }


def detect_regions(frame: dict) -> dict:
    """Stand-in for regions.py: OCR boxes, faces, brand mark, CTA."""
    return {
        "text_boxes": [],
        "word_count": None,
        "faces": [],
        "brand": None,
        "cta": None,
        "ocr_available": False,
    }


def render_heatmap(frame: dict, saliency: dict, **kwargs) -> None:
    """Stand-in for heatmap.py. Writes nothing and returns no object key."""
    return None


def measure_frames(media, *, wordmark=None, brand_names=None,
                   detect_regions=None, predict_saliency=None, **_) -> list:
    """Stand-in for measure.py: the record shape with nothing measured.

    EVERY KEY measure.py emits appears here, with the same TYPE. That is not
    padding - the stub record is what most of the suite runs against, so a key
    missing here is a key nothing tests. `text_boxes` being absent is exactly
    how psychology.py shipped a len() call against an integer that only failed
    on a real ad.
    """
    return [{"index": f["index"], "t": f["t"], "shot": 0,
             "seconds_into_shot": None,
             "saliency": {"concentration": None, "peak_count": None, "peaks": [],
                          "map_sum": None},
             "motion_energy": None, "gaze_stability": None,
             "frame_contrast": None, "frame_detail": None,
             "regions": {"word_count": None, "text_area_share": None,
                         "ocr_available": False, "text_boxes": 0, "text": "",
                         "faces": 0, "brand": None, "cta": None, "prices": []},
             "mass": {}}
            for f in (media.get("frames") or [])]


def summarise(records, media) -> dict:
    return {"frames": len(records), "duration_seconds": media.get("duration_seconds"),
            "shot_count": 0, "ocr_operational_fraction": None,
            "frames_with_text_fraction": None, "mean_concentration": None,
            "max_words_on_screen": None, "overloaded_shots": [],
            "brand": {"detected": False, "first_appearance_seconds": None,
                      "exposure_frames": 0, "exposure_seconds": 0.0},
            "cta": {"detected": False, "weak": False},
            "prices_detected": False, "faces_detected": False}


def saliency_info() -> dict:
    return {"saliency_method": "stub", "saliency_model": "stub",
            "saliency_trained": False}


def deps_kwargs() -> dict:
    """The vision callables as VisionDeps kwargs, in one place so app.py, the
    worker and the tests cannot drift apart on which stubs they wire in."""
    return {
        "sample_frames": sample_frames,
        "predict_saliency": predict_saliency,
        "detect_regions": detect_regions,
        "measure_frames": measure_frames,
        "summarise": summarise,
        "saliency_info": saliency_info,
        "render_heatmap": render_heatmap,
    }
