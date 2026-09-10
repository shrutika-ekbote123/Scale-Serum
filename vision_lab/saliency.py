"""
Step 9 - the attention model. Where a human eye would land, per frame.

TWO BACKENDS, AND THE DIFFERENCE MATTERS
    "onnx"      a trained saliency network from VL_MODEL_DIR. The real thing.
                Chosen by the Step 8 bake-off, which has NOT been run yet.
    "spectral"  Spectral Residual (Hou & Zhang, 2007) in numpy. A classical
                image-statistics method with no learning in it at all.

    The backend in use is reported in every payload as `saliency_method`, and
    `saliency_trained` says whether it learned anything from human eye-tracking.
    Do not let a client see scores produced by "spectral" and described as
    AI-predicted attention - it is a baseline that keeps the pipeline honest and
    testable until a real checkpoint lands, not a substitute for one.

WHY A BASELINE AT ALL
    Without it, nothing downstream could be built or tested: the timeline, the
    six scores, the heatmaps and the triggers all need a real map over real
    pixels. Spectral Residual gives that, from actual image content, today. When
    the bake-off finishes, `predict()` changes backend and nothing else in the
    codebase moves.

WHAT SPECTRAL RESIDUAL DOES NOT DO
    It has no notion of faces, text or brand marks - all three are strong human
    attractors and it will under-weight every one of them. It also has no centre
    bias, which real fixation data has a great deal of. Expect it to disagree
    with a trained model most on exactly the frames an ad cares about.

THE ONE INVARIANT
    Every map returned by predict() sums to 1.0. It is a spatial probability
    distribution, which is what makes `region mass` meaningful and comparable
    between frames. A normalisation bug here would silently corrupt every score
    downstream and would look completely fine in a rendered heatmap - so it is
    asserted in a test.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

import cv2
import numpy as np

from . import framework as fw

logger = logging.getLogger("vision_lab.saliency")

MODEL_DIR = os.environ.get("VL_MODEL_DIR", "")

METHOD_ONNX = "onnx"
METHOD_SPECTRAL = "spectral_residual"

# The fraction of pixels that counts as "where the eye is". 5% of a 640x360
# frame is ~11,500 px - about the size of a face or a headline block.
TOP_FRACTION = float(os.environ.get("VL_CONCENTRATION_TOP_FRACTION", 0.05))

_session = None
_session_meta: dict = {}


# --------------------------------------------------------------------------- model
def _load_onnx() -> Optional[tuple]:
    """Load the ONNX session once per process. None when no model is installed."""
    global _session, _session_meta
    if _session is not None:
        return _session, _session_meta
    if not MODEL_DIR or not os.path.isdir(MODEL_DIR):
        return None

    # Load the model the CONFIG names, not whatever sorts first.
    #
    # VL_MODEL_DIR holds more than one .onnx - the face detector lives there
    # too - and "first alphabetically" picked face_detection_yunet.onnx and ran
    # it as the saliency model, while describe() still reported UNISAL because
    # the name comes from config. Only the digest check made it visible.
    configured_file = fw.saliency_model().get("file")
    if configured_file:
        candidate = os.path.join(MODEL_DIR, configured_file)
        if not os.path.isfile(candidate):
            logger.warning("configured saliency model %s is not in %s",
                           configured_file, MODEL_DIR)
            return None
        models = [configured_file]
    else:
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".onnx")]
        if len(models) > 1:
            logger.warning("several .onnx files in %s and no `file` configured - "
                           "set saliency_model.file in vision_framework.json",
                           MODEL_DIR)
    if not models:
        return None

    try:
        import onnxruntime
    except ImportError:
        logger.warning("a model is present in %s but onnxruntime is not installed",
                       MODEL_DIR)
        return None

    path = os.path.join(MODEL_DIR, sorted(models)[0])
    configured = fw.saliency_model()

    # A silently swapped model would change every score with nothing to show for
    # it. The expected digest lives in vision_framework.json next to the name.
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    expected = configured.get("sha256")
    if expected and expected != digest:
        # REFUSE, do not warn and continue. A file that is not the one this
        # config describes would produce scores attributed to a model that never
        # ran them. Falling back to the classical baseline is a stated
        # degradation; running the wrong model silently is not.
        logger.error("saliency model digest mismatch: config expects %s but %s "
                     "is %s - refusing to load it. Re-run export_onnx.py, or "
                     "correct saliency_model.sha256.",
                     expected[:12], os.path.basename(path), digest[:12])
        return None

    _session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    _session_meta = {
        "name": configured.get("name") or os.path.basename(path),
        "sha256": digest,
        "input": _session.get_inputs()[0].name,
        # Whether the head emits probabilities or log-probabilities, and the
        # aspect-preserving input geometry. Both are properties of the exported
        # graph, so they live in config beside the model rather than as
        # constants here - a different model would need different answers.
        "output_space": configured.get("output_space"),
        "long_side": configured.get("input_long_side")
        or configured.get("input_width") or 384,
        "multiple": configured.get("input_multiple") or 32,
        "licence": configured.get("licence"),
    }
    logger.info("saliency model loaded: %s", _session_meta["name"])
    return _session, _session_meta


def method() -> str:
    return METHOD_ONNX if _load_onnx() else METHOD_SPECTRAL


def is_trained() -> bool:
    """Did this backend learn anything from human eye-tracking?"""
    return method() == METHOD_ONNX


def describe() -> dict:
    """What goes in the report so nobody has to guess what produced a number."""
    loaded = _load_onnx()
    if loaded:
        return {"saliency_method": METHOD_ONNX,
                "saliency_model": loaded[1]["name"],
                "saliency_trained": True,
                "saliency_licence": loaded[1].get("licence")}
    return {
        "saliency_method": METHOD_SPECTRAL,
        "saliency_model": "spectral_residual_baseline",
        "saliency_trained": False,
        "saliency_note": ("Classical image-statistics baseline, not a model trained "
                          "on eye-tracking. Scores are for development only until "
                          "a checkpoint is installed in VL_MODEL_DIR."),
    }


# --------------------------------------------------------------------------- maps
def _spectral_residual(gray: np.ndarray) -> np.ndarray:
    """Hou & Zhang (2007). The log-amplitude spectrum of a natural image is
    nearly scale-invariant; what departs from that average is what stands out."""
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    spectrum = np.fft.fft2(small)
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    residual = log_amplitude - cv2.blur(log_amplitude, (3, 3))
    reconstructed = np.fft.ifft2(np.exp(residual + 1j * phase))
    saliency = np.abs(reconstructed) ** 2
    return cv2.GaussianBlur(saliency, (9, 9), 2.5)


def _input_size(frame: np.ndarray, meta: dict) -> tuple[int, int]:
    """Model input size that PRESERVES the frame's aspect ratio.

    Every published saliency checkpoint ships a LANDSCAPE default (384x224,
    384x288) because the benchmarks are landscape photographs. ScaleSerum's
    creatives are 9:16 vertical.

    Squashing 960x1706 into 384x224 distorts the frame threefold - a headline
    becomes a smear, a face becomes a wide oval, and the model's spatial priors
    line up with nothing in the image. Measured during the Step 8 bake-off, that
    resize alone dropped UNISAL below a plain centre blob.

    The exported graph has dynamic height and width axes precisely so this can
    fit the long side and round to the encoder's stride instead.
    """
    long_side = int(meta.get("long_side") or 384)
    multiple = int(meta.get("multiple") or 32)
    height, width = frame.shape[:2]
    scale = long_side / max(height, width)
    out_w = max(multiple, int(round(width * scale / multiple)) * multiple)
    out_h = max(multiple, int(round(height * scale / multiple)) * multiple)
    return out_w, out_h


def _onnx_map(frame: np.ndarray, session, meta: dict) -> np.ndarray:
    out_w, out_h = _input_size(frame, meta)
    resized = cv2.resize(frame, (out_w, out_h),
                         interpolation=cv2.INTER_AREA)
    # ImageNet normalisation - the convention every candidate checkpoint uses.
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
             np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    output = np.squeeze(session.run(None, {meta["input"]: tensor})[0])
    output = output.astype(np.float32)

    # SOME MODELS RETURN LOG-PROBABILITIES, NOT PROBABILITIES.
    #
    # UNISAL's head is a log-softmax: every value is negative and exp() of it
    # sums to 1. Clipping those negatives to zero - the obvious thing to do to a
    # saliency map - yields an all-zero array that falls back to uniform. Every
    # score computed from it is then meaningless while still looking plausible,
    # which is exactly the bug that made this model appear WORSE than a plain
    # centre blob during the bake-off. `output_space` in vision_framework.json
    # states which it is; the all-negative check is the belt to that braces.
    if meta.get("output_space") == "log" or float(output.max()) <= 0.0:
        output = np.exp(output - float(output.max()))
    return output


def predict(frame: dict) -> dict:
    """The VisionDeps callable. One frame in, one measurement record out.

    Returns the normalised map alongside the statistics computed from it, so a
    caller that only needs numbers never has to hold pixel data.
    """
    pixels = frame.get("pixels")
    if pixels is None:
        return {"concentration": None, "peak_count": None, "peaks": [],
                "region_mass": {}, "map_sum": None, "map": None}

    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    loaded = _load_onnx()
    raw = _onnx_map(pixels, *loaded) if loaded else _spectral_residual(gray)

    height, width = gray.shape[:2]
    saliency = cv2.resize(raw, (width, height), interpolation=cv2.INTER_CUBIC)
    saliency = np.clip(saliency, 0, None)

    total = float(saliency.sum())
    if total <= 0:
        # A perfectly flat frame - black, white, or a solid colour card. Uniform
        # is the honest answer; zeros would make every region mass zero and read
        # as "nothing was salient" rather than "nothing was distinguishable".
        saliency = np.full_like(saliency, 1.0 / saliency.size)
    else:
        saliency = saliency / total

    return {
        "map": saliency,
        "map_sum": float(saliency.sum()),
        "concentration": concentration(saliency),
        **peaks(saliency),
    }


def concentration(saliency: np.ndarray) -> float:
    """Share of the map's mass sitting in its brightest TOP_FRACTION of pixels.

    High means the eye is pulled to one place. Low means the frame is visually
    flat or contested - which is what a wall of bullet points looks like.
    """
    flat = saliency.ravel()
    keep = max(1, int(flat.size * TOP_FRACTION))
    top = np.partition(flat, -keep)[-keep:]
    return round(float(top.sum()), 6)


def peaks(saliency: np.ndarray, limit: int = 3) -> dict:
    """The strongest attention centres, as fractional boxes.

    Fractional rather than pixel coordinates so the frontend can overlay them on
    a frame at any display size without knowing our working resolution.
    """
    height, width = saliency.shape[:2]
    scaled = (saliency / saliency.max() * 255).astype(np.uint8) if saliency.max() else \
        np.zeros_like(saliency, dtype=np.uint8)

    # Otsu finds the split between "attended" and "background" per frame rather
    # than against a fixed threshold that would suit one creative and not another.
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < (width * height) * 0.001:      # specks are not attention
            continue
        mass = float(saliency[y:y + h, x:x + w].sum())
        found.append({
            "box": [round(x / width, 4), round(y / height, 4),
                    round((x + w) / width, 4), round((y + h) / height, 4)],
            "share": round(mass, 4),
        })

    found.sort(key=lambda item: item["share"], reverse=True)
    for rank, item in enumerate(found[:limit], start=1):
        item["rank"] = rank
    return {"peak_count": len(found), "peaks": found[:limit]}


def region_mass(saliency: np.ndarray, box: list[float]) -> float:
    """How much predicted gaze falls inside a detected region.

    This is the join between "where the eye goes" and "what is on screen", and
    every marketing conclusion in the report is built on it.
    """
    if saliency is None or not box:
        return 0.0
    height, width = saliency.shape[:2]
    x0 = max(0, int(box[0] * width))
    y0 = max(0, int(box[1] * height))
    x1 = min(width, int(box[2] * width))
    y1 = min(height, int(box[3] * height))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return round(float(saliency[y0:y1, x0:x1].sum()), 6)
