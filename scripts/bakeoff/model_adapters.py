"""
One calling convention for every candidate: RGB frame in, saliency map out.

WHY ADAPTERS
    Each research repo has its own loading code, input size, normalisation and
    output shape. Wrapping them here means evaluate.py never learns any of that,
    and adding a candidate is one function rather than a branch in the scorer.

EVERY ADAPTER RETURNS A MAP THAT SUMS TO 1
    Same invariant as vision_lab/saliency.py. The metrics compare
    distributions, so a model that returned raw logits would score
    meaninglessly against one that returned probabilities - and the difference
    would look like a quality gap rather than a units mismatch.

MISSING IS NOT BROKEN
    A candidate whose weights or dependencies are absent is simply not offered.
    available() returns what can actually run, and evaluate.py scores that.
    Nothing raises, because "torch is not installed yet" should not stop the two
    controls from producing numbers.
"""
from __future__ import annotations

import functools
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

WORKSPACE = os.environ.get("VL_BAKEOFF_DIR", os.path.join(HERE, "workspace"))
CLONES = os.path.join(WORKSPACE, "models")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalise(saliency: np.ndarray, log_space: bool | None = None) -> np.ndarray:
    """To a distribution summing to 1, handling log-space outputs.

    SEVERAL OF THESE MODELS RETURN LOG-PROBABILITIES, NOT PROBABILITIES.
    UNISAL's final layer is a log-softmax: every value is negative and exp() of
    it sums to 1. Clipping those negatives to zero - the obvious thing to do to
    a saliency map - produces an all-zero array that falls back to uniform, and
    the model then scores BELOW a plain centre blob.

    That is a silent failure with a plausible-looking output, which is the worst
    kind. It looked like evidence that a published model is bad on ad creatives.
    It was evidence that the adapter was broken.

    `log_space=None` detects it: an all-negative map is log-space, because a
    probability never is.
    """
    saliency = saliency.astype(np.float32)
    if log_space is None:
        log_space = bool(saliency.size) and float(saliency.max()) <= 0.0
    if log_space:
        saliency = np.exp(saliency - float(saliency.max()))

    saliency = np.clip(saliency, 0, None)
    total = float(saliency.sum())
    return saliency / total if total > 0 else np.full_like(
        saliency, 1.0 / saliency.size)


def _fit(pixels: np.ndarray, long_side: int = 384,
         multiple: int = 32) -> tuple[int, int]:
    """Input size that PRESERVES the frame's aspect ratio.

    This matters more than it looks. Every candidate here was published with a
    landscape default (384x224, 384x288) because the benchmarks are landscape
    photographs and video. ScaleSerum's creatives are 9:16 vertical.

    Resizing 960x1706 into 384x224 distorts the frame by a factor of three -
    a headline across the bottom becomes a thin smear, a face becomes a wide
    oval, and the model's spatial priors no longer line up with anything in the
    image. Measured: UNISAL scored BELOW a plain centre blob that way, which is
    not a finding about UNISAL, it is a finding about the resize.

    These networks are fully convolutional, so they accept any size that is a
    multiple of the encoder stride. Fitting the long side and rounding both
    dimensions to `multiple` keeps the geometry honest.
    """
    h, w = pixels.shape[:2]
    scale = long_side / max(h, w)
    width = max(multiple, int(round(w * scale / multiple)) * multiple)
    height = max(multiple, int(round(h * scale / multiple)) * multiple)
    return width, height


def _to_tensor(pixels: np.ndarray, width: int | None = None,
               height: int | None = None, long_side: int = 384):
    import torch
    if width is None or height is None:
        width, height = _fit(pixels, long_side)
    resized = cv2.resize(pixels, (width, height), interpolation=cv2.INTER_AREA)
    array = resized.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))[None, ...]
    return torch.from_numpy(array)


# --------------------------------------------------------------------------- controls
def centre(pixels: np.ndarray) -> np.ndarray:
    """The control. Whatever a model scores, this is what it is worth for free."""
    import metrics as mx
    h, w = pixels.shape[:2]
    return mx.centre_prior(h, w)


def spectral(pixels: np.ndarray) -> np.ndarray:
    """What Vision Lab runs in production today - the thing to beat."""
    from vision_lab import saliency as sal
    result = sal.predict({"pixels": pixels})
    return result.get("map")


# --------------------------------------------------------------------------- candidates
@functools.lru_cache(maxsize=1)
def _unisal():
    """UNISAL in image mode.

    Its video path expects 16-32 consecutive frames at native frame rate; our
    samples are seconds apart, so feeding them as a clip would give the model
    nonsense motion. Image mode is the honest comparison.
    """
    path = os.path.join(CLONES, "unisal")
    weights = os.path.join(path, "training_runs", "pretrained_unisal",
                           "weights_best.pth")
    if not os.path.isfile(weights):
        return None
    try:
        import torch
        sys.path.insert(0, path)

        # UNISAL's own code hardcodes cuda:0 and its MobileNetV2 backbone calls
        # torch.load WITHOUT map_location, so constructing it on a CPU-only
        # machine dies deserialising CUDA tensors. Patching the default for the
        # duration of construction is less invasive than editing a vendored
        # repository we do not own and will re-clone.
        original_load = torch.load

        def cpu_first(*args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = cpu_first
        try:
            from unisal.model import UNISAL  # type: ignore
            model = UNISAL(bypass_rnn=True)
            model.load_state_dict(original_load(
                weights, map_location="cpu", weights_only=False), strict=True)
        finally:
            torch.load = original_load

        # device_count(), not is_available(): with CUDA_VISIBLE_DEVICES set to
        # hide the GPU, is_available() can still say True while there are no
        # devices to place a tensor on.
        device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
        model.to(device).eval()
        model._vl_device = device      # the adapter must send input to the same place
        return model
    except Exception as err:  # noqa: BLE001
        print(f"    (unisal unavailable: {type(err).__name__}: {err})")
        return None


def unisal(pixels: np.ndarray) -> np.ndarray | None:
    model = _unisal()
    if model is None:
        return None
    import torch
    with torch.no_grad():
        # Aspect-preserving - see _fit(). A fixed landscape size would
        # distort these vertical creatives by 3x.
        tensor = _to_tensor(pixels)[None, ...]     # (1,1,3,H,W)
        tensor = tensor.to(getattr(model, "_vl_device", "cpu"))
        out = model(tensor, source="SALICON", static=True)
    # UNISAL's head is a log-softmax - stated explicitly rather than left to
    # detection, so a future change to the model cannot silently flip meaning.
    return _normalise(np.squeeze(out.cpu().numpy()), log_space=True)


@functools.lru_cache(maxsize=1)
def _msinet():
    """MSI-Net. The licence-cleanest candidate: SALICON-only training, with MIT
    declared against the WEIGHTS themselves rather than only the code.

    Published as a TensorFlow SavedModel, so it needs TF here. If it wins, the
    export step converts it to ONNX and TensorFlow never reaches production -
    the worker would carry onnxruntime alone.
    """
    onnx = None
    saved_model = None
    for root, dirs, files in os.walk(os.path.join(CLONES, "msinet")):
        for name in files:
            if name.endswith(".onnx"):
                onnx = os.path.join(root, name)
            if name == "saved_model.pb":
                saved_model = root

    if onnx:
        try:
            import onnxruntime
            return ("onnx", onnxruntime.InferenceSession(
                onnx, providers=["CPUExecutionProvider"]))
        except Exception as err:  # noqa: BLE001
            print(f"    (msinet onnx unavailable: {type(err).__name__})")

    if saved_model:
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf
            return ("tf", tf.saved_model.load(saved_model))
        except Exception as err:  # noqa: BLE001
            print(f"    (msinet unavailable: {type(err).__name__}: "
                  f"{str(err)[:120]})")
    return None


def msinet(pixels: np.ndarray) -> np.ndarray | None:
    loaded = _msinet()
    if loaded is None:
        return None
    kind, session = loaded

    if kind == "tf":
        import tensorflow as tf
        # MSI-Net's published input is 240x320 landscape. Letterboxing rather
        # than squashing, for the same reason as everywhere else here: these
        # creatives are 9:16 and a straight resize distorts them threefold.
        letterboxed, pad = _letterbox(pixels, 320, 240)
        tensor = tf.convert_to_tensor(letterboxed[None, ...].astype(np.float32))
        out = session(tensor)
        out = list(out.values())[0] if isinstance(out, dict) else out
        return _normalise(_unletterbox(np.squeeze(out.numpy()), pad, (320, 240)))

    spec = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in spec.shape]
    height, width = (shape[2], shape[3]) if shape[1] == 3 else (shape[1], shape[2])
    # A fixed ONNX input size cannot be changed, so letterbox into it rather
    # than squashing: pad to the target aspect, then resize. The padding is
    # cropped back off the output so the returned map matches the real frame.
    letterboxed, pad = _letterbox(pixels, width, height)
    array = letterboxed.astype(np.float32) / 255.0
    if shape[1] == 3:
        array = np.transpose(array, (2, 0, 1))
    out = np.squeeze(session.run(None, {spec.name: array[None, ...]})[0])
    return _normalise(_unletterbox(out, pad, (width, height)))


def _letterbox(pixels: np.ndarray, width: int, height: int):
    h, w = pixels.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(pixels, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, pixels.shape[2]), dtype=pixels.dtype)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, (x0, y0, nw, nh)


def _unletterbox(saliency: np.ndarray, pad: tuple,
                 canvas: tuple[int, int]) -> np.ndarray:
    """Crop the padding back off the model's output.

    The map usually comes back smaller than the input it was letterboxed into,
    so the crop box is scaled by that ratio rather than used directly. Leaving
    the padding in would hand the metrics a band of dead zeros down each side,
    which quietly depresses every score the model gets.
    """
    x0, y0, nw, nh = pad
    canvas_w, canvas_h = canvas
    h, w = saliency.shape[:2]
    sx, sy = w / canvas_w, h / canvas_h
    left = int(round(x0 * sx))
    top = int(round(y0 * sy))
    right = max(left + 1, int(round((x0 + nw) * sx)))
    bottom = max(top + 1, int(round((y0 + nh) * sy)))
    return saliency[top:bottom, left:right]


@functools.lru_cache(maxsize=1)
def _transalnet():
    path = os.path.join(CLONES, "transalnet")
    weights = None
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".pth", ".pt")):
                weights = os.path.join(root, name)
    if not weights:
        return None                 # Google Drive download, needs a human
    try:
        import torch
        sys.path.insert(0, path)
        model = torch.load(weights, map_location="cpu", weights_only=False)
        if hasattr(model, "eval"):
            model.eval()
            return model
    except Exception as err:  # noqa: BLE001
        print(f"    (transalnet unavailable: {type(err).__name__}: {err})")
    return None


def transalnet(pixels: np.ndarray) -> np.ndarray | None:
    model = _transalnet()
    if model is None:
        return None
    import torch
    with torch.no_grad():
        out = model(_to_tensor(pixels))
    return _normalise(np.squeeze(out.cpu().numpy()))


# --------------------------------------------------------------------------- registry
ALL = {
    "centre (control)": centre,
    "spectral (production)": spectral,
    "unisal": unisal,
    "msinet": msinet,
    "transalnet": transalnet,
}


def available() -> dict:
    """Which candidates can actually run right now.

    Probed with a real 64x64 frame rather than by checking for files: a
    checkpoint that exists but will not load is not available, and finding that
    out here beats finding it out a hundred frames into a run.
    """
    probe = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    usable = {}
    for name, fn in ALL.items():
        try:
            result = fn(probe)
            if result is not None and np.isfinite(result).all():
                usable[name] = fn
        except Exception as err:  # noqa: BLE001
            print(f"  {name}: unavailable ({type(err).__name__})")
    return usable
