"""
Step 8, part 6 - convert the winning model to ONNX and install it.

    python scripts/bakeoff/export_onnx.py --model unisal
    python scripts/bakeoff/export_onnx.py --model unisal --dry-run

WHY ONNX AND NOT THE PYTORCH CHECKPOINT
    Production is a CPU-only VPS shared with four other features. Serving this
    through ONNX Runtime instead of PyTorch removes ~2.5 GB of torch and CUDA
    libraries the server would otherwise carry to run a 4M-parameter model, and
    is typically faster on CPU into the bargain.

    It also cuts the dependency on the model's own repository. The exported
    graph is self-contained: no vendored research code on the server, no
    `sys.path` surgery, nothing to re-clone.

WHAT IS VERIFIED BEFORE ANYTHING IS INSTALLED
    An export that silently changes the model is the worst outcome here,
    because every score downstream would shift with nothing to point at. So the
    ONNX output is compared against PyTorch on real frames and the export is
    REFUSED if they disagree beyond tolerance.

DYNAMIC INPUT SIZE IS NOT OPTIONAL
    ScaleSerum's creatives are 9:16 vertical; the model's published default is
    landscape. Exporting a fixed input size would bake a 3x distortion into the
    graph. The height and width axes are marked dynamic so the runtime can feed
    aspect-correct frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))

import model_adapters as adapters  # noqa: E402

WORKSPACE = os.environ.get("VL_BAKEOFF_DIR", os.path.join(HERE, "workspace"))
FRAMES_DIR = os.path.join(WORKSPACE, "ad_frames")
MODEL_DIR = os.environ.get("VL_MODEL_DIR", os.path.join(REPO, ".models"))
FRAMEWORK = os.path.join(REPO, "vision_lab", "vision_framework.json")

# Mean absolute difference between PyTorch and ONNX outputs, on maps that sum
# to 1. 1e-5 is comfortably above float32 noise and far below anything that
# would move a score.
TOLERANCE = 1e-5


class UnisalWrapper:
    """Adapts UNISAL's call signature for tracing.

    The model wants `(x, source=..., static=...)` and a 5-D tensor with a time
    axis. ONNX exports a plain forward(x), so the extras are bound here and the
    time axis is added inside - the exported graph then takes an ordinary
    (N, C, H, W) image like every other vision model.
    """

    @staticmethod
    def build(model):
        import torch

        class Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out = self.inner(x[:, None], source="SALICON", static=True)
                return out.squeeze(1)

        return Wrapped(model).eval()


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_frames(count: int = 5) -> list[np.ndarray]:
    manifest = json.load(open(os.path.join(FRAMES_DIR, "manifest.json"),
                              encoding="utf-8"))
    step = max(1, len(manifest["frames"]) // count)
    frames = []
    for entry in manifest["frames"][::step][:count]:
        image = cv2.imread(os.path.join(FRAMES_DIR, entry["file"]))
        if image is not None:
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frames


def export_unisal(dry_run: bool) -> dict:
    import torch

    model = adapters._unisal()
    if model is None:
        sys.exit("UNISAL is not loadable. Run fetch_models.py first.")
    model = model.to("cpu")
    wrapped = UnisalWrapper.build(model)

    dummy = torch.randn(1, 3, 384, 224)
    out_path = os.path.join(WORKSPACE, "unisal.onnx")
    os.makedirs(WORKSPACE, exist_ok=True)

    print("  tracing...", end="", flush=True)
    torch.onnx.export(
        wrapped, dummy, out_path,
        input_names=["image"], output_names=["saliency"],
        # Height and width dynamic - a fixed size would bake in the landscape
        # assumption and distort every vertical creative we analyse.
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                      "saliency": {0: "batch", 1: "height", 2: "width"}},
        opset_version=17, do_constant_folding=True)
    print(f" {os.path.getsize(out_path) / 1048576:.1f} MB")

    print("  verifying against PyTorch on real frames...")
    import onnxruntime
    session = onnxruntime.InferenceSession(out_path,
                                           providers=["CPUExecutionProvider"])
    worst = 0.0
    for index, pixels in enumerate(sample_frames(), start=1):
        tensor = adapters._to_tensor(pixels)
        with torch.no_grad():
            reference = wrapped(tensor).numpy()
        produced = session.run(None, {"image": tensor.numpy()})[0]
        # Compare as DISTRIBUTIONS, not raw log values: that is what the rest
        # of the system consumes, and it is the difference that would actually
        # change a score.
        a = adapters._normalise(np.squeeze(reference), log_space=True)
        b = adapters._normalise(np.squeeze(produced), log_space=True)
        delta = float(np.abs(a - b).mean())
        worst = max(worst, delta)
        print(f"    frame {index}: mean |difference| {delta:.2e}"
              f"  {'OK' if delta < TOLERANCE else 'FAILED'}")

    if worst >= TOLERANCE:
        sys.exit(f"\nEXPORT REFUSED: ONNX differs from PyTorch by {worst:.2e} "
                 f"(tolerance {TOLERANCE:.0e}). Installing this would silently "
                 f"change every score.")

    digest = sha256(out_path)
    record = {
        "name": "UNISAL",
        "file": "unisal.onnx",
        "sha256": digest,
        "licence": "Apache-2.0 (repository; weights committed within it)",
        "licence_note": ("The weights ship inside the Apache-2.0 repository. "
                         "Training data included Hollywood-2 and UCF-Sports - "
                         "see vision_lab/BAKEOFF_LICENCES.md before any "
                         "client-facing use."),
        "source": "https://github.com/rdroste/unisal",
        "citation": "Droste, Chen & Noble, Unified Image and Video Saliency Modelling, ECCV 2020",
        "trained_on": ["SALICON", "DHF1K", "Hollywood-2", "UCF-Sports"],
        "output_space": "log",
        "input_long_side": 384,
        "input_multiple": 32,
        "exported": datetime.now(timezone.utc).isoformat(),
        "size_mb": round(os.path.getsize(out_path) / 1048576, 1),
        "verified_max_delta": worst,
    }

    if dry_run:
        print("\n  --dry-run: not installing")
        return record

    os.makedirs(MODEL_DIR, exist_ok=True)
    installed = os.path.join(MODEL_DIR, "unisal.onnx")
    shutil.copy(out_path, installed)
    print(f"\n  installed -> {installed}")
    return record


def update_framework(record: dict) -> None:
    """Record what is running, so a swapped file cannot go unnoticed.

    saliency.py checks this digest at boot. Without it, replacing the model
    would change every score in the product with nothing in the report to say
    that anything had changed.
    """
    with open(FRAMEWORK, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["saliency_model"] = {
        **config.get("saliency_model", {}),
        "name": record["name"],
        "file": record["file"],
        "sha256": record["sha256"],
        "licence": record["licence"],
        "licence_note": record["licence_note"],
        "source": record["source"],
        "citation": record["citation"],
        "trained_on": record["trained_on"],
        # The head is a log-softmax. saliency.py MUST exponentiate before
        # normalising - clipping the negatives instead produces an all-zero map
        # that silently degrades to uniform, which is exactly the bug that made
        # this model look worse than a centre blob during the bake-off.
        "output_space": record["output_space"],
        "input_long_side": record["input_long_side"],
        "input_multiple": record["input_multiple"],
        "exported": record["exported"],
    }
    with open(FRAMEWORK, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"  recorded in {os.path.relpath(FRAMEWORK, REPO)}")


EXPORTERS = {"unisal": export_unisal}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unisal", choices=list(EXPORTERS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"\nExporting {args.model}\n")
    record = EXPORTERS[args.model](args.dry_run)
    if not args.dry_run:
        update_framework(record)

    print(f"\n  sha256 {record['sha256'][:16]}...")
    print(f"  licence: {record['licence']}")
    print("\nRestart the worker; /health will report saliency_trained=true.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
