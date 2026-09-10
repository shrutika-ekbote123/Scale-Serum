"""
Provenance: what a model's weights actually inherit, and the NOTICE file.

WHY THIS IS SEPARATE FROM THE LICENCE CHECK
    fetch_models.py reads the LICENSE file a repository ships. That grants
    rights over its CODE. It says nothing about the 15 MB .pth file, and the
    weights were produced from datasets with terms of their own.

    Reading only the repo licence made UNISAL - Apache-2.0, weights committed
    right there in the tree - look like the cleanest candidate. Its weights were
    trained on Hollywood-2, which is clips from commercial feature films. That
    is a question no LICENSE file in the model repo can answer, and it is
    invisible unless the training data is checked as its own thing.

WHAT A VERDICT HERE IS AND IS NOT
    It is a triage signal: green means nothing obvious to chase, amber means a
    specific thing to read, red means a specific thing that looks disqualifying.
    It is not legal advice, and the tool never marks anything as cleared - only
    a human can do that, which is why `weights_licence_declared` has to be
    filled in by hand after reading the actual model card.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from candidates import DATASETS  # noqa: E402

GREEN = "clean"
AMBER = "check"
RED = "likely disqualifying"


def dataset_concerns(candidate: dict) -> list[dict]:
    """The training sets with something specific to chase."""
    concerns = []
    for key in candidate.get("trained_on") or []:
        dataset = DATASETS.get(key)
        if dataset and dataset.get("concern"):
            concerns.append({"id": key, **dataset})
    return concerns


def weights_verdict(candidate: dict) -> tuple[str, list[str]]:
    """How much of a question mark hangs over these particular weights."""
    reasons: list[str] = []
    level = GREEN

    declared = candidate.get("weights_licence_declared")
    if declared:
        reasons.append(f"weights licence declared: {declared}")
    else:
        level = AMBER
        reasons.append("no licence declared against the WEIGHTS themselves - the "
                       "repo licence covers code only")

    for dataset in dataset_concerns(candidate):
        # Third-party footage the dataset authors did not own is a different
        # order of problem from a research-terms clause on a public image set.
        if "feature films" in dataset["what"] or "broadcast" in dataset["what"]:
            level = RED
            reasons.append(f"trained on {dataset['name']} - {dataset['what']}")
        else:
            level = RED if level == RED else AMBER
            reasons.append(f"{dataset['name']}: {dataset['concern'].split('.')[0]}")

    return level, reasons


def summarise(candidate: dict) -> dict:
    level, reasons = weights_verdict(candidate)
    return {
        "id": candidate["id"],
        "name": candidate["name"],
        "weights_source": candidate.get("weights_hint", ""),
        "weights_licence_declared": candidate.get("weights_licence_declared"),
        "weights_licence_url": candidate.get("weights_licence_url"),
        "trained_on": [DATASETS.get(k, {}).get("name", k)
                       for k in candidate.get("trained_on") or []],
        "level": level,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- NOTICE
# Third-party components Vision Lab introduces. Not the whole dependency tree -
# the things this feature added, which are the ones nobody else has reviewed.
#
# ffmpeg is the entry that matters most, and not for the reason people expect.
RUNTIME_COMPONENTS = [
    {
        "name": "FFmpeg",
        "used_for": "Decoding the creative, sampling frames, detecting shot changes, extracting audio",
        "licence": "LGPL-2.1+ or GPL-2+/GPL-3+ DEPENDING ON THE BUILD",
        "how_we_use_it": "invoked as a separate process via subprocess",
        "note": (
            "The Windows build used in development (gyan.dev full_build) reports "
            "--enable-gpl --enable-version3, so it is a GPL-3 build. Debian and "
            "Ubuntu's packaged ffmpeg is also typically GPL.\n\n"
            "This matters far less than it looks, because of HOW we use it: "
            "vision_lab/media.py shells out to the ffmpeg BINARY and reads its "
            "output. We do not link against libavcodec, and the GPL's copyleft "
            "reaches linked code, not separate programs exchanging data.\n\n"
            "Two rules keep it that way:\n"
            "  1. Install ffmpeg as a system package on the server "
            "(apt install ffmpeg). Never vendor the binary into this repo or "
            "into an image we distribute.\n"
            "  2. Never switch to a Python binding that LINKS ffmpeg libraries "
            "(PyAV, ffmpeg-python's native bindings) without a licence review - "
            "that changes the analysis completely.\n\n"
            "If ffmpeg ever has to ship WITH the product, use an LGPL build "
            "(no --enable-gpl) and comply with LGPL relinking terms."
        ),
    },
    {
        "name": "OpenCV (opencv-python-headless)",
        "used_for": "Frame resizing, template matching, colour maps, image encoding",
        "licence": "Apache-2.0 (OpenCV 4.5+)",
        "how_we_use_it": "imported as a Python library",
        "note": "Permissive. Preserve the notice if we ever distribute a bundle.",
    },
    {
        "name": "Tesseract OCR",
        "used_for": "Reading on-screen text, word counts, CTA and price detection",
        "licence": "Apache-2.0",
        "how_we_use_it": "invoked as a separate binary through pytesseract",
        "note": "Permissive. The binary is installed on the box, not vendored.",
    },
    {
        "name": "ONNX Runtime",
        "used_for": "Running the saliency model on CPU",
        "licence": "MIT",
        "how_we_use_it": "imported as a Python library",
        "note": "Optional - absent, saliency falls back to the classical baseline.",
    },
    {
        "name": "Pillow, NumPy, boto3, httpx",
        "used_for": "Image encoding, arrays, S3, HTTP",
        "licence": "MIT-BSD family / Apache-2.0",
        "how_we_use_it": "imported as Python libraries",
        "note": "Permissive; already in the wider service's dependency set.",
    },
]


def write_notice(path: str, chosen_model: dict | None = None) -> None:
    """Generate NOTICE.

    Cheap to keep current, awkward to reconstruct later. SaaS delivery probably
    does not oblige us to publish it today - but "probably" is doing work in
    that sentence, and the file costs nothing.
    """
    lines = [
        "# NOTICE - third-party components",
        "",
        f"Generated by `scripts/bakeoff/provenance.py` on "
        f"{datetime.now(timezone.utc):%Y-%m-%d}. Regenerate when a component changes.",
        "",
        "Covers the components **Vision Lab introduced**. The wider service's "
        "existing dependencies (FastAPI, motor, google-genai and so on) are not "
        "repeated here.",
        "",
        "## Why this file exists even though we do not distribute a binary",
        "",
        "Vision Lab runs server-side, and permissive licences attach most of their",
        "obligations to *distribution*. Running software to provide a service is",
        "usually not distribution - so today this file is mostly a record.",
        "",
        "It stops being a record the moment any of these happen:",
        "",
        "- Vision Lab is shipped on-premise to a customer",
        "- a model or binary is bundled into an image handed to someone else",
        "- inference moves into the browser",
        "",
        "Reconstructing provenance after the fact is the expensive version of this.",
        "",
        "---",
        "",
        "## Runtime components",
        "",
    ]

    for component in RUNTIME_COMPONENTS:
        lines += [
            f"### {component['name']}",
            "",
            f"- **Used for:** {component['used_for']}",
            f"- **Licence:** {component['licence']}",
            f"- **How we use it:** {component['how_we_use_it']}",
            "",
            component["note"],
            "",
        ]

    lines += ["---", "", "## Saliency model", ""]
    if chosen_model:
        lines += [
            f"- **Model:** {chosen_model.get('name')}",
            f"- **Code licence:** {chosen_model.get('licence')}",
            f"- **Weights licence:** "
            f"{chosen_model.get('weights_licence_declared') or 'NOT DECLARED - resolve before shipping'}",
            f"- **Trained on:** {', '.join(chosen_model.get('trained_on') or []) or 'unknown'}",
            f"- **SHA-256:** recorded in `vision_lab/vision_framework.json` and "
            f"verified by the worker at boot",
            "",
        ]
    else:
        lines += [
            "**None installed.** Vision Lab currently runs the Spectral Residual",
            "baseline implemented directly in `vision_lab/saliency.py` (Hou & Zhang,",
            "2007) - an algorithm, not a downloaded artefact, so it carries no",
            "third-party licence.",
            "",
            "When the Step 8 bake-off picks a model, record it here along with the",
            "licence covering its WEIGHTS - which is a separate question from the",
            "licence on its repository. See `vision_lab/BAKEOFF_LICENCES.md`.",
            "",
        ]

    lines += [
        "---",
        "",
        "## Evaluation datasets - never shipped",
        "",
        "MIT1003 and CAT2000 are downloaded to `scripts/bakeoff/workspace/` to",
        "benchmark candidate models. They are **evaluation inputs, not product",
        "components**: gitignored, never deployed, never redistributed, and absent",
        "from every artefact that reaches a customer.",
        "",
    ]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
