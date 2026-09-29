"""
Install the Sales Call Analyzer's voice models into SCA_MODEL_DIR.

    python scripts/fetch_sca_models.py                  # into $SCA_MODEL_DIR
    python scripts/fetch_sca_models.py /srv/models/sca  # or an explicit folder
    python scripts/fetch_sca_models.py --check          # verify, download nothing

Run once per server, and again only when a model changes. The folder must be
OUTSIDE the app directory: deploy/deploy.sh runs `git reset --hard`, and like
Vision Lab's VL_MODEL_DIR these files are deploy artefacts, not source.

Every file is pinned by SHA-256. A file that does not match is deleted and
reported - a silently different model would change every speaker decision, and
voiceprints enrolled under one model are not comparable with another.

Without these files the analyzer still runs: /health reports
sales_call_analyzer.speaker_refinement.voice_reason and speaker refinement and
voiceprints are skipped with that reason.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import httpx

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
MODELS = {
    # Speaker embeddings. Chosen by scripts/sca_eval/embedding_bakeoff.py:
    # 3D-Speaker ERes2Net, Apache-2.0.
    "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx": (
        f"{BASE}/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        "c59158379255ad66e161679cca6af8d52d51e389e3224ab7d7a7baae295c2db5"),
    # Voice activity detection. Silero VAD, MIT.
    "silero_vad.onnx": (
        f"{BASE}/asr-models/silero_vad.onnx",
        "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", nargs="?", default=os.environ.get("SCA_MODEL_DIR"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.folder:
        print("Set SCA_MODEL_DIR or pass the folder.")
        return 2
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)

    ok = True
    for name, (url, digest) in MODELS.items():
        path = folder / name
        if path.exists() and sha256(path) == digest:
            print(f"ok       {name}")
            continue
        if args.check:
            print(f"MISSING  {name}" if not path.exists() else f"MISMATCH {name}")
            ok = False
            continue
        print(f"fetching {name}")
        tmp = path.with_suffix(path.suffix + ".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        if sha256(tmp) != digest:
            tmp.unlink()
            print(f"MISMATCH {name}: the download did not match its pinned checksum")
            ok = False
            continue
        tmp.replace(path)
        print(f"ok       {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
