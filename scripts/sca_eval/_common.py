"""
Shared plumbing for the Sales Call Analyzer accuracy harness.

Everything this harness writes goes under scripts/sca_eval/workspace/, which is
gitignored: it holds real customer recordings, their transcripts and the CRM
names attached to them. None of it may ever be committed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORKSPACE = HERE / "workspace"

REAL_DIR = WORKSPACE / "real"          # real calls from scrumdb
FLEURS_DIR = WORKSPACE / "fleurs"      # labelled read speech, per language
SYNTH_DIR = WORKSPACE / "synthetic"    # generated two-speaker calls with exact truth
RUNS_DIR = WORKSPACE / "runs"          # cached raw Deepgram responses, per config
LABELS_DIR = WORKSPACE / "labels"      # human-corrected references for real calls
REPORTS_DIR = WORKSPACE / "reports"

sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from transcription.audio_slice import ffmpeg_path  # noqa: E402


def ffprobe_path() -> Optional[str]:
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return None
    candidate = Path(ffmpeg).with_name("ffprobe" + (".exe" if os.name == "nt" else ""))
    return str(candidate) if candidate.is_file() else None


def run_ffmpeg(args: list[str]) -> bytes:
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise SystemExit("ffmpeg is required: install it or set SCA_FFMPEG_DIR")
    done = subprocess.run([ffmpeg, "-v", "error", "-y", *args], capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode(errors="replace")[:400])
    return done.stdout


def decode_pcm(path: Path, *, rate: int = 16000, channels: int = 1):
    """The file as float32 samples, shape (n, channels)."""
    import numpy as np
    raw = run_ffmpeg(["-i", str(path), "-f", "s16le", "-ac", str(channels),
                      "-ar", str(rate), "-"])
    return np.frombuffer(raw, np.int16).astype(np.float32).reshape(-1, channels) / 32768.0


def to_telephony(src: Path, dst: Path) -> Path:
    """Degrade to what a phone line delivers: 8 kHz, G.711 mu-law, mono.

    Read speech and TTS are far cleaner than a real call. Without this every
    public-data number is optimistic; mycall3.mp3 has no energy above 4 kHz.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", "8000", "-acodec", "pcm_mulaw", str(dst)])
    return dst


def pg_connect():
    """scrumdb, read-only. This harness never writes to it."""
    import psycopg
    conn = psycopg.connect(
        host=os.environ["DB_HOST"], port=os.environ.get("DB_PORT", "5432"),
        dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"], connect_timeout=20,
        sslmode="require" if os.environ.get("DB_SSL", "").lower() == "true" else "prefer")
    conn.read_only = True
    return conn


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
