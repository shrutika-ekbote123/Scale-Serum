"""
Sarvam AI speech-to-text (Batch API) for a whole call, with speakers.

STATUS: STANDALONE. Nothing in sales_call_analyzer imports this yet.

SETTINGS - the ones the evaluation used (scripts/sca_eval/sarvam_eval.py):
    model saaras:v3, mode codemix (regional words in their script, English in
    Latin letters), language "unknown" (Sarvam detects it), diarization on,
    2 speakers. saaras:v4's keyterm option was tried and rejected: it wrote the
    keyterms over real words.

    The Batch API is used because diarization is only offered there. A job
    takes roughly 20-60 s for a 2-10 minute call.

COST (pay as you go, 2026-09-30): Rs 45 per audio hour with diarization,
    Rs 30 without - see estimated_cost_inr().

Never raises: a failure comes back as SarvamResult(ok=False, reason=...).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("transcription.sarvam_client")

MODEL = os.environ.get("SARVAM_STT_MODEL", "saaras:v3")
MODE = os.environ.get("SARVAM_STT_MODE", "codemix")
LANGUAGE = os.environ.get("SARVAM_STT_LANGUAGE", "unknown")
SPEAKERS = int(os.environ.get("SARVAM_STT_SPEAKERS", 2))
TIMEOUT_SECONDS = float(os.environ.get("SARVAM_STT_TIMEOUT_SECONDS", 900))
POLL_SECONDS = 5

INR_PER_HOUR_DIARIZED = 45.0
INR_PER_HOUR_PLAIN = 30.0

REASON_NOT_CONFIGURED = "sarvam_not_configured"
REASON_SDK_MISSING = "sarvam_sdk_missing"
REASON_PROVIDER_ERROR = "sarvam_provider_error"
REASON_NO_OUTPUT = "sarvam_no_output"
REASON_UNREADABLE = "sarvam_unreadable_output"
REASON_NO_CREDITS = "sarvam_no_credits"            # HTTP 402: the account needs topping up
REASON_RATE_LIMITED = "sarvam_rate_limited"        # HTTP 429


@dataclass
class SarvamResult:
    ok: bool
    response: Optional[dict] = None
    reason: Optional[str] = None
    error: Optional[str] = None
    ms: Optional[int] = None
    model: str = MODEL
    mode: str = MODE


def estimated_cost_inr(audio_seconds: float, *, diarization: bool = True) -> float:
    rate = INR_PER_HOUR_DIARIZED if diarization else INR_PER_HOUR_PLAIN
    return round(rate * max(audio_seconds, 0.0) / 3600, 4)


def _run_job(path: Path, out_dir: Path, api_key: str, speakers: int, timeout: float) -> dict:
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=api_key)
    job = client.speech_to_text_job.create_job(
        model=MODEL, mode=MODE, language_code=LANGUAGE,
        with_diarization=True, num_speakers=speakers)
    job.upload_files(file_paths=[str(path)])
    job.start()
    job.wait_until_complete(poll_interval=POLL_SECONDS, timeout=timeout)
    job.download_outputs(output_dir=str(out_dir))
    got = out_dir / (path.name + ".json")
    if not got.exists():
        found = list(out_dir.glob("*.json"))
        if not found:
            raise FileNotFoundError(REASON_NO_OUTPUT)
        got = found[0]
    return json.loads(got.read_text(encoding="utf-8"))


async def transcribe(audio: bytes, *, filename: str = "call.mp3",
                     speakers: int = SPEAKERS, api_key: Optional[str] = None,
                     timeout: float = TIMEOUT_SECONDS) -> SarvamResult:
    """One call's audio -> Sarvam's diarized batch response."""
    key = api_key or os.environ.get("SARVAM_API_KEY")
    if not key:
        return SarvamResult(ok=False, reason=REASON_NOT_CONFIGURED)
    try:
        import sarvamai  # noqa: F401
    except ImportError:
        return SarvamResult(ok=False, reason=REASON_SDK_MISSING)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sarvam_") as tmp:
        path = Path(tmp) / Path(filename).name
        path.write_bytes(audio)
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        try:
            response = await asyncio.to_thread(_run_job, path, out_dir, key, speakers, timeout)
        except FileNotFoundError:
            return SarvamResult(ok=False, reason=REASON_NO_OUTPUT,
                                ms=int((time.monotonic() - started) * 1000))
        except json.JSONDecodeError:
            return SarvamResult(ok=False, reason=REASON_UNREADABLE,
                                ms=int((time.monotonic() - started) * 1000))
        except Exception as err:  # noqa: BLE001 - opaque SDK and network errors
            status = getattr(err, "status_code", None)
            reason = {402: REASON_NO_CREDITS, 429: REASON_RATE_LIMITED}.get(status, REASON_PROVIDER_ERROR)
            logger.warning("Sarvam transcription failed: %s %s", type(err).__name__, status or "")
            return SarvamResult(ok=False, reason=reason,
                                error=f"{type(err).__name__} {status}" if status else type(err).__name__,
                                ms=int((time.monotonic() - started) * 1000))
    return SarvamResult(ok=True, response=response, ms=int((time.monotonic() - started) * 1000))
