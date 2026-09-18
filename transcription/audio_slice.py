"""
Cutting a short window out of a recording. ffmpeg, wrapped.

WHY A WINDOW AND NOT THE WHOLE FILE
    Language identification needs a sample, not the call. Gemini charges by
    audio length - about 25 tokens per second - so identifying the language of a
    60-minute call from the whole file costs roughly $0.07, while three minutes
    costs about $0.003 and answers the same question.

WHY NOT THE OPENING
    Measured on real calls: reps open in English and the customer's own language
    appears later. A window starting at 0:00 answers "English" for a call that is
    mostly Marathi, which is the exact failure this is meant to avoid. So the
    window starts after the greeting.

ffmpeg IS OPTIONAL, ALWAYS
    Not installed, unreadable audio, a timeout - every failure returns None and
    says why in the log. The caller then decides whether to send the whole file
    or skip identification entirely. Nothing here may fail an analysis.

    The binary lookup mirrors vision_lab/media.py deliberately rather than
    importing it: `transcription` is shared infrastructure and must not depend on
    a feature package. VL_FFMPEG_DIR is honoured so one server setting serves
    both.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger("transcription.audio_slice")

# Where the sample is taken from. 20 s skips the greeting and the "can I speak
# to..." exchange; 180 s is long enough to hear the customer at length while
# keeping the identification cost near a fifth of a cent.
WINDOW_START_SECONDS = float(os.environ.get("SCA_LANGUAGE_ID_WINDOW_START", 20))
WINDOW_SECONDS = float(os.environ.get("SCA_LANGUAGE_ID_WINDOW_SECONDS", 180))
FFMPEG_TIMEOUT_SECONDS = float(os.environ.get("SCA_FFMPEG_TIMEOUT_SECONDS", 60))


def ffmpeg_path() -> Optional[str]:
    """SCA_FFMPEG_DIR, then Vision Lab's VL_FFMPEG_DIR, then PATH."""
    for env in ("SCA_FFMPEG_DIR", "VL_FFMPEG_DIR"):
        directory = os.environ.get(env)
        if directory:
            candidate = os.path.join(directory, "ffmpeg" + (".exe" if os.name == "nt" else ""))
            if os.path.isfile(candidate):
                return candidate
    return shutil.which("ffmpeg")


def available() -> bool:
    return ffmpeg_path() is not None


async def window(audio: bytes, *, start: Optional[float] = None,
                 duration: Optional[float] = None,
                 timeout: Optional[float] = None) -> Optional[bytes]:
    """A mono 16 kHz MP3 slice of `audio`, or None if it could not be produced.

    Mono 16 kHz because this is heard by a language identifier, not a human:
    it is what speech models want and it keeps the payload small.

    A call shorter than the window start still yields nothing rather than an
    error - ffmpeg returns an empty stream and the caller degrades.
    """
    binary = ffmpeg_path()
    if not binary:
        logger.info("ffmpeg is not available; no audio window will be cut")
        return None
    if not audio:
        return None

    start = WINDOW_START_SECONDS if start is None else start
    duration = WINDOW_SECONDS if duration is None else duration
    args = [binary, "-v", "error", "-nostdin",
            "-ss", str(start), "-t", str(duration),
            "-i", "pipe:0",
            "-ac", "1", "-ar", "16000", "-f", "mp3", "pipe:1"]
    try:
        process = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
    except Exception as err:  # noqa: BLE001 - a missing or broken binary is not fatal
        logger.warning("ffmpeg could not be started: %s", type(err).__name__)
        return None

    try:
        out, err = await asyncio.wait_for(
            process.communicate(input=audio),
            timeout=FFMPEG_TIMEOUT_SECONDS if timeout is None else timeout)
    except asyncio.TimeoutError:
        logger.warning("ffmpeg timed out cutting the language-identification window")
        try:
            process.kill()
        except Exception:  # noqa: BLE001
            pass
        return None
    except Exception as err:  # noqa: BLE001
        logger.warning("ffmpeg failed: %s", type(err).__name__)
        return None

    if process.returncode != 0 or not out:
        # stderr can name the file; only the length is safe to log.
        logger.warning("ffmpeg produced no window (exit=%s, %d bytes of stderr)",
                       process.returncode, len(err or b""))
        return None
    return out
