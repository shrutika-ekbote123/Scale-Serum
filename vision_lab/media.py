"""
Step 7 - the media layer. ffmpeg and ffprobe, wrapped.

WHAT THIS OWNS
    Fetching the creative, finding out what it actually is, sampling frames at a
    fixed rate, detecting shot boundaries, and pulling the audio track out for
    transcription. Nothing here knows anything about attention or scoring.

WHY ffmpeg AND NOT cv2.VideoCapture
    OpenCV can decode video, but ffmpeg samples at an exact frame rate, gives us
    scene-change detection for free, and is needed for the audio track regardless.
    One tool for media rather than two that disagree at the edges.

FRAMES COME THROUGH A PIPE, NOT A TEMP DIRECTORY
    `-f rawvideo -pix_fmt rgb24 -` writes frames straight to stdout and numpy
    reshapes the buffer. No JPEG round-trip (which would add compression
    artefacts to the very pixels the saliency model reads) and no directory of
    files to clean up when a job is killed.

THE CREATIVE IS READ AND FORGOTTEN
    It is downloaded to a temp file, processed, and deleted in a finally block -
    including when the job fails. The signed URL is never logged.

WORKING RESOLUTION
    Frames are scaled to VL_WORK_WIDTH (640) on the way out. The saliency model
    wants ~384x224 and OCR wants enough pixels to read 20 px type; 640 serves
    both and keeps 48 frames of a 1080p ad under ~40 MB of RAM. The ORIGINAL
    dimensions are what gets reported.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

from . import (
    CREATIVE_TOO_LARGE,
    CREATIVE_UNREACHABLE,
    CREATIVE_UNUSABLE,
    DECODE_FAILED,
    DURATION_TOO_LONG,
    FFMPEG_UNAVAILABLE,
    KIND_IMAGE,
    NO_VIDEO_STREAM,
)

logger = logging.getLogger("vision_lab.media")

WORK_WIDTH = int(os.environ.get("VL_WORK_WIDTH", 640))
MAX_DOWNLOAD_MB = int(os.environ.get("VL_MAX_DOWNLOAD_MB", 400))
FFMPEG_TIMEOUT = int(os.environ.get("VL_FFMPEG_TIMEOUT_SECONDS", 600))
DOWNLOAD_TIMEOUT = int(os.environ.get("VL_DOWNLOAD_TIMEOUT_SECONDS", 600))

# Scene-change sensitivity. 0.4 is ffmpeg's usual working default for cuts in
# edited video; a lower value fires on camera movement, a higher one misses soft
# transitions. Config so it can be tuned without a code change.
SCENE_THRESHOLD = float(os.environ.get("VL_SCENE_THRESHOLD", 0.4))

# Above this the extracted WAV is dropped rather than held in memory and posted.
# Shares its variable with vision_lab/transcript.py: one cap, two places that
# must not disagree about it.
MAX_AUDIO_BYTES = int(os.environ.get("VL_MAX_AUDIO_BYTES", 60 * 1024 * 1024))


class MediaError(Exception):
    """Carries a reason code the pipeline turns into a stated failure."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(detail or reason)


# --------------------------------------------------------------------------- binaries
def _binary(name: str) -> str:
    """Find ffmpeg/ffprobe.

    winget adds them to PATH but an already-running shell does not see it, so a
    freshly provisioned box works while the terminal that provisioned it does
    not. VL_FFMPEG_DIR is the explicit escape hatch.
    """
    explicit = os.environ.get("VL_FFMPEG_DIR")
    if explicit:
        candidate = os.path.join(explicit, name + (".exe" if os.name == "nt" else ""))
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    raise MediaError(FFMPEG_UNAVAILABLE,
                     f"{name} is not on PATH. Install ffmpeg, or set VL_FFMPEG_DIR.")


def available() -> bool:
    try:
        _binary("ffmpeg")
        _binary("ffprobe")
        return True
    except MediaError:
        return False


def _run(args: list[str], *, capture: bool = True, timeout: int = FFMPEG_TIMEOUT):
    try:
        return subprocess.run(args, capture_output=capture, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as err:
        raise MediaError(DECODE_FAILED, "ffmpeg timed out") from err
    except OSError as err:
        raise MediaError(FFMPEG_UNAVAILABLE, str(err)) from err


# --------------------------------------------------------------------------- fetch
def download(url: str, dest: str) -> int:
    """Stream the creative to disk, refusing anything over the size cap.

    Streamed rather than read into memory: a 200 MB creative held in RAM inside a
    process that is about to allocate 48 frames of pixel data is how a small VPS
    gets OOM-killed mid-job.
    """
    import httpx

    limit = MAX_DOWNLOAD_MB * 1024 * 1024
    written = 0
    try:
        with httpx.stream("GET", url, timeout=DOWNLOAD_TIMEOUT,
                          follow_redirects=True) as response:
            if response.status_code >= 400:
                raise MediaError(CREATIVE_UNREACHABLE,
                                 f"HTTP {response.status_code} fetching the creative")
            declared = response.headers.get("content-length")
            if declared and int(declared) > limit:
                raise MediaError(CREATIVE_TOO_LARGE,
                                 f"{int(declared) / 1048576:.0f} MB exceeds "
                                 f"the {MAX_DOWNLOAD_MB} MB cap")
            with open(dest, "wb") as handle:
                for chunk in response.iter_bytes(1024 * 256):
                    written += len(chunk)
                    if written > limit:
                        raise MediaError(
                            CREATIVE_TOO_LARGE,
                            f"exceeded the {MAX_DOWNLOAD_MB} MB cap while downloading")
                    handle.write(chunk)
    except MediaError:
        raise
    except Exception as err:  # noqa: BLE001 - httpx raises a wide family
        raise MediaError(CREATIVE_UNREACHABLE, str(err)) from err

    if written == 0:
        raise MediaError(CREATIVE_UNUSABLE, "the creative is empty")
    return written


# --------------------------------------------------------------------------- probe
def probe(path: str) -> dict:
    """What the file actually is. ffprobe overrides whatever the caller claimed."""
    result = _run([_binary("ffprobe"), "-v", "error", "-print_format", "json",
                   "-show_format", "-show_streams", path], timeout=120)
    if result.returncode != 0:
        raise MediaError(CREATIVE_UNUSABLE,
                         (result.stderr or b"").decode("utf-8", "replace")[:200])
    try:
        info = json.loads(result.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError as err:
        raise MediaError(CREATIVE_UNUSABLE, "ffprobe returned nothing usable") from err

    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video:
        raise MediaError(NO_VIDEO_STREAM, "the file contains no video or image stream")

    duration = None
    for source in (info.get("format", {}).get("duration"), video.get("duration")):
        try:
            duration = float(source)
            break
        except (TypeError, ValueError):
            continue

    return {
        "duration_seconds": duration,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "codec": video.get("codec_name"),
        "fps": _parse_fps(video.get("avg_frame_rate")),
        "has_audio": audio is not None,
        "frame_count": _as_int(video.get("nb_frames")),
    }


def _parse_fps(rate: Optional[str]) -> Optional[float]:
    if not rate or "/" not in rate:
        return None
    numerator, denominator = rate.split("/", 1)
    try:
        denom = float(denominator)
        return round(float(numerator) / denom, 3) if denom else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- frames
def _work_size(width: int, height: int) -> tuple[int, int]:
    """Scale to WORK_WIDTH, keeping the aspect ratio, with even dimensions -
    rawvideo with an odd height silently produces a misaligned buffer."""
    if not width or not height:
        return WORK_WIDTH, 360
    if width <= WORK_WIDTH:
        return width - (width % 2), height - (height % 2)
    scaled = int(round(height * (WORK_WIDTH / width)))
    return WORK_WIDTH, scaled - (scaled % 2)


def extract_frames(path: str, *, sample_fps: float, max_frames: int,
                   width: int, height: int) -> list[np.ndarray]:
    """Sample frames at a fixed rate and return them as RGB arrays."""
    out_w, out_h = _work_size(width, height)
    args = [
        _binary("ffmpeg"), "-v", "error", "-i", path,
        "-vf", f"fps={sample_fps},scale={out_w}:{out_h}",
        "-frames:v", str(max_frames),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    result = _run(args)
    if result.returncode != 0 and not result.stdout:
        raise MediaError(DECODE_FAILED,
                         (result.stderr or b"").decode("utf-8", "replace")[:200])

    frame_bytes = out_w * out_h * 3
    count = len(result.stdout) // frame_bytes
    if count == 0:
        raise MediaError(DECODE_FAILED, "no frames could be decoded")

    buffer = np.frombuffer(result.stdout[: count * frame_bytes], dtype=np.uint8)
    return list(buffer.reshape(count, out_h, out_w, 3))


def detect_shots(path: str, duration: Optional[float]) -> list[float]:
    """Timestamps of scene changes.

    Feeds the novelty term in the attention index, and gates OCR - running text
    detection only on frames after a cut is roughly a 4x saving on the single
    most expensive step in the pipeline.
    """
    args = [_binary("ffmpeg"), "-v", "info", "-i", path,
            "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
            "-f", "null", "-"]
    result = _run(args)
    stderr = (result.stderr or b"").decode("utf-8", "replace")

    shots: list[float] = []
    for line in stderr.splitlines():
        if "pts_time:" not in line:
            continue
        try:
            value = line.split("pts_time:", 1)[1].split()[0]
            shots.append(round(float(value), 3))
        except (IndexError, ValueError):
            continue
    return sorted(set(shots))


def extract_audio(path: str, dest: str) -> Optional[str]:
    """Mono 16 kHz WAV for transcription (Step 16). None when there is no audio."""
    args = [_binary("ffmpeg"), "-v", "error", "-i", path, "-vn",
            "-ac", "1", "-ar", "16000", "-y", dest]
    result = _run(args)
    if result.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) == 0:
        return None
    return dest


def _audio_bytes(path: str, dest: str) -> Optional[bytes]:
    """The extracted WAV, read into memory. None on any failure.

    Never fatal: a creative whose audio track will not decode is still a
    perfectly analysable creative, and the report says the transcript is
    missing rather than failing the analysis over it.
    """
    try:
        if not extract_audio(path, dest):
            return None
        size = os.path.getsize(dest)
        if size > MAX_AUDIO_BYTES:
            logger.warning("extracted audio is %.0f MB, above the %.0f MB cap - "
                           "skipping the transcript", size / 1048576,
                           MAX_AUDIO_BYTES / 1048576)
            return None
        with open(dest, "rb") as handle:
            return handle.read()
    except Exception as err:  # noqa: BLE001
        logger.warning("audio extraction failed: %s", err)
        return None


# --------------------------------------------------------------------------- entry
def sample_frames(url: str, *, kind: str, sample_fps: float = 2.0,
                  max_frames: int = 120,
                  max_duration_seconds: Optional[float] = None) -> dict:
    """The VisionDeps callable. Fetch, probe, sample - then delete the file.

    Returns the media facts plus a list of frames, each `{t, index, pixels}`
    where pixels is an RGB uint8 array at the working resolution.
    """
    workdir = tempfile.mkdtemp(prefix="vl_")
    local = os.path.join(workdir, "creative")
    try:
        size = download(url, local)
        info = probe(local)

        duration = info["duration_seconds"]
        if (max_duration_seconds and duration
                and duration > float(max_duration_seconds)):
            raise MediaError(
                DURATION_TOO_LONG,
                f"{duration:.0f}s exceeds the {max_duration_seconds:.0f}s limit")

        if kind == KIND_IMAGE:
            pixels = extract_frames(local, sample_fps=1, max_frames=1,
                                    width=info["width"], height=info["height"])
            return {**info, "size_bytes": size, "shots": [],
                    "has_audio": False,
                    "frames": [{"t": 0.0, "index": 0, "pixels": pixels[0]}]}

        # THIN THE RATE RATHER THAN TRUNCATE THE AD.
        #
        # A 82 s creative at 2 fps needs 165 frames. Capping at max_frames would
        # sample the first 60 s densely and never look at the rest - which is
        # where the CTA, the brand reveal and the close live. Brand Memory and
        # the Recency trigger would both be computed from a video that stops
        # before the part they are about, and nothing in the output would say so.
        #
        # So when the cap binds we lower the rate and cover the whole runtime.
        # The effective rate is reported, because every per-second figure
        # downstream is derived from it.
        effective_fps = sample_fps
        if duration and duration * sample_fps > max_frames:
            effective_fps = round(max_frames / duration, 4)
            logger.info("thinning %.1f fps to %.3f fps to cover all %.0fs "
                        "within %d frames", sample_fps, effective_fps,
                        duration, max_frames)

        pixels = extract_frames(local, sample_fps=effective_fps,
                                max_frames=max_frames,
                                width=info["width"], height=info["height"])
        shots = detect_shots(local, duration)
        frames = [{"t": round(index / effective_fps, 3), "index": index,
                   "pixels": frame} for index, frame in enumerate(pixels)]

        # THE AUDIO COMES OUT NOW OR NOT AT ALL.
        #
        # This function deletes the downloaded video in its `finally`, so the
        # only chance to pull the track is here. Returned as BYTES rather than a
        # path: a path would outlive the workdir and leave the caller owning a
        # temp file it has no obvious reason to know about. A 90 s mono 16 kHz
        # WAV is ~2.9 MB, next to nothing beside the frames already in hand.
        audio = None
        if info.get("has_audio"):
            audio = _audio_bytes(local, os.path.join(workdir, "audio.wav"))

        covered = (len(frames) / effective_fps) if effective_fps else 0
        logger.info("sampled %d frames at %.3f fps [%.0fs of %.0fs, %d shots, %.0f MB]",
                    len(frames), effective_fps, covered, duration or 0,
                    len(shots), size / 1048576)
        return {**info, "size_bytes": size, "shots": shots, "frames": frames,
                "audio": audio,
                "audio_bytes": len(audio) if audio else 0,
                "requested_sample_fps": sample_fps,
                "sample_fps": effective_fps,
                "sample_fps_thinned": effective_fps != sample_fps,
                "coverage_seconds": round(covered, 2),
                "coverage_fraction": round(covered / duration, 4) if duration else None}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
