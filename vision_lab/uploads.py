"""
Uploading a creative directly to Vision Lab - POST /api/vision-lab/upload.

WHY THIS EXISTS, AND WHY IT IS NOT THE PRODUCTION PATH
    Vision Lab normally never touches the file. The caller puts the creative in
    S3 and sends a presigned link; the worker downloads it. That is still the
    right path for the ScaleSerum frontend: a 150 MB video uploaded through the
    AI service ties up a connection on this box for the whole transfer and
    duplicates what S3 already does.

    This exists for testing and for callers holding a file but no bucket access
    - one form-data request instead of upload, presign, copy, paste. It returns
    the same presigned link /analyze consumes, and can queue the analysis in the
    same call.

THE FILE NEVER SITS IN MEMORY, AND NEVER BLOCKS THE EVENT LOOP
    Starlette spools a multipart file to disk above 1 MB. boto3's
    upload_fileobj streams it from there in parts, and app.py runs it in a
    thread - so a 150 MB upload costs this process a temp file and a thread,
    not 150 MB of RAM and not a stalled onboarding request.

WHERE IT GOES
    s3://<AWS_S3_BUCKET>/<VL_UPLOAD_PREFIX>/<YYYY-MM-DD>/<uuid>/<safe-name>

    Its own prefix - not `vision-lab/` (heatmaps), not `vision-lab-test/` (the
    reference ads) - so a lifecycle rule can expire uploads on their own. The
    date partition is what keeps that rule a one-liner, and the uuid means two
    uploads of "final.mp4" never overwrite each other.

NO OPENCV HERE
    heatmap.py already has an S3 client, but importing it pulls in OpenCV. An
    upload needs boto3 and nothing else, so this module keeps its own.
"""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from typing import BinaryIO, Optional

from . import CREATIVE_TOO_LARGE, CREATIVE_UNUSABLE, UNSUPPORTED_FORMAT

S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "")
S3_REGION = os.environ.get("AWS_REGION", "")
UPLOAD_PREFIX = os.environ.get("VL_UPLOAD_PREFIX", "vision-lab-uploads")

# Same ceiling as a creative fetched by URL. A limit that differed by route
# would let a file the worker refuses to download be uploaded anyway.
MAX_UPLOAD_MB = int(os.environ.get("VL_MAX_UPLOAD_MB")
                    or os.environ.get("VL_MAX_DOWNLOAD_MB") or 400)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Multipart framing and the small form fields ride on top of the file. The
# early Content-Length check in app.py allows for them; the exact check runs on
# the file's real size once it has been received.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024

# How long the returned link stays usable. Deliberately longer than the 1 h a
# heatmap link gets: the worker downloads the video only when it reaches the
# job, and with one worker and a queue ahead of it an hour can pass before it
# does - at which point an expired link fails the job as creative_unreachable.
# Capped at S3's own 7-day maximum for SigV4.
URL_TTL_SECONDS = min(int(os.environ.get("VL_UPLOAD_URL_TTL_SECONDS", 6 * 3600)),
                      7 * 24 * 3600)

# What the pipeline can decode, by extension. Mirrors the lists
# pipeline.classify_creative accepts - a test asserts the two agree, because
# accepting an upload the worker then refuses is the worst of both.
MEDIA_TYPES = {
    ".mp4": ("video", "video/mp4"),
    ".m4v": ("video", "video/mp4"),
    ".mov": ("video", "video/quicktime"),
    ".webm": ("video", "video/webm"),
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".png": ("image", "image/png"),
    ".webp": ("image", "image/webp"),
}

_client = None


class UploadRefused(Exception):
    """An upload we will not store, with a reason code the caller can branch on."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        self.message = message
        super().__init__(message)


def storage_configured() -> bool:
    return bool(S3_BUCKET and os.environ.get("AWS_ACCESS_KEY_ID"))


def client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("s3", region_name=S3_REGION or None)
    return _client


# --------------------------------------------------------------------------- naming
def safe_filename(name: Optional[str]) -> str:
    """A filename that is safe to put in an S3 key.

    The client chooses the filename, so it is untrusted input. Anything that
    looks like a path is stripped - Windows and POSIX separators both, since a
    client on Windows can send "C:\\Users\\...\\ad.mp4" - and what remains is
    reduced to letters, digits, hyphens and underscores. "../../etc/passwd.mp4"
    becomes "passwd.mp4" and stays under the upload prefix.
    """
    base = re.split(r"[\\/]", name or "")[-1]
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_").lower()[:80] or "upload"
    ext = re.sub(r"[^A-Za-z0-9]+", "", ext).lower()[:5]
    return f"{stem}.{ext}" if ext else stem


def object_key(filename: Optional[str], *, today: Optional[datetime] = None,
               token: Optional[str] = None) -> str:
    day = (today or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return (f"{UPLOAD_PREFIX.strip('/')}/{day}/{token or uuid.uuid4().hex}/"
            f"{safe_filename(filename)}")


def creative_id_for(filename: Optional[str]) -> str:
    """A readable id for an upload that arrived without one, so it is still
    recognisable in /history: cre_ad3_1a2b3c4d rather than a bare uuid."""
    stem = safe_filename(filename).rsplit(".", 1)[0].replace("-", "_")[:40]
    return f"cre_{stem}_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- checks
def classify(filename: Optional[str], content_type: Optional[str]) -> tuple[str, str]:
    """(kind, mime) for a file we can analyse, or UploadRefused.

    The EXTENSION decides, because that is what the pipeline and ffprobe act
    on. The declared content type is believed when it rules the file OUT - the
    same rule as pipeline.classify_creative - and otherwise only has to agree.
    `application/octet-stream`, which clients send for anything they do not
    recognise, says nothing either way and is ignored.
    """
    name = safe_filename(filename)
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in MEDIA_TYPES:
        raise UploadRefused(
            UNSUPPORTED_FORMAT,
            f"'{ext or name}' is not a format Vision Lab analyses. "
            f"Use one of {', '.join(sorted(MEDIA_TYPES))}.")
    kind, mime = MEDIA_TYPES[ext]

    declared = (content_type or "").split(";", 1)[0].strip().lower()
    if declared and declared != "application/octet-stream":
        declared_kind = declared.split("/", 1)[0]
        if declared_kind not in ("video", "image"):
            raise UploadRefused(
                UNSUPPORTED_FORMAT,
                f"The file was sent as {declared}, which is not a video or an image.")
        if declared_kind != kind:
            raise UploadRefused(
                UNSUPPORTED_FORMAT,
                f"The file is named {ext} but was sent as {declared} - its name and "
                f"its declared type disagree about what it is.")
    return kind, mime


def measure(fileobj: BinaryIO) -> int:
    """Size in bytes, leaving the file positioned at the start for the upload."""
    fileobj.seek(0, os.SEEK_END)
    size = fileobj.tell()
    fileobj.seek(0)
    return size


def check_size(size: int) -> None:
    """Empty is unusable; over the cap is too large. Refused, never truncated."""
    if size <= 0:
        raise UploadRefused(CREATIVE_UNUSABLE, "The uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise UploadRefused(
            CREATIVE_TOO_LARGE,
            f"The file is {size / 1048576:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES / 1048576:.1f} MB.")


# --------------------------------------------------------------------------- S3
def put(fileobj: BinaryIO, key: str, mime: str) -> None:
    """Stream to S3. Synchronous - call it in a thread, never on the event loop.

    upload_fileobj switches to multipart above 8 MB and retries parts on its
    own, which is what a 150 MB video on an unreliable link needs.
    """
    client().upload_fileobj(fileobj, S3_BUCKET, key, ExtraArgs={"ContentType": mime})


def sign(key: Optional[str], ttl: Optional[int] = None) -> Optional[str]:
    """The presigned GET link /analyze consumes. A credential - never log it."""
    if not key or not storage_configured():
        return None
    return client().generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=int(ttl or URL_TTL_SECONDS))
