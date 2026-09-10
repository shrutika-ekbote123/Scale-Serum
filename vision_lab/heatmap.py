"""
Step 12 - the attention overlay, and getting it into S3.

WHAT IT DRAWS
    The saliency map as a colour ramp composited over the frame, with the top
    three attention centres ringed and numbered - the "Attention Report" image
    in the Vision Lab prototype.

WHY THE COLOURS ARE WHAT THEY ARE
    Cold blue through cyan and amber to red, the convention every eye-tracking
    report uses. Reading it needs no legend because the audience has seen it
    before; inventing a palette here would cost recognition for nothing.

STORE KEYS, SIGN AT READ TIME
    upload() returns an S3 OBJECT KEY, never a URL. A presigned URL persisted
    into a report expires, and a report read three weeks later would serve dead
    image links. The GET endpoint signs the key freshly on every read, which
    keeps the images private and never stale.

WHAT IS RENDERED, AND WHAT IS NOT
    Full overlays for the hero frame and each key moment; thumbnails for the
    strip. Not all 48 frames - that is 48 uploads and ~15 MB per analysis for
    images nobody opens. The timeline's "click a point to see that frame" is
    served on demand later.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("vision_lab.heatmap")

S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "")
S3_PREFIX = os.environ.get("VL_S3_PREFIX", "vision-lab")
S3_REGION = os.environ.get("AWS_REGION", "")

OVERLAY_ALPHA = float(os.environ.get("VL_HEATMAP_ALPHA", 0.55))
THUMBNAIL_WIDTH = int(os.environ.get("VL_THUMBNAIL_WIDTH", 320))

_client = None


# --------------------------------------------------------------------------- render
def colourise(saliency: np.ndarray, shape: tuple[int, int]):
    """Saliency to an RGB heat image, plus the per-pixel intensity behind it.

    Normalised per frame by its own maximum, not globally: the map already sums
    to 1, so a frame with one strong focus and a frame with three weak ones would
    otherwise render at wildly different brightness and read as "this frame got
    less attention", which is not what a saliency map means.

    The intensity is returned as well because the blend needs it - see render().
    """
    height, width = shape
    resized = cv2.resize(saliency.astype(np.float32), (width, height),
                         interpolation=cv2.INTER_CUBIC)
    peak = float(resized.max())
    intensity = (resized / peak) if peak > 0 else np.zeros((height, width),
                                                           dtype=np.float32)
    intensity = np.clip(intensity, 0.0, 1.0)
    coloured = cv2.applyColorMap((intensity * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB), intensity


def render(pixels: np.ndarray, saliency: np.ndarray,
           peaks: Optional[list[dict]] = None) -> bytes:
    """The overlay, as PNG bytes.

    THE ALPHA FOLLOWS THE ATTENTION. IT IS NOT UNIFORM.
        Blending the whole frame at a flat 55% - which this did - paints the
        COLD end of the colour map over the ad as well as the hot end. JET's
        cold end is dark blue, so every unattended area got 55% dark blue on top
        of it: on a real creative the white slide behind the headline came back
        as flat lavender and the ad was no longer visible underneath its own
        report.

        Weighting each pixel by its own predicted attention means an ignored
        region keeps the ad's real pixels, a hot region is fully coloured, and
        the gradient between them is itself the finding. That is also what makes
        the picture answer the question a marketer is asking - "where did the
        eye go ON MY AD" - rather than replacing the ad with a colour field.
    """
    height, width = pixels.shape[:2]
    heat, intensity = colourise(saliency, (height, width))

    weight = (intensity * OVERLAY_ALPHA)[..., None]
    blended = pixels.astype(np.float32) * (1.0 - weight) \
        + heat.astype(np.float32) * weight
    canvas = np.clip(blended, 0, 255).astype(np.uint8)

    for peak in (peaks or []):
        box = peak.get("box") or []
        if len(box) != 4:
            continue
        x0, y0 = int(box[0] * width), int(box[1] * height)
        x1, y1 = int(box[2] * width), int(box[3] * height)
        centre = ((x0 + x1) // 2, (y0 + y1) // 2)
        cv2.circle(canvas, centre, 18, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, centre, 17, (20, 20, 20), -1, lineType=cv2.LINE_AA)
        label = str(peak.get("rank", "?"))
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(canvas, label,
                    (centre[0] - size[0] // 2, centre[1] + size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA)

    return encode_png(canvas)


def thumbnail(pixels: np.ndarray, width: int = THUMBNAIL_WIDTH) -> bytes:
    height = int(pixels.shape[0] * (width / pixels.shape[1]))
    small = cv2.resize(pixels, (width, height), interpolation=cv2.INTER_AREA)
    return encode_png(small)


def encode_png(rgb: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("could not encode the overlay as PNG")
    return buffer.tobytes()


# --------------------------------------------------------------------------- storage
def storage_configured() -> bool:
    return bool(S3_BUCKET and os.environ.get("AWS_ACCESS_KEY_ID"))


def client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("s3", region_name=S3_REGION or None)
    return _client


def object_key(analysis_id: str, name: str) -> str:
    return f"{S3_PREFIX.strip('/')}/{analysis_id}/{name}"


def upload(analysis_id: str, name: str, payload: bytes,
           content_type: str = "image/png") -> Optional[str]:
    """Store one rendered image. Returns the KEY, never a URL - see the module
    docstring. None when object storage is not configured, which degrades the
    report to a heatmap-less one rather than failing the analysis."""
    if not storage_configured():
        return None
    key = object_key(analysis_id, name)
    try:
        client().put_object(Bucket=S3_BUCKET, Key=key, Body=payload,
                            ContentType=content_type)
        return key
    except Exception as err:  # noqa: BLE001
        logger.warning("heatmap upload failed [analysis_id=%s]: %s", analysis_id, err)
        return None


def signed_url(key: Optional[str], ttl: Optional[int] = None) -> Optional[str]:
    """Sign a stored key for reading, at request time.

    Called by the GET endpoint on every read so the link in a response is always
    live, however old the analysis is.
    """
    if not key or not storage_configured():
        return None
    seconds = int(ttl or os.environ.get("AWS_S3_URL_TTL_SECONDS", 3600))
    try:
        return client().generate_presigned_url(
            "get_object", Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=seconds)
    except Exception as err:  # noqa: BLE001
        logger.warning("could not sign %s: %s", key, err)
        return None


def delete_all(analysis_id: str) -> int:
    """Remove every rendered image for an analysis. Used by DELETE."""
    if not storage_configured():
        return 0
    prefix = f"{S3_PREFIX.strip('/')}/{analysis_id}/"
    removed = 0
    try:
        listing = client().list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        keys = [{"Key": item["Key"]} for item in listing.get("Contents", [])]
        if keys:
            client().delete_objects(Bucket=S3_BUCKET, Delete={"Objects": keys})
            removed = len(keys)
    except Exception as err:  # noqa: BLE001
        logger.warning("could not clean up %s: %s", prefix, err)
    return removed
