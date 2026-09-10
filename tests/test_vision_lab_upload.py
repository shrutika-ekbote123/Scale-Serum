"""POST /api/vision-lab/upload, driven through the real app with a fake bucket.

No network and no AWS: `put` and `sign` are replaced, so what these tests check
is everything around the transfer - what is accepted, what is refused and with
which reason, where the file lands, that it is never held or logged in a way it
should not be, and that analyze=true produces a job indistinguishable from one
submitted through /analyze.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

# Fixtures and the configured app come from the /analyze suite, so both files
# exercise the same wiring rather than two slightly different copies of it.
from test_vision_lab_api import (  # noqa: F401 - fixtures are used by name
    HEADERS,
    client,
    wired,
    work_the_queue,
)

import vision_lab as vl
from vision_lab import pipeline as pl
from vision_lab import uploads as up

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64    # a few bytes behind an mp4 header


@pytest.fixture
def s3(monkeypatch):
    """A fake bucket: records every put, signs with a host on the test allowlist."""
    stored = {}

    def put(fileobj, key, mime):
        stored[key] = {"bytes": fileobj.read(), "mime": mime}

    monkeypatch.setattr(up, "storage_configured", lambda: True)
    monkeypatch.setattr(up, "put", put)
    monkeypatch.setattr(up, "sign", lambda key, ttl=None:
                        f"https://bucket.s3.amazonaws.com/{key}?X-Amz-Signature=abc")
    return stored


def send_upload(client, name="Ad3.mp4", data=MP4, mime="video/mp4", headers=HEADERS,
                **form):
    return client.post("/api/vision-lab/upload", headers=headers,
                       files={"file": (name, data, mime)},
                       data={k: (str(v).lower() if isinstance(v, bool) else v)
                             for k, v in form.items()})


# =========================================================================== #
# Accepted
# =========================================================================== #
def test_upload_stores_the_file_and_returns_the_link_analyze_needs(wired, client, s3):
    response = send_upload(client)
    assert response.status_code == 200
    body = response.json()

    assert body["uploaded"] is True and body["reason"] is None
    upload = body["upload"]
    assert upload["object_key"].startswith(up.UPLOAD_PREFIX + "/")
    assert upload["object_key"].endswith("/ad3.mp4")
    assert upload["size_bytes"] == len(MP4)
    assert upload["mime_type"] == "video/mp4" and upload["kind"] == "video"
    assert upload["object_key"] in upload["creative_url"]
    assert upload["creative_url_expires_in_seconds"] == up.URL_TTL_SECONDS
    # Not asked to analyse, so nothing was queued.
    assert body["analysis"] is None

    # The whole file reached the bucket, from its start - measure() rewound it.
    assert s3[upload["object_key"]] == {"bytes": MP4, "mime": "video/mp4"}


def test_upload_can_queue_the_analysis_in_the_same_call(wired, client, s3):
    body = send_upload(client, creative_id="cre_up", ad_number="042",
                       division="directors_institute",
                       brand_names="Director's Institute, DI", analyze=True).json()

    analysis = body["analysis"]
    assert analysis["status"] == vl.STATUS_QUEUED
    assert analysis["creative_id"] == "cre_up"

    # The job carries OUR signed link and the parsed brand names.
    doc = asyncio.run(wired.get(analysis["analysis_id"]))
    assert doc["creative_url"] == body["upload"]["creative_url"]
    assert doc["request_snapshot"]["brand_names"] == ["Director's Institute", "DI"]

    # And it runs to completion like any other analysis.
    work_the_queue(wired)
    report = client.get(analysis["poll_url"], headers=HEADERS).json()
    assert report["status"] == vl.STATUS_COMPLETED


def test_an_upload_without_a_creative_id_gets_a_readable_one(wired, client, s3):
    """cre_ad3_1a2b3c4d rather than a bare uuid, so it is recognisable in history."""
    body = send_upload(client, analyze=True).json()
    assert body["analysis"]["creative_id"].startswith("cre_ad3_")


def test_octet_stream_is_judged_by_the_file_name(wired, client, s3):
    """What clients send for a type they do not recognise. It says nothing either
    way, so the extension decides."""
    body = send_upload(client, name="ad.mov", mime="application/octet-stream").json()
    assert body["uploaded"] is True
    assert body["upload"]["mime_type"] == "video/quicktime"


# =========================================================================== #
# Refused - and nothing reaches the bucket
# =========================================================================== #
def test_an_unsupported_file_is_refused(wired, client, s3):
    body = send_upload(client, name="brief.pdf", mime="application/pdf").json()
    assert body["uploaded"] is False
    assert body["reason"] == vl.UNSUPPORTED_FORMAT
    assert body["upload"] is None
    assert s3 == {}


def test_a_name_and_type_that_disagree_are_refused(wired, client, s3):
    """An .mp4 declared as image/png is a mismatch worth surfacing rather than
    guessing around - the rule pipeline.classify_creative applies to a URL."""
    body = send_upload(client, name="ad.mp4", mime="image/png").json()
    assert body["reason"] == vl.UNSUPPORTED_FORMAT
    assert s3 == {}


def test_a_file_field_with_no_file_name_is_reported_missing(wired, client, s3):
    """Measured against the live service, not assumed: the multipart parser
    DROPS a file part with an empty filename, so the route never runs and
    FastAPI answers 422 with the required `file` field missing. The Postman
    request treats exactly this as "no file chosen" and skips."""
    response = client.post("/api/vision-lab/upload", headers=HEADERS,
                           files={"file": ("", b"", "application/octet-stream")})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "file"]
    assert s3 == {}


def test_a_blank_file_name_that_reaches_the_route_is_no_creative(wired, client, s3):
    """A whitespace-only name survives the parser. Without the check it reached
    classify() as a file called "upload" and was refused as "'upload' is not a
    format" - true, and no help. It is the answer /analyze gives instead."""
    response = client.post("/api/vision-lab/upload", headers=HEADERS,
                           files={"file": ("   ", b"", "application/octet-stream")})
    assert response.status_code == 200
    assert response.json()["reason"] == vl.NO_CREATIVE
    assert s3 == {}


def test_an_empty_file_is_refused_as_unusable(wired, client, s3):
    body = send_upload(client, data=b"").json()
    assert body["reason"] == vl.CREATIVE_UNUSABLE
    assert s3 == {}


def test_a_file_over_the_limit_is_413_and_never_stored(wired, client, s3, monkeypatch):
    """The exact check, on the file's real size once received."""
    monkeypatch.setattr(up, "MAX_UPLOAD_BYTES", 10)
    response = send_upload(client)
    assert response.status_code == 413
    assert response.json()["reason"] == vl.CREATIVE_TOO_LARGE
    assert s3 == {}


def test_an_oversized_declared_length_is_refused_before_the_route_runs(
        wired, client, s3, monkeypatch):
    """By the time a route runs, the multipart body is already on disk - so the
    early answer has to come from the declared length, before the route."""
    monkeypatch.setattr(up, "MAX_UPLOAD_BYTES", 0)
    monkeypatch.setattr(up, "MULTIPART_OVERHEAD_BYTES", 0)

    def route_ran(*_args, **_kwargs):
        raise AssertionError("the route ran - the guard did not stop it")
    monkeypatch.setattr(up, "classify", route_ran)

    response = send_upload(client)
    assert response.status_code == 413
    assert response.json()["reason"] == vl.CREATIVE_TOO_LARGE
    assert s3 == {}


def test_the_size_guard_leaves_every_other_route_alone(wired, client, s3, monkeypatch):
    """It is registered app-wide, so it must provably touch one path only."""
    monkeypatch.setattr(up, "MAX_UPLOAD_BYTES", 0)
    monkeypatch.setattr(up, "MULTIPART_OVERHEAD_BYTES", 0)
    assert client.get("/health").status_code == 200


def test_upload_requires_the_api_key(wired, client, s3):
    assert send_upload(client, headers={}).status_code == 401
    assert s3 == {}


# =========================================================================== #
# Stated failures, never a 500
# =========================================================================== #
def test_upload_without_storage_says_so(wired, client, s3, monkeypatch):
    monkeypatch.setattr(up, "storage_configured", lambda: False)
    body = send_upload(client).json()
    assert body["uploaded"] is False
    assert body["reason"] == vl.OBJECT_STORAGE_NOT_CONFIGURED


def test_a_failed_put_is_a_stated_failure(wired, client, s3, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("connection reset by peer")
    monkeypatch.setattr(up, "put", broken)

    response = send_upload(client)
    assert response.status_code == 200
    assert response.json()["reason"] == vl.UPLOAD_FAILED


# =========================================================================== #
# Safety
# =========================================================================== #
def test_a_path_in_the_filename_cannot_escape_the_upload_prefix(wired, client, s3):
    body = send_upload(client, name="../../etc/passwd.mp4").json()
    key = body["upload"]["object_key"]
    assert key.startswith(up.UPLOAD_PREFIX + "/")
    assert key.endswith("/passwd.mp4")
    assert ".." not in key


def test_the_signed_link_is_never_logged(wired, client, s3, caplog):
    """The link is a credential. The key and size are logged; the URL is not."""
    with caplog.at_level(logging.INFO):
        send_upload(client, analyze=True)
    assert "X-Amz-Signature" not in caplog.text


def test_upload_accepts_exactly_what_the_pipeline_decodes():
    """Accepting a file the worker then refuses would be the worst of both."""
    videos = {ext for ext, (kind, _) in up.MEDIA_TYPES.items() if kind == "video"}
    images = {ext for ext, (kind, _) in up.MEDIA_TYPES.items() if kind == "image"}
    assert videos == set(pl.VIDEO_EXTENSIONS)
    assert images == set(pl.IMAGE_EXTENSIONS)
    for kind, mime in up.MEDIA_TYPES.values():
        assert mime in (pl.VIDEO_TYPES if kind == "video" else pl.IMAGE_TYPES)


@pytest.mark.parametrize("raw, safe", [
    ("Ad3.mp4", "ad3.mp4"),
    ("Ad 3 (FINAL).MP4", "ad-3-final.mp4"),
    ("../../etc/passwd.mp4", "passwd.mp4"),
    ("C:\\Users\\User\\Downloads\\board ad.mov", "board-ad.mov"),
    (".mp4", "upload.mp4"),
    ("", "upload"),
    (None, "upload"),
])
def test_safe_filename(raw, safe):
    assert up.safe_filename(raw) == safe
