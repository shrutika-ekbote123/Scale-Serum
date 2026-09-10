"""thumbnail_url on POST /analyze and GET /history, and the poster frame behind it.

WHAT IT MUST AND MUST NOT DO
    A new submission has no frames yet, so its thumbnail_url is null - never a
    placeholder that looks like an image. A completed analysis returns a signed
    link to a plain frame of the ad: the heatmap's frame, which is chosen to skip
    black openings and fades, rendered without the overlay. Analyses made before
    posters existed still get a thumbnail - the strip frame nearest that moment.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

# Fixtures and the configured app come from the /analyze suite.
from test_vision_lab_api import (  # noqa: F401 - fixtures are used by name
    BODY,
    HEADERS,
    client,
    wired,
    work_the_queue,
)
from test_vision_lab_pipeline import make_deps, make_store, seed

import vision_lab as vl
from vision_lab import heatmap as hm
from vision_lab import pipeline as pl
from vision_lab import stubs

SIGNED = "https://bucket.s3.amazonaws.com/{key}?X-Amz-Signature=fresh"


@pytest.fixture
def signing(monkeypatch):
    """Sign without AWS: the link names the key, so a test can see which image."""
    monkeypatch.setattr(hm, "signed_url",
                        lambda key, ttl=None: SIGNED.format(key=key) if key else None)


def submit(client, creative_id="cre_thumb"):
    return client.post("/api/vision-lab/analyze", json=dict(BODY, creative_id=creative_id),
                       headers=HEADERS).json()


def set_report(store, analysis_id, **fields):
    """The fake collection has no dotted-path $set, so replace the report whole."""
    doc = asyncio.run(store.get(analysis_id))
    report = dict(doc.get("report") or {}, **fields)
    asyncio.run(store.collection.update_one({"_id": analysis_id},
                                            {"$set": {"report": report}}))


# =========================================================================== #
# POST /analyze
# =========================================================================== #
def test_a_new_submission_has_no_thumbnail_yet(wired, client, signing):
    """Nothing has been downloaded, so there is no frame to show - and the field
    says so with null rather than being absent or faked."""
    body = submit(client)
    assert body["status"] == vl.STATUS_QUEUED
    assert "thumbnail_url" in body
    assert body["thumbnail_url"] is None


def test_resubmitting_a_completed_analysis_returns_its_poster(wired, client, signing):
    first = submit(client)
    work_the_queue(wired)
    key = f"vision-lab/{first['analysis_id']}/poster_55674.png"
    set_report(wired, first["analysis_id"],
               poster={"frame_time": 55.674, "object_key": key, "image_url": None})

    again = submit(client)
    assert again["idempotent_hit"] is True
    assert again["status"] == vl.STATUS_COMPLETED
    assert again["thumbnail_url"] == SIGNED.format(key=key)

    # The full report carries the same image, signed at read time.
    report = client.get(again["poll_url"], headers=HEADERS).json()
    assert report["poster"]["image_url"] == SIGNED.format(key=key)


def test_an_analysis_from_before_posters_uses_the_nearest_strip_frame(wired, client, signing):
    """The heatmap's moment is a representative frame; the strip frame nearest it
    is the same scene or close. Not the first frame, which is so often black."""
    first = submit(client)
    work_the_queue(wired)
    aid = first["analysis_id"]
    set_report(wired, aid,
               heatmap={"frame_time": 55.674, "object_key": None, "peaks": []},
               thumbnails=[{"t": 0.0, "object_key": f"vision-lab/{aid}/thumb_0.png"},
                           {"t": 27.5, "object_key": f"vision-lab/{aid}/thumb_27500.png"},
                           {"t": 55.0, "object_key": f"vision-lab/{aid}/thumb_55000.png"}])

    assert submit(client)["thumbnail_url"] == \
        SIGNED.format(key=f"vision-lab/{aid}/thumb_55000.png")


def test_with_no_image_at_all_the_thumbnail_is_null(wired, client, signing):
    """A completed analysis on a server without S3 has no pictures. null, not a
    broken link."""
    submit(client)
    work_the_queue(wired)
    assert submit(client)["thumbnail_url"] is None


def test_a_refused_submission_has_no_thumbnail(wired, client, signing):
    body = client.post("/api/vision-lab/analyze", headers=HEADERS,
                       json=dict(BODY, creative={"url": "https://evil.example.com/x.mp4"})).json()
    assert body["status"] == vl.STATUS_FAILED
    assert body["thumbnail_url"] is None


# =========================================================================== #
# GET /history
# =========================================================================== #
def test_history_rows_carry_their_thumbnail(wired, client, signing):
    done = submit(client, "cre_done")
    work_the_queue(wired)
    key = f"vision-lab/{done['analysis_id']}/poster_1000.png"
    set_report(wired, done["analysis_id"],
               poster={"frame_time": 1.0, "object_key": key, "image_url": None})
    waiting = submit(client, "cre_waiting")

    items = {i["analysis_id"]: i for i in
             client.get("/api/vision-lab/history", headers=HEADERS).json()["items"]}
    assert items[done["analysis_id"]]["thumbnail_url"] == SIGNED.format(key=key)
    assert items[waiting["analysis_id"]]["thumbnail_url"] is None


# =========================================================================== #
# The poster itself
# =========================================================================== #
def _framed(url, *, kind, sample_fps=2.0, max_frames=120, max_duration_seconds=None):
    """The stub media layer, with real pixels so there is something to render."""
    media = stubs.sample_frames(url, kind=kind, sample_fps=sample_fps,
                                max_frames=max_frames,
                                max_duration_seconds=max_duration_seconds)
    for frame in media["frames"]:
        frame["pixels"] = np.zeros((8, 8, 3), dtype=np.uint8)
    return media


def _run(render_heatmap):
    uploads = {}

    def upload(analysis_id, name, payload):
        uploads[name] = payload
        return f"vision-lab/{analysis_id}/{name}"

    async def scenario():
        store = make_store()
        await seed(store)
        claimed = await store.claim_next("worker-a")
        deps = make_deps(store, sample_frames=_framed,
                         predict_saliency=lambda frame: {"map": np.full((4, 4), 1 / 16),
                                                         "peaks": []},
                         render_heatmap=render_heatmap,
                         make_thumbnail=lambda pixels: b"PLAIN FRAME",
                         upload=upload)
        return (await pl.run_analysis(claimed, deps))["report"]

    return asyncio.run(scenario()), uploads


def test_the_poster_is_the_heatmaps_frame_without_the_overlay():
    report, uploads = _run(lambda pixels, smap, peaks: b"HEATMAP OVERLAY")

    poster = report["poster"]
    assert poster["object_key"].split("/")[-1].startswith("poster_")
    # The same moment as the heatmap - the frame chosen to skip blank openings...
    assert poster["frame_time"] == report["heatmap"]["frame_time"]
    # ...but the plain picture, not the one with the heat painted on it.
    name = poster["object_key"].split("/")[-1]
    assert uploads[name] == b"PLAIN FRAME"


def test_a_failed_heatmap_does_not_cost_the_poster():
    def broken(*_args):
        raise RuntimeError("render failed")

    report, _ = _run(broken)
    assert report["heatmap"]["object_key"] is None
    assert report["poster"]["object_key"]
