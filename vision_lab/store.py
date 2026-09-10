"""
Persistence - MongoDB, reusing the connection app.py already owns.

WHY THE COLLECTION IS INJECTED
    Same rule as sales_call_analyzer and purchase_probability_model: this
    package opens no connections. app.py holds the one motor client and hands
    the collections in, so there is one driver, one pool, and one place where
    storage can be unconfigured. worker.py is the single exception - it is a
    separate process, so it builds its own client and injects it here the same
    way.

WHAT IS STORED AND WHY
    `measurements`  the per-frame record: saliency stats, regions, OCR boxes.
                    This is the durable artefact - scores can be recomputed
                    from it when management sets real weights, with no ffmpeg,
                    no model and no provider call. Kept in its own collection
                    because it is an order of magnitude larger than the report
                    and nothing reading the report needs it.
    `report`        the assembled response, so a GET is a single read.
    `analysis`      the verified LLM ratings and evidence, kept for rescore.

WHAT IS NOT STORED
    The creative itself. Presigned heatmap URLs - those would leave a stored
    report serving dead image links weeks later, so we keep S3 object KEYS and
    sign them at read time.

THE ONE CREDENTIAL WE DO HOLD, AND FOR HOW LONG
    The Sales Call Analyzer never stores its signed audio URL because its job
    runs in-process and holds the request in memory. Vision Lab's worker is a
    SEPARATE PROCESS, so the only thing it can see is the job document - the
    signed creative URL therefore has to be written down.

    It is held in `creative_url`, and cleared the moment the worker has the
    bytes or the job reaches a terminal state, so it lives for the length of one
    download rather than the length of the record. It is never logged and never
    returned in any response.

    The better answer is to store the S3 bucket and key and have the worker
    presign it with its own credentials, which it already holds in order to
    write heatmaps. That removes the stored credential entirely, and it is the
    right change to make when boto3 arrives in Phase 4 - it is not done here
    only because there is no S3 client in the skeleton yet.

THE JOB DOCUMENT IS THE QUEUE
    claim_next() is an atomic find_one_and_update. Two workers cannot claim the
    same job, and there is no Redis, no Celery and no second datastore to keep
    alive.

IDEMPOTENCY
    A fingerprint over the inputs AND the versions that would shape the output.
    An identical request returns the stored analysis and costs nothing; a
    changed framework version or model is genuinely a different analysis and is
    allowed to run.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from . import (
    ACTIVE_STATUSES,
    CLAIMED_STATUSES,
    PROCESSING_INTERRUPTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_SKIPPED,
    TERMINAL_STATUSES,
)
from pymongo import ReturnDocument

from .models import AnalyzeRequest

logger = logging.getLogger("vision_lab.store")

# A job whose heartbeat is older than this was killed mid-flight - almost always
# by a pm2 restart during a deploy. It is reported as interrupted and can be
# re-run; it is never left spinning forever in "analyzing_frames". Longer than
# the sales-call equivalent because a long video legitimately takes minutes.
STALE_AFTER_SECONDS = int(os.environ.get("VL_JOB_STALE_SECONDS", 1800))

MEASUREMENTS_TTL_DAYS = int(os.environ.get("VL_MEASUREMENTS_TTL_DAYS", 180))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_analysis_id() -> str:
    """Same id style as brand_brain_id in app.py."""
    return uuid.uuid4().hex


def compute_fingerprint(request: AnalyzeRequest, *, framework_version: str,
                        psychology_version: str, prompt_version: str,
                        saliency_model: str, sample_fps: float) -> str:
    """Stable identity of "this exact analysis of this exact input".

    Versions are part of the key on purpose: re-running the same creative under
    a new framework or a new model is a different analysis and should be allowed
    to proceed, while an accidental double-submit of the same file is free.

    The URL is used as the identity of the creative, minus its query string - a
    presigned S3 link carries a signature and an expiry that change on every
    presign for the same object, so including them would defeat idempotency
    entirely.
    """
    url = (request.creative.url or "").split("?", 1)[0]
    material = "|".join([
        request.creative_id,
        url,
        str(request.creative.size_bytes or ""),
        request.ad_number or "",
        framework_version, psychology_version, prompt_version,
        saliency_model or "", f"{sample_fps:g}",
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def is_stale(doc: dict, max_seconds: int = STALE_AFTER_SECONDS,
             now: Optional[datetime] = None) -> bool:
    """Has a job a worker is PROCESSING stopped reporting for too long?

    A JOB WAITING IN THE QUEUE IS NEVER STALE.
        Staleness is how a worker that died mid-analysis gets noticed: its job
        stops heartbeating. A queued job has no worker yet, so nothing could
        have died - its "heartbeat" is only its creation time. Counting it made
        every job that waited 30 minutes behind a backlog fail as
        processing_interrupted without ever being analysed: at 60-100 s per ad,
        the back of a 20-30 ad queue was silently thrown away.

        A worker that is not running at all, with jobs piling up behind it, is a
        real failure too. It is reported by /health - queue.oldest_waiting_seconds
        and worker.reason - not by failing the jobs that are waiting for it.
    """
    if not doc or doc.get("status") not in CLAIMED_STATUSES:
        return False
    beat = doc.get("heartbeat_at") or doc.get("created_at")
    if not isinstance(beat, datetime):
        return True
    if beat.tzinfo is None:
        beat = beat.replace(tzinfo=timezone.utc)
    return ((now or now_utc()) - beat) > timedelta(seconds=max_seconds)


def reusable(doc: Optional[dict], force: bool) -> bool:
    """Is this stored analysis a valid answer to a repeat request?

    A completed or deliberately skipped analysis is. A failure is not - the
    caller asking again is exactly how a retry happens. An in-flight job is
    reusable (return its id, do not start a second one) unless it has gone
    stale.
    """
    if not doc or force:
        return False
    status = doc.get("status")
    if status in (STATUS_COMPLETED, STATUS_SKIPPED):
        return True
    if status in ACTIVE_STATUSES:
        return not is_stale(doc)
    return False


class AnalysisStore:
    """Thin async wrapper over the vision_lab collections."""

    def __init__(self, collection, measurements_collection=None):
        self.collection = collection
        self.measurements = measurements_collection

    # ------------------------------------------------------------------ reads
    async def get(self, analysis_id: str) -> Optional[dict]:
        return await self.collection.find_one({"_id": analysis_id})

    async def find_by_fingerprint(self, fingerprint: str) -> Optional[dict]:
        cursor = self.collection.find({"input_fingerprint": fingerprint}).sort("created_at", -1)
        results = await cursor.to_list(length=1)
        return results[0] if results else None

    async def latest_for_creative(self, creative_id: str) -> Optional[dict]:
        cursor = self.collection.find({"creative_id": creative_id}).sort("created_at", -1)
        results = await cursor.to_list(length=1)
        return results[0] if results else None

    async def history(self, *, division: Optional[str] = None,
                      ad_number: Optional[str] = None,
                      creative_id: Optional[str] = None,
                      limit: int = 25) -> list[dict]:
        """The History tab. Versioned by ad_number when the caller supplies one."""
        query: dict[str, Any] = {}
        if division:
            query["division"] = division
        if ad_number:
            query["ad_number"] = ad_number
        if creative_id:
            query["creative_id"] = creative_id
        cursor = self.collection.find(query).sort("created_at", -1)
        return await cursor.to_list(length=max(1, min(int(limit), 100)))

    async def get_measurements(self, analysis_id: str) -> Optional[dict]:
        """What /rescore reads. None when the analysis predates measurement
        storage or the TTL has expired it."""
        if self.measurements is None:
            return None
        return await self.measurements.find_one({"_id": analysis_id})

    # ----------------------------------------------------------------- writes
    async def create(self, *, analysis_id: str, request: AnalyzeRequest,
                     fingerprint: str, versions: dict) -> dict:
        """Insert the job document. Status starts at queued, which is also what
        makes it visible to claim_next()."""
        created = now_utc()
        doc = {
            "_id": analysis_id,
            "creative_id": request.creative_id,
            "brand_brain_id": request.brand_brain_id,
            "brand_id": request.brand_id,
            "division": request.division,
            "ad_number": request.ad_number,
            "campaign_id": request.campaign_id,
            "status": STATUS_QUEUED,
            "reason": None,
            "message": None,
            "attempts": 0,
            "created_at": created,
            "updated_at": created,
            "heartbeat_at": created,
            "claimed_by": None,
            "input_fingerprint": fingerprint,
            "versions": versions,
            # The signed URL, held only until the worker has the bytes. See the
            # module docstring - this is the one credential we write down, and
            # clear_creative_url() is what takes it away again.
            "creative_url": request.creative.url,
            # Same lifetime rule as creative_url: the wordmark may itself be a
            # presigned link, and the worker needs it exactly once.
            "wordmark_url": request.brand_assets.wordmark_url,
            # What we were asked to analyse. The signed URL is deliberately NOT
            # stored - it is a credential, and it expires. The URL minus its
            # query string is kept because it identifies the object.
            "request_snapshot": {
                "creative_object": (request.creative.url or "").split("?", 1)[0],
                "kind": request.creative.kind,
                "mime_type": request.creative.mime_type,
                "size_bytes": request.creative.size_bytes,
                "duration_hint": request.creative.duration_seconds,
                "funnel_stage": request.funnel_stage,
                "has_brand_assets": bool(request.brand_assets.wordmark_url
                                         or request.brand_assets.brand_names),
                "brand_names": list(request.brand_assets.brand_names or []),
                "options": request.options.model_dump(mode="json"),
            },
            "media": None,
            "analysis": None,
            "report": None,
            "processing": {},
        }
        await self.collection.insert_one(doc)
        return doc

    async def claim_next(self, worker_id: str) -> Optional[dict]:
        """Atomically take one queued job. THE QUEUE.

        find_one_and_update is atomic in MongoDB, so two workers racing on the
        same document cannot both win - the loser sees the already-updated
        status and moves on. No lock, no broker.
        """
        stamp = now_utc()
        return await self.collection.find_one_and_update(
            {"status": STATUS_QUEUED},
            {"$set": {"status": "probing", "claimed_by": worker_id,
                      "heartbeat_at": stamp, "updated_at": stamp},
             "$inc": {"attempts": 1}},
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
        )

    async def set_status(self, analysis_id: str, status: str, **fields: Any) -> None:
        stamp = now_utc()
        update = {"status": status, "updated_at": stamp, "heartbeat_at": stamp}
        update.update(fields)
        await self.collection.update_one({"_id": analysis_id}, {"$set": update})

    async def heartbeat(self, analysis_id: str) -> None:
        await self.collection.update_one(
            {"_id": analysis_id}, {"$set": {"heartbeat_at": now_utc()}})

    async def clear_creative_url(self, analysis_id: str) -> None:
        """Drop the signed URL once it has been used or the job has ended.

        Called on every exit path, success or failure. A credential that outlives
        its use is a credential waiting to be found in a database dump.
        """
        await self.collection.update_one(
            {"_id": analysis_id},
            {"$unset": {"creative_url": "", "wordmark_url": ""}})

    async def save_media(self, analysis_id: str, media: dict) -> None:
        """Persist what we learned about the file as soon as we know it, before
        the expensive work. A later failure still leaves the duration and
        dimensions on the record."""
        stamp = now_utc()
        await self.collection.update_one(
            {"_id": analysis_id},
            {"$set": {"media": media, "updated_at": stamp, "heartbeat_at": stamp}})

    async def save_measurements(self, analysis_id: str, measurements: dict) -> None:
        """Its own collection, with a TTL. This is what makes /rescore free."""
        if self.measurements is None:
            return
        await self.measurements.update_one(
            {"_id": analysis_id},
            {"$set": {"measurements": measurements,
                      "created_at": now_utc(),
                      "expires_at": now_utc() + timedelta(days=MEASUREMENTS_TTL_DAYS)}},
            upsert=True)

    async def complete(self, analysis_id: str, *, report: dict,
                       analysis: Optional[dict] = None,
                       processing: Optional[dict] = None,
                       status: str = STATUS_COMPLETED) -> None:
        stamp = now_utc()
        await self.collection.update_one(
            {"_id": analysis_id},
            {"$set": {"status": status, "report": report, "analysis": analysis,
                      "processing": processing or {},
                      "reason": None, "message": None,
                      "updated_at": stamp, "heartbeat_at": stamp}})

    async def fail(self, analysis_id: str, *, reason: str, message: str,
                   processing: Optional[dict] = None) -> None:
        stamp = now_utc()
        update = {"status": STATUS_FAILED, "reason": reason, "message": message,
                  "updated_at": stamp, "heartbeat_at": stamp}
        if processing is not None:
            update["processing"] = processing
        await self.collection.update_one({"_id": analysis_id}, {"$set": update})

    async def mark_interrupted(self, analysis_id: str) -> None:
        """A job that stopped reporting - almost always a pm2 restart mid-flight,
        which is to say every deploy.

        Recorded as a failure with its own reason so a retry is an explicit act
        rather than a job silently sitting in 'analyzing_frames' forever.
        """
        await self.collection.update_one(
            {"_id": analysis_id, "status": {"$nin": list(TERMINAL_STATUSES)}},
            {"$set": {"status": STATUS_FAILED, "reason": PROCESSING_INTERRUPTED,
                      "message": ("Processing was interrupted before it completed. "
                                  "Retry the analysis."),
                      "updated_at": now_utc()}})

    async def ensure_indexes(self) -> None:
        """Called once at startup. Cheap and idempotent in MongoDB."""
        try:
            await self.collection.create_index([("creative_id", 1), ("created_at", -1)])
            await self.collection.create_index("input_fingerprint")
            await self.collection.create_index([("status", 1), ("created_at", 1)])
            await self.collection.create_index(
                [("division", 1), ("ad_number", 1), ("created_at", -1)])
            if self.measurements is not None:
                await self.measurements.create_index("expires_at", expireAfterSeconds=0)
        except Exception as err:  # noqa: BLE001 - never block startup on an index
            logger.warning("vision lab index creation skipped: %s", err)
