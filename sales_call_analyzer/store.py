"""
Persistence - MongoDB, reusing the connection app.py already owns.

WHY THE COLLECTION IS INJECTED
    Same rule as purchase_probability_model: this package opens no connections.
    app.py holds the one motor client and hands the collection in, so there is
    one driver, one pool, and one place where storage can be unconfigured.

WHAT IS STORED AND WHY
    `analysis`      the verified per-criterion ratings and evidence. This is the
                    durable artefact: scores can be recomputed from it when
                    management sets real weights, with no provider call.
    `report`        the assembled response, so a GET is a single read.
    `transcript`    kept the moment it exists, BEFORE the LLM runs. A failed
                    analysis therefore never pays Deepgram twice.
    `raw_transcript` only when SCA_STORE_RAW_TRANSCRIPT is on. Word-level output
                    for a long call is megabytes with no consumer, so it is off
                    by default and goes to its own collection with a TTL.

WHAT IS NOT STORED
    Audio. Signed URLs. They are read and forgotten - the recording belongs to
    the backend, and a signed URL is a credential with an expiry.

IDEMPOTENCY
    A fingerprint over the inputs AND the versions that would shape the output.
    An identical request returns the stored analysis and costs nothing; a
    changed framework version or model is genuinely a different analysis and is
    allowed to run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from . import (
    ACTIVE_STATUSES,
    PROCESSING_INTERRUPTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_SKIPPED,
    TERMINAL_STATUSES,
)
from .models import AnalyzeRequest

logger = logging.getLogger("sales_call_analyzer.store")

# A job whose heartbeat is older than this was killed mid-flight - almost always
# by a pm2 restart during a deploy. It is reported as interrupted and can be
# re-run; it is never left spinning forever in "analyzing".
STALE_AFTER_SECONDS = int(os.environ.get("SCA_JOB_STALE_SECONDS", 900))

STORE_RAW_TRANSCRIPT = os.environ.get("SCA_STORE_RAW_TRANSCRIPT", "false").lower() == "true"
RAW_TRANSCRIPT_TTL_DAYS = int(os.environ.get("SCA_RAW_TRANSCRIPT_TTL_DAYS", 30))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_analysis_id() -> str:
    """Same id style as brand_brain_id in app.py."""
    return uuid.uuid4().hex


def compute_fingerprint(request: AnalyzeRequest, *, framework_version: str,
                        prompt_version: str, llm_model: str,
                        transcription_model: str) -> str:
    """Stable identity of "this exact analysis of this exact input".

    Versions are part of the key on purpose: re-running the same call under a
    new framework or a new model is a different analysis and should be allowed
    to proceed, while an accidental double-submit of the same call is free.
    """
    supplied = request.transcript
    transcript_material = ""
    if supplied:
        if supplied.segments:
            transcript_material = json.dumps(
                [[s.speaker_id, s.start, s.end, s.text] for s in supplied.segments],
                sort_keys=True, default=str)
        else:
            transcript_material = supplied.text or ""

    material = "|".join([
        request.call_id,
        (request.audio.url if request.audio else "") or "",
        hashlib.sha256(transcript_material.encode("utf-8")).hexdigest(),
        framework_version, prompt_version, llm_model, transcription_model,
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def is_stale(doc: dict, max_seconds: int = STALE_AFTER_SECONDS,
             now: Optional[datetime] = None) -> bool:
    """Has an in-flight job stopped reporting for longer than we tolerate?"""
    if not doc or doc.get("status") not in ACTIVE_STATUSES:
        return False
    beat = doc.get("heartbeat_at") or doc.get("created_at")
    if not isinstance(beat, datetime):
        return True
    if beat.tzinfo is None:
        beat = beat.replace(tzinfo=timezone.utc)
    return ((now or now_utc()) - beat) > timedelta(seconds=max_seconds)


class AnalysisStore:
    """Thin async wrapper over the `sales_call_analyses` collection."""

    def __init__(self, collection, raw_collection=None):
        self.collection = collection
        self.raw_collection = raw_collection

    # ------------------------------------------------------------------ reads
    async def get(self, analysis_id: str) -> Optional[dict]:
        return await self.collection.find_one({"_id": analysis_id})

    async def find_by_fingerprint(self, fingerprint: str) -> Optional[dict]:
        """The most recent analysis of an identical input, if any."""
        cursor = self.collection.find({"input_fingerprint": fingerprint}).sort("created_at", -1)
        results = await cursor.to_list(length=1)
        return results[0] if results else None

    async def latest_for_call(self, call_id: str) -> Optional[dict]:
        cursor = self.collection.find({"call_id": call_id}).sort("created_at", -1)
        results = await cursor.to_list(length=1)
        return results[0] if results else None

    async def stored_transcript(self, call_id: str) -> Optional[dict]:
        """A transcript already produced for this call, from any earlier attempt.

        This is the main cost control: an analysis that failed at the LLM step
        leaves its transcript behind, so the retry skips transcription entirely.
        """
        cursor = self.collection.find(
            {"call_id": call_id, "transcript": {"$ne": None}}).sort("created_at", -1)
        results = await cursor.to_list(length=1)
        return (results[0].get("transcript") if results else None)

    # ----------------------------------------------------------------- writes
    async def create(self, *, analysis_id: str, request: AnalyzeRequest,
                     fingerprint: str, versions: dict) -> dict:
        """Insert the job document. Status starts at queued."""
        created = now_utc()
        doc = {
            "_id": analysis_id,
            "call_id": request.call_id,
            "lead_id": request.lead_id,
            "brand_id": request.brand_id,
            "brand_brain_id": request.brand_brain_id,
            "status": STATUS_QUEUED,
            "reason": None,
            "message": None,
            "attempts": 0,
            "created_at": created,
            "updated_at": created,
            "heartbeat_at": created,
            "input_fingerprint": fingerprint,
            "versions": versions,
            # What we were asked to analyse. The signed audio URL is deliberately
            # NOT stored - it is a credential, and it expires.
            "request_snapshot": {
                "has_audio": bool(request.audio and request.audio.url),
                "has_transcript": bool(request.transcript),
                "call_metadata": request.call_metadata.model_dump(mode="json"),
                "rep": request.rep.model_dump(mode="json"),
                "customer": request.customer.model_dump(mode="json"),
                "product": request.product.model_dump(mode="json"),
            },
            "transcript": None,
            "analysis": None,
            "report": None,
            "processing": {},
        }
        await self.collection.insert_one(doc)
        return doc

    async def set_status(self, analysis_id: str, status: str, **fields: Any) -> None:
        update = {"status": status, "updated_at": now_utc(), "heartbeat_at": now_utc()}
        update.update(fields)
        await self.collection.update_one({"_id": analysis_id}, {"$set": update})

    async def heartbeat(self, analysis_id: str) -> None:
        await self.collection.update_one(
            {"_id": analysis_id}, {"$set": {"heartbeat_at": now_utc()}})

    async def increment_attempts(self, analysis_id: str) -> None:
        await self.collection.update_one(
            {"_id": analysis_id},
            {"$inc": {"attempts": 1}, "$set": {"heartbeat_at": now_utc()}})

    async def save_transcript(self, analysis_id: str, transcript: dict) -> None:
        """Persist as soon as the transcript exists, before the LLM runs."""
        await self.collection.update_one(
            {"_id": analysis_id},
            {"$set": {"transcript": transcript, "updated_at": now_utc(),
                      "heartbeat_at": now_utc()}})

    async def save_raw_transcript(self, analysis_id: str, raw: dict,
                                  enabled: Optional[bool] = None) -> bool:
        """Off by default. Word-level output is large and has no consumer beyond
        debugging a diarization complaint."""
        keep = STORE_RAW_TRANSCRIPT if enabled is None else bool(enabled)
        if not keep or self.raw_collection is None:
            return False
        await self.raw_collection.update_one(
            {"_id": analysis_id},
            {"$set": {"raw": raw, "created_at": now_utc(),
                      "expires_at": now_utc() + timedelta(days=RAW_TRANSCRIPT_TTL_DAYS)}},
            upsert=True)
        return True

    async def complete(self, analysis_id: str, *, report: dict, analysis: dict,
                       processing: dict, status: str = STATUS_COMPLETED,
                       blocked: Optional[dict] = None) -> None:
        """`blocked` records which criteria this call could not satisfy (no audio,
        no product shape). It is stored because a later rescore must reproduce
        the same not-applicable decisions without re-reading the context."""
        await self.collection.update_one(
            {"_id": analysis_id},
            {"$set": {"status": status, "report": report, "analysis": analysis,
                      "blocked_criteria": blocked or {},
                      "processing": processing, "reason": None, "message": None,
                      "updated_at": now_utc(), "heartbeat_at": now_utc()}})

    async def fail(self, analysis_id: str, *, reason: str, message: str,
                   report: Optional[dict] = None,
                   processing: Optional[dict] = None) -> None:
        update = {"status": STATUS_FAILED, "reason": reason, "message": message,
                  "updated_at": now_utc(), "heartbeat_at": now_utc()}
        if report is not None:
            update["report"] = report
        if processing is not None:
            update["processing"] = processing
        await self.collection.update_one({"_id": analysis_id}, {"$set": update})

    async def mark_interrupted(self, analysis_id: str) -> None:
        """A job that stopped reporting - almost always a restart mid-flight.

        Recorded as a failure with its own reason so a retry is an explicit act
        rather than a job silently sitting in 'analyzing' forever.
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
            await self.collection.create_index([("call_id", 1), ("created_at", -1)])
            await self.collection.create_index("input_fingerprint")
            await self.collection.create_index([("status", 1), ("heartbeat_at", 1)])
            if self.raw_collection is not None:
                await self.raw_collection.create_index("expires_at", expireAfterSeconds=0)
        except Exception as err:  # noqa: BLE001 - never block startup on an index
            logger.warning("sales-call analysis index creation skipped: %s", err)


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
