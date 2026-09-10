"""
The vision-worker process. pm2 entrypoint.

    pm2 start ecosystem.config.js        (see the vision-worker app)
    python -m vision_lab.worker          (locally)

WHY THIS IS A SEPARATE PROCESS
    The Sales Call Analyzer runs its jobs in a FastAPI BackgroundTask, which is
    correct there: Deepgram and Gemini are network waits, so the event loop
    stays free. Vision Lab is CPU-bound - ffmpeg decode, saliency inference and
    OCR will saturate a core for 30-120 seconds per ad. In the API process that
    would stall onboarding, Script Lab and every sales-call poll for the whole
    duration.

THE QUEUE IS THE JOB DOCUMENT
    claim_next() is an atomic find_one_and_update. Two workers cannot claim the
    same job. There is no Redis, no Celery and no second datastore to keep
    alive - and the document the API already polls is the same one the worker
    is working through.

THIS PROCESS OWNS ITS OWN MONGO CLIENT
    The rule elsewhere is that the package opens no connections and app.py
    injects them. This file is the exception because it IS a process, not a
    library - but it still injects into AnalysisStore rather than reaching into
    Mongo from the pipeline.

SHUTDOWN
    pm2 sends SIGTERM on every restart, which is to say on every deploy. We stop
    claiming immediately and let the job in flight finish if it can. Whatever is
    still running when the process dies is caught by the stale-job reaper and
    reported as processing_interrupted - never left spinning in an active
    status.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

# Import after load_dotenv so module-level env reads see the .env values.
from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from . import ANALYSIS_PROVIDER_ERROR, CLAIMED_STATUSES  # noqa: E402
from . import framework as fw  # noqa: E402
from . import pipeline as pl  # noqa: E402
from . import stubs  # noqa: E402
from .store import AnalysisStore, is_stale  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("vision_lab.worker")

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB", "scaleserum")
VL_ENABLED = os.environ.get("VL_ENABLED", "false").lower() == "true"

# The worker is a second process, so it reads these itself rather than being
# handed app.py's client. Both are optional here - unlike in app.py, which
# refuses to boot without a Gemini key - because a worker that can measure but
# not interpret still produces a report worth having.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

# Set false to force the stubs even where ffmpeg and OpenCV are installed -
# useful for reproducing a contract issue without waiting on real processing.
USE_REAL_VISION = os.environ.get("VL_USE_REAL_VISION", "true").lower() == "true"

IDLE_SLEEP_SECONDS = float(os.environ.get("VL_WORKER_IDLE_SLEEP", 3))
MAX_CONCURRENT = int(os.environ.get("VL_MAX_CONCURRENT_JOBS", 1))
REAP_EVERY_SECONDS = float(os.environ.get("VL_REAP_INTERVAL_SECONDS", 120))

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"

_stopping = False


def _request_stop(signum, _frame) -> None:
    """Stop claiming. The job in flight is given a chance to finish."""
    global _stopping
    _stopping = True
    logger.info("signal %s received - finishing the current job, then exiting", signum)


def build_deps(store: AnalysisStore) -> pl.VisionDeps:
    """Everything the pipeline needs from this process.

    The real vision modules when their tools are installed; the stubs otherwise.
    That check is at BOOT, not per job, so a box without ffmpeg says so once in
    the log rather than failing every analysis with the same message.
    """
    if not USE_REAL_VISION:
        logger.warning("running with STUB vision - no measurements will be real")
        return pl.VisionDeps(store=store, **stubs.deps_kwargs())

    from . import heatmap as hm
    from . import measure as ms
    from . import media as md
    from . import regions as reg
    from . import saliency as sal

    if not md.available():
        logger.error("ffmpeg is not installed - falling back to STUB vision. "
                     "Install ffmpeg, or set VL_FFMPEG_DIR.")
        return pl.VisionDeps(store=store, **stubs.deps_kwargs())

    info = sal.describe()
    logger.info("vision ready: saliency=%s trained=%s ocr=%s",
                info["saliency_method"], info["saliency_trained"],
                reg.ocr_available())
    if not info["saliency_trained"]:
        logger.warning("SALIENCY IS THE CLASSICAL BASELINE, not a trained model. "
                       "Measurements are real but must not be shown to a client "
                       "as AI-predicted attention until VL_MODEL_DIR has a "
                       "checkpoint from the Step 8 bake-off.")

    return pl.VisionDeps(
        store=store,
        sample_frames=md.sample_frames,
        predict_saliency=sal.predict,
        detect_regions=reg.detect,
        measure_frames=ms.measure_frames,
        summarise=ms.summarise,
        saliency_info=sal.describe,
        render_heatmap=hm.render,
        make_thumbnail=hm.thumbnail,
        upload=hm.upload,
        fetch_bytes=_fetch_bytes,
        decode_image=_decode_image,
        transcribe=_transcribe_if_configured(),
        llm_client=_llm_client(),
        llm_model=GEMINI_MODEL,
    )


def _transcribe_if_configured():
    """Deepgram, or None. None is a supported state, not a broken one: a report
    without a transcript still carries all six scores."""
    from . import transcript as tx
    if not tx.is_configured():
        logger.warning("DEEPGRAM_API_KEY is not set - creatives will be analysed "
                       "without a transcript.")
        return None
    return tx.transcribe_audio


def _llm_client():
    """Gemini, or None. Also a supported state: without it five metrics score
    fully, Clarity and Focus say why they are partial, and the report carries no
    written recommendations."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set - reports will carry "
                       "measurements and scores but no interpretation.")
        return None
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)


def _fetch_bytes(url: str) -> bytes:
    """Used only for the brand wordmark. The creative goes through media.py,
    which streams it to disk rather than into memory."""
    import httpx
    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    return response.content


def _decode_image(payload: bytes):
    import cv2
    import numpy as np
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB) if decoded is not None else None


async def reap_stale(store: AnalysisStore) -> int:
    """Report jobs whose worker died mid-flight.

    Usually a pm2 restart during a deploy. Without this a killed job sits in an
    active status forever and the caller polls it until they give up.
    """
    reaped = 0
    docs = await store.collection.find(
        {"status": {"$in": list(CLAIMED_STATUSES)}}).to_list(length=200)
    for doc in docs:
        if is_stale(doc):
            await store.mark_interrupted(doc["_id"])
            await store.clear_creative_url(doc["_id"])
            reaped += 1
            logger.warning("reaped stale job [analysis_id=%s status=%s]",
                           doc["_id"], doc.get("status"))
    return reaped


async def run_forever() -> int:
    if not MONGODB_URI:
        logger.error("MONGODB_URI is not set - the worker has no queue to read. Exiting.")
        return 1

    # Fail loudly here rather than once per job.
    fw.load_framework()
    fw.load_psychology()

    client = AsyncIOMotorClient(MONGODB_URI)
    database = client[MONGODB_DB]
    store = AnalysisStore(database["vision_lab_analyses"],
                          measurements_collection=database["vision_lab_measurements"])
    await store.ensure_indexes()

    deps = build_deps(store)
    in_flight: set[asyncio.Task] = set()
    last_reap = 0.0

    logger.info("vision-worker started [id=%s concurrency=%d enabled=%s]",
                WORKER_ID, MAX_CONCURRENT, VL_ENABLED)
    if not VL_ENABLED:
        logger.warning("VL_ENABLED is false - the worker will idle without claiming jobs. "
                       "Set VL_ENABLED=true in .env to turn Vision Lab on.")

    loop = asyncio.get_running_loop()
    while not _stopping:
        now = loop.time()
        if now - last_reap > REAP_EVERY_SECONDS:
            try:
                await reap_stale(store)
            except Exception:  # noqa: BLE001 - reaping must never kill the worker
                logger.exception("stale-job reaping failed")
            last_reap = now

        if not VL_ENABLED or len(in_flight) >= MAX_CONCURRENT:
            await asyncio.sleep(IDLE_SLEEP_SECONDS)
            in_flight = {task for task in in_flight if not task.done()}
            continue

        try:
            doc = await store.claim_next(WORKER_ID)
        except Exception:  # noqa: BLE001
            logger.exception("could not claim a job - retrying")
            await asyncio.sleep(IDLE_SLEEP_SECONDS)
            continue

        if not doc:
            await asyncio.sleep(IDLE_SLEEP_SECONDS)
            continue

        logger.info("claimed [analysis_id=%s creative_id=%s]",
                    doc["_id"], doc.get("creative_id"))
        task = asyncio.create_task(_guarded(doc, deps))
        in_flight.add(task)
        in_flight = {t for t in in_flight if not t.done()}

    if in_flight:
        logger.info("waiting for %d job(s) to finish", len(in_flight))
        await asyncio.gather(*in_flight, return_exceptions=True)

    client.close()
    logger.info("vision-worker stopped cleanly")
    return 0


async def _guarded(doc: dict, deps: pl.VisionDeps) -> None:
    """Run one job. The pipeline already persists every expected failure with a
    reason; this only catches the truly unexpected, because a task that dies
    silently would leave the row in an active status forever."""
    analysis_id = doc["_id"]
    try:
        await pl.run_analysis(doc, deps)
    except Exception:  # noqa: BLE001
        logger.exception("vision lab job crashed [analysis_id=%s]", analysis_id)
        try:
            await deps.store.fail(analysis_id, reason=ANALYSIS_PROVIDER_ERROR,
                                  message="The analysis did not complete.")
            await deps.store.clear_creative_url(analysis_id)
        except Exception:  # noqa: BLE001
            logger.exception("could not record failure [analysis_id=%s]", analysis_id)


def main() -> int:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        return asyncio.run(run_forever())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
