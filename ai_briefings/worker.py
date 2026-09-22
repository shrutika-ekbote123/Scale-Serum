"""
The briefing-worker process. pm2 entrypoint.

    pm2 start ecosystem.config.js --only briefing-worker     (once, then pm2 save)
    python -m ai_briefings.worker                            (locally)
    python -m ai_briefings.worker --once                     (one pass, then exit)

WHAT IT DOES
    Every few minutes it walks the brands with recent data. For each brand
    whose local clock has passed the generation hour (briefing_config.json,
    generation.local_hour), it generates YESTERDAY's briefings - once. A
    completed run for that brand and day is the "already done" marker; failed
    attempts are retried up to generation.max_attempts_per_day.

WHY A SEPARATE PROCESS
    The API process answers onboarding, Script Lab and every other request.
    A scheduler inside it would run once per uvicorn worker and die with each
    deploy mid-run. One small process, restarted by deploy.sh alongside the
    Vision Lab worker, is simpler to reason about.

THIS PROCESS OWNS ITS OWN CLIENTS
    Like vision_lab/worker.py, it opens its own Mongo and Gemini clients and
    injects them into the service - the package itself opens none.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from . import data as _data  # noqa: E402
from . import timezones as tzs  # noqa: E402
from .config import load_config  # noqa: E402
from .loaders import analyses_loader, brand_brain_loader, usage_pricer  # noqa: E402
from .service import BriefingDeps, generate_day  # noqa: E402
from .store import BriefingStore  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ai_briefings.worker")

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB", "scaleserum")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
INTERVAL_SECONDS = float(os.environ.get("BRIEFING_WORKER_INTERVAL_SECONDS", 300))
# A run still "running" after this long died with its process (a deploy, a crash).
STUCK_AFTER = timedelta(minutes=float(os.environ.get("BRIEFING_STUCK_MINUTES", 30)))

_stopping = False


def _request_stop(signum, _frame) -> None:
    global _stopping
    _stopping = True
    logger.info("signal %s received - finishing the current brand, then exiting", signum)


def _as_utc(value):
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


async def due(store: BriefingStore, brand: dict, now: datetime, cfg: dict):
    """The local day to generate for this brand now, or None."""
    zone = tzs.resolve(brand.get("timezone"))
    if tzs.local_now(now, zone).hour < cfg["generation"]["local_hour"]:
        return None
    day = tzs.yesterday(now, zone)
    runs = await store.runs_for(brand["id"], day.isoformat())
    if any(r["status"] == "completed" for r in runs):
        return None
    live = [r for r in runs if r["status"] in ("running", "queued")
            and now - (_as_utc(r.get("started_at") or r.get("queued_at")) or now) < STUCK_AFTER]
    if live:
        return None
    attempts = [r for r in runs if r.get("trigger") == "schedule"]
    if len(attempts) >= cfg["generation"]["max_attempts_per_day"]:
        return None
    return day


async def run_pass(deps: BriefingDeps) -> int:
    cfg = load_config()
    now = deps.now()
    try:
        brand_ids = await asyncio.to_thread(_data.active_brand_ids, now)
    except _data.DataUnavailable as err:
        logger.warning("could not list brands: %s", err)
        return 0
    generated = 0
    for brand_id in brand_ids:
        if _stopping:
            break
        try:
            brand = await asyncio.to_thread(_data.load_brand, brand_id)
            if not brand:
                continue
            day = await due(deps.store, brand, now, cfg)
            if day is None:
                continue
            logger.info("generating briefings [brand_id=%s date=%s]", brand_id, day)
            result = await generate_day(brand_id, deps, day=day, trigger="schedule")
            generated += result["status"] == "completed"
        except Exception:  # noqa: BLE001 - one brand must never stop the others
            logger.exception("briefing generation crashed [brand_id=%s]", brand_id)
    return generated


def build_deps() -> BriefingDeps:
    if not MONGODB_URI:
        raise SystemExit("MONGODB_URI is not set - the briefing worker has nowhere to store briefings.")
    db = AsyncIOMotorClient(MONGODB_URI)[MONGODB_DB]
    llm = None
    if GEMINI_API_KEY:
        from google import genai
        llm = genai.Client(api_key=GEMINI_API_KEY)
    else:
        logger.warning("GEMINI_API_KEY is not set - briefings will use template wording")
    return BriefingDeps(
        run_sync=asyncio.to_thread,
        store=BriefingStore(db["ai_briefings"], db["ai_briefing_runs"]),
        llm_client=llm, llm_model=GEMINI_MODEL,
        load_analyses=analyses_loader(db["sales_call_analyses"]),
        load_brand_brain=brand_brain_loader(db["brand_brains"]),
        price_usage=usage_pricer(GEMINI_MODEL),
    )


async def main(once: bool) -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    deps = build_deps()
    await deps.store.ensure_indexes()
    logger.info("briefing worker started [interval=%ss local_hour=%s]", INTERVAL_SECONDS,
                load_config()["generation"]["local_hour"])
    try:
        while not _stopping:
            count = await run_pass(deps)
            if count:
                logger.info("pass complete - %d brand(s) briefed", count)
            if once:
                break
            slept = 0.0
            while slept < INTERVAL_SECONDS and not _stopping:
                await asyncio.sleep(1)
                slept += 1
    finally:
        # Close the scrumdb pool while the interpreter is still whole; left to
        # the garbage collector it tries to join its threads during shutdown.
        from purchase_probability_model import inference
        if inference._POOL is not None:
            await asyncio.to_thread(inference._POOL.close)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate daily AI briefings.")
    parser.add_argument("--once", action="store_true", help="run one pass and exit")
    asyncio.run(main(parser.parse_args().once))
