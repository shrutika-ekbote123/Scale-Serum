"""
ScaleSerum - Brand Brain "ideal customer" persona rewriter (FastAPI).

One job: take the user's rough Q02 draft plus all the onboarding context, send it
to Google Gemini, and return a cleaned-up, optimized customer persona.

Run it:
    pip install -r requirements.txt
    copy .env.example .env   (then paste your GEMINI_API_KEY into .env)
    python app.py

Interactive docs (test the endpoint in your browser):
    http://localhost:3001/docs        <- Swagger UI

Your React frontend POSTs to:
    http://localhost:3001/api/brand-brain/rewrite-persona
"""

import os
import json
import functools
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import (BackgroundTasks, FastAPI, File, Form, HTTPException, Depends,
                     Query, Security, UploadFile)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorClient

from prompts import (
    PERSONA_SYSTEM_INSTRUCTION,
    FUNNEL_SYSTEM_INSTRUCTION,
    GAP_SYSTEM_INSTRUCTION,
    SCRIPT_SYSTEM_INSTRUCTION,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()  # reads GEMINI_API_KEY / GEMINI_MODEL / PORT from the .env file

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
PORT = int(os.environ.get("PORT", "3001"))
MAX_DRAFT = 4000  # guard against absurdly large input
MAX_SCRIPT = 8000  # ad scripts can be longer than a persona draft

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Copy .env.example to .env and set it.")

# One reusable Gemini client for the whole app. `.aio` gives us the async client.
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# MongoDB (Atlas) — stores the Brand Brain, keyed by a unique brand_brain_id.
# OPTIONAL: if MONGODB_URI is unset the app still runs; only the Brand Brain
# store/load endpoints are disabled (they return 503). motor = async driver, so
# DB calls don't block the FastAPI event loop.
# ---------------------------------------------------------------------------
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB", "scaleserum")

if MONGODB_URI:
    mongo_client = AsyncIOMotorClient(MONGODB_URI)
    brand_brains = mongo_client[MONGODB_DB]["brand_brains"]
    # Sales Call Analyzer: one document per analysis. The raw provider response
    # goes in its own collection with a TTL and is off unless
    # SCA_STORE_RAW_TRANSCRIPT is set - see sales_call_analyzer/store.py.
    sales_call_analyses = mongo_client[MONGODB_DB]["sales_call_analyses"]
    sales_call_transcripts_raw = mongo_client[MONGODB_DB]["sales_call_transcripts_raw"]
    # Vision Lab: one document per analysis, plus the per-frame measurement
    # record in its own collection with a TTL - it is an order of magnitude
    # larger than the report and only /rescore reads it.
    vision_lab_analyses = mongo_client[MONGODB_DB]["vision_lab_analyses"]
    vision_lab_measurements = mongo_client[MONGODB_DB]["vision_lab_measurements"]
else:
    mongo_client = None
    brand_brains = None
    sales_call_analyses = None
    sales_call_transcripts_raw = None
    vision_lab_analyses = None
    vision_lab_measurements = None


def _require_mongo():
    if brand_brains is None:
        raise HTTPException(
            status_code=503,
            detail="Brand Brain storage is not configured. Set MONGODB_URI in the environment.",
        )


# ---------------------------------------------------------------------------
# Logging. The rest of this file predates it and still uses print(); new code
# uses the logger. Level is env-driven so a server can be turned up without a
# code change. NEVER log an API key, a signed URL, transcript text or customer
# contact details - ids, counts and timings only.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("scaleserum")


# ---------------------------------------------------------------------------
# API-key auth: every /api/* endpoint requires the "X-API-Key" request header.
# Set API_KEY in .env to the secret you choose. If API_KEY is unset the API runs
# OPEN (handy for local dev) and logs a warning at startup. /health stays open.
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

if not API_KEY:
    print("WARNING: API_KEY is not set - the API is UNSECURED. Set API_KEY in .env to require a key.")


async def require_api_key(provided: Optional[str] = Security(api_key_header)):
    """Gate for all /api/* endpoints. Send the key in the 'X-API-Key' header."""
    if not API_KEY:
        return  # no key configured -> open (dev). Set API_KEY to enforce.
    if not provided or provided != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Send it in the 'X-API-Key' header.",
        )


# ---------------------------------------------------------------------------
# Sales Call Analyzer (transcription + diarization + framework evaluation).
#
# Imported defensively, like purchase_probability_model: a problem in this
# package must not stop onboarding and Script Lab from booting. If the import
# fails the endpoints report `analyzer_unavailable` instead of 500ing.
#
# Deepgram is configured here but NOT required at boot. GEMINI_API_KEY raises at
# import above because nothing works without it; Deepgram is one feature of
# several, so an unset key degrades that feature and says so.
# ---------------------------------------------------------------------------
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
DEEPGRAM_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-2")

SALES_CALL_ANALYZER_AVAILABLE = True
try:
    import sales_call_analyzer as sca
    from sales_call_analyzer import deepgram_client as _sca_deepgram
    from sales_call_analyzer import framework as _sca_framework
    from sales_call_analyzer import pipeline as _sca_pipeline
    from sales_call_analyzer import scoring as _sca_scoring
    from sales_call_analyzer import store as _sca_store
    from sales_call_analyzer.models import AnalyzeAccepted, AnalyzeRequest

    _sca_framework.load_framework()   # fail loudly here rather than per request
    _sca_framework.load_signals()
except Exception as _sca_import_error:  # pragma: no cover - import-time only
    SALES_CALL_ANALYZER_AVAILABLE = False
    _SCA_IMPORT_ERROR = repr(_sca_import_error)
    print(f"WARNING: sales_call_analyzer unavailable - {_SCA_IMPORT_ERROR}")

    # The routes below are defined unconditionally and annotate their body with
    # AnalyzeRequest. Without a stand-in, the failed import would raise NameError
    # at def time and take the WHOLE service down - onboarding and Script Lab
    # included - which is the exact opposite of what this guard is for.
    # With it, the endpoints still exist and return a clean 503.
    class AnalyzeRequest(BaseModel):  # type: ignore[no-redef]
        call_id: str = ""

    class AnalyzeAccepted(BaseModel):  # type: ignore[no-redef]
        analysis_id: str = ""
        call_id: str = ""
        status: str = "unavailable"

sales_call_store = None
if SALES_CALL_ANALYZER_AVAILABLE and sales_call_analyses is not None:
    sales_call_store = _sca_store.AnalysisStore(
        sales_call_analyses, raw_collection=sales_call_transcripts_raw)


# ---------------------------------------------------------------------------
# Vision Lab (AI-predicted attention analysis for ad creatives).
#
# Imported defensively for the same reason as the Sales Call Analyzer: a
# problem in this package - a missing model file, a malformed framework JSON -
# must not stop onboarding, Script Lab, purchase probability and sales calls
# from booting.
#
# TWO PROCESSES. This one accepts the job and serves the report; the actual
# frame decoding, saliency inference and OCR happen in vision_lab/worker.py,
# started by pm2 as "vision-worker". CPU-bound work does not belong in the
# process that also answers onboarding requests. The job document in MongoDB is
# the queue between them - there is no broker.
#
# VL_ENABLED gates the feature independently of whether it imported. It is
# false by default so that merging an unfinished Vision Lab onto main - which
# deploys itself - lands inert code rather than a half-built feature.
# ---------------------------------------------------------------------------
VISION_LAB_AVAILABLE = True
try:
    import vision_lab as vl
    from vision_lab import framework as _vl_framework
    from vision_lab import pipeline as _vl_pipeline
    from vision_lab import store as _vl_store
    from vision_lab.models import AnalyzeAccepted as VisionAccepted
    from vision_lab.models import AnalyzeRequest as VisionAnalyzeRequest
    from vision_lab.models import UploadAccepted as VisionUploadAccepted
    from vision_lab import uploads as _vl_uploads

    _vl_framework.load_framework()    # fail loudly here rather than per request
    _vl_framework.load_psychology()
except Exception as _vl_import_error:  # pragma: no cover - import-time only
    VISION_LAB_AVAILABLE = False
    _VL_IMPORT_ERROR = repr(_vl_import_error)
    print(f"WARNING: vision_lab unavailable - {_VL_IMPORT_ERROR}")

    # Same guard as the Sales Call Analyzer: the routes below annotate their
    # body with these types, so without a stand-in a failed import raises
    # NameError at def time and takes the WHOLE service down.
    class VisionAnalyzeRequest(BaseModel):  # type: ignore[no-redef]
        creative_id: str = ""

    class VisionAccepted(BaseModel):  # type: ignore[no-redef]
        analysis_id: str = ""
        creative_id: str = ""
        status: str = "unavailable"

    class VisionUploadAccepted(BaseModel):  # type: ignore[no-redef]
        uploaded: bool = False
        reason: Optional[str] = None

VL_ENABLED = os.environ.get("VL_ENABLED", "false").lower() == "true"

vision_lab_store = None
if VISION_LAB_AVAILABLE and vision_lab_analyses is not None:
    vision_lab_store = _vl_store.AnalysisStore(
        vision_lab_analyses, measurements_collection=vision_lab_measurements)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup: make sure the analysis indexes exist. Shutdown: close the shared
    Deepgram HTTP client. Both are no-ops when the feature is not configured."""
    if sales_call_store is not None:
        await sales_call_store.ensure_indexes()
    if vision_lab_store is not None:
        await vision_lab_store.ensure_indexes()
    yield
    if SALES_CALL_ANALYZER_AVAILABLE:
        await _sca_deepgram.aclose()


app = FastAPI(title="Brand Brain Persona Rewriter", version="1.0.0", lifespan=lifespan)

# Which frontend origins may call this API. Defaults to the local dev origins;
# override in production by setting ALLOWED_ORIGINS in .env to a comma-separated
# list (e.g. "https://app.scaleserum.com,https://staging.scaleserum.com").
DEFAULT_ORIGINS = [
    # local dev
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    # production frontend (add staging / other domains here or via ALLOWED_ORIGINS)
    "https://app.scaleserum.com",
]
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
] or DEFAULT_ORIGINS

# ---------------------------------------------------------------------------
# Vision Lab upload size guard - the route is POST /api/vision-lab/upload.
# ---------------------------------------------------------------------------
VISION_UPLOAD_PATH = "/api/vision-lab/upload"


class _VisionUploadSizeGuard:
    """Refuse an oversized upload BEFORE its body is read.

    By the time a route runs, FastAPI has already parsed the multipart body and
    spooled the whole file to disk - so a 5 GB upload checked inside the handler
    has already cost 5 GB of disk. This reads only the declared Content-Length
    and answers 413 without accepting the body. A client that lies, or sends no
    length, still meets the exact check on the real size inside the route; in
    production nginx's client_max_body_size is the outer wall.

    Pure ASGI rather than @app.middleware("http"): it touches ONE path and must
    not change how any other route behaves, and BaseHTTPMiddleware would wrap
    every request - the sales-call BackgroundTasks included - in machinery this
    does not need.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (VISION_LAB_AVAILABLE and scope.get("type") == "http"
                and scope.get("method") == "POST"
                and scope.get("path") == VISION_UPLOAD_PATH):
            declared = dict(scope.get("headers") or []).get(b"content-length", b"")
            limit = _vl_uploads.MAX_UPLOAD_BYTES + _vl_uploads.MULTIPART_OVERHEAD_BYTES
            if declared.isdigit() and int(declared) > limit:
                response = JSONResponse(status_code=413, content={
                    "uploaded": False, "reason": vl.CREATIVE_TOO_LARGE,
                    "message": (f"The upload is {int(declared) / 1048576:.1f} MB; the "
                                f"limit is {_vl_uploads.MAX_UPLOAD_BYTES / 1048576:.1f} MB."),
                    "upload": None, "analysis": None})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# Added BEFORE CORSMiddleware so it sits INSIDE it: a 413 refused here still
# carries the CORS headers, and a browser sees "too large" rather than an
# opaque CORS failure.
app.add_middleware(_VisionUploadSizeGuard)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response shapes (these also power the Swagger docs)
# ---------------------------------------------------------------------------
class BrandContext(BaseModel):
    businessType: Optional[str] = None       # Q01 selection
    industry: Optional[str] = None
    brandName: Optional[str] = None
    website: Optional[str] = None
    audienceShort: Optional[str] = None
    channels: Optional[List[str]] = None     # e.g. ["Meta", "Google"]
    adBudget: Optional[str] = None


class RewriteRequest(BaseModel):
    draft: str = Field(default="", description="The user's rough Q02 answer")
    context: BrandContext = Field(default_factory=BrandContext)


class RewriteResponse(BaseModel):
    optimized_persona: str
    raw: str
    fallback: bool = False


# ---- Q10: AI-suggested funnel (lead-to-sale journey) ----------------------
class BrandBrainAnswers(BaseModel):
    """All the Brand Brain answers we use to build the funnel. All optional so a
    partially-filled questionnaire still produces a sensible funnel."""
    businessType: Optional[str] = None           # business type
    idealCustomer: Optional[str] = None          # the optimized persona
    brandVoice: Optional[str] = None             # voice & tone
    language: Optional[str] = None               # content language
    trafficChannels: Optional[List[str]] = None  # channels they run
    salesCycle: Optional[str] = None             # e.g. "1-4 weeks"
    competitors: Optional[List[str]] = None      # top competitors
    marketingGoal: Optional[str] = None          # primary marketing goal
    journey: str = ""                            # lead-to-sale journey (may be empty)


class FunnelRequest(BaseModel):
    answers: BrandBrainAnswers = Field(default_factory=BrandBrainAnswers)
    context: BrandContext = Field(default_factory=BrandContext)  # business info etc.


class FunnelStage(BaseModel):
    stage: str            # short label, e.g. "Trial Pass Lead"
    description: str      # one line explaining what happens at this stage


class FunnelResponse(BaseModel):
    funnel: List[FunnelStage]
    optimized_journey: str
    fallback: bool = False


# ---- Step 11: gap analysis (find missing context, ask follow-ups) ---------
class GapRequest(BaseModel):
    """Everything collected so far. `exclude` lets the Reanalyze button ask for
    fresh gaps instead of repeating the ones already on screen."""
    answers: BrandBrainAnswers = Field(default_factory=BrandBrainAnswers)
    context: BrandContext = Field(default_factory=BrandContext)
    exclude: Optional[List[str]] = None   # gap titles already shown to the user
    max_gaps: int = 3                     # how many follow-up questions to return


class GapItem(BaseModel):
    id: str                 # stable id for the frontend (assigned server-side)
    title: str              # short name of the gap, e.g. "Customer objections"
    question: str           # the follow-up question to show the user
    why: str                # one line: why this matters for the downstream AI
    options: List[str]      # suggested tick-able options (the rectangular boxes)
    multi_select: bool = True


class GapResponse(BaseModel):
    gaps: List[GapItem]
    fallback: bool = False


# ---- Brand Brain storage (MongoDB) ----------------------------------------
class BrandBrainSaveRequest(BaseModel):
    """The full Brand Brain to persist: the answers + business context."""
    answers: BrandBrainAnswers = Field(default_factory=BrandBrainAnswers)
    context: BrandContext = Field(default_factory=BrandContext)


class BrandBrainSaveResponse(BaseModel):
    brand_brain_id: str   # give this to the main backend to store on its brand record


class BrandBrainDoc(BaseModel):
    brand_brain_id: str
    answers: BrandBrainAnswers
    context: BrandContext



# ---- Script Lab: test / review an ad script -------------------------------
class ScriptTestRequest(BaseModel):
    """An ad script plus the sales-team selections, reviewed against the brand's
    full Brand Brain context. Pass `brand_brain_id` to load the context from the
    DB; or send `answers` + `context` inline (used as a fallback if no id / not found)."""
    script: str = ""                             # the ad script to review
    marketingAngle: Optional[str] = None         # e.g. "Original", "Authority"
    funnelStage: Optional[str] = None            # e.g. "Cold, Top of Funnel"
    adSource: Optional[str] = None               # e.g. "meta"
    region: Optional[str] = None
    adName: Optional[str] = None                 # metadata, echoed for reference
    adNumber: Optional[str] = None
    brand_brain_id: Optional[str] = None         # preferred: load context from Mongo by this id
    answers: BrandBrainAnswers = Field(default_factory=BrandBrainAnswers)  # inline fallback
    context: BrandContext = Field(default_factory=BrandContext)           # inline fallback


class EmotionalAngle(BaseModel):
    label: str = ""      # e.g. "Story / narrative with aspirational underpinning"
    status: str = ""     # "ANGLE WORKS" | "ANGLE WEAK" | "ANGLE OFF"
    critique: str = ""


class DimensionScores(BaseModel):
    attention: int = 0                   # each 0-100
    resonance: int = 0
    conversion: int = 0
    creative: int = 0
    marketing_angle_execution: int = 0   # how consistently the chosen angle is expressed


class ContextAlignment(BaseModel):
    """Did the script follow the brief? Each is "Strong" | "Moderate" | "Weak"."""
    brand_voice_fit: str = ""
    funnel_stage_fit: str = ""
    marketing_angle_fit: str = ""


class SectionScore(BaseModel):
    section: str         # "Hook", "Problem / Tension", ...
    score: int           # 0-10
    comment: str = ""


class Improvement(BaseModel):
    title: str
    why_it_matters: str = ""
    suggested_rewrite: str = ""
    metrics_impacted: str = ""


class ScriptTestResponse(BaseModel):
    overall_score: int                 # 0-100
    verdict: str                       # one-line summary
    verdict_band: str                  # banded rating label
    emotional_angle: EmotionalAngle
    context_alignment: ContextAlignment  # did it follow the brief?
    dimension_scores: DimensionScores
    section_breakdown: List[SectionScore]
    improvements: List[Improvement]
    fallback: bool = False


# System prompts live in prompts.py (imported at the top).

# The exact shape we force Gemini to return: {"optimized_persona": "..."}
RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={"optimized_persona": types.Schema(type=types.Type.STRING)},
    required=["optimized_persona"],
)


def build_context_block(ctx: BrandContext) -> str:
    """Turn the onboarding fields into a readable list, skipping empty ones."""
    channels = ctx.channels or []
    channels = ", ".join(str(c) for c in channels)

    rows = [
        ("Business type (Q01)", ctx.businessType),
        ("Industry", ctx.industry),
        ("Brand / sub-account", ctx.brandName),
        ("Website", ctx.website),
        ("Audience (short)", ctx.audienceShort),
        ("Traffic channels", channels),
        ("Monthly ad budget", ctx.adBudget),
    ]
    lines = [f"- {label}: {value}" for label, value in rows if value and str(value).strip()]
    return "\n".join(lines) if lines else "(no extra context)"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness plus which optional subsystems are configured.

    Booleans and names only - never a key. The deploy health check reads `ok`
    and is unaffected by the extra fields.
    """
    return {
        "ok": True,
        "model": GEMINI_MODEL,
        "sales_call_analyzer": {
            "available": SALES_CALL_ANALYZER_AVAILABLE,
            "transcription": "configured" if DEEPGRAM_API_KEY else "not_configured",
            "storage": "configured" if sales_call_store is not None else "not_configured",
        },
        "vision_lab": await _vision_lab_health(),
    }


@app.post("/api/brand-brain/rewrite-persona", response_model=RewriteResponse,
          dependencies=[Depends(require_api_key)])
async def rewrite_persona(body: RewriteRequest):
    draft = (body.draft or "")[:MAX_DRAFT].strip()

    prompt = "\n".join(
        [
            "BUSINESS CONTEXT:",
            build_context_block(body.context),
            "",
            "CLIENT'S ROUGH DRAFT OF THE IDEAL CUSTOMER:",
            draft or "(empty)",
            "",
            "Rewrite the ideal customer persona following your rules.",
        ]
    )

    try:
        # Async call -> the server stays free to handle other requests while we
        # wait on Gemini.
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=PERSONA_SYSTEM_INSTRUCTION,
                temperature=0.4,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                # Don't let a slow model call hang the user's "Next" click (ms).
                http_options=types.HttpOptions(timeout=12_000),
            ),
        )

        parsed = json.loads(response.text)
        optimized = str(parsed.get("optimized_persona") or "").strip()

        return RewriteResponse(optimized_persona=optimized or draft, raw=draft)

    except Exception as err:  # noqa: BLE001 - we deliberately never block onboarding
        # Non-blocking contract: hand the raw draft back so the UI can proceed.
        print(f"rewrite-persona failed: {err}")
        return RewriteResponse(optimized_persona=draft, raw=draft, fallback=True)


# ---------------------------------------------------------------------------
# Q10: AI-suggested funnel (lead-to-sale journey)
# ---------------------------------------------------------------------------
FUNNEL_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "funnel": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "stage": types.Schema(type=types.Type.STRING),
                    "description": types.Schema(type=types.Type.STRING),
                },
                required=["stage", "description"],
            ),
        ),
        "optimized_journey": types.Schema(type=types.Type.STRING),
    },
    required=["funnel", "optimized_journey"],
)


def build_answers_block(a: BrandBrainAnswers) -> str:
    """Flatten the Brand Brain answers into a readable list, skipping empties."""
    channels = ", ".join(str(c) for c in (a.trafficChannels or []))
    competitors = ", ".join(str(c) for c in (a.competitors or []))
    rows = [
        ("Business type", a.businessType),
        ("Ideal customer", a.idealCustomer),
        ("Brand voice", a.brandVoice),
        ("Content language", a.language),
        ("Traffic channels", channels),
        ("Sales cycle", a.salesCycle),
        ("Competitors", competitors),
        ("Primary marketing goal", a.marketingGoal),
    ]
    lines = [f"- {label}: {value}" for label, value in rows if value and str(value).strip()]
    return "\n".join(lines) if lines else "(no answers provided)"


@app.post("/api/brand-brain/suggest-funnel", response_model=FunnelResponse,
          dependencies=[Depends(require_api_key)])
async def suggest_funnel(body: FunnelRequest):
    journey = (body.answers.journey or "")[:MAX_DRAFT].strip()

    prompt = "\n".join(
        [
            "BUSINESS CONTEXT:",
            build_context_block(body.context),
            "",
            "BRAND BRAIN ANSWERS:",
            build_answers_block(body.answers),
            "",
            "CLIENT'S DESCRIBED LEAD-TO-SALE JOURNEY (Q10):",
            journey or "(empty)",
            "",
            "Produce the ordered funnel and the cleaned-up journey following your rules.",
        ]
    )

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=FUNNEL_SYSTEM_INSTRUCTION,
                temperature=0.4,
                response_mime_type="application/json",
                response_schema=FUNNEL_RESPONSE_SCHEMA,
                http_options=types.HttpOptions(timeout=12_000),
            ),
        )

        parsed = json.loads(response.text)
        stages = [
            FunnelStage(stage=str(s.get("stage", "")).strip(),
                        description=str(s.get("description", "")).strip())
            for s in (parsed.get("funnel") or [])
            if str(s.get("stage", "")).strip()
        ]
        optimized_journey = str(parsed.get("optimized_journey") or "").strip()

        # If the model returned nothing usable, fall back rather than error.
        if not stages:
            raise ValueError("model returned no funnel stages")

        return FunnelResponse(
            funnel=stages,
            optimized_journey=optimized_journey or journey,
        )

    except Exception as err:  # noqa: BLE001 - never block onboarding
        # Non-blocking fallback: a generic starter funnel so the UI still has chips.
        print(f"suggest-funnel failed: {err}")
        fallback_funnel = [
            FunnelStage(stage="Ad Click", description="Prospect clicks an ad or link."),
            FunnelStage(stage="Lead", description="Prospect submits their details."),
            FunnelStage(stage="Qualified", description="Lead is contacted and qualified."),
            FunnelStage(stage="Purchase", description="Lead converts into a paying customer."),
            FunnelStage(stage="Retention", description="Customer is retained and re-engaged."),
        ]
        return FunnelResponse(funnel=fallback_funnel, optimized_journey=journey, fallback=True)


# ---------------------------------------------------------------------------
# Step 11: gap analysis (analyze all answers -> find missing context)
# ---------------------------------------------------------------------------
GAP_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "gaps": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "title": types.Schema(type=types.Type.STRING),
                    "question": types.Schema(type=types.Type.STRING),
                    "why": types.Schema(type=types.Type.STRING),
                    "options": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                    ),
                },
                required=["title", "question", "why", "options"],
            ),
        ),
    },
    required=["gaps"],
)


@app.post("/api/brand-brain/analyze-gaps", response_model=GapResponse,
          dependencies=[Depends(require_api_key)])
async def analyze_gaps(body: GapRequest):
    max_gaps = max(1, min(int(body.max_gaps or 3), 6))
    already_shown = ", ".join(body.exclude or []) or "(none)"

    prompt = "\n".join(
        [
            "BUSINESS CONTEXT:",
            build_context_block(body.context),
            "",
            "ALL BRAND BRAIN ANSWERS:",
            build_answers_block(body.answers),
            "",
            f"Journey (Q10): {body.answers.journey or '(empty)'}",
            "",
            f"Return AT MOST {max_gaps} gaps, ordered by importance.",
            f"Already shown to the user (do NOT repeat these): {already_shown}",
        ]
    )

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GAP_SYSTEM_INSTRUCTION,
                # Slightly higher so "Reanalyze" surfaces different angles.
                temperature=0.7,
                response_mime_type="application/json",
                response_schema=GAP_RESPONSE_SCHEMA,
                http_options=types.HttpOptions(timeout=12_000),
            ),
        )

        parsed = json.loads(response.text)
        gaps: List[GapItem] = []
        for i, g in enumerate(parsed.get("gaps") or [], start=1):
            title = str(g.get("title", "")).strip()
            options = [str(o).strip() for o in (g.get("options") or []) if str(o).strip()]
            if not title or not options:
                continue
            gaps.append(GapItem(
                id=f"gap_{i}",
                title=title,
                question=str(g.get("question", "")).strip(),
                why=str(g.get("why", "")).strip(),
                options=options,
            ))
            if len(gaps) >= max_gaps:
                break

        if not gaps:
            raise ValueError("model returned no usable gaps")

        return GapResponse(gaps=gaps)

    except Exception as err:  # noqa: BLE001 - never block onboarding
        # Non-blocking fallback: a couple of broadly-useful gaps so the step still
        # renders. These are generic on purpose (used only when the AI call fails).
        print(f"analyze-gaps failed: {err}")
        fallback_gaps = [
            GapItem(
                id="gap_1",
                title="Customer objections",
                question="What are the main objections that stop people from buying?",
                why="Ad Review & Script Lab need known objections to write rebuttals.",
                options=["Price too high", "No time", "Tried before - didn't work",
                         "Skeptical of results", "Needs partner approval"],
            ),
            GapItem(
                id="gap_2",
                title="Proof & credibility",
                question="What proof do you have that you can show in ads?",
                why="Creative angles rely on proof (testimonials, data, guarantees).",
                options=["Client testimonials", "Before/after results", "Case studies",
                         "Money-back guarantee", "Awards / certifications"],
            ),
        ]
        return GapResponse(gaps=fallback_gaps[:max_gaps], fallback=True)


# ---------------------------------------------------------------------------
# Brand Brain storage: persist the Brand Brain and hand back a brand_brain_id
# (the main backend stores step 1-3 itself; the Brand Brain lives here).
# ---------------------------------------------------------------------------
@app.post("/api/brand-brain/save", response_model=BrandBrainSaveResponse,
          dependencies=[Depends(require_api_key)])
async def _brand_brain(body: BrandBrainSaveRequest):
    """Called at 'Finish & Train AI'. Stores the Brand Brain, returns a new
    unique brand_brain_id for the main backend to keep on its brand record."""
    _require_mongo()
    brand_brain_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    await brand_brains.insert_one(
        {
            "_id": brand_brain_id,
            "answers": body.answers.model_dump(),
            "context": body.context.model_dump(),
            "created_at": now,
            "updated_at": now,
        }
    )
    return BrandBrainSaveResponse(brand_brain_id=brand_brain_id)


@app.put("/api/brand-brain/{brand_brain_id}", response_model=BrandBrainSaveResponse,
         dependencies=[Depends(require_api_key)])
async def update_brand_brain(brand_brain_id: str, body: BrandBrainSaveRequest):
    """Update (or create) the Brand Brain for an existing id - e.g. if the user
    edits the brand later."""
    _require_mongo()
    now = datetime.now(timezone.utc)
    await brand_brains.update_one(
        {"_id": brand_brain_id},
        {
            "$set": {
                "answers": body.answers.model_dump(),
                "context": body.context.model_dump(),
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return BrandBrainSaveResponse(brand_brain_id=brand_brain_id)


@app.get("/api/brand-brain/{brand_brain_id}", response_model=BrandBrainDoc,
         dependencies=[Depends(require_api_key)])
async def get_brand_brain(brand_brain_id: str):
    """Fetch a stored Brand Brain (handy for verifying / debugging)."""
    _require_mongo()
    doc = await brand_brains.find_one({"_id": brand_brain_id})
    if not doc:
        raise HTTPException(status_code=404, detail="brand_brain_id not found")
    return BrandBrainDoc(
        brand_brain_id=brand_brain_id,
        answers=BrandBrainAnswers(**(doc.get("answers") or {})),
        context=BrandContext(**(doc.get("context") or {})),
    )


# ---------------------------------------------------------------------------
# Script Lab: review an ad script against the brand's Brand Brain context
# ---------------------------------------------------------------------------
SCRIPT_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "overall_score": types.Schema(type=types.Type.INTEGER),
        "verdict": types.Schema(type=types.Type.STRING),
        "verdict_band": types.Schema(type=types.Type.STRING),
        "emotional_angle": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "label": types.Schema(type=types.Type.STRING),
                "status": types.Schema(type=types.Type.STRING),
                "critique": types.Schema(type=types.Type.STRING),
            },
            required=["label", "status", "critique"],
        ),
        "context_alignment": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "brand_voice_fit": types.Schema(type=types.Type.STRING),
                "funnel_stage_fit": types.Schema(type=types.Type.STRING),
                "marketing_angle_fit": types.Schema(type=types.Type.STRING),
            },
            required=["brand_voice_fit", "funnel_stage_fit", "marketing_angle_fit"],
        ),
        "dimension_scores": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "attention": types.Schema(type=types.Type.INTEGER),
                "resonance": types.Schema(type=types.Type.INTEGER),
                "conversion": types.Schema(type=types.Type.INTEGER),
                "creative": types.Schema(type=types.Type.INTEGER),
                "marketing_angle_execution": types.Schema(type=types.Type.INTEGER),
            },
            required=["attention", "resonance", "conversion", "creative",
                      "marketing_angle_execution"],
        ),
        "section_breakdown": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "section": types.Schema(type=types.Type.STRING),
                    "score": types.Schema(type=types.Type.INTEGER),
                    "comment": types.Schema(type=types.Type.STRING),
                },
                required=["section", "score", "comment"],
            ),
        ),
        "improvements": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "title": types.Schema(type=types.Type.STRING),
                    "why_it_matters": types.Schema(type=types.Type.STRING),
                    "suggested_rewrite": types.Schema(type=types.Type.STRING),
                    "metrics_impacted": types.Schema(type=types.Type.STRING),
                },
                required=["title", "why_it_matters", "suggested_rewrite", "metrics_impacted"],
            ),
        ),
    },
    required=[
        "overall_score", "verdict", "verdict_band", "emotional_angle",
        "context_alignment", "dimension_scores", "section_breakdown", "improvements",
    ],
)


def _clamp(value, lo, hi, default=0):
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default


def _band_for(score: int) -> str:
    if score >= 90:
        return "No changes needed"
    if score >= 70:
        return "Minor tweaks only"
    if score >= 50:
        return "Needs work before going live"
    return "Rewrite required"


def build_script_meta_block(body: "ScriptTestRequest") -> str:
    rows = [
        ("Marketing angle", body.marketingAngle),
        ("Funnel stage", body.funnelStage),
        ("Ad source", body.adSource),
        ("Region", body.region),
        ("Ad name", body.adName),
        ("Ad number", body.adNumber),
    ]
    lines = [f"- {label}: {value}" for label, value in rows if value and str(value).strip()]
    return "\n".join(lines) if lines else "(no selections provided)"


@app.post("/api/script-lab/test-script", response_model=ScriptTestResponse,
          dependencies=[Depends(require_api_key)])
async def test_script(body: ScriptTestRequest):
    script = (body.script or "")[:MAX_SCRIPT].strip()

    # Resolve the brand context: prefer the stored Brand Brain (by id); otherwise
    # use whatever was sent inline in the request.
    answers = body.answers
    context = body.context
    if body.brand_brain_id and brand_brains is not None:
        doc = await brand_brains.find_one({"_id": body.brand_brain_id})
        if doc:
            answers = BrandBrainAnswers(**(doc.get("answers") or {}))
            context = BrandContext(**(doc.get("context") or {}))

    prompt = "\n".join(
        [
            "BUSINESS CONTEXT:",
            build_context_block(context),
            "",
            "BRAND BRAIN (what this brand stands for):",
            build_answers_block(answers),
            "",
            "SALES-TEAM SELECTIONS FOR THIS TEST:",
            build_script_meta_block(body),
            "",
            "AD SCRIPT TO REVIEW:",
            script or "(empty)",
            "",
            "Review the script and return the structured critique following your rules.",
        ]
    )

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SCRIPT_SYSTEM_INSTRUCTION,
                temperature=0.4,
                response_mime_type="application/json",
                response_schema=SCRIPT_RESPONSE_SCHEMA,
                # This critique is large + reasoned, so it needs longer than the
                # onboarding endpoints. "Test Script" is a deliberate click with a
                # loading state, so a longer wait is acceptable.
                http_options=types.HttpOptions(timeout=45_000),
            ),
        )

        parsed = json.loads(response.text)

        sections = [
            SectionScore(
                section=str(s.get("section", "")).strip(),
                score=_clamp(s.get("score"), 0, 10),
                comment=str(s.get("comment", "")).strip(),
            )
            for s in (parsed.get("section_breakdown") or [])
            if str(s.get("section", "")).strip()
        ]
        if not sections:
            raise ValueError("model returned no section breakdown")

        overall = _clamp(parsed.get("overall_score"), 0, 100, default=50)
        ea = parsed.get("emotional_angle") or {}
        ca = parsed.get("context_alignment") or {}
        ds = parsed.get("dimension_scores") or {}
        improvements = [
            Improvement(
                title=str(i.get("title", "")).strip(),
                why_it_matters=str(i.get("why_it_matters", "")).strip(),
                suggested_rewrite=str(i.get("suggested_rewrite", "")).strip(),
                metrics_impacted=str(i.get("metrics_impacted", "")).strip(),
            )
            for i in (parsed.get("improvements") or [])
            if str(i.get("title", "")).strip()
        ]

        return ScriptTestResponse(
            overall_score=overall,
            verdict=str(parsed.get("verdict") or "").strip(),
            # Trust the band only if it's one of ours; else derive from the score.
            verdict_band=str(parsed.get("verdict_band") or "").strip() or _band_for(overall),
            emotional_angle=EmotionalAngle(
                label=str(ea.get("label", "")).strip(),
                status=str(ea.get("status", "")).strip(),
                critique=str(ea.get("critique", "")).strip(),
            ),
            context_alignment=ContextAlignment(
                brand_voice_fit=str(ca.get("brand_voice_fit", "")).strip(),
                funnel_stage_fit=str(ca.get("funnel_stage_fit", "")).strip(),
                marketing_angle_fit=str(ca.get("marketing_angle_fit", "")).strip(),
            ),
            dimension_scores=DimensionScores(
                attention=_clamp(ds.get("attention"), 0, 100),
                resonance=_clamp(ds.get("resonance"), 0, 100),
                conversion=_clamp(ds.get("conversion"), 0, 100),
                creative=_clamp(ds.get("creative"), 0, 100),
                marketing_angle_execution=_clamp(ds.get("marketing_angle_execution"), 0, 100),
            ),
            section_breakdown=sections,
            improvements=improvements,
        )

    except Exception as err:  # noqa: BLE001 - never block the sales team
        # Non-blocking fallback: a neutral scorecard so the UI still renders.
        print(f"test-script failed: {err}")
        neutral_sections = [
            SectionScore(section=name, score=5, comment="Couldn't analyze automatically - review manually.")
            for name in [
                "Hook", "Problem / Tension", "Solution / Offer",
                "Social Proof / Credibility", "Call to Action", "Pacing & Tightness",
            ]
        ]
        return ScriptTestResponse(
            overall_score=50,
            verdict="Couldn't complete the AI review - try again.",
            verdict_band="Needs work before going live",
            emotional_angle=EmotionalAngle(
                label=body.marketingAngle or "",
                status="",
                critique="The angle could not be assessed automatically.",
            ),
            context_alignment=ContextAlignment(),
            dimension_scores=DimensionScores(
                attention=50, resonance=50, conversion=50, creative=50,
                marketing_angle_execution=50,
            ),
            section_breakdown=neutral_sections,
            improvements=[
                Improvement(
                    title="Re-run the analysis",
                    why_it_matters="The automated review did not complete for this script.",
                    suggested_rewrite="Click Regenerate, or review the script manually against the brand voice.",
                    metrics_impacted="",
                )
            ],
            fallback=True,
        )


# ---------------------------------------------------------------------------
# Purchase Probability (baseline MVP + signal layers)
#
# Scores a lead with the frozen baseline model in purchase_probability_model/.
# The model reads PostgreSQL READ-ONLY; this service never writes to it.
#
# THREE things are returned and they are NOT the same thing:
#   * purchase_probability - the real calibrated model output, as a percentage.
#     Base rate is ~1.1%, so genuine values sit roughly in 0.3%-4%. It is never
#     rescaled to look bigger, and the layers below never touch it.
#   * engagement / brand_brain - what the lead has DONE (admissible clicks and
#     journey timeline) and how well it fits what the brand said it wants (the
#     Brand Brain document from MongoDB). Both are bounded heuristic priors.
#   * lead_priority - the ranking signal those layers produce. Sort your list on
#     this; quote purchase_probability as the probability. It reports
#     `calibrated: false` about itself, because it is.
#
#   * lead_summary - the CRM lead card: score, temperature, realised revenue,
#     days to convert, and an estimated lifetime value. Read off the leads table,
#     so it is present even when the model cannot score the lead. None of it is a
#     model input: these columns are mutated at payment time and leak the outcome.
#     `lead_score` is the CRM's engagement score out of 100 and is NOT a
#     probability - showing it as one is the exact confusion this endpoint exists
#     to remove.
#
# Touchpoints in the response are DISPLAY history. Base-model inputs are reported
# under `model_features`, layer inputs under each layer. Never conflate them.
# ---------------------------------------------------------------------------
PURCHASE_PROBABILITY_AVAILABLE = True
try:
    from purchase_probability_model import (
        predict_for_lead as _pp_predict,
        resolve_brand_brain_ref as _pp_brand_ref,
    )
except Exception as _pp_import_error:  # pragma: no cover - import-time only
    PURCHASE_PROBABILITY_AVAILABLE = False
    _PP_IMPORT_ERROR = repr(_pp_import_error)
    print(f"WARNING: purchase_probability_model unavailable - {_PP_IMPORT_ERROR}")


def _pp_dead_layers(message: str) -> dict:
    """Layer and lead-card blocks for the import-failed path, matching the live
    response shape.

    Unlike the in-package unavailable paths, this one cannot fill the lead card:
    the inference package failed to import, so there is no database connection to
    read the lead row with. Everything is null and says why.
    """
    empty = {"available": False, "reason": "model_artefacts_unavailable",
             "message": message, "factors": [], "total_applied": 0.0,
             "total_raw": 0.0, "clamped": False, "bounds": None}
    return {
        "lead_summary": {
            "available": False, "reason": "model_artefacts_unavailable",
            "message": message,
            "lead_score": None, "lead_score_max": 100, "temperature": None,
            "status": None, "stage": None, "source": None, "created_at": None,
            "touchpoint_count": None, "total_revenue": None, "currency": None,
            "payment_count": 0, "converted": None, "converted_at": None,
            "days_to_convert": None,
            "lifetime_value": {
                "amount": None, "currency": None, "estimated": True,
                "available": False, "reason": "model_artefacts_unavailable",
                "basis": None, "expected_amount": None,
                "potential_amount": None, "potential_basis": None,
                "probability_used": None, "order_value_used": None,
                "order_value_basis": None, "order_count": None,
                "average_order_value": None, "median_order_value": None},
            "basis": "lead row could not be read",
        },
        "lead_score": None, "temperature": None, "total_revenue": None,
        "days_to_convert": None, "lifetime_value": None,
        "engagement": {**empty, "observed": None, "channel": None, "window": None,
                       "timeline": [], "sources": None,
                       "recency_half_life_hours": None},
        "brand_brain": {**empty, "brand_brain_id": None, "profile": None},
        "lead_priority": {"probability": None, "probability_percent": None,
                          "score": None, "percentile": None, "decile": None,
                          "priority": "Unavailable", "calibrated": False,
                          "layers_applied": [], "basis": "no layers applied",
                          "log_odds": None},
        "ranking_factors": [], "signal_version": None,
    }


async def _load_brand_brain(brand_brain_id: Optional[str]) -> Optional[dict]:
    """Fetch the Brand Brain document. Async so the Mongo round trip does not
    block the event loop, and so the inference package never needs a Mongo driver.

    A missing document is not an error: the brand-fit layer simply reports
    `no_brand_brain` and contributes nothing to the score.
    """
    if not brand_brain_id or brand_brains is None:
        return None
    try:
        return await brand_brains.find_one({"_id": brand_brain_id})
    except Exception:
        return None


@app.get("/api/purchase-probability/{lead_id}",
         dependencies=[Depends(require_api_key)])
async def purchase_probability(
    lead_id: str,
    brand_brain_id: Optional[str] = Query(
        None,
        description="Override the Brand Brain used for brand-fit context. Normally "
                    "omitted - it is resolved from the lead's brand automatically."),
):
    """Purchase probability, engagement, brand fit, ranking and the lead card.

    Follows the house convention: always HTTP 200. When the lead cannot be scored
    the response carries `fallback: true` and `availability.available: false`
    rather than a fabricated number. `null` and `0` mean different things here,
    and so do a layer that found nothing (`available: false`) and a layer that
    found something bad (a negative contribution).

    `lead_summary` (mirrored as `lead_score`, `temperature`, `total_revenue`,
    `days_to_convert` and `lifetime_value` at the top level) is the CRM lead card.
    It comes from the leads table, not the model, so it stays populated on an
    unscorable lead - only the probability goes null. `lead_score` is the CRM's
    0-100 engagement score; rendering it as a purchase probability is wrong, and
    `lifetime_value` is an estimate (probability x the brand median order), not
    money received - `total_revenue` is the money actually received.
    """
    if not PURCHASE_PROBABILITY_AVAILABLE:
        message = "Model artefacts are not available on this server."
        return {
            "lead_id": lead_id,
            "purchase_probability": None, "purchase_probability_percent": None,
            "probability": None, "percentile": None, "decile": None,
            "priority": "Unavailable", "top_factors": [], "why": None,
            "summary": {"available": False, "headline": None, "text": None,
                        "sentences": [], "positive": [], "negative": [],
                        "standing": None,
                        "counts": {"total": 0, "positive": 0, "negative": 0},
                        "note": None, "basis": None},
            "model_factors": [], "model_features": None,
            "touchpoint_count": 0, "touchpoints": [],
            "model": {"name": "purchase_probability", "version": "baseline_mvp",
                      "status": "baseline_mvp"},
            "availability": {"available": False,
                             "reason": "model_artefacts_unavailable",
                             "message": message},
            "fallback": True, "reason": "model_artefacts_unavailable",
            **_pp_dead_layers(message),
        }

    # Resolve which Brand Brain belongs to this lead's brand, then load it from
    # Mongo. Both steps are skipped when Mongo is not configured, so the endpoint
    # costs exactly what it used to on deployments without a Brand Brain store.
    ref: dict = {}
    brand_brain: Optional[dict] = None
    if brand_brains is not None:
        ref = await run_in_threadpool(_pp_brand_ref, lead_id)
        brand_brain = await _load_brand_brain(brand_brain_id or ref.get("brand_brain_id"))

    # Inference is synchronous (psycopg + sklearn); keep it off the event loop.
    result = await run_in_threadpool(
        functools.partial(_pp_predict, lead_id, brand_brain=brand_brain))

    # Say which brand we looked at even when it has no Brand Brain, so the UI can
    # explain the gap ("Brand X has not completed onboarding") instead of showing
    # an unexplained empty panel.
    block = result.get("brand_brain")
    if isinstance(block, dict):
        block["brand_id"] = ref.get("brand_id")
        block["brand_name"] = ref.get("brand_name")
        block["resolved_brand_brain_id"] = brand_brain_id or ref.get("brand_brain_id")
        block["brand_brain_store"] = (
            "configured" if brand_brains is not None else "not_configured")
    return result


# ---------------------------------------------------------------------------
# Sales Call Analyzer
#
# Transcribes and diarizes a recorded sales call (Deepgram), evaluates it against
# the six-stage sales framework (Gemini), verifies every claim against the
# transcript, and scores it DETERMINISTICALLY in Python. The model never
# produces a number - see sales_call_analyzer/scoring.py.
#
# ASYNCHRONOUS. Transcription plus analysis takes longer than a request should be
# held open, so POST returns an analysis_id immediately and the backend polls the
# GET. Processing runs in a FastAPI BackgroundTask bounded by a concurrency gate,
# because this is a single pm2 fork process shared with onboarding and Script Lab.
#
# Path style follows the rest of this file (/api/<feature>/...). Versioning is
# carried in the payload - framework_version, prompt_version, transcript_version -
# so the contract can evolve without a URL change. If a /v1 prefix is ever wanted
# it should be applied to every endpoint in this service, not just this one.
# ---------------------------------------------------------------------------
SALES_CALL_POLL_SECONDS = int(os.environ.get("SCA_POLL_INTERVAL_SECONDS", 5))


def _require_sales_call_analyzer():
    if not SALES_CALL_ANALYZER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="The Sales Call Analyzer is not available on this server.")
    if sales_call_store is None:
        raise HTTPException(
            status_code=503,
            detail=("Sales call analysis storage is not configured. "
                    "Set MONGODB_URI in the environment."))


async def _sca_resolve_brand_ref(lead_id: str) -> dict:
    """Which brand (and Brand Brain) a lead belongs to.

    Reuses the purchase-probability lookup rather than adding a second query. It
    is synchronous psycopg, so it goes to the threadpool like every other
    database call in this file.
    """
    if not PURCHASE_PROBABILITY_AVAILABLE or not lead_id:
        return {}
    return await run_in_threadpool(_pp_brand_ref, lead_id)


def _sales_call_deps():
    """Everything the pipeline needs from this process. The analyzer package owns
    no clients or connections of its own."""
    return _sca_pipeline.PipelineDeps(
        store=sales_call_store,
        llm_client=client,
        llm_model=GEMINI_MODEL,
        transcribe=_sca_deepgram.transcribe,
        load_brand_brain=_load_brand_brain,          # the existing helper, reused
        resolve_brand_ref=_sca_resolve_brand_ref,
        transcription_model=DEEPGRAM_MODEL,
    )


async def _sales_call_job(analysis_id, body, created_at):
    """Background job body. Every expected failure is already persisted with a
    reason by the pipeline; this only guards the truly unexpected, because a task
    that dies silently would leave the row in an active status forever."""
    try:
        await _sca_pipeline.run_analysis(analysis_id, body, _sales_call_deps(),
                                         created_at=created_at)
    except Exception:  # noqa: BLE001
        logger.exception("sales call job crashed [analysis_id=%s]", analysis_id)
        try:
            await sales_call_store.fail(
                analysis_id, reason=sca.ANALYSIS_PROVIDER_ERROR,
                message="The analysis did not complete.")
        except Exception:  # noqa: BLE001
            logger.exception("could not record failure [analysis_id=%s]", analysis_id)


def _poll_url(analysis_id: str) -> str:
    return f"/api/sales-calls/analysis/{analysis_id}"


@app.post("/api/sales-calls/analyze", dependencies=[Depends(require_api_key)])
async def analyze_sales_call(body: AnalyzeRequest, background: BackgroundTasks):
    """Submit a call for analysis. Returns immediately with an analysis_id.

    This is NOT the report - poll GET /api/sales-calls/analysis/{analysis_id}.

    Idempotent: an identical submission returns the existing analysis and costs
    nothing. Send options.force_reanalysis=true to override.
    """
    _require_sales_call_analyzer()

    cfg = _sca_framework.load_framework()
    fingerprint = _sca_store.compute_fingerprint(
        body,
        framework_version=cfg["framework_version"],
        prompt_version=_sca_pipeline.analyzer_mod.PROMPT_VERSION,
        llm_model=GEMINI_MODEL,
        transcription_model=DEEPGRAM_MODEL,
    )

    # Already analysed, or already running? Hand back what exists.
    existing = await sales_call_store.find_by_fingerprint(fingerprint)
    if _sca_store.reusable(existing, force=body.options.force_reanalysis):
        return AnalyzeAccepted(
            analysis_id=existing["_id"], call_id=existing["call_id"],
            status=existing["status"], created_at=existing.get("created_at"),
            idempotent_hit=True, poll_url=_poll_url(existing["_id"]),
            suggested_poll_interval_seconds=SALES_CALL_POLL_SECONDS)

    analysis_id = _sca_store.new_analysis_id()
    versions = {"framework_version": cfg["framework_version"],
                "signals_version": _sca_framework.load_signals()["signals_version"],
                "prompt_version": _sca_pipeline.analyzer_mod.PROMPT_VERSION,
                "llm_model": GEMINI_MODEL,
                "transcription_model": DEEPGRAM_MODEL}
    doc = await sales_call_store.create(analysis_id=analysis_id, request=body,
                                        fingerprint=fingerprint, versions=versions)

    # Nothing to analyse. Recorded as a failed analysis rather than a 4xx, so the
    # backend gets an auditable row and the same reason vocabulary as every other
    # failure - consistent with the always-200 contract used across this API.
    if not (_sca_pipeline.transcript_mod.has_audio(body.audio)
            or _sca_pipeline.transcript_mod.has_text(body.transcript)):
        await sales_call_store.fail(
            analysis_id, reason=sca.NO_AUDIO_OR_TRANSCRIPT,
            message=sca.REASON_TEXT[sca.NO_AUDIO_OR_TRANSCRIPT])
        return AnalyzeAccepted(
            analysis_id=analysis_id, call_id=body.call_id, status=sca.STATUS_FAILED,
            created_at=doc["created_at"], poll_url=_poll_url(analysis_id),
            availability=_sca_pipeline.unavailable(sca.NO_AUDIO_OR_TRANSCRIPT),
            reason=sca.NO_AUDIO_OR_TRANSCRIPT,
            message=sca.REASON_TEXT[sca.NO_AUDIO_OR_TRANSCRIPT])

    background.add_task(_sales_call_job, analysis_id, body, doc["created_at"])
    logger.info("sales call analysis queued [analysis_id=%s call_id=%s lead_id=%s]",
                analysis_id, body.call_id, body.lead_id)

    return AnalyzeAccepted(
        analysis_id=analysis_id, call_id=body.call_id, status=sca.STATUS_QUEUED,
        created_at=doc["created_at"], poll_url=_poll_url(analysis_id),
        suggested_poll_interval_seconds=SALES_CALL_POLL_SECONDS)


async def _sales_call_payload(doc: dict) -> dict:
    """Turn a stored job document into the polling response.

    A job whose heartbeat has stopped is reaped here: a pm2 restart mid-flight
    (that is, every deploy) would otherwise leave a row reporting "analyzing"
    forever. It becomes a stated failure the caller can retry.
    """
    if _sca_store.is_stale(doc):
        await sales_call_store.mark_interrupted(doc["_id"])
        doc = await sales_call_store.get(doc["_id"]) or doc

    status = doc.get("status")
    if status in sca.TERMINAL_STATUSES and doc.get("report"):
        report = dict(doc["report"])
        report["status"] = status
        return report

    return {
        "analysis_id": doc["_id"],
        "call_id": doc.get("call_id"),
        "lead_id": doc.get("lead_id"),
        "status": status,
        "availability": {
            "available": status != sca.STATUS_FAILED,
            "reason": doc.get("reason"),
            "message": doc.get("message"),
        },
        "scores": None,
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "attempts": doc.get("attempts", 0),
        "poll_url": _poll_url(doc["_id"]),
        "suggested_poll_interval_seconds": SALES_CALL_POLL_SECONDS,
        "fallback": status == sca.STATUS_FAILED,
    }


@app.get("/api/sales-calls/analysis/{analysis_id}",
         dependencies=[Depends(require_api_key)])
async def get_sales_call_analysis(analysis_id: str):
    """Poll for status, then read the finished report.

    While processing: a small envelope with `status` and `scores: null`.
    When complete: the full report. On failure: the same shape with
    `availability.available=false`, a stable `reason`, and `scores: null` - a
    failed analysis never invents a scorecard.
    """
    _require_sales_call_analyzer()
    doc = await sales_call_store.get(analysis_id)
    if not doc:
        raise HTTPException(status_code=404, detail="analysis_id not found")
    return await _sales_call_payload(doc)


@app.get("/api/sales-calls/analysis/by-call/{call_id}",
         dependencies=[Depends(require_api_key)])
async def get_latest_sales_call_analysis(call_id: str):
    """The most recent analysis for a call - for a backend that kept the call_id
    but not the analysis_id."""
    _require_sales_call_analyzer()
    doc = await sales_call_store.latest_for_call(call_id)
    if not doc:
        raise HTTPException(status_code=404, detail="no analysis exists for that call_id")
    return await _sales_call_payload(doc)


@app.post("/api/sales-calls/analysis/{analysis_id}/rescore",
          dependencies=[Depends(require_api_key)])
async def rescore_sales_call_analysis(analysis_id: str):
    """Recompute the scores from the stored ratings under the current framework.

    No Deepgram call, no Gemini call, no cost. This is the payoff for storing
    per-criterion ratings rather than only numbers: when management sets real
    weights or a real rating scale, historical calls can be brought onto the new
    configuration without re-analysing them.
    """
    _require_sales_call_analyzer()
    doc = await sales_call_store.get(analysis_id)
    if not doc:
        raise HTTPException(status_code=404, detail="analysis_id not found")

    analysis = doc.get("analysis")
    if doc.get("status") != sca.STATUS_COMPLETED or not analysis:
        raise HTTPException(status_code=409,
                            detail="Only a completed analysis can be rescored.")

    cfg = _sca_framework.load_framework()
    result = _sca_scoring.rescore(analysis, cfg, doc.get("blocked_criteria") or {})

    report = dict(doc.get("report") or {})
    report["scores"] = result["scores"].model_dump(mode="json")
    report["stage_evaluations"] = [s.model_dump(mode="json")
                                   for s in result["stage_evaluations"]]
    report["status"] = sca.STATUS_COMPLETED
    await sales_call_store.complete(
        analysis_id, report=report, analysis=analysis,
        blocked=doc.get("blocked_criteria") or {},
        processing=doc.get("processing") or {})
    logger.info("analysis rescored [analysis_id=%s framework_version=%s]",
                analysis_id, cfg["framework_version"])
    return report


# ---------------------------------------------------------------------------
# Vision Lab
#
# Uploads a creative? No. The creative is up to 200 MB and nginx caps request
# bodies at 1 MB, so the backend presigns an upload straight to S3 and sends us
# a URL. We read it and forget it.
#
# ASYNCHRONOUS. Frame decode, saliency inference and OCR take one to two
# minutes, so POST returns an analysis_id immediately and the caller polls the
# GET. Unlike the Sales Call Analyzer this does NOT use a BackgroundTask: the
# work is CPU-bound and would stall every other endpoint in this process. It
# runs in vision_lab/worker.py under pm2, which claims jobs from MongoDB.
#
# Path style follows the rest of this file (/api/<feature>/...). Versioning is
# carried in the payload - framework_version, psychology_version,
# prompt_version, saliency_model - so the contract can evolve without a URL
# change.
# ---------------------------------------------------------------------------
VISION_LAB_POLL_SECONDS = int(os.environ.get("VL_POLL_INTERVAL_SECONDS", 5))

# Which hosts a creative URL may point at. Accepting an arbitrary URL and
# fetching it server-side is server-side request forgery; the allowlist is what
# bounds it. VL_ALLOW_ANY_URL exists for local development only and must stay
# false in production.
VL_ALLOWED_URL_HOSTS = [
    host.strip().lower()
    for host in os.environ.get("VL_ALLOWED_URL_HOSTS", "").split(",")
    if host.strip()
]
VL_ALLOW_ANY_URL = os.environ.get("VL_ALLOW_ANY_URL", "false").lower() == "true"


async def _vision_lab_health() -> dict:
    """Which Vision Lab subsystems are configured, and is the worker alive?

    The worker is a separate process, so a deploy can leave the API healthy
    while every job sits in `queued` forever. Its heartbeat is reported here so
    that failure is visible rather than silent.
    """
    health = {
        "available": VISION_LAB_AVAILABLE,
        "enabled": VL_ENABLED,
        "storage": "configured" if vision_lab_store is not None else "not_configured",
        "worker": {"seen_seconds_ago": None, "healthy": False, "in_flight": 0},
    }
    if not VISION_LAB_AVAILABLE or vision_lab_store is None:
        return health

    cfg_model = _vl_framework.saliency_model()
    health["model"] = "configured" if cfg_model.get("name") else "not_configured"
    health["framework_version"] = _vl_framework.load_framework()["framework_version"]

    # The interpretation layer is OPTIONAL, and a report says so when it is
    # missing - but an integrator seeing reports with no recommendations needs
    # to be able to tell "this server cannot interpret" from "this ad had
    # nothing wrong with it" without opening a report to find out.
    from vision_lab import transcript as _vl_transcript
    health["transcription"] = ("configured" if _vl_transcript.is_configured()
                               else "not_configured")
    health["interpretation"] = "configured" if GEMINI_API_KEY else "not_configured"

    # An in-flight job's heartbeat is the cheapest liveness signal we have that
    # does not need the worker to write a second document.
    try:
        active = await vision_lab_analyses.find(
            {"status": {"$in": list(vl.ACTIVE_STATUSES)}}
        ).to_list(length=20)
        health["worker"]["in_flight"] = len(active)
        beats = [doc.get("heartbeat_at") for doc in active if doc.get("heartbeat_at")]
        if beats:
            newest = max(beats)
            if newest.tzinfo is None:
                newest = newest.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - newest).total_seconds()
            health["worker"]["seen_seconds_ago"] = round(age, 1)
            health["worker"]["healthy"] = age < _vl_store.STALE_AFTER_SECONDS
        else:
            # Nothing in flight is not evidence of a dead worker, only that
            # there is nothing to do. Reported as unknown, not unhealthy.
            health["worker"]["healthy"] = None
    except Exception as err:  # noqa: BLE001 - health must never 500
        logger.warning("vision lab health probe failed: %s", err)
    return health


def _require_vision_lab():
    if not VISION_LAB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Vision Lab is not available on this server.")
    if not VL_ENABLED:
        raise HTTPException(
            status_code=503,
            detail=("Vision Lab is not enabled on this server. "
                    "Set VL_ENABLED=true in the environment."))
    if vision_lab_store is None:
        raise HTTPException(
            status_code=503,
            detail=("Vision Lab storage is not configured. "
                    "Set MONGODB_URI in the environment."))


def _vision_url_allowed(url: str) -> bool:
    """Bound what the worker will fetch.

    Without this, anyone who can call the API can make the server issue GETs to
    arbitrary hosts, including private addresses on the server's own network.
    """
    if not url or not url.lower().startswith("https://"):
        return False
    if VL_ALLOW_ANY_URL:
        return True
    if not VL_ALLOWED_URL_HOSTS:
        return False
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    host = host.lower()
    return any(host == allowed or host.endswith("." + allowed)
               for allowed in VL_ALLOWED_URL_HOSTS)


def _vision_poll_url(analysis_id: str) -> str:
    return f"/api/vision-lab/analysis/{analysis_id}"


@app.post("/api/vision-lab/analyze", response_model=VisionAccepted,
          dependencies=[Depends(require_api_key)])
async def analyze_creative(body: VisionAnalyzeRequest):
    """Submit a creative for attention analysis. Returns immediately.

    This is NOT the report - poll GET /api/vision-lab/analysis/{analysis_id}.

    Idempotent: an identical submission returns the existing analysis and costs
    nothing. Send options.force_reanalysis=true to override.
    """
    _require_vision_lab()

    cfg = _vl_framework.load_framework()
    sampling = _vl_framework.sampling(cfg)
    sample_fps = float(body.options.sample_fps or sampling.get("sample_fps") or 2.0)
    versions = {
        **_vl_framework.versions(),
        "prompt_version": _vl_pipeline.PROMPT_VERSION,
        "saliency_model": _vl_framework.saliency_model(cfg).get("name") or "stub",
    }

    fingerprint = _vl_store.compute_fingerprint(
        body,
        framework_version=versions["framework_version"],
        psychology_version=versions["psychology_version"],
        prompt_version=versions["prompt_version"],
        saliency_model=versions["saliency_model"],
        sample_fps=sample_fps,
    )

    # Already analysed, or already running? Hand back what exists.
    existing = await vision_lab_store.find_by_fingerprint(fingerprint)
    if _vl_store.reusable(existing, force=body.options.force_reanalysis):
        return VisionAccepted(
            analysis_id=existing["_id"], creative_id=existing.get("creative_id", ""),
            status=existing["status"], created_at=existing.get("created_at"),
            idempotent_hit=True, poll_url=_vision_poll_url(existing["_id"]),
            suggested_poll_interval_seconds=VISION_LAB_POLL_SECONDS)

    analysis_id = _vl_store.new_analysis_id()
    doc = await vision_lab_store.create(analysis_id=analysis_id, request=body,
                                        fingerprint=fingerprint, versions=versions)

    # Nothing to analyse, or somewhere we refuse to fetch from. Recorded as a
    # failed analysis rather than a 4xx, so the caller gets an auditable row and
    # the same reason vocabulary as every other failure - consistent with the
    # always-200 contract used across this API.
    refusal = None
    if not body.creative.url:
        refusal = vl.NO_CREATIVE
    elif not _vision_url_allowed(body.creative.url):
        refusal = vl.URL_NOT_ALLOWED

    if refusal:
        await vision_lab_store.fail(analysis_id, reason=refusal,
                                    message=vl.REASON_TEXT[refusal])
        await vision_lab_store.clear_creative_url(analysis_id)
        return VisionAccepted(
            analysis_id=analysis_id, creative_id=body.creative_id,
            status=vl.STATUS_FAILED, created_at=doc["created_at"],
            poll_url=_vision_poll_url(analysis_id),
            availability=_vl_pipeline.unavailable(refusal),
            reason=refusal, message=vl.REASON_TEXT[refusal])

    # No background task here. The document is now queued, and vision-worker
    # picks it up - see vision_lab/worker.py.
    logger.info("vision lab analysis queued [analysis_id=%s creative_id=%s ad_number=%s]",
                analysis_id, body.creative_id, body.ad_number)

    return VisionAccepted(
        analysis_id=analysis_id, creative_id=body.creative_id,
        status=vl.STATUS_QUEUED, created_at=doc["created_at"],
        poll_url=_vision_poll_url(analysis_id),
        suggested_poll_interval_seconds=VISION_LAB_POLL_SECONDS)


# FastAPI refuses even to DEFINE a route with File()/Form() when python-multipart
# is missing - it raises at import, which would take the whole service down:
# onboarding, Script Lab, sales calls. Same rule as the guarded vision_lab
# import above - one missing dependency disables one feature, and says so.
import importlib.util as _importlib_util  # noqa: E402

_MULTIPART_AVAILABLE = bool(_importlib_util.find_spec("python_multipart")
                            or _importlib_util.find_spec("multipart"))
if not _MULTIPART_AVAILABLE:
    print("WARNING: python-multipart is not installed - "
          "POST /api/vision-lab/upload is disabled")

if _MULTIPART_AVAILABLE:
    @app.post(VISION_UPLOAD_PATH, response_model=VisionUploadAccepted,
              dependencies=[Depends(require_api_key)])
    async def upload_creative(
        file: UploadFile = File(..., description="The video or image to analyse."),
        creative_id: Optional[str] = Form(None),
        division: Optional[str] = Form(None),
        ad_number: Optional[str] = Form(None),
        brand_names: Optional[str] = Form(
            None, description="Comma-separated, e.g. Director's Institute,DI"),
        analyze: bool = Form(False, description="Also queue the analysis, in this call."),
    ):
        """Upload a creative straight to Vision Lab. multipart/form-data.

        For testing, and for callers holding a file but no bucket access. The
        ScaleSerum frontend should keep putting creatives in S3 itself and
        sending the link to /analyze - see vision_lab/uploads.py for why.

        Returns the stored object and the presigned link /analyze consumes.
        With analyze=true it also queues the analysis and returns, under
        `analysis`, exactly what /analyze would have.

        A file over the size limit is a 413: we refuse to RECEIVE it, a
        protocol-level answer like a missing key. Everything else we decline to
        store is a 200 with a stable `reason`, as everywhere else in this API.
        """
        _require_vision_lab()

        def declined(reason: str, message: str, status_code: int = 200):
            payload = {"uploaded": False, "reason": reason, "message": message,
                       "upload": None, "analysis": None}
            if status_code != 200:
                return JSONResponse(status_code=status_code, content=payload)
            return VisionUploadAccepted(**payload)

        if not _vl_uploads.storage_configured():
            return declined(vl.OBJECT_STORAGE_NOT_CONFIGURED,
                            "Upload storage is not configured. Set AWS_S3_BUCKET and "
                            "the AWS credentials in the environment.")

        # A file part whose name is blank. An EMPTY filename never gets here -
        # the multipart parser drops that part, and FastAPI answers 422 with the
        # required `file` field missing. A whitespace-only name does get here,
        # and without this would reach classify() as a file called "upload" and
        # come back "'upload' is not a format Vision Lab analyses" - true, and
        # no help. It is no_creative, as /analyze says for a missing creative.
        if not (file.filename or "").strip():
            return declined(vl.NO_CREATIVE,
                            "No file was uploaded. Choose one in the `file` field.")

        try:
            kind, mime = _vl_uploads.classify(file.filename, file.content_type)
            size = _vl_uploads.measure(file.file)
            _vl_uploads.check_size(size)
        except _vl_uploads.UploadRefused as refused:
            return declined(refused.reason, refused.message,
                            413 if refused.reason == vl.CREATIVE_TOO_LARGE else 200)

        key = _vl_uploads.object_key(file.filename)
        try:
            # boto3 is synchronous. On the event loop a 150 MB transfer would
            # stall every other endpoint in this process until it finished.
            await run_in_threadpool(_vl_uploads.put, file.file, key, mime)
        except Exception as err:  # noqa: BLE001 - reported, never a 500
            logger.warning("vision lab upload failed [key=%s]: %s", key, err)
            return declined(vl.UPLOAD_FAILED, vl.REASON_TEXT[vl.UPLOAD_FAILED])

        creative_url = _vl_uploads.sign(key)
        # The key and size are logged. The link never is - it is a credential.
        logger.info("vision lab creative uploaded [key=%s size=%d mime=%s]",
                    key, size, mime)

        upload = {"object_key": key, "filename": file.filename or "",
                  "size_bytes": size, "mime_type": mime, "kind": kind,
                  "creative_url": creative_url,
                  "creative_url_expires_in_seconds": _vl_uploads.URL_TTL_SECONDS}

        analysis = None
        if analyze:
            names = [n.strip() for n in (brand_names or "").split(",") if n.strip()]
            analysis = await analyze_creative(VisionAnalyzeRequest(
                creative_id=creative_id or _vl_uploads.creative_id_for(file.filename),
                division=division, ad_number=ad_number,
                creative={"url": creative_url, "kind": kind, "mime_type": mime},
                brand_assets={"brand_names": names},
            ))

        return VisionUploadAccepted(uploaded=True, upload=upload, analysis=analysis)


async def _vision_payload(doc: dict) -> dict:
    """Turn a stored job document into the polling response.

    A job whose heartbeat has stopped is reaped here as well as in the worker: a
    pm2 restart mid-flight (that is, every deploy) would otherwise leave a row
    reporting "analyzing_frames" forever. It becomes a stated failure the caller
    can retry.
    """
    if _vl_store.is_stale(doc):
        await vision_lab_store.mark_interrupted(doc["_id"])
        await vision_lab_store.clear_creative_url(doc["_id"])
        doc = await vision_lab_store.get(doc["_id"]) or doc

    status = doc.get("status")
    if status in vl.TERMINAL_STATUSES and doc.get("report"):
        report = dict(doc["report"])
        report["status"] = status
        return _sign_images(report)

    return {
        "analysis_id": doc["_id"],
        "creative_id": doc.get("creative_id"),
        "ad_number": doc.get("ad_number"),
        "division": doc.get("division"),
        "status": status,
        "availability": {
            "available": status != vl.STATUS_FAILED,
            "reason": doc.get("reason"),
            "message": doc.get("message"),
        },
        # Duplicated at the top level as well as inside `availability`, matching
        # the POST response. Carrying it in only one place on one endpoint means
        # the frontend has to look somewhere different depending on which call
        # it made, which is exactly the kind of trap that ships as a bug.
        "reason": doc.get("reason"),
        "message": doc.get("message"),
        "media": doc.get("media"),
        "scores": None,
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "attempts": doc.get("attempts", 0),
        "poll_url": _vision_poll_url(doc["_id"]),
        "suggested_poll_interval_seconds": VISION_LAB_POLL_SECONDS,
        "fallback": status == vl.STATUS_FAILED,
    }


def _sign_images(report: dict) -> dict:
    """Turn stored S3 object keys into presigned URLs, at READ time.

    The report holds keys, never URLs. A presigned URL written into the document
    would expire, and an analysis opened three weeks later would serve dead image
    links with nothing to explain why. Signing here means the link in a response
    is always live, however old the analysis is - and the images stay private.
    """
    if not VISION_LAB_AVAILABLE:
        return report
    try:
        from vision_lab import heatmap as _vl_heatmap
    except Exception:  # noqa: BLE001 - OpenCV missing must not break a GET
        return report

    heatmap = report.get("heatmap") or {}
    if heatmap.get("object_key"):
        heatmap = dict(heatmap)
        heatmap["image_url"] = _vl_heatmap.signed_url(heatmap["object_key"])
        report["heatmap"] = heatmap

    thumbnails = report.get("thumbnails") or []
    if thumbnails:
        report["thumbnails"] = [
            {**item, "image_url": _vl_heatmap.signed_url(item.get("object_key"))}
            for item in thumbnails]
    return report


@app.get("/api/vision-lab/analysis/{analysis_id}",
         dependencies=[Depends(require_api_key)])
async def get_vision_analysis(analysis_id: str):
    """Poll for status, then read the finished report.

    While processing: a small envelope with `status` and `scores: null`.
    When complete: the full report. On failure: the same shape with
    `availability.available=false`, a stable `reason`, and `scores: null` - a
    failed analysis never invents a scorecard.
    """
    _require_vision_lab()
    doc = await vision_lab_store.get(analysis_id)
    if not doc:
        raise HTTPException(status_code=404, detail="analysis_id not found")
    return await _vision_payload(doc)


@app.get("/api/vision-lab/analysis/by-creative/{creative_id}",
         dependencies=[Depends(require_api_key)])
async def get_latest_vision_analysis(creative_id: str):
    """The most recent analysis for a creative - for a backend that kept the
    creative_id but not the analysis_id."""
    _require_vision_lab()
    doc = await vision_lab_store.latest_for_creative(creative_id)
    if not doc:
        raise HTTPException(status_code=404,
                            detail="no analysis exists for that creative_id")
    return await _vision_payload(doc)


@app.get("/api/vision-lab/history", dependencies=[Depends(require_api_key)])
async def vision_lab_history(
    division: Optional[str] = Query(None),
    ad_number: Optional[str] = Query(None),
    creative_id: Optional[str] = Query(None),
    limit: int = Query(25, ge=1, le=100),
):
    """The History tab: previous analyses, newest first.

    Filter by ad_number to get the version history of one ad, which is what the
    optional Ad Number field on the upload form is for.
    """
    _require_vision_lab()
    docs = await vision_lab_store.history(division=division, ad_number=ad_number,
                                          creative_id=creative_id, limit=limit)
    items = [{
        "analysis_id": doc["_id"],
        "creative_id": doc.get("creative_id"),
        "ad_number": doc.get("ad_number"),
        "division": doc.get("division"),
        "status": doc.get("status"),
        "overall_score": ((doc.get("report") or {}).get("overall") or {}).get("score"),
        "created_at": doc.get("created_at"),
    } for doc in docs]
    return {"items": items, "count": len(items)}


@app.post("/api/vision-lab/analysis/{analysis_id}/rescore",
          dependencies=[Depends(require_api_key)])
async def rescore_vision_analysis(analysis_id: str):
    """Recompute the scores from the stored measurements under the current config.

    No ffmpeg, no saliency inference, no Gemini call, no cost. This is the payoff
    for storing per-frame measurements rather than only numbers: when management
    sets real weights or real bands, historical creatives can be brought onto the
    new configuration without re-processing a single frame.
    """
    _require_vision_lab()
    doc = await vision_lab_store.get(analysis_id)
    if not doc:
        raise HTTPException(status_code=404, detail="analysis_id not found")
    if doc.get("status") != vl.STATUS_COMPLETED:
        raise HTTPException(status_code=409,
                            detail="Only a completed analysis can be rescored.")

    stored = await vision_lab_store.get_measurements(analysis_id)
    if not stored:
        raise HTTPException(
            status_code=409,
            detail=("The per-frame measurements for this analysis are no longer "
                    "stored, so it cannot be rescored. Re-run the analysis."))

    # THE PAYOFF FOR STORING MEASUREMENTS RATHER THAN ONLY NUMBERS.
    #
    # The timeline and all six scores are recomputed from the per-frame record
    # under whatever vision_framework.json now says. No ffmpeg, no saliency
    # model, no Gemini, no network - which is what lets management set real
    # weights and bring every historical creative onto them without
    # re-processing a single frame.
    measurements = stored.get("measurements") or {}
    frames = measurements.get("frames") or []
    stored_summary = dict(measurements.get("summary") or {})
    sample_fps = measurements.get("sample_fps") or stored_summary.get("sample_fps") or 2.0

    report = dict(doc.get("report") or {})
    media = doc.get("media") or report.get("media") or {}
    cfg = _vl_framework.load_framework()

    recomputed = await run_in_threadpool(
        _vl_rescore, frames, stored_summary, media, cfg, sample_fps,
        (media.get("kind") or report.get("media", {}).get("kind")),
        doc.get("analysis") or {})

    report.update(recomputed)
    report["config_disclosure"] = _vl_framework.config_disclosure()
    report["versions"] = {**report.get("versions", {}), **_vl_framework.versions()}
    report["status"] = vl.STATUS_COMPLETED
    await vision_lab_store.complete(analysis_id, report=report,
                                    analysis=doc.get("analysis"),
                                    processing=doc.get("processing") or {})
    logger.info("vision lab analysis rescored [analysis_id=%s framework_version=%s]",
                analysis_id, _vl_framework.load_framework()["framework_version"])
    return report


def _vl_rescore(frames: list, stored_summary: dict, media: dict, cfg: dict,
                sample_fps: float, kind: str, analysis: dict) -> dict:
    """Rebuild the summary, timeline, key moments, scores and defects from the
    stored per-frame records.

    THE SUMMARY IS RECOMPUTED, NOT REUSED. Reading speed is a CONFIG value, and
    it is what decides whether a slide counts as overloaded. Reusing the stored
    summary would mean management could change reading speed and the overloaded
    slides - and so Cognitive Demand, and so the defect list - would not move,
    which defeats the point of rescoring at all.

    THE MODEL IS NOT CALLED AGAIN. Its ordinal ratings and the key message it
    named are stored on the analysis and reused, so Clarity keeps its blend and
    Focus keeps its window without a second Gemini call or a second bill.
    Recommendations are then RE-ANCHORED against the recomputed defects: if new
    config means a slide is no longer overloaded, the written advice about it
    goes too, rather than being left behind pointing at a finding that has
    stopped existing.

    Synchronous and CPU-light: a few hundred arithmetic operations over ~120
    records. Nothing here needs a GPU or a second process.
    """
    from vision_lab import defects as _df
    from vision_lab import measure as _ms
    from vision_lab import scoring as _sc
    from vision_lab import timeline as _tl

    summary = (_ms.summarise(frames, {**media, "sample_fps": sample_fps})
               if frames else dict(stored_summary))
    summary["sample_fps"] = sample_fps

    timeline = (_tl.build(frames, {**media, "sample_fps": summary["sample_fps"]},
                          cfg) if kind == vl.KIND_VIDEO and frames else None)
    moments = (_tl.key_moments(timeline["points"], frames, timeline["weak_zones"])
               if timeline else [])
    scored = _sc.score_all(frames, summary, timeline or {"points": []}, cfg,
                           llm_ratings=(analysis or {}).get("llm_ratings"),
                           key_message=(analysis or {}).get("key_message"))
    found = _df.detect(frames, summary, timeline, cfg)

    return {
        "scores": scored["scores"],
        "overall": scored["overall"],
        "defects": found,
        "recommendations": _vl_reanchor(
            ((analysis or {}).get("interpretation") or {}).get("recommendations"),
            found),
        "timeline": ({k: v for k, v in timeline.items() if k != "signals"}
                     if timeline else None),
        "key_moments": moments,
        "measurements": summary,
    }


def _vl_reanchor(recommendations: Optional[list], defects: list) -> list:
    """Point each stored recommendation at the recomputed defect, or drop it.

    A recommendation quotes the numbers its defect was found with. After a
    config change those numbers can move, so the anchor is refreshed from the
    new defect - and one whose defect no longer exists is dropped rather than
    republished against a finding that is gone.
    """
    by_id = {d["defect_id"]: d for d in defects or []}
    kept = []
    for item in recommendations or []:
        defect = by_id.get(item.get("defect_id"))
        if not defect:
            continue
        item = {**item,
                "anchor": {"defect_id": defect["defect_id"],
                           "t_start": defect.get("t_start"),
                           "t_end": defect.get("t_end"),
                           "measured": defect.get("measured", {})},
                "severity": defect.get("severity"),
                "scores_impacted": defect.get("scores_impacted", []),
                "rank": len(kept) + 1}
        kept.append(item)
    return kept


@app.delete("/api/vision-lab/analysis/{analysis_id}",
            dependencies=[Depends(require_api_key)])
async def delete_vision_analysis(analysis_id: str):
    """Remove an analysis and its measurements.

    Called by the backend when a customer deletes a creative. We do not touch
    the creative itself - that belongs to the backend.
    """
    _require_vision_lab()
    doc = await vision_lab_store.get(analysis_id)
    if not doc:
        raise HTTPException(status_code=404, detail="analysis_id not found")
    await vision_lab_analyses.delete_one({"_id": analysis_id})
    if vision_lab_measurements is not None:
        await vision_lab_measurements.delete_one({"_id": analysis_id})

    # The rendered overlays go too. We do NOT touch the creative itself - that
    # belongs to the backend.
    removed = 0
    try:
        from vision_lab import heatmap as _vl_heatmap
        removed = await run_in_threadpool(_vl_heatmap.delete_all, analysis_id)
    except Exception as err:  # noqa: BLE001
        logger.warning("could not remove stored images [analysis_id=%s]: %s",
                       analysis_id, err)

    logger.info("vision lab analysis deleted [analysis_id=%s images=%d]",
                analysis_id, removed)
    return {"deleted": True, "analysis_id": analysis_id, "images_removed": removed}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
