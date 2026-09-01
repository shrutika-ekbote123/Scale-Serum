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
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
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
else:
    mongo_client = None
    brand_brains = None


def _require_mongo():
    if brand_brains is None:
        raise HTTPException(
            status_code=503,
            detail="Brand Brain storage is not configured. Set MONGODB_URI in the environment.",
        )


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


app = FastAPI(title="Brand Brain Persona Rewriter", version="1.0.0")

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
    return {"ok": True, "model": GEMINI_MODEL}


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
# Purchase Probability (baseline MVP)
#
# Scores a lead with the frozen baseline model in purchase_probability_model/.
# The model reads PostgreSQL READ-ONLY; this service never writes to it.
#
# Two things are returned and they are NOT the same thing:
#   * purchase_probability - the real calibrated model output, as a percentage.
#     Base rate is ~1.1%, so genuine values sit roughly in 0.3%-4%. It is never
#     rescaled to look bigger.
#   * percentile / decile / priority - relative ranking against the frozen
#     out-of-fold reference. This is the signal to prioritise leads with.
#
# Touchpoints in the response are DISPLAY history. Model inputs are reported
# separately under `model_features`. They must not be conflated.
# ---------------------------------------------------------------------------
PURCHASE_PROBABILITY_AVAILABLE = True
try:
    from purchase_probability_model import predict_for_lead as _pp_predict
except Exception as _pp_import_error:  # pragma: no cover - import-time only
    PURCHASE_PROBABILITY_AVAILABLE = False
    _PP_IMPORT_ERROR = repr(_pp_import_error)
    print(f"WARNING: purchase_probability_model unavailable - {_PP_IMPORT_ERROR}")


@app.get("/api/purchase-probability/{lead_id}",
         dependencies=[Depends(require_api_key)])
async def purchase_probability(lead_id: str):
    """Calibrated purchase probability, ranking and contributing factors for a lead.

    Follows the house convention: always HTTP 200. When the lead cannot be scored
    the response carries `fallback: true` and `availability.available: false`
    rather than a fabricated number. `null` and `0` mean different things here.
    """
    if not PURCHASE_PROBABILITY_AVAILABLE:
        return {
            "lead_id": lead_id,
            "purchase_probability": None, "purchase_probability_percent": None,
            "probability": None, "percentile": None, "decile": None,
            "priority": "Unavailable", "top_factors": [], "model_features": None,
            "touchpoint_count": 0, "touchpoints": [],
            "model": {"name": "purchase_probability", "version": "baseline_mvp",
                      "status": "baseline_mvp"},
            "availability": {"available": False,
                             "reason": "model_artefacts_unavailable",
                             "message": "Model artefacts are not available on this server."},
            "fallback": True, "reason": "model_artefacts_unavailable",
        }

    # Inference is synchronous (psycopg + sklearn); keep it off the event loop.
    return await run_in_threadpool(_pp_predict, lead_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
