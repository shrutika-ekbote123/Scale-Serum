"""
Context assembly - everything the analysis needs to know about who was on the
call, what was being sold and what the brand stands for.

WHY CONTEXT IS A MODULE AND NOT A PROMPT STRING
    The same sales behaviour is good or bad depending on the customer, the
    product and the price. Assembling that context deliberately - and reporting
    exactly what was and was not available - is what separates "the rep did not
    follow the script" from "the rep adapted to this customer".

WHAT THIS IS NOT
    It is not a rule engine. There are no per-region, per-product or
    per-profile behaviour rules here, and none should be added without a written
    basis from the business. Region, language, price band and profile are
    supplied to the model as FACTS about this call. They inform interpretation
    and recommendations; they never silently change a weight or a score.

BRAND BRAIN
    Reused, not rebuilt. The document is fetched by app.py with the existing
    `_load_brand_brain()` helper and the existing `resolve_brand_brain_ref()`
    lookup, then passed in here. This package never opens a Mongo connection -
    the same rule purchase_probability_model follows.

PII
    Email and phone number are part of the report (the UI shows them) but are
    NOT rendered into the prompt: the model does not need them to evaluate a
    sales call, and every field left out is a field that cannot leak into an
    LLM provider's logs.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from . import NO_BRAND_BRAIN, NO_CUSTOMER_CONTEXT, NO_PRODUCT_CONTEXT
from .models import (
    AnalyzeRequest,
    CallMetadata,
    ContextUsed,
    CustomerInfo,
    NormalizedTranscript,
    ProductInfo,
    RepInfo,
)
from .transcript import SOURCE_DEEPGRAM

# Framework requirement names. Declared in sales_framework.json; satisfied here.
REQ_VOCAL_TONE = "vocal_tone"
REQ_PRIOR_INTERACTION = "prior_interaction"
REQ_STRUCTURED_PROGRAMME = "structured_programme"
REQ_PAYMENT_PROCESS = "payment_process"


class BrandContext(BaseModel):
    """The Brand Brain, reduced to what a call review can use."""
    available: bool = False
    reason: Optional[str] = None
    brand_brain_id: Optional[str] = None
    brand_id: Optional[str] = None
    brand_name: Optional[str] = None
    business_type: Optional[str] = None
    industry: Optional[str] = None
    ideal_customer: Optional[str] = None
    brand_voice: Optional[str] = None
    language: Optional[str] = None
    sales_cycle: Optional[str] = None
    marketing_goal: Optional[str] = None
    audience_short: Optional[str] = None
    journey: Optional[str] = None
    competitors: list[str] = Field(default_factory=list)


class AnalysisContext(BaseModel):
    """Everything assembled, ready to render into the prompt."""
    call_id: str
    lead_id: Optional[str] = None
    brand: BrandContext = Field(default_factory=BrandContext)
    customer: CustomerInfo = Field(default_factory=CustomerInfo)
    product: ProductInfo = Field(default_factory=ProductInfo)
    rep: RepInfo = Field(default_factory=RepInfo)
    call: CallMetadata = Field(default_factory=CallMetadata)
    missing: list[str] = Field(default_factory=list)


# =========================================================================== #
# Assembly
# =========================================================================== #
def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_brand_context(brand_brain: Optional[dict],
                        brand_ref: Optional[dict] = None) -> BrandContext:
    """Reduce the Brand Brain document to the fields a call review can use.

    A missing Brand Brain is not an error. It is reported, and the criteria that
    genuinely depend on brand context become not-applicable rather than scoring
    low - otherwise every brand that skipped onboarding would look like it had
    bad salespeople.
    """
    ref = brand_ref or {}
    if not brand_brain:
        return BrandContext(
            available=False,
            reason=NO_BRAND_BRAIN,
            brand_id=_clean(ref.get("brand_id")),
            brand_name=_clean(ref.get("brand_name")),
            brand_brain_id=_clean(ref.get("brand_brain_id")),
        )

    answers = brand_brain.get("answers") or {}
    ctx = brand_brain.get("context") or {}
    competitors = [str(c).strip() for c in (answers.get("competitors") or []) if str(c).strip()]

    return BrandContext(
        available=True,
        brand_brain_id=_clean(brand_brain.get("_id")) or _clean(ref.get("brand_brain_id")),
        brand_id=_clean(ref.get("brand_id")),
        brand_name=_clean(ctx.get("brandName")) or _clean(ref.get("brand_name")),
        business_type=_clean(answers.get("businessType")) or _clean(ctx.get("businessType")),
        industry=_clean(ctx.get("industry")),
        ideal_customer=_clean(answers.get("idealCustomer")),
        brand_voice=_clean(answers.get("brandVoice")),
        language=_clean(answers.get("language")),
        sales_cycle=_clean(answers.get("salesCycle")),
        marketing_goal=_clean(answers.get("marketingGoal")),
        audience_short=_clean(ctx.get("audienceShort")),
        journey=_clean(answers.get("journey")),
        competitors=competitors,
    )


def build_context(request: AnalyzeRequest, brand_brain: Optional[dict] = None,
                  brand_ref: Optional[dict] = None) -> AnalysisContext:
    """Assemble the call's context and record what was missing."""
    brand = parse_brand_context(brand_brain, brand_ref)

    missing: list[str] = []
    if not brand.available:
        missing.append(NO_BRAND_BRAIN)

    customer = request.customer
    if not any([customer.name, customer.region, customer.designation, customer.industry,
                customer.profile, customer.language]):
        missing.append(NO_CUSTOMER_CONTEXT)

    product = request.product
    if not any([product.name, product.description, product.price, product.price_band]):
        missing.append(NO_PRODUCT_CONTEXT)

    return AnalysisContext(
        call_id=request.call_id,
        lead_id=request.lead_id,
        brand=brand,
        customer=customer,
        product=product,
        rep=request.rep,
        call=request.call_metadata,
        missing=missing,
    )


def satisfied_requirements(ctx: AnalysisContext,
                           transcript: NormalizedTranscript) -> set[str]:
    """Which framework requirements this call can actually meet.

    Conservative by design: an unknown is not a "yes". A criterion whose
    requirement cannot be confirmed becomes not-applicable, which is honest,
    rather than being scored on a guess.
    """
    satisfied: set[str] = set()

    # Tone, energy and interruption only exist in audio-derived transcripts.
    if transcript.source == SOURCE_DEEPGRAM and transcript.timestamps_available:
        satisfied.add(REQ_VOCAL_TONE)

    profile = ctx.customer.profile or {}
    if ctx.customer.previous_interactions or profile.get("source") or profile.get("campaign"):
        satisfied.add(REQ_PRIOR_INTERACTION)

    # Explicit true only. `None` means the backend did not tell us, and guessing
    # that a product is a structured programme would invent a criterion to
    # judge the rep against.
    if ctx.product.is_structured_programme is True:
        satisfied.add(REQ_STRUCTURED_PROGRAMME)
    if ctx.product.sold_on_call is True:
        satisfied.add(REQ_PAYMENT_PROCESS)

    return satisfied


def to_context_used(ctx: AnalysisContext, transcript: NormalizedTranscript,
                    unmet: dict[str, str]) -> ContextUsed:
    """What the report says about the context it had. Missing context is stated
    so a thin report can be explained rather than looking like a bad call."""
    return ContextUsed(
        brand_brain={
            "available": ctx.brand.available,
            "reason": ctx.brand.reason,
            "brand_brain_id": ctx.brand.brand_brain_id,
            "brand_id": ctx.brand.brand_id,
            "brand_name": ctx.brand.brand_name,
        },
        customer={
            "name_known": bool(ctx.customer.name),
            "region": ctx.customer.region,
            "language": ctx.customer.language,
            "designation": ctx.customer.designation,
            "industry": ctx.customer.industry,
            "awareness_level": ctx.customer.awareness_level,
            "profile_fields": sorted((ctx.customer.profile or {}).keys()),
            "previous_interactions": len(ctx.customer.previous_interactions or []),
        },
        product={
            "name": ctx.product.name,
            "price": ctx.product.price,
            "currency": ctx.product.currency,
            "price_band": ctx.product.price_band,
            "complexity": ctx.product.complexity,
            "is_structured_programme": ctx.product.is_structured_programme,
            "sold_on_call": ctx.product.sold_on_call,
        },
        rep={"name_known": bool(ctx.rep.name), "designation": ctx.rep.designation},
        call={
            "direction": ctx.call.direction,
            "disposition": ctx.call.disposition,
            "duration_seconds": ctx.call.duration_seconds,
            "has_remarks": bool((ctx.call.remarks or "").strip()),
            "transcript_source": transcript.source,
            "diarization_available": transcript.diarization_available,
            "timestamps_available": transcript.timestamps_available,
            "speaker_count": transcript.speaker_count,
        },
        missing=list(ctx.missing),
        unmet_requirements=unmet,
    )


# =========================================================================== #
# Prompt rendering
# =========================================================================== #
def _block(rows: list[tuple[str, Any]], empty: str) -> str:
    """Label/value lines with empty fields skipped.

    Same idiom as build_context_block / build_answers_block in app.py: an absent
    field is left out entirely rather than rendered as "unknown", because a
    model reads "Region: unknown" as a fact about the customer.
    """
    lines = [f"- {label}: {value}" for label, value in rows
             if value is not None and str(value).strip()]
    return "\n".join(lines) if lines else empty


def render_brand_block(brand: BrandContext) -> str:
    if not brand.available:
        return ("(no Brand Brain is linked to this brand - judge the call on the customer "
                "and product context only, and do not assume what this brand's approved "
                "messaging or positioning is)")
    return _block([
        ("Brand", brand.brand_name),
        ("Business type", brand.business_type),
        ("Industry", brand.industry),
        ("Ideal customer", brand.ideal_customer),
        ("Brand voice", brand.brand_voice),
        ("Selling language", brand.language),
        ("Typical sales cycle", brand.sales_cycle),
        ("Primary marketing goal", brand.marketing_goal),
        ("Audience (short)", brand.audience_short),
        ("Lead-to-sale journey", brand.journey),
        ("Competitors", ", ".join(brand.competitors)),
    ], "(Brand Brain is linked but empty)")


def render_customer_block(customer: CustomerInfo) -> str:
    """Customer facts for interpretation. Email and phone are deliberately
    excluded - they are in the report, not in the prompt."""
    profile_rows = [(f"Profile: {k}", v) for k, v in sorted((customer.profile or {}).items())
                    if k.lower() not in ("email", "phone", "mobile", "contact")]
    return _block([
        ("Name", customer.name),
        ("Region", customer.region),
        ("Preferred language", customer.language),
        ("Designation", customer.designation),
        ("Industry", customer.industry),
        ("Experience", customer.experience),
        ("Awareness of the product", customer.awareness_level),
        ("Previous interactions", "; ".join(customer.previous_interactions or [])),
        *profile_rows,
    ], "(no customer details were supplied)")


def render_product_block(product: ProductInfo) -> str:
    price = None
    if product.price is not None:
        price = f"{product.currency or ''} {product.price:,.0f}".strip()
    return _block([
        ("Product", product.name),
        ("Description", product.description),
        ("Price", price),
        ("Price band", product.price_band),
        ("Complexity", product.complexity),
        ("Structured programme", product.is_structured_programme),
        ("Sold and paid for on the call", product.sold_on_call),
    ], "(no product details were supplied)")


def render_call_block(call: CallMetadata, rep: RepInfo,
                      transcript: NormalizedTranscript) -> str:
    duration = call.duration_seconds or transcript.duration_seconds
    return _block([
        ("Direction", call.direction),
        ("Occurred at", call.occurred_at.isoformat() if call.occurred_at else None),
        ("Duration", f"{duration:.0f} seconds" if duration else None),
        ("CRM disposition recorded by the representative", call.disposition),
        ("Representative's own remarks", call.remarks),
        ("Representative", rep.name),
        ("Representative designation", rep.designation),
        ("Transcript source", transcript.source),
        ("Speaker attribution available", transcript.diarization_available),
        ("Timings available", transcript.timestamps_available),
        ("Speakers detected", transcript.speaker_count or None),
    ], "(no call metadata was supplied)")


def known_context_factors(ctx: AnalysisContext) -> list[str]:
    """Which adaptation factors this call actually supplies. Only these may be
    reasoned about - the rest are unknown, not assumed."""
    factors: list[str] = []
    if ctx.customer.region:
        factors.append("region")
    if ctx.customer.language or ctx.brand.language:
        factors.append("language")
    if ctx.product.name or ctx.product.description:
        factors.append("product")
    if ctx.product.price is not None or ctx.product.price_band:
        factors.append("price band")
    if ctx.product.complexity:
        factors.append("product complexity")
    if ctx.customer.designation or ctx.customer.industry or ctx.customer.experience:
        factors.append("customer profile")
    if ctx.customer.awareness_level:
        factors.append("customer awareness")
    if ctx.customer.previous_interactions:
        factors.append("previous interactions")
    if ctx.brand.available:
        factors.append("brand positioning")
    if ctx.brand.sales_cycle:
        factors.append("brand sales cycle")
    if ctx.call.direction:
        factors.append("call direction")
    return factors


def render_for_prompt(ctx: AnalysisContext, transcript: NormalizedTranscript) -> str:
    """The whole context block, in the order the prompt expects it."""
    factors = known_context_factors(ctx)
    factor_line = (", ".join(factors) if factors
                   else "(none - no contextual factors were supplied for this call)")
    return "\n".join([
        "BRAND CONTEXT (Brand Brain):",
        render_brand_block(ctx.brand),
        "",
        "CUSTOMER CONTEXT:",
        render_customer_block(ctx.customer),
        "",
        "PRODUCT CONTEXT:",
        render_product_block(ctx.product),
        "",
        "CALL METADATA:",
        render_call_block(ctx.call, ctx.rep, transcript),
        "",
        f"CONTEXT FACTORS AVAILABLE FOR THIS CALL: {factor_line}",
        "Judge the representative on how well they adapted to THIS customer, product and "
        "price, using only the factors listed above. Where a factor is not listed, it is "
        "unknown - do not assume it, and do not apply generalisations about how people "
        "from a region, industry or demographic behave.",
    ])
