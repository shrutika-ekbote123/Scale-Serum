"""Context assembly: Brand Brain reuse, customer/product/region facts, and the
requirement gates that turn unknowns into not-applicable rather than guesses.

Pure unit tests. The Brand Brain fixture is shaped like the real MongoDB
`brand_brains` document written by /api/brand-brain/save.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import context as ctxmod  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from sales_call_analyzer.models import (  # noqa: E402
    AnalyzeRequest,
    CallMetadata,
    CustomerInfo,
    ProductInfo,
    RepInfo,
    SuppliedSegment,
    SuppliedTranscript,
)

# Shaped exactly like the real brand_brains document.
BRAND_BRAIN = {
    "_id": "bb_test_id",
    "answers": {
        "businessType": "Education / Coaching / Consulting",
        "idealCustomer": "Mid-career professionals seeking a recognised certification",
        "brandVoice": "Warm, credible, never pushy",
        "language": "English and Hindi",
        "trafficChannels": ["Google Search", "Meta (Facebook + Instagram)"],
        "salesCycle": "1-4 weeks",
        "competitors": ["Competitor A", "Competitor B"],
        "marketingGoal": "Lead quality (higher intent)",
        "journey": "Webinar registration, then a consultative call.",
    },
    "context": {"industry": "Education / Coaching", "brandName": "EdTech Pro",
                "audienceShort": "Mid-career professionals", "website": "https://example.com"},
}

BRAND_REF = {"brand_id": "brand_1", "brand_name": "EdTech Pro", "brand_brain_id": "bb_test_id"}


def audio_transcript():
    return tr.from_deepgram({
        "metadata": {"duration": 60.0},
        "results": {"utterances": [
            {"speaker": 0, "start": 0.0, "end": 3.0, "transcript": "Hello.", "confidence": 0.9},
            {"speaker": 1, "start": 3.2, "end": 6.0, "transcript": "Hi there.", "confidence": 0.9},
        ]}})


def text_transcript():
    return tr.from_supplied_text(SuppliedTranscript(text="Rajan: Hello.\nMeera: Hi there."))


def request(**kwargs):
    base = dict(call_id="call_1", lead_id="lead_1")
    base.update(kwargs)
    return AnalyzeRequest(**base)


# --------------------------------------------------------------------------- brand brain
def test_brand_brain_is_reused_not_rebuilt():
    brand = ctxmod.parse_brand_context(BRAND_BRAIN, BRAND_REF)
    assert brand.available is True
    assert brand.brand_name == "EdTech Pro"
    assert brand.brand_voice == "Warm, credible, never pushy"
    assert brand.sales_cycle == "1-4 weeks"
    assert brand.competitors == ["Competitor A", "Competitor B"]
    assert brand.brand_brain_id == "bb_test_id"


def test_missing_brand_brain_is_reported_not_fatal():
    brand = ctxmod.parse_brand_context(None, BRAND_REF)
    assert brand.available is False
    assert brand.reason == sca.NO_BRAND_BRAIN
    # We still say which brand we looked at, so the UI can explain the gap.
    assert brand.brand_name == "EdTech Pro"
    assert brand.brand_id == "brand_1"


def test_missing_brand_brain_prompt_block_forbids_assuming_messaging():
    text = ctxmod.render_brand_block(ctxmod.parse_brand_context(None, None))
    assert "do not assume" in text.lower()


def test_analysis_continues_without_brand_brain():
    ctx = ctxmod.build_context(request(), brand_brain=None, brand_ref=BRAND_REF)
    assert sca.NO_BRAND_BRAIN in ctx.missing
    assert ctx.call_id == "call_1"          # the analysis is still assembled


# --------------------------------------------------------------------------- missing context
def test_missing_customer_and_product_context_are_recorded():
    ctx = ctxmod.build_context(request(), BRAND_BRAIN, BRAND_REF)
    assert sca.NO_CUSTOMER_CONTEXT in ctx.missing
    assert sca.NO_PRODUCT_CONTEXT in ctx.missing
    assert sca.NO_BRAND_BRAIN not in ctx.missing


def test_supplied_context_clears_the_missing_flags():
    ctx = ctxmod.build_context(
        request(customer=CustomerInfo(name="Meera Patel", region="Maharashtra"),
                product=ProductInfo(name="Career Accelerator", price=200000, currency="INR")),
        BRAND_BRAIN, BRAND_REF)
    assert ctx.missing == []


# --------------------------------------------------------------------------- requirement gates
def test_tone_requires_audio():
    ctx = ctxmod.build_context(request(), BRAND_BRAIN, BRAND_REF)
    assert ctxmod.REQ_VOCAL_TONE in ctxmod.satisfied_requirements(ctx, audio_transcript())
    assert ctxmod.REQ_VOCAL_TONE not in ctxmod.satisfied_requirements(ctx, text_transcript())


def test_unknown_product_shape_does_not_become_a_yes():
    """`None` means the backend did not say. Guessing would invent a criterion
    to judge the representative against."""
    ctx = ctxmod.build_context(request(product=ProductInfo(name="Something")),
                               BRAND_BRAIN, BRAND_REF)
    satisfied = ctxmod.satisfied_requirements(ctx, audio_transcript())
    assert ctxmod.REQ_STRUCTURED_PROGRAMME not in satisfied
    assert ctxmod.REQ_PAYMENT_PROCESS not in satisfied


def test_explicit_product_flags_satisfy_their_requirements():
    ctx = ctxmod.build_context(
        request(product=ProductInfo(name="Career Accelerator",
                                    is_structured_programme=True, sold_on_call=True)),
        BRAND_BRAIN, BRAND_REF)
    satisfied = ctxmod.satisfied_requirements(ctx, audio_transcript())
    assert ctxmod.REQ_STRUCTURED_PROGRAMME in satisfied
    assert ctxmod.REQ_PAYMENT_PROCESS in satisfied


def test_prior_interaction_comes_from_recorded_history_only():
    empty = ctxmod.build_context(request(), BRAND_BRAIN, BRAND_REF)
    assert ctxmod.REQ_PRIOR_INTERACTION not in ctxmod.satisfied_requirements(empty, audio_transcript())

    known = ctxmod.build_context(
        request(customer=CustomerInfo(previous_interactions=["Webinar, 6 Sept 2026"])),
        BRAND_BRAIN, BRAND_REF)
    assert ctxmod.REQ_PRIOR_INTERACTION in ctxmod.satisfied_requirements(known, audio_transcript())


def test_unmet_requirements_block_the_right_criteria():
    from sales_call_analyzer import framework as fw
    cfg = fw.load_framework()
    ctx = ctxmod.build_context(request(), BRAND_BRAIN, BRAND_REF)
    satisfied = ctxmod.satisfied_requirements(ctx, text_transcript())
    blocked = fw.criteria_blocked_by_requirements(cfg, satisfied)
    assert "opening_tone" in blocked            # no audio -> no tone
    assert "pitch_structure_explained" in blocked
    assert "probing_challenges" not in blocked  # nothing about this needs audio


# --------------------------------------------------------------------------- PII
def test_email_and_phone_never_reach_the_prompt():
    customer = CustomerInfo(name="Meera Patel", email="meera@example.com",
                            phone="+919876543210", region="Maharashtra",
                            profile={"email": "dupe@example.com", "seniority": "Manager"})
    text = ctxmod.render_customer_block(customer)
    assert "Meera Patel" in text
    assert "Maharashtra" in text
    assert "Manager" in text
    assert "meera@example.com" not in text
    assert "dupe@example.com" not in text
    assert "9876543210" not in text


def test_full_prompt_context_carries_no_contact_details():
    ctx = ctxmod.build_context(
        request(customer=CustomerInfo(name="Meera Patel", email="meera@example.com",
                                      phone="+919876543210")),
        BRAND_BRAIN, BRAND_REF)
    text = ctxmod.render_for_prompt(ctx, audio_transcript())
    assert "meera@example.com" not in text
    assert "9876543210" not in text


# --------------------------------------------------------------------------- adaptation factors
def test_only_supplied_factors_are_offered_for_adaptation():
    bare = ctxmod.build_context(request(), None, None)
    assert ctxmod.known_context_factors(bare) == []

    rich = ctxmod.build_context(
        request(customer=CustomerInfo(region="Maharashtra", designation="Senior Manager",
                                      awareness_level="problem-aware"),
                product=ProductInfo(name="Career Accelerator", price=200000,
                                    currency="INR", complexity="high"),
                call_metadata=CallMetadata(direction="outbound")),
        BRAND_BRAIN, BRAND_REF)
    factors = ctxmod.known_context_factors(rich)
    for expected in ("region", "product", "price band", "product complexity",
                     "customer profile", "customer awareness", "brand positioning",
                     "call direction"):
        assert expected in factors


def test_prompt_forbids_regional_generalisation():
    ctx = ctxmod.build_context(
        request(customer=CustomerInfo(region="Maharashtra")), BRAND_BRAIN, BRAND_REF)
    text = ctxmod.render_for_prompt(ctx, audio_transcript())
    assert "Maharashtra" in text
    assert "do not apply generalisations" in text
    assert "region" in text


def test_absent_fields_are_omitted_not_rendered_as_unknown():
    """A model reads 'Region: unknown' as a fact about the customer."""
    text = ctxmod.render_customer_block(CustomerInfo(name="Meera Patel"))
    assert "unknown" not in text.lower()
    assert "None" not in text


def test_price_is_rendered_with_its_currency():
    text = ctxmod.render_product_block(
        ProductInfo(name="Career Accelerator", price=200000, currency="INR"))
    assert "INR 200,000" in text


# --------------------------------------------------------------------------- context_used
def test_context_used_reports_what_was_available():
    ctx = ctxmod.build_context(
        request(customer=CustomerInfo(name="Meera Patel", region="Maharashtra",
                                      email="meera@example.com"),
                product=ProductInfo(name="Career Accelerator"),
                call_metadata=CallMetadata(direction="outbound", disposition="SQL")),
        BRAND_BRAIN, BRAND_REF)
    used = ctxmod.to_context_used(ctx, audio_transcript(), unmet={"vocal_tone": "needs audio"})

    assert used.brand_brain["available"] is True
    assert used.customer["region"] == "Maharashtra"
    assert used.customer["name_known"] is True
    assert "email" not in used.customer          # the flag, never the value
    assert used.call["disposition"] == "SQL"
    assert used.call["transcript_source"] == tr.SOURCE_DEEPGRAM
    assert used.unmet_requirements == {"vocal_tone": "needs audio"}


def test_supplied_structured_transcript_does_not_satisfy_tone():
    """Speaker labels are not tone. Only audio-derived transcripts carry it."""
    supplied = SuppliedTranscript(segments=[
        SuppliedSegment(speaker_id="speaker_0", start=0.0, end=2.0, text="Hello."),
        SuppliedSegment(speaker_id="speaker_1", start=2.1, end=4.0, text="Hi."),
    ])
    transcript = tr.from_supplied_structured(supplied)
    ctx = ctxmod.build_context(request(), BRAND_BRAIN, BRAND_REF)
    assert ctxmod.REQ_VOCAL_TONE not in ctxmod.satisfied_requirements(ctx, transcript)
