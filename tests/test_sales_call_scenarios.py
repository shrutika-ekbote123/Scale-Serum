"""End-to-end scenario coverage - the 22 cases the feature must handle.

Each test runs the real pipeline with fake providers, so transcription,
diarization, context assembly, evidence verification, scoring and reporting all
execute exactly as they will in production.

    1  audio, two speakers                 12  strong buying signals
    2  audio, three or more speakers       13  weak buying signals
    3  transcript only                     14  different regions
    4  audio plus plain transcript         15  different products
    5  Hindi-English code-switching        16  different customer profiles
    6  very short call                     17  missing Brand Brain
    7  long call                           18  missing customer context
    8  poor-quality audio                  19  duplicate analysis request
    9  No Answer                           20  Gemini failure
    10 Invalid number                      21  Deepgram failure
    11 strong objections                   22  partial pipeline failure

The recurring assertion across the context scenarios (14-16) is the one that
matters most: context changes what the model is TOLD and what is APPLICABLE, and
never silently changes the arithmetic. Identical ratings must produce an
identical score in Mumbai and in Manchester.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import analyzer as an  # noqa: E402
from sales_call_analyzer import framework as fw  # noqa: E402
from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import store as st  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from sales_call_analyzer.deepgram_client import DeepgramError  # noqa: E402
from sales_call_analyzer.models import (  # noqa: E402
    AnalyzeRequest,
    AudioRef,
    CallMetadata,
    CustomerInfo,
    ProductInfo,
    RepInfo,
    SuppliedTranscript,
)
from test_sales_call_pipeline import (  # noqa: E402
    FakeClient,
    FakeCollection,
    FakeResponse,
)


@pytest.fixture(scope="module")
def cfg():
    return fw.load_framework()


@pytest.fixture
def store():
    return st.AnalysisStore(FakeCollection(), raw_collection=FakeCollection())


# --------------------------------------------------------------------------- helpers
def dg(utterances, duration=None, detected="en"):
    """A Deepgram response from (speaker, text) pairs, with plausible timings."""
    built, clock = [], 0.0
    for speaker, text in utterances:
        words = max(len(text.split()), 1)
        span = words * 0.4
        built.append({"speaker": speaker, "start": round(clock, 2),
                      "end": round(clock + span, 2), "transcript": text,
                      "confidence": 0.93})
        clock += span + 0.3
    return {"metadata": {"duration": duration or round(clock, 2)},
            "results": {"channels": [{"detected_language": detected,
                                      "alternatives": [{"transcript": " ", "words": []}]}],
                        "utterances": built}}


TWO_SPEAKER = dg([
    (0, "Hi Meera, this is Rajan Kumar from EdTech Pro. Is now a good time to talk?"),
    (1, "Yes that works. I registered for the masterclass last month."),
    (0, "Wonderful. What is driving your interest in the programme right now?"),
    (1, "My main problem is I do not have a recognised certification for senior roles."),
    (0, "Understood. The Career Accelerator ends with an accredited certificate."),
    (1, "That sounds useful but the fee is higher than I expected."),
    (0, "We offer a three part instalment plan. Shall I send the payment link today?"),
    (1, "Yes please send it across and I will complete it this evening."),
])


def quote_from(transcript, index, words=5):
    """A verbatim fragment of a real segment, so evidence verifies."""
    return " ".join(transcript.segments[index].text.split()[:words])


def llm_json(cfg, transcript, *, ratings=None, not_applicable=(), extra=None):
    """A model response anchored to real segments.

    `ratings` maps criterion_id -> rating; anything unlisted is "adequate".
    """
    ratings = ratings or {}
    anchor = [{"segment_index": 0, "speaker_id": transcript.segments[0].speaker_id,
               "quote": quote_from(transcript, 0)}]
    criteria = []
    for stage in cfg["stages"]:
        for crit in stage["criteria"]:
            cid = crit["id"]
            if cid in not_applicable:
                criteria.append({"criterion_id": cid, "applicable": False,
                                 "not_applicable_reason": "The call gave no opportunity.",
                                 "observation": "Not observed.", "evidence": []})
                continue
            criteria.append({"criterion_id": cid, "applicable": True,
                             "rating": ratings.get(cid, "adequate"),
                             "confidence": "high", "observation": "Observed on the call.",
                             "evidence": anchor})
    payload = {"summary": "Scenario call.", "criteria": criteria,
               "stages": [{"stage_id": s["id"], "assessment": "Handled.",
                           "confidence": "high"} for s in cfg["stages"]]}
    payload.update(extra or {})
    return json.dumps(payload)


def request(**kwargs):
    base = dict(
        call_id="call_scenario", lead_id="lead_1",
        audio=AudioRef(url="https://rec.example.com/a.mp3"),
        rep=RepInfo(name="Rajan Kumar"),
        customer=CustomerInfo(name="Meera Patel"),
        product=ProductInfo(name="Career Accelerator"),
        call_metadata=CallMetadata(direction="outbound", disposition="SQL"),
    )
    base.update(kwargs)
    return AnalyzeRequest(**base)


def deps_for(store, *, deepgram=TWO_SPEAKER, llm=None, transcribe=None,
             brand_brain="default"):
    calls = {"transcribe": 0}

    async def fake_transcribe(url, **kwargs):
        calls["transcribe"] += 1
        return deepgram

    async def load_brand_brain(_id):
        if brand_brain == "default":
            return {"_id": "bb1", "answers": {"businessType": "Education / Coaching",
                                              "brandVoice": "Warm, credible",
                                              "salesCycle": "1-4 weeks"},
                    "context": {"brandName": "EdTech Pro"}}
        return brand_brain

    async def resolve_brand_ref(_lead_id):
        return {"brand_id": "brand_1", "brand_name": "EdTech Pro", "brand_brain_id": "bb1"}

    d = pl.PipelineDeps(
        store=store, llm_client=FakeClient(llm or []), llm_model="gemini-test",
        transcribe=transcribe or fake_transcribe,
        load_brand_brain=load_brand_brain, resolve_brand_ref=resolve_brand_ref,
        transcription_model="nova-test")
    d.extra["calls"] = calls
    return d


def run(coro):
    return asyncio.run(coro)


async def _execute(store, request_obj, deps):
    analysis_id = st.new_analysis_id()
    await store.create(analysis_id=analysis_id, request=request_obj,
                       fingerprint=st.new_analysis_id(), versions={})
    return await pl.run_analysis(analysis_id, request_obj, deps)


def execute(store, cfg, request_obj=None, *, deepgram=TWO_SPEAKER, ratings=None,
            not_applicable=(), extra=None, transcribe=None, llm=None,
            brand_brain="default"):
    """Run one scenario, building the model response from the real transcript."""
    request_obj = request_obj or request()
    if llm is None:
        preview = tr.from_deepgram(deepgram)
        if not preview.segments and request_obj.transcript:
            preview = (tr.from_supplied_structured(request_obj.transcript)
                       if tr.is_structured(request_obj.transcript)
                       else tr.from_supplied_text(request_obj.transcript))
        if request_obj.audio is None and request_obj.transcript is not None:
            preview = (tr.from_supplied_structured(request_obj.transcript)
                       if tr.is_structured(request_obj.transcript)
                       else tr.from_supplied_text(request_obj.transcript))
        llm = ([FakeResponse(llm_json(cfg, preview, ratings=ratings,
                                      not_applicable=not_applicable, extra=extra))]
               if preview.segments else [FakeResponse("{}")])
    deps = deps_for(store, deepgram=deepgram, llm=llm, transcribe=transcribe,
                    brand_brain=brand_brain)
    report = run(_execute(store, request_obj, deps))
    return report, deps


# =========================================================================== #
# 1-2. Speaker counts
# =========================================================================== #
def test_01_audio_with_two_speakers(store, cfg):
    report, deps = execute(store, cfg)
    assert report.status == sca.STATUS_COMPLETED
    assert report.transcript.speaker_count == 2
    roles = {s.speaker_id: s.role for s in report.transcript.speakers}
    assert roles["speaker_0"] == sca.ROLE_SALES_REP
    assert roles["speaker_1"] == sca.ROLE_CUSTOMER
    assert report.scores.overall is not None
    assert deps.extra["calls"]["transcribe"] == 1


def test_02_conference_call_with_four_speakers(store, cfg):
    four = dg([
        (0, "Good morning, this is Rajan Kumar calling from EdTech Pro."),
        (1, "Hello, my husband and my manager have joined the call as well."),
        (2, "Yes, I wanted to understand the fee structure before we decide."),
        (3, "And I would like to know how much time off she will need."),
        (1, "We will make the decision together after this call."),
    ])
    report, _ = execute(store, cfg, deepgram=four)
    assert report.transcript.speaker_count == 4
    roles = {s.speaker_id: s.role for s in report.transcript.speakers}
    assert roles["speaker_0"] == sca.ROLE_SALES_REP
    # Nobody self-identified as Meera, so no one is promoted to customer.
    assert sca.ROLE_CUSTOMER not in roles.values()
    assert [roles[f"speaker_{i}"] for i in (1, 2, 3)] == [sca.ROLE_PARTICIPANT] * 3
    assert all(s.name is None for s in report.transcript.speakers
               if s.role != sca.ROLE_SALES_REP)


# =========================================================================== #
# 3-4. Input combinations
# =========================================================================== #
def test_03_transcript_only_skips_transcription(store, cfg):
    supplied = SuppliedTranscript(text=(
        "Rajan: Hi Meera, this is Rajan Kumar from EdTech Pro.\n"
        "Meera: I need a recognised certification for senior roles.\n"
        "Rajan: The Career Accelerator ends with an accredited certificate."))
    report, deps = execute(store, cfg, request(audio=None, transcript=supplied))
    assert deps.extra["calls"]["transcribe"] == 0
    assert report.transcript.source == tr.SOURCE_SUPPLIED_TEXT
    assert report.transcript.timestamps_available is False
    assert report.analysis_quality.degraded is True
    assert report.scores.overall is not None      # still scorable, just degraded


def test_04_audio_plus_plain_transcript_prefers_the_audio(store, cfg):
    plain = SuppliedTranscript(text="we discussed the programme and the fee")
    report, deps = execute(store, cfg, request(transcript=plain))
    assert deps.extra["calls"]["transcribe"] == 1
    assert report.transcript.source == tr.SOURCE_DEEPGRAM
    assert report.transcript.timestamps_available is True


# =========================================================================== #
# 5. Multilingual / code-switching
# =========================================================================== #
def test_05_hindi_english_code_switched_call(store, cfg):
    mixed = dg([
        (0, "Namaste Meera ji, main Rajan Kumar bol raha hoon EdTech Pro se."),
        (1, "Haan ji boliye, maine masterclass attend ki thi last month."),
        (0, "Bahut accha. Aapko certification ki zaroorat kis liye hai?"),
        (1, "Mujhe senior role chahiye but recognised certificate nahin hai."),
        (0, "Our Career Accelerator programme mein accredited certificate milta hai."),
    ], detected="multi")
    report, _ = execute(store, cfg, deepgram=mixed)

    assert report.status == sca.STATUS_COMPLETED
    assert report.transcript.language_detected == "multi"
    assert report.transcript.multilingual is True
    # Text is preserved verbatim - never translated or normalised away.
    assert "zaroorat" in report.transcript.segments[2].text
    # Evidence anchored to code-switched text still verifies.
    assert report.stage_evaluations[0].criteria[0].evidence[0].verified is True


def test_05b_devanagari_evidence_verifies(store, cfg):
    devanagari = dg([
        (0, "नमस्ते मीरा जी, मैं राजन कुमार बोल रहा हूँ EdTech Pro से।"),
        (1, "मुझे certification चाहिए senior role के लिए।"),
    ], detected="hi")
    report, _ = execute(store, cfg, deepgram=devanagari)
    assert report.status == sca.STATUS_COMPLETED
    evidence = report.stage_evaluations[0].criteria[0].evidence[0]
    assert evidence.verified is True
    assert "नमस्ते" in evidence.quote


# =========================================================================== #
# 6-7. Call length
# =========================================================================== #
def test_06_very_short_call_scores_only_what_happened(store, cfg):
    short = dg([(0, "Hello, this is Rajan from EdTech Pro."),
                (1, "Sorry, I am driving right now.")])
    later_stages = [c["id"] for s in cfg["stages"] for c in s["criteria"]
                    if s["id"] in ("product_pitching", "objection_handling", "closing")]
    report, _ = execute(store, cfg, deepgram=short, not_applicable=later_stages)

    by_stage = {s.stage_id: s for s in report.scores.stages}
    for stage_id in ("product_pitching", "objection_handling", "closing"):
        assert by_stage[stage_id].score is None            # null, never 0
        assert by_stage[stage_id].status == "not_applicable"
    assert report.scores.overall is not None               # the opening still counts
    assert report.scores.stages_not_applicable == 3
    # The 18 the model marked not-applicable, plus any blocked by an unmet
    # requirement (here `purpose_reference`, since no previous interaction was
    # supplied). Both are absent opportunity, and neither is scored zero.
    assert report.analysis_quality.criteria_not_applicable >= len(later_stages)
    criteria = {c.criterion_id: c for st_ in report.stage_evaluations for c in st_.criteria}
    assert criteria["purpose_reference"].status == "not_applicable"
    assert criteria["purpose_reference"].score is None


def test_07_long_call_keeps_stable_evidence_anchors(store, cfg):
    turns = []
    for i in range(300):
        speaker = i % 2
        turns.append((speaker, f"This is turn number {i} discussing the programme in detail."))
    long_call = dg(turns)
    report, _ = execute(store, cfg, deepgram=long_call)

    assert report.status == sca.STATUS_COMPLETED
    assert report.transcript.segment_count == 300
    assert [s.index for s in report.transcript.segments[:5]] == [0, 1, 2, 3, 4]
    assert report.transcript.segments[-1].index == 299
    assert report.transcript.duration_seconds > 1000
    assert report.scores.overall is not None


def test_07b_prompt_rendering_is_bounded(cfg):
    """A long transcript must not produce an unbounded prompt."""
    turns = [(i % 2, "A long sentence repeated many times to inflate the transcript." * 5)
             for i in range(400)]
    transcript = tr.from_deepgram(dg(turns))
    rendered = tr.render_for_prompt(transcript, max_chars=an.MAX_TRANSCRIPT_CHARS)
    assert len(rendered) <= an.MAX_TRANSCRIPT_CHARS


# =========================================================================== #
# 8. Poor audio
# =========================================================================== #
def test_08_poor_quality_audio_is_flagged_not_silently_scored(store, cfg):
    noisy = dg([(0, "Hello this is Rajan from ... hmm"),
                (1, "Sorry I cannot ... you very well")])
    for utterance in noisy["results"]["utterances"]:
        utterance["confidence"] = 0.29

    report, _ = execute(store, cfg, deepgram=noisy)
    assert report.status == sca.STATUS_COMPLETED
    assert report.analysis_quality.transcript_confidence < 0.4
    assert tr.WARN_LOW_CONFIDENCE in report.analysis_quality.warnings
    assert report.scores.overall is not None    # analysable, but the caller is warned


def test_08b_silent_recording_is_not_a_zero_score(store, cfg):
    silent = {"metadata": {"duration": 4.0}, "results": {"channels": []}}
    report, _ = execute(store, cfg, deepgram=silent, llm=[FakeResponse("{}")])
    assert report.status == sca.STATUS_FAILED
    assert report.availability.reason == sca.TRANSCRIPT_EMPTY
    assert report.scores is None


# =========================================================================== #
# 9-10. Dispositions
# =========================================================================== #
@pytest.mark.parametrize("disposition", ["No Answer", "Invalid number"])
def test_09_10_dispositions_are_not_hard_coded(store, cfg, disposition):
    """No business rule has been configured, so nothing is skipped by default."""
    report, _ = execute(store, cfg,
                        request(call_metadata=CallMetadata(disposition=disposition)))
    assert report.status == sca.STATUS_COMPLETED
    assert report.call.disposition == disposition
    assert report.call.disposition_source == "rep_reported"


@pytest.mark.parametrize("disposition", ["No Answer", "Invalid number"])
def test_09_10b_configured_skip_produces_a_transcript_and_no_llm_spend(
        store, cfg, disposition, monkeypatch):
    configured = copy.deepcopy(cfg)
    configured["disposition_policy"]["skip_full_evaluation"] = ["No Answer", "Invalid number"]
    configured["disposition_policy"]["confirmed"] = True
    monkeypatch.setattr(fw, "load_framework", lambda *a, **k: configured)

    report, deps = execute(store, cfg,
                           request(call_metadata=CallMetadata(disposition=disposition)))
    assert report.status == sca.STATUS_SKIPPED
    assert report.availability.reason == sca.DISPOSITION_EXCLUDED
    assert report.scores is None
    assert report.transcript.segment_count > 0
    assert deps.llm_client.models.calls == 0


# =========================================================================== #
# 11-13. Objections and buying intent
# =========================================================================== #
def test_11_strong_objections_are_reported_with_evidence(store, cfg):
    preview = tr.from_deepgram(TWO_SPEAKER)
    extra = {"objections": [
        {"summary": "The fee is higher than expected", "category": "price",
         "raised_by": "speaker_1", "handled": "partially_resolved",
         "evidence": [{"segment_index": 5, "quote": quote_from(preview, 5)}]},
        {"summary": "No recognised certification yet", "category": "fit",
         "raised_by": "speaker_1", "handled": "resolved",
         "evidence": [{"segment_index": 3, "quote": quote_from(preview, 3)}]},
    ]}
    report, _ = execute(store, cfg, extra=extra)

    assert len(report.objections) == 2
    price = next(o for o in report.objections if o.category == "price")
    assert price.handled == "partially_resolved"
    assert price.evidence[0].verified is True
    assert price.evidence[0].start is not None


def test_12_strong_buying_signals_pass_through(store, cfg):
    preview = tr.from_deepgram(TWO_SPEAKER)
    extra = {"buying_signals": [
        {"type": "buying_intent", "summary": "Asked for the payment link",
         "strength": "strong", "speaker_id": "speaker_1",
         "evidence": [{"segment_index": 7, "quote": quote_from(preview, 7)}]},
        {"type": "commitment", "summary": "Committed to paying this evening",
         "strength": "strong",
         "evidence": [{"segment_index": 7, "quote": quote_from(preview, 7, 8)}]},
    ]}
    report, _ = execute(store, cfg, extra=extra)
    assert {s.type for s in report.buying_signals} == {"buying_intent", "commitment"}
    assert report.buying_signals[0].label == "Buying intent"     # from the vocabulary
    assert all(s.evidence[0].verified for s in report.buying_signals)


def test_13_weak_intent_does_not_change_the_arithmetic(store, cfg):
    """Signals are reported, not scored. Two calls with identical ratings score
    identically whatever their signals - which is what stops an unvalidated
    behavioural taxonomy leaking into a number."""
    preview = tr.from_deepgram(TWO_SPEAKER)
    strong = {"buying_signals": [
        {"type": "buying_intent", "summary": "Asked for the link", "strength": "strong",
         "evidence": [{"segment_index": 7, "quote": quote_from(preview, 7)}]}]}
    weak = {"customer_signals": [
        {"type": "hesitation", "summary": "Wants to think about it", "strength": "weak",
         "evidence": [{"segment_index": 5, "quote": quote_from(preview, 5)}]}]}

    hot, _ = execute(store, cfg, request(call_id="hot"), extra=strong)
    cold, _ = execute(store, cfg, request(call_id="cold"), extra=weak)

    assert hot.scores.overall == cold.scores.overall
    assert len(hot.buying_signals) == 1 and cold.buying_signals == []
    assert cold.customer_signals[0].type == "hesitation"


# =========================================================================== #
# 14-16. Context variation
# =========================================================================== #
@pytest.mark.parametrize("region", ["Maharashtra", "Tamil Nadu", "Greater Manchester", None])
def test_14_region_is_context_never_an_arithmetic_change(store, cfg, region):
    """Region informs interpretation. It must not move a score by itself."""
    report, _ = execute(store, cfg, request(
        call_id=f"call_{region}", customer=CustomerInfo(name="Meera Patel", region=region)))
    assert report.scores.overall_100 == 67    # identical ratings, identical score
    assert report.context_used.customer["region"] == region


def test_14b_region_reaches_the_prompt_without_a_stereotype(store, cfg):
    from sales_call_analyzer import context as ctxmod
    ctx = ctxmod.build_context(
        request(customer=CustomerInfo(name="Meera Patel", region="Maharashtra")),
        None, None)
    prompt = ctxmod.render_for_prompt(ctx, tr.from_deepgram(TWO_SPEAKER))
    assert "Maharashtra" in prompt
    assert "do not apply generalisations" in prompt


@pytest.mark.parametrize("product,expected_blocked", [
    (ProductInfo(name="Starter Toolkit", price=2000, currency="INR",
                 is_structured_programme=False, sold_on_call=True), True),
    (ProductInfo(name="Career Accelerator", price=200000, currency="INR",
                 is_structured_programme=True, sold_on_call=False), False),
])
def test_15_product_shape_changes_applicability_not_weights(store, cfg, product,
                                                            expected_blocked):
    """A 2,000 rupee toolkit and a 2 lakh programme are judged on different
    criteria - but the weighting is identical, because no per-product business
    rule has been configured."""
    report, _ = execute(store, cfg, request(call_id=f"call_{product.price}",
                                            product=product))
    criteria = {c.criterion_id: c for s in report.stage_evaluations for c in s.criteria}
    assert (criteria["pitch_structure_explained"].status == "not_applicable") is expected_blocked
    assert report.scores.weighting == fw.WEIGHTING_EQUAL
    assert report.context_used.product["price"] == product.price


def test_15b_price_and_complexity_reach_the_prompt(store, cfg):
    from sales_call_analyzer import context as ctxmod
    text = ctxmod.render_product_block(
        ProductInfo(name="Career Accelerator", price=200000, currency="INR",
                    complexity="high", price_band="high-ticket"))
    assert "INR 200,000" in text
    assert "high-ticket" in text


@pytest.mark.parametrize("profile", [
    CustomerInfo(name="Meera Patel", designation="Operations Manager",
                 industry="Logistics", awareness_level="problem-aware"),
    CustomerInfo(name="Meera Patel", designation="Founder",
                 industry="SaaS", awareness_level="most-aware"),
    CustomerInfo(name="Meera Patel"),
])
def test_16_customer_profile_is_reported_and_never_rescores(store, cfg, profile):
    report, _ = execute(store, cfg, request(
        call_id=f"call_{profile.designation}", customer=profile))
    assert report.scores.overall_100 == 67
    assert report.context_used.customer["designation"] == profile.designation


def test_16b_a_thin_profile_is_declared_not_invented(store, cfg):
    """With no CRM names, the brand name still identifies the representative
    ("...from EdTech Pro"), which is legitimate evidence. What must NOT happen is
    a name appearing from nowhere."""
    report, _ = execute(store, cfg, request(customer=CustomerInfo(), rep=RepInfo()))
    assert sca.NO_CUSTOMER_CONTEXT in report.context_used.missing
    assert report.context_used.customer["name_known"] is False

    rep = next(s for s in report.transcript.speakers if s.role == sca.ROLE_SALES_REP)
    assert rep.role_basis == sca.ROLE_BASIS_REP_SELF_INTRO
    assert rep.role_confidence == "medium"        # the brand name, not a person's name
    # Nothing may be named when the CRM supplied no names.
    assert all(s.name is None for s in report.transcript.speakers)


def test_16c_with_no_identifying_evidence_every_role_stays_unknown(store, cfg):
    anonymous = dg([
        (0, "Hello, thanks for taking my call today."),
        (1, "No problem, what is this regarding?"),
        (0, "I wanted to discuss the programme you enquired about."),
    ])
    report, _ = execute(store, cfg, request(customer=CustomerInfo(), rep=RepInfo()),
                        deepgram=anonymous)
    assert all(s.role == sca.ROLE_UNKNOWN for s in report.transcript.speakers)
    assert all(s.role_basis == sca.ROLE_BASIS_UNRESOLVED
               for s in report.transcript.speakers)
    assert all(s.name is None for s in report.transcript.speakers)


# =========================================================================== #
# 17-18. Missing context
# =========================================================================== #
def test_17_missing_brand_brain_is_survivable(store, cfg):
    report, _ = execute(store, cfg, brand_brain=None)
    assert report.status == sca.STATUS_COMPLETED
    assert report.context_used.brand_brain["available"] is False
    assert sca.NO_BRAND_BRAIN in report.context_used.missing
    assert report.scores.overall is not None


def test_18_missing_customer_and_product_context_is_survivable(store, cfg):
    report, _ = execute(store, cfg, request(customer=CustomerInfo(),
                                            product=ProductInfo()))
    assert report.status == sca.STATUS_COMPLETED
    assert sca.NO_CUSTOMER_CONTEXT in report.context_used.missing
    assert sca.NO_PRODUCT_CONTEXT in report.context_used.missing


# =========================================================================== #
# 19-22. Duplicates and failures
# =========================================================================== #
def test_19_duplicate_request_is_detected_by_fingerprint(store, cfg):
    kwargs = dict(framework_version=cfg["framework_version"], prompt_version="p1",
                  llm_model="m", transcription_model="t")
    first = st.compute_fingerprint(request(), **kwargs)
    assert first == st.compute_fingerprint(request(), **kwargs)

    execute(store, cfg)
    existing = run(store.find_by_fingerprint(
        list(store.collection.docs.values())[0]["input_fingerprint"]))
    assert st.reusable(existing, force=False) is True


def test_20_gemini_failure_keeps_the_transcript(store, cfg):
    report, deps = execute(store, cfg,
                           llm=[FakeResponse("not json"), FakeResponse("still not json")])
    assert report.status == sca.STATUS_FAILED
    assert report.availability.reason == sca.ANALYSIS_FAILED_AFTER_TRANSCRIPTION
    assert report.scores is None
    assert report.transcript.segment_count == 8
    assert deps.extra["calls"]["transcribe"] == 1


def test_21_deepgram_failure_is_reported_with_its_reason(store, cfg):
    async def failing(url, **kwargs):
        raise DeepgramError(sca.TRANSCRIPTION_TIMEOUT, "timed out", retryable=True)

    report, _ = execute(store, cfg, transcribe=failing, llm=[FakeResponse("{}")])
    assert report.status == sca.STATUS_FAILED
    assert report.availability.reason == sca.TRANSCRIPTION_TIMEOUT
    assert report.scores is None


def test_22_partial_failure_lets_the_retry_skip_transcription(store, cfg):
    """The single biggest cost control: transcription is paid for once, ever."""
    failed, first_deps = execute(store, cfg,
                                 llm=[FakeResponse("bad"), FakeResponse("bad")])
    assert failed.status == sca.STATUS_FAILED
    assert first_deps.extra["calls"]["transcribe"] == 1

    retried, second_deps = execute(store, cfg)
    assert retried.status == sca.STATUS_COMPLETED
    assert second_deps.extra["calls"]["transcribe"] == 0
    assert retried.scores.overall is not None
