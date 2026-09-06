"""
Orchestration - the job body, start to finish.

    input -> transcript -> speakers -> context -> LLM -> evidence -> scoring
          -> report -> persisted

DESIGN NOTES
    * The transcript is persisted the moment it exists, before the LLM runs. A
      failure after that point never re-transcribes on retry, which is the
      single biggest cost control in the feature.
    * Every failure ends as a stated reason with scores=null. A failed analysis
      never becomes a neutral scorecard, because a sales manager cannot tell an
      invented evaluation from a real one.
    * A concurrency gate bounds how many calls are in flight at once. This runs
      in a single pm2 fork process alongside onboarding and Script Lab; two
      simultaneous 40-minute transcriptions must not starve them.
    * All dependencies are injected. The package opens no connection and reads
      no global client, so the whole pipeline is testable without a network.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from . import (
    ANALYSIS_FAILED_AFTER_TRANSCRIPTION,
    ANALYSIS_INVALID_OUTPUT,
    NO_AUDIO_OR_TRANSCRIPT,
    REASON_TEXT,
    STATUS_ANALYZING,
    STATUS_COMPLETED,
    STATUS_SCORING,
    STATUS_SKIPPED,
    STATUS_TRANSCRIBING,
    TRANSCRIPT_EMPTY,
    TRANSCRIPTION_NOT_CONFIGURED,
)
from . import analyzer as analyzer_mod
from . import context as context_mod
from . import evidence as evidence_mod
from . import framework as fw
from . import report as report_mod
from . import scoring as scoring_mod
from . import speakers as speakers_mod
from . import transcript as transcript_mod
from .deepgram_client import DeepgramError
from .models import (
    AnalyzeRequest,
    Availability,
    CallSummaryBlock,
    NormalizedTranscript,
    ProcessingInfo,
    SalesCallAnalysis,
)
from .store import AnalysisStore

logger = logging.getLogger("sales_call_analyzer.pipeline")

MAX_CONCURRENT_JOBS = int(os.environ.get("SCA_MAX_CONCURRENT_JOBS", 2))
_GATE: Optional[asyncio.Semaphore] = None


def gate() -> asyncio.Semaphore:
    """Lazily created so the semaphore binds to the running event loop."""
    global _GATE
    if _GATE is None:
        _GATE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    return _GATE


@dataclass
class PipelineDeps:
    """Everything the pipeline needs from the outside world.

    Injected rather than imported so the whole run is testable offline and so
    this package keeps its rule of owning no connections.
    """
    store: AnalysisStore
    llm_client: Any
    llm_model: str
    transcribe: Callable[..., Awaitable[dict]]
    load_brand_brain: Callable[[Optional[str]], Awaitable[Optional[dict]]] = None
    resolve_brand_ref: Callable[[str], Awaitable[dict]] = None
    transcription_model: str = "unknown"
    store_raw_transcript: Optional[bool] = None
    extra: dict = field(default_factory=dict)


class PipelineFailure(Exception):
    """A stated, reportable failure. Carries the stable reason code."""

    def __init__(self, reason: str, message: Optional[str] = None,
                 transcript: Optional[NormalizedTranscript] = None):
        super().__init__(message or REASON_TEXT.get(reason, reason))
        self.reason = reason
        self.message = message or REASON_TEXT.get(reason, reason)
        self.transcript = transcript


# =========================================================================== #
# Transcription
# =========================================================================== #
async def resolve_transcript(request: AnalyzeRequest, deps: PipelineDeps,
                             analysis_id: str,
                             stored: Optional[dict] = None) -> tuple[NormalizedTranscript, str]:
    """Produce the normalised transcript. Returns (transcript, strategy).

    A transcript already produced for this call short-circuits everything: an
    earlier attempt that died at the LLM step has already paid for it.
    """
    if stored:
        logger.info("reusing stored transcript [analysis_id=%s]", analysis_id)
        # Reuse the transcription, never the role resolution: the CRM names in
        # THIS request may differ from the ones the earlier run had.
        return (transcript_mod.clear_role_annotations(NormalizedTranscript(**stored)),
                "reused_stored_transcript")

    strategy, _why = transcript_mod.select_input_strategy(request.audio, request.transcript)

    if strategy == transcript_mod.STRATEGY_NONE:
        raise PipelineFailure(NO_AUDIO_OR_TRANSCRIPT)

    if strategy == transcript_mod.STRATEGY_USE_SUPPLIED_STRUCTURED:
        return transcript_mod.from_supplied_structured(request.transcript), strategy

    if strategy == transcript_mod.STRATEGY_USE_SUPPLIED_TEXT:
        return transcript_mod.from_supplied_text(request.transcript), strategy

    # Audio. If transcription fails but a plain-text transcript was also
    # supplied, degrade to it rather than losing the call entirely.
    await deps.store.set_status(analysis_id, STATUS_TRANSCRIBING)
    language = request.options.language_hint
    if not language and request.transcript:
        language = request.transcript.language
    try:
        raw = await deps.transcribe(request.audio.url,
                                    mime_type=request.audio.mime_type,
                                    language=language)
    except DeepgramError as err:
        if transcript_mod.has_text(request.transcript):
            logger.warning("transcription failed (%s); falling back to the supplied "
                           "text transcript [analysis_id=%s]", err.reason, analysis_id)
            return (transcript_mod.from_supplied_text(request.transcript),
                    "supplied_text_after_transcription_failure")
        raise PipelineFailure(err.reason, err.message) from err

    await deps.store.save_raw_transcript(analysis_id, raw,
                                         enabled=(request.options.store_raw_transcript
                                                  if request.options.store_raw_transcript is not None
                                                  else deps.store_raw_transcript))
    return transcript_mod.from_deepgram(
        raw, language_hint=request.options.language_hint), strategy


# =========================================================================== #
# The run
# =========================================================================== #
async def run_analysis(analysis_id: str, request: AnalyzeRequest,
                       deps: PipelineDeps,
                       created_at: Optional[datetime] = None) -> SalesCallAnalysis:
    """Execute one analysis end to end and persist the outcome.

    Never raises for an expected failure: the failure is persisted with its
    reason and returned as a report with scores=null. Unexpected exceptions are
    caught, logged and recorded the same way, because a job that dies silently
    leaves a row stuck in 'analyzing' forever.
    """
    async with gate():
        return await _run(analysis_id, request, deps, created_at)


async def _run(analysis_id: str, request: AnalyzeRequest, deps: PipelineDeps,
               created_at: Optional[datetime]) -> SalesCallAnalysis:
    started = time.monotonic()
    processing = ProcessingInfo(
        transcription_provider="deepgram",
        transcription_model=deps.transcription_model,
        llm_model=deps.llm_model,
        prompt_version=analyzer_mod.PROMPT_VERSION,
        started_at=datetime.now(timezone.utc),
    )
    cfg = fw.load_framework()
    signals = fw.load_signals()
    processing.framework_version = cfg["framework_version"]
    processing.signals_version = signals["signals_version"]

    transcript: Optional[NormalizedTranscript] = None
    ctx = None
    context_used = None

    await deps.store.increment_attempts(analysis_id)

    try:
        # ---- 1. transcript ------------------------------------------------
        stored = await deps.store.stored_transcript(request.call_id)
        t0 = time.monotonic()
        transcript, strategy = await resolve_transcript(request, deps, analysis_id, stored)
        if strategy != "reused_stored_transcript":
            processing.transcription_ms = int((time.monotonic() - t0) * 1000)
            processing.audio_seconds_submitted = (
                transcript.duration_seconds
                if strategy == transcript_mod.STRATEGY_TRANSCRIBE_AUDIO else None)

        if not transcript.segments:
            raise PipelineFailure(TRANSCRIPT_EMPTY, transcript=transcript)

        # ---- 2. speakers ---------------------------------------------------
        brand_ref = {}
        if deps.resolve_brand_ref and request.lead_id:
            try:
                brand_ref = await deps.resolve_brand_ref(request.lead_id) or {}
            except Exception as err:  # noqa: BLE001 - context is never fatal
                logger.warning("brand ref lookup failed [analysis_id=%s]: %s",
                               analysis_id, type(err).__name__)

        transcript = speakers_mod.resolve_roles(
            transcript, rep=request.rep, customer=request.customer,
            brand_name=brand_ref.get("brand_name"))

        # Persisted BEFORE the LLM runs, so a later failure never re-transcribes.
        await deps.store.save_transcript(analysis_id, transcript.model_dump(mode="json"))

        # ---- 3. context ----------------------------------------------------
        brand_brain = None
        if deps.load_brand_brain:
            try:
                brand_brain = await deps.load_brand_brain(
                    request.brand_brain_id or brand_ref.get("brand_brain_id"))
            except Exception as err:  # noqa: BLE001 - a missing Brand Brain is not fatal
                logger.warning("brand brain load failed [analysis_id=%s]: %s",
                               analysis_id, type(err).__name__)

        ctx = context_mod.build_context(request, brand_brain, brand_ref)
        satisfied = context_mod.satisfied_requirements(ctx, transcript)
        blocked = fw.criteria_blocked_by_requirements(cfg, satisfied)
        unmet = fw.unmet_requirements(cfg, satisfied)
        context_used = context_mod.to_context_used(ctx, transcript, unmet)

        # ---- 4. policy: is this call to be evaluated at all? ----------------
        skip, skip_reason = fw.skip_decision(
            cfg, request.call_metadata.disposition,
            request.call_metadata.duration_seconds or transcript.duration_seconds,
            transcript.segment_count)
        if skip:
            return await _finish_skipped(analysis_id, ctx, transcript, context_used,
                                         processing, skip_reason, deps, created_at, started)

        # ---- 5. the model --------------------------------------------------
        await deps.store.set_status(analysis_id, STATUS_ANALYZING)
        try:
            outcome = await analyzer_mod.analyze(
                deps.llm_client, deps.llm_model, ctx, transcript, cfg, signals, blocked)
        except analyzer_mod.AnalyzerError as err:
            # The transcript survived and is already stored; say so, so a retry
            # is known to be cheap.
            raise PipelineFailure(
                ANALYSIS_FAILED_AFTER_TRANSCRIPTION
                if err.reason == ANALYSIS_INVALID_OUTPUT else err.reason,
                err.message, transcript=transcript) from err

        for key, value in (outcome["meta"] or {}).items():
            if hasattr(processing, key):
                setattr(processing, key, value)

        # ---- 6. evidence, then scoring -------------------------------------
        await deps.store.set_status(analysis_id, STATUS_SCORING)
        verified, evidence_stats = evidence_mod.verify_analysis(
            outcome["analysis"], transcript, cfg)
        scoring = scoring_mod.score_analysis(verified, cfg, blocked)

        # ---- 7. report ------------------------------------------------------
        processing.total_ms = int((time.monotonic() - started) * 1000)
        processing.completed_at = datetime.now(timezone.utc)
        report = report_mod.build_report(
            analysis_id=analysis_id, ctx=ctx, transcript=transcript,
            analysis=verified, scoring=scoring, signals=signals,
            context_used=context_used, evidence_stats=evidence_stats,
            processing=processing, lead_id=request.lead_id,
            brand_id=brand_ref.get("brand_id") or request.brand_id,
            created_at=created_at, status=STATUS_COMPLETED,
            degraded=_is_degraded(transcript, strategy))

        await deps.store.complete(
            analysis_id,
            report=report.model_dump(mode="json"),
            analysis=verified,
            blocked=blocked,
            processing=processing.model_dump(mode="json"))
        logger.info("analysis complete [analysis_id=%s call_id=%s total_ms=%s "
                    "criteria_scored=%s]", analysis_id, request.call_id,
                    processing.total_ms, scoring["counts"]["criteria_scored"])
        return report

    except PipelineFailure as err:
        return await _finish_failed(analysis_id, request, err.reason, err.message,
                                    err.transcript or transcript, ctx, context_used,
                                    processing, deps, created_at, started)
    except Exception as err:  # noqa: BLE001 - a dead job must never stay "analyzing"
        logger.exception("analysis crashed [analysis_id=%s]", analysis_id)
        return await _finish_failed(analysis_id, request, "analysis_provider_error",
                                    f"The analysis did not complete: {type(err).__name__}.",
                                    transcript, ctx, context_used, processing, deps,
                                    created_at, started)


def _is_degraded(transcript: NormalizedTranscript, strategy: str) -> bool:
    """Degraded means the analysis ran on less than it should have had."""
    return (not transcript.diarization_available
            or not transcript.timestamps_available
            or strategy == "supplied_text_after_transcription_failure")


async def _finish_skipped(analysis_id, ctx, transcript, context_used, processing,
                          reason, deps: PipelineDeps, created_at, started):
    """Configured policy says this call is not evaluated. Not a failure: the
    transcript is real and is kept, there is simply no scorecard."""
    processing.total_ms = int((time.monotonic() - started) * 1000)
    report = report_mod.build_failed_report(
        analysis_id=analysis_id, call_id=ctx.call_id, reason=reason,
        status=STATUS_SKIPPED, lead_id=ctx.lead_id, transcript=transcript,
        context_used=context_used, processing=processing, created_at=created_at)
    await deps.store.complete(analysis_id, report=report.model_dump(mode="json"),
                              analysis={}, processing=processing.model_dump(mode="json"),
                              status=STATUS_SKIPPED)
    await deps.store.set_status(analysis_id, STATUS_SKIPPED, reason=reason,
                                message=REASON_TEXT.get(reason, reason))
    logger.info("analysis skipped by policy [analysis_id=%s reason=%s]", analysis_id, reason)
    return report


async def _finish_failed(analysis_id, request, reason, message, transcript, ctx,
                         context_used, processing, deps: PipelineDeps, created_at, started):
    processing.total_ms = int((time.monotonic() - started) * 1000)
    # Keep the call card on a failure where we got far enough to assemble the
    # context: the UI can still show who was called and what the rep recorded.
    call_block = None
    if ctx is not None:
        call_block = CallSummaryBlock(
            call_id=ctx.call_id, disposition=ctx.call.disposition,
            direction=ctx.call.direction, occurred_at=ctx.call.occurred_at,
            duration_seconds=ctx.call.duration_seconds,
            remarks=ctx.call.remarks, provider=ctx.call.provider,
            recording_reference=ctx.call.recording_reference,
            rep=ctx.rep, customer=ctx.customer)
    report = report_mod.build_failed_report(
        analysis_id=analysis_id, call_id=request.call_id, reason=reason,
        message=message, lead_id=request.lead_id, transcript=transcript,
        context_used=context_used, processing=processing, created_at=created_at,
        call=call_block)
    await deps.store.fail(analysis_id, reason=reason, message=message,
                          report=report.model_dump(mode="json"),
                          processing=processing.model_dump(mode="json"))
    logger.warning("analysis failed [analysis_id=%s call_id=%s reason=%s transcript_kept=%s]",
                   analysis_id, request.call_id, reason, transcript is not None)
    return report


def unavailable(reason: str, message: Optional[str] = None) -> Availability:
    return Availability(available=False, reason=reason,
                        message=message or REASON_TEXT.get(reason, reason))


def transcription_unavailable() -> Availability:
    return unavailable(TRANSCRIPTION_NOT_CONFIGURED)
