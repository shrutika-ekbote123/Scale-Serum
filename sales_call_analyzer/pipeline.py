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
    * Every exit - completed, skipped, failed - carries an estimated cost block
      (`processing.cost`). A failed analysis still cost money, and a pricing bug
      must never fail an analysis, so the estimate is best-effort.
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

import billing
from transcription import audio_slice as _audio_slice
from transcription import deepgram_client as _dg_client
from transcription import language_id as _language_id

from . import (
    ANALYSIS_FAILED_AFTER_TRANSCRIPTION,
    LANGUAGE_BASIS_DEFAULT,
    LANGUAGE_BASIS_DETECTED,
    LANGUAGE_BASIS_SUPPLIED,
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

STRATEGY_REUSED = "reused_stored_transcript"
STRATEGY_FALLBACK_TEXT = "supplied_text_after_transcription_failure"

# Transcription failures that happen before Deepgram does any billable work.
# Any other failure while transcribing may or may not have been billed.
_TRANSCRIPTION_NOT_BILLED = {
    NO_AUDIO_OR_TRANSCRIPT,
    TRANSCRIPTION_NOT_CONFIGURED,
    _dg_client.TRANSCRIPTION_RATE_LIMITED,
    _dg_client.AUDIO_TOO_LARGE,
}


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
    # Language identification. Injected, so a test needs neither Gemini nor
    # ffmpeg, and an unconfigured server simply skips the step.
    identify_language: Optional[Callable[..., Awaitable[Any]]] = None
    fetch_audio: Optional[Callable[[str], Awaitable[tuple]]] = None
    language_id_mode: Optional[str] = None
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
# How much say the language detector gets.
#
#   off      never runs. The default until it has been watched on live calls.
#   shadow   runs and records what it WOULD have chosen, changes nothing. This
#            is how we find out whether it is right without risking a report.
#   on       acts on the answer.
#
# Default off on purpose, like VL_ENABLED: merging a feature that costs money
# per call must land inert, and be switched on deliberately.
LANGUAGE_ID_OFF = "off"
LANGUAGE_ID_SHADOW = "shadow"
LANGUAGE_ID_ON = "on"
LANGUAGE_ID_MODE = os.environ.get("SCA_LANGUAGE_ID", LANGUAGE_ID_OFF).strip().lower()

# Without ffmpeg the whole recording is sent instead of a window, so it is only
# worth doing for a short file - 8 MB is roughly 8 minutes of speech MP3.
LANGUAGE_ID_MAX_WHOLE_BYTES = int(os.environ.get("SCA_LANGUAGE_ID_MAX_WHOLE_BYTES", 8 * 1024 * 1024))

REASON_LANGUAGE_ID_AUDIO_UNAVAILABLE = "language_id_audio_unavailable"
REASON_LANGUAGE_ID_FILE_TOO_LONG = "language_id_file_too_long_for_whole_file_check"


def _requested_language(request: AnalyzeRequest) -> Optional[str]:
    """The language this request asks Deepgram for, before server defaults."""
    language = request.options.language_hint
    if not language and request.transcript:
        language = request.transcript.language
    return language


async def _decide_language(request: AnalyzeRequest, deps: PipelineDeps,
                           processing: ProcessingInfo,
                           analysis_id: str) -> Optional[str]:
    """Which language to ask Deepgram for. None means the server default.

    Order: what the caller told us, then what the audio sounds like, then the
    default. The caller wins because a rep who has just had the conversation is
    a better source than any detector.

    Never raises. Every failure here leaves the default in place, because a
    wrong-but-complete transcript beats a failed analysis.
    """
    hint = _requested_language(request)
    if hint:
        processing.language_basis = LANGUAGE_BASIS_SUPPLIED
        return hint

    processing.language_basis = LANGUAGE_BASIS_DEFAULT
    mode = (deps.language_id_mode or LANGUAGE_ID_MODE or LANGUAGE_ID_OFF).strip().lower()
    if mode == LANGUAGE_ID_OFF or not deps.identify_language or not deps.fetch_audio:
        return None

    try:
        audio, content_type = await deps.fetch_audio(request.audio.url)
    except Exception as err:  # noqa: BLE001 - Deepgram may still reach the URL itself
        logger.warning("language identification could not fetch the audio "
                       "[analysis_id=%s]: %s", analysis_id, type(err).__name__)
        processing.language_detection_reason = REASON_LANGUAGE_ID_AUDIO_UNAVAILABLE
        return None

    mime = request.audio.mime_type or content_type or "audio/mpeg"
    sample = await _audio_slice.window(audio)
    sample_mime = "audio/mpeg"
    if not sample:
        # No ffmpeg. The whole file answers the same question, but it is priced
        # by length, so only a short recording is worth sending.
        if len(audio) > LANGUAGE_ID_MAX_WHOLE_BYTES:
            processing.language_detection_reason = REASON_LANGUAGE_ID_FILE_TOO_LONG
            return None
        sample, sample_mime = audio, mime

    decision = await deps.identify_language(sample, mime_type=sample_mime)
    _merge_meta(processing, decision.as_meta())
    if not decision.ok:
        return None

    language, why = _dg_client.language_for(decision.dominant_non_english)
    processing.language_decision = why
    if language is None:
        return None

    if mode != LANGUAGE_ID_ON:
        # Shadow: record the choice, change nothing.
        processing.language_shadow_choice = language
        logger.info("language id would have used %s [analysis_id=%s, mode=shadow]",
                    language, analysis_id)
        return None

    processing.language_basis = LANGUAGE_BASIS_DETECTED
    logger.info("language id chose %s [analysis_id=%s]", language, analysis_id)
    return language


async def resolve_transcript(request: AnalyzeRequest, deps: PipelineDeps,
                             analysis_id: str,
                             stored: Optional[dict] = None,
                             processing: Optional[ProcessingInfo] = None,
                             ) -> tuple[NormalizedTranscript, str]:
    """Produce the normalised transcript. Returns (transcript, strategy).

    A transcript already produced for this call short-circuits everything: an
    earlier attempt that died at the LLM step has already paid for it.
    """
    if stored:
        logger.info("reusing stored transcript [analysis_id=%s]", analysis_id)
        # Reuse the transcription, never the role resolution: the CRM names in
        # THIS request may differ from the ones the earlier run had.
        return (transcript_mod.clear_role_annotations(NormalizedTranscript(**stored)),
                STRATEGY_REUSED)

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
    processing = processing if processing is not None else ProcessingInfo()
    language = await _decide_language(request, deps, processing, analysis_id)
    # What was actually sent, which is what billing prices: multi is charged at
    # the multilingual rate.
    processing.transcription_language_sent = _dg_client.effective_language(language)
    try:
        raw = await deps.transcribe(request.audio.url,
                                    mime_type=request.audio.mime_type,
                                    language=language)
    except DeepgramError as err:
        if transcript_mod.has_text(request.transcript):
            logger.warning("transcription failed (%s); falling back to the supplied "
                           "text transcript [analysis_id=%s]", err.reason, analysis_id)
            return (transcript_mod.from_supplied_text(request.transcript),
                    STRATEGY_FALLBACK_TEXT)
        raise PipelineFailure(err.reason, err.message) from err

    await deps.store.save_raw_transcript(analysis_id, raw,
                                         enabled=(request.options.store_raw_transcript
                                                  if request.options.store_raw_transcript is not None
                                                  else deps.store_raw_transcript))
    return transcript_mod.from_deepgram(
        raw, language_hint=processing.transcription_language_sent), strategy


# =========================================================================== #
# Cost
# =========================================================================== #
def _merge_meta(processing: ProcessingInfo, meta: Optional[dict]) -> None:
    for key, value in (meta or {}).items():
        if hasattr(processing, key):
            setattr(processing, key, value)


async def _prior_costs(store: AnalysisStore, analysis_id: str) -> list:
    """A re-run of the same analysis_id must not erase what the earlier run spent."""
    try:
        doc = await store.get(analysis_id)
    except Exception:  # noqa: BLE001 - cost history is never worth failing a run
        return []
    prior = (doc or {}).get("processing") or {}
    runs = list(prior.get("cost_prior_runs") or [])
    if prior.get("cost"):
        runs.append(prior["cost"])
    out = []
    for raw in runs:
        try:
            out.append(billing.CostBreakdown(**raw))
        except Exception:  # noqa: BLE001
            logger.warning("skipping unreadable prior cost block [analysis_id=%s]", analysis_id)
    return out


def _transcription_outcome(processing: ProcessingInfo,
                           failure_reason: Optional[str]) -> tuple[str, Optional[str]]:
    """Was Deepgram billed for this run? (outcome, reason)."""
    strategy = processing.transcript_strategy
    if strategy == transcript_mod.STRATEGY_TRANSCRIBE_AUDIO:
        return billing.CHARGED, None
    if strategy == STRATEGY_REUSED:
        return billing.NOT_CHARGED, billing.REASON_REUSED_TRANSCRIPT
    if strategy in (transcript_mod.STRATEGY_USE_SUPPLIED_STRUCTURED,
                    transcript_mod.STRATEGY_USE_SUPPLIED_TEXT):
        return billing.NOT_CHARGED, billing.REASON_SUPPLIED_TRANSCRIPT
    if strategy == STRATEGY_FALLBACK_TEXT:
        return billing.CHARGE_UNKNOWN, billing.REASON_TRANSCRIPTION_FAILED_UNKNOWN
    if strategy is None and failure_reason and failure_reason not in _TRANSCRIPTION_NOT_BILLED:
        # Deepgram was called and errored. A timeout or provider error can still
        # be billed on their side, so this is unknown rather than zero.
        return billing.CHARGE_UNKNOWN, billing.REASON_TRANSCRIPTION_FAILED_UNKNOWN
    return billing.NOT_CHARGED, billing.REASON_NO_TRANSCRIPTION


def _attach_cost(processing: ProcessingInfo, failure_reason: Optional[str] = None) -> None:
    """Estimated provider cost for this run. Best-effort: never fails the analysis."""
    try:
        pricing = billing.load_pricing()
        outcome, reason = _transcription_outcome(processing, failure_reason)
        at = processing.started_at
        deepgram = billing.deepgram_cost(
            pricing, outcome=outcome, reason=reason,
            model=processing.transcription_model,
            audio_seconds=processing.audio_seconds_submitted,
            channels_processed=processing.channels_processed,
            multichannel_requested=processing.multichannel_requested,
            language_sent=processing.transcription_language_sent, at=at)
        # Language identification is a Gemini call on the same model, so it is
        # folded into the same block rather than going quietly unbilled. The
        # per-step token counts stay on `processing` for anyone checking.
        def _total(analysis: Optional[int], language: Optional[int]) -> Optional[int]:
            present = [v for v in (analysis, language) if v is not None]
            return sum(present) if present else None

        gemini = billing.gemini_cost(
            pricing, model_requested=processing.llm_model,
            model_version=processing.llm_model_version,
            attempts=processing.llm_attempts + (1 if processing.language_detection_ms else 0),
            input_tokens=_total(processing.llm_input_tokens,
                                processing.language_id_input_tokens),
            output_tokens=_total(processing.llm_output_tokens,
                                 processing.language_id_output_tokens),
            thinking_tokens=_total(processing.llm_thinking_tokens,
                                   processing.language_id_thinking_tokens),
            cached_tokens=_total(processing.llm_cached_tokens,
                                 processing.language_id_cached_tokens), at=at)
        processing.billed_channels = deepgram.billed_channels
        processing.cost = billing.combine(pricing, deepgram, gemini, at=at)
    except Exception:  # noqa: BLE001 - a pricing bug must never fail an analysis
        logger.exception("cost estimate failed; the analysis continues without one")
        processing.cost = None


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

    processing.cost_prior_runs = await _prior_costs(deps.store, analysis_id)
    await deps.store.increment_attempts(analysis_id)

    try:
        # ---- 1. transcript ------------------------------------------------
        stored = await deps.store.stored_transcript(request.call_id)
        t0 = time.monotonic()
        transcript, strategy = await resolve_transcript(request, deps, analysis_id, stored,
                                                        processing)
        processing.transcript_strategy = strategy
        if strategy != STRATEGY_REUSED:
            processing.transcription_ms = int((time.monotonic() - t0) * 1000)
            processing.audio_seconds_submitted = (
                transcript.duration_seconds
                if strategy == transcript_mod.STRATEGY_TRANSCRIBE_AUDIO else None)
        if strategy == transcript_mod.STRATEGY_TRANSCRIBE_AUDIO:
            processing.channels_processed = transcript.channels_processed
            processing.multichannel_requested = _dg_client.MULTICHANNEL

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
            # Rejected and failed attempts were still billed; keep their usage.
            _merge_meta(processing, err.meta)
            # The transcript survived and is already stored; say so, so a retry
            # is known to be cheap.
            raise PipelineFailure(
                ANALYSIS_FAILED_AFTER_TRANSCRIPTION
                if err.reason == ANALYSIS_INVALID_OUTPUT else err.reason,
                err.message, transcript=transcript) from err

        _merge_meta(processing, outcome["meta"])

        # ---- 6. evidence, then scoring -------------------------------------
        await deps.store.set_status(analysis_id, STATUS_SCORING)
        verified, evidence_stats = evidence_mod.verify_analysis(
            outcome["analysis"], transcript, cfg)
        scoring = scoring_mod.score_analysis(verified, cfg, blocked)

        # ---- 7. report ------------------------------------------------------
        processing.total_ms = int((time.monotonic() - started) * 1000)
        processing.completed_at = datetime.now(timezone.utc)
        _attach_cost(processing)
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
                    "criteria_scored=%s cost_usd=%s]", analysis_id, request.call_id,
                    processing.total_ms, scoring["counts"]["criteria_scored"],
                    processing.cost.total_usd if processing.cost else None)
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
            or strategy == STRATEGY_FALLBACK_TEXT)


async def _finish_skipped(analysis_id, ctx, transcript, context_used, processing,
                          reason, deps: PipelineDeps, created_at, started):
    """Configured policy says this call is not evaluated. Not a failure: the
    transcript is real and is kept, there is simply no scorecard."""
    processing.total_ms = int((time.monotonic() - started) * 1000)
    _attach_cost(processing)
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
    _attach_cost(processing, failure_reason=reason)
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
