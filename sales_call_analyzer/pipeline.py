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
from collections import Counter
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
from transcription import sarvam_cleanup as _sarvam_cleanup
from transcription import segment_transcribe as _segment
from transcription import voice as _voice

from . import ROLE_SALES_REP as _ROLE_SALES_REP
from . import diarization as diarization_mod
from transcription.romanize import indic_share as _indic_share
from . import prosody as prosody_mod
from . import tone as tone_mod
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
    VoiceAnalysis,
    VoiceMoment,
    VoiceSpeaker,
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
    # Speaker refinement (diarization.py). All injected: tests need no model,
    # and a server without sherpa-onnx or the model file skips the step.
    speaker_refine_mode: Optional[str] = None
    load_voiceprint: Optional[Callable[[Optional[str]], Awaitable[Any]]] = None
    embed_voice: Optional[Callable[[Any], Any]] = None
    speech_regions: Optional[Callable[[Any], Any]] = None
    voice_model: Optional[str] = None
    # Segment pass: (clips, language) -> transcription.segment_transcribe.ClipResult.
    segment_transcribe: Optional[Callable[..., Awaitable[Any]]] = None
    segment_pass_mode: Optional[str] = None
    segment_pass_model: Optional[str] = None
    segment_pass_scope: Optional[str] = None
    # Tone step: (clips) -> tone.ToneResult.
    classify_tone: Optional[Callable[..., Awaitable[Any]]] = None
    tone_mode: Optional[str] = None
    tone_model: Optional[str] = None
    # Which provider transcribes audio: "deepgram" (default) or "sarvam".
    transcriber: Optional[str] = None
    # (audio bytes, filename=) -> transcription.sarvam_client.SarvamResult
    sarvam_transcribe: Optional[Callable[..., Awaitable[Any]]] = None
    # (samples, entries, indices, terms, brand_terms) -> sarvam_recheck.RecheckResult
    sarvam_recheck: Optional[Callable[..., Awaitable[Any]]] = None
    sarvam_model: Optional[str] = None
    sarvam_recheck_mode: Optional[str] = None
    sarvam_voiceprint_mode: Optional[str] = None
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


# Speaker refinement: off | shadow | on, like language identification.
#   off     never runs.
#   shadow  runs, records what it WOULD change in processing.speaker_refinement,
#           changes nothing. Watch it on live calls first.
#   on      the refined speakers are the transcript's speakers.
# Default off: it costs ~27 s of CPU per 10-minute call in the API process.
SPEAKER_REFINE_OFF = "off"
SPEAKER_REFINE_SHADOW = "shadow"
SPEAKER_REFINE_ON = "on"
SPEAKER_REFINE_MODE = os.environ.get("SCA_SPEAKER_REFINE", SPEAKER_REFINE_OFF).strip().lower()

REASON_REFINE_AUDIO_UNAVAILABLE = "audio_unavailable"
REASON_REFINE_UNDECODABLE = "audio_not_decodable"


async def _audio_bytes(request: AnalyzeRequest, deps: PipelineDeps, box: dict,
                       analysis_id: str) -> Optional[bytes]:
    """The recording, fetched at most once per run and shared by language
    identification and speaker refinement. None if it cannot be fetched."""
    if "bytes" in box:
        return box["bytes"]
    box["bytes"] = None
    if not deps.fetch_audio or not transcript_mod.has_audio(request.audio):
        return None
    try:
        box["bytes"], box["content_type"] = await deps.fetch_audio(request.audio.url)
    except Exception as err:  # noqa: BLE001 - Deepgram may still reach the URL itself
        logger.warning("could not fetch the recording [analysis_id=%s]: %s",
                       analysis_id, type(err).__name__)
    return box["bytes"]


def _requested_language(request: AnalyzeRequest) -> Optional[str]:
    """The language this request asks Deepgram for, before server defaults."""
    language = request.options.language_hint
    if not language and request.transcript:
        language = request.transcript.language
    return language


async def _decide_language(request: AnalyzeRequest, deps: PipelineDeps,
                           processing: ProcessingInfo,
                           analysis_id: str, box: Optional[dict] = None) -> Optional[str]:
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

    box = {} if box is None else box
    audio = await _audio_bytes(request, deps, box, analysis_id)
    if audio is None:
        processing.language_detection_reason = REASON_LANGUAGE_ID_AUDIO_UNAVAILABLE
        return None
    content_type = box.get("content_type")

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

    language, why = _dg_client.language_for(decision.dominant_non_english,
                                            decision.english_share,
                                            decision.dominant_share)
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


async def _refine_speakers(raw: dict, request: AnalyzeRequest, deps: PipelineDeps,
                           processing: ProcessingInfo, box: dict,
                           analysis_id: str) -> tuple[dict, Optional[dict], Optional[list]]:
    """(raw to use, rep voice match, speech regions). Never raises.

    Checks Deepgram's speakers against the voices (diarization.py): a speaker-
    per-channel recording is attributed by channel; otherwise stretches are
    compared by voice, and against the rep's enrolled voiceprint when there is
    one. In shadow mode everything is measured and recorded, nothing changes.
    """
    mode = (deps.speaker_refine_mode or SPEAKER_REFINE_MODE or SPEAKER_REFINE_OFF).strip().lower()
    if mode == SPEAKER_REFINE_OFF:
        return raw, None, None
    info: dict = {"mode": mode, "applied": False}
    processing.speaker_refinement = info

    embed = deps.embed_voice
    regions_fn = deps.speech_regions
    model = deps.voice_model
    if embed is None:
        usable, why = _voice.availability()
        if not usable:
            info["reason"] = why
            return raw, None, None
        embed, regions_fn, model = _voice.embed, regions_fn or _voice.speech_regions, _voice.model_id()
    info["model"] = model

    audio = await _audio_bytes(request, deps, box, analysis_id)
    if audio is None:
        info["reason"] = REASON_REFINE_AUDIO_UNAVAILABLE
        return raw, None, None
    try:
        channels = await _audio_slice.channel_count(audio) or 1
        decoded = await _audio_slice.decode(audio, channels=2 if channels >= 2 else 1)
    except Exception:  # noqa: BLE001
        decoded = None
    if decoded is None or not len(decoded):
        info["reason"] = REASON_REFINE_UNDECODABLE
        return raw, None, None
    samples = decoded.mean(axis=1) if decoded.shape[1] > 1 else decoded[:, 0]
    box["samples"] = samples

    voiceprint = None
    if deps.load_voiceprint and request.rep.id:
        try:
            voiceprint = await deps.load_voiceprint(request.rep.id)
        except Exception as err:  # noqa: BLE001 - a missing voiceprint is not an error
            logger.warning("voiceprint lookup failed [analysis_id=%s]: %s",
                           analysis_id, type(err).__name__)
    info["voiceprint_used"] = voiceprint is not None

    regions = None
    try:
        profile = (diarization_mod.stereo_profile(decoded) if decoded.shape[1] > 1
                   else {"speaker_split": False})
        if decoded.shape[1] > 1:
            info["stereo"] = profile
        if profile.get("speaker_split"):
            refined, report = diarization_mod.attribute_by_channel(raw, decoded)
            if voiceprint is not None and report.applied:
                report.voice_scores = await asyncio.to_thread(
                    diarization_mod.score_speakers, refined, samples, embed, voiceprint)
                best = max(report.voice_scores, key=report.voice_scores.get, default=None)
                if best and report.voice_scores[best] >= diarization_mod.REP_MATCH_AT:
                    report.rep_speaker_id = best
        else:
            refined, report = await asyncio.to_thread(
                diarization_mod.refine, raw, samples, embed, voiceprint, model)
        if regions_fn is not None:
            regions = await asyncio.to_thread(regions_fn, samples)
    except Exception as err:  # noqa: BLE001 - refinement is an improvement, never a dependency
        logger.warning("speaker refinement failed [analysis_id=%s]: %s",
                       analysis_id, type(err).__name__)
        info["reason"] = diarization_mod.REASON_ERROR
        return raw, None, None

    info.update({k: v for k, v in report.as_dict().items() if k != "applied"})
    info["would_apply"] = report.applied
    if mode != SPEAKER_REFINE_ON or not report.applied:
        return raw, None, regions

    info["applied"] = True
    rep_voice = None
    if report.rep_speaker_id:
        rep_voice = {"rep_id": request.rep.id, "speaker_id": report.rep_speaker_id,
                     "score": report.voice_scores.get(report.rep_speaker_id), "model": model}
    return refined, rep_voice, regions


# Segment pass: off | shadow | on, like language identification.
#   off     never runs (default).
#   shadow  re-transcribes and records processing.segment_pass, changes nothing.
#   on      the customer's turns carry the re-transcribed words.
SEGMENT_PASS_OFF = "off"
SEGMENT_PASS_SHADOW = "shadow"
SEGMENT_PASS_ON = "on"
SEGMENT_PASS_MODE = os.environ.get("SCA_SEGMENT_PASS", SEGMENT_PASS_OFF).strip().lower()
# A regional language must be at least this % of the call's speech to trigger it.
SEGMENT_PASS_MIN_SHARE = float(os.environ.get("SCA_SEGMENT_PASS_MIN_SHARE", 10))
SEGMENT_PASS_MIN_SECONDS = 0.3
SEGMENT_PASS_PAD_SECONDS = 0.15
# Which turns are re-transcribed on a regional call:
#   customer  every turn except the rep's (the first design)
#   mixed     those, plus any rep turn whose Deepgram text contains Indian script
#   all       every turn
# The customer-only design assumed the rep speaks English. On a real Marathi
# call (testaudio/mycall4.mp3) the rep switched into Marathi too, and his turns
# stayed garbled: "अपना अपना business growth site की अमचा platform बदल inquiry"
# for "आपण आपल्या business growth साठी आमच्या platform बद्दल inquiry".
SEGMENT_SCOPE_CUSTOMER = "customer"
SEGMENT_SCOPE_MIXED = "mixed"
SEGMENT_SCOPE_ALL = "all"
SEGMENT_PASS_SCOPE = os.environ.get("SCA_SEGMENT_PASS_SCOPE", SEGMENT_SCOPE_ALL).strip().lower()
# A clip may extend into the silence after its turn - up to this much, and never
# into the next turn. Deepgram's turn boundaries can drop a turn's last words
# into the gap: "मग scale serum वेगळं काय करेल?" came back as "मग ScaleCRM".
SEGMENT_PASS_GAP_FILL_SECONDS = float(os.environ.get("SCA_SEGMENT_PASS_GAP_FILL_SECONDS", 1.0))
SEGMENT_PASS_NEXT_TURN_GUARD_SECONDS = 0.25
# Indian-script share of a Deepgram turn above which, in "mixed" scope, it is
# treated as code-switched.
SEGMENT_MIXED_MIN_INDIC_SHARE = 0.1
SEGMENT_TEXT_SOURCE = "gemini_segment_pass"

REASON_SEGMENT_NO_REGIONAL = "no_regional_language_detected"
REASON_SEGMENT_WHOLE_CALL = "whole_call_already_in_regional_language"
REASON_SEGMENT_NO_SEGMENTS = "no_segments_to_retranscribe"


def _regional_language(request: AnalyzeRequest, processing: ProcessingInfo) -> Optional[str]:
    """The regional language this call's customer speaks, if one was detected
    or supplied. Hindi is excluded: multi already transcribes it well (8% WER on
    the customer's turns of the synthetic Hindi calls)."""
    hint = (request.options.language_hint or "").strip().lower()
    if hint in _dg_client.REGIONAL_LANGUAGES:
        return hint
    code = (processing.language_detected or "").strip().lower()
    if code not in _dg_client.REGIONAL_LANGUAGES:
        return None
    share = next((entry.get("share_percent") for entry in processing.language_detection_shares or []
                  if (entry.get("code") or "").lower() == code), None)
    if share is not None and share < SEGMENT_PASS_MIN_SHARE:
        return None
    return code


def _segments_in_scope(transcript: NormalizedTranscript, scope: str) -> tuple[list, bool]:
    """(segments to re-transcribe, whether any rep turn was left out)."""
    reps = {s.speaker_id for s in transcript.speakers if s.role == _ROLE_SALES_REP}
    timed = [s for s in transcript.segments if s.start is not None and s.end is not None
             and s.end - s.start >= SEGMENT_PASS_MIN_SECONDS]
    if scope == SEGMENT_SCOPE_ALL or not reps:
        return timed, False
    chosen = []
    for s in timed:
        if s.speaker_id not in reps:
            chosen.append(s)
        elif scope == SEGMENT_SCOPE_MIXED and _indic_share(s.text) >= SEGMENT_MIXED_MIN_INDIC_SHARE:
            chosen.append(s)
    return chosen, len(chosen) < len(timed)


def _clip_bounds(transcript: NormalizedTranscript, seg) -> tuple[float, float]:
    """Where to cut a turn's clip: a small pad before it, and into the silence
    after it up to the next turn - never into another turn's words."""
    starts = sorted(s.start for s in transcript.segments if s.start is not None)
    ends = sorted(s.end for s in transcript.segments if s.end is not None)
    previous_end = max((e for e in ends if e <= seg.start), default=None)
    next_start = min((b for b in starts if b >= seg.end), default=None)
    start = seg.start - SEGMENT_PASS_PAD_SECONDS
    if previous_end is not None:
        start = max(start, previous_end)
    end = seg.end + max(SEGMENT_PASS_PAD_SECONDS, SEGMENT_PASS_GAP_FILL_SECONDS)
    if next_start is not None:
        # Stop short of the next turn: Deepgram starts a word's timing a little
        # late, so right up to next_start the clip caught the next speaker's
        # first syllable ("आप-" before "आपण", measured on mycall4.mp3).
        end = min(end, max(next_start - SEGMENT_PASS_NEXT_TURN_GUARD_SECONDS,
                           seg.end + SEGMENT_PASS_PAD_SECONDS))
    return max(start, 0.0), end


async def _segment_pass(transcript: NormalizedTranscript, request: AnalyzeRequest,
                        deps: PipelineDeps, processing: ProcessingInfo, box: dict,
                        analysis_id: str, brand_name: Optional[str] = None) -> bool:
    """Re-transcribe a regional call's turns in their language. True if any
    text changed. Never raises.

    Which turns depends on SEGMENT_PASS_SCOPE (default: all). Runs after roles
    are resolved, so the customer-only and mixed scopes can leave the rep's
    English turns alone.
    """
    mode = (deps.segment_pass_mode or SEGMENT_PASS_MODE or SEGMENT_PASS_OFF).strip().lower()
    if mode == SEGMENT_PASS_OFF:
        return False
    info: dict = {"mode": mode, "applied": False, "model": deps.segment_pass_model}
    processing.segment_pass = info

    language = _regional_language(request, processing)
    if language is None:
        info["reason"] = REASON_SEGMENT_NO_REGIONAL
        return False
    info["language"] = language
    if (processing.transcription_language_sent or "").lower() == language:
        info["reason"] = REASON_SEGMENT_WHOLE_CALL
        return False
    if deps.segment_transcribe is None:
        info["reason"] = _segment.REASON_NOT_CONFIGURED
        return False

    scope = (deps.segment_pass_scope or SEGMENT_PASS_SCOPE or SEGMENT_SCOPE_ALL).strip().lower()
    segments, rep_excluded = _segments_in_scope(transcript, scope)
    info["scope"] = scope
    info["segments"] = len(segments)
    info["rep_excluded"] = rep_excluded
    if not segments:
        info["reason"] = REASON_SEGMENT_NO_SEGMENTS
        return False

    samples = box.get("samples")
    if samples is None:
        audio = await _audio_bytes(request, deps, box, analysis_id)
        decoded = await _audio_slice.decode(audio) if audio else None
        if decoded is None or not len(decoded):
            info["reason"] = REASON_REFINE_AUDIO_UNAVAILABLE
            return False
        samples = box["samples"] = decoded[:, 0]

    rate = 16000
    clips = []
    for s in segments:
        a, b = _clip_bounds(transcript, s)
        clips.append(samples[int(a * rate):min(int(b * rate), len(samples))])
    info["audio_seconds"] = round(sum(len(c) for c in clips) / rate, 1)

    try:
        keyterms = transcription_keyterms(request, brand_name)
        extra = {"keyterms": keyterms} if keyterms else {}
        result = await deps.segment_transcribe(clips, language, **extra)
    except Exception as err:  # noqa: BLE001 - the original words are always kept
        logger.warning("segment pass failed [analysis_id=%s]: %s", analysis_id, type(err).__name__)
        info["reason"] = _segment.REASON_PROVIDER_ERROR
        return False

    processing.segment_pass_input_tokens = result.input_tokens
    processing.segment_pass_output_tokens = result.output_tokens
    processing.segment_pass_thinking_tokens = result.thinking_tokens
    processing.segment_pass_cached_tokens = result.cached_tokens
    info.update({"requests": result.requests, "ms": result.ms, "reason": result.reason,
                 "rejected": dict(Counter(result.rejected.values()))})
    texts = {seg.index: text for seg, text in zip(segments, result.texts) if text}
    info["replaced"] = len(texts)
    if mode != SEGMENT_PASS_ON or not texts:
        return False
    transcript_mod.replace_segment_texts(transcript, texts, SEGMENT_TEXT_SOURCE)
    info["applied"] = True
    return True


# Tone step: off | shadow | on.
#   off     never runs (default).
#   shadow  measures and listens, records processing.tone, changes nothing.
#   on      the report carries `voice`, and the analysis is told how it was said.
TONE_OFF = "off"
TONE_SHADOW = "shadow"
TONE_ON = "on"
TONE_MODE = os.environ.get("SCA_TONE", TONE_OFF).strip().lower()
_ROLE_WORDS = {"sales_rep": "the sales rep", "customer": "the customer",
               "participant": "another participant"}


async def _tone_pass(transcript: NormalizedTranscript, request: AnalyzeRequest,
                     deps: PipelineDeps, processing: ProcessingInfo, box: dict,
                     analysis_id: str) -> None:
    """How it was said: measure every turn, listen to the ones that matter.

    Sets transcript.voice in "on" mode. Never raises; any failure leaves the
    analysis exactly as it would have been without this step.
    """
    mode = (deps.tone_mode or TONE_MODE or TONE_OFF).strip().lower()
    if mode == TONE_OFF:
        return
    info: dict = {"mode": mode, "applied": False, "model": deps.tone_model,
                  "prompt_version": tone_mod.PROMPT_VERSION}
    processing.tone = info
    if deps.classify_tone is None:
        info["reason"] = tone_mod.REASON_NOT_CONFIGURED
        return

    samples = box.get("samples")
    if samples is None:
        audio = await _audio_bytes(request, deps, box, analysis_id)
        decoded = await _audio_slice.decode(audio) if audio else None
        if decoded is None or not len(decoded):
            info["reason"] = REASON_REFINE_AUDIO_UNAVAILABLE
            return
        samples = box["samples"] = decoded[:, 0]

    try:
        measures, speakers = await asyncio.to_thread(prosody_mod.analyse, transcript, samples)
    except Exception as err:  # noqa: BLE001
        logger.warning("prosody failed [analysis_id=%s]: %s", analysis_id, type(err).__name__)
        info["reason"] = "prosody_error"
        return
    roles = {s.speaker_id: s.role for s in transcript.speakers}
    chosen = tone_mod.select_moments(transcript, measures, roles)
    info["moments"] = len(chosen)
    if not chosen:
        info["reason"] = tone_mod.REASON_NO_MOMENTS
        return

    rate = 16000
    by_index = {s.index: s for s in transcript.segments}
    by_measure = {m.index: m for m in measures}
    clips = [tone_mod.ToneClip(
        samples=samples[int(by_index[i].start * rate):int(by_index[i].end * rate)],
        role=_ROLE_WORDS.get(roles.get(by_index[i].speaker_id), "a speaker"),
        # The measured delivery is NOT sent: on CREMA-D it did not change
        # accuracy (48.0% with, 48.5% without) and made neutral less stable.
        text=by_index[i].text) for i in chosen]
    try:
        result = await deps.classify_tone(clips)
    except Exception as err:  # noqa: BLE001
        logger.warning("tone step failed [analysis_id=%s]: %s", analysis_id, type(err).__name__)
        info["reason"] = tone_mod.REASON_PROVIDER_ERROR
        return
    processing.tone_input_tokens = result.input_tokens
    processing.tone_output_tokens = result.output_tokens
    processing.tone_thinking_tokens = result.thinking_tokens
    info.update({"requests": result.requests, "ms": result.ms, "reason": result.reason})
    if not result.ok:
        return

    moments = []
    counts: dict[str, dict[str, int]] = {}
    for i, label in zip(chosen, result.labels):
        if not label:
            continue
        seg, measure = by_index[i], by_measure.get(i)
        z = dict(measure.z) if measure else {}
        role = roles.get(seg.speaker_id, "unknown")
        support = tone_mod.acoustic_support(label["tone"], z)
        if support == "contradicts" and label["confidence"] != "low":
            # When the measured voice points the other way, Gemini was right 23%
            # of the time on CREMA-D (10/44), against 53-57% otherwise.
            label = {**label, "confidence": "low",
                     "cue": (label["cue"] + " [measured delivery disagrees]").strip()}
        moments.append(VoiceMoment(
            segment_index=i, speaker_id=seg.speaker_id, role=role, start=seg.start,
            end=seg.end, quote=seg.text[:240], measured=z, acoustic_support=support, **label))
        counts.setdefault(role, {})
        counts[role][label["tone"]] = counts[role].get(label["tone"], 0) + 1
    info["labelled"] = len(moments)
    voice = VoiceAnalysis(
        model=deps.tone_model, prompt_version=tone_mod.PROMPT_VERSION,
        speakers=[VoiceSpeaker(role=roles.get(sp.speaker_id, "unknown"), **sp.as_dict())
                  for sp in speakers],
        moments=moments, tone_counts=counts)
    if mode != TONE_ON:
        info["tone_counts"] = counts      # shadow: what it heard, nothing changed
        return
    transcript.voice = voice
    info["applied"] = True


# =========================================================================== #
# Sarvam (SCA_TRANSCRIBER=sarvam)
# =========================================================================== #
# Measured against the current pipeline (scripts/sca_eval/README.md, 2026-09-30):
# on 19 calls with exact truth Sarvam put 95% of words on the right role (Deepgram
# + our refinement: 75%) and wrote regional languages far better (FLEURS: 7 of 8
# languages better). Its weaknesses - echoed fragments, stray scripts, misheard
# brand and product names - are fixed by transcription/sarvam_cleanup.py, and
# the few turns it cannot fix from text are re-checked by Gemini
# (sarvam_recheck.py). With Sarvam:
#   * no Gemini language identification: Sarvam detects the language itself
#   * no segment pass: Sarvam already writes the regional language
#   * no re-splitting of speakers (diarization.refine): Sarvam's speakers are
#     better, and it gives no word timings to split on. The rep's voiceprint is
#     still matched, to say which speaker the rep is.
# If Sarvam fails (not configured, no credits, outage), the call falls back to
# Deepgram and processing.sarvam says why.
TRANSCRIBER_DEEPGRAM = "deepgram"
TRANSCRIBER_SARVAM = "sarvam"
TRANSCRIBER = os.environ.get("SCA_TRANSCRIBER", TRANSCRIBER_DEEPGRAM).strip().lower()
# on | off. The Gemini re-check of flagged turns (~Rs 0.2-0.6 a call when any).
SARVAM_RECHECK_MODE = os.environ.get("SCA_SARVAM_RECHECK", "on").strip().lower()
# on | off. Matching the rep's enrolled voiceprint to a Sarvam speaker. Only
# labels a speaker, never moves words, so it does not wait on SCA_SPEAKER_REFINE.
SARVAM_VOICEPRINT_MODE = os.environ.get("SCA_SARVAM_VOICEPRINT", "on").strip().lower()

LANGUAGE_BASIS_TRANSCRIBER = "detected_by_transcriber"
REASON_TRANSCRIBED_BY_SARVAM = "transcribed_by_sarvam_no_deepgram_charge"
REASON_SARVAM_NOT_CONFIGURED = "sarvam_not_configured"


def _transcriber(deps: PipelineDeps) -> str:
    return (deps.transcriber or TRANSCRIBER or TRANSCRIBER_DEEPGRAM).strip().lower()


def sarvam_terms(request: AnalyzeRequest, brand_name: Optional[str] = None) -> list[str]:
    """Brand and product names the clean-up writes correctly. Never people's
    names: the CRM's are often placeholders, and a wrong one written into the
    transcript would be worse than a misspelt right one."""
    return [t for t in dict.fromkeys([brand_name, request.product.name]) if t and t.strip()]


async def _transcribe_with_sarvam(request: AnalyzeRequest, deps: PipelineDeps,
                                  processing: ProcessingInfo, box: dict, analysis_id: str,
                                  brand_name: Optional[str]) -> Optional[tuple[dict, Optional[str]]]:
    """(Deepgram-shaped response, language code) from Sarvam, cleaned and
    re-checked; None when Sarvam could not be used. Never raises."""
    info: dict = {"model": deps.sarvam_model, "applied": False}
    processing.sarvam = info
    if deps.sarvam_transcribe is None:
        info["reason"] = REASON_SARVAM_NOT_CONFIGURED
        return None
    audio = await _audio_bytes(request, deps, box, analysis_id)
    if audio is None:
        info["reason"] = REASON_REFINE_AUDIO_UNAVAILABLE
        return None
    filename = os.path.basename((request.audio.url or "").split("?")[0]) or "call.mp3"
    try:
        result = await deps.sarvam_transcribe(audio, filename=filename)
    except Exception as err:  # noqa: BLE001 - the client never raises; belt and braces
        logger.warning("Sarvam failed [analysis_id=%s]: %s", analysis_id, type(err).__name__)
        info["reason"] = "sarvam_provider_error"
        return None
    info["ms"] = result.ms
    if not result.ok:
        info.update({"reason": result.reason, "error": result.error})
        logger.warning("Sarvam failed [analysis_id=%s]: %s", analysis_id, result.reason)
        return None
    response = result.response or {}
    info["model"] = result.model
    info["mode"] = result.mode
    info["language_code"] = response.get("language_code")

    terms = sarvam_terms(request, brand_name)
    vocabulary = list(request.product.terms or [])
    cleaned = _sarvam_cleanup.clean(response, terms, vocabulary)
    report = cleaned.report()
    info["cleanup"] = {k: report[k] for k in ("counts", "duplicates_removed", "script_fixes",
                                              "term_fixes", "suspects")}

    samples = None
    try:
        decoded = await _audio_slice.decode(audio)
        if decoded is not None and len(decoded):
            samples = box["samples"] = decoded[:, 0]
    except Exception:  # noqa: BLE001 - only the re-check and the voice steps need it
        samples = None

    mode = (deps.sarvam_recheck_mode or SARVAM_RECHECK_MODE).strip().lower()
    if cleaned.suspect_indices:
        if mode != "on":
            info["recheck"] = {"reason": "off"}
        elif deps.sarvam_recheck is None or samples is None:
            info["recheck"] = {"reason": "not_configured" if deps.sarvam_recheck is None
                               else REASON_REFINE_AUDIO_UNAVAILABLE}
        else:
            try:
                done = await deps.sarvam_recheck(samples, cleaned.entries, cleaned.suspect_indices,
                                                 terms + vocabulary, terms)
            except Exception as err:  # noqa: BLE001 - an optional step never fails the call
                logger.warning("Sarvam re-check failed [analysis_id=%s]: %s",
                               analysis_id, type(err).__name__)
                info["recheck"] = {"reason": "recheck_error", "error": type(err).__name__}
            else:
                info["recheck"] = done.report()
                processing.sarvam_recheck_input_tokens = done.input_tokens
                processing.sarvam_recheck_output_tokens = done.output_tokens
                processing.sarvam_recheck_thinking_tokens = done.thinking_tokens

    raw = _sarvam_cleanup.to_deepgram_shape(cleaned.entries)
    if samples is not None:
        raw["metadata"]["duration"] = round(len(samples) / 16000, 3)
    raw["metadata"]["provider"] = TRANSCRIBER_SARVAM
    await deps.store.save_raw_transcript(analysis_id, response,
                                         enabled=(request.options.store_raw_transcript
                                                  if request.options.store_raw_transcript is not None
                                                  else deps.store_raw_transcript))
    code = (response.get("language_code") or "").split("-")[0].lower() or None
    processing.transcription_provider = TRANSCRIBER_SARVAM
    processing.transcription_model = result.model
    processing.transcription_language_sent = None
    processing.language_basis = LANGUAGE_BASIS_TRANSCRIBER
    processing.language_detected = code if code and code != "en" else None
    info["applied"] = True
    return raw, code


async def _sarvam_rep_voice(raw: dict, request: AnalyzeRequest, deps: PipelineDeps,
                            processing: ProcessingInfo, box: dict,
                            analysis_id: str) -> Optional[dict]:
    """Which Sarvam speaker is the rep, by the rep's enrolled voiceprint. Never raises."""
    mode = (deps.sarvam_voiceprint_mode or SARVAM_VOICEPRINT_MODE).strip().lower()
    samples = box.get("samples")
    if mode != "on" or not request.rep.id or deps.load_voiceprint is None or samples is None:
        return None
    embed, model = deps.embed_voice, deps.voice_model
    if embed is None:
        usable, why = _voice.availability()
        if not usable:
            processing.speaker_refinement = {"transcriber": TRANSCRIBER_SARVAM, "reason": why}
            return None
        embed, model = _voice.embed, _voice.model_id()
    try:
        voiceprint = await deps.load_voiceprint(request.rep.id)
    except Exception as err:  # noqa: BLE001 - a missing voiceprint is not an error
        logger.warning("voiceprint lookup failed [analysis_id=%s]: %s",
                       analysis_id, type(err).__name__)
        voiceprint = None
    info: dict = {"transcriber": TRANSCRIBER_SARVAM, "model": model,
                  "voiceprint_used": voiceprint is not None}
    processing.speaker_refinement = info
    if voiceprint is None:
        return None
    try:
        scores = await asyncio.to_thread(diarization_mod.score_speakers, raw, samples,
                                         embed, voiceprint)
    except Exception as err:  # noqa: BLE001
        logger.warning("voiceprint match failed [analysis_id=%s]: %s",
                       analysis_id, type(err).__name__)
        info["reason"] = diarization_mod.REASON_ERROR
        return None
    info["voice_scores"] = scores
    best = max(scores, key=scores.get, default=None)
    if best is None or scores[best] < diarization_mod.REP_MATCH_AT:
        info["reason"] = "no_speaker_matches_the_voiceprint"
        return None
    info["rep_speaker_id"] = best
    return {"rep_id": request.rep.id, "speaker_id": best, "score": scores[best], "model": model}


def transcription_keyterms(request: AnalyzeRequest,
                           brand_name: Optional[str] = None) -> list[str]:
    """Names Deepgram should expect: brand, product, rep, customer.

    Brand and rep first - they are what role resolution needs most and what
    survives the MAX_KEYTERMS cap. Full names and first names both, because the
    rep says "this is Rohan" and the CRM stores "Rohan Mehta".
    """
    terms: list[Optional[str]] = [brand_name, request.product.name]
    for person in (request.rep.name, request.customer.name):
        terms.append(person)
        parts = (person or "").split()
        if len(parts) > 1:
            terms.append(parts[0])
    return _dg_client.clean_keyterms(terms)


async def resolve_transcript(request: AnalyzeRequest, deps: PipelineDeps,
                             analysis_id: str,
                             stored: Optional[dict] = None,
                             processing: Optional[ProcessingInfo] = None,
                             brand_name: Optional[str] = None,
                             box: Optional[dict] = None,
                             ) -> tuple[NormalizedTranscript, str]:
    """Produce the normalised transcript. Returns (transcript, strategy).

    A transcript already produced for this call short-circuits everything: an
    earlier attempt that died at the LLM step has already paid for it.
    """
    # A transcript made by the other provider is not reused: switching
    # SCA_TRANSCRIBER is a request for that provider's transcript.
    made_by = (stored or {}).get("source")
    if (stored and made_by in (TRANSCRIBER_DEEPGRAM, TRANSCRIBER_SARVAM)
            and transcript_mod.has_audio(request.audio) and made_by != _transcriber(deps)):
        logger.info("stored transcript is from %s, not %s; transcribing again [analysis_id=%s]",
                    made_by, _transcriber(deps), analysis_id)
        stored = None
    if stored:
        logger.info("reusing stored transcript [analysis_id=%s]", analysis_id)
        # Reuse the transcription, never the role resolution: the CRM names in
        # THIS request may differ from the ones the earlier run had.
        return (transcript_mod.label_speakers(
                    transcript_mod.clear_role_annotations(NormalizedTranscript(**stored))),
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
    box = {} if box is None else box

    if _transcriber(deps) == TRANSCRIBER_SARVAM:
        got = await _transcribe_with_sarvam(request, deps, processing, box, analysis_id,
                                            brand_name)
        if got is not None:
            raw, code = got
            rep_voice = await _sarvam_rep_voice(raw, request, deps, processing, box, analysis_id)
            transcript = transcript_mod.from_deepgram(raw, language_hint=code)
            transcript.source = transcript_mod.SOURCE_SARVAM
            transcript.rep_voice = rep_voice
            return transcript, strategy
        processing.sarvam["fallback"] = TRANSCRIBER_DEEPGRAM
        processing.transcription_provider = TRANSCRIBER_DEEPGRAM
        logger.warning("Sarvam unavailable (%s); transcribing with Deepgram [analysis_id=%s]",
                       processing.sarvam.get("reason"), analysis_id)

    language = await _decide_language(request, deps, processing, analysis_id, box)
    # What was actually sent, which is what billing prices: multi is charged at
    # the multilingual rate.
    processing.transcription_language_sent = _dg_client.effective_language(language)
    keyterms = transcription_keyterms(request, brand_name)
    processing.transcription_keyterm_count = len(keyterms)
    extra = {"keyterms": keyterms} if keyterms else {}
    try:
        raw = await deps.transcribe(request.audio.url,
                                    mime_type=request.audio.mime_type,
                                    language=language, **extra)
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
    raw, rep_voice, regions = await _refine_speakers(raw, request, deps, processing, box,
                                                    analysis_id)
    transcript = transcript_mod.from_deepgram(
        raw, language_hint=processing.transcription_language_sent)
    transcript = transcript_mod.apply_vad(transcript, regions)
    transcript.rep_voice = rep_voice
    return transcript, strategy


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
        # Sarvam transcribed it: Sarvam is billed for the audio, Deepgram is not.
        sarvam = None
        if processing.transcription_provider == TRANSCRIBER_SARVAM:
            sarvam = billing.sarvam_cost(
                pricing, outcome=outcome, reason=reason, model=processing.transcription_model,
                audio_seconds=processing.audio_seconds_submitted, at=at)
            outcome, reason = billing.NOT_CHARGED, REASON_TRANSCRIBED_BY_SARVAM
        elif processing.sarvam:
            # Tried and fell back to Deepgram. A refused job (no credits, not
            # configured) is not billed; an error after upload may have been.
            failed = processing.sarvam.get("reason") == "sarvam_provider_error"
            sarvam = billing.sarvam_cost(
                pricing, outcome=billing.CHARGE_UNKNOWN if failed else billing.NOT_CHARGED,
                reason=processing.sarvam.get("reason"), model=processing.sarvam.get("model"),
                audio_seconds=None, at=at)
        recheck_requests = int(((processing.sarvam or {}).get("recheck") or {}).get("requests") or 0)
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
        def _total(*values: Optional[int]) -> Optional[int]:
            present = [v for v in values if v is not None]
            return sum(present) if present else None

        gemini = billing.gemini_cost(
            pricing, model_requested=processing.llm_model,
            model_version=processing.llm_model_version,
            attempts=(processing.llm_attempts + (1 if processing.language_detection_ms else 0)
                      + int((processing.segment_pass or {}).get("requests") or 0)
                      + int((processing.tone or {}).get("requests") or 0)
                      + recheck_requests),
            input_tokens=_total(processing.llm_input_tokens,
                                processing.language_id_input_tokens,
                                processing.segment_pass_input_tokens,
                                processing.tone_input_tokens,
                                processing.sarvam_recheck_input_tokens),
            output_tokens=_total(processing.llm_output_tokens,
                                 processing.language_id_output_tokens,
                                 processing.segment_pass_output_tokens,
                                 processing.tone_output_tokens,
                                 processing.sarvam_recheck_output_tokens),
            thinking_tokens=_total(processing.llm_thinking_tokens,
                                   processing.language_id_thinking_tokens,
                                   processing.segment_pass_thinking_tokens,
                                   processing.tone_thinking_tokens,
                                   processing.sarvam_recheck_thinking_tokens),
            cached_tokens=_total(processing.llm_cached_tokens,
                                 processing.language_id_cached_tokens,
                                 processing.segment_pass_cached_tokens), at=at)
        processing.billed_channels = deepgram.billed_channels
        processing.cost = billing.combine(pricing, deepgram, gemini, at=at, sarvam=sarvam)
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
        # The brand is looked up first: its name is a transcription keyterm as
        # well as a role-resolution clue.
        brand_ref = {}
        if deps.resolve_brand_ref and request.lead_id:
            try:
                brand_ref = await deps.resolve_brand_ref(request.lead_id) or {}
            except Exception as err:  # noqa: BLE001 - context is never fatal
                logger.warning("brand ref lookup failed [analysis_id=%s]: %s",
                               analysis_id, type(err).__name__)

        if not brand_ref.get("brand_name") and request.brand_name:
            brand_ref = {**brand_ref, "brand_name": request.brand_name}

        stored = await deps.store.stored_transcript(request.call_id)
        t0 = time.monotonic()
        # The recording, fetched once and shared by language identification,
        # speaker refinement and the segment pass; released before the LLM step.
        box: dict = {}
        transcript, strategy = await resolve_transcript(request, deps, analysis_id, stored,
                                                        processing,
                                                        brand_name=brand_ref.get("brand_name"),
                                                        box=box)
        processing.transcript_strategy = strategy
        if strategy != STRATEGY_REUSED:
            processing.transcription_ms = int((time.monotonic() - t0) * 1000)
            processing.audio_seconds_submitted = (
                transcript.duration_seconds
                if strategy == transcript_mod.STRATEGY_TRANSCRIBE_AUDIO else None)
        sarvam_used = processing.transcription_provider == TRANSCRIBER_SARVAM
        if strategy == transcript_mod.STRATEGY_TRANSCRIBE_AUDIO and not sarvam_used:
            processing.channels_processed = transcript.channels_processed
            processing.multichannel_requested = _dg_client.MULTICHANNEL

        if not transcript.segments:
            raise PipelineFailure(TRANSCRIPT_EMPTY, transcript=transcript)

        # ---- 2. speakers ---------------------------------------------------
        # Voice evidence counts only for the rep it was measured against: a
        # reused transcript may be re-analysed with a different rep.
        rep_voice = transcript.rep_voice
        if rep_voice and rep_voice.get("rep_id") != request.rep.id:
            rep_voice = None
        transcript = speakers_mod.resolve_roles(
            transcript, rep=request.rep, customer=request.customer,
            brand_name=brand_ref.get("brand_name"), rep_voice=rep_voice)

        # ---- 2b. the customer's words, in their own language ----------------
        # Only on a fresh transcription: a reused transcript already has them.
        # Not after Sarvam, which already writes the customer's language.
        if strategy == transcript_mod.STRATEGY_TRANSCRIBE_AUDIO and not sarvam_used:
            if await _segment_pass(transcript, request, deps, processing, box, analysis_id,
                                   brand_name=brand_ref.get("brand_name")):
                # New words can carry the name evidence that was garbled before.
                transcript = speakers_mod.resolve_roles(
                    transcript, rep=request.rep, customer=request.customer,
                    brand_name=brand_ref.get("brand_name"), rep_voice=rep_voice)
        # ---- 2c. how it was said ------------------------------------------
        if strategy == transcript_mod.STRATEGY_TRANSCRIBE_AUDIO:
            await _tone_pass(transcript, request, deps, processing, box, analysis_id)
        box.clear()         # release the recording before the LLM step

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
