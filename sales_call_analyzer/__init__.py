"""Sales Call Analyzer - transcription, diarization, framework evaluation and
deterministic scoring for recorded sales calls.

Pipeline (see pipeline.py for the orchestration):

    audio or transcript
      -> deepgram_client.py / transcript.py   what was said, by whom, when
      -> speakers.py                          speaker_id -> role, only on evidence
      -> context.py                           lead + brand brain + product + region
      -> analyzer.py                          Gemini: interpretation and ratings
      -> evidence.py                          every claim checked against the transcript
      -> scoring.py                           criterion -> stage -> overall, in Python
      -> report.py                            assembled, never rescored
      -> store.py                             persisted to MongoDB

DIVISION OF LABOUR (these are correctness requirements, not style preferences)
    Deepgram  says what was said, who spoke and when.
    Gemini    interprets: needs, objections, signals, evidence, ratings, prose.
    Python    validates, scores, aggregates, persists and orchestrates.

    Gemini never produces a score. It produces an ordinal rating per criterion
    and the evidence for it; scoring.py turns ratings into numbers using
    sales_framework.json. That is what makes a score reproducible, re-derivable
    under new weights, and immune to anything a caller says on the call.

NO INVENTED BUSINESS VALUES
    Weights, the rating scale, score bands, the not-applicable policy and the
    disposition policy are all null in sales_framework.json because management
    has not set them. Placeholders are neutral (equal weighting, uniform rating
    spacing) and every response says which values were still unconfirmed.
"""
from .framework import (  # noqa: F401
    FrameworkConfigError,
    RATING_SCALE_CONFIGURED,
    RATING_SCALE_UNIFORM,
    WEIGHTING_CONFIGURED,
    WEIGHTING_EQUAL,
    config_disclosure,
    criteria_blocked_by_requirements,
    criterion_index,
    load_framework,
    load_signals,
    not_applicable_mode,
    rating_scale,
    render_for_prompt,
    skip_decision,
    stage_index,
    unmet_requirements,
    weighting_mode,
)

# --------------------------------------------------------------------------- statuses
# The lifecycle of an analysis. The UI may branch on these.
STATUS_QUEUED = "queued"
STATUS_TRANSCRIBING = "transcribing"
STATUS_ANALYZING = "analyzing"
STATUS_SCORING = "scoring"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"          # transcribed, but evaluation skipped by policy

TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED)
ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_TRANSCRIBING, STATUS_ANALYZING, STATUS_SCORING)

# --------------------------------------------------------------------------- speakers
# Roles are annotations on top of the diarization ids, never a rename of them.
# UNKNOWN is a valid, expected outcome and the UI must render it.
ROLE_SALES_REP = "sales_rep"
ROLE_CUSTOMER = "customer"
ROLE_PARTICIPANT = "participant"
ROLE_UNKNOWN = "unknown"
SPEAKER_ROLES = (ROLE_SALES_REP, ROLE_CUSTOMER, ROLE_PARTICIPANT, ROLE_UNKNOWN)

# How a role was decided. Always reported next to the role.
ROLE_BASIS_UNRESOLVED = "unresolved"
ROLE_BASIS_REP_SELF_INTRO = "crm_rep_self_introduction"
ROLE_BASIS_CUSTOMER_NAME = "crm_customer_name_match"
ROLE_BASIS_CUSTOMER_ADDRESSED = "crm_customer_name_addressed"
ROLE_BASIS_ELIMINATION = "single_other_speaker_by_elimination"
ROLE_BASIS_SUPPLIED = "supplied_by_caller"

# --------------------------------------------------------------------------- reason codes
# Stable identifiers. The UI and the backend may branch on these; do not reword
# them without telling the backend team.

# input
NO_AUDIO_OR_TRANSCRIPT = "no_audio_or_transcript"
AUDIO_UNREACHABLE = "audio_unreachable"
AUDIO_UNUSABLE = "audio_unusable"
AUDIO_TOO_LARGE = "audio_too_large"

# transcription
TRANSCRIPTION_NOT_CONFIGURED = "transcription_not_configured"
TRANSCRIPTION_PROVIDER_ERROR = "transcription_provider_error"
TRANSCRIPTION_RATE_LIMITED = "transcription_rate_limited"
TRANSCRIPTION_TIMEOUT = "transcription_timeout"
TRANSCRIPT_EMPTY = "transcript_empty"
TRANSCRIPT_TOO_SHORT = "transcript_too_short"
TRANSCRIPT_TOO_LONG = "transcript_too_long"

# analysis
ANALYSIS_NOT_CONFIGURED = "analysis_not_configured"
ANALYSIS_PROVIDER_ERROR = "analysis_provider_error"
ANALYSIS_INVALID_OUTPUT = "analysis_invalid_output"
ANALYSIS_FAILED_AFTER_TRANSCRIPTION = "analysis_failed_after_transcription"

# storage / lifecycle
STORAGE_NOT_CONFIGURED = "storage_not_configured"
PROCESSING_INTERRUPTED = "processing_interrupted"
ANALYSIS_NOT_FOUND = "analysis_not_found"

# policy (not errors)
DISPOSITION_EXCLUDED = "disposition_excluded_by_policy"
BELOW_MINIMUM_DURATION = "call_shorter_than_configured_minimum"
BELOW_MINIMUM_SEGMENTS = "transcript_shorter_than_configured_minimum"

# context (not errors - reported, never fatal)
NO_BRAND_BRAIN = "no_brand_brain"
NO_CUSTOMER_CONTEXT = "no_customer_context"
NO_PRODUCT_CONTEXT = "no_product_context"

REASON_TEXT = {
    NO_AUDIO_OR_TRANSCRIPT: "The request contained neither a call recording nor a transcript.",
    AUDIO_UNREACHABLE: "The call recording could not be fetched.",
    AUDIO_UNUSABLE: "The call recording could not be decoded as audio.",
    AUDIO_TOO_LARGE: "The call recording is larger than this service accepts.",
    TRANSCRIPTION_NOT_CONFIGURED: "Transcription is not configured on this server.",
    TRANSCRIPTION_PROVIDER_ERROR: "The transcription provider returned an error.",
    TRANSCRIPTION_RATE_LIMITED: "The transcription provider rate-limited this request.",
    TRANSCRIPTION_TIMEOUT: "Transcription did not complete in time.",
    TRANSCRIPT_EMPTY: "No speech was found in this call.",
    TRANSCRIPT_TOO_SHORT: "The transcript is too short to evaluate against the sales framework.",
    TRANSCRIPT_TOO_LONG: "The transcript is longer than the analyzer accepts in one pass.",
    ANALYSIS_NOT_CONFIGURED: "The analysis model is not configured on this server.",
    ANALYSIS_PROVIDER_ERROR: "The analysis model returned an error.",
    ANALYSIS_INVALID_OUTPUT: "The analysis model did not return a usable result.",
    ANALYSIS_FAILED_AFTER_TRANSCRIPTION: (
        "The call was transcribed but the analysis failed. The transcript was kept, so a "
        "retry does not re-transcribe."),
    STORAGE_NOT_CONFIGURED: "Analysis storage is not configured on this server.",
    PROCESSING_INTERRUPTED: "Processing was interrupted before it completed. Retry the analysis.",
    ANALYSIS_NOT_FOUND: "No analysis exists with that id.",
    DISPOSITION_EXCLUDED: "This disposition is excluded from full evaluation by configuration.",
    BELOW_MINIMUM_DURATION: "The call is shorter than the configured minimum for evaluation.",
    BELOW_MINIMUM_SEGMENTS: "The transcript is shorter than the configured minimum for evaluation.",
    NO_BRAND_BRAIN: "No Brand Brain is linked to this brand, so brand context was unavailable.",
    NO_CUSTOMER_CONTEXT: "No customer details were supplied, so customer context was unavailable.",
    NO_PRODUCT_CONTEXT: "No product details were supplied, so product context was unavailable.",
}

ANALYZER_NAME = "sales_call_analyzer"
ANALYZER_STATUS = "mvp_v1"

__all__ = [
    "FrameworkConfigError",
    "load_framework", "load_signals", "config_disclosure", "criterion_index",
    "stage_index", "rating_scale", "weighting_mode", "not_applicable_mode",
    "unmet_requirements", "criteria_blocked_by_requirements", "skip_decision",
    "render_for_prompt",
    "RATING_SCALE_UNIFORM", "RATING_SCALE_CONFIGURED",
    "WEIGHTING_EQUAL", "WEIGHTING_CONFIGURED",
    "STATUS_QUEUED", "STATUS_TRANSCRIBING", "STATUS_ANALYZING", "STATUS_SCORING",
    "STATUS_COMPLETED", "STATUS_FAILED", "STATUS_SKIPPED",
    "TERMINAL_STATUSES", "ACTIVE_STATUSES",
    "ROLE_SALES_REP", "ROLE_CUSTOMER", "ROLE_PARTICIPANT", "ROLE_UNKNOWN", "SPEAKER_ROLES",
    "ROLE_BASIS_UNRESOLVED", "ROLE_BASIS_REP_SELF_INTRO", "ROLE_BASIS_CUSTOMER_NAME",
    "ROLE_BASIS_CUSTOMER_ADDRESSED",
    "ROLE_BASIS_ELIMINATION", "ROLE_BASIS_SUPPLIED",
    "REASON_TEXT", "ANALYZER_NAME", "ANALYZER_STATUS",
]
