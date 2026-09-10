"""Vision Lab - AI-predicted attention analysis for ad creatives.

Pipeline (see pipeline.py for the orchestration):

    creative URL (image or video)
      -> media.py         ffmpeg: probe, sample frames, detect shots, extract audio
      -> saliency.py      ONNX: where a human eye would land, per frame
      -> regions.py       OCR, faces, brand mark, CTA - what is actually on screen
      -> measure.py       the two joined into one measurement record per frame
      -> timeline.py      the attention index, weak zones, key moments
      -> transcript       Deepgram, joined to the timeline by time span
      -> psychology.py    the measured psychological triggers
      -> analyzer.py      Gemini: interpretation, ordinal ratings, fix prose
      -> evidence.py      every claim checked against the measurement record
      -> scoring.py       measurements + ratings -> six scores, in Python
      -> report.py        assembled, never rescored
      -> store.py         persisted to MongoDB

DIVISION OF LABOUR (these are correctness requirements, not style preferences)
    CV        measures: where gaze goes, what is on screen, when, how big, how long.
    Deepgram  measures: what is said, and when.
    Gemini    interprets: what the creative is doing, ordinal ratings, prose.
    Python    validates, scores, aggregates, persists and orchestrates.

    Gemini never produces a number. It produces an ordinal rating per criterion
    and the evidence for it; scoring.py turns measurements and ratings into
    numbers using vision_framework.json. That is what makes a score
    reproducible, re-derivable under new weights, and immune to anything
    written in the ad copy.

A SALIENCY MAP IS NOT AN ATTENTION CURVE
    Saliency maps are spatial probability distributions - every frame sums to 1,
    so the model cannot say whether a frame held the viewer. The per-second
    "attention" figure is a DERIVED COMPOSITE INDEX over concentration, motion,
    face presence, gaze stability, text load and novelty. It is reported as
    basis="derived_composite" and must never be described as measured attention.

NO INVENTED BUSINESS VALUES
    Metric weights, score bands, the weak-zone threshold and the timeline
    coefficients are all null in vision_framework.json because management has
    not set them. Placeholders are neutral (equal weighting, uniform spacing)
    and every response reports which values were still unconfirmed.

ASYNCHRONOUS BY DESIGN
    Frame decode, saliency inference and OCR are CPU-bound and take one to two
    minutes. POST returns an analysis_id immediately and the caller polls the
    GET. Processing runs in a SEPARATE pm2 process (worker.py) - not a
    BackgroundTask - because CPU-bound work in the API process would stall
    onboarding, Script Lab and every sales-call poll for its duration.
"""
from .framework import (  # noqa: F401
    FrameworkConfigError,
    METRIC_IDS,
    WEIGHTING_CONFIGURED,
    WEIGHTING_EQUAL,
    bands,
    config_disclosure,
    load_framework,
    load_psychology,
    metric_index,
    reading_speed,
    sampling,
    trigger_index,
    weak_zone_rule,
    weighting_mode,
)

# --------------------------------------------------------------------------- versions
# BUMP THIS WHENEVER A REPORT GAINS OR LOSES SOMETHING A CALLER WOULD NOTICE.
#
# It is part of the idempotency fingerprint, so bumping it means the next
# identical request is re-analysed rather than served from the previous result.
# Forgetting to bump it is not a cosmetic slip: after Milestone D added the
# transcript, the triggers and the recommendations, a re-submitted creative kept
# returning its old Milestone-A report - correct by the fingerprint, and useless
# to the caller, who had no way to tell a stale shape from a new one.
#
# Lives here because both pipeline.py and report.py need it and neither may
# import the other.
PROMPT_VERSION = "vl_2026_09_d"

# --------------------------------------------------------------------------- statuses
# The lifecycle of an analysis. The UI may branch on these, and may be shown to
# the user as progress copy - they are deliberately narrated rather than a
# single "processing".
STATUS_QUEUED = "queued"
STATUS_PROBING = "probing"                    # reading duration / dimensions
STATUS_ANALYZING_FRAMES = "analyzing_frames"  # saliency + OCR + detection
STATUS_TRANSCRIBING = "transcribing"
STATUS_INTERPRETING = "interpreting"          # the LLM pass
STATUS_SCORING = "scoring"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"                    # analysed, but not scored by policy

TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED)
ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_PROBING, STATUS_ANALYZING_FRAMES,
                   STATUS_TRANSCRIBING, STATUS_INTERPRETING, STATUS_SCORING)
# Of those, the ones in which a WORKER HOLDS the job and writes a heartbeat.
# `queued` is not among them: nobody holds a queued job, so there is nothing
# that could have stopped reporting. See store.is_stale.
CLAIMED_STATUSES = (STATUS_PROBING, STATUS_ANALYZING_FRAMES, STATUS_TRANSCRIBING,
                    STATUS_INTERPRETING, STATUS_SCORING)

# --------------------------------------------------------------------------- creative kinds
KIND_VIDEO = "video"
KIND_IMAGE = "image"
CREATIVE_KINDS = (KIND_VIDEO, KIND_IMAGE)

# --------------------------------------------------------------------------- key moments
MOMENT_PEAK = "peak"    # global maximum of the attention index
MOMENT_HERO = "hero"    # strongest moment in the second half
MOMENT_KEY = "key"      # first frame where the message region takes majority mass
MOMENT_WEAK = "weak"    # inside a detected weak zone
MOMENT_LABELS = (MOMENT_PEAK, MOMENT_HERO, MOMENT_KEY, MOMENT_WEAK)

# --------------------------------------------------------------------------- trigger status
TRIGGER_PRESENT = "present"
TRIGGER_WEAK = "weak"
TRIGGER_ABSENT = "absent"
TRIGGER_NOT_APPLICABLE = "not_applicable"   # the creative had no opportunity - NOT a zero
TRIGGER_UNSUPPORTED = "unsupported"         # claimed, but evidence failed verification

# --------------------------------------------------------------------------- reason codes
# Stable identifiers. The UI and the backend may branch on these; do not reword
# them without telling both teams.

# input
NO_CREATIVE = "no_creative"
CREATIVE_UNREACHABLE = "creative_unreachable"
CREATIVE_UNUSABLE = "creative_unusable"
CREATIVE_TOO_LARGE = "creative_too_large"
UNSUPPORTED_FORMAT = "unsupported_format"
DURATION_TOO_LONG = "duration_too_long"
URL_NOT_ALLOWED = "url_not_allowed"

# media
DECODE_FAILED = "decode_failed"
NO_VIDEO_STREAM = "no_video_stream"
NO_AUDIO_STREAM = "no_audio_stream"          # reported, never fatal
FFMPEG_UNAVAILABLE = "ffmpeg_unavailable"

# vision
SALIENCY_MODEL_UNAVAILABLE = "saliency_model_unavailable"
OCR_UNAVAILABLE = "ocr_unavailable"          # reported, never fatal
INFERENCE_FAILED = "inference_failed"

# transcription
TRANSCRIPTION_NOT_CONFIGURED = "transcription_not_configured"
TRANSCRIPTION_PROVIDER_ERROR = "transcription_provider_error"
TRANSCRIPT_EMPTY = "transcript_empty"

# analysis
ANALYSIS_NOT_CONFIGURED = "analysis_not_configured"
ANALYSIS_PROVIDER_ERROR = "analysis_provider_error"
ANALYSIS_INVALID_OUTPUT = "analysis_invalid_output"

# storage / lifecycle
STORAGE_NOT_CONFIGURED = "storage_not_configured"
OBJECT_STORAGE_NOT_CONFIGURED = "object_storage_not_configured"
UPLOAD_FAILED = "upload_failed"               # POST /upload could not reach S3
PROCESSING_INTERRUPTED = "processing_interrupted"
ANALYSIS_NOT_FOUND = "analysis_not_found"
VISION_LAB_DISABLED = "vision_lab_disabled"

# context (not errors - reported, never fatal)
NO_BRAND_BRAIN = "no_brand_brain"
NO_BRAND_ASSETS = "no_brand_assets"

REASON_TEXT = {
    NO_CREATIVE: "No creative was supplied to analyse.",
    CREATIVE_UNREACHABLE: "The creative could not be downloaded. The link may have expired.",
    CREATIVE_UNUSABLE: "The file could not be read as an image or a video.",
    CREATIVE_TOO_LARGE: "The creative is larger than this service accepts.",
    UNSUPPORTED_FORMAT: "That file type is not supported. Use JPG, PNG, WebP, MP4, MOV or WebM.",
    DURATION_TOO_LONG: "The video is longer than this service analyses.",
    URL_NOT_ALLOWED: "The creative URL is not on the allowed host list.",
    DECODE_FAILED: "The video could not be decoded.",
    NO_VIDEO_STREAM: "The file contains no video track.",
    NO_AUDIO_STREAM: "The video has no audio track, so there is no transcript.",
    FFMPEG_UNAVAILABLE: "Video processing is not available on this server.",
    SALIENCY_MODEL_UNAVAILABLE: "The attention model is not installed on this server.",
    OCR_UNAVAILABLE: "On-screen text could not be read, so text-based scores are reduced.",
    INFERENCE_FAILED: "The attention analysis did not complete.",
    TRANSCRIPTION_NOT_CONFIGURED: "Transcription is not configured on this server.",
    TRANSCRIPTION_PROVIDER_ERROR: "The transcription provider could not be reached.",
    TRANSCRIPT_EMPTY: "No speech was found in the audio.",
    ANALYSIS_NOT_CONFIGURED: "The interpretation step is not configured on this server.",
    ANALYSIS_PROVIDER_ERROR: "The interpretation step could not be reached.",
    ANALYSIS_INVALID_OUTPUT: "The interpretation step returned something unusable.",
    STORAGE_NOT_CONFIGURED: "Analysis storage is not configured. Set MONGODB_URI.",
    OBJECT_STORAGE_NOT_CONFIGURED: "Heatmap storage is not configured.",
    UPLOAD_FAILED: "The file could not be stored. Try the upload again.",
    PROCESSING_INTERRUPTED: "Processing was interrupted before it completed. Retry the analysis.",
    ANALYSIS_NOT_FOUND: "No analysis exists with that id.",
    VISION_LAB_DISABLED: "Vision Lab is not enabled on this server.",
    NO_BRAND_BRAIN: "No Brand Brain was supplied, so brand context was not used.",
    NO_BRAND_ASSETS: "No brand mark was supplied, so brand detection used the brand name only.",
}
