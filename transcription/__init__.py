"""
Shared transcription. Deepgram, used by more than one feature.

WHY THIS IS A TOP-LEVEL PACKAGE
    It began inside sales_call_analyzer, which was right while one feature used
    it. Vision Lab needs the same client for the audio track of a video ad, and
    `vision_lab` importing from `sales_call_analyzer` would be the wrong
    dependency edge - a change made for sales calls would then be able to break
    creative analysis, and the next feature that needs speech would inherit the
    same tangle.

    So the client moved here and both features import from a shared place.
    `sales_call_analyzer.deepgram_client` still exists as a re-export, because
    nothing about that feature should have to change to make room for this one.

REASON CODES LIVE HERE
    They used to come from sales_call_analyzer/__init__.py. Keeping them there
    would have left the shared client importing from one of its own consumers.
"""

# Stable identifiers. Both features branch on these; do not reword them.
AUDIO_UNREACHABLE = "audio_unreachable"
AUDIO_TOO_LARGE = "audio_too_large"
TRANSCRIPTION_NOT_CONFIGURED = "transcription_not_configured"
TRANSCRIPTION_PROVIDER_ERROR = "transcription_provider_error"
TRANSCRIPTION_RATE_LIMITED = "transcription_rate_limited"
TRANSCRIPTION_TIMEOUT = "transcription_timeout"

from .deepgram_client import (  # noqa: E402,F401
    DeepgramError,
    aclose,
    build_params,
    get_client,
    is_configured,
    transcribe,
)
