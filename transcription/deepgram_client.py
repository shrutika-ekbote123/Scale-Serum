"""
Deepgram pre-recorded transcription + speaker diarization.

WHY httpx AND NOT THE SDK
    One endpoint, one POST, and we need exact control of timeouts, retries and
    what is allowed into a log line. httpx is already installed (google-genai
    depends on it) and the repo's style is plain HTTP against services rather
    than an SDK per vendor.

SECURITY
    The API key is read from the environment, sent in an Authorization header,
    and never appears in a log line, an error message, a URL or a response.
    Signed audio URLs are bearer credentials too, so they are never logged
    either - only their host is.

AVAILABILITY
    A missing key does NOT stop the app booting. GEMINI_API_KEY raises at import
    in app.py because nothing works without it; Deepgram is one feature of many,
    so an unset key degrades that feature and reports
    `transcription_not_configured`, matching how MONGODB_URI and the purchase
    probability model already degrade.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from . import (
    AUDIO_TOO_LARGE,
    AUDIO_UNREACHABLE,
    TRANSCRIPTION_NOT_CONFIGURED,
    TRANSCRIPTION_PROVIDER_ERROR,
    TRANSCRIPTION_RATE_LIMITED,
    TRANSCRIPTION_TIMEOUT,
)

logger = logging.getLogger("transcription.deepgram")

DEEPGRAM_ENDPOINT = os.environ.get("DEEPGRAM_ENDPOINT", "https://api.deepgram.com/v1/listen")

# Model choice is env-driven, exactly like GEMINI_MODEL, so it can be changed
# without a deploy.
#
# WHY nova-3 WITH language=multi - measured 2026-09-17 on four real calls
#     The previous default, nova-2 with detect_language, picks ONE language for
#     the whole file. Our reps open in English and do ~85% of the talking, so
#     detection answers "en" with high confidence and the customer's Hindi or
#     Marathi is transcribed as garbled English or dropped. On a real
#     English-then-Hindi call that lost half the conversation: 107 words against
#     215 with multi, and one speaker found instead of two.
#
#     language=multi is Deepgram's code-switching mode. One pass covers a call
#     that is all English, all Hindi, or English switching into Hindi.
DEEPGRAM_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-3")

# Sent when the request carries no options.language_hint.
#
# multi does NOT cover Marathi, Tamil, Telugu, Kannada, Bengali, Gujarati,
# Punjabi or Urdu - Hindi is its only Indian language. Those calls need their
# code passed as language_hint (nova-3 with language=mr produced correct Marathi
# where multi drifted into Hindi). Detecting them automatically is Phase 2.
DEEPGRAM_LANGUAGE = os.environ.get("DEEPGRAM_LANGUAGE") or "multi"

# Deepgram's own language detection is OFF because it is confidently wrong on
# these calls: it answered "en" at 98.5% confidence for a Marathi customer even
# when handed only that customer's audio. Set true to restore the old behaviour.
DEEPGRAM_DETECT_LANGUAGE = os.environ.get("DEEPGRAM_DETECT_LANGUAGE", "false").lower() == "true"

# Never sent today, so Deepgram merges all channels into one stream. Deepgram
# bills each channel separately when multichannel is on (a 10-minute stereo file
# becomes 20 billed minutes), so this is recorded on every analysis for billing.
MULTICHANNEL = False

DEEPGRAM_TIMEOUT_SECONDS = float(os.environ.get("DEEPGRAM_TIMEOUT_SECONDS", 300))
DEEPGRAM_CONNECT_TIMEOUT = float(os.environ.get("DEEPGRAM_CONNECT_TIMEOUT", 15))
DEEPGRAM_MAX_RETRIES = int(os.environ.get("DEEPGRAM_MAX_RETRIES", 2))

# Byte-forwarding path only. The URL path never loads the file into this process.
MAX_AUDIO_BYTES = int(os.environ.get("SCA_MAX_AUDIO_BYTES", 60 * 1024 * 1024))

# Prefer handing Deepgram the URL: no bytes through this box, no memory spike on
# the single pm2 fork process. Set false when recordings live somewhere only
# this service can reach.
USE_URL_INGESTION = os.environ.get("SCA_DEEPGRAM_URL_INGESTION", "true").lower() == "true"

_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


class DeepgramError(Exception):
    """A transcription failure with a stable reason code and a retry verdict."""

    def __init__(self, reason: str, message: str, *, retryable: bool = False,
                 status_code: Optional[int] = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.retryable = retryable
        self.status_code = status_code


def is_configured() -> bool:
    return bool(os.environ.get("DEEPGRAM_API_KEY"))


def _api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise DeepgramError(
            TRANSCRIPTION_NOT_CONFIGURED,
            "DEEPGRAM_API_KEY is not set on this server.", retryable=False)
    return key


def _safe_url(url: str) -> str:
    """A signed URL is a credential. Only its host is ever safe to log."""
    try:
        return urlsplit(url).netloc or "unknown-host"
    except Exception:
        return "unparseable-url"


async def get_client() -> httpx.AsyncClient:
    """One reusable client for the process, like the Gemini client in app.py."""
    global _client
    if _client is not None and not _client.is_closed:
        return _client
    async with _client_lock:
        if _client is None or _client.is_closed:
            _client = httpx.AsyncClient(
                timeout=httpx.Timeout(DEEPGRAM_TIMEOUT_SECONDS,
                                      connect=DEEPGRAM_CONNECT_TIMEOUT),
                follow_redirects=True,
            )
        return _client


async def aclose() -> None:
    """Called from the app's shutdown hook."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def effective_language(language: Optional[str] = None) -> Optional[str]:
    """The `language` value actually sent, or None when Deepgram detects it.
    Billing needs it: language=multi is priced at the multilingual rate."""
    return language or DEEPGRAM_LANGUAGE


# Which single language codes nova-3 accepts, by what they mean for us.
#
#   MULTI_COVERED   already handled by language=multi. Asking for "hi" by name
#                   would transcribe the rep's English as broken Hindi, so a
#                   detected en or hi means "change nothing".
#   REGIONAL        NOT in multi (Hindi is its only Indian language). These must
#                   be requested by name or the customer's half is lost.
#   UNSUPPORTED     Deepgram has no model for these at all.
MULTI_COVERED_LANGUAGES = {"en", "hi"}
REGIONAL_LANGUAGES = {"mr", "ta", "te", "kn", "bn", "gu", "pa", "ur"}
UNSUPPORTED_LANGUAGES = {"ml"}

LANGUAGE_COVERED_BY_MULTI = "covered_by_multi"
LANGUAGE_REGIONAL = "regional_code"
LANGUAGE_UNSUPPORTED = "unsupported_by_provider"
LANGUAGE_UNKNOWN = "unknown_language_code"


def language_for(detected: Optional[str]) -> tuple[Optional[str], str]:
    """(language to send, why). None means "leave the default alone".

    An unknown or unsupported language deliberately falls back to the default
    rather than being passed through: sending a code Deepgram does not accept
    fails the whole transcription, and a wrong-but-complete transcript beats no
    transcript at all.
    """
    code = (detected or "").strip().lower()
    if not code or code in MULTI_COVERED_LANGUAGES:
        return None, LANGUAGE_COVERED_BY_MULTI
    if code in REGIONAL_LANGUAGES:
        return code, LANGUAGE_REGIONAL
    if code in UNSUPPORTED_LANGUAGES:
        return None, LANGUAGE_UNSUPPORTED
    return None, LANGUAGE_UNKNOWN


def build_params(language: Optional[str] = None) -> dict[str, Any]:
    """Query parameters for a pre-recorded request.

    `diarize` is the reason this integration exists. `utterances` is what turns
    word-level diarization into speaker turns, which is exactly our internal
    segment shape - without it we would be re-implementing turn segmentation
    from words.
    """
    params: dict[str, Any] = {
        "model": DEEPGRAM_MODEL,
        "diarize": "true",
        "utterances": "true",
        "punctuate": "true",
        "smart_format": "true",
    }
    if MULTICHANNEL:
        params["multichannel"] = "true"
    chosen = effective_language(language)
    if chosen:
        params["language"] = chosen
    elif DEEPGRAM_DETECT_LANGUAGE:
        params["detect_language"] = "true"
    return params


def _classify(status: int, body: str) -> DeepgramError:
    if status == 429:
        return DeepgramError(TRANSCRIPTION_RATE_LIMITED,
                             "Transcription provider rate limit reached.",
                             retryable=True, status_code=status)
    if status in (401, 403):
        # A bad key never fixes itself by retrying.
        return DeepgramError(TRANSCRIPTION_NOT_CONFIGURED,
                             "Transcription provider rejected the credentials.",
                             retryable=False, status_code=status)
    if status >= 500:
        return DeepgramError(TRANSCRIPTION_PROVIDER_ERROR,
                             f"Transcription provider error (HTTP {status}).",
                             retryable=True, status_code=status)
    return DeepgramError(TRANSCRIPTION_PROVIDER_ERROR,
                         f"Transcription request rejected (HTTP {status}): {body[:200]}",
                         retryable=False, status_code=status)


async def _post(client: httpx.AsyncClient, *, params: dict, headers: dict,
                json_body: Optional[dict] = None,
                content: Optional[bytes] = None) -> dict:
    try:
        response = await client.post(DEEPGRAM_ENDPOINT, params=params, headers=headers,
                                     json=json_body, content=content)
    except httpx.TimeoutException as err:
        raise DeepgramError(TRANSCRIPTION_TIMEOUT,
                            "Transcription did not complete in time.",
                            retryable=True) from err
    except httpx.HTTPError as err:
        raise DeepgramError(TRANSCRIPTION_PROVIDER_ERROR,
                            f"Could not reach the transcription provider: {type(err).__name__}",
                            retryable=True) from err

    if response.status_code >= 400:
        raise _classify(response.status_code, response.text)

    try:
        return response.json()
    except ValueError as err:
        raise DeepgramError(TRANSCRIPTION_PROVIDER_ERROR,
                            "Transcription provider returned a non-JSON response.",
                            retryable=True) from err


async def _fetch_audio(client: httpx.AsyncClient, url: str) -> tuple[bytes, Optional[str]]:
    """Byte-forwarding path. Streamed and capped so one large upload cannot
    exhaust the single worker process."""
    try:
        async with client.stream("GET", url) as response:
            if response.status_code >= 400:
                raise DeepgramError(
                    AUDIO_UNREACHABLE,
                    f"The recording could not be fetched (HTTP {response.status_code}).",
                    retryable=response.status_code >= 500)
            declared = response.headers.get("content-length")
            if declared and int(declared) > MAX_AUDIO_BYTES:
                raise DeepgramError(AUDIO_TOO_LARGE,
                                    "The recording is larger than this service accepts.",
                                    retryable=False)
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise DeepgramError(AUDIO_TOO_LARGE,
                                        "The recording is larger than this service accepts.",
                                        retryable=False)
                chunks.append(chunk)
            return b"".join(chunks), response.headers.get("content-type")
    except DeepgramError:
        raise
    except httpx.TimeoutException as err:
        raise DeepgramError(AUDIO_UNREACHABLE, "Fetching the recording timed out.",
                            retryable=True) from err
    except httpx.HTTPError as err:
        raise DeepgramError(AUDIO_UNREACHABLE,
                            f"The recording could not be fetched: {type(err).__name__}",
                            retryable=True) from err


async def fetch_audio(audio_url: str) -> tuple[bytes, Optional[str]]:
    """Download a recording into memory, capped at MAX_AUDIO_BYTES.

    Public because language identification needs the bytes even when Deepgram
    itself is handed the URL. Same client, same cap, same reason codes - one
    place that knows how to fetch a recording.
    """
    if not (audio_url or "").strip():
        raise DeepgramError(AUDIO_UNREACHABLE, "No recording URL was supplied.",
                            retryable=False)
    return await _fetch_audio(await get_client(), audio_url)


async def transcribe(audio_url: str, *, mime_type: Optional[str] = None,
                     language: Optional[str] = None,
                     max_retries: Optional[int] = None) -> dict:
    """Transcribe and diarize a recording. Returns the raw Deepgram response.

    Normalisation is transcript.py's job - this module never interprets the
    payload, so a change in Deepgram's response shape has exactly one place to
    be handled.
    """
    key = _api_key()
    if not (audio_url or "").strip():
        raise DeepgramError(AUDIO_UNREACHABLE, "No recording URL was supplied.",
                            retryable=False)

    client = await get_client()
    params = build_params(language)
    attempts = DEEPGRAM_MAX_RETRIES if max_retries is None else max_retries
    host = _safe_url(audio_url)

    last: Optional[DeepgramError] = None
    for attempt in range(attempts + 1):
        try:
            if USE_URL_INGESTION:
                headers = {"Authorization": f"Token {key}", "Content-Type": "application/json"}
                return await _post(client, params=params, headers=headers,
                                   json_body={"url": audio_url})

            audio, detected_type = await _fetch_audio(client, audio_url)
            if not audio:
                raise DeepgramError(AUDIO_UNREACHABLE, "The recording was empty.",
                                    retryable=False)
            headers = {"Authorization": f"Token {key}",
                       "Content-Type": mime_type or detected_type or "application/octet-stream"}
            return await _post(client, params=params, headers=headers, content=audio)

        except DeepgramError as err:
            last = err
            if not err.retryable or attempt >= attempts:
                break
            # Exponential backoff with jitter, so a provider blip does not turn
            # into a synchronised retry storm from every queued job.
            delay = min(2.0 ** attempt, 8.0) + random.uniform(0, 0.5)
            logger.warning(
                "deepgram attempt %s/%s failed (%s), retrying in %.1fs [host=%s]",
                attempt + 1, attempts + 1, err.reason, delay, host)
            await asyncio.sleep(delay)

    assert last is not None
    logger.error("deepgram failed: %s [host=%s status=%s]", last.reason, host, last.status_code)
    raise last
