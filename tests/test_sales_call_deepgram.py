"""Deepgram client: request shape, error classification, retries and secrecy.

No network. httpx.MockTransport stands in for the provider, so the retry and
classification logic is exercised exactly as it will run in production.
"""
from __future__ import annotations

import asyncio
import os
import sys

import httpx
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import deepgram_client as dg  # noqa: E402

OK_BODY = {"metadata": {"duration": 12.0},
           "results": {"utterances": [{"speaker": 0, "start": 0.0, "end": 2.0,
                                       "transcript": "Hello.", "confidence": 0.9}]}}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key-not-real")
    yield
    asyncio.run(dg.aclose())


def install(handler, monkeypatch):
    """Point the module's shared client at a mock transport."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(dg, "_client", client)
    return client


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def no_backoff(monkeypatch):
    """Skip the retry backoff so the retry tests are instant. The original
    sleep is captured first - patching asyncio.sleep with a lambda that calls
    asyncio.sleep would recurse."""
    real_sleep = asyncio.sleep

    async def instant(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant)


# --------------------------------------------------------------------------- config
def test_is_configured_follows_the_environment(monkeypatch):
    assert dg.is_configured() is True
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    assert dg.is_configured() is False


def test_missing_key_is_a_stated_reason_not_a_crash(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    with pytest.raises(dg.DeepgramError) as err:
        run(dg.transcribe("https://example.com/rec.mp3"))
    assert err.value.reason == sca.TRANSCRIPTION_NOT_CONFIGURED
    assert err.value.retryable is False


def test_diarization_is_always_requested():
    params = dg.build_params()
    assert params["diarize"] == "true"
    assert params["utterances"] == "true"     # utterances give us speaker turns
    assert params["model"]


def test_language_is_detected_unless_pinned(monkeypatch):
    monkeypatch.setattr(dg, "DEEPGRAM_LANGUAGE", None)
    monkeypatch.setattr(dg, "DEEPGRAM_DETECT_LANGUAGE", True)
    assert dg.build_params().get("detect_language") == "true"
    assert dg.build_params("en-IN")["language"] == "en-IN"
    assert "detect_language" not in dg.build_params("en-IN")


# --------------------------------------------------------------------------- happy path
def test_url_ingestion_sends_the_url_and_the_key_in_a_header(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return httpx.Response(200, json=OK_BODY)

    install(handler, monkeypatch)
    monkeypatch.setattr(dg, "USE_URL_INGESTION", True)

    result = run(dg.transcribe("https://recordings.example.com/rec.mp3"))
    assert result["metadata"]["duration"] == 12.0
    assert seen["auth"] == "Token test-key-not-real"
    assert "diarize=true" in seen["url"]
    assert "recordings.example.com/rec.mp3" in seen["body"]
    # The key must never travel in the query string.
    assert "test-key-not-real" not in seen["url"]


def test_byte_ingestion_fetches_then_posts(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, content=b"ID3fake-audio-bytes",
                                  headers={"content-type": "audio/mpeg"})
        assert request.content == b"ID3fake-audio-bytes"
        return httpx.Response(200, json=OK_BODY)

    install(handler, monkeypatch)
    monkeypatch.setattr(dg, "USE_URL_INGESTION", False)

    assert run(dg.transcribe("https://recordings.example.com/rec.mp3"))["metadata"]
    assert calls == ["GET", "POST"]


def test_oversized_audio_is_refused_before_it_is_buffered(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5000,
                              headers={"content-length": "5000", "content-type": "audio/mpeg"})

    install(handler, monkeypatch)
    monkeypatch.setattr(dg, "USE_URL_INGESTION", False)
    monkeypatch.setattr(dg, "MAX_AUDIO_BYTES", 1000)

    with pytest.raises(dg.DeepgramError) as err:
        run(dg.transcribe("https://recordings.example.com/big.mp3"))
    assert err.value.reason == sca.AUDIO_TOO_LARGE


# --------------------------------------------------------------------------- failures
def test_server_error_is_retried_then_reported(monkeypatch, no_backoff):
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, text="upstream unavailable")

    install(handler, monkeypatch)
    monkeypatch.setattr(dg, "USE_URL_INGESTION", True)

    with pytest.raises(dg.DeepgramError) as err:
        run(dg.transcribe("https://recordings.example.com/rec.mp3", max_retries=2))
    assert err.value.reason == sca.TRANSCRIPTION_PROVIDER_ERROR
    assert attempts["n"] == 3       # first try plus two retries


def test_a_transient_error_that_then_succeeds_returns_the_result(monkeypatch, no_backoff):
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(500, text="blip")
        return httpx.Response(200, json=OK_BODY)

    install(handler, monkeypatch)
    monkeypatch.setattr(dg, "USE_URL_INGESTION", True)

    assert run(dg.transcribe("https://x/rec.mp3", max_retries=2))["metadata"]
    assert attempts["n"] == 2


def test_rate_limit_is_retryable(monkeypatch, no_backoff):
    install(lambda request: httpx.Response(429, text="slow down"), monkeypatch)
    monkeypatch.setattr(dg, "USE_URL_INGESTION", True)

    with pytest.raises(dg.DeepgramError) as err:
        run(dg.transcribe("https://x/rec.mp3", max_retries=1))
    assert err.value.reason == sca.TRANSCRIPTION_RATE_LIMITED
    assert err.value.retryable is True


def test_bad_credentials_are_never_retried(monkeypatch):
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(401, text="unauthorized")

    install(handler, monkeypatch)
    monkeypatch.setattr(dg, "USE_URL_INGESTION", True)

    with pytest.raises(dg.DeepgramError) as err:
        run(dg.transcribe("https://x/rec.mp3", max_retries=3))
    assert err.value.reason == sca.TRANSCRIPTION_NOT_CONFIGURED
    assert attempts["n"] == 1       # a bad key never fixes itself by retrying


def test_timeout_is_classified_as_a_timeout(monkeypatch, no_backoff):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    install(handler, monkeypatch)
    monkeypatch.setattr(dg, "USE_URL_INGESTION", True)

    with pytest.raises(dg.DeepgramError) as err:
        run(dg.transcribe("https://x/rec.mp3", max_retries=0))
    assert err.value.reason == sca.TRANSCRIPTION_TIMEOUT


def test_unreachable_recording_is_its_own_reason(monkeypatch):
    install(lambda request: httpx.Response(404, text="gone"), monkeypatch)
    monkeypatch.setattr(dg, "USE_URL_INGESTION", False)

    with pytest.raises(dg.DeepgramError) as err:
        run(dg.transcribe("https://x/missing.mp3", max_retries=0))
    assert err.value.reason == sca.AUDIO_UNREACHABLE


def test_missing_url_fails_before_any_call(monkeypatch):
    def handler(request):
        raise AssertionError("no request should be made")

    install(handler, monkeypatch)
    with pytest.raises(dg.DeepgramError) as err:
        run(dg.transcribe("   "))
    assert err.value.reason == sca.AUDIO_UNREACHABLE


# --------------------------------------------------------------------------- secrecy
def test_only_the_host_of_a_signed_url_is_loggable():
    signed = "https://storage.example.com/rec.mp3?X-Amz-Signature=deadbeefsecret"
    safe = dg._safe_url(signed)
    assert safe == "storage.example.com"
    assert "deadbeefsecret" not in safe


def test_error_messages_never_carry_the_key(monkeypatch):
    install(lambda request: httpx.Response(403, text="forbidden"), monkeypatch)
    monkeypatch.setattr(dg, "USE_URL_INGESTION", True)
    with pytest.raises(dg.DeepgramError) as err:
        run(dg.transcribe("https://x/rec.mp3", max_retries=0))
    assert "test-key-not-real" not in str(err.value)
