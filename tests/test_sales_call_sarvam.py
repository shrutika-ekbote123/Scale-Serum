"""The analyzer with SCA_TRANSCRIBER=sarvam.

Sarvam transcribes; its transcript is cleaned (transcription/sarvam_cleanup.py)
and the flagged turns re-checked by Gemini; no Gemini language identification,
no segment pass, no Deepgram charge. A Sarvam failure falls back to Deepgram.
Roles: South Indian introductions and honorifics, and the rep's voiceprint.

No network: Sarvam, Gemini, ffmpeg and the voice model are fakes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import billing  # noqa: E402
from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import speakers as sp  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from sales_call_analyzer.models import CustomerInfo, ProductInfo, RepInfo  # noqa: E402
from transcription import sarvam_client  # noqa: E402
from transcription import sarvam_recheck as rc  # noqa: E402
from test_sales_call_pipeline import (  # noqa: E402
    FakeCollection,
    cfg,  # noqa: F401 - fixture
    create_and_run,
    make_deps,
    make_request,
    run,
    store,  # noqa: F401 - fixture
    collection,  # noqa: F401 - fixture
)

RATE = 16000


def sarvam_response(language="mr-IN"):
    entries = [
        (0, 0.0, 6.0, "Hi Meera, this is Rajan Kumar from Ed Tek Pro. Is now a good time?"),
        (1, 6.4, 12.0, "Yes. My main problem is I don't have a recognised certification."),
        (1, 4.6, 5.4, "Is now a good time"),                     # echo of the rep
        (0, 12.5, 20.0, "That is exactly what the free fund policy and our programme cover."),
        (1, 20.5, 24.0, "ହଁ okay, tell me more."),                 # stray Odia
    ]
    return {"language_code": language, "transcript": " ".join(e[3] for e in entries),
            "diarized_transcript": {"entries": [
                {"speaker_id": str(s), "start_time_seconds": a, "end_time_seconds": b,
                 "transcript": t} for s, a, b, t in entries]}}


@pytest.fixture
def decoded(monkeypatch):
    """ffmpeg stand-in: 30 s of silence for any audio."""
    async def fake_decode(audio, *, channels=1):
        return np.zeros((30 * RATE, channels), np.float32)
    monkeypatch.setattr(pl._audio_slice, "decode", fake_decode)


def sarvam_deps(store, cfg, *, response=None, result=None, recheck=None):
    deps = make_deps(store, cfg)
    calls = deps.extra["calls"]
    calls.update({"sarvam": 0, "recheck": [], "language_id": 0})

    async def fetch_audio(url):
        return b"audio-bytes", "audio/mpeg"

    async def sarvam_transcribe(audio, *, filename):
        calls["sarvam"] += 1
        calls["filename"] = filename
        return result or sarvam_client.SarvamResult(ok=True, response=response or sarvam_response(),
                                                    ms=1200)

    async def identify_language(*a, **k):
        calls["language_id"] += 1
        raise AssertionError("language identification must not run with Sarvam")

    async def sarvam_recheck(samples, entries, indices, terms, brand_terms):
        calls["recheck"].append({"indices": list(indices), "terms": list(terms)})
        if recheck:
            return recheck(entries, indices)
        out = rc.RecheckResult(ok=True, requests=1, input_tokens=800, output_tokens=100,
                               thinking_tokens=200)
        for e in entries:
            if e.index in indices and "free fund" in e.text:
                out.changes.append({"index": e.index, "before": e.text,
                                    "after": e.text.replace("free fund", "refund")})
                e.text = e.text.replace("free fund", "refund")
        return out

    deps.transcriber = pl.TRANSCRIBER_SARVAM
    deps.fetch_audio = fetch_audio
    deps.sarvam_transcribe = sarvam_transcribe
    deps.sarvam_recheck = sarvam_recheck
    deps.sarvam_model = "saaras:v3"
    deps.identify_language = identify_language
    deps.language_id_mode = pl.LANGUAGE_ID_ON
    deps.segment_pass_mode = pl.SEGMENT_PASS_ON
    return deps


def request_with_terms():
    return make_request(product=ProductInfo(name="Career Accelerator", price=200000,
                                            currency="INR", terms=["refund", "ChatGPT"]))


# =========================================================================== #
# The Sarvam path
# =========================================================================== #
def test_sarvam_transcribes_and_deepgram_is_not_called(store, cfg, decoded):
    deps = sarvam_deps(store, cfg)
    _, report, deps = run(create_and_run(store, cfg, request_with_terms(), deps))
    calls = deps.extra["calls"]
    assert calls["sarvam"] == 1 and calls["transcribe"] == 0 and calls["language_id"] == 0
    assert calls["filename"] == "a.mp3"
    doc = store.collection.docs[next(iter(store.collection.docs))]
    processing = doc["processing"]
    assert processing["transcription_provider"] == "sarvam"
    assert processing["transcription_model"] == "saaras:v3"
    assert processing["language_detected"] == "mr"
    assert processing["language_basis"] == pl.LANGUAGE_BASIS_TRANSCRIBER
    assert doc["transcript"]["source"] == "sarvam"


def test_the_clean_up_runs_before_anything_reads_the_transcript(store, cfg, decoded):
    deps = sarvam_deps(store, cfg)
    run(create_and_run(store, cfg, request_with_terms(), deps))
    doc = store.collection.docs[next(iter(store.collection.docs))]
    texts = [s["text"] for s in doc["transcript"]["segments"]]
    assert "Is now a good time" not in texts                                   # echo removed
    assert any("EdTech Pro" in t for t in texts)                               # brand fixed
    assert any(t.startswith("हँ okay") for t in texts)                         # script fixed
    assert any("refund policy" in t for t in texts)                            # re-checked
    counts = doc["processing"]["sarvam"]["cleanup"]["counts"]
    assert counts["duplicates_removed"] == 1 and counts["script_fixes"] == 1


def test_recheck_gets_the_flagged_turns_and_the_product_terms(store, cfg, decoded):
    deps = sarvam_deps(store, cfg)
    run(create_and_run(store, cfg, request_with_terms(), deps))
    sent = deps.extra["calls"]["recheck"]
    assert len(sent) == 1
    assert "refund" in sent[0]["terms"] and "EdTech Pro" in sent[0]["terms"]
    # people's names are never handed to the clean-up as terms to write in
    assert "Rajan Kumar" not in sent[0]["terms"]


def test_recheck_off_leaves_the_flagged_turns(store, cfg, decoded):
    deps = sarvam_deps(store, cfg)
    deps.sarvam_recheck_mode = "off"
    run(create_and_run(store, cfg, request_with_terms(), deps))
    assert deps.extra["calls"]["recheck"] == []
    doc = store.collection.docs[next(iter(store.collection.docs))]
    assert any("free fund" in s["text"] for s in doc["transcript"]["segments"])


def test_no_segment_pass_after_sarvam(store, cfg, decoded):
    deps = sarvam_deps(store, cfg)
    run(create_and_run(store, cfg, request_with_terms(), deps))
    doc = store.collection.docs[next(iter(store.collection.docs))]
    assert doc["processing"].get("segment_pass") is None


def test_roles_are_resolved_on_the_sarvam_transcript(store, cfg, decoded):
    deps = sarvam_deps(store, cfg)
    run(create_and_run(store, cfg, request_with_terms(), deps))
    doc = store.collection.docs[next(iter(store.collection.docs))]
    roles = {s["speaker_id"]: s["role"] for s in doc["transcript"]["speakers"]}
    assert roles == {"speaker_0": "sales_rep", "speaker_1": "customer"}


def test_cost_bills_sarvam_and_gemini_not_deepgram(store, cfg, decoded):
    deps = sarvam_deps(store, cfg)
    _, report, _ = run(create_and_run(store, cfg, request_with_terms(), deps))
    doc = store.collection.docs[next(iter(store.collection.docs))]
    cost = doc["processing"]["cost"]
    assert cost["deepgram"]["charged"] is False
    assert cost["deepgram"]["reason"] == pl.REASON_TRANSCRIBED_BY_SARVAM
    assert cost["sarvam"]["charged"] is True and cost["sarvam"]["audio_seconds"] == 30.0
    assert cost["sarvam"]["inr"] == pytest.approx(30 / 3600 * 45, abs=1e-3)
    assert cost["gemini"]["input_tokens"] >= 800          # the re-check is billed
    usage = report.usage.cost
    assert usage.sarvam_inr == pytest.approx(0.375, abs=1e-3)
    assert "sarvam_recheck" in report.usage.tokens


# =========================================================================== #
# Falling back to Deepgram
# =========================================================================== #
def test_a_failed_sarvam_job_falls_back_to_deepgram(store, cfg, decoded):
    deps = sarvam_deps(store, cfg, result=sarvam_client.SarvamResult(
        ok=False, reason=sarvam_client.REASON_NO_CREDITS, error="ApiError 402"))
    deps.identify_language = None
    _, report, deps = run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["calls"]["transcribe"] == 1
    doc = store.collection.docs[next(iter(store.collection.docs))]
    processing = doc["processing"]
    assert processing["transcription_provider"] == "deepgram"
    assert processing["sarvam"]["reason"] == sarvam_client.REASON_NO_CREDITS
    assert processing["sarvam"]["fallback"] == "deepgram"
    assert processing["cost"]["deepgram"]["charged"] is True
    assert processing["cost"]["sarvam"]["charged"] is False
    assert report.status == "completed"


def test_sarvam_not_configured_falls_back_to_deepgram(store, cfg, decoded):
    deps = sarvam_deps(store, cfg)
    deps.sarvam_transcribe = None
    deps.identify_language = None
    run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["calls"]["transcribe"] == 1
    doc = store.collection.docs[next(iter(store.collection.docs))]
    assert doc["processing"]["sarvam"]["reason"] == pl.REASON_SARVAM_NOT_CONFIGURED


def test_a_deepgram_transcript_is_not_reused_once_sarvam_is_chosen(store, cfg, decoded):
    run(create_and_run(store, cfg, make_request(), make_deps(store, cfg)))     # Deepgram run
    deps = sarvam_deps(store, cfg)
    run(create_and_run(store, cfg, make_request(), deps))                      # same call_id
    assert deps.extra["calls"]["sarvam"] == 1


def test_a_sarvam_transcript_is_reused_by_a_sarvam_retry(store, cfg, decoded):
    run(create_and_run(store, cfg, make_request(), sarvam_deps(store, cfg)))
    deps = sarvam_deps(store, cfg)
    run(create_and_run(store, cfg, make_request(), deps))
    assert deps.extra["calls"]["sarvam"] == 0


# =========================================================================== #
# The rep's voiceprint picks the rep among Sarvam's speakers
# =========================================================================== #
def test_voiceprint_names_the_rep_when_no_name_is_said(store, cfg, decoded, monkeypatch):
    response = sarvam_response()
    for e in response["diarized_transcript"]["entries"]:          # no names, no brand
        e["transcript"] = "हाँ जी okay" if e["speaker_id"] == "1" else "आपका demo हो गया right"
    deps = sarvam_deps(store, cfg, response=response)
    rep_vector = np.ones(4, np.float32) / 2

    async def load_voiceprint(rep_id):
        return rep_vector if rep_id == "rep_1" else None

    monkeypatch.setattr(pl.diarization_mod, "score_speakers",
                        lambda raw, samples, embed, vp: {"speaker_0": 0.71, "speaker_1": 0.05})
    deps.load_voiceprint = load_voiceprint
    deps.embed_voice = lambda clip: rep_vector
    deps.voice_model = "fake-voice"
    request = make_request(rep=RepInfo(id="rep_1", name="Placeholder Name"),
                           customer=CustomerInfo(name="Someone Else"))
    run(create_and_run(store, cfg, request, deps))
    doc = store.collection.docs[next(iter(store.collection.docs))]
    roles = {s["speaker_id"]: (s["role"], s["role_basis"]) for s in doc["transcript"]["speakers"]}
    assert roles["speaker_0"][0] == "sales_rep" and roles["speaker_1"][0] == "customer"
    assert doc["processing"]["speaker_refinement"]["rep_speaker_id"] == "speaker_0"


def test_a_voice_below_the_match_threshold_names_nobody(store, cfg, decoded, monkeypatch):
    deps = sarvam_deps(store, cfg)

    async def load_voiceprint(rep_id):
        return np.ones(4, np.float32)

    monkeypatch.setattr(pl.diarization_mod, "score_speakers",
                        lambda *a: {"speaker_0": 0.12, "speaker_1": 0.08})
    deps.load_voiceprint = load_voiceprint
    deps.embed_voice = lambda clip: np.ones(4, np.float32)
    run(create_and_run(store, cfg, make_request(rep=RepInfo(id="rep_1", name="Rajan Kumar")), deps))
    doc = store.collection.docs[next(iter(store.collection.docs))]
    assert doc["processing"]["speaker_refinement"]["reason"] == "no_speaker_matches_the_voiceprint"


# =========================================================================== #
# Roles from South Indian introductions (any transcriber)
# =========================================================================== #
def _roles(entries, *, brand=None, customer=None):
    raw = {"metadata": {"duration": 60}, "results": {"utterances": [
        {"speaker": s, "start": a, "end": b, "transcript": t, "confidence": 1.0}
        for s, a, b, t in entries]}}
    t = sp.resolve_roles(tr.from_deepgram(raw), rep=RepInfo(), customer=CustomerInfo(name=customer),
                         brand_name=brand)
    return {s.speaker_id: s.role for s in t.speakers}


def test_malayalam_introduction_names_the_rep():
    roles = _roles([(0, 0, 1, "Hello"),
                    (1, 3, 11, "ഹലോ, ഞാൻ ABC Educational Institute-ൽ നിന്ന് വിളിക്കുകയാണ്.")],
                   brand="ABC Educational Institute")
    assert roles == {"speaker_0": "customer", "speaker_1": "sales_rep"}


def test_tamil_introduction_names_the_rep():
    roles = _roles([(1, 0, 1, "Hello."),
                    (0, 4, 11, "ஹலோ. நான் ABC Educational Institute-ல இருந்து பேசுறேன்.")],
                   brand="ABC Educational Institute")
    assert roles == {"speaker_0": "sales_rep", "speaker_1": "customer"}


def test_telugu_introduction_names_the_rep():
    roles = _roles([(0, 0, 5, "నమస్తే, నేను ABC Hospital నుండి మాట్లాడుతున్నాను"),
                    (1, 6, 8, "చెప్పండి")], brand="ABC Hospital")
    assert roles == {"speaker_0": "sales_rep", "speaker_1": "customer"}


def test_telugu_honorific_after_the_customers_name_names_the_customer():
    # mycall6: the booking assistant addresses the customer "... Mondal గారు".
    roles = _roles([(0, 0, 10, "దయచేసి మీ పేరు తెలపండి"),
                    (1, 11, 15, "name శుభోజిత్ మండల్"),
                    (0, 16, 25, "ధన్యవాదాలు. Bheem Shubhajeet Mondal గారు. ఏ రోజు?"),
                    (1, 26, 28, "General Physician")], customer="Shubhojit Mondal")
    assert roles == {"speaker_0": "sales_rep", "speaker_1": "customer"}


# =========================================================================== #
# Billing
# =========================================================================== #
def test_sarvam_cost_is_rupees_by_the_hour():
    pricing = billing.load_pricing(force=True)
    cost = billing.sarvam_cost(pricing, outcome=billing.CHARGED, model="saaras:v3", audio_seconds=600)
    assert cost.inr == 7.5 and cost.usd == pytest.approx(7.5 / 88, abs=1e-6)
    plain = billing.sarvam_cost(pricing, outcome=billing.CHARGED, model="saaras:v3",
                                audio_seconds=600, diarization=False)
    assert plain.inr == 5.0


def test_unknown_sarvam_model_is_unpriced_not_free():
    pricing = billing.load_pricing(force=True)
    cost = billing.sarvam_cost(pricing, outcome=billing.CHARGED, model="saaras:v9", audio_seconds=60)
    assert cost.charged and cost.usd is None and cost.reason == billing.REASON_RATE_NOT_CONFIGURED


def test_a_deepgram_run_has_an_empty_sarvam_block():
    pricing = billing.load_pricing(force=True)
    dg = billing.deepgram_cost(pricing, outcome=billing.CHARGED, model="nova-3", audio_seconds=60,
                               channels_processed=1, language_sent="multi")
    gm = billing.gemini_cost(pricing, model_requested="gemini-3.8-flash")
    total = billing.combine(pricing, dg, gm)
    assert total.sarvam.charged is False and total.total_usd == dg.usd
    assert billing.REASON_NO_TRANSCRIPTION not in total.notes


def test_a_failing_recheck_never_fails_the_analysis(store, cfg, decoded):
    def boom(entries, indices):
        raise RuntimeError("provider down")
    deps = sarvam_deps(store, cfg, recheck=boom)
    _, report, _ = run(create_and_run(store, cfg, request_with_terms(), deps))
    assert report.status == "completed"
    doc = store.collection.docs[next(iter(store.collection.docs))]
    assert doc["processing"]["sarvam"]["recheck"] == {"reason": "recheck_error", "error": "RuntimeError"}
    assert any("free fund" in s["text"] for s in doc["transcript"]["segments"])    # Sarvam's text kept


def test_app_wires_the_recheck_to_the_module(monkeypatch):
    # 2026-10-01: the wrapper shared its name with the module it called, so
    # every flagged call crashed with "'function' object has no attribute".
    import asyncio
    import app as app_module
    seen = {}

    async def fake_recheck(client, model, samples, entries, indices, terms, brand_terms=None):
        seen.update(model=model, indices=indices, brand_terms=brand_terms)
        return rc.RecheckResult(ok=True)

    monkeypatch.setattr(app_module._sca_sarvam_rc, "recheck", fake_recheck)
    out = asyncio.run(app_module._sca_sarvam_recheck(np.zeros(10), [], [3], ["refund"], ["Brand"]))
    assert out.ok and seen["indices"] == [3] and seen["brand_terms"] == ["Brand"]
    deps = app_module._sales_call_deps()
    assert deps.sarvam_recheck is app_module._sca_sarvam_recheck


# =========================================================================== #
# Display labels: Speaker 1 / Speaker 2 for people, roles for scoring
# =========================================================================== #
def _raw(*turns):
    return {"metadata": {"duration": 30}, "results": {"utterances": [
        {"speaker": s, "start": a, "end": b, "transcript": t, "confidence": 1.0}
        for s, a, b, t in turns]}}


def test_speakers_are_numbered_in_the_order_they_first_speak():
    # Diarization called the customer speaker_1 but they spoke first.
    t = tr.from_deepgram(_raw((1, 0, 1, "Hello?"), (0, 1, 4, "Hi, this is Rajan from EdTech Pro"),
                              (1, 4, 5, "Yes")))
    assert [s.speaker_label for s in t.segments] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert {s.speaker_id: s.label for s in t.speakers} == {"speaker_1": "Speaker 1",
                                                           "speaker_0": "Speaker 2"}


def test_labels_do_not_change_roles():
    t = tr.from_deepgram(_raw((1, 0, 1, "Hello?"), (0, 1, 4, "Hi, this is Rajan from EdTech Pro")))
    t = sp.resolve_roles(t, rep=RepInfo(name="Rajan"), customer=CustomerInfo(), brand_name="EdTech Pro")
    by_id = {s.speaker_id: s for s in t.speakers}
    assert by_id["speaker_0"].role == "sales_rep" and by_id["speaker_0"].label == "Speaker 2"
    assert by_id["speaker_1"].role == "customer" and by_id["speaker_1"].label == "Speaker 1"
    # the scoring prompt still describes speakers by id and role
    assert "speaker_0: role=sales_rep" in sp.render_for_prompt(t)


def test_labelling_is_idempotent_and_survives_reuse():
    t = tr.from_deepgram(_raw((1, 0, 1, "Hello?"), (0, 1, 4, "Hi")))
    stored = tr.clear_role_annotations(tr.NormalizedTranscript(**t.model_dump(mode="json")))
    again = tr.label_speakers(tr.label_speakers(stored))
    assert [s.speaker_label for s in again.segments] == ["Speaker 1", "Speaker 2"]


def test_evidence_quotes_carry_the_speaker_label(store, cfg, decoded):
    deps = sarvam_deps(store, cfg)
    _, report, _ = run(create_and_run(store, cfg, request_with_terms(), deps))
    quotes = [e for st in report.stage_evaluations for c in st.criteria for e in c.evidence]
    assert quotes and all(e.speaker_label == "Speaker 1" for e in quotes if e.speaker_id == "speaker_0")
    doc = store.collection.docs[next(iter(store.collection.docs))]
    assert doc["transcript"]["segments"][0]["speaker_label"] == "Speaker 1"
