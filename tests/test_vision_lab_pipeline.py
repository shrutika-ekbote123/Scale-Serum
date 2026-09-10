"""The Vision Lab store, pipeline and worker loop, entirely offline.

No MongoDB, no ffmpeg, no ONNX model, no network. The vision layer arrives
through VisionDeps, so everything above it is testable on a bare CI runner -
the same arrangement that lets the sales-call suite run without Deepgram.

FakeCollection is shared with test_vision_lab_api.py.
"""
from __future__ import annotations

import asyncio
import copy
import os
import sys
from datetime import timedelta

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

import vision_lab as vl  # noqa: E402
from vision_lab import framework as fw  # noqa: E402
from vision_lab import pipeline as pl  # noqa: E402
from vision_lab import stubs  # noqa: E402
from vision_lab import store as st  # noqa: E402
from vision_lab.models import AnalyzeRequest  # noqa: E402


# =========================================================================== #
# Fakes
# =========================================================================== #
class FakeCursor:
    def __init__(self, docs):
        self.docs = docs

    def sort(self, key, direction=1):
        self.docs.sort(key=lambda d: (d.get(key) is None, d.get(key)),
                       reverse=direction < 0)
        return self

    async def to_list(self, length=None):
        return self.docs[:length] if length else self.docs


class FakeCollection:
    """Enough of motor's surface for the store, with real matching semantics."""

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.indexes: list = []

    def _matches(self, doc, query):
        for key, expected in query.items():
            value = doc.get(key)
            if isinstance(expected, dict):
                if "$ne" in expected and value == expected["$ne"]:
                    return False
                if "$nin" in expected and value in expected["$nin"]:
                    return False
                if "$in" in expected and value not in expected["$in"]:
                    return False
            elif value != expected:
                return False
        return True

    def _apply(self, target, update):
        target.update(update.get("$set") or {})
        for key, amount in (update.get("$inc") or {}).items():
            target[key] = (target.get(key) or 0) + amount
        for key in (update.get("$unset") or {}):
            target.pop(key, None)

    async def find_one(self, query):
        for doc in self.docs.values():
            if self._matches(doc, query):
                return copy.deepcopy(doc)
        return None

    def find(self, query):
        return FakeCursor([copy.deepcopy(d) for d in self.docs.values()
                           if self._matches(d, query)])

    async def insert_one(self, doc):
        self.docs[doc["_id"]] = copy.deepcopy(doc)

    async def update_one(self, query, update, upsert=False):
        for doc in self.docs.values():
            if self._matches(doc, query):
                self._apply(doc, update)
                return
        if upsert:
            target = {"_id": query.get("_id")}
            self.docs[target["_id"]] = target
            self._apply(target, update)

    async def find_one_and_update(self, query, update, sort=None,
                                  return_document=True):
        candidates = [d for d in self.docs.values() if self._matches(d, query)]
        if sort:
            key, direction = sort[0]
            candidates.sort(key=lambda d: (d.get(key) is None, d.get(key)),
                            reverse=direction < 0)
        if not candidates:
            return None
        target = candidates[0]
        before = copy.deepcopy(target)
        self._apply(target, update)
        return copy.deepcopy(target) if return_document else before

    async def delete_one(self, query):
        for key, doc in list(self.docs.items()):
            if self._matches(doc, query):
                del self.docs[key]
                return

    async def create_index(self, *args, **kwargs):
        self.indexes.append((args, kwargs))


def make_store():
    return st.AnalysisStore(FakeCollection(), measurements_collection=FakeCollection())


def make_request(**overrides):
    payload = {
        "creative_id": "cre_test_1",
        "ad_number": "042",
        "division": "directors_institute",
        "creative": {
            "url": "https://bucket.s3.amazonaws.com/test/ad.mp4?X-Amz-Signature=abc",
            "kind": "video",
            "mime_type": "video/mp4",
        },
    }
    payload.update(overrides)
    return AnalyzeRequest(**payload)


def make_deps(store, **overrides):
    kwargs = dict(stubs.deps_kwargs())
    kwargs.update(overrides)
    return pl.VisionDeps(store=store, **kwargs)


async def seed(store, request=None, **doc_overrides):
    request = request or make_request()
    versions = {**fw.versions(), "prompt_version": pl.PROMPT_VERSION,
                "saliency_model": "stub"}
    fingerprint = st.compute_fingerprint(
        request, framework_version=versions["framework_version"],
        psychology_version=versions["psychology_version"],
        prompt_version=versions["prompt_version"],
        saliency_model="stub", sample_fps=2.0)
    doc = await store.create(analysis_id=st.new_analysis_id(), request=request,
                             fingerprint=fingerprint, versions=versions)
    if doc_overrides:
        await store.collection.update_one({"_id": doc["_id"]}, {"$set": doc_overrides})
        doc = await store.get(doc["_id"])
    return doc


# =========================================================================== #
# Config
# =========================================================================== #
def test_framework_loads_and_declares_its_placeholders():
    cfg = fw.load_framework()
    assert cfg["framework_version"] == "vision_v1"
    assert len(cfg["metrics"]) == 6

    disclosure = fw.config_disclosure()
    # Nothing here is confirmed yet, and the API must say so on every response.
    assert disclosure["weighting"] == fw.WEIGHTING_EQUAL
    assert disclosure["bands"] == "thresholds_not_configured"
    assert disclosure["weak_zone_rule"] == "placeholder"
    assert disclosure["unconfirmed"]


def test_psychology_has_exactly_fifteen_triggers_and_blind_spot_is_unscored():
    cfg = fw.load_psychology()
    triggers = cfg["triggers"]
    assert len(triggers) == 15

    index = fw.trigger_index(cfg)
    # Eight are measured rather than judged - that is what stops the psychology
    # layer being a horoscope.
    measured = [t for t in triggers if t["detection"] == "measured"]
    assert len(measured) >= 6

    # Blind-Spot Bias is a property of the marketer, not of the frames.
    assert index["blind_spot_bias"]["scored"] is False

    # Anchoring is not applicable without a price, and that is not a zero.
    assert index["anchoring"]["not_applicable_reason"] == "no_price_shown"


def test_bands_are_absent_until_configured():
    band, reason = fw.bands()
    assert band is None
    assert reason == "thresholds_not_configured"


# =========================================================================== #
# Store
# =========================================================================== #
def test_fingerprint_ignores_the_signature_but_not_the_version():
    """A presigned URL changes on every presign of the same object. If the
    signature were part of the identity, idempotency would never hit."""
    base = dict(framework_version="vision_v1", psychology_version="triggers_v1",
                prompt_version="p1", saliency_model="m1", sample_fps=2.0)

    signed_once = make_request(creative={
        "url": "https://bucket.s3.amazonaws.com/test/ad.mp4?X-Amz-Signature=AAA"})
    signed_again = make_request(creative={
        "url": "https://bucket.s3.amazonaws.com/test/ad.mp4?X-Amz-Signature=ZZZ"})
    assert (st.compute_fingerprint(signed_once, **base)
            == st.compute_fingerprint(signed_again, **base))

    # A new framework version is genuinely a different analysis.
    changed = dict(base, framework_version="vision_v2")
    assert (st.compute_fingerprint(signed_once, **base)
            != st.compute_fingerprint(signed_once, **changed))


def test_reusable_rules():
    assert st.reusable({"status": vl.STATUS_COMPLETED}, force=False) is True
    assert st.reusable({"status": vl.STATUS_COMPLETED}, force=True) is False
    # A failure is not reusable - asking again is exactly how a retry happens.
    assert st.reusable({"status": vl.STATUS_FAILED}, force=False) is False
    assert st.reusable(None, force=False) is False


def test_stale_detection_only_applies_to_active_jobs():
    fresh = {"status": vl.STATUS_ANALYZING_FRAMES, "heartbeat_at": st.now_utc()}
    dead = {"status": vl.STATUS_ANALYZING_FRAMES,
            "heartbeat_at": st.now_utc() - timedelta(seconds=st.STALE_AFTER_SECONDS + 60)}
    done = {"status": vl.STATUS_COMPLETED,
            "heartbeat_at": st.now_utc() - timedelta(days=30)}

    assert st.is_stale(fresh) is False
    assert st.is_stale(dead) is True
    assert st.is_stale(done) is False


def test_claim_next_gives_a_job_to_exactly_one_worker():
    async def scenario():
        store = make_store()
        await seed(store)

        first = await store.claim_next("worker-a")
        second = await store.claim_next("worker-b")

        assert first is not None
        assert first["claimed_by"] == "worker-a"
        assert first["status"] == vl.STATUS_PROBING
        assert first["attempts"] == 1
        # The job is no longer queued, so the second worker finds nothing.
        assert second is None

    asyncio.run(scenario())


def test_creative_url_is_cleared_and_never_survives_the_job():
    async def scenario():
        store = make_store()
        doc = await seed(store)
        assert doc["creative_url"]

        await store.clear_creative_url(doc["_id"])
        assert (await store.get(doc["_id"])).get("creative_url") is None

    asyncio.run(scenario())


def test_history_filters_by_division_and_ad_number():
    async def scenario():
        store = make_store()
        await seed(store, make_request(ad_number="042"))
        await seed(store, make_request(creative_id="cre_2", ad_number="043"))

        all_rows = await store.history()
        just_042 = await store.history(ad_number="042")

        assert len(all_rows) == 2
        assert len(just_042) == 1
        assert just_042[0]["ad_number"] == "042"

    asyncio.run(scenario())


# =========================================================================== #
# Pipeline
# =========================================================================== #
def test_run_analysis_walks_the_statuses_and_completes():
    async def scenario():
        store = make_store()
        doc = await seed(store)
        claimed = await store.claim_next("worker-a")

        seen = []
        original = store.set_status

        async def spy(analysis_id, status, **fields):
            seen.append(status)
            await original(analysis_id, status, **fields)

        store.set_status = spy
        final = await pl.run_analysis(claimed, make_deps(store))

        assert seen == [vl.STATUS_PROBING, vl.STATUS_ANALYZING_FRAMES, vl.STATUS_SCORING]
        assert final["status"] == vl.STATUS_COMPLETED
        # The credential is gone by the time the job ends.
        assert final.get("creative_url") is None

    asyncio.run(scenario())


def test_the_report_is_complete_and_states_every_absence():
    async def scenario():
        store = make_store()
        await seed(store)
        claimed = await store.claim_next("worker-a")
        final = await pl.run_analysis(claimed, make_deps(store))
        report = final["report"]

        # Nothing in the payload is a placeholder any more. The stub deps supply
        # no transcription and no LLM, which is a SUPPORTED state - each block
        # says so rather than being left out.
        assert report["stub"] is False
        assert report["transcript"]["available"] is False
        assert report["transcript"]["reason"]
        assert report["interpretation"]["available"] is False
        assert report["interpretation"]["reason"] == "analysis_not_configured"
        assert report["recommendations"] == []
        # The 15 triggers are still reported, each with a status and a reason -
        # an unjudgeable trigger is not a silently missing one.
        assert len(report["psychology"]["triggers"]) == 15

        # Scores come from MEASUREMENTS. The stubs measure nothing, so
        # every metric must report null WITH A REASON rather than inventing a
        # number from absent data - that distinction is the whole contract.
        for metric_id, entry in report["scores"].items():
            assert entry["score"] is None, metric_id
            assert entry["reason"], f"{metric_id} is null with no stated reason"
        assert report["overall"]["score"] is None
        assert report["overall"]["metrics_scored"] == 0
        assert len(report["overall"]["metrics_missing"]) == 6
        # Why there is no score, and why there is no band, are separate answers.
        assert report["overall"]["score_reason"] == "no metric could be scored"

        # ...but structurally complete, so the frontend can build against it.
        assert set(report["scores"]) == set(fw.METRIC_IDS)
        for key in ("media", "heatmap", "key_moments", "timeline", "psychology",
                    "recommendations", "versions", "config_disclosure"):
            assert key in report, key

        # Cognitive Demand is the one metric where lower is better, and the UI
        # needs to know that rather than infer it.
        assert report["scores"]["cognitive_demand"]["direction"] == "lower_better"
        # No bands are configured, so none are invented.
        assert report["overall"]["band"] is None
        assert report["overall"]["band_reason"] == "thresholds_not_configured"

    asyncio.run(scenario())


def test_measurements_are_stored_separately_for_rescore():
    async def scenario():
        store = make_store()
        await seed(store)
        claimed = await store.claim_next("worker-a")
        await pl.run_analysis(claimed, make_deps(store))

        stored = await store.get_measurements(claimed["_id"])
        assert stored is not None
        # 24 s at 2 fps, per the stub.
        assert len(stored["measurements"]["frames"]) == 48
        assert "summary" in stored["measurements"]
        assert stored["measurements"]["sample_fps"] == 2.0

    asyncio.run(scenario())


# =========================================================================== #
# Milestone D - the interpretation layer, wired
# =========================================================================== #
def audible_sample_frames(url, *, kind, sample_fps=2.0, max_frames=120,
                          max_duration_seconds=None):
    """The stub media layer, plus an audio track to transcribe."""
    media = stubs.sample_frames(url, kind=kind, sample_fps=sample_fps,
                                max_frames=max_frames,
                                max_duration_seconds=max_duration_seconds)
    return {**media, "audio": b"RIFF....WAVEfmt ", "audio_bytes": 16}


DEEPGRAM_REPLY = {"results": {"channels": [{"alternatives": [{"transcript": "x"}]}],
                              "utterances": [
                                  {"start": 0.0, "end": 3.0, "confidence": 0.9,
                                   "transcript": "Most operations directors think the bottleneck is headcount."},
                                  {"start": 12.0, "end": 15.0, "confidence": 0.9,
                                   "transcript": "Applications close on the thirtieth."}]}}


class SilentProvider:
    """A transcription provider that is simply down. Not an error case for the
    analysis - a supported state that the report has to state."""

    def __call__(self, audio):
        raise RuntimeError("provider unreachable")


def test_a_transcript_reaches_the_report_joined_to_the_timeline():
    async def scenario():
        store = make_store()
        await seed(store)
        claimed = await store.claim_next("worker-a")

        async def transcribe(audio):
            assert audio == b"RIFF....WAVEfmt "   # the bytes, not a second fetch
            return DEEPGRAM_REPLY

        final = await pl.run_analysis(claimed, make_deps(
            store, sample_frames=audible_sample_frames, transcribe=transcribe))

        block = final["report"]["transcript"]
        assert block["available"] is True
        assert len(block["segments"]) == 2
        assert block["stats"]["words"] == 13
        # Every line carries the two things the transcript panel renders.
        for segment in block["segments"]:
            assert "attention" in segment
            assert "in_weak_zone" in segment
        # The provider payload never reaches the response shape.
        assert "confidence" not in block["segments"][0]

    asyncio.run(scenario())


def test_a_transcription_outage_degrades_the_report_it_does_not_fail_it():
    async def scenario():
        store = make_store()
        await seed(store)
        claimed = await store.claim_next("worker-a")

        final = await pl.run_analysis(claimed, make_deps(
            store, sample_frames=audible_sample_frames,
            transcribe=SilentProvider()))

        assert final["status"] == vl.STATUS_COMPLETED
        report = final["report"]
        assert report["transcript"]["available"] is False
        assert report["transcript"]["reason"] == vl.TRANSCRIPTION_PROVIDER_ERROR
        assert vl.TRANSCRIPTION_PROVIDER_ERROR in report["notes"]
        # The scorecard is untouched: not one of the six metrics depends on speech.
        assert set(report["scores"]) == set(fw.METRIC_IDS)

    asyncio.run(scenario())


class FakeGemini:
    """Returns a fixed interpretation, including one invented defect and one
    fabricated timestamp - because that is what has to be caught."""

    PAYLOAD = {
        "summary": "The hook lands; the middle asks the viewer to wait.",
        "clarity": {"rating": "adequate", "why": "the CTA is legible"},
        "key_message": {"element": "headline", "t": 2.0,
                        "quote": "Most operations directors think the bottleneck is headcount."},
        "recommendations": [
            {"defect_id": "no_cta", "title": "Ask for something",
             "why": "Nothing on screen tells the viewer what to do.",
             "fix": "Put a single instruction on the close."},
            {"defect_id": "the_soundtrack_is_tired", "title": "Change the music",
             "why": "It feels dated.", "fix": "Licence something current."}],
        "triggers": [{"id": "framing_effect", "rating": "strong",
                      "note": "the cost of standing still is named first",
                      "evidence": [{"t": 900.0, "quote": "Never said in this ad."}]}],
        "observations": ["The close assumes the viewer already knows the brand."],
    }

    def __init__(self):
        import json as _json
        payload = self.PAYLOAD

        class Models:
            calls = 0

            async def generate_content(self, **kwargs):
                Models.calls += 1
                self.kwargs = kwargs
                return type("R", (), {"text": _json.dumps(payload)})()

        self.models = Models()
        self.aio = type("Aio", (), {"models": self.models})()


def test_the_interpretation_reaches_the_report_and_the_invented_parts_do_not():
    async def scenario():
        store = make_store()
        await seed(store)
        claimed = await store.claim_next("worker-a")
        client = FakeGemini()

        async def transcribe(audio):
            return DEEPGRAM_REPLY

        final = await pl.run_analysis(claimed, make_deps(
            store, sample_frames=audible_sample_frames, transcribe=transcribe,
            llm_client=client, llm_model="fake-model"))

        report = final["report"]
        assert report["interpretation"]["available"] is True
        assert report["summary"].startswith("The hook lands")

        # The stub vision layer measures nothing, so no defect was found - and a
        # model writing about a defect nobody found gets nothing published.
        # This is the guarantee, restated end to end: no recommendation without
        # a measured finding behind it.
        assert report["recommendations"] == []
        assert "the_soundtrack_is_tired" in \
            report["interpretation"]["dropped_invented_recommendations"]

        # The fabricated quote at 900 s was checked and dropped.
        audit = report["interpretation"]["evidence_audit"]
        assert audit["dropped"] >= 1
        framing = next(t for t in report["psychology"]["triggers"]
                       if t["id"] == "framing_effect")
        assert framing["status"] == vl.TRIGGER_UNSUPPORTED

        # The ratings are stored apart from the report so /rescore can reuse
        # them without a second call to the model.
        stored = await store.get(claimed["_id"])
        assert stored["analysis"]["llm_ratings"] == {"clarity": "adequate"}
        assert stored["analysis"]["key_message"]["t"] == 2.0
        assert client.models.calls == 1

    asyncio.run(scenario())


def test_without_a_model_the_report_still_carries_measurements_and_scores():
    async def scenario():
        store = make_store()
        await seed(store)
        claimed = await store.claim_next("worker-a")

        final = await pl.run_analysis(claimed, make_deps(store))

        report = final["report"]
        assert final["status"] == vl.STATUS_COMPLETED
        assert report["interpretation"]["available"] is False
        assert report["interpretation"]["reason"] == vl.ANALYSIS_NOT_CONFIGURED
        assert report["summary"] == ""
        # And the 15 triggers are still reported, each saying where it stands.
        assert len(report["psychology"]["triggers"]) == 15
        assert all(t["status"] for t in report["psychology"]["triggers"])

    asyncio.run(scenario())


def test_a_missing_creative_fails_with_a_stated_reason_not_an_exception():
    async def scenario():
        store = make_store()
        doc = await seed(store)
        await store.clear_creative_url(doc["_id"])
        claimed = await store.claim_next("worker-a")

        final = await pl.run_analysis(claimed, make_deps(store))

        assert final["status"] == vl.STATUS_FAILED
        assert final["reason"] == vl.NO_CREATIVE
        assert final["message"]
        # A failed analysis never invents a scorecard.
        assert final.get("report") is None

    asyncio.run(scenario())


def test_an_unreachable_creative_is_reported_not_raised():
    async def scenario():
        store = make_store()
        await seed(store)
        claimed = await store.claim_next("worker-a")

        def explode(*_args, **_kwargs):
            raise OSError("connection reset")

        final = await pl.run_analysis(claimed, make_deps(store, sample_frames=explode))

        assert final["status"] == vl.STATUS_FAILED
        assert final["reason"] == vl.CREATIVE_UNREACHABLE

    asyncio.run(scenario())


def test_an_unsupported_format_is_refused_before_anything_is_downloaded():
    async def scenario():
        store = make_store()
        await seed(store, make_request(creative={
            "url": "https://bucket.s3.amazonaws.com/test/deck.pdf"}))
        claimed = await store.claim_next("worker-a")

        touched = []
        final = await pl.run_analysis(
            claimed, make_deps(store, sample_frames=lambda *a, **k: touched.append(1)))

        assert final["status"] == vl.STATUS_FAILED
        assert final["reason"] == vl.UNSUPPORTED_FORMAT
        assert touched == []   # nothing was fetched

    asyncio.run(scenario())


def test_a_static_image_produces_one_frame_and_no_timeline():
    async def scenario():
        store = make_store()
        await seed(store, make_request(creative={
            "url": "https://bucket.s3.amazonaws.com/test/banner.png",
            "kind": "image", "mime_type": "image/png"}))
        claimed = await store.claim_next("worker-a")
        final = await pl.run_analysis(claimed, make_deps(store))
        report = final["report"]

        assert report["media"]["frames_analyzed"] == 1
        # One contract, not two - the field exists and is null.
        assert report["timeline"] is None

    asyncio.run(scenario())


def test_classify_creative_prefers_the_declared_kind_then_mime_then_extension():
    assert pl.classify_creative("https://x/y", None, "image") == vl.KIND_IMAGE
    assert pl.classify_creative("https://x/y", "video/mp4", None) == vl.KIND_VIDEO
    assert pl.classify_creative("https://x/y", "video/mp4; codecs=avc1", None) == vl.KIND_VIDEO
    assert pl.classify_creative("https://x/y.mov?sig=1", None, None) == vl.KIND_VIDEO
    with pytest.raises(pl.AnalysisFailure) as err:
        pl.classify_creative("https://x/y.pdf", None, None)
    assert err.value.reason == vl.UNSUPPORTED_FORMAT


def test_a_declared_non_media_mime_is_refused_even_when_the_key_looks_fine():
    """An S3 object with ContentType application/pdf and a .mp4 key is a real
    mismatch. Guessing from the extension would hand a PDF to ffmpeg."""
    with pytest.raises(pl.AnalysisFailure) as err:
        pl.classify_creative("https://x/ad.mp4?X-Amz-Signature=a",
                             "application/pdf", None)
    assert err.value.reason == vl.UNSUPPORTED_FORMAT

    # octet-stream is the exception: it means "unlabelled", not "not media", and
    # is what several uploaders send for a perfectly good MP4.
    assert pl.classify_creative("https://x/ad.mp4", "application/octet-stream",
                                None) == vl.KIND_VIDEO


def test_an_interrupted_job_becomes_a_retryable_failure():
    """A pm2 restart mid-analysis is every deploy. The row must not sit in an
    active status forever."""
    async def scenario():
        store = make_store()
        doc = await seed(store)
        await store.claim_next("worker-a")
        await store.collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"heartbeat_at": st.now_utc()
                      - timedelta(seconds=st.STALE_AFTER_SECONDS + 60)}})

        stale = await store.get(doc["_id"])
        assert st.is_stale(stale)

        await store.mark_interrupted(doc["_id"])
        after = await store.get(doc["_id"])
        assert after["status"] == vl.STATUS_FAILED
        assert after["reason"] == vl.PROCESSING_INTERRUPTED

    asyncio.run(scenario())


def test_mark_interrupted_never_touches_a_finished_analysis():
    async def scenario():
        store = make_store()
        doc = await seed(store)
        await store.complete(doc["_id"], report={"ok": True})
        await store.mark_interrupted(doc["_id"])

        after = await store.get(doc["_id"])
        assert after["status"] == vl.STATUS_COMPLETED
        assert after["reason"] is None

    asyncio.run(scenario())
