"""
Orchestration: status transitions, heartbeats, and the one place a failure
becomes a stated reason rather than a stack trace.

THE INJECTION SEAM
    Everything expensive or external arrives in VisionDeps: frame sampling,
    saliency inference, region detection, measurement, heatmap rendering,
    object storage, transcription, the LLM client. Nothing in this file imports
    ffmpeg, ONNX, OpenCV or an HTTP client.

    That is not tidiness. It is what lets the whole API, measurement, scoring and
    trigger surface be tested on a bare CI runner with no ffmpeg binary and no
    model weights - exactly as the Sales Call Analyzer tests run with no
    Deepgram and no MongoDB.

THE ORDER OF THE PASSES IS A CORRECTNESS REQUIREMENT
    Measure, then time, then find defects, then transcribe, then count the
    triggers, THEN interpret, and only then score. The model is shown the
    defects so it can write about them and cannot nominate its own; scoring runs
    last because two of the six metrics take an ordinal rating from that pass -
    and every one of them still produces a number if the pass never happens.

FAILURE IS A STATUS, NOT AN EXCEPTION
    Every expected failure is persisted with a reason from __init__.py and the
    job ends `failed`. The only thing that escapes to the worker is genuinely
    unexpected, and the worker records that too - a task that dies silently
    would leave the row in an active status forever.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from . import (
    ANALYSIS_NOT_CONFIGURED,
    PROMPT_VERSION,  # noqa: F401 - re-exported; the store and tests read pl.PROMPT_VERSION
    CREATIVE_UNREACHABLE,
    KIND_IMAGE,
    KIND_VIDEO,
    NO_AUDIO_STREAM,
    NO_BRAND_ASSETS,
    NO_BRAND_BRAIN,
    NO_CREATIVE,
    OCR_UNAVAILABLE,
    REASON_TEXT,
    STATUS_ANALYZING_FRAMES,
    STATUS_COMPLETED,
    STATUS_INTERPRETING,
    STATUS_PROBING,
    STATUS_SCORING,
    STATUS_TRANSCRIBING,
    TRANSCRIPTION_NOT_CONFIGURED,
    TRANSCRIPTION_PROVIDER_ERROR,
    TRANSCRIPT_EMPTY,
    UNSUPPORTED_FORMAT,
)
from . import analyzer as an
from . import defects as df
from . import evidence as ev
from . import framework as fw
from . import psychology as psy
from . import regions as rg
from . import report as rp
from . import scoring as sc
from . import timeline as tl
from . import transcript as tx
from .store import AnalysisStore

logger = logging.getLogger("vision_lab.pipeline")

# Everything we can actually decode. Anything else is refused with a stated
# reason rather than attempted and failed halfway through.
VIDEO_TYPES = ("video/mp4", "video/quicktime", "video/webm")
IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# How many frames get a rendered overlay. Not all 48 - that is 48 uploads and
# ~15 MB per analysis for images nobody opens. The hero frame plus a sampled
# strip is what the report actually shows.
STRIP_FRAMES = 6


class AnalysisFailure(Exception):
    """An expected failure with a reason code the UI can branch on."""

    def __init__(self, reason: str, message: Optional[str] = None):
        self.reason = reason
        self.message = message or REASON_TEXT.get(reason, "The analysis did not complete.")
        super().__init__(self.message)


@dataclass
class VisionDeps:
    """Everything the pipeline needs from the process that hosts it.

    The package owns no clients, no connections and no binaries of its own.
    """
    store: AnalysisStore
    sample_frames: Callable          # media.py    - Step 7
    predict_saliency: Callable       # saliency.py - Step 9
    detect_regions: Callable         # regions.py  - Step 10
    measure_frames: Callable         # measure.py  - Step 11
    summarise: Callable              # measure.py  - Step 11
    saliency_info: Callable          # saliency.py - what produced these numbers
    render_heatmap: Optional[Callable] = None      # heatmap.py - Step 12
    make_thumbnail: Optional[Callable] = None
    upload: Optional[Callable] = None              # S3
    fetch_bytes: Optional[Callable] = None         # for the brand wordmark
    decode_image: Optional[Callable] = None
    transcribe: Optional[Callable] = None          # Deepgram - Step 16
    llm_client: Any = None                         # Gemini   - Step 18
    llm_model: str = ""
    load_brand_brain: Optional[Callable] = None


def unavailable(reason: str, message: Optional[str] = None) -> dict:
    return {"available": False, "reason": reason,
            "message": message or REASON_TEXT.get(reason)}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- input
def classify_creative(url: str, mime_type: Optional[str],
                      declared_kind: Optional[str]) -> str:
    """video or image, from whatever the caller told us.

    The caller's `kind` wins when it is one we support, then the mime type, then
    the file extension. ffprobe overrides all of it in media.py - this is only
    enough to reject something we could never process before downloading it.

    A DECLARED mime type is believed, including when it rules the file out. If
    the caller says application/pdf we refuse, even where the key happens to end
    in .mp4: an object whose declared content type contradicts its key is a
    mismatch worth surfacing, not one to guess around. Falling through to the
    extension there would have us download a PDF and hand it to ffmpeg.
    """
    if declared_kind in (KIND_VIDEO, KIND_IMAGE):
        return declared_kind
    if mime_type:
        lowered = mime_type.lower().split(";", 1)[0].strip()
        if lowered in VIDEO_TYPES or lowered.startswith("video/"):
            return KIND_VIDEO
        if lowered in IMAGE_TYPES or lowered.startswith("image/"):
            return KIND_IMAGE
        # Not a media type at all. The caller told us; believe them.
        if "/" in lowered and lowered != "application/octet-stream":
            raise AnalysisFailure(UNSUPPORTED_FORMAT)
    path = (url or "").split("?", 1)[0].lower()
    if path.endswith(VIDEO_EXTENSIONS):
        return KIND_VIDEO
    if path.endswith(IMAGE_EXTENSIONS):
        return KIND_IMAGE
    raise AnalysisFailure(UNSUPPORTED_FORMAT)


def has_creative(doc: dict) -> bool:
    return bool(doc.get("creative_url"))


# --------------------------------------------------------------------------- run
async def run_analysis(doc: dict, deps: VisionDeps) -> dict:
    """Process one claimed job. Returns the final document state.

    Works from the job DOCUMENT rather than the request model, because the
    worker is a separate process and the document is all it can see.
    """
    analysis_id = doc["_id"]
    store = deps.store
    snapshot = doc.get("request_snapshot") or {}
    started = now_utc()
    notes: list[str] = []

    try:
        # ---------------------------------------------------------- probing
        if not has_creative(doc):
            raise AnalysisFailure(NO_CREATIVE)

        url = doc["creative_url"]
        kind = classify_creative(url, snapshot.get("mime_type"), snapshot.get("kind"))
        await store.set_status(analysis_id, STATUS_PROBING)

        cfg = fw.load_framework()
        sampling = fw.sampling(cfg)
        requested_fps = (snapshot.get("options") or {}).get("sample_fps")
        sample_fps = float(requested_fps or sampling.get("sample_fps") or 2.0)

        try:
            media = await _maybe_await(
                deps.sample_frames, url, kind=kind, sample_fps=sample_fps,
                max_frames=int(sampling.get("max_frames") or 120),
                max_duration_seconds=sampling.get("max_duration_seconds"))
        except AnalysisFailure:
            raise
        except Exception as err:  # noqa: BLE001
            # media.py raises MediaError with its own reason code; anything else
            # is genuinely a fetch problem.
            reason = getattr(err, "reason", None) or CREATIVE_UNREACHABLE
            logger.warning("frame sampling failed [analysis_id=%s reason=%s]: %s",
                           analysis_id, reason, err)
            raise AnalysisFailure(reason) from err

        # media.py may have thinned the rate to cover a long creative within
        # the frame cap; its value wins over what we asked for.
        sample_fps = float(media.get("sample_fps") or sample_fps)
        await store.save_media(analysis_id, rp.media_summary(media, kind, sample_fps))

        # The wordmark is fetched while we still hold its URL; both credentials
        # then go at once.
        wordmark = await _load_wordmark(doc.get("wordmark_url"), deps)
        await store.clear_creative_url(analysis_id)

        # -------------------------------------------------- analyzing frames
        await store.set_status(analysis_id, STATUS_ANALYZING_FRAMES)

        measurements = await _maybe_await(
            deps.measure_frames, media,
            wordmark=wordmark,
            brand_names=(snapshot.get("brand_names") or []),
            detect_regions=deps.detect_regions,
            predict_saliency=deps.predict_saliency)
        await store.heartbeat(analysis_id)

        summary = await _maybe_await(deps.summarise, measurements, media)
        vision = deps.saliency_info()

        await store.save_measurements(analysis_id, {
            "frames": measurements,
            "summary": summary,
            "sample_fps": sample_fps,
            "shots": media.get("shots") or [],
            "vision": vision,
        })

        heatmaps = await _render_heatmaps(
            analysis_id, media, measurements, deps, wordmark=wordmark,
            brand_names=snapshot.get("brand_names") or [])
        await store.heartbeat(analysis_id)

        if not snapshot.get("has_brand_assets"):
            notes.append(NO_BRAND_ASSETS)
        if not doc.get("brand_brain_id"):
            notes.append(NO_BRAND_BRAIN)
        if not summary.get("frames_with_text_fraction"):
            notes.append(OCR_UNAVAILABLE)

        # ------------------------------------------------ timeline and defects
        # Every number below is arithmetic over the measurement record and
        # vision_framework.json - no model, no LLM, no network.
        summary["sample_fps"] = sample_fps
        timeline = tl.build(measurements, media, cfg) if kind == KIND_VIDEO else None
        moments = (tl.key_moments(timeline["points"], measurements,
                                  timeline["weak_zones"]) if timeline else [])

        # The measured half of the fix recommendations. These are handed to
        # Gemini to write prose ABOUT - the model never nominates a defect of
        # its own, which is what keeps every recommendation anchored to a
        # timestamp and a counted number.
        found = df.detect(measurements, summary, timeline, cfg)

        # ------------------------------------------------------ transcript (16)
        transcript = await _transcribe(analysis_id, media, timeline, deps, notes)

        # ------------------------------------------------------ psychology (17)
        brand_brain = await _load_brand_brain(doc, deps)
        triggers = psy.detect(measurements, summary, timeline, transcript,
                              fw.load_psychology(), brand_brain)

        # -------------------------------------------- interpretation (18 + 19)
        interpretation = await _interpret(
            analysis_id, deps, media=media, summary=summary,
            timeline=timeline or {"points": []}, defects=found,
            transcript=transcript, triggers=triggers, brand_brain=brand_brain)

        # Ratings and the named key message are the ONLY things the model
        # contributes to a number, and both arrive as words this file hands to
        # scoring.py for conversion. If the model was unavailable, five metrics
        # still score and two say why they could not - see analyzer.py.
        ratings = _ratings_from(interpretation)
        scored = sc.score_all(measurements, summary,
                              timeline or {"points": []}, cfg,
                              llm_ratings=ratings,
                              key_message=interpretation.get("key_message"))

        # The judged triggers get their verified ratings folded back in here,
        # AFTER evidence.py has checked what the model cited.
        triggers = psy.apply_interpretation(triggers, interpretation)

        # ---------------------------------------------------------- scoring
        await store.set_status(analysis_id, STATUS_SCORING)

        report = rp.build_report(doc, media=media, summary=summary, heatmaps=heatmaps,
                              kind=kind, sample_fps=sample_fps, notes=notes,
                              vision=vision, timeline=timeline,
                              key_moments=moments, scored=scored,
                              defects=found, transcript=transcript,
                              triggers=triggers, interpretation=interpretation)

        processing = {
            "started_at": started,
            "finished_at": now_utc(),
            "seconds": round((now_utc() - started).total_seconds(), 2),
            "frames_analyzed": len(measurements),
            "shots": len(media.get("shots") or []),
            "defects_found": len(found),
            "saliency_method": vision.get("saliency_method"),
            "transcribed": bool((transcript or {}).get("segments")),
            "interpreted": bool(interpretation.get("available", True)),
        }
        # `analysis` holds what the model returned, kept apart from the report so
        # /rescore can reuse the ratings without a second Gemini call.
        await store.complete(analysis_id, report=report, processing=processing,
                             analysis={
                                 "interpretation": interpretation,
                                 "llm_ratings": ratings,
                                 "key_message": interpretation.get("key_message"),
                                 "prompt_version": an.PROMPT_VERSION,
                             })
        logger.info("vision lab analysis completed [analysis_id=%s frames=%d "
                    "shots=%d method=%s %.1fs]", analysis_id, len(measurements),
                    len(media.get("shots") or []), vision.get("saliency_method"),
                    processing["seconds"])

    except AnalysisFailure as failure:
        await store.fail(analysis_id, reason=failure.reason, message=failure.message)
        await store.clear_creative_url(analysis_id)
        logger.info("vision lab analysis failed [analysis_id=%s reason=%s]",
                    analysis_id, failure.reason)

    return await store.get(analysis_id) or doc


async def _maybe_await(func: Callable, *args, **kwargs):
    """Deps may be sync (ffmpeg, ONNX, OpenCV) or async (HTTP). Accept both
    rather than forcing every implementation into one shape."""
    result = func(*args, **kwargs)
    if hasattr(result, "__await__"):
        return await result
    return result


async def _transcribe(analysis_id: str, media: dict, timeline: Optional[dict],
                      deps: VisionDeps, notes: list[str]) -> dict:
    """The spoken track, joined to the attention timeline.

    NEVER FATAL. A silent cutdown, an unconfigured provider and a provider
    outage all produce a transcript block that states its reason - they do not
    end the analysis, because not one of the six scores depends on speech.
    """
    if not media.get("has_audio"):
        notes.append(NO_AUDIO_STREAM)
        return tx.unavailable(NO_AUDIO_STREAM)

    audio = media.get("audio")
    if not audio:
        notes.append(NO_AUDIO_STREAM)
        return tx.unavailable(NO_AUDIO_STREAM,
                              "The audio track could not be extracted.")
    if not deps.transcribe:
        notes.append(TRANSCRIPTION_NOT_CONFIGURED)
        return tx.unavailable(TRANSCRIPTION_NOT_CONFIGURED)

    await deps.store.set_status(analysis_id, STATUS_TRANSCRIBING)
    try:
        raw = await _maybe_await(deps.transcribe, audio)
    except Exception as err:  # noqa: BLE001
        reason = getattr(err, "reason", None) or TRANSCRIPTION_PROVIDER_ERROR
        logger.warning("transcription failed [analysis_id=%s reason=%s]: %s",
                       analysis_id, reason, err)
        notes.append(reason)
        return tx.unavailable(reason, str(err)[:200])
    finally:
        # The WAV has served its purpose; a few MB per job adds up in a worker
        # that never restarts between analyses.
        media.pop("audio", None)

    transcript = tx.normalise(raw)
    if not transcript.get("segments"):
        notes.append(TRANSCRIPT_EMPTY)
        return tx.unavailable(TRANSCRIPT_EMPTY)

    tx.join_to_timeline(transcript, timeline)
    transcript["available"] = True
    transcript["reason"] = None
    transcript["stats"] = tx.summarise(transcript)
    return transcript


async def _load_brand_brain(doc: dict, deps: VisionDeps) -> Optional[dict]:
    """The brand's persona, if one was supplied. Reported when absent - two
    triggers are judged against it and are `not_applicable` without it."""
    brand_brain_id = doc.get("brand_brain_id")
    if not brand_brain_id or not deps.load_brand_brain:
        return None
    try:
        return await _maybe_await(deps.load_brand_brain, brand_brain_id)
    except Exception as err:  # noqa: BLE001
        logger.warning("could not load the brand brain %s: %s", brand_brain_id, err)
        return None


async def _interpret(analysis_id: str, deps: VisionDeps, *, media: dict,
                     summary: dict, timeline: dict, defects: list[dict],
                     transcript: Optional[dict], triggers: dict,
                     brand_brain: Optional[dict]) -> dict:
    """Step 18 then Step 19: the model reads, then every claim is checked.

    Verification is NOT optional and NOT a later pass - nothing the model wrote
    leaves this function without having been checked against the measurement
    record, so no caller can accidentally publish the unverified version.
    """
    if not deps.llm_client:
        return an.unavailable(ANALYSIS_NOT_CONFIGURED)

    await deps.store.set_status(analysis_id, STATUS_INTERPRETING)
    context = an.build_context(media=media, summary=summary, timeline=timeline,
                               defects=defects, transcript=transcript,
                               triggers=triggers, brand_brain=brand_brain)
    try:
        parsed = await an.interpret(context, client=deps.llm_client,
                                    model=deps.llm_model)
    except an.AnalyzerError as err:
        logger.warning("interpretation failed [analysis_id=%s reason=%s]: %s",
                       analysis_id, err.reason, err)
        return an.unavailable(err.reason, str(err)[:200])
    except Exception as err:  # noqa: BLE001
        logger.warning("interpretation failed unexpectedly [analysis_id=%s]: %s",
                       analysis_id, err)
        return an.unavailable(an.ANALYSIS_PROVIDER_ERROR, str(err)[:200])

    verified = ev.verify(parsed, defects=defects, summary=summary,
                         timeline=timeline, transcript=transcript,
                         duration=media.get("duration_seconds"),
                         context=context)
    verified["available"] = True
    verified["reason"] = None
    return verified


def _ratings_from(interpretation: dict) -> dict:
    """The ordinal ratings scoring.py is allowed to convert. Nothing else from
    the model reaches a number."""
    rating = ((interpretation.get("clarity") or {}).get("rating") or "").lower()
    return {"clarity": rating} if rating in an.RATING_LEVELS else {}


async def _load_wordmark(url: Optional[str], deps: VisionDeps):
    """The brand mark, decoded, or None. Never fatal: without it brand detection
    falls back to matching the brand name in OCR text and says so."""
    if not url or not deps.fetch_bytes or not deps.decode_image:
        return None
    try:
        payload = await _maybe_await(deps.fetch_bytes, url)
        return deps.decode_image(payload)
    except Exception as err:  # noqa: BLE001
        logger.warning("could not load the brand wordmark: %s", err)
        return None


async def _render_heatmaps(analysis_id: str, media: dict, measurements: list,
                           deps: VisionDeps, wordmark=None,
                           brand_names: Optional[list] = None) -> dict:
    """Overlay for the strongest frame, plus a thumbnail strip.

    The saliency map is recomputed for the handful of frames that get rendered
    rather than carried for all 48 - holding 48 float maps for the sake of six
    images is the wrong trade on a small box.
    """
    if not deps.render_heatmap or not deps.upload or not measurements:
        return {"hero": None, "strip": [], "storage": "not_configured"}

    frames = media.get("frames") or []
    if not frames:
        return {"hero": None, "strip": [], "storage": "no_frames"}

    # Rank by concentration, but only among frames that are worth showing.
    #
    # TWO WAYS A FRAME CAN BE EMPTY, and both used to win this ranking. A
    # near-black end card drives the saliency map degenerate: with no structure
    # to find, the mass collapses onto one artefact and concentration reads 1.0,
    # the highest possible "the eye is locked here" produced by a frame with
    # nothing in it. A white flash between two shots does the same while looking
    # nothing like it - black letterbox bars over a blank body give it HIGH
    # contrast, so it passed the degeneracy check and was chosen as the hero of
    # a real report, which came back as a near-blank page with one marker on it.
    #
    # is_representative rules out both: flat frames by contrast, transitions by
    # edge detail.
    from .measure import is_representative
    usable = [r for r in measurements if is_representative(r)]
    candidates = usable or measurements
    scored = [(r.get("saliency", {}).get("concentration") or 0.0, r["index"])
              for r in candidates]
    hero_index = max(scored)[1] if scored else 0
    if not usable:
        logger.warning("every frame is low-contrast [analysis_id=%s]", analysis_id)

    hero = None
    try:
        frame = frames[hero_index]
        saliency = deps.predict_saliency(frame)
        if saliency.get("map") is not None:
            # NAME EACH PEAK, do not just place it. Coordinates tell the frontend
            # where to draw the numbered marker; the label tells the marketer
            # what the eye actually went to, which is the part they can act on.
            #
            # Regions are re-detected for this ONE frame rather than read from
            # the measurement record, because measure.py stores `text_boxes` as
            # a count - carrying the geometry of every word across 120 frames to
            # label three markers would be the wrong trade.
            peaks = saliency.get("peaks") or []
            try:
                regions = await _maybe_await(deps.detect_regions, frame,
                                             wordmark=wordmark,
                                             brand_names=brand_names or [])
                peaks = rg.label_peaks(peaks, regions)
            except Exception as err:  # noqa: BLE001 - an unlabelled peak is still a peak
                logger.warning("could not label the peaks [analysis_id=%s]: %s",
                               analysis_id, err)

            png = deps.render_heatmap(frame["pixels"], saliency["map"], peaks)
            key = await _maybe_await(deps.upload, analysis_id,
                                     f"heat_{int(frame['t'] * 1000)}.png", png)
            hero = {"frame_time": frame["t"], "object_key": key, "peaks": peaks}
    except Exception as err:  # noqa: BLE001 - a missing picture is not a failed analysis
        logger.warning("hero heatmap failed [analysis_id=%s]: %s", analysis_id, err)

    # THE POSTER - a plain frame for previews: the /analyze and /history
    # thumbnail_url, and any list view.
    #
    # The same frame the heatmap uses, which is chosen above to skip black
    # openings, fades and white flashes - rendered WITHOUT the overlay. Not the
    # first frame: ads so often open on black that a History list built from
    # first frames would be a column of black squares. Its own try, so a failed
    # heatmap does not cost the poster.
    poster = None
    if deps.make_thumbnail:
        try:
            frame = frames[hero_index]
            key = await _maybe_await(
                deps.upload, analysis_id, f"poster_{int(frame['t'] * 1000)}.png",
                deps.make_thumbnail(frame["pixels"]))
            if key:
                poster = {"frame_time": frame["t"], "object_key": key}
        except Exception as err:  # noqa: BLE001 - a missing picture is not a failed analysis
            logger.warning("poster failed [analysis_id=%s]: %s", analysis_id, err)

    strip = []
    if deps.make_thumbnail and len(frames) > 1:
        step = max(1, len(frames) // STRIP_FRAMES)
        for frame in frames[::step][:STRIP_FRAMES]:
            try:
                key = await _maybe_await(
                    deps.upload, analysis_id,
                    f"thumb_{int(frame['t'] * 1000)}.png",
                    deps.make_thumbnail(frame["pixels"]))
                strip.append({"t": frame["t"], "object_key": key})
            except Exception as err:  # noqa: BLE001
                logger.warning("thumbnail failed at %.1fs: %s", frame["t"], err)

    return {"hero": hero, "strip": strip, "poster": poster,
            "storage": "configured" if hero and hero.get("object_key")
            else "not_configured"}
