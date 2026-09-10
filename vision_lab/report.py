"""
Step 20 - assembly. Turns everything the pipeline produced into the response.

ASSEMBLE, NEVER COMPUTE
    Every number here was produced by scoring.py, timeline.py or measure.py and
    is copied in as it stands. If this module ever starts calculating, a report
    and a /rescore of the same analysis can disagree, and there is no way for a
    reader to tell which of them is wrong.

EVERY BLOCK SAYS WHETHER IT IS THERE
    A missing transcript, an unavailable interpretation and a trigger nobody
    could judge are each reported with a reason rather than omitted. An absent
    key and a null are the same thing to a frontend and very different things to
    a marketer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from . import KIND_VIDEO, PROMPT_VERSION, STATUS_COMPLETED
from . import analyzer as an
from . import framework as fw


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def media_summary(media: dict, kind: str, sample_fps: float) -> dict:
    return {
        "kind": kind,
        "duration_seconds": media.get("duration_seconds"),
        "width": media.get("width"),
        "height": media.get("height"),
        "frames_analyzed": len(media.get("frames") or []),
        "sample_fps": sample_fps,
        "has_audio": bool(media.get("has_audio")),
        "codec": media.get("codec"),
        "source_fps": media.get("fps"),
        "size_bytes": media.get("size_bytes"),
        "shot_count": len(media.get("shots") or []),
        "requested_sample_fps": media.get("requested_sample_fps"),
        "sample_fps_thinned": bool(media.get("sample_fps_thinned")),
        "coverage_seconds": media.get("coverage_seconds"),
        "coverage_fraction": media.get("coverage_fraction"),
    }


def build_report(doc: dict, *, media: dict, summary: dict, heatmaps: dict,
                 kind: str, sample_fps: float, notes: list[str],
                 vision: dict, timeline: Optional[dict] = None,
                 key_moments: Optional[list] = None,
                 scored: Optional[dict] = None,
                 defects: Optional[list] = None,
                 transcript: Optional[dict] = None,
                 triggers: Optional[dict] = None,
                 interpretation: Optional[dict] = None) -> dict:
    """The response, field for field. See the module docstring for the rules."""
    cfg = fw.load_framework()
    metrics = fw.metric_index(cfg)
    band, band_reason = fw.bands(cfg)
    is_video = kind == KIND_VIDEO

    # THE FALLBACK CARRIES THE SAME KEYS AS THE REAL THING.
    #
    # A block whose shape depends on whether a pass ran is a block a frontend
    # cannot code against: `overall.metrics_scored` was simply absent here,
    # while every scored report has it. Same fields, null values.
    scored = scored or {}
    scores = scored.get("scores") or {
        metric_id: {
            "score": None,
            "label": metrics[metric_id].get("label", ""),
            "direction": metrics[metric_id].get("direction", "higher_better"),
            "basis": metrics[metric_id].get("basis", "measured"),
            "reason": "not_scored",
            "signals": {},
        }
        for metric_id in fw.METRIC_IDS
    }
    overall = scored.get("overall") or {
        "score": None, "band": band, "band_reason": band_reason,
        "score_reason": "not_scored", "weighting": fw.weighting_mode(cfg),
        "metrics_scored": 0, "metrics_missing": list(fw.METRIC_IDS)}

    interpretation = interpretation or {}
    interpreted = bool(interpretation.get("available"))
    hero = heatmaps.get("hero") or {}
    return {
        # Nothing in this payload is a placeholder any more: frames, saliency,
        # OCR, the timeline, all six scores, the transcript, the 15 triggers and
        # the recommendations are each either real or state why they are absent.
        "stub": False,
        "analysis_id": doc["_id"],
        "creative_id": doc.get("creative_id"),
        "ad_number": doc.get("ad_number"),
        "division": doc.get("division"),
        "status": STATUS_COMPLETED,
        "availability": {"available": True, "reason": None, "message": None},

        "media": media_summary(media, kind, sample_fps),
        "vision": vision,

        "overall": overall,
        # The written summary is the model's. Absent when it could not run - the
        # scores below do not wait on it and are not affected by it.
        "summary": interpretation.get("summary") or "",
        "scores": scores,

        # Real, and the point of this milestone.
        "measurements": summary,

        "heatmap": {
            "frame_time": hero.get("frame_time"),
            "object_key": hero.get("object_key"),
            "image_url": None,          # signed at read time by the GET endpoint
            "peaks": hero.get("peaks") or [],
        },
        "thumbnails": heatmaps.get("strip") or [],
        # A plain frame of the ad, no overlay - what thumbnail_url points to on
        # /analyze and /history. The heatmap's frame, so never a black opening.
        # Signed at read time like every other image.
        "poster": {
            "frame_time": (heatmaps.get("poster") or {}).get("frame_time"),
            "object_key": (heatmaps.get("poster") or {}).get("object_key"),
            "image_url": None,
        },
        "key_moments": key_moments or [],
        # `signals` is dropped from the response: six numbers per frame over 120
        # frames is a large payload nothing renders. It stays in the stored
        # measurement record for /rescore.
        "timeline": ({k: v for k, v in timeline.items() if k != "signals"}
                     if timeline else None),
        "transcript": _transcript_block(transcript),
        "psychology": triggers or {
            "triggers": [],
            "coverage": {"present": 0, "weak": 0, "absent": 0, "not_applicable": 0},
        },
        # The measured findings, worst first. Every recommendation below is
        # written ABOUT one of these and carries its timestamp and its counted
        # number - there is no recommendation without a defect behind it.
        "defects": defects or [],
        "recommendations": interpretation.get("recommendations") or [],
        "key_message": interpretation.get("key_message"),
        "observations": interpretation.get("observations") or [],
        "interpretation": {
            "available": interpreted,
            "reason": interpretation.get("reason"),
            "message": interpretation.get("message"),
            "prompt_version": interpretation.get("prompt_version"),
            # What the model tried to say and could not support. Published
            # rather than swallowed: a caller measuring how often the model
            # invents things needs to be able to see it.
            "evidence_audit": interpretation.get("evidence_audit"),
            "dropped_invented_recommendations":
                interpretation.get("dropped_invented_recommendations") or [],
        },

        "notes": notes,
        "versions": {
            **fw.versions(),
            "prompt_version": PROMPT_VERSION,
            "interpret_prompt_version": an.PROMPT_VERSION,
            "saliency_model": vision.get("saliency_model"),
        },
        "config_disclosure": fw.config_disclosure(),
        "created_at": doc.get("created_at"),
        "updated_at": now_utc(),
    }


def _transcript_block(transcript: Optional[dict]) -> Optional[dict]:
    """The transcript as the report publishes it.

    `confidence` is dropped per line - it is a provider detail nothing renders -
    but `attention` and `in_weak_zone` are kept, because those are what draw the
    warning icon beside a line in the prototype's transcript panel.
    """
    if not transcript:
        return None
    return {
        "available": bool(transcript.get("segments")),
        "reason": transcript.get("reason"),
        "message": transcript.get("message"),
        "language": transcript.get("language"),
        "stats": transcript.get("stats") or {},
        "segments": [
            {"t": s.get("t"), "end": s.get("end"), "text": s.get("text"),
             "attention": s.get("attention"),
             "in_weak_zone": bool(s.get("in_weak_zone"))}
            for s in transcript.get("segments") or []],
    }
