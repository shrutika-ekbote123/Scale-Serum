"""
Report assembly - PRESENTATION ONLY.

WHAT THIS DOES
    Takes the verified analysis, the computed scores, the transcript and the
    context, and assembles the object the API returns and the Sales Calls UI
    renders.

WHAT THIS MUST NOT DO
    Change a score, a rating, an evidence anchor, or which findings exist. Every
    number here is passed through untouched. Same contract as explain.py in the
    purchase probability package: it arranges and labels, it never rescores.

NO PADDING
    If only three highlights are genuinely supported, the report shows three.
    Filling a list to reach a target count would put sentences in front of a
    sales manager that no evidence supports.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .models import (
    AnalysisQuality,
    Availability,
    CallSummaryBlock,
    ContextAdaptation,
    ContextUsed,
    CustomerNeed,
    CustomerSignal,
    Evidence,
    Highlight,
    Insight,
    NormalizedTranscript,
    Objection,
    PitchStructureAssessment,
    ProcessingInfo,
    RepTechnique,
    SalesCallAnalysis,
    Scores,
    StageEvaluation,
)
from . import STATUS_COMPLETED, REASON_TEXT
from .context import AnalysisContext


def _evidence(items: Any) -> list[Evidence]:
    out: list[Evidence] = []
    for item in items or []:
        if isinstance(item, Evidence):
            out.append(item)
        elif isinstance(item, dict):
            out.append(Evidence(**{k: v for k, v in item.items()
                                   if k in Evidence.model_fields}))
    return out


def _build(model, item: dict, **overrides):
    data = {k: v for k, v in (item or {}).items() if k in model.model_fields}
    data["evidence"] = _evidence((item or {}).get("evidence"))
    data.update(overrides)
    return model(**data)


def _insights(items: Any) -> list[Insight]:
    return [_build(Insight, item) for item in items or [] if (item or {}).get("text")]


def _label_for(kind: str, signals: dict, block: str) -> str:
    for entry in (signals.get(block) or {}).get("types") or []:
        if entry["id"] == kind:
            return entry.get("label") or kind
    return kind


def build_report(*, analysis_id: str, ctx: AnalysisContext,
                 transcript: NormalizedTranscript,
                 analysis: dict, scoring: dict, signals: dict,
                 context_used: ContextUsed,
                 evidence_stats: dict,
                 processing: ProcessingInfo,
                 lead_id: Optional[str] = None,
                 brand_id: Optional[str] = None,
                 created_at: Optional[datetime] = None,
                 status: str = STATUS_COMPLETED,
                 availability: Optional[Availability] = None,
                 degraded: bool = False,
                 extra_warnings: Optional[list[str]] = None) -> SalesCallAnalysis:
    """Assemble the final report. Pure: no clock beyond `updated_at`, no I/O."""
    scores: Optional[Scores] = scoring.get("scores") if scoring else None
    stage_evaluations: list[StageEvaluation] = (scoring or {}).get("stage_evaluations") or []
    counts = (scoring or {}).get("counts") or {}

    warnings = list(transcript.quality.warnings) + list(extra_warnings or [])

    quality = AnalysisQuality(
        criteria_total=counts.get("criteria_total", 0),
        criteria_scored=counts.get("criteria_scored", 0),
        criteria_not_applicable=counts.get("criteria_not_applicable", 0),
        criteria_unsupported=counts.get("criteria_unsupported", 0),
        evidence_anchors_total=evidence_stats.get("evidence_anchors_total", 0),
        evidence_anchors_dropped=evidence_stats.get("evidence_anchors_dropped", 0),
        transcript_confidence=transcript.quality.mean_confidence,
        degraded=degraded,
        warnings=warnings,
    )

    call = CallSummaryBlock(
        call_id=ctx.call_id,
        disposition=ctx.call.disposition,
        disposition_source="rep_reported",   # the AI never replaces the CRM outcome
        direction=ctx.call.direction,
        occurred_at=ctx.call.occurred_at,
        duration_seconds=ctx.call.duration_seconds or transcript.duration_seconds,
        remarks=ctx.call.remarks,
        provider=ctx.call.provider,
        recording_reference=ctx.call.recording_reference,
        rep=ctx.rep,
        customer=ctx.customer,
    )

    pitch = analysis.get("pitch_structure")
    adaptation = analysis.get("context_adaptation")

    return SalesCallAnalysis(
        analysis_id=analysis_id,
        call_id=ctx.call_id,
        lead_id=lead_id or ctx.lead_id,
        brand_id=brand_id or ctx.brand.brand_id,
        status=status,
        availability=availability or Availability(available=True),
        created_at=created_at,
        updated_at=datetime.now(timezone.utc),
        call=call,
        scores=scores,
        stage_evaluations=stage_evaluations,
        summary=analysis.get("summary", ""),
        highlights=[_build(Highlight, h) for h in analysis.get("highlights") or []],
        strengths=_insights(analysis.get("strengths")),
        weaknesses=_insights(analysis.get("weaknesses")),
        recommendations=_insights(analysis.get("recommendations")),
        customer_needs=[_build(CustomerNeed, n) for n in analysis.get("customer_needs") or []],
        objections=[_build(Objection, o) for o in analysis.get("objections") or []],
        buying_signals=[_build(CustomerSignal, s,
                               label=_label_for(s.get("type", ""), signals, "customer_signals"))
                        for s in analysis.get("buying_signals") or []],
        customer_signals=[_build(CustomerSignal, s,
                                 label=_label_for(s.get("type", ""), signals, "customer_signals"))
                          for s in analysis.get("customer_signals") or []],
        rep_techniques=[_build(RepTechnique, t,
                               label=_label_for(t.get("type", ""), signals, "rep_techniques"))
                        for t in analysis.get("rep_techniques") or []],
        pitch_structure=_build(PitchStructureAssessment, pitch) if pitch else None,
        # Reported alongside the scores, never folded into them: whether context
        # adaptation should move a number is a business decision nobody has made.
        context_adaptation=(_build(ContextAdaptation, adaptation, affects_score=False)
                            if adaptation else None),
        transcript=transcript,
        context_used=context_used,
        analysis_quality=quality,
        processing=processing,
        fallback=False,
    )


def build_failed_report(*, analysis_id: str, call_id: str, reason: str,
                        message: Optional[str] = None,
                        status: str = "failed",
                        lead_id: Optional[str] = None,
                        transcript: Optional[NormalizedTranscript] = None,
                        context_used: Optional[ContextUsed] = None,
                        processing: Optional[ProcessingInfo] = None,
                        created_at: Optional[datetime] = None,
                        call: Optional[CallSummaryBlock] = None) -> SalesCallAnalysis:
    """A failure, stated honestly.

    `scores` stays null. A neutral scorecard would look like a real evaluation
    of a real call, and a sales manager has no way to tell the difference - so
    a failed analysis never invents one. Any transcript we did manage to produce
    is kept, both for the reader and so a retry does not pay for it again.
    """
    return SalesCallAnalysis(
        analysis_id=analysis_id,
        call_id=call_id,
        lead_id=lead_id,
        status=status,
        availability=Availability(available=False, reason=reason,
                                  message=message or REASON_TEXT.get(reason, reason)),
        created_at=created_at,
        updated_at=datetime.now(timezone.utc),
        call=call,
        scores=None,
        transcript=transcript,
        context_used=context_used or ContextUsed(),
        processing=processing or ProcessingInfo(),
        fallback=True,
    )
