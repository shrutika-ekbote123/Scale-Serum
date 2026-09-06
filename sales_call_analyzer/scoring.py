"""
Deterministic scoring - criterion, then stage, then overall. In Python, never in
the model.

THE RULE THIS FILE EXISTS TO ENFORCE
    The LLM produces an ordinal rating per criterion and the evidence for it.
    Every number in the report is computed here, from those ratings and
    sales_framework.json. Consequences, all of them deliberate:

      * The same ratings always produce the same score.
      * When management finally sets weights, historical calls can be rescored
        from their stored ratings without re-running Deepgram or Gemini.
      * Nothing said on the call can move a number, because the thing that
        reads the call never touches the arithmetic.

NO INVENTED BUSINESS VALUES
    Weights are equal until configured. The rating scale is uniform until
    configured. Bands do not exist until configured. Every response reports
    which of those were still placeholders.

ABSENT OPPORTUNITY IS NOT A ZERO
    A criterion the call gave no chance for is excluded from the denominator by
    default, not scored zero. Marking a representative down because the
    customer raised no objection would be an artefact of the framework, not a
    finding about the call. `not_applicable_policy.mode` in the config can
    change this to "zero" when the business decides.

UNSUPPORTED IS NOT A ZERO EITHER
    A criterion whose evidence failed verification is excluded and reported as
    unsupported. We do not know how the representative did; that is different
    from knowing they did badly.
"""
from __future__ import annotations

from typing import Optional

from . import framework as fw
from .models import (
    CriterionEvaluation,
    Insight,
    Scores,
    StageEvaluation,
    StageScore,
)

STATUS_SCORED = "scored"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_UNSUPPORTED = "unsupported"

NOT_EVALUATED_REASON = "The analysis did not return an evaluation for this criterion."


def _weight(value: Optional[float]) -> float:
    """A null weight means equal weighting - the absence of a decision, not a
    decision that everything is worth 1."""
    return 1.0 if value is None else float(value)


def _weighted_mean(pairs: list[tuple[float, float]]) -> Optional[float]:
    total_weight = sum(w for _, w in pairs)
    if total_weight <= 0:
        return None
    return sum(score * w for score, w in pairs) / total_weight


def _insights_for_stage(items: list[dict], stage_id: str) -> list[Insight]:
    return [Insight(**{k: v for k, v in item.items() if k in Insight.model_fields})
            for item in items or [] if item.get("stage_id") == stage_id]


def score_analysis(analysis: dict, cfg: dict,
                   blocked: Optional[dict[str, str]] = None) -> dict:
    """Turn verified ratings into stage evaluations and scores.

    Returns {"stage_evaluations": [...], "scores": Scores, "counts": {...}}.
    Pure: no network, no database, no LLM, no clock.
    """
    blocked = blocked or {}
    scale, scale_mode = fw.rating_scale(cfg)
    score_max = float(cfg["score_max"])
    na_mode = fw.not_applicable_mode(cfg)
    weights_cfg = cfg.get("weights") or {}

    by_id = {c["criterion_id"]: c for c in analysis.get("criteria") or []}
    stage_notes = {s["stage_id"]: s for s in analysis.get("stages") or []}

    stage_evaluations: list[StageEvaluation] = []
    stage_pairs: list[tuple[float, float]] = []
    stage_scores: list[StageScore] = []

    totals = {"criteria_total": 0, "criteria_scored": 0,
              "criteria_not_applicable": 0, "criteria_unsupported": 0}

    for stage in sorted(cfg["stages"], key=lambda s: s.get("order", 0)):
        criteria: list[CriterionEvaluation] = []
        pairs: list[tuple[float, float]] = []
        counts = {"scored": 0, "not_applicable": 0, "unsupported": 0}

        # A stage the config allows to be skipped entirely - "the call ended
        # before any product was presented" - excuses its criteria too. What is
        # NOT excused is a single always-applies criterion being dropped while
        # the rest of its stage is evaluated: that is the model declining to
        # answer, and it quietly shrinks the denominator.
        stage_excused = bool(stage.get("may_be_not_applicable")) and all(
            (by_id.get(c["id"]) is not None
             and not by_id[c["id"]].get("applicable", True))
            or c["id"] in blocked
            for c in stage["criteria"])

        for spec in stage["criteria"]:
            cid = spec["id"]
            totals["criteria_total"] += 1
            item = by_id.get(cid)
            weight = _weight(spec.get("weight"))

            evaluation = CriterionEvaluation(
                criterion_id=cid,
                name=spec["name"],
                stage_id=stage["id"],
                score_max=score_max,
                evidence=[e for e in (item or {}).get("evidence", [])],
                observation=(item or {}).get("observation", ""),
                missing_behaviour=(item or {}).get("missing_behaviour"),
                recommendation=(item or {}).get("recommendation"),
                confidence=(item or {}).get("confidence", "medium"),
                evidence_backed=bool((item or {}).get("evidence_backed", False)),
            )

            # 1. this call could not satisfy the criterion's requirements
            if cid in blocked:
                evaluation.applicable = False
                evaluation.status = STATUS_NOT_APPLICABLE
                evaluation.not_applicable_reason = blocked[cid]
            # 2. the analysis never evaluated it
            elif item is None:
                evaluation.status = STATUS_UNSUPPORTED
                evaluation.not_applicable_reason = NOT_EVALUATED_REASON
            # 3a. the model declined to rate a criterion the config says ALWAYS
            #     applies. Not accepted silently: the config is the authority on
            #     which criteria may be excused, and a criterion quietly leaving
            #     the denominator changes the score. Reported as unsupported so
            #     the reader can see the model dodged it.
            elif (not item.get("applicable", True)
                  and not spec.get("may_be_not_applicable")
                  and not stage_excused):
                evaluation.status = STATUS_UNSUPPORTED
                evaluation.not_applicable_reason = (
                    "The analysis marked this not applicable, but the framework says it "
                    "always applies to a call of this kind, so it was not excused. "
                    f"Stated reason: {item.get('not_applicable_reason') or '(none given)'}")
            # 3b. the call genuinely gave no opportunity for it
            elif not item.get("applicable", True):
                evaluation.applicable = False
                evaluation.status = STATUS_NOT_APPLICABLE
                evaluation.not_applicable_reason = (
                    item.get("not_applicable_reason")
                    or "The call gave no opportunity to observe this.")
            # 4. evidence did not hold up, or no valid rating came back
            elif item.get("status") == STATUS_UNSUPPORTED or item.get("rating") is None:
                evaluation.status = STATUS_UNSUPPORTED
            # 5. scored
            else:
                rating = item["rating"]
                evaluation.rating = rating
                evaluation.score = round(scale[rating] * score_max, 2)
                evaluation.status = STATUS_SCORED

            if evaluation.status == STATUS_SCORED:
                pairs.append((evaluation.score, weight))
                counts["scored"] += 1
            elif evaluation.status == STATUS_NOT_APPLICABLE:
                counts["not_applicable"] += 1
                if na_mode == "zero":
                    # Configured business choice: an absent opportunity scores 0.
                    evaluation.score = 0.0
                    pairs.append((0.0, weight))
            else:
                counts["unsupported"] += 1

            criteria.append(evaluation)

        totals["criteria_scored"] += counts["scored"]
        totals["criteria_not_applicable"] += counts["not_applicable"]
        totals["criteria_unsupported"] += counts["unsupported"]

        stage_score = _weighted_mean(pairs)
        if stage_score is not None:
            status = STATUS_SCORED
            reason = None
        elif counts["not_applicable"] and not counts["unsupported"]:
            status = STATUS_NOT_APPLICABLE
            reason = stage.get("not_applicable_when") or (
                "The call gave no opportunity to observe this stage.")
        else:
            status = STATUS_UNSUPPORTED
            reason = "No criterion in this stage could be evaluated with evidence."

        note = stage_notes.get(stage["id"], {})
        stage_evaluations.append(StageEvaluation(
            stage_id=stage["id"],
            name=stage["name"],
            order=stage.get("order", 0),
            objective=stage.get("objective", ""),
            kpis=list(stage.get("kpis") or []),
            score=round(stage_score, 2) if stage_score is not None else None,
            score_max=score_max,
            status=status,
            not_applicable_reason=reason,
            assessment=note.get("assessment", ""),
            confidence=note.get("confidence", "medium"),
            criteria=criteria,
            strengths=_insights_for_stage(analysis.get("strengths"), stage["id"]),
            weaknesses=_insights_for_stage(analysis.get("weaknesses"), stage["id"]),
            recommendations=_insights_for_stage(analysis.get("recommendations"), stage["id"]),
            criteria_scored=counts["scored"],
            criteria_not_applicable=counts["not_applicable"],
            criteria_unsupported=counts["unsupported"],
        ))
        stage_scores.append(StageScore(
            stage_id=stage["id"], name=stage["name"], order=stage.get("order", 0),
            score=round(stage_score, 2) if stage_score is not None else None,
            score_max=score_max, status=status))

        if stage_score is not None:
            stage_pairs.append((stage_score, _weight(stage.get("weight"))))

    overall = _weighted_mean(stage_pairs)
    scored_stages = sum(1 for s in stage_scores if s.status == STATUS_SCORED)
    na_stages = sum(1 for s in stage_scores if s.status == STATUS_NOT_APPLICABLE)

    scores = Scores(
        overall=round(overall, 2) if overall is not None else None,
        overall_100=round(overall / score_max * 100) if overall is not None else None,
        score_max=score_max,
        band=None,
        band_reason=("thresholds_not_configured" if cfg.get("bands") is None
                     else "not_evaluated"),
        weighting=fw.weighting_mode(cfg),
        rating_scale_mode=scale_mode,
        framework_version=cfg["framework_version"],
        stage_weights_confirmed=bool(weights_cfg.get("stage_weights_confirmed")),
        criterion_weights_confirmed=bool(weights_cfg.get("criterion_weights_confirmed")),
        not_applicable_mode=na_mode,
        stages=stage_scores,
        stages_scored=scored_stages,
        stages_not_applicable=na_stages,
        basis=_basis(scale_mode, fw.weighting_mode(cfg), na_mode, totals),
    )

    return {"stage_evaluations": stage_evaluations, "scores": scores, "counts": totals}


def _basis(scale_mode: str, weighting: str, na_mode: str, totals: dict) -> str:
    """One line saying how this number was produced, in the response itself.

    The house style: a score that cannot explain itself should not be shown.
    """
    parts = [
        f"{totals['criteria_scored']} of {totals['criteria_total']} criteria scored",
        f"{totals['criteria_not_applicable']} not applicable",
        f"{totals['criteria_unsupported']} unsupported",
    ]
    weighting_text = ("equal weighting (no business weights configured)"
                      if weighting == fw.WEIGHTING_EQUAL else "configured weights")
    scale_text = ("uniform rating spacing (no business rating scale configured)"
                  if scale_mode == fw.RATING_SCALE_UNIFORM else "configured rating scale")
    na_text = ("not-applicable criteria excluded from the average" if na_mode == "exclude"
               else "not-applicable criteria scored as zero")
    return f"{', '.join(parts)}; {weighting_text}; {scale_text}; {na_text}."


def rescore(stored_analysis: dict, cfg: dict,
            blocked: Optional[dict[str, str]] = None) -> dict:
    """Recompute scores from a stored analysis without any provider call.

    This is the payoff for storing ratings rather than only numbers: when
    management sets real weights, every historical call can be re-scored for
    free. `stored_analysis` is the verified analysis dict persisted by the
    pipeline.
    """
    return score_analysis(stored_analysis, cfg, blocked)
