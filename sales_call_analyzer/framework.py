"""
Sales Framework configuration - loading, validation and derived values.

WHY THIS FILE EXISTS
    Every business value in the framework (weights, the rating scale, bands,
    the not-applicable policy, the disposition policy) lives in
    sales_framework.json, not in Python. Management has not finalised any of
    them. This module loads that file, validates it hard enough that a typo
    fails at startup rather than silently mis-scoring a call, and derives the
    neutral placeholders used while the real values are undecided.

TWO PLACEHOLDERS, AND WHY THEY ARE NOT BUSINESS DECISIONS
    * Equal weighting when a weight is null. Treating every criterion and every
      stage alike is the absence of a judgement, not a judgement.
    * Uniform rating spacing when rating_scale is null. With N ordered rating
      levels, level i maps to i/(N-1). Nothing here decides that "adequate" is
      worth more or less than an even share of the ladder - it just spaces the
      ladder evenly, which is what "we have not been told" looks like in
      numbers.

    Both are reported in every API response (`weighting`, `rating_scale_mode`,
    `*_confirmed: false`) so a placeholder can never be read as a decision.
    Setting real values is a JSON edit plus a framework_version bump.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_PATH = os.path.join(PKG_DIR, "sales_framework.json")
SIGNALS_PATH = os.path.join(PKG_DIR, "signals_config.json")

# Rating scale modes, reported on every scored response.
RATING_SCALE_UNIFORM = "uniform_placeholder"
RATING_SCALE_CONFIGURED = "configured"

# Weighting modes, reported on every scored response.
WEIGHTING_EQUAL = "equal_unweighted_placeholder"
WEIGHTING_CONFIGURED = "configured"

_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()


class FrameworkConfigError(ValueError):
    """The framework or signals config is unusable. Raised at load time so a bad
    config fails loudly instead of producing quietly wrong scores."""


# --------------------------------------------------------------------------- loading
def _load_json(path: str, label: str) -> dict:
    if not os.path.exists(path):
        raise FrameworkConfigError(f"{label} is missing: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as err:
        raise FrameworkConfigError(f"{label} is not valid JSON: {err}") from err
    if not isinstance(data, dict):
        raise FrameworkConfigError(f"{label} must be a JSON object")
    return data


def load_framework(force: bool = False) -> dict:
    """Load, validate and cache sales_framework.json."""
    if not force and "framework" in _CACHE:
        return _CACHE["framework"]
    with _LOCK:
        if not force and "framework" in _CACHE:
            return _CACHE["framework"]
        cfg = _load_json(FRAMEWORK_PATH, "sales_framework.json")
        validate_framework(cfg)
        _CACHE["framework"] = cfg
        return cfg


def load_signals(force: bool = False) -> dict:
    """Load, validate and cache signals_config.json."""
    if not force and "signals" in _CACHE:
        return _CACHE["signals"]
    with _LOCK:
        if not force and "signals" in _CACHE:
            return _CACHE["signals"]
        cfg = _load_json(SIGNALS_PATH, "signals_config.json")
        validate_signals(cfg)
        _CACHE["signals"] = cfg
        return cfg


# --------------------------------------------------------------------------- validation
def _check_weight(value: Any, where: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise FrameworkConfigError(
            f"{where}: weight must be null (equal weighting) or a positive number, got {value!r}")


def validate_framework(cfg: dict) -> None:
    """Structural validation. Deliberately strict about the things that would
    otherwise produce a plausible-looking but wrong score."""
    if not str(cfg.get("framework_version") or "").strip():
        raise FrameworkConfigError("framework_version is required")

    score_max = cfg.get("score_max")
    if not isinstance(score_max, (int, float)) or isinstance(score_max, bool) or score_max <= 0:
        raise FrameworkConfigError(f"score_max must be a positive number, got {score_max!r}")

    levels = cfg.get("rating_levels")
    if not isinstance(levels, list) or len(levels) < 2:
        raise FrameworkConfigError("rating_levels must be a list of at least two ordered levels")
    if len(set(levels)) != len(levels):
        raise FrameworkConfigError("rating_levels contains duplicates")
    if not all(isinstance(lv, str) and lv.strip() for lv in levels):
        raise FrameworkConfigError("rating_levels must all be non-empty strings")

    scale = cfg.get("rating_scale")
    if scale is not None:
        if not isinstance(scale, dict):
            raise FrameworkConfigError("rating_scale must be null or an object keyed by rating level")
        missing = [lv for lv in levels if lv not in scale]
        if missing:
            raise FrameworkConfigError(f"rating_scale is missing levels: {missing}")
        for lv, val in scale.items():
            if lv not in levels:
                raise FrameworkConfigError(f"rating_scale has unknown level {lv!r}")
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not 0.0 <= val <= 1.0:
                raise FrameworkConfigError(
                    f"rating_scale[{lv!r}] must be a fraction between 0.0 and 1.0, got {val!r}")

    policy = cfg.get("not_applicable_policy") or {}
    mode = policy.get("mode", "exclude")
    if mode not in ("exclude", "zero"):
        raise FrameworkConfigError(
            f"not_applicable_policy.mode must be 'exclude' or 'zero', got {mode!r}")

    stages = cfg.get("stages")
    if not isinstance(stages, list) or not stages:
        raise FrameworkConfigError("stages must be a non-empty list")

    seen_stage_ids: set[str] = set()
    seen_criterion_ids: set[str] = set()
    requirements = set((cfg.get("requirements") or {}).keys())

    for stage in stages:
        sid = str(stage.get("id") or "").strip()
        if not sid:
            raise FrameworkConfigError("every stage needs an id")
        if sid in seen_stage_ids:
            raise FrameworkConfigError(f"duplicate stage id: {sid}")
        seen_stage_ids.add(sid)
        if not str(stage.get("name") or "").strip():
            raise FrameworkConfigError(f"stage {sid}: name is required")
        _check_weight(stage.get("weight"), f"stage {sid}")

        criteria = stage.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise FrameworkConfigError(f"stage {sid}: criteria must be a non-empty list")

        for crit in criteria:
            cid = str(crit.get("id") or "").strip()
            if not cid:
                raise FrameworkConfigError(f"stage {sid}: every criterion needs an id")
            if cid in seen_criterion_ids:
                raise FrameworkConfigError(f"duplicate criterion id: {cid}")
            seen_criterion_ids.add(cid)
            if not str(crit.get("name") or "").strip():
                raise FrameworkConfigError(f"criterion {cid}: name is required")
            _check_weight(crit.get("weight"), f"criterion {cid}")
            for req in crit.get("requires") or []:
                if req not in requirements:
                    raise FrameworkConfigError(
                        f"criterion {cid}: requires unknown requirement {req!r}; "
                        f"add it to the 'requirements' block")

    bands = cfg.get("bands")
    if bands is not None and not isinstance(bands, list):
        raise FrameworkConfigError("bands must be null or a list of band objects")


def validate_signals(cfg: dict) -> None:
    if not str(cfg.get("signals_version") or "").strip():
        raise FrameworkConfigError("signals_version is required")
    for block in ("customer_signals", "rep_techniques"):
        types = (cfg.get(block) or {}).get("types")
        if not isinstance(types, list) or not types:
            raise FrameworkConfigError(f"{block}.types must be a non-empty list")
        ids = [str(t.get("id") or "") for t in types]
        if not all(ids):
            raise FrameworkConfigError(f"{block}.types: every entry needs an id")
        if len(set(ids)) != len(ids):
            raise FrameworkConfigError(f"{block}.types contains duplicate ids")
    pitch = (cfg.get("pitch_structures") or {}).get("types")
    if not isinstance(pitch, list) or not pitch:
        raise FrameworkConfigError("pitch_structures.types must be a non-empty list")


# --------------------------------------------------------------------------- derived values
def rating_scale(cfg: dict) -> tuple[dict[str, float], str]:
    """(level -> fraction of score_max, mode).

    Configured scale wins. Otherwise the ordered rating_levels are spaced
    uniformly over [0, 1], which is the neutral stand-in for a decision nobody
    has made. See the module docstring.
    """
    levels: list[str] = list(cfg["rating_levels"])
    configured = cfg.get("rating_scale")
    if configured:
        return {lv: float(configured[lv]) for lv in levels}, RATING_SCALE_CONFIGURED
    span = len(levels) - 1
    return {lv: (i / span) for i, lv in enumerate(levels)}, RATING_SCALE_UNIFORM


def weighting_mode(cfg: dict) -> str:
    weights = cfg.get("weights") or {}
    confirmed = bool(weights.get("stage_weights_confirmed")) and \
        bool(weights.get("criterion_weights_confirmed"))
    any_set = any(s.get("weight") is not None for s in cfg["stages"]) or \
        any(c.get("weight") is not None for s in cfg["stages"] for c in s["criteria"])
    return WEIGHTING_CONFIGURED if (confirmed and any_set) else WEIGHTING_EQUAL


def not_applicable_mode(cfg: dict) -> str:
    return (cfg.get("not_applicable_policy") or {}).get("mode", "exclude")


def stage_index(cfg: dict) -> dict[str, dict]:
    return {s["id"]: s for s in cfg["stages"]}


def criterion_index(cfg: dict) -> dict[str, dict]:
    """criterion_id -> {**criterion, stage_id, stage_name}. The lookup used to
    reject criterion ids the model invented."""
    out: dict[str, dict] = {}
    for stage in cfg["stages"]:
        for crit in stage["criteria"]:
            out[crit["id"]] = {**crit, "stage_id": stage["id"], "stage_name": stage["name"]}
    return out


def unmet_requirements(cfg: dict, satisfied: set[str]) -> dict[str, str]:
    """Requirements the current call cannot satisfy, with the reason text from
    the config. Criteria tagged with these become not-applicable rather than
    being guessed at (e.g. tone criteria on a pasted text transcript)."""
    reqs = cfg.get("requirements") or {}
    return {name: text for name, text in reqs.items() if name not in satisfied}


def criteria_blocked_by_requirements(cfg: dict, satisfied: set[str]) -> dict[str, str]:
    """criterion_id -> reason, for every criterion whose requirements are unmet."""
    reqs = cfg.get("requirements") or {}
    blocked: dict[str, str] = {}
    for stage in cfg["stages"]:
        for crit in stage["criteria"]:
            unmet = [r for r in (crit.get("requires") or []) if r not in satisfied]
            if unmet:
                blocked[crit["id"]] = reqs.get(unmet[0], f"requires {unmet[0]}")
    return blocked


def skip_decision(cfg: dict, disposition: Optional[str], duration_seconds: Optional[float],
                  segment_count: Optional[int]) -> tuple[bool, Optional[str]]:
    """Should the six-stage evaluation be skipped for this call?

    Driven entirely by disposition_policy in the config, which ships empty: no
    business rule about which dispositions are worth evaluating has been set, so
    by default nothing is skipped. Returns (skip, reason_code).
    """
    policy = cfg.get("disposition_policy") or {}

    skip_list = {str(d).strip().lower() for d in (policy.get("skip_full_evaluation") or [])}
    if disposition and str(disposition).strip().lower() in skip_list:
        return True, "disposition_excluded_by_policy"

    min_duration = policy.get("minimum_duration_seconds")
    if min_duration is not None and duration_seconds is not None and duration_seconds < float(min_duration):
        return True, "call_shorter_than_configured_minimum"

    min_segments = policy.get("minimum_segments")
    if min_segments is not None and segment_count is not None and segment_count < int(min_segments):
        return True, "transcript_shorter_than_configured_minimum"

    return False, None


def config_disclosure(cfg: dict) -> dict:
    """What the API reports about its own configuration on every scored call, so
    nobody mistakes a placeholder for a management decision."""
    scale, scale_mode = rating_scale(cfg)
    weights = cfg.get("weights") or {}
    policy = cfg.get("not_applicable_policy") or {}
    disposition = cfg.get("disposition_policy") or {}
    return {
        "framework_version": cfg["framework_version"],
        "score_max": cfg["score_max"],
        "weighting": weighting_mode(cfg),
        "stage_weights_confirmed": bool(weights.get("stage_weights_confirmed")),
        "criterion_weights_confirmed": bool(weights.get("criterion_weights_confirmed")),
        "rating_scale_mode": scale_mode,
        "rating_scale": {k: round(v, 6) for k, v in scale.items()},
        "not_applicable_mode": policy.get("mode", "exclude"),
        "not_applicable_policy_confirmed": bool(policy.get("confirmed")),
        "bands_configured": cfg.get("bands") is not None,
        "disposition_policy_confirmed": bool(disposition.get("confirmed")),
    }


def render_for_prompt(cfg: dict, blocked: Optional[dict[str, str]] = None) -> str:
    """The framework as the model sees it: stages, objectives, KPIs and criterion
    ids. Criteria whose requirements this call cannot meet are marked so the
    model reports them not-applicable instead of inventing an assessment.

    No weights, no rating scale and no score_max are ever rendered here - the
    model must not know how its ratings will be turned into numbers, because it
    must not be optimising for a number.
    """
    blocked = blocked or {}
    lines: list[str] = []
    for stage in sorted(cfg["stages"], key=lambda s: s.get("order", 0)):
        lines.append(f"STAGE {stage.get('order')}: {stage['name']}  (stage_id: {stage['id']})")
        lines.append(f"  Objective: {stage.get('objective', '')}")
        kpis = ", ".join(stage.get("kpis") or [])
        if kpis:
            lines.append(f"  KPIs: {kpis}")
        if stage.get("not_applicable_when"):
            lines.append(f"  This stage is not applicable when: {stage['not_applicable_when']}")
        lines.append("  Criteria:")
        for crit in stage["criteria"]:
            suffix = ""
            if crit["id"] in blocked:
                suffix = f"  [NOT APPLICABLE for this call: {blocked[crit['id']]}]"
            elif crit.get("may_be_not_applicable"):
                suffix = "  [may be not applicable]"
            lines.append(f"    - {crit['id']}: {crit['name']}{suffix}")
        lines.append("")
    return "\n".join(lines).strip()
