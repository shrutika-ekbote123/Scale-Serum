"""Sales Framework configuration: loading, validation and derived placeholders.

Pure unit tests - no network, no database, no API keys. The load-bearing
assertions are the ones about what the config must NOT quietly contain: no
invented weights, no invented rating scale, no invented thresholds, and no
disposition rule nobody signed off on.
"""
from __future__ import annotations

import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import sales_call_analyzer as sca  # noqa: E402
from sales_call_analyzer import framework as fw  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return fw.load_framework()


@pytest.fixture(scope="module")
def signals():
    return fw.load_signals()


# --------------------------------------------------------------------------- shape
def test_six_mandatory_stages_in_order(cfg):
    expected = ["call_opening", "purpose_of_call", "probing",
                "product_pitching", "objection_handling", "closing"]
    got = [s["id"] for s in sorted(cfg["stages"], key=lambda s: s["order"])]
    assert got == expected
    assert [s["order"] for s in cfg["stages"]] == [1, 2, 3, 4, 5, 6]


def test_every_stage_has_objective_kpis_and_criteria(cfg):
    for stage in cfg["stages"]:
        assert stage["objective"].strip(), stage["id"]
        assert stage["kpis"], stage["id"]
        assert stage["criteria"], stage["id"]
        for crit in stage["criteria"]:
            assert crit["name"].strip(), crit["id"]


def test_criterion_ids_are_globally_unique(cfg):
    ids = [c["id"] for s in cfg["stages"] for c in s["criteria"]]
    assert len(ids) == len(set(ids))


def test_criterion_index_maps_back_to_its_stage(cfg):
    index = fw.criterion_index(cfg)
    assert index["probing_active_listening"]["stage_id"] == "probing"
    assert index["closing_signoff"]["stage_id"] == "closing"
    assert len(index) == sum(len(s["criteria"]) for s in cfg["stages"])


# --------------------------------------------------------------------------- no invented business values
def test_no_business_weights_are_set(cfg):
    """Management has not set weights. Nothing may ship pretending otherwise."""
    for stage in cfg["stages"]:
        assert stage["weight"] is None, f"stage {stage['id']} has an invented weight"
        for crit in stage["criteria"]:
            assert crit["weight"] is None, f"criterion {crit['id']} has an invented weight"
    assert cfg["weights"]["stage_weights_confirmed"] is False
    assert cfg["weights"]["criterion_weights_confirmed"] is False


def test_no_rating_scale_no_bands_no_disposition_rules(cfg):
    assert cfg["rating_scale"] is None
    assert cfg["bands"] is None
    assert cfg["disposition_policy"]["skip_full_evaluation"] == []
    assert cfg["disposition_policy"]["minimum_duration_seconds"] is None
    assert cfg["disposition_policy"]["minimum_segments"] is None
    assert cfg["disposition_policy"]["confirmed"] is False
    assert cfg["not_applicable_policy"]["confirmed"] is False


def test_weighting_mode_reports_placeholder(cfg):
    assert fw.weighting_mode(cfg) == fw.WEIGHTING_EQUAL


def test_uniform_rating_scale_is_evenly_spaced(cfg):
    scale, mode = fw.rating_scale(cfg)
    assert mode == fw.RATING_SCALE_UNIFORM
    assert scale["absent"] == 0.0
    assert scale["strong"] == 1.0
    # Evenly spaced: the gap between consecutive levels is identical.
    values = [scale[lv] for lv in cfg["rating_levels"]]
    gaps = [round(b - a, 9) for a, b in zip(values, values[1:])]
    assert len(set(gaps)) == 1


def test_configured_rating_scale_wins(cfg):
    custom = copy.deepcopy(cfg)
    custom["rating_scale"] = {"absent": 0.0, "weak": 0.2, "adequate": 0.8, "strong": 1.0}
    scale, mode = fw.rating_scale(custom)
    assert mode == fw.RATING_SCALE_CONFIGURED
    assert scale["adequate"] == 0.8


def test_configured_weights_flip_the_reported_mode(cfg):
    custom = copy.deepcopy(cfg)
    custom["stages"][0]["weight"] = 2.0
    custom["weights"]["stage_weights_confirmed"] = True
    custom["weights"]["criterion_weights_confirmed"] = True
    assert fw.weighting_mode(custom) == fw.WEIGHTING_CONFIGURED


def test_config_disclosure_admits_what_is_unconfirmed(cfg):
    d = fw.config_disclosure(cfg)
    assert d["weighting"] == fw.WEIGHTING_EQUAL
    assert d["rating_scale_mode"] == fw.RATING_SCALE_UNIFORM
    assert d["stage_weights_confirmed"] is False
    assert d["criterion_weights_confirmed"] is False
    assert d["bands_configured"] is False
    assert d["disposition_policy_confirmed"] is False
    assert d["framework_version"] == cfg["framework_version"]


# --------------------------------------------------------------------------- validation
def test_duplicate_criterion_id_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["stages"][1]["criteria"][0]["id"] = bad["stages"][0]["criteria"][0]["id"]
    with pytest.raises(fw.FrameworkConfigError, match="duplicate criterion id"):
        fw.validate_framework(bad)


def test_negative_weight_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["stages"][0]["weight"] = -1
    with pytest.raises(fw.FrameworkConfigError, match="positive number"):
        fw.validate_framework(bad)


def test_unknown_requirement_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["stages"][0]["criteria"][0]["requires"] = ["telepathy"]
    with pytest.raises(fw.FrameworkConfigError, match="unknown requirement"):
        fw.validate_framework(bad)


def test_incomplete_rating_scale_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["rating_scale"] = {"absent": 0.0, "strong": 1.0}
    with pytest.raises(fw.FrameworkConfigError, match="missing levels"):
        fw.validate_framework(bad)


def test_out_of_range_rating_scale_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["rating_scale"] = {"absent": 0.0, "weak": 0.3, "adequate": 0.6, "strong": 11}
    with pytest.raises(fw.FrameworkConfigError, match="fraction between"):
        fw.validate_framework(bad)


def test_bad_not_applicable_mode_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["not_applicable_policy"]["mode"] = "penalise"
    with pytest.raises(fw.FrameworkConfigError, match="exclude"):
        fw.validate_framework(bad)


# --------------------------------------------------------------------------- requirements
def test_tone_criteria_are_blocked_without_audio(cfg):
    """A pasted text transcript carries no tone, so tone criteria must become
    not-applicable rather than being guessed at."""
    blocked = fw.criteria_blocked_by_requirements(cfg, satisfied=set())
    assert "opening_tone" in blocked
    assert "objection_listens" in blocked
    assert "objection_tone" in blocked
    assert "probing_challenges" not in blocked


def test_nothing_is_blocked_when_all_requirements_are_met(cfg):
    all_reqs = set(cfg["requirements"].keys())
    assert fw.criteria_blocked_by_requirements(cfg, satisfied=all_reqs) == {}
    assert fw.unmet_requirements(cfg, satisfied=all_reqs) == {}


# --------------------------------------------------------------------------- skip policy
def test_nothing_is_skipped_by_default(cfg):
    for disposition in ("No Answer", "Invalid number", "SQL", None):
        skip, reason = fw.skip_decision(cfg, disposition, duration_seconds=3,
                                        segment_count=1)
        assert skip is False, disposition
        assert reason is None


def test_skip_applies_once_configured(cfg):
    custom = copy.deepcopy(cfg)
    custom["disposition_policy"]["skip_full_evaluation"] = ["No Answer", "Invalid number"]
    custom["disposition_policy"]["minimum_duration_seconds"] = 30

    skip, reason = fw.skip_decision(custom, "no answer", 300, 40)
    assert (skip, reason) == (True, sca.DISPOSITION_EXCLUDED)

    skip, reason = fw.skip_decision(custom, "SQL", 12, 40)
    assert (skip, reason) == (True, sca.BELOW_MINIMUM_DURATION)

    skip, reason = fw.skip_decision(custom, "SQL", 300, 40)
    assert (skip, reason) == (False, None)


# --------------------------------------------------------------------------- prompt rendering
def test_prompt_rendering_hides_the_scoring_machinery(cfg):
    """The model must not see weights or the rating scale - it must not be
    optimising for a number it is not allowed to produce."""
    text = fw.render_for_prompt(cfg, blocked={})
    assert "Call Opening" in text
    assert "probing_active_listening" in text
    assert "weight" not in text.lower()
    assert "rating_scale" not in text
    assert "score_max" not in text


def test_prompt_rendering_marks_blocked_criteria(cfg):
    blocked = fw.criteria_blocked_by_requirements(cfg, satisfied=set())
    text = fw.render_for_prompt(cfg, blocked=blocked)
    assert "NOT APPLICABLE for this call" in text
    line = next(ln for ln in text.splitlines() if "opening_tone:" in ln)
    assert "NOT APPLICABLE" in line


# --------------------------------------------------------------------------- signals config
def test_signal_vocabularies_are_closed_and_unique(signals):
    for block in ("customer_signals", "rep_techniques"):
        ids = [t["id"] for t in signals[block]["types"]]
        assert len(ids) == len(set(ids))
        assert all(t.get("definition", "").strip() for t in signals[block]["types"])
    assert "buying_intent" in [t["id"] for t in signals["customer_signals"]["types"]]
    assert "social_proof" in [t["id"] for t in signals["rep_techniques"]["types"]]


def test_pitch_structures_do_not_privilege_one_shape(signals):
    ids = [t["id"] for t in signals["pitch_structures"]["types"]]
    for expected in ("consultative_discovery_first", "feature_led", "roi_led", "direct_offer"):
        assert expected in ids
    assert "insufficient_evidence" in signals["pitch_structures"]["fit_values"]


def test_context_factors_are_signals_not_rules(signals):
    factors = signals["context_factors"]
    assert "region" in factors["factors"]
    assert "insufficient_evidence" in factors["assessment_values"]
    # No per-region / per-product behavioural rule tables may ship here.
    assert "rules" not in factors
