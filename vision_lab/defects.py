"""
Step 14b - the defect list. What is measurably wrong with this creative.

WHY THIS FILE IS THE HINGE OF THE WHOLE DESIGN
    Milestone D hands these to Gemini and asks it to write the prose a client
    reads. The rule is that the model writes ABOUT defects Python found; it
    never nominates its own.

    Without this list, the obvious shortcut is to ask the model what is wrong
    with the ad - and then every recommendation is an opinion. With it, every
    recommendation is anchored to a timestamp and a counted number that
    evidence.py can verify. "34 words on screen for 2.5 s" is checkable. "The
    pacing feels rushed" is not.

    So nothing here is a judgement. Each defect is a rule over the measurement
    record, carrying the numbers that produced it.

SCORES_IMPACTED IS DERIVED, NOT ASSERTED
    Each defect names the metrics whose inputs it is built from - the same
    signals scoring.py consumed. It is not the model's view of what a problem
    affects, and it is not a guess.

WHAT IS DELIBERATELY ABSENT
    `unclosed_promise` - whether the close restates the opening claim - needs
    the transcript, which arrives in Milestone D. It is listed in
    VISION_LAB_PLAN.md and is not implemented here rather than being faked from
    on-screen text alone.
"""
from __future__ import annotations

from typing import Optional

from . import framework as fw

# Defect ids. Stable - the UI and the prompt may branch on these.
DEAD_ZONE = "dead_zone"
OVERLOADED_SLIDE = "overloaded_slide"
OVERHELD_SLIDE = "overheld_slide"
LATE_BRAND = "late_brand_appearance"
BRAND_UNSEEN = "brand_never_attended"
BRAND_ABSENT = "brand_not_detected"
WEAK_CTA = "weak_cta"
NO_CTA = "cta_not_detected"
COMPETING_PEAKS = "competing_peaks"
FLAT_HOOK = "flat_hook"

SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"

# How late a brand may first appear before it is a finding, as a fraction of
# runtime. A business decision that has not been made - the plan lists "brand
# appearance targets" as unconfirmed - so this is a stated placeholder.
LATE_BRAND_FRACTION = 0.35

# Mean simultaneous attention centres above which a frame is contested.
COMPETING_PEAK_THRESHOLD = 3.5

# Floors for reporting an overloaded slide at all. Arithmetic will find a
# five-word caption half a second short; that is true and not a finding, and
# listing it buries the slide that genuinely cannot be read.
MIN_SHORTFALL_SECONDS = 1.0
MIN_OVERLOAD_WORDS = 8


def _defect(defect_id: str, severity: str, scores_impacted: list[str],
            measured: dict, t_start: Optional[float] = None,
            t_end: Optional[float] = None, note: str = "") -> dict:
    return {
        "defect_id": defect_id,
        "severity": severity,
        "t_start": t_start,
        "t_end": t_end,
        "scores_impacted": scores_impacted,
        "measured": {k: v for k, v in measured.items() if v is not None},
        "note": note,
    }


def detect(measurements: list[dict], summary: dict,
           timeline: Optional[dict], cfg: Optional[dict] = None) -> list[dict]:
    """Every measurable defect, ranked worst first.

    Ranking is by severity then by how much of the runtime the defect covers -
    a two-second dead zone matters more than a half-second one, and the list is
    ordered so a caller taking the top three takes the three that matter.
    """
    cfg = cfg or fw.load_framework()
    found: list[dict] = []
    timeline = timeline or {}

    found += _dead_zones(timeline, summary)
    found += _reading_load(summary, cfg)
    found += _brand(measurements, summary, timeline)
    found += _cta(summary)
    found += _contested_frames(measurements)
    found += _hook(timeline, cfg)

    order = {SEVERITY_HIGH: 0, SEVERITY_MEDIUM: 1, SEVERITY_LOW: 2}
    return sorted(found, key=lambda d: (order.get(d["severity"], 3),
                                        -(d["measured"].get("seconds") or 0)))


# --------------------------------------------------------------------------- rules
def _dead_zones(timeline: dict, summary: dict) -> list[dict]:
    """Stretches where the attention index stays low for long enough to matter."""
    out = []
    for zone in timeline.get("weak_zones") or []:
        seconds = zone["seconds"]
        severity = (SEVERITY_HIGH if seconds >= 2.0 else
                    SEVERITY_MEDIUM if seconds >= 1.0 else SEVERITY_LOW)
        out.append(_defect(
            DEAD_ZONE, severity,
            ["engagement", "attention"],
            {"seconds": seconds, "min_attention": zone["min_attention"],
             "mean_attention": zone["mean_attention"]},
            t_start=zone["t_start"], t_end=zone["t_end"],
            note=(f"attention falls to {zone['min_attention']:.0f} and stays "
                  f"below the weak threshold for {seconds:.1f}s")))
    return out


def _reading_load(summary: dict, cfg: dict) -> list[dict]:
    """Copy that cannot be read in the time it is held, and copy held long
    after it has been."""
    speed = fw.reading_speed(cfg)
    out = []
    for shot in summary.get("overloaded_shots") or []:
        needed = shot["seconds_needed"]
        available = shot["seconds_available"]
        shortfall = needed - available
        overrun = needed / max(0.1, available)

        # A FLOOR, because arithmetic finds things nobody would call a problem.
        # A five-word caption half a second short is technically overloaded and
        # is not worth a client's attention; reporting it buries the slide that
        # genuinely is. Both the absolute shortfall and the word count have to
        # clear a bar.
        if shortfall < MIN_SHORTFALL_SECONDS or shot["words"] < MIN_OVERLOAD_WORDS:
            continue

        # Severity weighs the SHORTFALL as well as the ratio. Sixty-one words
        # 1.3s short is a worse experience than four words 0.3s short, even
        # though the ratio says otherwise - the viewer has far more to lose.
        severity = (SEVERITY_HIGH if (overrun >= 2.0 or shortfall >= 4.0) else
                    SEVERITY_MEDIUM if (overrun >= 1.25 or shortfall >= 1.5)
                    else SEVERITY_LOW)
        out.append(_defect(
            OVERLOADED_SLIDE, severity,
            ["cognitive_demand", "focus"],
            {"words": shot["words"], "seconds_available": available,
             "seconds_needed": needed,
             "seconds_short": round(shortfall, 2),
             "reading_speed_words_per_second": speed,
             "seconds": available},
            t_start=shot.get("t_start"), t_end=shot.get("t_end"),
            note=(f"{shot['words']} words need {needed:.1f}s at {speed:g} "
                  f"words/second but are held for {available:.1f}s - "
                  f"{shortfall:.1f}s short")))
    return out


def _brand(measurements: list[dict], summary: dict,
           timeline: dict) -> list[dict]:
    """Brand findings come from DETECTION, not from saliency.

    The Step 8 bake-off measured every candidate model at or below chance on
    brand marks, so the model is never asked where the logo is - only whether
    the moment it appeared was being attended.
    """
    brand = summary.get("brand") or {}
    if not brand.get("detected"):
        return [_defect(BRAND_ABSENT, SEVERITY_HIGH, ["brand_memory"],
                        {"exposure_seconds": 0.0},
                        note=("no brand mark or brand name was found in the "
                              "creative - supply a wordmark if one should be "
                              "detected"))]

    out = []
    fraction = brand.get("first_appearance_fraction")
    if fraction is not None and fraction > LATE_BRAND_FRACTION:
        severity = (SEVERITY_HIGH if fraction > 0.6 else SEVERITY_MEDIUM)
        out.append(_defect(
            LATE_BRAND, severity, ["brand_memory"],
            {"first_appearance_seconds": brand.get("first_appearance_seconds"),
             "first_appearance_fraction": round(fraction, 3),
             "threshold_fraction": LATE_BRAND_FRACTION,
             "threshold_confirmed": False},
            t_start=brand.get("first_appearance_seconds"),
            note=(f"the brand first appears "
                  f"{fraction * 100:.0f}% of the way through")))

    # Exposure without memory: on screen, but never while anyone is looking.
    points = timeline.get("points") or []
    values = sorted(p["attention"] for p in points
                    if p.get("attention") is not None)
    if values:
        top_quartile = values[int(len(values) * 0.75)]
        brand_frames = [m for m in measurements
                        if (m.get("regions") or {}).get("brand")]
        attended = sum(
            1 for m in brand_frames
            if next((p.get("attention") or 0) for p in points
                    if p.get("t") == m.get("t")) >= top_quartile)
        overlap = attended / len(brand_frames) if brand_frames else 0.0
        if brand_frames and overlap < 0.15:
            out.append(_defect(
                BRAND_UNSEEN, SEVERITY_MEDIUM, ["brand_memory"],
                {"exposure_seconds": brand.get("exposure_seconds"),
                 "attention_peak_overlap": round(overlap, 3),
                 "seconds": brand.get("exposure_seconds")},
                note=(f"the brand is on screen for "
                      f"{brand.get('exposure_seconds', 0):.1f}s but only "
                      f"{overlap * 100:.0f}% of that overlaps a high-attention "
                      f"moment - exposure without memory")))
    return out


def _cta(summary: dict) -> list[dict]:
    cta = summary.get("cta") or {}
    if not cta.get("detected"):
        return [_defect(NO_CTA, SEVERITY_HIGH, ["clarity"], {},
                        note="no call to action was found on screen")]
    if cta.get("weak"):
        return [_defect(WEAK_CTA, SEVERITY_MEDIUM, ["clarity"], {},
                        note=("the call to action uses weak wording - the same "
                              "list Script Lab flags"))]
    return []


def _contested_frames(measurements: list[dict]) -> list[dict]:
    """Frames pulling the eye several ways at once."""
    counts = [(m.get("saliency") or {}).get("peak_count")
              for m in measurements]
    usable = [c for c in counts if c is not None]
    if not usable:
        return []
    mean = sum(usable) / len(usable)
    if mean < COMPETING_PEAK_THRESHOLD:
        return []
    return [_defect(
        COMPETING_PEAKS, SEVERITY_MEDIUM, ["focus", "cognitive_demand"],
        {"mean_competing_peaks": round(mean, 2),
         "threshold": COMPETING_PEAK_THRESHOLD},
        note=(f"an average of {mean:.1f} attention centres compete per frame - "
              f"the eye is being asked to do several things at once"))]


def _hook(timeline: dict, cfg: dict) -> list[dict]:
    """A flat opening. The first seconds decide whether the rest is watched."""
    hook_seconds = fw.hook_seconds(cfg)
    early = [p["attention"] for p in (timeline.get("points") or [])
             if p.get("attention") is not None
             and (p.get("t") or 0) <= hook_seconds]
    if not early:
        return []
    mean = sum(early) / len(early)
    rest = [p["attention"] for p in (timeline.get("points") or [])
            if p.get("attention") is not None
            and (p.get("t") or 0) > hook_seconds]
    if not rest:
        return []
    baseline = sum(rest) / len(rest)
    if mean >= baseline:
        return []
    return [_defect(
        FLAT_HOOK, SEVERITY_HIGH if mean < baseline - 15 else SEVERITY_MEDIUM,
        ["attention", "engagement"],
        {"hook_attention": round(mean, 1),
         "rest_of_ad_attention": round(baseline, 1),
         "hook_seconds": hook_seconds, "seconds": hook_seconds},
        t_start=0.0, t_end=hook_seconds,
        note=(f"the first {hook_seconds:g}s score {mean:.0f} against "
              f"{baseline:.0f} for the rest of the ad - the opening is the "
              f"weaker part"))]
