"""
Step 14 - the six scores. DETERMINISTIC, in Python, never in the model.

THE RULE THIS FILE EXISTS TO ENFORCE
    Every number in the report is computed here, from the stored measurements
    and vision_framework.json. Consequences, all deliberate:

      * The same measurements always produce the same score.
      * When management sets real weights, historical creatives are rescored
        from stored measurements with no ffmpeg, no model and no provider call.
      * Nothing written in the ad copy can move a number, because the thing
        that reads the ad never touches the arithmetic.

    If you find yourself reading an LLM response in this file, stop. That
    belongs in analyzer.py, and it returns an ordinal RATING which this file
    converts - never a number.

NO INVENTED BUSINESS VALUES
    Metric weights, bands and thresholds are null in config until management
    sets them. Placeholders are neutral - equal weighting - and every response
    reports which values were still unconfirmed.

A MISSING SIGNAL IS NOT A ZERO
    A metric whose inputs could not be measured returns null with a stated
    reason, and is excluded from the overall rather than dragging it down.
    "We could not measure this" and "this creative did badly" are different
    findings and the report must not conflate them.

BRAND MEMORY DOES NOT TRUST SALIENCY TO FIND THE LOGO
    Measured in the Step 8 bake-off: UNISAL scores 1.18 on brand marks and
    MSI-Net 0.49 - below chance. Models trained on photographs and film have no
    reason to prioritise a small high-contrast graphic. So brand DETECTION
    supplies where the logo is and when; saliency is asked only whether that
    moment was being attended. Asking the model to find logos would be asking a
    question it demonstrably cannot answer.
"""
from __future__ import annotations

import logging
from typing import Optional

from . import framework as fw

logger = logging.getLogger("vision_lab.scoring")

# Reasons a metric could not be scored. Reported, never silently zeroed.
NO_MEASUREMENTS = "no_measurements"
NO_TEXT_READ = "ocr_unavailable"
NO_BRAND_DETECTED = "no_brand_detected"
NO_CTA_DETECTED = "no_cta_detected"
NO_TIMELINE = "no_timeline"
AWAITING_LLM = "llm_rating_not_available"
# The key message was named, but it is carried by the VOICEOVER - there is no
# copy on screen at that moment for gaze to land on. Not a failure of the ad,
# and reported as its own answer rather than as a low score.
KEY_MESSAGE_NOT_ON_SCREEN = "key_message_not_on_screen"
KEY_MESSAGE_OUT_OF_RANGE = "key_message_outside_sampled_frames"


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> int:
    return int(round(max(low, min(high, value))))


def _mean(values: list) -> Optional[float]:
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else None


def _score(value: Optional[float], reason: Optional[str] = None,
           basis: str = "measured", **signals) -> dict:
    return {
        "score": None if value is None else _clamp(value),
        "basis": basis,
        "reason": reason,
        "signals": {k: v for k, v in signals.items() if v is not None},
    }


# --------------------------------------------------------------------------- metrics
def attention(measurements: list[dict], timeline: dict, cfg: dict) -> dict:
    """Does the opening capture gaze? Measured over the hook window only."""
    hook = fw.hook_seconds(cfg)
    early = [m for m in measurements if (m.get("t") or 0) <= hook]
    if not early:
        return _score(None, NO_MEASUREMENTS)

    concentration = _mean([(m.get("saliency") or {}).get("concentration")
                           for m in early])
    if concentration is None:
        return _score(None, NO_MEASUREMENTS)

    peaks = [(m.get("saliency") or {}).get("peaks") or [] for m in early]
    strongest = _mean([p[0]["share"] if p else None for p in peaks])
    index = _mean([p["attention"] for p in (timeline.get("points") or [])
                   if (p.get("t") or 0) <= hook])

    # Concentration carries it, the strongest single region confirms it, and the
    # index grounds it against the rest of the ad.
    parts = [v for v in (concentration / 0.45 * 100 if concentration else None,
                         (strongest or 0) / 0.5 * 100 if strongest else None,
                         index) if v is not None]
    return _score(_mean(parts), basis="measured",
                  hook_seconds=hook, mean_concentration=round(concentration, 4),
                  strongest_region_share=round(strongest, 4) if strongest else None)


def focus(measurements: list[dict], cfg: dict,
          key_message: Optional[dict] = None) -> dict:
    """How much gaze lands on the thing carrying the message.

    THE KEY MESSAGE IS WHAT MAKES THIS METRIC MEAN WHAT IT CLAIMS
        Without it, "the message" is every word OCR read - a legal line, a
        price and the headline all counted alike. analyzer.py names which
        element carries the central claim and when, so the measurement can be
        taken WHERE THAT CLAIM IS ON SCREEN rather than averaged over the whole
        creative. The model names the moment; the number is still measured here
        from the saliency mass at that moment.

        Without a key message the whole-creative average stands in, and `basis`
        says `measured_partial` so nobody reads it as the finished measure.
    """
    window, fallback_reason = _key_window(measurements, key_message)
    ratios, competing = [], []
    for record in (window or measurements):
        mass = record.get("mass") or {}
        regions = record.get("regions") or {}

        # ALL the text, against the area it occupies - not the largest single
        # word box. OCR returns words; the message is the block of them, and the
        # biggest single word covers a few percent of the frame, so measuring
        # mass on it alone reads as near-zero focus on an ad whose copy is in
        # fact being looked at.
        text_mass = mass.get("text")
        text_area = regions.get("text_area_share")
        if text_mass is not None and text_area and text_area > 0.002:
            # Chance-relative: 1.0 means gaze falls on the copy exactly in
            # proportion to how much of the frame it covers. An absolute share
            # cannot distinguish "the eye is drawn to the headline" from "the
            # headline is simply large", which is the whole question here.
            ratios.append(text_mass / text_area)

        peak_count = (record.get("saliency") or {}).get("peak_count")
        if peak_count is not None:
            competing.append(peak_count)

    if not ratios:
        return _score(None, NO_TEXT_READ, basis="measured_partial")

    concentration = _mean(ratios) or 0.0
    contest = _mean(competing) or 1.0
    # A frame pulling the eye in six directions is unfocused however much mass
    # sits on the copy.
    penalty = max(0.0, min(0.4, (contest - 2.0) * 0.06))

    # 1.0 = chance, 3.0 = strongly drawn. The bake-off measured UNISAL at 3.35
    # on hand-marked headline regions, so 3.0 is a demanding but reachable top.
    scaled = min(1.0, max(0.0, (concentration - 1.0) / 2.0)) * 100
    signals = {"copy_attention_vs_chance": round(concentration, 2),
               "mean_competing_peaks": round(contest, 2)}

    if window:
        return _score(scaled * (1 - penalty), basis="measured",
                      key_message_t=(key_message or {}).get("t"),
                      key_message_element=(key_message or {}).get("element"),
                      frames_at_key_message=len(window), **signals)

    return _score(scaled * (1 - penalty),
                  reason=fallback_reason, basis="measured_partial", **signals)


# How much of the creative counts as "at the key message". Two seconds either
# side of the named moment: long enough to survive a rounded timestamp, short
# enough that the number is about that element and not the whole ad.
KEY_MESSAGE_WINDOW_SECONDS = 2.0
MIN_KEY_FRAMES = 2

# A text region has to cover at least this much of the frame to be measurable at
# all. Below it, `text_mass / text_area` divides by a rounding error.
MIN_TEXT_AREA = 0.002


def _key_window(measurements: list[dict], key_message: Optional[dict]):
    """The frames where the key message is ON SCREEN, or None to use them all.

    A SPOKEN KEY MESSAGE IS NOT A FOCUS FAILURE, AND THIS IS WHY THE CHECK IS
    HERE RATHER THAN TRUSTED TO THE MODEL
        On a real ad the model named the central claim at 70.8 s - correctly,
        because that is where the voiceover states it. There is no copy on
        screen at that moment, so measuring "did gaze land on the message" there
        measured gaze against text that was not there, and Focus scored 0 on an
        ad whose copy was in fact being read.

        Gaze cannot land on a sentence nobody drew. So the window is used only
        when the frames in it actually carry measurable text; otherwise Focus
        falls back to the whole-creative figure and says why.

    A window of one frame is refused too: a single sample is not a measurement,
    and reporting basis="measured" off it would overstate what we know.
    """
    when = (key_message or {}).get("t")
    if when is None:
        return None, AWAITING_LLM
    try:
        when = float(when)
    except (TypeError, ValueError):
        return None, AWAITING_LLM

    # WHAT THE MODEL SAYS, THEN WHAT THE FRAMES SAY - both have to agree.
    #
    # A claim delivered by the voiceover has no visual element to measure, even
    # when other copy happens to be on screen at that second. Measuring gaze
    # against THAT text answers a question nobody asked: it is not the thing
    # carrying the message. On the real ad this scored Focus 0 twice - once
    # because there was no text at all, and once because there was the wrong
    # text.
    carrier = (key_message or {}).get("carrier")
    if carrier in ("voiceover", "visual"):
        return None, KEY_MESSAGE_NOT_ON_SCREEN

    window = [r for r in measurements
              if r.get("t") is not None
              and abs(float(r["t"]) - when) <= KEY_MESSAGE_WINDOW_SECONDS]
    if len(window) < MIN_KEY_FRAMES:
        return None, KEY_MESSAGE_OUT_OF_RANGE

    visible = [r for r in window
               if ((r.get("regions") or {}).get("text_area_share") or 0) > MIN_TEXT_AREA]
    if len(visible) < MIN_KEY_FRAMES:
        return None, KEY_MESSAGE_NOT_ON_SCREEN
    return visible, None


def cognitive_demand(measurements: list[dict], summary: dict, cfg: dict) -> dict:
    """Effort to understand. LOWER IS BETTER. Entirely measured, no LLM.

    Reading load is the core: words on screen against the time they are held,
    at the configured reading speed. That is the measurement the dead-zone
    finding is built on and the one the report quotes back to the user.
    """
    if not summary.get("frames_with_text_fraction"):
        return _score(None, NO_TEXT_READ)

    reading_speed = fw.reading_speed(cfg)
    overloaded = summary.get("overloaded_shots") or []
    shots = max(1, summary.get("shot_count") or 1)

    # Share of shots asking for more reading than they allow time for.
    overload_share = len(overloaded) / shots
    worst = max((s["seconds_needed"] / max(0.1, s["seconds_available"])
                 for s in overloaded), default=1.0)

    cut_rate = summary.get("cut_rate_per_minute") or 0.0
    text_share = _mean([(m.get("regions") or {}).get("text_area_share")
                        for m in measurements]) or 0.0
    competing = _mean([(m.get("saliency") or {}).get("peak_count")
                       for m in measurements]) or 1.0

    demand = (
        overload_share * 45          # slides that cannot be read in time
        + min(1.0, (worst - 1) / 2) * 25   # how badly the worst one overruns
        + min(1.0, cut_rate / 30) * 12     # cutting faster than the eye settles
        + min(1.0, text_share / 0.25) * 10
        + min(1.0, max(0.0, competing - 2) / 6) * 8
    )
    return _score(demand, basis="measured",
                  reading_speed_words_per_second=reading_speed,
                  overloaded_shots=len(overloaded),
                  worst_overrun_ratio=round(worst, 2),
                  cut_rate_per_minute=cut_rate)


def clarity(measurements: list[dict], summary: dict, cfg: dict,
            llm_rating: Optional[str] = None) -> dict:
    """Message and CTA unambiguous. The only metric an LLM contributes to - and
    it contributes a RATING, converted here, never a number."""
    cta = summary.get("cta") or {}
    if not cta.get("detected"):
        return _score(None, NO_CTA_DETECTED, basis="measured_partial")

    heights = []
    durations = 0
    for record in measurements:
        found = (record.get("regions") or {}).get("cta")
        if found:
            durations += 1
            box = found.get("box")
            if box:
                heights.append(box[3] - box[1])

    sample_fps = summary.get("sample_fps") or 2.0
    seconds_on_screen = durations / sample_fps
    legibility = (_mean(heights) or 0.0) / 0.05      # 5% of frame height = clear
    measured = (
        min(1.0, legibility) * 45
        + min(1.0, seconds_on_screen / 3.0) * 35
        + (0 if cta.get("weak") else 20)             # a weak CTA is a real defect
    )

    if llm_rating is None:
        return _score(measured, reason=AWAITING_LLM, basis="measured_partial",
                      cta_seconds_on_screen=round(seconds_on_screen, 2),
                      cta_weak=cta.get("weak"))

    scale, _mode = fw.rating_scale(cfg)
    rated = scale.get(llm_rating, 0.5) * 100
    weight = (cfg.get("metrics") and next(
        (m.get("llm_rating_weight") for m in cfg["metrics"]
         if m["id"] == "clarity"), None))
    blend = 0.5 if weight is None else float(weight)
    return _score(measured * (1 - blend) + rated * blend, basis="hybrid",
                  llm_rating=llm_rating, llm_blend=blend,
                  cta_seconds_on_screen=round(seconds_on_screen, 2))


def brand_memory(measurements: list[dict], summary: dict, timeline: dict,
                 cfg: dict) -> dict:
    """Likelihood the brand is remembered.

    DETECTION says where the logo is and when. Saliency is asked only whether
    that moment was being attended - see the module docstring for why.
    """
    brand = summary.get("brand") or {}
    if not brand.get("detected"):
        return _score(None, NO_BRAND_DETECTED,
                      exposure_seconds=0.0, detected=False)

    duration = summary.get("duration_seconds") or 0
    first_fraction = brand.get("first_appearance_fraction")
    exposure = brand.get("exposure_seconds") or 0.0

    # Early is better, and the penalty is not linear: a logo at 80% of runtime
    # is far worse than twice as bad as one at 40%.
    earliness = (100 * (1 - first_fraction) ** 1.5) if first_fraction is not None else 50

    # THE SIGNAL THAT MATTERS: was the brand on screen while attention was high?
    # A logo held during a dead zone is exposure without memory.
    points = timeline.get("points") or []
    values = sorted(p["attention"] for p in points if p.get("attention") is not None)
    overlap = None
    if values:
        top_quartile = values[int(len(values) * 0.75)]
        brand_frames = [m for m in measurements
                        if (m.get("regions") or {}).get("brand")]
        attended = 0
        for record in brand_frames:
            point = next((p for p in points if p.get("t") == record.get("t")), None)
            if point and (point.get("attention") or 0) >= top_quartile:
                attended += 1
        overlap = attended / len(brand_frames) if brand_frames else 0.0

    parts = [
        earliness * 0.35,
        min(1.0, exposure / max(1.0, duration * 0.15)) * 100 * 0.25,
        (overlap if overlap is not None else 0.3) * 100 * 0.40,
    ]
    return _score(sum(parts), basis="measured",
                  first_appearance_seconds=brand.get("first_appearance_seconds"),
                  first_appearance_fraction=first_fraction,
                  exposure_seconds=exposure,
                  attention_peak_overlap=round(overlap, 3) if overlap is not None else None)


def engagement(timeline: dict, summary: dict, cfg: dict) -> dict:
    """Pull to keep watching, from the SHAPE of the timeline."""
    points = [p for p in (timeline.get("points") or [])
              if p.get("engagement") is not None]
    if not points:
        return _score(None, NO_TIMELINE)

    values = [p["engagement"] for p in points]
    area = sum(values) / len(values)

    zones = timeline.get("weak_zones") or []
    dead_seconds = sum(z["seconds"] for z in zones)
    duration = summary.get("duration_seconds") or (len(values) / 2)
    dead_share = dead_seconds / duration if duration else 0.0

    hook = values[:max(1, len(values) // 10)]
    hook_strength = sum(hook) / len(hook)
    novelty = min(1.0, (summary.get("cut_rate_per_minute") or 0) / 12)

    score = (area * 0.45
             + hook_strength * 0.25
             + novelty * 100 * 0.15
             + max(0.0, 1 - dead_share * 3) * 100 * 0.15)
    return _score(score, basis="measured",
                  mean_engagement=round(area, 1),
                  hook_strength=round(hook_strength, 1),
                  dead_zone_seconds=round(dead_seconds, 2),
                  weak_zones=len(zones))


# --------------------------------------------------------------------------- overall
def score_all(measurements: list[dict], summary: dict, timeline: dict,
              cfg: Optional[dict] = None,
              llm_ratings: Optional[dict] = None,
              key_message: Optional[dict] = None) -> dict:
    """Every metric, plus the overall. The one entry point.

    `llm_ratings` and `key_message` are the ONLY things the interpretation pass
    contributes. Both are optional: with neither, five metrics score fully and
    two report `llm_rating_not_available` - the scorecard never waits on Gemini.
    """
    cfg = cfg or fw.load_framework()
    metrics = fw.metric_index(cfg)
    ratings = llm_ratings or {}

    computed = {
        "attention": attention(measurements, timeline, cfg),
        "focus": focus(measurements, cfg, key_message),
        "cognitive_demand": cognitive_demand(measurements, summary, cfg),
        "clarity": clarity(measurements, summary, cfg, ratings.get("clarity")),
        "brand_memory": brand_memory(measurements, summary, timeline, cfg),
        "engagement": engagement(timeline, summary, cfg),
    }

    scores = {}
    for metric_id, result in computed.items():
        definition = metrics[metric_id]
        scores[metric_id] = {
            "score": result["score"],
            "label": definition.get("label", ""),
            "direction": definition.get("direction", "higher_better"),
            "basis": result["basis"],
            "reason": result["reason"],
            "signals": result["signals"],
        }

    return {"scores": scores, "overall": overall(scores, cfg)}


def overall(scores: dict, cfg: dict) -> dict:
    """Weighted mean, with lower-is-better metrics inverted first.

    A metric that could not be measured is EXCLUDED from the denominator, not
    scored zero - and the response says how many were excluded, so a score
    computed from four metrics is never mistaken for one computed from six.
    """
    metrics = fw.metric_index(cfg)
    weighting = fw.weighting_mode(cfg)

    pairs = []
    missing = []
    for metric_id, entry in scores.items():
        if entry["score"] is None:
            missing.append(metric_id)
            continue
        value = entry["score"]
        if metrics[metric_id].get("direction") == "lower_better":
            value = 100 - value
        weight = metrics[metric_id].get("weight")
        pairs.append((value, 1.0 if weight is None else float(weight)))

    band, band_reason = fw.bands(cfg)
    if not pairs:
        # band_reason stays about BANDS. Why there is no score is a separate
        # question, answered by metrics_scored and metrics_missing - folding
        # both into one field would leave a caller unable to tell "no
        # thresholds configured" from "nothing could be measured".
        return {"score": None, "band": None, "band_reason": band_reason,
                "score_reason": "no metric could be scored",
                "weighting": weighting, "metrics_scored": 0,
                "metrics_missing": missing}

    total_weight = sum(w for _, w in pairs)
    value = sum(v * w for v, w in pairs) / total_weight

    return {
        "score": _clamp(value),
        "band": _band_for(value, band),
        "band_reason": band_reason,
        "weighting": weighting,
        "metrics_scored": len(pairs),
        "metrics_missing": missing,
    }


def _band_for(value: float, bands: Optional[list]) -> Optional[str]:
    """None until management configures bands. Script Lab's 90/70/50 ad-script
    thresholds carry no authority over creative attention and are not borrowed."""
    if not bands:
        return None
    for band in bands:
        if value >= float(band.get("min", 0)):
            return band.get("name")
    return None
