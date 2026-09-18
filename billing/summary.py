"""
Date-range bill - adding up stored cost blocks. PURE, no I/O.

WHY PYTHON AND NOT A MONGO AGGREGATE
    Volume is small (hundreds to low thousands of analyses a month), analyses
    that predate cost tracking must be re-priced from their stored usage anyway,
    and one code path is easier to trust than two that must agree.

WHAT IS COUNTED
    Every run's spend, including earlier runs of the same analysis_id
    (`processing.cost_prior_runs`) and failed analyses - a failed call still
    cost money. Analyses still in progress are counted but not priced.
"""
from __future__ import annotations

from typing import Iterable, Optional

from .cost import (
    CHARGED,
    NOT_CHARGED,
    NOTE_BILLED_CHANNELS_UNCONFIRMED,
    NOTE_FX_FIXED_RATE,
    NOTE_GEMINI_ALIAS_UNCONFIRMED,
    CostBreakdown,
    combine,
    deepgram_cost,
    gemini_cost,
)

TERMINAL_STATUSES = ("completed", "failed", "skipped")
_TOKEN_KEYS = ("llm_input_tokens", "llm_output_tokens", "llm_thinking_tokens", "llm_cached_tokens")


def backfill_cost(doc: dict, pricing: dict) -> Optional[CostBreakdown]:
    """Price an analysis that predates cost tracking from the usage it stored.

    Old analyses recorded neither channel count nor language, so they are
    assumed to be 1 billed channel at the monolingual rate, and marked
    `backfilled` so the bill says so.
    """
    processing = doc.get("processing") or {}
    audio = processing.get("audio_seconds_submitted")
    tokens_present = any(processing.get(key) is not None for key in _TOKEN_KEYS)
    if audio is None and not tokens_present and not processing.get("llm_attempts"):
        return None
    at = processing.get("started_at") or doc.get("created_at")
    deepgram = deepgram_cost(
        pricing, outcome=CHARGED if audio is not None else NOT_CHARGED,
        model=processing.get("transcription_model"), audio_seconds=audio,
        channels_processed=processing.get("channels_processed"),
        language_sent=processing.get("transcription_language_sent"), at=at)
    gemini = gemini_cost(
        pricing, model_requested=processing.get("llm_model"),
        model_version=processing.get("llm_model_version"),
        attempts=processing.get("llm_attempts") or (1 if tokens_present else 0),
        input_tokens=processing.get("llm_input_tokens"),
        output_tokens=processing.get("llm_output_tokens"),
        thinking_tokens=processing.get("llm_thinking_tokens"),
        cached_tokens=processing.get("llm_cached_tokens"), at=at)
    return combine(pricing, deepgram, gemini, at=at, backfilled=True)


def _costs_for(doc: dict, pricing: dict) -> tuple[list[CostBreakdown], bool]:
    """(every run's cost for this analysis, whether the latest was backfilled)."""
    processing = doc.get("processing") or {}
    costs: list[CostBreakdown] = []
    for raw in processing.get("cost_prior_runs") or []:
        try:
            costs.append(CostBreakdown(**raw))
        except Exception:  # noqa: BLE001 - a malformed old block is skipped, not fatal
            continue
    if processing.get("cost"):
        try:
            costs.append(CostBreakdown(**processing["cost"]))
            return costs, False
        except Exception:  # noqa: BLE001 - fall through to re-pricing from usage
            pass
    backfilled = backfill_cost(doc, pricing)
    if backfilled is not None:
        costs.append(backfilled)
        return costs, True
    return costs, False


def _bucket(buckets: dict, key: tuple, **fields) -> dict:
    if key not in buckets:
        buckets[key] = {**fields, "runs": 0, "priced_usd_partial": 0.0, "unpriced": 0}
    return buckets[key]


def _finish(bucket: dict) -> dict:
    out = dict(bucket)
    out["priced_usd_partial"] = round(out["priced_usd_partial"], 6)
    out["usd"] = None if out["unpriced"] else out["priced_usd_partial"]
    return out


def summarize(docs: Iterable[dict], pricing: dict) -> dict:
    counts = {"analyses": 0, "completed": 0, "failed": 0, "skipped": 0, "in_progress": 0,
              "runs_priced": 0, "zero_deepgram_cost": 0, "unpriced": 0,
              "backfilled": 0, "no_cost_data": 0}
    deepgram_total = {"provider": "deepgram", "runs_charged": 0, "audio_seconds": 0.0,
                      "billed_seconds": 0.0, "priced_usd_partial": 0.0, "unpriced": 0}
    gemini_total = {"provider": "gemini", "runs_charged": 0, "attempts": 0, "input_tokens": 0,
                    "cached_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
                    "priced_usd_partial": 0.0, "unpriced": 0}
    by_model: dict = {}
    by_channels: dict = {}
    notes_seen: dict[str, int] = {}
    versions: set[str] = set()
    fx_seen: set[float] = set()
    total_partial = 0.0
    inr_partial = 0.0
    inr_complete = True

    for doc in docs:
        status = doc.get("status")
        if status not in TERMINAL_STATUSES:
            counts["in_progress"] += 1
            continue
        counts["analyses"] += 1
        counts[status] += 1

        costs, backfilled = _costs_for(doc, pricing)
        if backfilled:
            counts["backfilled"] += 1
        if not costs:
            counts["no_cost_data"] += 1
            continue

        for cost in costs:
            counts["runs_priced"] += 1
            if cost.pricing_version:
                versions.add(cost.pricing_version)
            if cost.usd_to_inr:
                fx_seen.add(cost.usd_to_inr)
            for note in cost.notes:
                notes_seen[note] = notes_seen.get(note, 0) + 1
            if cost.total_usd is None:
                counts["unpriced"] += 1
            total_partial += cost.priced_usd_partial
            if cost.priced_inr_partial is None:
                inr_complete = False
            else:
                inr_partial += cost.priced_inr_partial

            dg = cost.deepgram
            if not dg.charged:
                counts["zero_deepgram_cost"] += 1
            else:
                deepgram_total["runs_charged"] += 1
                deepgram_total["audio_seconds"] += dg.audio_seconds or 0.0
                deepgram_total["billed_seconds"] += dg.billed_seconds or 0.0
                deepgram_total["priced_usd_partial"] += dg.usd or 0.0
                deepgram_total["unpriced"] += int(dg.usd is None)
                row = _bucket(by_model, ("deepgram", dg.model, dg.language_variant),
                              provider="deepgram", model=dg.model,
                              language_variant=dg.language_variant, billed_seconds=0.0,
                              rate_per_hour_usd=dg.rate_per_hour_usd,
                              rate_confirmed=dg.rate_confirmed)
                row["runs"] += 1
                row["billed_seconds"] += dg.billed_seconds or 0.0
                row["priced_usd_partial"] += dg.usd or 0.0
                row["unpriced"] += int(dg.usd is None)
                if dg.rate_confirmed is False:
                    row["rate_confirmed"] = False
                channel_row = _bucket(by_channels, (dg.billed_channels, dg.billed_channels_basis),
                                      billed_channels=dg.billed_channels,
                                      basis=dg.billed_channels_basis,
                                      audio_seconds=0.0, billed_seconds=0.0)
                channel_row["runs"] += 1
                channel_row["audio_seconds"] += dg.audio_seconds or 0.0
                channel_row["billed_seconds"] += dg.billed_seconds or 0.0
                channel_row["priced_usd_partial"] += dg.usd or 0.0
                channel_row["unpriced"] += int(dg.usd is None)

            gm = cost.gemini
            if gm.charged:
                gemini_total["runs_charged"] += 1
                gemini_total["attempts"] += gm.attempts
                for key in ("input_tokens", "cached_tokens", "output_tokens", "thinking_tokens"):
                    gemini_total[key] += getattr(gm, key) or 0
                gemini_total["priced_usd_partial"] += gm.usd or 0.0
                gemini_total["unpriced"] += int(gm.usd is None)
                model = gm.model_priced or gm.model_requested
                row = _bucket(by_model, ("gemini", model, None), provider="gemini",
                              model=model, input_tokens=0, output_tokens=0,
                              thinking_tokens=0, rate_confirmed=gm.rate_confirmed)
                row["runs"] += 1
                row["input_tokens"] += gm.input_tokens or 0
                row["output_tokens"] += gm.output_tokens or 0
                row["thinking_tokens"] += gm.thinking_tokens or 0
                row["priced_usd_partial"] += gm.usd or 0.0
                row["unpriced"] += int(gm.usd is None)
                if gm.rate_confirmed is False:
                    row["rate_confirmed"] = False

    any_unpriced = counts["unpriced"] > 0
    total_partial = round(total_partial, 6)
    for block in (deepgram_total, gemini_total):
        block["priced_usd_partial"] = round(block["priced_usd_partial"], 6)
        block["usd"] = None if block["unpriced"] else block["priced_usd_partial"]
    deepgram_total["audio_seconds"] = round(deepgram_total["audio_seconds"], 3)
    deepgram_total["billed_seconds"] = round(deepgram_total["billed_seconds"], 3)
    deepgram_total["billed_minutes"] = round(deepgram_total["billed_seconds"] / 60, 2)
    deepgram_total["billed_hours"] = round(deepgram_total["billed_seconds"] / 3600, 4)
    # cached tokens are part of input and are not counted twice; thinking is
    # counted because Gemini bills it at the output rate.
    gemini_total["total_tokens"] = (gemini_total["input_tokens"]
                                    + gemini_total["output_tokens"]
                                    + gemini_total["thinking_tokens"])

    model_rows = [_finish(row) for row in by_model.values()]
    channel_rows = sorted((_finish(row) for row in by_channels.values()),
                          key=lambda r: (r["billed_channels"] or 0, r["basis"] or ""))
    for row in model_rows + channel_rows:
        for key in ("audio_seconds", "billed_seconds"):
            if key in row:
                row[key] = round(row[key], 3)

    return {
        "currency": "USD",
        "pricing_version_current": pricing.get("pricing_version"),
        "pricing_versions_seen": sorted(versions),
        "usd_to_inr_seen": sorted(fx_seen),
        "totals": {
            "usd": None if any_unpriced else total_partial,
            "priced_usd_partial": total_partial,
            "inr": (round(inr_partial, 2) if (inr_complete and not any_unpriced and fx_seen)
                    else None),
            "priced_inr_partial": round(inr_partial, 2) if fx_seen else None,
            "complete": not any_unpriced and counts["no_cost_data"] == 0,
        },
        "tokens": {
            "input": gemini_total["input_tokens"],
            "output": gemini_total["output_tokens"],
            "thinking": gemini_total["thinking_tokens"],
            "cached": gemini_total["cached_tokens"],
            "total": gemini_total["total_tokens"],
        },
        "per_analysis": _per_analysis(counts, total_partial, gemini_total, deepgram_total),
        "by_provider": [deepgram_total, gemini_total],
        "by_model": model_rows,
        "by_billed_channels": channel_rows,
        "counts": counts,
        "notes_seen": notes_seen,
        "disclaimers": _disclaimers(notes_seen, model_rows, counts),
    }


def _per_analysis(counts: dict, total_usd: float, gemini: dict, deepgram: dict) -> dict:
    """Averages, for answering "what does a call cost us?" without a calculator.

    Divided by analyses rather than runs: a retry is part of what one analysis
    cost, not a second analysis.
    """
    analyses = counts["analyses"]
    if not analyses:
        return {"analyses": 0, "usd": None, "inr": None, "tokens": None,
                "billed_minutes": None}
    return {
        "analyses": analyses,
        "usd": round(total_usd / analyses, 6),
        "tokens": round(gemini["total_tokens"] / analyses, 1),
        "billed_minutes": round(deepgram["billed_seconds"] / 60 / analyses, 2),
    }


def _disclaimers(notes: dict, model_rows: list[dict], counts: dict) -> list[str]:
    out = ["Estimated from the rates in billing/pricing.json and the usage each analysis "
           "recorded. This is not a provider invoice."]
    placeholders = sorted({row["model"] for row in model_rows
                           if row.get("rate_confirmed") is False and row.get("model")})
    if placeholders:
        out.append(f"Rates for {', '.join(placeholders)} are placeholders, not confirmed prices.")
    if notes.get(NOTE_BILLED_CHANNELS_UNCONFIRMED):
        out.append("Billed channels are Deepgram's own processed-channel count (or 1 for merged "
                   "audio when it reported none). Not yet checked against Deepgram's usage export.")
    if notes.get(NOTE_GEMINI_ALIAS_UNCONFIRMED):
        out.append("gemini-flash-latest is assumed to resolve to the model it is priced as; "
                   "Google can change that at any time.")
    if notes.get(NOTE_FX_FIXED_RATE):
        out.append("Rupee amounts use a fixed configured exchange rate, not a live one.")
    if counts["backfilled"]:
        out.append(f"{counts['backfilled']} analyses predate cost tracking; their cost was "
                   "recomputed from stored usage, assuming 1 billed channel.")
    if counts["unpriced"]:
        out.append(f"{counts['unpriced']} runs could not be fully priced, so totals.usd is null; "
                   "priced_usd_partial covers everything that could be priced.")
    if counts["no_cost_data"]:
        out.append(f"{counts['no_cost_data']} analyses recorded no usage and are not included.")
    if counts["in_progress"]:
        out.append(f"{counts['in_progress']} analyses were still in progress and are not included.")
    return out
