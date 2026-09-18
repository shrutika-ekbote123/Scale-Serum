"""
Estimated provider cost for one run - PURE, no I/O.

THE RULES
    Deepgram   billed_seconds = audio_seconds x billed_channels
               usd = billed_seconds / 3600 x rate_per_hour
               Deepgram bills total processed audio: a 10-minute file processed
               as 2 channels is 20 minutes. Channels sent WITHOUT multichannel
               are merged into one stream, billed per the config rule.
    Gemini     usd = [(input - cached) x input_rate + cached x cached_rate
                      + (output + thinking) x output_rate] / 1,000,000
               Thinking tokens are billed at the output rate. prompt_token_count
               already INCLUDES cached tokens, hence the subtraction.

UNKNOWN IS NOT ZERO
    A charge we cannot price (unknown model, null rate, missing usage, a
    provider call that failed after it may have been billed) is `usd: null` with
    a stated reason. Only work that provably cost nothing - a reused or supplied
    transcript, no LLM call - is `usd: 0`. The overall total is null whenever any
    charged part is unpriced; `priced_usd_partial` still adds up what could be
    priced.

This module knows nothing about sales calls. The caller says whether a
transcription was charged; that keeps it reusable by Vision Lab and others.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .pricing import as_utc_datetime, resolve_alias, resolve_rate

# What the caller reports about a transcription step.
CHARGED = "charged"
NOT_CHARGED = "not_charged"
CHARGE_UNKNOWN = "unknown"

# Where billed_channels came from.
BASIS_DEEPGRAM_METADATA = "deepgram_metadata"
BASIS_CONFIG_RULE = "config_rule"

# Reasons on a single provider block.
REASON_RATE_UNCONFIRMED_NULL = "rate_unconfirmed_null"
REASON_USAGE_NOT_REPORTED = "usage_not_reported"
REASON_NO_LLM_CALL = "no_llm_call"
REASON_NO_TRANSCRIPTION = "no_transcription_call"
REASON_REUSED_TRANSCRIPT = "reused_stored_transcript_no_deepgram_charge"
REASON_SUPPLIED_TRANSCRIPT = "supplied_transcript_no_deepgram_charge"
REASON_TRANSCRIPTION_FAILED_UNKNOWN = "transcription_failed_charge_unknown"

# Notes on the whole breakdown: what is estimated or not yet confirmed.
NOTE_DEEPGRAM_RATE_UNCONFIRMED = "deepgram_rate_unconfirmed"
NOTE_BILLED_CHANNELS_UNCONFIRMED = "billed_channels_unconfirmed"
NOTE_GEMINI_RATE_UNCONFIRMED = "gemini_rate_unconfirmed"
NOTE_GEMINI_ALIAS_UNCONFIRMED = "gemini_alias_unconfirmed"
NOTE_FX_FIXED_RATE = "fx_fixed_rate"
NOTE_FX_NOT_CONFIGURED = "fx_rate_not_configured"
NOTE_BACKFILLED = "backfilled_from_stored_usage_assumed_1_channel"

_UNCONFIRMED_NOTES = {
    NOTE_DEEPGRAM_RATE_UNCONFIRMED, NOTE_BILLED_CHANNELS_UNCONFIRMED,
    NOTE_GEMINI_RATE_UNCONFIRMED, NOTE_GEMINI_ALIAS_UNCONFIRMED,
    NOTE_FX_NOT_CONFIGURED, NOTE_BACKFILLED,
}


class DeepgramCost(BaseModel):
    charged: bool = False
    model: Optional[str] = None
    language_variant: Optional[str] = None          # monolingual | multilingual
    rate_per_hour_usd: Optional[float] = None
    rate_confirmed: Optional[bool] = None
    rate_effective_from: Optional[str] = None
    audio_seconds: Optional[float] = None
    # Channels Deepgram says it processed (metadata.channels). NOT the channels
    # in the file: a stereo file sent without multichannel is merged and reports 1.
    channels_processed: Optional[int] = None
    multichannel_requested: bool = False
    billed_channels: Optional[int] = None
    billed_channels_basis: Optional[str] = None     # deepgram_metadata | config_rule
    billed_channels_confirmed: Optional[bool] = None
    billed_seconds: Optional[float] = None
    usd: Optional[float] = None
    reason: Optional[str] = None


class GeminiCost(BaseModel):
    charged: bool = False
    model_requested: Optional[str] = None
    model_priced: Optional[str] = None
    alias_confirmed: Optional[bool] = None
    attempts: int = 0
    input_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
    rate_input_per_1m_usd: Optional[float] = None
    rate_output_per_1m_usd: Optional[float] = None
    rate_cached_input_per_1m_usd: Optional[float] = None
    rate_confirmed: Optional[bool] = None
    rate_effective_from: Optional[str] = None
    usd: Optional[float] = None
    reason: Optional[str] = None


class CostBreakdown(BaseModel):
    deepgram: DeepgramCost = Field(default_factory=DeepgramCost)
    gemini: GeminiCost = Field(default_factory=GeminiCost)
    total_usd: Optional[float] = None
    priced_usd_partial: float = 0.0
    usd_to_inr: Optional[float] = None
    total_inr: Optional[float] = None
    priced_inr_partial: Optional[float] = None
    pricing_version: Optional[str] = None
    rates_as_of: Optional[str] = None               # the UTC day whose rates were used
    estimated: bool = True
    backfilled: bool = False
    confirmed: bool = False                         # every rate, rule and alias confirmed
    notes: list[str] = Field(default_factory=list)


def _round6(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 6)


def deepgram_cost(pricing: dict, *, outcome: str, reason: Optional[str] = None,
                  model: Optional[str], audio_seconds: Optional[float],
                  channels_processed: Optional[int] = None,
                  multichannel_requested: bool = False,
                  language_sent: Optional[str] = None, at: Any = None) -> DeepgramCost:
    """Cost of one transcription. `outcome` is CHARGED, NOT_CHARGED or CHARGE_UNKNOWN."""
    variant = "multilingual" if (language_sent or "").strip().lower() == "multi" else "monolingual"
    cost = DeepgramCost(model=model, language_variant=variant, audio_seconds=audio_seconds,
                        channels_processed=channels_processed,
                        multichannel_requested=bool(multichannel_requested))

    if outcome == NOT_CHARGED:
        cost.usd = 0.0
        cost.billed_seconds = 0.0
        cost.reason = reason or REASON_NO_TRANSCRIPTION
        return cost

    cost.charged = True
    if outcome == CHARGE_UNKNOWN:
        cost.reason = reason or REASON_TRANSCRIPTION_FAILED_UNKNOWN
        return cost

    # Deepgram bills every channel it processes. Its own count is used when it
    # reported one; the config rule covers merged audio when it did not. With
    # multichannel and no count, the number of channels is genuinely unknown.
    rule = pricing["deepgram"]["merged_audio_billed_channels"]
    if channels_processed:
        cost.billed_channels = int(channels_processed)
        cost.billed_channels_basis = BASIS_DEEPGRAM_METADATA
    elif not multichannel_requested:
        cost.billed_channels = int(rule["value"])
        cost.billed_channels_basis = BASIS_CONFIG_RULE
    cost.billed_channels_confirmed = bool(rule["confirmed"])

    period, why = resolve_rate(pricing, "deepgram", model, at)
    if period is not None:
        cost.rate_per_hour_usd = period["rates"].get(variant)
        cost.rate_confirmed = bool(period["confirmed"])
        cost.rate_effective_from = period["effective_from"]

    if audio_seconds is None or cost.billed_channels is None:
        cost.reason = REASON_USAGE_NOT_REPORTED
        return cost
    billed = float(audio_seconds) * cost.billed_channels
    cost.billed_seconds = round(billed, 3)
    if period is None:
        cost.reason = why
        return cost
    if cost.rate_per_hour_usd is None:
        cost.reason = REASON_RATE_UNCONFIRMED_NULL
        return cost
    cost.usd = _round6(billed / 3600.0 * cost.rate_per_hour_usd)
    return cost


def gemini_cost(pricing: dict, *, model_requested: Optional[str],
                model_version: Optional[str] = None, attempts: int = 0,
                input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
                thinking_tokens: Optional[int] = None, cached_tokens: Optional[int] = None,
                at: Any = None) -> GeminiCost:
    """Cost of every LLM attempt in one run, including rejected attempts."""
    tokens = (input_tokens, output_tokens, thinking_tokens, cached_tokens)
    cost = GeminiCost(model_requested=model_requested, attempts=int(attempts or 0),
                      input_tokens=input_tokens, output_tokens=output_tokens,
                      thinking_tokens=thinking_tokens, cached_tokens=cached_tokens)

    if not cost.attempts and all(t is None for t in tokens):
        cost.usd = 0.0
        cost.reason = REASON_NO_LLM_CALL
        return cost
    cost.charged = True

    # Prefer the version the provider reported; fall back to what was requested.
    models = pricing["gemini"]["models"]
    for candidate in (model_version, model_requested):
        name, alias_confirmed = resolve_alias(pricing, candidate, at)
        if name in models:
            cost.model_priced = name
            cost.alias_confirmed = alias_confirmed
            break

    period, why = resolve_rate(pricing, "gemini", cost.model_priced, at)
    if period is not None:
        rates = period["rates"]
        cost.rate_input_per_1m_usd = rates.get("input")
        cost.rate_output_per_1m_usd = rates.get("output")
        cost.rate_cached_input_per_1m_usd = rates.get("cached_input")
        cost.rate_confirmed = bool(period["confirmed"])
        cost.rate_effective_from = period["effective_from"]

    if all(t is None for t in tokens):
        cost.reason = REASON_USAGE_NOT_REPORTED
        return cost
    if period is None:
        cost.reason = why
        return cost
    if None in (cost.rate_input_per_1m_usd, cost.rate_output_per_1m_usd,
                cost.rate_cached_input_per_1m_usd):
        cost.reason = REASON_RATE_UNCONFIRMED_NULL
        return cost

    inp = int(input_tokens or 0)
    cached = int(cached_tokens or 0)
    uncached = max(inp - cached, 0)
    billed_output = int(output_tokens or 0) + int(thinking_tokens or 0)
    cost.usd = _round6((uncached * cost.rate_input_per_1m_usd
                        + cached * cost.rate_cached_input_per_1m_usd
                        + billed_output * cost.rate_output_per_1m_usd) / 1_000_000)
    return cost


def combine(pricing: dict, deepgram: DeepgramCost, gemini: GeminiCost, *,
            at: Any = None, backfilled: bool = False) -> CostBreakdown:
    """One run's breakdown, with every estimate and unconfirmed value named."""
    parts = (deepgram, gemini)
    unpriced = any(part.charged and part.usd is None for part in parts)
    partial = round(sum(part.usd or 0.0 for part in parts), 6)
    total = None if unpriced else partial

    fx_cfg = pricing.get("fx") or {}
    fx = fx_cfg.get("usd_to_inr")

    notes: list[str] = []
    for part in parts:
        if part.reason and part.reason not in notes:
            notes.append(part.reason)
    if deepgram.charged and deepgram.rate_confirmed is False:
        notes.append(NOTE_DEEPGRAM_RATE_UNCONFIRMED)
    if deepgram.charged and deepgram.billed_channels_confirmed is False:
        notes.append(NOTE_BILLED_CHANNELS_UNCONFIRMED)
    if gemini.charged and gemini.rate_confirmed is False:
        notes.append(NOTE_GEMINI_RATE_UNCONFIRMED)
    if gemini.charged and gemini.alias_confirmed is False:
        notes.append(NOTE_GEMINI_ALIAS_UNCONFIRMED)
    if fx is None:
        notes.append(NOTE_FX_NOT_CONFIGURED)
    elif fx_cfg.get("fixed_rate", True):
        notes.append(NOTE_FX_FIXED_RATE)
    if backfilled:
        notes.append(NOTE_BACKFILLED)

    moment = as_utc_datetime(at)
    return CostBreakdown(
        deepgram=deepgram,
        gemini=gemini,
        total_usd=total,
        priced_usd_partial=partial,
        usd_to_inr=fx,
        total_inr=round(total * fx, 2) if (total is not None and fx) else None,
        priced_inr_partial=round(partial * fx, 2) if fx else None,
        pricing_version=pricing.get("pricing_version"),
        rates_as_of=moment.date().isoformat() if moment else None,
        backfilled=backfilled,
        confirmed=total is not None and not any(n in _UNCONFIRMED_NOTES for n in notes),
        notes=notes,
    )
