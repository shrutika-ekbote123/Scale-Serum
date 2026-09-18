"""Provider billing - turning recorded usage into an estimated bill.

    pricing.json   dated, versioned provider rates (no invented prices)
    pricing.py     load, validate, look up the rate in force on a date
    cost.py        pure cost of one run: Deepgram audio x channels, Gemini tokens
    summary.py     pure date-range bill over stored analyses

Shared on purpose, like `transcription/`: nothing here imports a feature
package, so Vision Lab or next-action can price their own provider calls later
without depending on the Sales Call Analyzer.

Everything is an ESTIMATE from configured rates. Unconfirmed rates, the merged-
channel billing assumption and the fixed exchange rate are named on every
breakdown and every bill.
"""
from .cost import (  # noqa: F401
    BASIS_CONFIG_RULE,
    BASIS_DEEPGRAM_METADATA,
    CHARGE_UNKNOWN,
    CHARGED,
    NOT_CHARGED,
    NOTE_BACKFILLED,
    NOTE_BILLED_CHANNELS_UNCONFIRMED,
    NOTE_DEEPGRAM_RATE_UNCONFIRMED,
    NOTE_FX_FIXED_RATE,
    NOTE_FX_NOT_CONFIGURED,
    NOTE_GEMINI_ALIAS_UNCONFIRMED,
    NOTE_GEMINI_RATE_UNCONFIRMED,
    REASON_NO_LLM_CALL,
    REASON_NO_TRANSCRIPTION,
    REASON_RATE_UNCONFIRMED_NULL,
    REASON_REUSED_TRANSCRIPT,
    REASON_SUPPLIED_TRANSCRIPT,
    REASON_TRANSCRIPTION_FAILED_UNKNOWN,
    REASON_USAGE_NOT_REPORTED,
    CostBreakdown,
    DeepgramCost,
    GeminiCost,
    combine,
    deepgram_cost,
    gemini_cost,
)
from .pricing import (  # noqa: F401
    REASON_RATE_NOT_CONFIGURED,
    PricingConfigError,
    as_utc_datetime,
    find_period,
    load_pricing,
    resolve_alias,
    resolve_rate,
    validate_pricing,
)
from .summary import backfill_cost, summarize  # noqa: F401
