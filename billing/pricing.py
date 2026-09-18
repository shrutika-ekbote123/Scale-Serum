"""
Provider rates - loading, validation and date-effective lookup.

NO INVENTED PRICES
    Every rate lives in pricing.json with the date it took effect and whether it
    is a confirmed published price. A placeholder is allowed, but it must say
    `confirmed: false`, and every bill built from it then says so. Changing a
    rate is a JSON edit plus a `pricing_version` bump, never a code change.

DATE-EFFECTIVE, ON PURPOSE
    A rate is chosen by the date the work RAN, not by today. Gemini's price
    doubles on 2027-01-01; a September 2026 call must keep its September 2026
    price when someone pulls the bill in February 2027. Past periods are never
    edited in place - a new period is added instead.

FAIL LOUDLY AT LOAD, NEVER AT RUN TIME
    A malformed file raises PricingConfigError when it is loaded, so a typo
    shows up at startup. Callers that price a live analysis catch errors and
    carry on without a cost block - a pricing bug must never fail an analysis.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timezone
from typing import Any, Optional

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRICING_PATH = os.path.join(PKG_DIR, "pricing.json")

DEEPGRAM_RATE_KEYS = ("monolingual", "multilingual")
GEMINI_RATE_KEYS = ("input", "output", "cached_input")

REASON_RATE_NOT_CONFIGURED = "rate_not_configured"

_CACHE: dict[str, dict] = {}
_LOCK = threading.Lock()


class PricingConfigError(ValueError):
    """pricing.json is unusable. Raised at load time so a typo is loud rather
    than a quietly wrong bill."""


# --------------------------------------------------------------------------- loading
def pricing_path() -> str:
    """BILLING_PRICING_PATH overrides the shipped file (tests, trial rates)."""
    return os.environ.get("BILLING_PRICING_PATH") or DEFAULT_PRICING_PATH


def load_pricing(force: bool = False) -> dict:
    """Load, validate and cache the pricing config. Cached per path."""
    path = pricing_path()
    if not force and path in _CACHE:
        return _CACHE[path]
    with _LOCK:
        if not force and path in _CACHE:
            return _CACHE[path]
        if not os.path.exists(path):
            raise PricingConfigError(f"pricing config is missing: {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except json.JSONDecodeError as err:
            raise PricingConfigError(f"pricing config is not valid JSON: {err}") from err
        validate_pricing(cfg)
        _CACHE[path] = cfg
        return cfg


# --------------------------------------------------------------------------- validation
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_date(value: Any, where: str) -> Optional[date]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PricingConfigError(f"{where}: dates must be YYYY-MM-DD strings, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as err:
        raise PricingConfigError(f"{where}: invalid date {value!r}") from err


def _check_periods(periods: Any, where: str, rate_keys: Optional[tuple]) -> None:
    """Periods must be well-formed and must not overlap. `effective_to` is
    exclusive, so one period ending on the day the next begins is not an overlap."""
    if not isinstance(periods, list) or not periods:
        raise PricingConfigError(f"{where}: periods must be a non-empty list")
    spans: list[tuple[date, Optional[date]]] = []
    for i, period in enumerate(periods):
        at = f"{where}.periods[{i}]"
        if not isinstance(period, dict):
            raise PricingConfigError(f"{at}: must be an object")
        start = _parse_date(period.get("effective_from"), f"{at}.effective_from")
        if start is None:
            raise PricingConfigError(f"{at}: effective_from is required")
        end = _parse_date(period.get("effective_to"), f"{at}.effective_to")
        if end is not None and end <= start:
            raise PricingConfigError(f"{at}: effective_to must be after effective_from")
        if not isinstance(period.get("confirmed"), bool):
            raise PricingConfigError(f"{at}: confirmed must be true or false")
        if rate_keys is not None:
            rates = period.get("rates")
            if not isinstance(rates, dict):
                raise PricingConfigError(f"{at}: rates must be an object")
            for key in rate_keys:
                if key not in rates:
                    raise PricingConfigError(f"{at}: rates.{key} is missing (use null if unknown)")
                value = rates[key]
                if value is not None and (not _is_number(value) or value < 0):
                    raise PricingConfigError(f"{at}: rates.{key} must be null or a number >= 0")
            if period["confirmed"] and any(rates[key] is None for key in rate_keys):
                raise PricingConfigError(f"{at}: a period with a null rate cannot be confirmed")
        spans.append((start, end))
    spans.sort(key=lambda span: span[0])
    for (_s1, end1), (start2, _e2) in zip(spans, spans[1:]):
        if end1 is None or end1 > start2:
            raise PricingConfigError(f"{where}: rate periods overlap")


def validate_pricing(cfg: Any) -> None:
    if not isinstance(cfg, dict):
        raise PricingConfigError("pricing config must be a JSON object")
    if not str(cfg.get("pricing_version") or "").strip():
        raise PricingConfigError("pricing_version is required")

    fx = cfg.get("fx") or {}
    rate = fx.get("usd_to_inr")
    if rate is not None and (not _is_number(rate) or rate <= 0):
        raise PricingConfigError("fx.usd_to_inr must be null or a positive number")

    deepgram = cfg.get("deepgram")
    if not isinstance(deepgram, dict):
        raise PricingConfigError("deepgram section is required")
    rule = deepgram.get("merged_audio_billed_channels") or {}
    value = rule.get("value")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PricingConfigError("deepgram.merged_audio_billed_channels.value must be an integer >= 1")
    if not isinstance(rule.get("confirmed"), bool):
        raise PricingConfigError("deepgram.merged_audio_billed_channels.confirmed must be true or false")
    models = deepgram.get("models")
    if not isinstance(models, dict) or not models:
        raise PricingConfigError("deepgram.models must be a non-empty object")
    for name, model in models.items():
        _check_periods((model or {}).get("periods"), f"deepgram.models.{name}", DEEPGRAM_RATE_KEYS)

    gemini = cfg.get("gemini")
    if not isinstance(gemini, dict):
        raise PricingConfigError("gemini section is required")
    gmodels = gemini.get("models")
    if not isinstance(gmodels, dict) or not gmodels:
        raise PricingConfigError("gemini.models must be a non-empty object")
    for name, model in gmodels.items():
        _check_periods((model or {}).get("periods"), f"gemini.models.{name}", GEMINI_RATE_KEYS)
    for alias, periods in (gemini.get("aliases") or {}).items():
        _check_periods(periods, f"gemini.aliases.{alias}", None)
        for i, period in enumerate(periods):
            if period.get("resolves_to") not in gmodels:
                raise PricingConfigError(
                    f"gemini.aliases.{alias}.periods[{i}]: resolves_to "
                    f"{period.get('resolves_to')!r} is not a configured model")


# --------------------------------------------------------------------------- lookup
def as_utc_datetime(value: Any) -> Optional[datetime]:
    """A datetime (naive means UTC, which is how MongoDB returns them), an ISO
    string (how processing is stored), a date, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return None


def _rate_day(at: Any) -> date:
    """The UTC day whose rates apply. Falls back to today only when the work
    carries no timestamp at all."""
    moment = as_utc_datetime(at) or datetime.now(timezone.utc)
    return moment.date()


def find_period(periods: list[dict], at: Any) -> Optional[dict]:
    day = _rate_day(at)
    for period in periods:
        start = date.fromisoformat(period["effective_from"])
        end = date.fromisoformat(period["effective_to"]) if period.get("effective_to") else None
        if start <= day and (end is None or day < end):
            return period
    return None


def resolve_alias(cfg: dict, model: Optional[str], at: Any) -> tuple[Optional[str], Optional[bool]]:
    """(model it is priced as, alias confirmed). A name that is not an alias
    resolves to itself with `None` for confirmed - there was nothing to confirm."""
    if not model:
        return None, None
    periods = ((cfg.get("gemini") or {}).get("aliases") or {}).get(model)
    if not periods:
        return model, None
    period = find_period(periods, at)
    if period is None:
        return model, None
    return period["resolves_to"], bool(period["confirmed"])


def resolve_rate(cfg: dict, provider: str, model: Optional[str],
                 at: Any) -> tuple[Optional[dict], Optional[str]]:
    """(rate period, reason). A model with no period in force is unpriced, never
    priced at zero."""
    models = (cfg.get(provider) or {}).get("models") or {}
    if not model or model not in models:
        return None, REASON_RATE_NOT_CONFIGURED
    period = find_period(models[model]["periods"], at)
    if period is None:
        return None, REASON_RATE_NOT_CONFIGURED
    return period, None
