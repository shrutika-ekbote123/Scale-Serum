"""Display strings, computed once in Python.

The writer is given these strings rather than raw floats, so "Rs 1.9L" in the
briefing is the same "1.9L" the number check allows - and the model never does
arithmetic or unit conversion itself.
"""
from __future__ import annotations

from typing import Optional

RUPEE = "₹"
UP = "▲"
DOWN = "▼"
TIMES = "×"


def money(amount: Optional[float], currency: Optional[str] = "INR") -> Optional[str]:
    """Indian units: Rs 950, Rs 90.3K, Rs 14.2L, Rs 2.7Cr."""
    if amount is None:
        return None
    symbol = RUPEE if (currency or "INR").upper() == "INR" else f"{(currency or '').upper()} "
    value = float(amount)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e7:
        body = f"{_trim(value / 1e7)}Cr"
    elif value >= 1e5:
        body = f"{_trim(value / 1e5)}L"
    elif value >= 1e3:
        body = f"{_trim(value / 1e3)}K"
    else:
        body = f"{value:,.0f}"
    return f"{sign}{symbol}{body}"


def _trim(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def ratio(value: Optional[float]) -> Optional[str]:
    """ROAS: 5.9x."""
    if value is None:
        return None
    return f"{value:.1f}{TIMES}"


def pct(value: Optional[float], digits: int = 1) -> Optional[str]:
    """A share already in 0..1 -> "12.8%"."""
    if value is None:
        return None
    text = f"{value * 100:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text}%"


def delta_count(current: Optional[float], previous: Optional[float]) -> Optional[str]:
    """(24, 18) -> "▲6"; no previous -> None; no change -> "no change"."""
    if current is None or previous is None:
        return None
    diff = round(current - previous)
    if diff == 0:
        return "no change"
    return f"{UP if diff > 0 else DOWN}{abs(diff):,}"


def delta_pct(current: Optional[float], previous: Optional[float]) -> Optional[str]:
    """Relative change: +18% / -6%. None when there is no base to compare with."""
    if current is None or not previous:
        return None
    change = (current - previous) / previous
    return f"{'+' if change >= 0 else '-'}{abs(change) * 100:.0f}%"


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or not den:
        return None
    return num / den


def campaign_name(raw: Optional[str], limit: int = 70) -> str:
    """Underscores to spaces, trimmed. The name is not rewritten: a media buyer
    has to be able to find the campaign in Ads Manager from what we print."""
    text = " ".join((raw or "Unnamed campaign").replace("_", " ").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
