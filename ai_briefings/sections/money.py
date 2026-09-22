"""Money actually received: non-backfilled payments from touchpoint_events.

Shared by Ads (blended ROAS), Leads (buyers) and All, so the revenue figure is
computed once and every tab quotes the same number.
"""
from __future__ import annotations

from ..timezones import Window
from .common import within


def payments_in(payments: list[dict], window: Window) -> list[dict]:
    return [p for p in payments if within(p["occurred_at"], window)]


def revenue(payments: list[dict], window: Window) -> float:
    return float(sum(p["value"] for p in payments_in(payments, window)))


def first_time_buyers(payments: list[dict], window: Window) -> list[dict]:
    """Payments that are a lead's first within the loaded history and fall in
    the window. A repeat purchase is revenue, not a new customer."""
    first: dict[str, dict] = {}
    for p in sorted(payments, key=lambda p: p["occurred_at"]):
        if p["lead_id"] and p["lead_id"] not in first:
            first[p["lead_id"]] = p
    return [p for p in first.values() if within(p["occurred_at"], window)]
