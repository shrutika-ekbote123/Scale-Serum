"""The brand's local day.

`brands.timezone` is free text written by the onboarding form, in at least three
shapes: an IANA name ("Asia/Kolkata"), a label ("IST - UTC+5:30 (India Standard
Time)") and a bare "UTC". "Yesterday" is only right once that is a real zone,
so this module turns every shape into one and says how it got there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_ZONE = "Asia/Kolkata"

# Abbreviations seen in the onboarding labels. Only unambiguous ones: "IST" is
# India here, never Irish or Israel Standard Time.
_ABBREVIATIONS = {
    "IST": "Asia/Kolkata",
    "GMT": "UTC",
    "UTC": "UTC",
    "GST": "Asia/Dubai",
    "SGT": "Asia/Singapore",
}

_OFFSET = re.compile(r"(?:UTC|GMT)\s*([+\-−])\s*(\d{1,2})(?::?(\d{2}))?", re.IGNORECASE)


@dataclass(frozen=True)
class BrandZone:
    tz: tzinfo
    name: str          # what to report: an IANA name or "UTC+05:30"
    source: str        # iana | offset | abbreviation | default


def resolve(raw: Optional[str]) -> BrandZone:
    text = (raw or "").strip()
    if text:
        try:
            return BrandZone(ZoneInfo(text), text, "iana")
        except (ZoneInfoNotFoundError, ValueError):
            pass
        m = _OFFSET.search(text)
        if m:
            sign = -1 if m.group(1) in "-−" else 1
            minutes = int(m.group(2)) * 60 + int(m.group(3) or 0)
            if minutes == 0:
                return BrandZone(timezone.utc, "UTC", "offset")
            delta = timedelta(minutes=sign * minutes)
            hh, mm = divmod(minutes, 60)
            name = f"UTC{'+' if sign > 0 else '-'}{hh:02d}:{mm:02d}"
            return BrandZone(timezone(delta, name), name, "offset")
        head = re.split(r"[\s\-–—(]+", text.upper())[0]
        if head in _ABBREVIATIONS:
            zone = _ABBREVIATIONS[head]
            return BrandZone(ZoneInfo(zone), zone, "abbreviation")
    return BrandZone(ZoneInfo(DEFAULT_ZONE), DEFAULT_ZONE, "default")


@dataclass(frozen=True)
class Window:
    start: datetime    # UTC, inclusive
    end: datetime      # UTC, exclusive


def day_window(day: date, zone: BrandZone) -> Window:
    start = datetime.combine(day, time.min, tzinfo=zone.tz)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=zone.tz)
    return Window(start.astimezone(timezone.utc), end.astimezone(timezone.utc))


def span_window(first: date, last: date, zone: BrandZone) -> Window:
    """Local days first..last inclusive."""
    return Window(day_window(first, zone).start, day_window(last, zone).end)


def local_date(at: datetime, zone: BrandZone) -> date:
    return at.astimezone(zone.tz).date()


def local_now(now: datetime, zone: BrandZone) -> datetime:
    return now.astimezone(zone.tz)


def yesterday(now: datetime, zone: BrandZone) -> date:
    return local_now(now, zone).date() - timedelta(days=1)


def date_label(day: date) -> str:
    """Tuesday, 3 June 2026 - the prototype's heading format."""
    return f"{day.strftime('%A')}, {day.day} {day.strftime('%B %Y')}"


def short_label(day: date) -> str:
    """Mon, 2 Jun - the Past Briefings column."""
    return f"{day.strftime('%a')}, {day.day} {day.strftime('%b')}"
