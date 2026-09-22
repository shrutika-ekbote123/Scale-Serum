"""Shared pieces for the section builders.

A builder is a pure function: raw rows in (from data.load_day), a SectionResult
out. No I/O, no clock of its own, no model - so each can be tested on a
hand-built day and gives the same answer every time.

SectionResult
    section      sales | ads | whatsapp | leads | all
    available    False when the brand has no such data at all (no WhatsApp
                 account, no calls ever). The tab then says why.
    reason       machine code for an unavailable section
    message      the sentence the tab shows instead of a briefing
    facts        numbers, raw and formatted, that the writer may quote
    kpis         the headline numbers, each {value, display, previous, delta}
    watch        [{type, severity, entity, text}] most severe first
    template     deterministic wording: {summary, top, watch, blocks}
                 used when the model is unavailable or fails validation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

from ..timezones import BrandZone, Window, day_window, span_window


@dataclass
class DayContext:
    day: date
    zone: BrandZone
    cfg: dict
    currency: str
    now: datetime
    analyses: dict = field(default_factory=dict)   # analysis_id -> report dict
    hot: Optional[dict] = None                     # hot_leads.score() output

    @property
    def d(self) -> Window:
        return day_window(self.day, self.zone)

    def day_n(self, n: int) -> Window:
        """The local day n days before D (n=0 is D)."""
        return day_window(self.day - timedelta(days=n), self.zone)

    def last_days(self, days: int, ending_days_ago: int = 0) -> Window:
        """`days` local days ending `ending_days_ago` days before D, inclusive."""
        last = self.day - timedelta(days=ending_days_ago)
        return span_window(last - timedelta(days=days - 1), last, self.zone)

    @property
    def is_latest(self) -> bool:
        """True when D is the brand's yesterday - the only day whose "right now"
        state (open qualified leads, awaiting replies) describes D."""
        return self.now.astimezone(self.zone.tz).date() - timedelta(days=1) <= self.day


def within(ts: Optional[datetime], window: Window) -> bool:
    return ts is not None and window.start <= ts < window.end


def watch(cfg: dict, kind: str, severity: str, entity: Optional[str] = None,
          **fields: Any) -> dict:
    return {"type": kind, "severity": severity, "entity": entity,
            "text": cfg["watch_templates"][kind].format(**fields)}


def sort_watch(items: list[dict], cfg: dict) -> list[dict]:
    rank = cfg["severity_rank"]
    return sorted(items, key=lambda w: rank.get(w["severity"], 9))


def kpi(value, display, previous=None, delta=None, **extra) -> dict:
    return {"value": value, "display": display, "previous": previous, "delta": delta, **extra}


def result(section: str, *, facts: dict, kpis: dict, watch_items: list, template: dict,
           cfg: dict, extras: Optional[dict] = None) -> dict:
    out = {"section": section, "available": True, "reason": None, "message": None,
           "facts": facts, "kpis": kpis, "watch": sort_watch(watch_items, cfg),
           "template": template}
    out.update(extras or {})
    return out


def unavailable(section: str, reason: str, message: str) -> dict:
    return {"section": section, "available": False, "reason": reason, "message": message,
            "facts": {}, "kpis": {}, "watch": [],
            "template": {"summary": message, "top": "", "watch": "", "blocks": []}}


def watch_sentence(items: list[dict], limit: int = 2) -> str:
    return " ".join(w["text"] for w in items[:limit])


def join_parts(parts: list[Optional[str]], sep: str = ", ") -> str:
    return sep.join(p for p in parts if p)
