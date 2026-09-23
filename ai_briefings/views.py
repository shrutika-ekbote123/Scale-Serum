"""Read side: stored briefings shaped for one viewer.

No generation and no scrumdb reads here - every endpoint is a Mongo lookup, so
the Briefings page loads instantly.

Two things are filtered per viewer (see access.py):
  * tabs    - a viewer only gets the sections their permissions allow. Their
              "All" tab is then composed on read from those sections, because
              the stored All briefing may mention the others.
  * reps    - without team view, the Sales Team tab shows the team numbers and
              the viewer's own rep row, never a colleague's name.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from . import timezones as tzs
from .access import Viewer
from .config import SECTIONS, load_config
from .store import BriefingStore, briefing_id


class Forbidden(Exception):
    pass


FAR_FUTURE = "9999-12-31"


def _brand_yesterday(doc: dict, now: datetime) -> Optional[str]:
    zone = tzs.resolve((doc.get("timezone") or {}).get("name"))
    return tzs.yesterday(now, zone).isoformat()


def _public_sales(doc: dict, viewer: Viewer) -> dict:
    """The Sales Team tab for a viewer without team view."""
    doc = dict(doc)
    own = [r for r in doc.get("reps") or [] if viewer.user_id and r.get("user_id") == viewer.user_id]
    summary = dict(doc.get("summary") or {})
    summary["top"] = ""
    summary["watch"] = own[0]["watch"] if own and own[0].get("watch") else ""
    summary["watch_label"] = "Coach"
    doc["summary"] = summary
    doc["reps"] = own
    doc["blocks"] = [b for b in doc.get("blocks") or [] if b["key"] != "reps"]
    doc["watch"] = [w for w in doc.get("watch") or [] if w["type"] != "missed_callbacks"]
    facts = dict(doc.get("facts") or {})
    facts.pop("top_rep", None)
    facts.pop("coach", None)
    doc["facts"] = facts
    return doc


def _for_viewer(doc: dict, viewer: Viewer) -> dict:
    if doc["section"] == "sales" and not viewer.team_view:
        return _public_sales(doc, viewer)
    return doc


def _compose_all(docs: dict, day: str, viewer: Viewer, cfg: dict) -> Optional[dict]:
    """An All tab from only the sections this viewer may see."""
    live = [docs[s] for s in SECTIONS if s in docs and docs[s].get("available")]
    if not live:
        return None
    first = live[0]
    rank = cfg["severity_rank"]
    watch = sorted((dict(w, section=d["section"]) for d in live for w in d.get("watch") or []),
                   key=lambda w: rank.get(w["severity"], 9))[:3]
    sales = docs.get("sales")
    return {
        "_id": briefing_id(first["brand_id"], day, "all"), "briefing_id": None,
        "brand_id": first["brand_id"], "brand_name": first.get("brand_name"),
        "date": day, "date_label": first["date_label"], "short_label": first["short_label"],
        "section": "all", "section_label": cfg["sections"]["all"]["label"],
        "available": True, "reason": None, "message": None,
        "summary": {"label": cfg["sections"]["all"]["summary_label"],
                    "text": " ".join(f"{d['summary']['label']}: {d['summary']['text']}" for d in live),
                    "top_label": "Top", "top": "", "watch_label": "Watch",
                    "watch": " ".join(w["text"] for w in watch[:2])},
        "blocks": [{"key": d["section"], "title": d["section_label"],
                    "bullets": [d["summary"]["text"]]} for d in live]
                  + ([{"key": "notable_changes", "title": "Notable changes",
                       "bullets": [w["text"] for w in watch]}] if watch else []),
        "watch": watch, "kpis": {k: v for d in live for k, v in (d.get("kpis") or {}).items()},
        "facts": {"sections": [d["section"] for d in live]},
        "timezone": first.get("timezone"), "data_as_of": first.get("data_as_of"),
        "wording": {"source": "composed", "fallback_reason": None},
        "fallback": any(d.get("fallback") for d in live),
        "generated_at": max(d["generated_at"] for d in live),
        "score_card": sales.get("score_card") if sales and sales.get("available") else None,
        "composed_for_viewer": True,
    }


def shape(doc: dict, *, viewer: Viewer, requested: Optional[str], stale: bool,
          tabs: list[dict]) -> dict:
    keep = ("briefing_id", "brand_id", "brand_name", "date", "date_label", "short_label",
            "section", "section_label", "available", "reason", "message", "summary", "blocks",
            "watch", "kpis", "score_card", "reps", "timezone", "data_as_of", "wording",
            "fallback", "generated_at", "versions", "usage", "cost")
    out = {k: doc.get(k) for k in keep}
    out["facts"] = doc.get("facts")
    out["requested_date"] = requested
    out["stale"] = stale
    out["tabs"] = tabs
    out["viewer"] = {"team_view": viewer.team_view, "sections": viewer.sections(),
                     "service": viewer.service}
    return out


def not_generated(brand_id: str, section: str, requested: Optional[str], viewer: Viewer,
                  tabs: list[dict]) -> dict:
    cfg = load_config()
    return {
        "briefing_id": None, "brand_id": brand_id, "section": section,
        "section_label": cfg["sections"][section]["label"], "date": None,
        "available": False, "reason": "not_generated",
        "message": "No briefing has been generated for this brand yet.",
        "summary": None, "blocks": [], "watch": [], "kpis": {}, "score_card": None,
        "requested_date": requested, "stale": False, "tabs": tabs,
        "viewer": {"team_view": viewer.team_view, "sections": viewer.sections(),
                   "service": viewer.service},
    }


async def _day_docs(store: BriefingStore, brand_id: str, day: str) -> dict:
    out = {}
    for s in ("all",) + SECTIONS:
        doc = await store.get_for(brand_id, day, s)
        if doc:
            out[s] = doc
    return out


def _tabs(docs: dict, viewer: Viewer) -> list[dict]:
    cfg = load_config()
    tabs = [{"section": "all", "label": cfg["sections"]["all"]["label"], "available": True,
             "reason": None}]
    for s in viewer.sections():
        doc = docs.get(s)
        tabs.append({"section": s, "label": cfg["sections"][s]["label"],
                     "available": bool(doc and doc.get("available")),
                     "reason": (doc or {}).get("reason") or (None if doc else "not_generated")})
    return tabs


async def today(store: BriefingStore, brand_id: str, section: str, viewer: Viewer,
                requested: Optional[str], now: datetime) -> dict:
    cfg = load_config()
    allowed = viewer.sections()
    if not allowed or (section != "all" and section not in allowed):
        raise Forbidden(section)

    probe = await store.latest(brand_id, "all", FAR_FUTURE)
    if probe is None:
        return not_generated(brand_id, section, requested, viewer, _tabs({}, viewer))
    target = requested or _brand_yesterday(probe, now)

    docs = await _day_docs(store, brand_id, target)
    stale = False
    if not docs:
        earlier = await store.latest(brand_id, "all", target)
        if earlier is None:
            return not_generated(brand_id, section, requested, viewer, _tabs({}, viewer))
        docs = await _day_docs(store, brand_id, earlier["date"])
        stale = True
    day = next(iter(docs.values()))["date"]

    if section == "all" and not (viewer.full_access and viewer.team_view):
        visible = {s: _for_viewer(docs[s], viewer) for s in allowed if s in docs}
        doc = _compose_all(visible, day, viewer, cfg)
        if doc is None:
            doc = docs.get("all") or next(iter(docs.values()))
            doc = {**doc, "available": False, "reason": "no_data",
                   "message": "None of your sections has data for this day."}
    else:
        doc = docs.get(section)
        if doc is None:
            return not_generated(brand_id, section, requested, viewer, _tabs(docs, viewer))
        doc = _for_viewer(doc, viewer)
    return shape(doc, viewer=viewer, requested=requested, stale=stale,
                 tabs=_tabs(docs, viewer))


async def score(store: BriefingStore, brand_id: str, viewer: Viewer,
                requested: Optional[str], now: datetime) -> dict:
    if "sales" not in viewer.sections():
        raise Forbidden("sales")
    out = await today(store, brand_id, "sales", viewer, requested, now)
    return {"brand_id": brand_id, "date": out.get("date"), "date_label": out.get("date_label"),
            "requested_date": requested, "stale": out.get("stale"),
            "available": out.get("available"), "reason": out.get("reason"),
            "message": out.get("message"), "score_card": out.get("score_card"),
            "reps": out.get("reps") or [], "team_view": viewer.team_view,
            "generated_at": out.get("generated_at")}


async def history(store: BriefingStore, brand_id: str, section: str, viewer: Viewer,
                  before: Optional[str], limit: int) -> dict:
    allowed = viewer.sections()
    if not allowed or (section != "all" and section not in allowed):
        raise Forbidden(section)
    if section == "all":
        wanted = list(allowed) + (["all"] if viewer.full_access and viewer.team_view else [])
    else:
        wanted = [section]
    # Paged by DAY, not by document, so one day's tabs never split across pages.
    docs = await store.history(brand_id, wanted, before, limit * len(wanted) + 1)
    days = sorted({d["date"] for d in docs}, reverse=True)
    page_days = set(days[:limit])
    items = [{"briefing_id": d["_id"], "date": d["date"], "date_label": d.get("date_label"),
              "short_label": d.get("short_label"), "section": d["section"],
              "section_label": d.get("section_label"),
              "summary": (d.get("summary") or {}).get("text"),
              "fallback": d.get("fallback"), "generated_at": d.get("generated_at")}
             for d in docs if d.get("available") and d["date"] in page_days]
    order = {s: i for i, s in enumerate(("all",) + SECTIONS)}
    items.sort(key=lambda i: (i["date"], -order.get(i["section"], 9)), reverse=True)
    next_before = min(page_days) if len(days) > limit else None
    return {"brand_id": brand_id, "section": section, "items": items,
            "next_before": next_before}


async def one(store: BriefingStore, bid: str, viewer: Viewer) -> Optional[dict]:
    doc = await store.get(bid)
    if doc is None:
        return None
    allowed = viewer.sections()
    if doc["section"] == "all":
        if not (viewer.full_access and viewer.team_view):
            raise Forbidden("all")
    elif doc["section"] not in allowed:
        raise Forbidden(doc["section"])
    return shape(_for_viewer(doc, viewer), viewer=viewer, requested=doc["date"],
                 stale=False, tabs=[])


async def latest(store: BriefingStore, brand_id: str, viewer: Viewer, now: datetime) -> dict:
    """The dashboard header card: the All briefing, compact."""
    out = await today(store, brand_id, "all", viewer, None, now)
    watch = (out.get("watch") or [])[:1]
    return {"brand_id": brand_id, "available": out.get("available"),
            "reason": out.get("reason"), "message": out.get("message"),
            "date": out.get("date"), "date_label": out.get("date_label"),
            "stale": out.get("stale"), "generated_at": out.get("generated_at"),
            "summary": out.get("summary"), "top_watch": watch[0] if watch else None,
            "fallback": out.get("fallback")}
