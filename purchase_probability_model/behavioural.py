"""
Behavioural layer - click actions and journey timeline for one lead.

WHAT THIS IS FOR
    The calibrated base model scores a lead from its form submission alone. This
    module answers the other half of the question the product asks: what has this
    person actually *done* since, and how recently.

WHAT IT MAY READ (and why the filter is not optional)
    scrumdb's attribution pipeline writes touchpoint rows at conversion time and
    backdates occurred_at, manufacturing journeys that never happened. All 234
    ad_click rows and 225/228 call rows were written at or after their lead's
    payment. A behavioural feature built naively on those rows is reading the
    receipt, not the journey.

    So every row admitted here must satisfy BOTH:
        * it is not backdated   - created_at - occurred_at <= tolerance
        * it was written before the observation window closed
    and 'payment' is excluded outright: it is the label.

    tracking_events (the ss.js collector) is trustworthy by construction - rows
    are written as they arrive - but it is currently EMPTY in this database, so in
    practice this layer reports no_behavioural_data for almost every lead today.
    That is a data-collection gap, not a failure. Absent signal is reported as
    absent; it is never scored as a negative.

NOTHING HERE IS A BASE-MODEL FEATURE. The base model's inputs are unchanged.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

# Why a lead has no usable behavioural history. Stable identifiers.
NO_BEHAVIOURAL_DATA = "no_behavioural_data"
ONLY_FORM_SUBMISSION = "only_form_submission"

# The first payment WRITE time (not its backdated occurred_at) closes the window:
# browsing done after someone has already paid must not raise a "will they buy"
# score. created_at is assigned by the database and cannot be backdated.
_SQL_PAYMENT_WRITE_CUTOFF = """
    SELECT MIN(te.created_at)
    FROM touchpoint_events te
    WHERE te.lead_id = %s AND te.type = 'payment'
"""

# Admissible, non-label touchpoints. Both anti-backfill clauses are required.
_SQL_ADMISSIBLE_BEHAVIOUR = """
    SELECT te.type::text AS type, te.occurred_at, te.created_at,
           te.channel, te.source, te.provider,
           te.utm_source, te.utm_medium, te.utm_campaign,
           te.fbclid, te.gclid, te.li_fat_id
    FROM touchpoint_events te
    WHERE te.lead_id = %(lead_id)s
      AND te.type <> 'payment'
      AND te.created_at - te.occurred_at <= %(tolerance)s
      AND te.created_at <= %(window_end)s
    ORDER BY te.occurred_at ASC, te.created_at ASC
    LIMIT %(limit)s
"""

# On-site events, joined to the lead through every visitor id we can resolve.
# received_at (server write time) is authoritative; client_ts is client-supplied
# and therefore never used for admissibility.
_SQL_TRACKING_EVENTS = """
    WITH vis AS (
        SELECT tv.visitor_id AS visitor_id
        FROM tracking_visitors tv
        WHERE tv.lead_id = %(lead_id)s
        UNION
        SELECT li.key_value AS visitor_id
        FROM lead_identities li
        WHERE li.lead_id = %(lead_id)s AND li.key_type = 'anon_cookie'
    )
    SELECT te.name, te.path, te.url, te.session_id, te.received_at,
           te.click::text AS click, te.utm::text AS utm, te.referrer
    FROM tracking_events te
    WHERE te.brand_id = %(brand_id)s
      AND te.visitor_id IN (SELECT visitor_id FROM vis)
      AND te.received_at >= %(window_start)s
      AND te.received_at <= %(window_end)s
    ORDER BY te.received_at ASC
    LIMIT %(limit)s
"""


# --------------------------------------------------------------------------- helpers
def _loads(raw: Any) -> dict:
    """Tolerant JSON read. A malformed column must not sink the whole response."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def _hours(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / 3600.0


def normalise_channel(raw: Optional[str], aliases: dict) -> Optional[str]:
    """Map a free-text channel / utm_source onto a canonical channel key."""
    if not raw:
        return None
    text = re.sub(r"[^a-z0-9+ ]+", " ", str(raw).strip().lower())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    for canonical, variants in aliases.items():
        for variant in variants:
            v = str(variant).strip().lower()
            if not v:
                continue
            if text == v or re.search(
                    r"(^|[^a-z0-9])" + re.escape(v) + r"([^a-z0-9]|$)", text):
                return canonical
    return None


def _classify(name: Optional[str], path: Optional[str], cfg: dict) -> set:
    """Tag one on-site event. An event can be both a click and high-intent."""
    ec = cfg["event_classification"]
    name_s = (name or "").lower()
    path_s = (path or "").lower()
    tags = set()
    if re.search(ec["click_names"], name_s):
        tags.add("click")
    if re.search(ec["page_view_names"], name_s):
        tags.add("page_view")
    if re.search(ec["form_names"], name_s):
        tags.add("form")
    if re.search(ec["high_intent_event_names"], name_s) or (
            path_s and re.search(ec["high_intent_paths"], path_s)):
        tags.add("high_intent")
    return tags


# --------------------------------------------------------------------------- collection
def collect_behaviour(cur, lead: dict, cfg: dict, now: Optional[datetime] = None) -> dict:
    """Gather admissible click + timeline evidence for one lead.

    `cur` is a live read-only cursor. The returned dict is always well formed;
    when there is nothing admissible to report it carries available=False and a
    reason, never zeroed-out counters presented as observations.
    """
    now = now or datetime.now(timezone.utc)
    lead_id = lead["id"]
    brand_id = lead.get("brand_id")
    t0 = lead["created_at"]
    win = cfg["window"]
    tolerance = timedelta(seconds=int(win["backdate_tolerance_seconds"]))
    limit = int(win["max_events"])

    # ---- observation window -------------------------------------------------
    window_end, end_reason = now, "now"
    try:
        cur.execute(_SQL_PAYMENT_WRITE_CUTOFF, (lead_id,))
        row = cur.fetchone()
        if row and row[0] and row[0] < window_end:
            window_end, end_reason = row[0], "first_payment_write"
    except Exception:
        pass  # no cutoff available -> fall back to `now`, which is never unsafe
    window_start = t0 - timedelta(days=int(win["pre_lead_lookback_days"]))

    # ---- admissible touchpoints --------------------------------------------
    tp_rows = []
    try:
        cur.execute(_SQL_ADMISSIBLE_BEHAVIOUR, {
            "lead_id": lead_id, "tolerance": tolerance,
            "window_end": window_end, "limit": limit})
        cols = [d[0] for d in cur.description]
        tp_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception:
        tp_rows = []

    # ---- on-site tracking events -------------------------------------------
    tr_rows = []
    if brand_id is not None:
        try:
            cur.execute(_SQL_TRACKING_EVENTS, {
                "lead_id": lead_id, "brand_id": brand_id,
                "window_start": window_start, "window_end": window_end,
                "limit": limit})
            cols = [d[0] for d in cur.description]
            tr_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            tr_rows = []  # tables may not exist on every deployment

    # ---- fold both sources into one timeline -------------------------------
    clicks = page_views = form_submits = high_intent = 0
    ad_click_arrival = False
    sessions = set()
    days = set()
    stamps = []
    timeline = []
    channel_candidates = []

    for r in tp_rows:
        occurred = r.get("occurred_at") or r.get("created_at")
        if occurred is None:
            continue
        kind = r.get("type") or "custom"
        if kind == "form_submit":
            form_submits += 1
        elif kind == "ad_click":
            clicks += 1
            ad_click_arrival = True
        elif kind == "page_view":
            page_views += 1
        elif kind == "call":
            high_intent += 1
        if any(r.get(k) for k in ("fbclid", "gclid", "li_fat_id")):
            ad_click_arrival = True
        for key in ("channel", "utm_source", "source", "provider"):
            if r.get(key):
                channel_candidates.append(str(r[key]))
        stamps.append(occurred)
        days.add(occurred.date().isoformat())
        timeline.append({
            "at": occurred.isoformat(), "source": "touchpoint_events",
            "kind": kind, "channel": r.get("channel"),
            "campaign": r.get("utm_campaign"), "utm_source": r.get("utm_source"),
        })

    for r in tr_rows:
        occurred = r.get("received_at")
        if occurred is None:
            continue
        tags = _classify(r.get("name"), r.get("path"), cfg)
        if "click" in tags:
            clicks += 1
        if "page_view" in tags:
            page_views += 1
        if "form" in tags:
            form_submits += 1
        if "high_intent" in tags:
            high_intent += 1
        click_ids = _loads(r.get("click"))
        if any(click_ids.get(k) for k in ("fbclid", "gclid", "li_fat_id")):
            ad_click_arrival = True
        utm = _loads(r.get("utm"))
        if utm.get("source"):
            channel_candidates.append(str(utm["source"]))
        if r.get("session_id"):
            sessions.add(str(r["session_id"]))
        stamps.append(occurred)
        days.add(occurred.date().isoformat())
        timeline.append({
            "at": occurred.isoformat(), "source": "tracking_events",
            "kind": r.get("name"), "path": r.get("path"),
            "tags": sorted(tags) or None,
        })

    timeline.sort(key=lambda e: e["at"])
    events_total = len(timeline)
    channel_raw = channel_candidates[0] if channel_candidates else None
    channel = normalise_channel(channel_raw, cfg["channel_aliases"])
    window_block = {"start": window_start.isoformat(),
                    "end": window_end.isoformat(), "end_reason": end_reason}
    sources_block = {"tracking_events": len(tr_rows), "touchpoint_events": len(tp_rows)}

    # A lone admitting form submission is not behavioural evidence - it is the
    # thing the base model already scored. Reporting it as engagement would
    # double-count it, and its age is already covered by the sales-cycle factor.
    has_signal = bool(tr_rows) or form_submits > 1 or (events_total - form_submits) > 0
    if not has_signal:
        reason = ONLY_FORM_SUBMISSION if events_total else NO_BEHAVIOURAL_DATA
        return {
            "available": False,
            "reason": reason,
            "message": (
                "Only the original form submission is on record, so there is no "
                "post-submission behaviour to read."
                if reason == ONLY_FORM_SUBMISSION else
                "No admissible click or on-site activity is recorded for this lead. "
                "The on-site collector has not reported events for it."),
            "observed": None,
            "channel": channel,
            "channel_raw": channel_raw,
            "window": window_block,
            "timeline": timeline,
            "sources": sources_block,
        }

    first_seen, last_seen = min(stamps), max(stamps)
    active_days = len(days)
    return {
        "available": True,
        "reason": None,
        "message": None,
        "observed": {
            "clicks": clicks,
            "page_views": page_views,
            "form_submits": form_submits,
            "high_intent_hits": high_intent,
            "sessions": len(sessions),
            "events_total": events_total,
            "active_days": active_days,
            "returning": active_days > 1,
            "ad_click_arrival": ad_click_arrival,
            "first_seen_at": first_seen.isoformat(),
            "last_seen_at": last_seen.isoformat(),
            "recency_hours": round(_hours(window_end, last_seen), 3),
            "span_hours": round(_hours(last_seen, first_seen), 3),
            "hours_to_first_activity": round(_hours(first_seen, t0), 3),
            "events_per_active_day": (round(events_total / active_days, 3)
                                      if active_days else None),
        },
        "channel": channel,
        "channel_raw": channel_raw,
        "window": window_block,
        "timeline": timeline,
        "sources": sources_block,
    }
