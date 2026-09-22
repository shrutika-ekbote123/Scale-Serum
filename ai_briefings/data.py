"""Read-only PostgreSQL access for AI Briefings.

Synchronous psycopg, run in a thread by the caller. Connections are borrowed
from the purchase-probability pool, like the AI Suggested Next Action, so one
DB_POOL_MAX covers every feature that reads scrumdb.

Every query is bounded by an explicit UTC window computed from the brand's
local day, so a briefing for a past date reads the data as it stood on that
date wherever the schema allows it (created_at / occurred_at / insight date).
Contact details are never selected, with one exception: the hot-lead scorer
needs the lead's email to classify it (gmail / corporate), exactly as the
purchase-probability endpoint does. It is used in memory and never returned.

Traps this module is written around (see the data audit in the plan):
  * insights exist at several entity levels; only `campaign` rows are read.
  * payments written by the attribution backfill (`payload ? 'backfill'`) are
    excluded - they are not money received on that day.
  * campaign names say "_Paused" on ACTIVE campaigns; status comes from
    effective_status, and budgets from ad sets when the campaign has none.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from purchase_probability_model.inference import _borrow

from .timezones import BrandZone, Window, day_window, span_window


class DataUnavailable(Exception):
    """scrumdb could not be read. Distinct from 'brand not found'."""


_NOT_BACKFILL = """NOT COALESCE(jsonb_typeof(te.payload::jsonb) = 'object'
                                AND te.payload::jsonb ? 'backfill', false)"""

_SQL_BRAND = """
    SELECT id, name, timezone, currency, brand_brain_id FROM brands WHERE id = %s
"""

_SQL_ACTIVE_BRANDS = """
    SELECT b.id FROM brands b
    WHERE EXISTS (SELECT 1 FROM leads l WHERE l.brand_id = b.id AND l.created_at >= %(since)s)
       OR EXISTS (SELECT 1 FROM meta_insights m WHERE m.brand_id = b.id AND m.date >= %(since_d)s)
       OR EXISTS (SELECT 1 FROM google_insights g WHERE g.brand_id = b.id AND g.date >= %(since_d)s)
       OR EXISTS (SELECT 1 FROM sales_calls s WHERE s.brand_id = b.id AND s.occurred_at >= %(since)s)
"""

# ---------------------------------------------------------------------- ads
_SQL_INSIGHTS = """
    SELECT 'meta', entity_id, date, spend::float, impressions::float, clicks::float,
           leads::float, purchases::float, revenue::float, conversions::float
    FROM meta_insights
    WHERE brand_id = %(b)s AND entity_type = 'campaign' AND date BETWEEN %(d0)s AND %(d1)s
    UNION ALL
    SELECT 'google', entity_id, date, spend::float, impressions::float, clicks::float,
           leads::float, purchases::float, revenue::float, conversions::float
    FROM google_insights
    WHERE brand_id = %(b)s AND entity_type = 'campaign' AND date BETWEEN %(d0)s AND %(d1)s
    UNION ALL
    SELECT 'linkedin', entity_id, date, spend::float, impressions::float, clicks::float,
           leads::float, purchases::float, revenue::float, conversions::float
    FROM linkedin_insights
    WHERE brand_id = %(b)s AND entity_type = 'campaign' AND date BETWEEN %(d0)s AND %(d1)s
"""

# Meta budgets: the campaign's own (CBO) or the sum of its ACTIVE ad sets (ABO).
_SQL_META_CAMPAIGNS = """
    SELECT mc.campaign_id, mc.name, mc.effective_status, mc.daily_budget::float,
           (SELECT sum(ma.daily_budget)::float FROM meta_ad_sets ma
             WHERE ma.brand_id = mc.brand_id AND ma.campaign_id = mc.campaign_id
               AND ma.effective_status = 'ACTIVE')
    FROM meta_campaigns mc
    WHERE mc.brand_id = %(b)s AND mc.campaign_id = ANY(%(ids)s)
"""

_SQL_GOOGLE_CAMPAIGNS = """
    SELECT campaign_id, name, status, daily_budget::float
    FROM google_campaigns WHERE brand_id = %(b)s AND campaign_id = ANY(%(ids)s)
"""

_SQL_LINKEDIN_CAMPAIGNS = """
    SELECT campaign_id, name, status, daily_budget::float
    FROM linkedin_campaigns WHERE brand_id = %(b)s AND campaign_id = ANY(%(ids)s)
"""

# Which platforms the brand has connected, and when each last produced data.
_SQL_FRESHNESS = """
    SELECT 'meta', (SELECT count(*) FROM meta_ad_accounts WHERE brand_id = %(b)s),
           (SELECT max(date) FROM meta_insights WHERE brand_id = %(b)s AND entity_type = 'campaign'),
           (SELECT max(last_synced_at) FROM meta_ad_accounts WHERE brand_id = %(b)s)
    UNION ALL
    SELECT 'google', (SELECT count(*) FROM google_ad_accounts WHERE brand_id = %(b)s),
           (SELECT max(date) FROM google_insights WHERE brand_id = %(b)s AND entity_type = 'campaign'),
           (SELECT max(last_synced_at) FROM google_ad_accounts WHERE brand_id = %(b)s)
    UNION ALL
    SELECT 'linkedin', (SELECT count(*) FROM linkedin_ad_accounts WHERE brand_id = %(b)s),
           (SELECT max(date) FROM linkedin_insights WHERE brand_id = %(b)s AND entity_type = 'campaign'),
           (SELECT max(last_synced_at) FROM linkedin_ad_accounts WHERE brand_id = %(b)s)
"""

# The creative analyzer overwrites one row per ad and range, so only the latest
# verdict survives. For day D we take verdicts analysed on D or the morning after.
_SQL_CREATIVES = """
    SELECT DISTINCT ON (ad_id) ad_id, ad_name, verdict, creative_score::float, analyzed_at
    FROM mi_ad_creative_analyses
    WHERE brand_id = %(b)s AND platform = 'meta' AND range_key = %(range_key)s
      AND analyzed_at >= %(t0)s AND analyzed_at < %(t1)s
    ORDER BY ad_id, analyzed_at DESC
"""

# ------------------------------------------------------------------ revenue
_SQL_PAYMENTS = f"""
    SELECT te.lead_id, te.occurred_at, te.value::float,
           NULLIF(split_part(te.payload::jsonb #>> '{{data,order,order_id}}', '_', 2), '')
    FROM touchpoint_events te
    WHERE te.brand_id = %(b)s AND te.type = 'payment' AND te.value > 0
      AND te.occurred_at >= %(t0)s AND te.occurred_at < %(t1)s
      AND {_NOT_BACKFILL}
"""

# -------------------------------------------------------------------- leads
_SQL_LEADS = """
    SELECT id, created_at, funnel, status::text
    FROM leads
    WHERE brand_id = %(b)s AND created_at >= %(t0)s AND created_at < %(t1)s
"""

_SQL_OPEN_QUALIFIED = """
    SELECT count(*) FROM leads WHERE brand_id = %(b)s AND status = 'qualified'
"""

# Hot-lead candidates: open, old enough to have been worked, and never worked.
# "Worked" = a sales call, or an outbound WhatsApp message, before the end of D.
# The form payload is the one the purchase-probability model admits (written at
# lead creation, not backdated), so the score is the model's, not a guess.
_SQL_UNTOUCHED = f"""
    SELECT l.id, l.created_at, l.email, f.payload
    FROM leads l
    JOIN LATERAL (
        SELECT te.payload::jsonb AS payload FROM touchpoint_events te
        WHERE te.lead_id = l.id AND te.type = 'form_submit'
          AND te.created_at <= l.created_at + INTERVAL '1 hour'
          AND te.created_at - te.occurred_at <= INTERVAL '1 hour'
        ORDER BY te.occurred_at ASC, te.created_at ASC LIMIT 1
    ) f ON true
    WHERE l.brand_id = %(b)s AND l.created_at >= %(t0)s AND l.created_at < %(cutoff)s
      AND l.status::text <> 'lost'
      AND NOT EXISTS (SELECT 1 FROM touchpoint_events te
                      WHERE te.lead_id = l.id AND te.type = 'payment'
                        AND te.occurred_at < %(t1)s AND {_NOT_BACKFILL})
      AND NOT EXISTS (SELECT 1 FROM sales_calls sc
                      WHERE sc.lead_id = l.id AND sc.occurred_at < %(t1)s)
      AND NOT EXISTS (SELECT 1 FROM whatsapp_conversations wc
                      JOIN whatsapp_messages wm ON wm.conversation_id = wc.id
                      WHERE wc.lead_id = l.id AND wm.direction::text = 'outbound'
                        AND wm.created_at < %(t1)s)
"""

# -------------------------------------------------------------------- sales
_SQL_CALLS = """
    SELECT id, lead_id, rep_name, rep_user_id, occurred_at, duration_seconds::float,
           disposition::text, callback_at, analysis_id, score::float
    FROM sales_calls
    WHERE brand_id = %(b)s AND occurred_at >= %(t0)s AND occurred_at < %(t1)s
"""

# ----------------------------------------------------------------- whatsapp
_SQL_WA_ACCOUNTS = "SELECT count(*) FROM whatsapp_accounts WHERE brand_id = %(b)s"

_SQL_WA_BROADCASTS = """
    SELECT id, name, status, recipient_count, sent_count, delivered_count, read_count,
           replied_count, failed_count, estimated_cost::float, sent_at
    FROM whatsapp_broadcasts
    WHERE brand_id = %(b)s AND sent_at >= %(t0)s AND sent_at < %(t1)s
"""

_SQL_WA_INBOUND = """
    SELECT wm.conversation_id, wc.lead_id, wm.created_at
    FROM whatsapp_messages wm
    JOIN whatsapp_conversations wc ON wc.id = wm.conversation_id
    WHERE wm.brand_id = %(b)s AND wm.direction::text = 'inbound'
      AND wm.created_at >= %(t0)s AND wm.created_at < %(t1)s
"""

_SQL_WA_AWAITING = """
    SELECT count(*) FROM whatsapp_conversations
    WHERE brand_id = %(b)s AND last_direction::text = 'inbound'
      AND last_message_at >= %(t0)s AND last_message_at < %(t1)s
"""

# approved_at is never filled in, so "approved yesterday" is read from the
# status change's updated_at.
_SQL_WA_TEMPLATES = """
    SELECT status::text, count(*),
           array_agg(name) FILTER (WHERE status::text = 'approved'
                                   AND updated_at >= %(t0)s AND updated_at < %(t1)s)
    FROM whatsapp_templates WHERE brand_id = %(b)s GROUP BY 1
"""


def _rows(cur, sql: str, params: dict) -> list[tuple]:
    cur.execute(sql, params)
    return cur.fetchall()


def load_brand(brand_id: str, conn=None) -> Optional[dict]:
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            cur.execute(_SQL_BRAND, (brand_id,))
            row = cur.fetchone()
    except Exception as err:  # noqa: BLE001 - driver errors are varied
        raise DataUnavailable(type(err).__name__) from err
    if not row:
        return None
    return {"id": str(row[0]), "name": row[1], "timezone": row[2],
            "currency": row[3] or "INR", "brand_brain_id": row[4]}


def active_brand_ids(now: datetime, days: int = 35, conn=None) -> list[str]:
    """Brands with any leads, ad data or calls recently - the worker's roster."""
    since = now - timedelta(days=days)
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            rows = _rows(cur, _SQL_ACTIVE_BRANDS, {"since": since, "since_d": since.date()})
    except Exception as err:  # noqa: BLE001
        raise DataUnavailable(type(err).__name__) from err
    return [str(r[0]) for r in rows]


def load_day(brand_id: str, day: date, zone: BrandZone, cfg: dict, now: datetime,
             conn=None) -> dict:
    """Everything the four section builders need for one brand-local day.

    Raises DataUnavailable when scrumdb cannot be read. The hot-lead candidates
    are loaded here too; scoring them is hot_leads.py's job."""
    d = day_window(day, zone)
    lookback = max(cfg["sales"]["closure_credit_days"] + cfg["sales"]["window_days"],
                   cfg["leads"]["funnel_lookback_days"], 14) + 1
    wide = span_window(day - timedelta(days=lookback), day, zone)
    b = brand_id
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            insights = _rows(cur, _SQL_INSIGHTS, {"b": b, "d0": day - timedelta(days=14), "d1": day})
            ids = {p: sorted({r[1] for r in insights if r[0] == p}) for p in ("meta", "google", "linkedin")}
            campaigns = {}
            if ids["meta"]:
                for r in _rows(cur, _SQL_META_CAMPAIGNS, {"b": b, "ids": ids["meta"]}):
                    campaigns[("meta", r[0])] = {"name": r[1], "status": r[2],
                                                 "daily_budget": r[3] or r[4], "budget_level":
                                                 "campaign" if r[3] else ("ad_set" if r[4] else None)}
            for platform, sql in (("google", _SQL_GOOGLE_CAMPAIGNS), ("linkedin", _SQL_LINKEDIN_CAMPAIGNS)):
                if ids[platform]:
                    for r in _rows(cur, sql, {"b": b, "ids": ids[platform]}):
                        campaigns[(platform, r[0])] = {"name": r[1], "status": r[2],
                                                       "daily_budget": r[3],
                                                       "budget_level": "campaign" if r[3] else None}
            freshness = [{"platform": r[0], "accounts": int(r[1] or 0), "last_date": r[2],
                          "last_synced_at": r[3]}
                         for r in _rows(cur, _SQL_FRESHNESS, {"b": b})]
            creatives = [{"ad_id": r[0], "ad_name": r[1], "verdict": r[2], "score": r[3],
                          "analyzed_at": r[4]}
                         for r in _rows(cur, _SQL_CREATIVES, {
                             "b": b, "range_key": cfg["ads"]["creative_range_key"],
                             "t0": d.start, "t1": d.end + timedelta(days=1)})]

            payments = [{"lead_id": str(r[0]) if r[0] else None, "occurred_at": r[1],
                         "value": r[2], "product_code": r[3]}
                        for r in _rows(cur, _SQL_PAYMENTS, {"b": b, "t0": wide.start, "t1": d.end})]

            leads = [{"id": str(r[0]), "created_at": r[1], "funnel": r[2], "status": r[3]}
                     for r in _rows(cur, _SQL_LEADS, {"b": b, "t0": wide.start, "t1": d.end})]
            open_qualified = int(_rows(cur, _SQL_OPEN_QUALIFIED, {"b": b})[0][0] or 0)

            hot_t0 = d.end - timedelta(days=cfg["leads"]["hot_lookback_days"])
            cutoff = d.end - timedelta(hours=cfg["leads"]["untouched_hours"])
            untouched = [{"id": str(r[0]), "created_at": r[1], "email": r[2], "payload": r[3]}
                         for r in _rows(cur, _SQL_UNTOUCHED, {"b": b, "t0": hot_t0,
                                                              "cutoff": cutoff, "t1": d.end})]

            calls = [{"id": str(r[0]), "lead_id": str(r[1]) if r[1] else None,
                      "rep_name": r[2], "rep_user_id": str(r[3]) if r[3] else None,
                      "occurred_at": r[4], "duration_seconds": r[5], "disposition": r[6],
                      "callback_at": r[7], "analysis_id": r[8], "score": r[9]}
                     for r in _rows(cur, _SQL_CALLS, {"b": b, "t0": wide.start,
                                                      "t1": min(d.end, now)})]

            wa_accounts = int(_rows(cur, _SQL_WA_ACCOUNTS, {"b": b})[0][0] or 0)
            whatsapp = {"accounts": wa_accounts}
            if wa_accounts:
                reply_days = cfg["whatsapp"]["reply_to_sql_days"]
                whatsapp.update({
                    "broadcasts": [{"id": str(r[0]), "name": r[1], "status": r[2],
                                    "recipients": r[3] or 0, "sent": r[4] or 0,
                                    "delivered": r[5] or 0, "read": r[6] or 0,
                                    "replied": r[7] or 0, "failed": r[8] or 0,
                                    "estimated_cost": r[9], "sent_at": r[10]}
                                   for r in _rows(cur, _SQL_WA_BROADCASTS,
                                                  {"b": b, "t0": d.start, "t1": d.end})],
                    "inbound": [{"conversation_id": str(r[0]),
                                 "lead_id": str(r[1]) if r[1] else None, "at": r[2]}
                                for r in _rows(cur, _SQL_WA_INBOUND, {
                                    "b": b, "t0": d.end - timedelta(days=reply_days),
                                    "t1": d.end})],
                    "awaiting_reply": int(_rows(cur, _SQL_WA_AWAITING, {
                        "b": b, "t0": d.end - timedelta(days=cfg["whatsapp"]["awaiting_reply_days"]),
                        "t1": d.end})[0][0] or 0),
                    "templates": {r[0]: {"count": int(r[1]), "approved_today": list(r[2] or [])}
                                  for r in _rows(cur, _SQL_WA_TEMPLATES,
                                                 {"b": b, "t0": d.start, "t1": d.end})},
                })
    except Exception as err:  # noqa: BLE001
        raise DataUnavailable(type(err).__name__) from err

    return {
        "insights": [{"platform": r[0], "campaign_id": r[1], "date": r[2], "spend": r[3] or 0.0,
                      "impressions": r[4] or 0.0, "clicks": r[5] or 0.0, "leads": r[6] or 0.0,
                      "purchases": r[7] or 0.0, "revenue": r[8] or 0.0, "conversions": r[9] or 0.0}
                     for r in insights],
        "campaigns": {f"{p}:{cid}": v for (p, cid), v in campaigns.items()},
        "freshness": freshness,
        "creatives": creatives,
        "payments": payments,
        "leads": leads,
        "open_qualified": open_qualified,
        "untouched": untouched,
        "calls": calls,
        "whatsapp": whatsapp,
    }
