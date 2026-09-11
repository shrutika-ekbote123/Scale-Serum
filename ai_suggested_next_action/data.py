"""Read-only PostgreSQL access for the AI Suggested Next Action.

Everything here is synchronous psycopg; app.py runs it in the threadpool. It
borrows connections from the purchase-probability pool rather than opening a
second pool against the same remote database - one handshake budget, one
DB_POOL_MAX, for both features.

Contact details (email, phone) are never selected. The only phone-derived value
is a COUNT of distinct numbers, used to flag a lead with conflicting identities.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from purchase_probability_model.inference import _borrow


class DataUnavailable(Exception):
    """The database could not be read. Distinct from 'lead not found'."""


_SQL_LEAD = """
    SELECT l.id, l.brand_id, l.full_name, l.status::text, l.created_at,
           l.designation, b.name, b.timezone, b.currency, b.brand_brain_id
    FROM leads l
    LEFT JOIN brands b ON b.id = l.brand_id
    WHERE l.id = %s
"""

# `synthetic` marks rows a seeding script wrote (payload.backfill). They look
# like intent - "Webinar Attended", "WhatsApp Follow-up Reply" - but never
# happened, so the rules ignore them and the response says so.
_SQL_TOUCHPOINTS = """
    SELECT te.type::text, te.occurred_at, te.created_at, te.source, te.channel,
           te.value::float, te.currency,
           NULLIF(split_part(te.payload::jsonb #>> '{data,order,order_id}', '_', 2), ''),
           COALESCE(jsonb_typeof(te.payload::jsonb) = 'object'
                    AND te.payload::jsonb ? 'backfill', false)
    FROM touchpoint_events te
    WHERE te.lead_id = %s
    ORDER BY te.occurred_at ASC, te.created_at ASC
"""

_SQL_CALLS = """
    SELECT occurred_at, disposition::text, direction::text, callback_at,
           analysis_id, analysis_status
    FROM sales_calls
    WHERE lead_id = %s
    ORDER BY occurred_at ASC NULLS FIRST, created_at ASC
"""

_SQL_WHATSAPP = """
    SELECT count(*) OVER (), last_direction::text, last_message_at, unread_count
    FROM whatsapp_conversations
    WHERE lead_id = %s
    ORDER BY last_message_at DESC NULLS LAST
    LIMIT 1
"""

_SQL_PHONE_COUNT = r"""
    SELECT count(DISTINCT right(regexp_replace(key_value, '\D', '', 'g'), 10))
    FROM lead_identities
    WHERE lead_id = %s AND key_type = 'phone'
"""

# One row per distinct paid amount, with the product code most often seen at
# that price. Brand-wide, so it is cached rather than re-run per lead.
_SQL_BRAND_PAYMENTS = """
    SELECT te.value::float, count(*)::int,
           mode() WITHIN GROUP (ORDER BY NULLIF(split_part(
               te.payload::jsonb #>> '{data,order,order_id}', '_', 2), ''))
    FROM touchpoint_events te
    WHERE te.brand_id = %s
      AND te.type = 'payment'
      AND te.value > 0
      AND NOT COALESCE(jsonb_typeof(te.payload::jsonb) = 'object'
                       AND te.payload::jsonb ? 'backfill', false)
    GROUP BY te.value
"""

# How long this brand's leads take from their first form to their first payment.
_SQL_BRAND_WINDOWS = """
    WITH f AS (SELECT lead_id, min(occurred_at) AS t FROM touchpoint_events
               WHERE brand_id = %s AND type = 'form_submit' AND lead_id IS NOT NULL
               GROUP BY lead_id),
         p AS (SELECT lead_id, min(occurred_at) AS t FROM touchpoint_events
               WHERE brand_id = %s AND type = 'payment' AND lead_id IS NOT NULL
               GROUP BY lead_id)
    SELECT count(*)::int,
           percentile_cont(0.5)  WITHIN GROUP (ORDER BY extract(epoch FROM p.t - f.t) / 3600)::float,
           percentile_cont(0.75) WITHIN GROUP (ORDER BY extract(epoch FROM p.t - f.t) / 3600)::float
    FROM f JOIN p USING (lead_id)
    WHERE p.t >= f.t
"""


def load_lead_context(lead_id: str, conn=None) -> Optional[dict]:
    """Everything the rules need about one lead. None when the lead does not
    exist; DataUnavailable when the database cannot be read."""
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            cur.execute(_SQL_LEAD, (lead_id,))
            row = cur.fetchone()
            if not row:
                return None
            lead = {
                "id": str(row[0]), "brand_id": str(row[1]) if row[1] else None,
                "full_name": row[2], "status": row[3], "created_at": row[4],
                "designation": row[5],
            }
            brand = {"id": lead["brand_id"], "name": row[6], "timezone": row[7],
                     "currency": row[8], "brand_brain_id": row[9]}

            cur.execute(_SQL_TOUCHPOINTS, (lead_id,))
            touchpoints = [{
                "type": r[0], "occurred_at": r[1], "created_at": r[2], "source": r[3],
                "channel": r[4], "value": r[5], "currency": r[6],
                "product_code": r[7], "synthetic": bool(r[8]),
            } for r in cur.fetchall()]

            cur.execute(_SQL_CALLS, (lead_id,))
            calls = [{
                "occurred_at": r[0], "disposition": r[1], "direction": r[2],
                "callback_at": r[3], "analysis_id": r[4], "analysis_status": r[5],
            } for r in cur.fetchall()]

            cur.execute(_SQL_WHATSAPP, (lead_id,))
            wa = cur.fetchone()
            whatsapp = ({"conversations": int(wa[0]), "last_direction": wa[1],
                         "last_message_at": wa[2], "unread_count": wa[3] or 0}
                        if wa else {"conversations": 0, "last_direction": None,
                                    "last_message_at": None, "unread_count": 0})

            cur.execute(_SQL_PHONE_COUNT, (lead_id,))
            phone_count = int((cur.fetchone() or [0])[0] or 0)
    except Exception as err:  # noqa: BLE001 - driver errors are varied
        raise DataUnavailable(type(err).__name__) from err

    return {"lead": lead, "brand": brand, "touchpoints": touchpoints,
            "calls": calls, "whatsapp": whatsapp, "phone_count": phone_count}


class _TTLCache:
    def __init__(self):
        self._data: dict = {}
        self._lock = threading.Lock()

    def get(self, key, ttl):
        with self._lock:
            hit = self._data.get(key)
            if hit and time.monotonic() - hit[0] < ttl:
                return hit[1]
        return None

    def put(self, key, value):
        with self._lock:
            self._data[key] = (time.monotonic(), value)


_PAYMENTS_CACHE = _TTLCache()
_WINDOWS_CACHE = _TTLCache()


def brand_payments(brand_id: Optional[str], ttl: float = 900, conn=None) -> list[tuple]:
    """(amount, count, product_code) per distinct paid amount for a brand. An empty
    list on any failure - tiers are a refinement, never a reason to fail."""
    if not brand_id:
        return []
    cached = _PAYMENTS_CACHE.get(brand_id, ttl)
    if cached is not None:
        return cached
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            cur.execute(_SQL_BRAND_PAYMENTS, (brand_id,))
            rows = [(float(r[0]), int(r[1]), r[2]) for r in cur.fetchall()]
    except Exception:  # noqa: BLE001
        return []
    _PAYMENTS_CACHE.put(brand_id, rows)
    return rows


def brand_windows(brand_id: Optional[str], ttl: float = 900, conn=None) -> dict:
    """Hours from first form to first payment for this brand's paying leads."""
    empty = {"converters": 0, "p50_hours": None, "p75_hours": None}
    if not brand_id:
        return empty
    cached = _WINDOWS_CACHE.get(brand_id, ttl)
    if cached is not None:
        return cached
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            cur.execute(_SQL_BRAND_WINDOWS, (brand_id, brand_id))
            row = cur.fetchone()
    except Exception:  # noqa: BLE001
        return empty
    result = {"converters": int(row[0] or 0) if row else 0,
              "p50_hours": row[1] if row else None,
              "p75_hours": row[2] if row else None}
    _WINDOWS_CACHE.put(brand_id, result)
    return result
