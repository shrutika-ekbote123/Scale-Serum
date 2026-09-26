"""Read-only PostgreSQL access for the Creative Coach.

Synchronous psycopg, run in a thread by the caller, borrowing the
purchase-probability pool like AI Briefings does - one DB_POOL_MAX covers every
feature that reads scrumdb. Every connection in that pool is read_only; this
module has no write path and must never grow one. The coach thread is written
by the main backend, not here.

TRAPS THIS MODULE IS WRITTEN AROUND (each one is a wrong answer if ignored)

  * meta_insights holds the SAME spend at campaign, adset and ad level. Only
    entity_type = 'ad' is read here. Summing across levels triples the spend.

  * Derived metrics are recomputed from the summed totals, never averaged out
    of the daily rows. The mean of daily CTRs is not the CTR of the period: a
    day with 8 impressions would weigh as much as a day with 80,000.

  * sl_script_lab_tests.source_ad_id is not always a real ad id - live data
    contains 'mi_named_0', '202', '31241551'. The join is therefore VERIFIED:
    no matching insight rows means no performance tier, rather than silence
    that the caller could mistake for "this ad got no clicks".

  * The JSON columns are camelCase (brandVoiceFit, whyItMatters) while the API
    is snake_case. Normalisation happens here, on read, so nothing downstream
    has to know which convention a given row was written with.

  * ad_number is a small per-brand sequence, NOT the Meta ad id. Versions of
    one ad are matched on (brand_id, ad_number); real ad performance is matched
    on source_ad_id. Confusing the two coaches the wrong creative.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any, Optional

from purchase_probability_model.inference import _borrow


class DataUnavailable(Exception):
    """scrumdb could not be read. Distinct from 'the test does not exist'."""


# --------------------------------------------------------------------- queries
_SQL_TEST = """
    SELECT id, brand_id, ad_number, version, label, ad_name, source_ad_id,
           source_platform, funnel_stage, marketing_angle, script_text,
           score, attention, resonance, creative, conversion,
           marketing_angle_execution, verdict, verdict_band, ai_verdict,
           emotional_angle, context_alignment, section_breakdown, improvements,
           what_worked, what_to_fix, top_recommendation, ai_fallback,
           created_by_user_id, created_at
    FROM sl_script_lab_tests WHERE id = %(id)s
"""

_SQL_BRAND = """
    SELECT id, name, timezone, currency, brand_brain_id FROM brands WHERE id = %(b)s
"""

# Every OTHER brand's name. The coach must never name one of these to this
# brand's user - and it has already tried: on live data the model wrote DI a
# rewrite advertising Lawtorney, because DI's Brand Brain holds Lawtorney's
# content. Without this list the check has nothing to check against.
_SQL_OTHER_BRAND_NAMES = """
    SELECT DISTINCT name FROM brands
    WHERE id <> %(b)s AND name IS NOT NULL AND length(btrim(name)) >= 4
"""

# Ad-level only, and the derived rates are computed from the totals below.
# The brand_id predicate is not decoration: 109 stored tests carry a
# source_ad_id that exists under a DIFFERENT brand. Dropping it would coach one
# brand using another brand's delivery.
_SQL_AD_PERFORMANCE = """
    SELECT sum(spend)::float, sum(impressions)::float, sum(clicks)::float,
           sum(leads)::float, sum(purchases)::float, sum(revenue)::float,
           count(*)::int, min(date), max(date)
    FROM meta_insights
    WHERE brand_id = %(b)s AND entity_type = 'ad' AND entity_id = %(ad)s
      AND date >= %(d0)s AND date <= %(d1)s
"""

# The same totals with no date bound. A script is often tested long after the ad
# it came from stopped running: the recent window is then empty while the ad has
# a full, useful history. "It ran in June and stopped" beats "no data".
_SQL_AD_PERFORMANCE_LIFETIME = """
    SELECT sum(spend)::float, sum(impressions)::float, sum(clicks)::float,
           sum(leads)::float, sum(purchases)::float, sum(revenue)::float,
           count(*)::int, min(date), max(date)
    FROM meta_insights
    WHERE brand_id = %(b)s AND entity_type = 'ad' AND entity_id = %(ad)s
"""

_SQL_AD_META = """
    SELECT ad_id, name, status, effective_status, ad_set_id, campaign_id
    FROM meta_ads WHERE brand_id = %(b)s AND ad_id = %(ad)s
"""

# The brand's own analysed creatives, best and worst, with the copy that ran.
#
# RANKED AND LIMITED IN SQL, NOT IN PYTHON. DI has 431 analysed creatives, every
# one carrying its full script copy. Fetching them all to keep five meant moving
# most of a megabyte across the network on every `improve` turn, and the ranking
# is something the database does better anyway.
#
# DISTINCT ON keeps one row per ad: the analyser overwrites per ad and range, so
# only the newest analysis of each ad counts. Platform is matched when the test
# names one - a Meta script is coached against Meta creatives.
_SQL_BRAND_CREATIVES = """
    WITH latest AS (
        SELECT DISTINCT ON (ad_id)
               ad_id, ad_name, creative_score::float AS score, verdict, format,
               hook_analysis, script_copy, analyzed_at
        FROM mi_ad_creative_analyses
        WHERE brand_id = %(b)s AND script_copy IS NOT NULL AND creative_score IS NOT NULL
          -- platform is a Postgres ENUM, so it is compared as text: an untyped
          -- parameter has no operator against the enum type and the query fails
          -- outright rather than degrading.
          AND (%(platform)s::text IS NULL OR platform::text = %(platform)s::text)
        ORDER BY ad_id, analyzed_at DESC
    )
    (SELECT 'winner' AS kind, * FROM latest ORDER BY score DESC, analyzed_at DESC
     LIMIT %(w)s)
    UNION ALL
    (SELECT 'loser' AS kind, * FROM latest ORDER BY score ASC, analyzed_at DESC
     LIMIT %(l)s)
"""

# Long-running competitor ads are the ones still paying for themselves.
_SQL_COMPETITOR_HOOKS = """
    SELECT ad_archive_id, hook, creative_type, est_run_days, status
    FROM mi_competitor_ads
    WHERE brand_id = %(b)s AND hook IS NOT NULL AND length(btrim(hook)) > 0
    ORDER BY est_run_days DESC NULLS LAST, last_seen_at DESC
    LIMIT %(n)s
"""

# Earlier tests of the SAME ad for this brand - the "scored against its history"
# corpus. Excludes the test being coached.
_SQL_VERSIONS = """
    SELECT id, version, label, score, verdict, verdict_band, marketing_angle,
           funnel_stage, ai_fallback, created_at
    FROM sl_script_lab_tests
    WHERE brand_id = %(b)s AND ad_number = %(n)s AND id <> %(id)s
    ORDER BY created_at DESC
    LIMIT %(lim)s
"""


# ----------------------------------------------------------------- normalising
_ALIASES = {
    # context_alignment
    "brandvoicefit": "brand_voice_fit",
    "funnelstagefit": "funnel_stage_fit",
    "marketinganglefit": "marketing_angle_fit",
    # improvements
    "whyitmatters": "why_it_matters",
    "suggestedrewrite": "suggested_rewrite",
    "metricsimpacted": "metrics_impacted",
}


def _snake(key: str) -> str:
    """camelCase -> snake_case for the handful of keys the JSON columns use.

    A lookup table rather than a regex: the set is small, closed and known, and
    a table cannot surprise us by renaming a key we did not anticipate."""
    flat = key.replace("_", "").lower()
    return _ALIASES.get(flat, key)


def _normalise(value: Any) -> Any:
    """Recursively rename camelCase keys in a JSON column's contents."""
    if isinstance(value, dict):
        return {_snake(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def _json(value: Any, default: Any) -> Any:
    """psycopg returns jsonb already decoded; text columns arrive as strings."""
    if value is None:
        return default
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return default
    return _normalise(value)


def _rows(cur, sql: str, params: dict) -> list:
    cur.execute(sql, params)
    return cur.fetchall()


# --------------------------------------------------------------------- loaders
def load_test(test_id: str, conn=None) -> Optional[dict]:
    """The tested script and its stored review. None when the id is unknown."""
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            row = _rows(cur, _SQL_TEST, {"id": test_id})
    except Exception as err:  # noqa: BLE001 - driver errors are varied
        raise DataUnavailable(type(err).__name__) from err
    if not row:
        return None
    r = row[0]
    return {
        "test_id": str(r[0]), "brand_id": str(r[1]) if r[1] else None,
        "ad_number": r[2], "version": r[3], "label": r[4], "ad_name": r[5],
        "source_ad_id": (r[6] or "").strip() or None, "source_platform": r[7],
        "funnel_stage": r[8], "marketing_angle": r[9], "script_text": r[10] or "",
        "score": r[11], "attention": r[12], "resonance": r[13], "creative": r[14],
        "conversion": r[15], "marketing_angle_execution": r[16],
        "verdict": r[17], "verdict_band": r[18], "ai_verdict": r[19],
        "emotional_angle": _json(r[20], {}),
        "context_alignment": _json(r[21], {}),
        "section_breakdown": _json(r[22], []),
        "improvements": _json(r[23], []),
        "what_worked": _json(r[24], []),
        "what_to_fix": _json(r[25], []),
        "top_recommendation": r[26],
        "ai_fallback": bool(r[27]),
        "created_by_user_id": str(r[28]) if r[28] else None,
        "created_at": r[29],
    }


def load_brand(brand_id: str, conn=None) -> Optional[dict]:
    """The brand row - crucially `brand_brain_id`, the ONLY correct way to find
    a brand's Brand Brain. Several brands share a display name in live data, so
    matching on name coaches against the wrong brand's voice."""
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            row = _rows(cur, _SQL_BRAND, {"b": brand_id})
    except Exception as err:  # noqa: BLE001
        raise DataUnavailable(type(err).__name__) from err
    if not row:
        return None
    r = row[0]
    return {"id": str(r[0]), "name": r[1], "timezone": r[2],
            "currency": r[3] or "INR", "brand_brain_id": r[4]}


def load_other_brand_names(brand_id: str, conn=None) -> list:
    """Names of every brand except this one, for the leak check.

    Names shorter than four characters are excluded by the query: "DI" and "T"
    are real brand names here and would match inside ordinary words. A brand
    that shares a name with this one (there are four called "Lawttorney") is
    dropped too - flagging a brand's own name is a false alarm, and a noisy
    gate is a gate that gets switched off."""
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            rows = _rows(cur, _SQL_OTHER_BRAND_NAMES, {"b": brand_id})
            own = _rows(cur, _SQL_BRAND, {"b": brand_id})
    except Exception as err:  # noqa: BLE001
        raise DataUnavailable(type(err).__name__) from err
    own_name = (own[0][1] or "").strip().lower() if own else ""
    return sorted({(r[0] or "").strip() for r in rows
                   if (r[0] or "").strip().lower() != own_name})


def load_ad_performance(brand_id: str, ad_id: str, days: int = 30,
                        today: Optional[date] = None, conn=None) -> Optional[dict]:
    """Ad-level delivery for one ad over the last `days`.

    When the recent window is empty the ad's whole history is read instead and
    the result is labelled `window: "lifetime"`, so the coach can say when the ad
    actually ran rather than implying it never delivered.

    Returns None when the ad id matches no insight rows at all - the normal
    outcome for a script that was never published, and for the junk ad ids that
    exist in live data. The caller must treat None as "no performance tier" and
    make no claims, rather than reporting zeroes."""
    end = today or date.today()
    start = end - timedelta(days=days - 1)
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            row = _rows(cur, _SQL_AD_PERFORMANCE,
                        {"b": brand_id, "ad": ad_id, "d0": start, "d1": end})[0]
            window = f"{days}d"
            if not row[6]:
                row = _rows(cur, _SQL_AD_PERFORMANCE_LIFETIME,
                            {"b": brand_id, "ad": ad_id})[0]
                window = "lifetime"
            meta = _rows(cur, _SQL_AD_META, {"b": brand_id, "ad": ad_id})
    except Exception as err:  # noqa: BLE001
        raise DataUnavailable(type(err).__name__) from err

    spend, impressions, clicks, leads, purchases, revenue, rows, first, last = row
    if not rows:
        return None

    impressions = impressions or 0.0
    clicks = clicks or 0.0
    spend = spend or 0.0
    # Rates are derived from the totals. Averaging the daily rates would let a
    # day with a handful of impressions count as much as a day with 80,000.
    perf = {
        "ad_id": ad_id,
        "window": window,
        "window_days": days if window != "lifetime" else None,
        "days_with_data": int(rows),
        "first_date": first.isoformat() if first else None,
        "last_date": last.isoformat() if last else None,
        # True when nothing was delivered inside the requested window: the ad
        # has a history but is not currently running.
        "stale": window == "lifetime",
        "spend": round(spend, 2),
        "impressions": int(impressions),
        "clicks": int(clicks),
        "leads": int(leads or 0),
        "purchases": int(purchases or 0),
        "ctr_pct": round(clicks / impressions * 100, 2) if impressions else None,
        "cpm": round(spend / impressions * 1000, 2) if impressions else None,
        "cpc": round(spend / clicks, 2) if clicks else None,
        "cpl": round(spend / leads, 2) if leads else None,
        # revenue is deliberately NOT returned. Payments in scrumdb are written
        # by an attribution backfill and cannot be split by ad, so any per-ad
        # ROAS would be fiction. See AI_BRIEFINGS_API.md.
    }
    if meta:
        m = meta[0]
        perf.update({"ad_name": m[1], "status": m[2], "effective_status": m[3]})
    return perf


def load_brand_creatives(brand_id: str, winners: int = 3, losers: int = 2,
                         platform: Optional[str] = None, conn=None) -> dict:
    """This brand's best and worst analysed creatives, with the copy that ran.

    The coach quotes these back as evidence, so both ends matter: what worked is
    a pattern to copy, what failed is a trap to avoid repeating.

    A brand with only a handful of analysed ads can have the same creative rank
    as both, so winners take precedence and losers are deduplicated against
    them: showing one ad as this brand's best AND worst is a bug the reader
    would notice before we did."""
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            rows = _rows(cur, _SQL_BRAND_CREATIVES,
                         {"b": brand_id, "platform": platform,
                          "w": max(0, winners), "l": max(0, losers)})
    except Exception as err:  # noqa: BLE001
        raise DataUnavailable(type(err).__name__) from err

    out: dict = {"winners": [], "losers": []}
    seen: set = set()
    for row in rows:
        kind = row[0]
        item = {"ad_id": row[1], "ad_name": row[2], "score": row[3], "verdict": row[4],
                "format": row[5], "hook_analysis": row[6], "script_copy": row[7],
                "analyzed_at": row[8].isoformat() if row[8] else None}
        if kind == "winner":
            seen.add(item["ad_id"])
            out["winners"].append(item)
        elif item["ad_id"] not in seen:
            out["losers"].append(item)
    return out


def load_competitor_hooks(brand_id: str, limit: int = 5, conn=None) -> list:
    """Competitor hooks, longest-running first - a proxy for what is working."""
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            rows = _rows(cur, _SQL_COMPETITOR_HOOKS, {"b": brand_id, "n": limit})
    except Exception as err:  # noqa: BLE001
        raise DataUnavailable(type(err).__name__) from err
    return [{"ad_archive_id": r[0], "hook": (r[1] or "").strip(), "creative_type": r[2],
             "est_run_days": r[3], "status": r[4]} for r in rows]


def load_versions(brand_id: str, ad_number, test_id: str, limit: int = 5,
                  conn=None) -> list:
    """Earlier tested versions of the same ad, newest first.

    Matched on (brand_id, ad_number) - the per-brand sequence - not on the Meta
    ad id, because a rewritten script is tested before it is ever published."""
    if ad_number is None:
        return []
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            rows = _rows(cur, _SQL_VERSIONS,
                         {"b": brand_id, "n": ad_number, "id": test_id, "lim": limit})
    except Exception as err:  # noqa: BLE001
        raise DataUnavailable(type(err).__name__) from err
    return [{"test_id": str(r[0]), "version": r[1], "label": r[2], "score": r[3],
             "verdict": r[4], "verdict_band": r[5], "marketing_angle": r[6],
             "funnel_stage": r[7], "ai_fallback": bool(r[8]),
             "created_at": r[9].isoformat() if r[9] else None}
            for r in rows]
