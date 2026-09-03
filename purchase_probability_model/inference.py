"""
Purchase Probability — production inference.

Loads the frozen baseline model package and scores a single lead.

DESIGN RULES (these are correctness requirements, not style preferences)
  * Read-only PostgreSQL. This module never writes.
  * Point-in-time safe. Features come from the lead row and the FIRST ADMISSIBLE
    form_submit only, where admissible means the touchpoint row was written at
    lead-creation time and is not backdated. Payment / ad_click / call events are
    never features - the audit showed they are written at or after conversion with
    a backdated occurred_at.
  * Displayed touchpoints and model features are two different things and are
    computed by two different queries. They must never be conflated.
  * No fabrication. If the lead cannot be scored safely the result is UNAVAILABLE,
    which is not the same as a probability of zero.

LAYERS (added after the frozen baseline; see blend.py for the governing rule)
  * `probability` / `purchase_probability` remain EXACTLY the calibrated base
    model output. Nothing below touches them.
  * `engagement` reads admissible click + timeline evidence (behavioural.py).
  * `brand_brain` reads the brand's own onboarding answers from MongoDB, supplied
    by the caller (brand_fit.py) - this module never opens a Mongo connection.
  * `lead_priority` is the ranking signal those two layers produce, explicitly
    marked uncalibrated. Sort leads on it; quote `purchase_probability` as the
    probability.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from . import behavioural as _behavioural
from . import blend as _blend
from . import brand_fit as _brand_fit
from . import explain as _explain

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# Reasons a lead cannot be scored. Stable identifiers - the UI may branch on these.
UNAVAILABLE_LEAD_NOT_FOUND = "lead_not_found"
UNAVAILABLE_NO_FORM_PAYLOAD = "no_admissible_form_payload"
UNAVAILABLE_MODEL_MISSING = "model_artefacts_unavailable"
UNAVAILABLE_DB_ERROR = "database_unavailable"
UNAVAILABLE_INVALID_DATA = "invalid_lead_data"

# Why the lead card's expected value has no number. Separate from the scoring
# reasons above because the card can be fully populated for a lead the model
# cannot score, and a scorable lead can still have nothing to price against.
LTV_NO_PROBABILITY = "probability_unavailable"
LTV_NO_ORDER_HISTORY = "no_paid_orders_for_brand"

_REASON_TEXT = {
    UNAVAILABLE_LEAD_NOT_FOUND: "Lead not found.",
    UNAVAILABLE_NO_FORM_PAYLOAD: (
        "No admissible form submission was recorded for this lead, so the model "
        "has no point-in-time input to score."),
    UNAVAILABLE_MODEL_MISSING: "Model artefacts are not available on this server.",
    UNAVAILABLE_DB_ERROR: "Lead data source is unavailable.",
    UNAVAILABLE_INVALID_DATA: "Lead data could not be interpreted safely.",
}

_ARTEFACTS: Optional[dict] = None
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- artefacts
def load_artefacts(force: bool = False) -> dict:
    """Load and cache the model package. Raises FileNotFoundError if incomplete."""
    global _ARTEFACTS
    if _ARTEFACTS is not None and not force:
        return _ARTEFACTS
    with _LOCK:
        if _ARTEFACTS is not None and not force:
            return _ARTEFACTS
        import joblib
        import numpy as np

        need = ["model.pkl", "calibration.pkl", "feature_schema.json",
                "model_metadata.json", "percentile_reference.json"]
        missing = [f for f in need if not os.path.exists(os.path.join(PKG_DIR, f))]
        if missing:
            raise FileNotFoundError(f"model package incomplete, missing: {missing}")

        schema = json.load(open(os.path.join(PKG_DIR, "feature_schema.json"), encoding="utf-8"))
        ref = json.load(open(os.path.join(PKG_DIR, "percentile_reference.json"), encoding="utf-8"))
        meta = json.load(open(os.path.join(PKG_DIR, "model_metadata.json"), encoding="utf-8"))

        # Optional on purpose. The engagement / brand-brain layers are an addition
        # to the frozen baseline, not a prerequisite for it: if their config is
        # missing the base model still scores and the layers report unavailable.
        signal_cfg = None
        signal_path = os.path.join(PKG_DIR, "signal_config.json")
        if os.path.exists(signal_path):
            try:
                signal_cfg = json.load(open(signal_path, encoding="utf-8"))
            except Exception:
                signal_cfg = None

        _ARTEFACTS = {
            "model": joblib.load(os.path.join(PKG_DIR, "model.pkl")),
            "calibration": joblib.load(os.path.join(PKG_DIR, "calibration.pkl")),
            "schema": schema,
            "metadata": meta,
            "percentile_grid": np.asarray(ref["percentile_grid"], dtype=float),
            "percentile_quantiles": np.asarray(ref["quantiles"], dtype=float),
            "decile_bands": ref["decile_bands"],
            "placeholders": set(schema["company_placeholder_values"]),
            "signal_config": signal_cfg,
        }
        return _ARTEFACTS


# --------------------------------------------------------------------------- database
#
# The database is remote, so a TCP+TLS handshake costs ~3s. Opening one per request
# does not survive concurrency: 20 simultaneous requests queue past connect_timeout
# and legitimate leads degrade to "Unavailable". A pool pays the handshake once and
# hands the same connections back out.
_POOL = None
_POOL_LOCK = threading.Lock()


def _conn_kwargs() -> dict:
    return dict(
        host=os.environ["DB_HOST"], port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        connect_timeout=int(os.environ.get("DB_CONNECT_TIMEOUT", 10)),
    )


def _make_read_only(conn):
    """Applied to every pooled connection: this module cannot write."""
    conn.read_only = True
    return conn


def _get_pool():
    """Lazy pool. Returns None if psycopg_pool is unavailable, so the module still
    works (one connection per call) rather than failing to import."""
    global _POOL
    if _POOL is not None:
        return _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            return _POOL
        try:
            from psycopg_pool import ConnectionPool
        except ImportError:
            return None
        _POOL = ConnectionPool(
            kwargs=_conn_kwargs(),
            min_size=int(os.environ.get("DB_POOL_MIN", 1)),
            max_size=int(os.environ.get("DB_POOL_MAX", 10)),
            timeout=float(os.environ.get("DB_POOL_TIMEOUT", 20)),
            max_idle=float(os.environ.get("DB_POOL_MAX_IDLE", 300)),
            configure=_make_read_only,
            open=True,
            name="purchase_probability",
        )
        return _POOL


def _connect():
    """Single unpooled connection. Used only when no pool is available."""
    import psycopg
    conn = psycopg.connect(**_conn_kwargs())
    conn.read_only = True          # hard guarantee: this module cannot write
    return conn


@contextmanager
def _borrow(conn=None):
    """Yield a usable connection.

    Caller-supplied connections are used as-is and never closed. Otherwise a pooled
    connection is borrowed and returned, falling back to a fresh connection when no
    pool exists.
    """
    if conn is not None:
        yield conn
        return
    pool = _get_pool()
    if pool is not None:
        with pool.connection() as pooled:
            yield pooled
        return
    own = _connect()
    try:
        yield own
    finally:
        try:
            own.close()
        except Exception:
            pass


_SQL_LEAD = """
    SELECT id, created_at, email, brand_id
    FROM leads
    WHERE id = %s
"""

# CRM card state. A SEPARATE query on the same row, deliberately: leads.score,
# temperature, status, revenue, touchpoint_count and time_to_convert_days are all
# written or mutated at payment time, so they leak the outcome. Keeping them out
# of _SQL_LEAD means they cannot reach build_features by accident - the same
# separation the display-touchpoint query above exists to enforce. Safe to SHOW,
# unsafe to LEARN FROM.
_SQL_LEAD_CARD = """
    SELECT created_at, score, temperature::text, status::text, stage, source,
           revenue, converted_at, time_to_convert_days, touchpoint_count
    FROM leads
    WHERE id = %s
"""

_CARD_COLUMNS = ("created_at", "score", "temperature", "status", "stage", "source",
                 "revenue", "converted_at", "time_to_convert_days", "touchpoint_count")

# The admissibility rule, identical to the frozen training query:
#   written at lead-creation time  AND  not backdated.
_SQL_ADMISSIBLE_FORM = """
    SELECT te.payload::jsonb AS payload, te.occurred_at, te.created_at
    FROM touchpoint_events te
    WHERE te.lead_id = %s
      AND te.type = 'form_submit'
      AND te.created_at <= %s + INTERVAL '1 hour'
      AND te.created_at - te.occurred_at <= INTERVAL '1 hour'
    ORDER BY te.occurred_at ASC, te.created_at ASC
    LIMIT 1
"""

# Display only. Deliberately unfiltered: the user is entitled to see the real
# history. None of this reaches the model.
_SQL_TOUCHPOINTS_DISPLAY = """
    SELECT te.type::text AS type, te.occurred_at, te.created_at,
           te.channel, te.provider, te.source,
           te.utm_source, te.utm_medium, te.utm_campaign,
           te.value, te.currency
    FROM touchpoint_events te
    WHERE te.lead_id = %s
    ORDER BY te.occurred_at ASC, te.created_at ASC
"""

# Which Brand Brain document belongs to this lead's brand. The document itself
# lives in MongoDB and is fetched by the caller (see app.py) - keeping the Mongo
# round trip in the async layer avoids a second driver and a second connection
# pool inside this package.
_SQL_BRAND_BRAIN_REF = """
    SELECT l.brand_id, b.name, b.brand_brain_id
    FROM leads l
    LEFT JOIN brands b ON b.id = l.brand_id
    WHERE l.id = %s
"""


# Brand-level paid-order statistics. Used only to express the lifetime-value
# estimate in money, and aggregated over the whole brand, so it carries no
# information about this lead's own outcome.
_SQL_BRAND_ORDER_STATS = """
    SELECT count(*)::bigint,
           avg(te.value)::float,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY te.value)::float,
           mode() WITHIN GROUP (ORDER BY te.currency)
    FROM touchpoint_events te
    WHERE te.brand_id = %s
      AND te.type = 'payment'
      AND te.value IS NOT NULL
      AND te.value > 0
"""

# One aggregate over a brand's whole payment history is the same answer for every
# lead in that brand and moves slowly, so it is cached rather than re-run per
# request. A stale order value shifts an estimate by a few rupees; a scan per
# lead-detail page view is a real cost.
_ORDER_STATS_TTL = float(os.environ.get("PP_ORDER_STATS_TTL_SECONDS", 900))
_ORDER_STATS: dict = {}
_ORDER_STATS_LOCK = threading.Lock()


def _brand_order_stats(cur, brand_id) -> dict:
    """Mean / median paid order value for a brand. `{}` when there is nothing to
    price against - a brand with no payments is not an error."""
    if not brand_id:
        return {}
    key = str(brand_id)
    now = time.monotonic()
    with _ORDER_STATS_LOCK:
        hit = _ORDER_STATS.get(key)
        if hit and now - hit[0] < _ORDER_STATS_TTL:
            return hit[1]
    try:
        cur.execute(_SQL_BRAND_ORDER_STATS, (brand_id,))
        row = cur.fetchone()
    except Exception:
        # Never fail a prediction over a presentational aggregate.
        return {}
    stats: dict = {}
    if row and row[0]:
        stats = {
            "order_count": int(row[0]),
            "average_order_value": round(float(row[1]), 2) if row[1] is not None else None,
            "median_order_value": round(float(row[2]), 2) if row[2] is not None else None,
            "currency": row[3],
        }
    with _ORDER_STATS_LOCK:
        _ORDER_STATS[key] = (now, stats)
    return stats


def resolve_brand_brain_ref(lead_id: str, conn=None) -> dict:
    """Look up the Brand Brain id for a lead's brand.

    Cheap indexed lookup, safe to call before scoring. Returns a well-formed dict
    even when the lead, the brand or the link is missing - a brand that has not
    completed onboarding simply has no brand_brain_id, which is not an error.
    """
    result = {"lead_id": str(lead_id), "brand_id": None,
              "brand_name": None, "brand_brain_id": None}
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            cur.execute(_SQL_BRAND_BRAIN_REF, (lead_id,))
            row = cur.fetchone()
    except Exception:
        return result
    if row:
        result["brand_id"] = str(row[0]) if row[0] else None
        result["brand_name"] = row[1]
        result["brand_brain_id"] = row[2]
    return result


# --------------------------------------------------------------------------- features
def _norm_company(raw: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (raw or "").strip()).lower()


def build_features(lead: dict, payload: dict, schema: dict) -> dict:
    """Reproduce the frozen V1 seven-feature construction exactly.

    Mirrors final_trainset_v3.sql. Point-in-time values come from the immutable
    form payload; leads.email is used only as a fallback for the email class,
    which the audit verified is unmutated on 157/157 positives.
    """
    contact = ((payload or {}).get("data") or {}).get("contact") or {}

    def _s(v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    pit_email = _s(contact.get("email"))
    pit_company = _s(contact.get("company"))
    pit_jobtitle = _s(contact.get("jobTitle"))
    pit_locale = _s(contact.get("locale"))
    eff_email = pit_email or _s(lead.get("email"))

    rx = schema["seniority_regex"]
    if pit_jobtitle is None:
        seniority = "other"
    elif re.search(rx["other_student"], pit_jobtitle, re.I):
        seniority = "other"
    elif re.search(rx["founder_c_level"], pit_jobtitle, re.I):
        seniority = "founder_c_level"
    elif re.search(rx["vp_director"], pit_jobtitle, re.I):
        seniority = "vp_director"
    elif re.search(rx["manager"], pit_jobtitle, re.I):
        seniority = "manager"
    else:
        seniority = "individual_contributor"

    erx = schema["email_regex"]
    if eff_email is None:
        email_class = "other_freemail"
    elif re.search(erx["gmail"], eff_email, re.I):
        email_class = "gmail"
    elif re.search(erx["other_freemail"], eff_email, re.I):
        email_class = "other_freemail"
    else:
        email_class = "corporate"

    locale = pit_locale if pit_locale in schema["locale_keep"] else "other"

    clip = float(schema["constants"]["company_len_p99_clip"])
    company_len = float(min(len(pit_company or ""), clip))

    placeholder = int(_norm_company(pit_company) in set(schema["company_placeholder_values"]))

    t0 = lead["created_at"]
    hour = t0.hour
    hour_sin = round(math.sin(2 * math.pi * hour / 24), 6)
    hour_cos = round(math.cos(2 * math.pi * hour / 24), 6)

    return {
        "f_seniority": seniority,
        "f_email_class": email_class,
        "f_locale": locale,
        "f_company_len": company_len,
        "f_company_is_placeholder": float(placeholder),
        "f_hour_sin": hour_sin,
        "f_hour_cos": hour_cos,
    }


# --------------------------------------------------------------------------- scoring
def _percentile_of(prob: float, art: dict) -> float:
    import numpy as np
    q, g = art["percentile_quantiles"], art["percentile_grid"]
    idx = int(np.searchsorted(q, prob, side="right"))
    idx = max(0, min(idx, len(g) - 1))
    return float(g[idx])


def _decile_of(percentile: float) -> int:
    return int(min(10, max(1, math.ceil(percentile / 10.0) if percentile > 0 else 1)))


def _priority_of(decile: int, bands: dict) -> str:
    for name, ds in bands.items():
        if decile in ds:
            return name
    return "Low"


def _contributions(features: dict, art: dict) -> list[dict]:
    """Signed contribution to the model's log-odds: transformed value x coefficient.

    One-hot columns of the same source feature are summed. The hour sine/cosine
    pair is summed into a single factor - they are one feature in two columns and
    are meaningless read apart.
    """
    import pandas as pd

    schema = art["schema"]
    pre = art["model"].named_steps["pre"]
    coefs = art["model"].named_steps["clf"].coef_[0]
    row = pd.DataFrame([{k: features[k] for k in schema["features"]}])
    x = pre.transform(row)[0]

    grouped: dict[str, float] = {}
    for col, val, coef in zip(schema["design_columns"], x, coefs):
        src = schema["design_column_to_feature"][col]
        if src in ("f_hour_sin", "f_hour_cos"):
            src = "f_hour_time_pattern"
        grouped[src] = grouped.get(src, 0.0) + float(val) * float(coef)

    labels = schema["labels"]
    out = []
    for src, contrib in grouped.items():
        if abs(contrib) < 1e-12:
            continue
        out.append({
            "feature": src,
            "label": labels.get(src, src),
            "value": features.get(src) if src != "f_hour_time_pattern" else None,
            "direction": "positive" if contrib > 0 else "negative",
            "contribution": round(contrib, 6),
            "explanation": ("Contributed positively to this prediction."
                            if contrib > 0 else
                            "Contributed negatively to this prediction."),
        })
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return out


# --------------------------------------------------------------------------- layers
SIGNAL_CONFIG_MISSING = "signal_config_unavailable"

# The base model could not score the lead, so the layers were never run. This is
# reported instead of the base reason, because "no_admissible_form_payload" is not
# a fact about the Brand Brain and must not be shown as one.
LAYERS_NOT_RUN = "base_model_unavailable"

_LAYER_KEYS = ("e_ad_click_arrival", "e_click_volume", "e_high_intent_pages",
               "e_page_view_volume", "e_session_return", "e_recency",
               "e_engagement_velocity", "b_icp_seniority_fit", "b_channel_fit",
               "b_business_type_email_fit", "b_language_locale_fit",
               "b_sales_cycle_timing")


def _empty_layers(behaviour: Optional[dict] = None,
                  reason: str = SIGNAL_CONFIG_MISSING) -> dict:
    """Layer blocks for responses that could not run the layers at all.

    The shape is identical to a live response so the UI never has to branch on
    whether the keys exist - only on `available`.
    """
    # A layer reports why the LAYER has nothing to say. Passing the base model's
    # failure reason through would claim, for instance, that a brand's Brand Brain
    # is unavailable because the lead had no admissible form payload - two
    # unrelated things.
    if reason == SIGNAL_CONFIG_MISSING:
        layer_reason = reason
        message = ("Signal configuration is not available on this server, so no "
                   "behavioural or brand context was applied.")
    else:
        layer_reason = LAYERS_NOT_RUN
        message = ("The base model could not score this lead, so the ranking layers "
                   "were not applied. "
                   + _REASON_TEXT.get(reason, "The lead could not be scored."))
    behaviour = behaviour or {}
    return {
        "engagement": {
            "available": False,
            # Behaviour collection runs independently of scoring, so when it did
            # run its own finding is the more informative answer.
            "reason": behaviour.get("reason") or layer_reason,
            "message": behaviour.get("message") or message,
            "observed": behaviour.get("observed"),
            "channel": behaviour.get("channel"),
            "window": behaviour.get("window"),
            "timeline": behaviour.get("timeline", []),
            "sources": behaviour.get("sources"),
            "recency_half_life_hours": None,
            "factors": [],
            "total_applied": 0.0, "total_raw": 0.0, "clamped": False, "bounds": None,
        },
        "brand_brain": {
            "available": False, "reason": layer_reason, "message": message,
            "brand_brain_id": None, "profile": None, "factors": [],
            "total_applied": 0.0, "total_raw": 0.0, "clamped": False, "bounds": None,
        },
        "lead_priority": {
            "probability": None, "probability_percent": None, "score": None,
            "percentile": None, "decile": None, "priority": "Unavailable",
            "calibrated": False, "layers_applied": [],
            "basis": "no layers applied",
            "log_odds": None,
        },
        "ranking_factors": [],
        "signal_version": None,
    }


def _tagged(factors: list, layer: str, basis: str) -> list:
    """Stamp provenance onto factors so a reader can tell measured from prior."""
    out = []
    for f in factors:
        g = dict(f)
        g.setdefault("layer", layer)
        g.setdefault("basis", basis)
        g.setdefault("status", "observed")
        out.append(g)
    return out


def _build_layers(art: dict, lead: dict, features: dict, base_prob: float,
                  behaviour: Optional[dict], brand_brain: Optional[dict],
                  model_factors: list, now: datetime) -> dict:
    """Run the engagement + brand-brain layers and assemble the ranking signal."""
    cfg = art.get("signal_config")
    if not cfg:
        return _empty_layers(behaviour)

    profile = _brand_fit.parse_brand_profile(brand_brain, cfg)
    eng_factors, half_life = _blend.engagement_factors(behaviour, profile, cfg)
    brand_factors = _brand_fit.brand_fit_factors(profile, features, behaviour or {},
                                                 lead, cfg, now)
    combo = _blend.combine_layers(base_prob, eng_factors, brand_factors, cfg)

    adjusted_p = combo["adjusted"]["probability"]
    percentile = _percentile_of(adjusted_p, art)
    decile = _decile_of(percentile)

    behaviour = behaviour or {}
    eng_available = bool(behaviour.get("available"))
    applied = []
    if eng_available:
        applied.append("engagement")
    if profile.get("available"):
        applied.append("brand_brain")

    ranking = (_tagged(model_factors, "model", "calibrated_model")
               + eng_factors + brand_factors)
    ranking.sort(key=lambda d: abs(float(d.get("contribution") or 0.0)), reverse=True)

    return {
        "engagement": {
            "available": eng_available,
            "reason": behaviour.get("reason"),
            "message": behaviour.get("message"),
            "observed": behaviour.get("observed"),
            "channel": behaviour.get("channel"),
            "channel_raw": behaviour.get("channel_raw"),
            "window": behaviour.get("window"),
            "timeline": behaviour.get("timeline", []),
            "sources": behaviour.get("sources"),
            "recency_half_life_hours": half_life,
            "factors": eng_factors,
            "total_raw": combo["engagement"]["total_raw"],
            "total_applied": combo["engagement"]["total_applied"],
            "clamped": combo["engagement"]["clamped"],
            "bounds": combo["engagement"]["bounds"],
        },
        "brand_brain": {
            "available": bool(profile.get("available")),
            "reason": profile.get("reason"),
            "message": profile.get("message"),
            "brand_brain_id": profile.get("brand_brain_id"),
            "profile": profile if profile.get("available") else None,
            "factors": brand_factors,
            "total_raw": combo["brand_brain"]["total_raw"],
            "total_applied": combo["brand_brain"]["total_applied"],
            "clamped": combo["brand_brain"]["clamped"],
            "bounds": combo["brand_brain"]["bounds"],
        },
        "lead_priority": {
            # The number to SORT ON. Deliberately not called a probability of
            # purchase in the UI: it is the calibrated base moved by bounded,
            # unfitted priors, so it ranks honestly but does not calibrate.
            "probability": adjusted_p,
            "probability_percent": round(adjusted_p * 100.0, 2),
            "score": int(round(percentile)),
            "percentile": int(round(percentile)),
            "decile": decile,
            "priority": _priority_of(decile, art["decile_bands"]),
            "calibrated": False,
            "layers_applied": applied,
            "basis": ("calibrated base model, adjusted by bounded heuristic "
                      "engagement and brand-fit priors"),
            "log_odds": {
                "base": combo["base"]["log_odds"],
                "engagement": combo["engagement"]["total_applied"],
                "brand_brain": combo["brand_brain"]["total_applied"],
                "total": combo["adjusted"]["log_odds"],
            },
            "percentile_basis": ("base-model out-of-fold reference distribution; the "
                                 "mapping is monotone, so ranking holds, but the "
                                 "absolute percentile is approximate once adjusted"),
        },
        "ranking_factors": ranking,
        "signal_version": combo["signal_version"],
    }


def _unavailable(lead_id: str, reason: str, touchpoints: Optional[list] = None,
                 behaviour: Optional[dict] = None,
                 summary: Optional[dict] = None) -> dict:
    art_meta = {"name": "purchase_probability", "version": "baseline_mvp",
                "status": "baseline_mvp"}
    tps = touchpoints or []
    # The model has no number, but the lead card is not the model. Whenever the
    # lead row was read the caller passes the real summary in, so history, score,
    # temperature and revenue still render; only the probability is null.
    summary = summary or _empty_lead_summary(reason)
    return {
        "lead_id": lead_id,
        "purchase_probability": None,
        "purchase_probability_percent": None,
        "probability": None,
        "percentile": None,
        "decile": None,
        "priority": "Unavailable",
        "top_factors": [],
        "why": None,
        "model_factors": [],
        "model_features": None,
        "touchpoint_count": len(tps),
        "touchpoints": tps,
        "model": art_meta,
        "availability": {"available": False, "reason": reason,
                         "message": _REASON_TEXT.get(reason, "Unavailable.")},
        "fallback": True,
        "reason": reason,
        "lead_summary": summary,
        **_card_fields(summary),
        # Layers cannot rank a lead the base model could not score, but the blocks
        # are always present so the response shape never changes.
        **_empty_layers(behaviour, reason=reason),
    }


def _row_to_touchpoint(r: dict) -> dict:
    """Only fields that genuinely exist. Nulls are preserved, never invented."""
    tp = {
        "type": r.get("type"),
        "occurred_at": r["occurred_at"].isoformat() if r.get("occurred_at") else None,
        "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
        "channel": r.get("channel"),
        "provider": r.get("provider"),
        "source": r.get("source"),
        "campaign": r.get("utm_campaign"),
        "utm_source": r.get("utm_source"),
        "utm_medium": r.get("utm_medium"),
        "value": float(r["value"]) if r.get("value") is not None else None,
        "currency": r.get("currency"),
    }
    return {k: v for k, v in tp.items() if v is not None} | {
        "type": tp["type"], "occurred_at": tp["occurred_at"]}


# --------------------------------------------------------------------------- lead card
#
# The lead card is CRM state, not model output: score, temperature, realised
# revenue, touchpoint count and days-to-convert are read off the lead row. It is
# populated whenever the lead row could be read, INCLUDING on responses where the
# model has no probability to give - an unscorable lead still has a history.
#
# The one derived number is `lifetime_value`, and it announces itself as one.


def _lifetime_value(probability, order_stats: dict, currency, reason=None,
                    realised=None) -> dict:
    """What this lead is worth, in money.

    Once a lead has actually paid, its value is not a forecast any more, so
    `amount` is the recorded revenue and `estimated` is false. Until then `amount`
    is `probability x the brand's median paid order`: the median, not the mean,
    because order values are long-tailed and one outsized order drags a brand's
    mean far away from what a typical lead is worth. Both are reported so a caller
    that prefers the mean can recompute.

    Three numbers, because a value tile can honestly mean any of them:
      * `amount`           - the headline. Recorded revenue if the lead paid,
                             otherwise the probability-weighted estimate.
      * `expected_amount`  - always the probability-weighted estimate, so a
                             converted lead can still be compared with an open one.
      * `potential_amount` - the order itself, undiscounted. What this lead would
                             be worth IF they converted. Independent of the model,
                             so it survives an unscorable lead.
    The flat `lifetime_value` field mirrors `amount`.

    Null, never a substitute number, when the lead has not paid and there is no
    calibrated probability or no brand order history to price against.
    """
    order_value = order_stats.get("median_order_value")
    out = {
        "amount": None,
        "currency": currency or order_stats.get("currency"),
        "estimated": True,
        "available": False,
        "reason": None,
        "basis": None,
        "expected_amount": None,
        "potential_amount": order_value,
        "potential_basis": ("the brand's median paid order - what this lead would be "
                            "worth if they converted, not discounted by how likely "
                            "that is" if order_value else None),
        "probability_used": probability,
        "order_value_used": order_value,
        "order_value_basis": "median paid order for this brand" if order_value else None,
        "order_count": order_stats.get("order_count"),
        "average_order_value": order_stats.get("average_order_value"),
        "median_order_value": order_value,
    }
    if probability is not None and order_value:
        out["expected_amount"] = round(float(probability) * order_value, 2)

    # A lead that has paid is not a forecast. Report what it actually paid, and
    # keep the estimate alongside rather than replacing one with the other.
    if realised:
        out["available"] = True
        out["estimated"] = False
        out["amount"] = round(float(realised), 2)
        out["basis"] = "revenue actually recorded against this lead"
        return out

    if probability is None:
        out["reason"] = reason or LTV_NO_PROBABILITY
        return out
    if not order_value:
        out["reason"] = LTV_NO_ORDER_HISTORY
        return out
    out["available"] = True
    out["amount"] = out["expected_amount"]
    out["basis"] = ("calibrated purchase probability x the brand's median paid order "
                    "value - an expected value, not recorded revenue")
    return out


def _empty_lead_summary(reason: str, message=None) -> dict:
    """Lead-card block for responses that never read the lead row. Same shape as a
    populated one, so the UI branches on `available` and nothing else."""
    return {
        "available": False,
        "reason": reason,
        "message": message or _REASON_TEXT.get(reason, "Lead details are unavailable."),
        "lead_score": None, "lead_score_max": 100, "temperature": None,
        "status": None, "stage": None, "source": None, "created_at": None,
        "touchpoint_count": None,
        "total_revenue": None, "currency": None, "payment_count": 0,
        "converted": None, "converted_at": None, "days_to_convert": None,
        "lifetime_value": _lifetime_value(None, {}, None, reason),
        "basis": "lead row could not be read",
    }


def _lead_summary(card: dict, touchpoints: list, order_stats: dict,
                  probability=None, reason=None) -> dict:
    """The lead card: CRM score and temperature, realised revenue, expected value.

    `card` is the _SQL_LEAD_CARD row - deliberately not the model's lead dict, so
    that no column in here can be mistaken for a feature.
    """
    payments = [t for t in touchpoints if t.get("type") == "payment"]
    paid = [float(t["value"]) for t in payments if t.get("value") is not None]

    # leads.revenue is the CRM's own total; the lead's payment touchpoints are the
    # fallback when it has not been written. 0.0 with no payments means "nothing
    # yet" and is not the same as null - the card renders both as a dash, but only
    # one of the two is a measurement.
    revenue = card.get("revenue")
    revenue = float(revenue) if revenue is not None else None
    if not revenue and paid:
        revenue = round(sum(paid), 2)

    currency = (next((t.get("currency") for t in payments if t.get("currency")), None)
                or order_stats.get("currency"))
    converted_at = card.get("converted_at")
    created_at = card.get("created_at")

    return {
        "available": True,
        "reason": None,
        "message": None,
        # The CRM's own engagement-quality score - the number beside the
        # temperature on the card badge. It is NOT the model output and must never
        # be rendered as a purchase probability.
        "lead_score": int(card["score"]) if card.get("score") is not None else None,
        "lead_score_max": 100,
        "temperature": card.get("temperature"),
        "status": card.get("status"),
        "stage": card.get("stage"),
        "source": card.get("source"),
        "created_at": created_at.isoformat() if created_at else None,
        # The CRM's counter, which is what the card shows. `touchpoint_count` at
        # the top level of the response is the length of the displayed history and
        # can legitimately differ from it.
        "touchpoint_count": (int(card["touchpoint_count"])
                             if card.get("touchpoint_count") is not None
                             else len(touchpoints)),
        "total_revenue": revenue,
        "currency": currency,
        "payment_count": len(payments),
        "converted": bool(converted_at) or bool(revenue),
        "converted_at": converted_at.isoformat() if converted_at else None,
        # Null when the lead has not converted. Do not render that as 0 - zero days
        # to convert is a same-day purchase, which is a very different fact.
        "days_to_convert": (int(card["time_to_convert_days"])
                            if card.get("time_to_convert_days") is not None else None),
        "lifetime_value": _lifetime_value(probability, order_stats, currency,
                                          reason, revenue),
        "basis": ("leads table plus this lead's payment touchpoints - CRM display "
                  "state, mutated at payment time and therefore never a model input"),
    }


def _card_fields(summary: dict) -> dict:
    """The scalars the lead card puts in its header tiles, mirrored to the top
    level. `lead_summary` stays authoritative and carries the provenance."""
    return {
        "lead_score": summary.get("lead_score"),
        "temperature": summary.get("temperature"),
        "total_revenue": summary.get("total_revenue"),
        "days_to_convert": summary.get("days_to_convert"),
        "lifetime_value": (summary.get("lifetime_value") or {}).get("amount"),
    }


def predict_for_lead(lead_id: str, conn=None, brand_brain: Optional[dict] = None,
                     now: Optional[datetime] = None) -> dict:
    """Score one lead. Never raises for expected conditions - returns UNAVAILABLE.

    `brand_brain` is the MongoDB Brand Brain document for this lead's brand, if
    the caller has one (see `resolve_brand_brain_ref`). It is optional: without it
    the brand-fit layer reports `no_brand_brain` and contributes nothing, and the
    calibrated probability is unchanged either way.

    `now` fixes the clock for every time-based factor in one place, so recency and
    lead-age can never disagree by a few milliseconds within a single response.
    """
    try:
        art = load_artefacts()
    except Exception:
        return _unavailable(lead_id, UNAVAILABLE_MODEL_MISSING)

    now = now or datetime.now(timezone.utc)
    cfg = art.get("signal_config")
    touchpoints: list[dict] = []
    behaviour: Optional[dict] = None
    lead: Optional[dict] = None
    card: dict = {}
    order_stats: dict = {}
    lead_missing = False
    try:
        with _borrow(conn) as db, db.cursor() as cur:
            cur.execute(_SQL_LEAD, (lead_id,))
            row = cur.fetchone()
            if row is None:
                lead_missing = True
            else:
                lead = {"id": row[0], "created_at": row[1],
                        "email": row[2], "brand_id": row[3]}

                # ---- CARD state: CRM display columns, in their own dict so a
                # post-conversion value can never be handed to build_features.
                cur.execute(_SQL_LEAD_CARD, (lead_id,))
                card_row = cur.fetchone()
                card = dict(zip(_CARD_COLUMNS, card_row)) if card_row else {}

                # ---- DISPLAY history (never a model input) ----
                cur.execute(_SQL_TOUCHPOINTS_DISPLAY, (lead_id,))
                cols = [d[0] for d in cur.description]
                touchpoints = [_row_to_touchpoint(dict(zip(cols, r))) for r in cur.fetchall()]

                # ---- CARD input: what a lead is worth in this brand. Brand-level,
                # cached, and never a model feature.
                order_stats = _brand_order_stats(cur, lead.get("brand_id"))

                # ---- MODEL input: first admissible form_submit only ----
                cur.execute(_SQL_ADMISSIBLE_FORM, (lead_id, lead["created_at"]))
                form = cur.fetchone()

                # ---- LAYER input: admissible clicks + timeline. Separate query,
                # separate admissibility rule, and never a base-model feature.
                if cfg:
                    try:
                        behaviour = _behavioural.collect_behaviour(cur, lead, cfg, now)
                    except Exception:
                        behaviour = None
    except Exception:
        # The lead row may already be in hand when a later query failed; show what
        # was actually read rather than blanking the card as well.
        return _unavailable(
            lead_id, UNAVAILABLE_DB_ERROR, touchpoints, behaviour,
            _lead_summary(card, touchpoints, order_stats, None, UNAVAILABLE_DB_ERROR)
            if card else None)

    if lead_missing:
        return _unavailable(lead_id, UNAVAILABLE_LEAD_NOT_FOUND)

    if form is None:
        return _unavailable(
            lead_id, UNAVAILABLE_NO_FORM_PAYLOAD, touchpoints, behaviour,
            _lead_summary(card, touchpoints, order_stats, None,
                          UNAVAILABLE_NO_FORM_PAYLOAD))

    payload, form_occurred_at, form_created_at = form[0], form[1], form[2]
    bad_data = _lead_summary(card, touchpoints, order_stats, None,
                             UNAVAILABLE_INVALID_DATA)
    try:
        features = build_features(lead, payload or {}, art["schema"])
        prob = _score(features, art)
    except Exception:
        return _unavailable(lead_id, UNAVAILABLE_INVALID_DATA, touchpoints, behaviour,
                            bad_data)

    if prob is None or not math.isfinite(prob) or not (0.0 <= prob <= 1.0):
        return _unavailable(lead_id, UNAVAILABLE_INVALID_DATA, touchpoints, behaviour,
                            bad_data)

    percentile = _percentile_of(prob, art)
    decile = _decile_of(percentile)
    model_factors = _contributions(features, art)

    # Engagement + brand fit. These produce `lead_priority`; they do not and must
    # not alter `probability` / `purchase_probability` below.
    try:
        layers = _build_layers(art, lead, features, prob, behaviour, brand_brain,
                               model_factors, now)
    except Exception:
        layers = _empty_layers(behaviour)

    summary = _lead_summary(card, touchpoints, order_stats, prob)

    # The lead card's factor list: base-model factors plus the touchpoint-derived
    # engagement factors and the brand-fit factors, rendered in plain language.
    # Each row carries `affects`, because only the base-model rows sum to
    # `purchase_probability` - see explain.py for why that field is not optional.
    top_factors = _explain.merge_top_factors(
        model_factors,
        (layers.get("engagement") or {}).get("factors"),
        (layers.get("brand_brain") or {}).get("factors"),
        features,
        (layers.get("engagement") or {}).get("observed"),
    )

    return {
        "lead_id": str(lead_id),
        # Calibrated probability expressed as a percentage. 2.3 means 2.3%.
        # This is the real model output - never rescaled for presentation.
        "purchase_probability": round(prob * 100.0, 2),
        "purchase_probability_percent": int(round(prob * 100.0)),
        "probability": round(prob, 8),
        # Relative ranking against the frozen out-of-fold reference distribution.
        "percentile": int(round(percentile)),
        "decile": decile,
        "priority": _priority_of(decile, art["decile_bands"]),
        # Plain-language, all layers, each row tagged with what it moves. The
        # untouched base-model-only list is still available as `model_factors`
        # below, and the raw merged list as `ranking_factors`.
        "top_factors": top_factors,
        "why": _explain.baseline_block(
            art["metadata"], art["schema"].get("constants"), prob),
        # The calibrated explanation on its own: exactly the factors that sum, in
        # log-odds, to `probability`. Unchanged by the presentation layer, and by
        # the brand brain. Use this, not `top_factors`, for any arithmetic.
        "model_factors": model_factors,
        "model_features": {
            "feature_version": art["schema"]["feature_version"],
            "values": features,
            "source": "first admissible form_submit (written at lead creation, not backdated)",
            "form_occurred_at": form_occurred_at.isoformat() if form_occurred_at else None,
            "form_created_at": form_created_at.isoformat() if form_created_at else None,
        },
        "touchpoint_count": len(touchpoints),
        "touchpoints": touchpoints,
        "model": {
            "name": art["metadata"]["model_name"],
            "version": art["metadata"]["version"],
            "status": art["metadata"]["model_status"],
            "feature_version": art["metadata"]["feature_version"],
            "description": art["metadata"]["description"],
        },
        "availability": {"available": True, "reason": None, "message": None},
        "fallback": False,
        "scored_at": now.isoformat(),
        # CRM lead card. `lead_summary.lead_score` is the CRM's engagement score
        # and has nothing to do with `purchase_probability` above - the two are
        # displayed side by side and must never be substituted for one another.
        "lead_summary": summary,
        **_card_fields(summary),
        # `engagement`, `brand_brain`, `lead_priority`, `ranking_factors`,
        # `signal_version`. Sort on lead_priority; quote purchase_probability.
        **layers,
    }


def _score(features: dict, art: dict) -> float:
    import numpy as np
    import pandas as pd
    schema = art["schema"]
    row = pd.DataFrame([{k: features[k] for k in schema["features"]}])
    raw = float(art["model"].predict_proba(row)[0, 1])
    eps = 1e-15
    lo = math.log(min(max(raw, eps), 1 - eps) / min(max(1 - raw, eps), 1 - eps))
    cal = art["calibration"].predict_proba(np.array([[lo]]))[0, 1]
    return float(cal)
