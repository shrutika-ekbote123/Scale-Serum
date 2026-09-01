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
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from contextlib import contextmanager
from typing import Any, Optional

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# Reasons a lead cannot be scored. Stable identifiers - the UI may branch on these.
UNAVAILABLE_LEAD_NOT_FOUND = "lead_not_found"
UNAVAILABLE_NO_FORM_PAYLOAD = "no_admissible_form_payload"
UNAVAILABLE_MODEL_MISSING = "model_artefacts_unavailable"
UNAVAILABLE_DB_ERROR = "database_unavailable"
UNAVAILABLE_INVALID_DATA = "invalid_lead_data"

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
        _ARTEFACTS = {
            "model": joblib.load(os.path.join(PKG_DIR, "model.pkl")),
            "calibration": joblib.load(os.path.join(PKG_DIR, "calibration.pkl")),
            "schema": schema,
            "metadata": meta,
            "percentile_grid": np.asarray(ref["percentile_grid"], dtype=float),
            "percentile_quantiles": np.asarray(ref["quantiles"], dtype=float),
            "decile_bands": ref["decile_bands"],
            "placeholders": set(schema["company_placeholder_values"]),
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


def _unavailable(lead_id: str, reason: str, touchpoints: Optional[list] = None) -> dict:
    art_meta = {"name": "purchase_probability", "version": "baseline_mvp",
                "status": "baseline_mvp"}
    tps = touchpoints or []
    return {
        "lead_id": lead_id,
        "purchase_probability": None,
        "purchase_probability_percent": None,
        "probability": None,
        "percentile": None,
        "decile": None,
        "priority": "Unavailable",
        "top_factors": [],
        "model_features": None,
        "touchpoint_count": len(tps),
        "touchpoints": tps,
        "model": art_meta,
        "availability": {"available": False, "reason": reason,
                         "message": _REASON_TEXT.get(reason, "Unavailable.")},
        "fallback": True,
        "reason": reason,
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


def predict_for_lead(lead_id: str, conn=None) -> dict:
    """Score one lead. Never raises for expected conditions - returns UNAVAILABLE."""
    try:
        art = load_artefacts()
    except Exception:
        return _unavailable(lead_id, UNAVAILABLE_MODEL_MISSING)

    touchpoints: list[dict] = []
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

                # ---- DISPLAY history (never a model input) ----
                cur.execute(_SQL_TOUCHPOINTS_DISPLAY, (lead_id,))
                cols = [d[0] for d in cur.description]
                touchpoints = [_row_to_touchpoint(dict(zip(cols, r))) for r in cur.fetchall()]

                # ---- MODEL input: first admissible form_submit only ----
                cur.execute(_SQL_ADMISSIBLE_FORM, (lead_id, lead["created_at"]))
                form = cur.fetchone()
    except Exception:
        return _unavailable(lead_id, UNAVAILABLE_DB_ERROR, touchpoints)

    if lead_missing:
        return _unavailable(lead_id, UNAVAILABLE_LEAD_NOT_FOUND)

    if form is None:
        return _unavailable(lead_id, UNAVAILABLE_NO_FORM_PAYLOAD, touchpoints)

    payload, form_occurred_at, form_created_at = form[0], form[1], form[2]
    try:
        features = build_features(lead, payload or {}, art["schema"])
        prob = _score(features, art)
    except Exception:
        return _unavailable(lead_id, UNAVAILABLE_INVALID_DATA, touchpoints)

    if prob is None or not math.isfinite(prob) or not (0.0 <= prob <= 1.0):
        return _unavailable(lead_id, UNAVAILABLE_INVALID_DATA, touchpoints)

    percentile = _percentile_of(prob, art)
    decile = _decile_of(percentile)
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
        "top_factors": _contributions(features, art),
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
