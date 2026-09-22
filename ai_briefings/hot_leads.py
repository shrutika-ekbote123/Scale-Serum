"""Which untouched leads are "hot", by the purchase-probability model.

Not leads.score: on scrumdb that column is written at the moment of purchase
(1,110 of the 1,151 leads scoring 80-100 have already paid), so "hot leads" by
score is a list of buyers. The purchase-probability model scores a lead from
its first admissible form only, which is known the moment the lead arrives.

Candidates are scored in one batch with the same artefacts, feature builder,
calibration and priority bands as GET /api/purchase-probability/{lead_id}, so
a lead counted here as High shows as High on its own card too.
"""
from __future__ import annotations

import math
from typing import Optional

from purchase_probability_model import inference as _pp


def score(candidates: list[dict], priorities: list[str]) -> dict:
    """{"available", "reason", "scored", "hot": [{"lead_id", "probability", "priority"}]}.

    `hot` is sorted most likely first. A candidate whose features cannot be built
    is skipped and counted, never guessed."""
    if not candidates:
        return {"available": True, "reason": None, "scored": 0, "skipped": 0, "hot": []}
    try:
        art = _pp.load_artefacts()
    except Exception:  # noqa: BLE001 - model files missing on this box
        return {"available": False, "reason": "model_artefacts_unavailable",
                "scored": 0, "skipped": 0, "hot": []}

    import numpy as np
    import pandas as pd

    schema = art["schema"]
    rows, ids, skipped = [], [], 0
    for cand in candidates:
        try:
            features = _pp.build_features(
                {"created_at": cand["created_at"], "email": cand.get("email")},
                cand.get("payload") or {}, schema)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        rows.append({k: features[k] for k in schema["features"]})
        ids.append(cand["id"])
    if not rows:
        return {"available": True, "reason": None, "scored": 0, "skipped": skipped, "hot": []}

    raw = art["model"].predict_proba(pd.DataFrame(rows))[:, 1]
    eps = 1e-15
    clipped = np.clip(raw, eps, 1 - eps)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrated = art["calibration"].predict_proba(log_odds)[:, 1]

    wanted = set(priorities)
    hot = []
    for lead_id, prob in zip(ids, calibrated):
        prob = float(prob)
        if not math.isfinite(prob):
            skipped += 1
            continue
        priority = _pp._priority_of(_pp._decile_of(_pp._percentile_of(prob, art)),
                                    art["decile_bands"])
        if priority in wanted:
            hot.append({"lead_id": lead_id, "probability": round(prob, 4),
                        "priority": priority})
    hot.sort(key=lambda h: h["probability"], reverse=True)
    return {"available": True, "reason": None, "scored": len(ids), "skipped": skipped,
            "hot": hot}


def empty(reason: Optional[str] = None) -> dict:
    return {"available": reason is None, "reason": reason, "scored": 0, "skipped": 0, "hot": []}
