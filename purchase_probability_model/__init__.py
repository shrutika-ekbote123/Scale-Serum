"""Purchase Probability — production model package (baseline MVP).

Public surface:
    predict_for_lead(lead_id, conn=None) -> dict
    load_artefacts()                     -> dict
    build_features(lead, payload, schema)-> dict

The model is a frozen logistic-regression baseline. It is deliberately versioned
as a single product-level model - `purchase_probability`, status `baseline_mvp`.
Research experiment history lives outside this package.
"""
from .inference import (  # noqa: F401
    UNAVAILABLE_DB_ERROR,
    UNAVAILABLE_INVALID_DATA,
    UNAVAILABLE_LEAD_NOT_FOUND,
    UNAVAILABLE_MODEL_MISSING,
    UNAVAILABLE_NO_FORM_PAYLOAD,
    build_features,
    load_artefacts,
    predict_for_lead,
)

MODEL_NAME = "purchase_probability"
MODEL_STATUS = "baseline_mvp"

__all__ = [
    "predict_for_lead", "load_artefacts", "build_features",
    "MODEL_NAME", "MODEL_STATUS",
    "UNAVAILABLE_LEAD_NOT_FOUND", "UNAVAILABLE_NO_FORM_PAYLOAD",
    "UNAVAILABLE_MODEL_MISSING", "UNAVAILABLE_DB_ERROR", "UNAVAILABLE_INVALID_DATA",
]
