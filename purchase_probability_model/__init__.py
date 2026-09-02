"""Purchase Probability — production model package (baseline MVP + signal layers).

Public surface:
    predict_for_lead(lead_id, conn=None, brand_brain=None, now=None) -> dict
    resolve_brand_brain_ref(lead_id, conn=None)                      -> dict
    load_artefacts()                                                 -> dict
    build_features(lead, payload, schema)                            -> dict

The probability itself is a frozen logistic-regression baseline, versioned as a
single product-level model - `purchase_probability`, status `baseline_mvp`.
Research experiment history lives outside this package.

Layered on top of it, and reported separately so the two are never confused:
    * engagement  - admissible click actions and journey timeline (behavioural.py)
    * brand_brain - the brand's own onboarding answers from MongoDB (brand_fit.py)
    * lead_priority - the ranking signal those layers produce (blend.py)

`probability` / `purchase_probability` are the calibrated model output and are
never modified by the layers. `lead_priority` is uncalibrated by construction and
says so in the response.
"""
from .behavioural import (  # noqa: F401
    NO_BEHAVIOURAL_DATA,
    ONLY_FORM_SUBMISSION,
    collect_behaviour,
)
from .blend import combine_layers, engagement_factors  # noqa: F401
from .brand_fit import (  # noqa: F401
    NO_BRAND_BRAIN,
    brand_fit_factors,
    parse_brand_profile,
    parse_sales_cycle_days,
)
from .inference import (  # noqa: F401
    LAYERS_NOT_RUN,
    LTV_NO_ORDER_HISTORY,
    LTV_NO_PROBABILITY,
    SIGNAL_CONFIG_MISSING,
    UNAVAILABLE_DB_ERROR,
    UNAVAILABLE_INVALID_DATA,
    UNAVAILABLE_LEAD_NOT_FOUND,
    UNAVAILABLE_MODEL_MISSING,
    UNAVAILABLE_NO_FORM_PAYLOAD,
    build_features,
    load_artefacts,
    predict_for_lead,
    resolve_brand_brain_ref,
)

MODEL_NAME = "purchase_probability"
MODEL_STATUS = "baseline_mvp"

__all__ = [
    "predict_for_lead", "resolve_brand_brain_ref", "load_artefacts", "build_features",
    "collect_behaviour", "engagement_factors", "combine_layers",
    "parse_brand_profile", "parse_sales_cycle_days", "brand_fit_factors",
    "MODEL_NAME", "MODEL_STATUS",
    "UNAVAILABLE_LEAD_NOT_FOUND", "UNAVAILABLE_NO_FORM_PAYLOAD",
    "UNAVAILABLE_MODEL_MISSING", "UNAVAILABLE_DB_ERROR", "UNAVAILABLE_INVALID_DATA",
    "NO_BEHAVIOURAL_DATA", "ONLY_FORM_SUBMISSION", "NO_BRAND_BRAIN",
    "SIGNAL_CONFIG_MISSING", "LAYERS_NOT_RUN",
    "LTV_NO_PROBABILITY", "LTV_NO_ORDER_HISTORY",
]
