"""
Request / response models for Vision Lab.

Conventions follow app.py and sales_call_analyzer/models.py: pydantic
BaseModel, almost everything Optional with a safe default, so a partially
filled request still produces a usable analysis rather than a 422. Only
`creative_id` and the creative URL are genuinely required.

Free-string enums on purpose. The house contract is "always return 200 with a
stated reason", not "reject the caller", so values that arrive from the backend
are typed `str` and normalised in code rather than constrained by `Literal`.
The valid values are listed in __init__.py.

The completed report is returned as a plain dict assembled by report.py rather
than a model, matching the Sales Call Analyzer. Its shape is documented in
VISION_LAB_PLAN.md section 4.2 and pinned by tests/test_vision_lab_api.py.
"""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# =========================================================================== #
# Request
# =========================================================================== #
class CreativeRef(BaseModel):
    """Where the creative lives. We read it and never store it.

    The URL is a credential with an expiry - a presigned S3 GET link. It is
    never persisted and never logged, exactly as the Sales Call Analyzer
    treats a call recording.
    """
    url: Optional[str] = None                 # https URL, ideally short-lived and signed
    kind: Optional[str] = None                # "video" | "image" - probed if absent
    mime_type: Optional[str] = None           # e.g. "video/mp4"
    duration_seconds: Optional[float] = None  # a hint; ffprobe decides
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None
    expires_at: Optional[datetime] = None     # signed-URL expiry, diagnostics only


class BrandAssets(BaseModel):
    """What lets us detect the brand on screen.

    Without a wordmark we fall back to matching the brand name in OCR text and
    report `no_brand_assets`, so Brand Memory is still scored but on a weaker
    basis - and the response says which basis was used.
    """
    wordmark_url: Optional[str] = None
    brand_names: list[str] = Field(default_factory=list)


class AnalyzeOptions(BaseModel):
    force_reanalysis: bool = False
    sample_fps: Optional[float] = None    # overrides the configured default
    store_raw_maps: Optional[bool] = None


class AnalyzeRequest(BaseModel):
    creative_id: str                                  # the caller's id - required
    creative: CreativeRef = Field(default_factory=CreativeRef)
    brand_brain_id: Optional[str] = None
    brand_id: Optional[str] = None
    division: Optional[str] = None                    # the Division selector in the UI
    ad_number: Optional[str] = None                   # enables versioned history
    campaign_id: Optional[str] = None
    funnel_stage: Optional[str] = None                # cold | warm | hot | retargeting
    brand_assets: BrandAssets = Field(default_factory=BrandAssets)
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)


# =========================================================================== #
# Response
# =========================================================================== #
class Availability(BaseModel):
    """Always present. `available=false` with a stable reason is how this API
    reports failure - not a 4xx, and never an invented scorecard."""
    available: bool = True
    reason: Optional[str] = None
    message: Optional[str] = None


class AnalyzeAccepted(BaseModel):
    """The POST response. This is NOT the report - poll `poll_url`."""
    analysis_id: str
    creative_id: str = ""
    status: str = "queued"
    created_at: Optional[datetime] = None
    poll_url: Optional[str] = None
    suggested_poll_interval_seconds: int = 5
    idempotent_hit: bool = False
    availability: Availability = Field(default_factory=Availability)
    reason: Optional[str] = None
    message: Optional[str] = None


class UploadedCreative(BaseModel):
    """Where an uploaded creative landed, and the link that analyses it."""
    object_key: str
    filename: str
    size_bytes: int
    mime_type: str
    kind: str
    # The same presigned link /analyze consumes. It is a credential: anyone
    # holding it can read the file until it expires.
    creative_url: Optional[str] = None
    creative_url_expires_in_seconds: Optional[int] = None


class UploadAccepted(BaseModel):
    """The POST /upload response. `analysis` is present only when the upload
    was also asked to analyse - and is then exactly what POST /analyze returns."""
    uploaded: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    upload: Optional[UploadedCreative] = None
    analysis: Optional[AnalyzeAccepted] = None


class MetricScore(BaseModel):
    score: Optional[int] = None
    label: str = ""
    direction: str = "higher_better"
    basis: str = "measured"          # measured | hybrid
    signals: dict[str, Any] = Field(default_factory=dict)


class HistoryItem(BaseModel):
    """One row of the History tab."""
    analysis_id: str
    creative_id: Optional[str] = None
    ad_number: Optional[str] = None
    division: Optional[str] = None
    status: str = ""
    overall_score: Optional[int] = None
    created_at: Optional[datetime] = None


class HistoryResponse(BaseModel):
    items: list[HistoryItem] = Field(default_factory=list)
    count: int = 0
