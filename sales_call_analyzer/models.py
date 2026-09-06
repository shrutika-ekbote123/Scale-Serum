"""
Request / response models for the Sales Call Analyzer.

Conventions follow app.py: pydantic BaseModel, almost everything Optional with a
safe default, so a partially-filled request still produces a usable analysis
rather than a 422. Only `call_id` is genuinely required.

Free-string enums on purpose. Where a value comes from the backend or from the
model, it is typed `str` and normalised in code rather than constrained by
`Literal`, because the house contract is "always return 200 with a stated
reason", not "reject the caller". The valid values are listed in __init__.py
and validated where it matters (scoring, evidence).
"""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# =========================================================================== #
# Request
# =========================================================================== #
class AudioRef(BaseModel):
    """Where the recording lives. The AI service reads it and never stores it."""
    url: Optional[str] = None                 # https URL, ideally short-lived and signed
    mime_type: Optional[str] = None           # e.g. "audio/mpeg"
    duration_seconds: Optional[float] = None  # backend's own measurement, if it has one
    expires_at: Optional[datetime] = None     # signed-URL expiry, for diagnostics only


class SuppliedSegment(BaseModel):
    """One turn of a transcript the caller already has (e.g. from a dialer)."""
    speaker_id: Optional[str] = None     # "speaker_0", "0", "Rep" - normalised on ingest
    speaker_label: Optional[str] = None  # free-text label if the provider used names
    role: Optional[str] = None           # sales_rep | customer | participant | unknown
    name: Optional[str] = None
    start: Optional[float] = None        # seconds from call start
    end: Optional[float] = None
    text: str = ""
    confidence: Optional[float] = None


class SuppliedTranscript(BaseModel):
    """A transcript the caller already has.

    `format` is advisory - normalisation decides for itself whether the payload
    is really structured (has speakers and timestamps) or plain text, because a
    caller labelling plain text as "structured" must not silently disable
    diarization.
    """
    format: Optional[str] = None                        # "text" | "structured"
    text: Optional[str] = None                          # plain-text transcript
    segments: list[SuppliedSegment] = Field(default_factory=list)
    language: Optional[str] = None


class CallMetadata(BaseModel):
    direction: Optional[str] = None            # "inbound" | "outbound"
    occurred_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    disposition: Optional[str] = None          # CRM outcome: SQL / MQL / Non SQL / ...
    remarks: Optional[str] = None              # rep's own notes
    provider: Optional[str] = None             # e.g. "Dialer"
    recording_reference: Optional[str] = None  # opaque backend id; never a signed URL


class RepInfo(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    designation: Optional[str] = None


class CustomerInfo(BaseModel):
    """Everything the CRM knows about the person on the other end. Used as
    context for interpretation, and to resolve who is speaking."""
    id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    region: Optional[str] = None
    language: Optional[str] = None
    designation: Optional[str] = None
    industry: Optional[str] = None
    experience: Optional[str] = None
    awareness_level: Optional[str] = None
    profile: dict[str, Any] = Field(default_factory=dict)   # anything else the CRM holds
    previous_interactions: list[str] = Field(default_factory=list)


class ProductInfo(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    price_band: Optional[str] = None    # caller's own label, e.g. "high-ticket"
    complexity: Optional[str] = None
    is_structured_programme: Optional[bool] = None   # drives the structured_programme requirement
    sold_on_call: Optional[bool] = None              # drives the payment_process requirement


class AnalyzeOptions(BaseModel):
    force_reanalysis: bool = False   # bypass the idempotency hit and re-run
    language_hint: Optional[str] = None
    store_raw_transcript: Optional[bool] = None   # overrides SCA_STORE_RAW_TRANSCRIPT


class AnalyzeRequest(BaseModel):
    """What the backend posts. Only `call_id` is required; at least one of
    `audio` or `transcript` must carry something, which is reported as a stated
    failure reason rather than a 422."""
    call_id: str
    lead_id: Optional[str] = None
    brand_id: Optional[str] = None
    brand_brain_id: Optional[str] = None     # normally resolved from the lead
    audio: Optional[AudioRef] = None
    transcript: Optional[SuppliedTranscript] = None
    call_metadata: CallMetadata = Field(default_factory=CallMetadata)
    rep: RepInfo = Field(default_factory=RepInfo)
    customer: CustomerInfo = Field(default_factory=CustomerInfo)
    product: ProductInfo = Field(default_factory=ProductInfo)
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)


# =========================================================================== #
# Transcript (internal, normalised)
# =========================================================================== #
class TranscriptSpeaker(BaseModel):
    """A speaker as diarization found them.

    `speaker_id` is the primary identity and is never renamed. `role` and `name`
    are annotations that stay unresolved when the evidence does not support them
    - "unknown" is a valid, expected outcome.
    """
    speaker_id: str
    role: str = "unknown"                  # sales_rep | customer | participant | unknown
    name: Optional[str] = None             # only ever from CRM context, never invented
    role_basis: str = "unresolved"         # stable code explaining how the role was decided
    role_confidence: str = "none"          # high | medium | low | none
    talk_time_seconds: float = 0.0
    turn_count: int = 0
    word_count: int = 0


class TranscriptSegment(BaseModel):
    """One speaker turn. `index` is the anchor every piece of evidence cites."""
    index: int
    speaker_id: str
    start: Optional[float] = None
    end: Optional[float] = None
    text: str = ""
    confidence: Optional[float] = None


class TranscriptQuality(BaseModel):
    mean_confidence: Optional[float] = None
    low_confidence_ratio: Optional[float] = None
    usable: bool = True
    warnings: list[str] = Field(default_factory=list)


class NormalizedTranscript(BaseModel):
    transcript_version: str = "v1"
    source: str = "deepgram"          # deepgram | supplied_structured | supplied_text
    language: Optional[str] = None
    language_detected: Optional[str] = None
    multilingual: bool = False
    diarization_available: bool = False
    timestamps_available: bool = False
    speaker_count: int = 0
    segment_count: int = 0
    duration_seconds: Optional[float] = None
    word_count: int = 0
    speakers: list[TranscriptSpeaker] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    quality: TranscriptQuality = Field(default_factory=TranscriptQuality)


# =========================================================================== #
# Analysis
# =========================================================================== #
class Evidence(BaseModel):
    """A pointer back into the transcript.

    `start`/`end` are copied from the cited segment, never taken from the model,
    so a timestamp cannot be fabricated. `verified` is set by evidence.py.
    """
    segment_index: int
    speaker_id: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    quote: str = ""
    verified: bool = False


class CriterionEvaluation(BaseModel):
    criterion_id: str
    name: str = ""
    stage_id: str = ""
    applicable: bool = True
    not_applicable_reason: Optional[str] = None
    rating: Optional[str] = None           # one of framework rating_levels
    score: Optional[float] = None          # computed by scoring.py, never by the model
    score_max: Optional[float] = None
    confidence: str = "medium"             # high | medium | low
    status: str = "scored"                 # scored | not_applicable | unsupported
    observation: str = ""
    evidence_backed: bool = True
    missing_behaviour: Optional[str] = None
    recommendation: Optional[str] = None
    evidence: list[Evidence] = Field(default_factory=list)


class StageEvaluation(BaseModel):
    stage_id: str
    name: str = ""
    order: int = 0
    objective: str = ""
    kpis: list[str] = Field(default_factory=list)
    score: Optional[float] = None          # null when not applicable - never 0 as a stand-in
    score_max: Optional[float] = None
    status: str = "scored"                 # scored | not_applicable | unsupported
    not_applicable_reason: Optional[str] = None
    assessment: str = ""
    confidence: str = "medium"
    criteria: list[CriterionEvaluation] = Field(default_factory=list)
    strengths: list["Insight"] = Field(default_factory=list)
    weaknesses: list["Insight"] = Field(default_factory=list)
    recommendations: list["Insight"] = Field(default_factory=list)
    criteria_scored: int = 0
    criteria_not_applicable: int = 0
    criteria_unsupported: int = 0


class Insight(BaseModel):
    """A statement about the call that must be traceable to the transcript.

    `evidence_backed` is false only for claims about ABSENCE - "never asked about
    budget" cannot be quoted, because the quote does not exist. Claims that
    something happened are dropped when their evidence does not verify.
    """
    text: str
    detail: Optional[str] = None
    stage_id: Optional[str] = None
    evidence_backed: bool = True
    evidence: list[Evidence] = Field(default_factory=list)


class Highlight(BaseModel):
    text: str
    type: str = "neutral"          # positive | negative | neutral
    stage_id: Optional[str] = None
    evidence_backed: bool = True
    evidence: list[Evidence] = Field(default_factory=list)


class CustomerSignal(BaseModel):
    type: str                       # from signals_config.customer_signals
    label: str = ""
    speaker_id: Optional[str] = None
    summary: str = ""
    strength: Optional[str] = None  # strong | moderate | weak
    evidence: list[Evidence] = Field(default_factory=list)


class RepTechnique(BaseModel):
    type: str                       # from signals_config.rep_techniques
    label: str = ""
    speaker_id: Optional[str] = None
    summary: str = ""
    effectiveness: str = "insufficient_evidence"
    rationale: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class Objection(BaseModel):
    summary: str
    category: Optional[str] = None      # price | timing | trust | authority | fit | other
    raised_by: Optional[str] = None     # speaker_id
    handled: Optional[str] = None       # resolved | partially_resolved | unresolved | ignored
    handling_notes: Optional[str] = None
    evidence: list[Evidence] = Field(default_factory=list)


class CustomerNeed(BaseModel):
    summary: str
    kind: str = "need"                  # need | pain_point | goal
    addressed: Optional[bool] = None
    evidence: list[Evidence] = Field(default_factory=list)


class PitchStructureAssessment(BaseModel):
    structure: str = "none_discernible"
    fit: str = "insufficient_evidence"
    rationale: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class ContextAdaptation(BaseModel):
    """How well the rep adapted to THIS customer, product and context.

    Reported alongside the stage scores and deliberately excluded from the
    overall score: whether context adaptation should move a number is a business
    decision nobody has made yet.
    """
    assessment: str = "insufficient_evidence"
    rationale: str = ""
    factors_considered: list[str] = Field(default_factory=list)
    affects_score: bool = False
    evidence: list[Evidence] = Field(default_factory=list)


class StageScore(BaseModel):
    """Compact stage score for list views and the score header."""
    stage_id: str
    name: str
    order: int
    score: Optional[float] = None
    score_max: Optional[float] = None
    status: str = "scored"


class Scores(BaseModel):
    overall: Optional[float] = None
    overall_100: Optional[int] = None
    score_max: Optional[float] = None
    band: Optional[str] = None
    band_reason: str = "thresholds_not_configured"
    weighting: str = "equal_unweighted_placeholder"
    rating_scale_mode: str = "uniform_placeholder"
    framework_version: str = ""
    stage_weights_confirmed: bool = False
    criterion_weights_confirmed: bool = False
    not_applicable_mode: str = "exclude"
    stages: list[StageScore] = Field(default_factory=list)
    stages_scored: int = 0
    stages_not_applicable: int = 0
    basis: str = ""


class AnalysisQuality(BaseModel):
    criteria_total: int = 0
    criteria_scored: int = 0
    criteria_not_applicable: int = 0
    criteria_unsupported: int = 0
    evidence_anchors_total: int = 0
    evidence_anchors_dropped: int = 0
    transcript_confidence: Optional[float] = None
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)


class ProcessingInfo(BaseModel):
    transcription_ms: Optional[int] = None
    llm_ms: Optional[int] = None
    total_ms: Optional[int] = None
    transcription_provider: Optional[str] = None
    transcription_model: Optional[str] = None
    llm_model: Optional[str] = None
    attempts: int = 0
    llm_attempts: int = 0
    prompt_version: Optional[str] = None
    framework_version: Optional[str] = None
    signals_version: Optional[str] = None
    audio_seconds_submitted: Optional[float] = None
    llm_input_tokens: Optional[int] = None
    llm_output_tokens: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ContextUsed(BaseModel):
    """What context the analysis actually had. Missing context is reported, not
    silently tolerated, so a thin report can be explained."""
    brand_brain: dict[str, Any] = Field(default_factory=dict)
    customer: dict[str, Any] = Field(default_factory=dict)
    product: dict[str, Any] = Field(default_factory=dict)
    rep: dict[str, Any] = Field(default_factory=dict)
    call: dict[str, Any] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    unmet_requirements: dict[str, str] = Field(default_factory=dict)


class CallSummaryBlock(BaseModel):
    """The call card the Sales Calls UI renders above the scores."""
    call_id: str
    disposition: Optional[str] = None
    disposition_source: str = "rep_reported"
    direction: Optional[str] = None
    occurred_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    remarks: Optional[str] = None
    provider: Optional[str] = None
    recording_reference: Optional[str] = None
    rep: RepInfo = Field(default_factory=RepInfo)
    customer: CustomerInfo = Field(default_factory=CustomerInfo)


class Availability(BaseModel):
    """Same contract as the Purchase Probability endpoint: unavailable is not
    zero, and the reason code is stable enough for the UI to branch on."""
    available: bool = True
    reason: Optional[str] = None
    message: Optional[str] = None


class SalesCallAnalysis(BaseModel):
    """The full report. Returned by GET once status is completed."""
    analysis_id: str
    call_id: str
    lead_id: Optional[str] = None
    brand_id: Optional[str] = None
    status: str = "completed"
    availability: Availability = Field(default_factory=Availability)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    call: Optional[CallSummaryBlock] = None
    scores: Optional[Scores] = None
    stage_evaluations: list[StageEvaluation] = Field(default_factory=list)

    summary: str = ""
    highlights: list[Highlight] = Field(default_factory=list)
    strengths: list[Insight] = Field(default_factory=list)
    weaknesses: list[Insight] = Field(default_factory=list)
    recommendations: list[Insight] = Field(default_factory=list)

    customer_needs: list[CustomerNeed] = Field(default_factory=list)
    objections: list[Objection] = Field(default_factory=list)
    buying_signals: list[CustomerSignal] = Field(default_factory=list)
    customer_signals: list[CustomerSignal] = Field(default_factory=list)
    rep_techniques: list[RepTechnique] = Field(default_factory=list)
    pitch_structure: Optional[PitchStructureAssessment] = None
    context_adaptation: Optional[ContextAdaptation] = None

    transcript: Optional[NormalizedTranscript] = None
    context_used: ContextUsed = Field(default_factory=ContextUsed)
    analysis_quality: AnalysisQuality = Field(default_factory=AnalysisQuality)
    processing: ProcessingInfo = Field(default_factory=ProcessingInfo)
    fallback: bool = False


class AnalyzeAccepted(BaseModel):
    """The immediate POST response. This is NOT the report - poll for that."""
    analysis_id: str
    call_id: str
    status: str
    created_at: Optional[datetime] = None
    idempotent_hit: bool = False
    poll_url: str = ""
    suggested_poll_interval_seconds: int = 5
    availability: Availability = Field(default_factory=Availability)
    reason: Optional[str] = None
    message: Optional[str] = None


StageEvaluation.model_rebuild()
