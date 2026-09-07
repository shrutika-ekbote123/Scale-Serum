# Sales Call Analyzer — Handoff for Backend & Frontend

**Status:** built, tested, ready to deploy. Not yet integrated into the ScaleSerum app.
**Owner:** AI service (`Marketing_tool` repo)
**Audience:** backend and frontend teams

---

## Contents

1. [What this is](#1-what-this-is)
2. [How it works](#2-how-it-works)
3. [Prerequisites — read before integrating](#3-prerequisites--read-before-integrating)
4. [API reference](#4-api-reference)
5. [Backend integration guide](#5-backend-integration-guide)
6. [Frontend rendering guide](#6-frontend-rendering-guide)
7. [Complete field reference](#7-complete-field-reference)
8. [Failure handling](#8-failure-handling)
9. [Known limitations](#9-known-limitations)
10. [Awaiting management decisions](#10-awaiting-management-decisions)

---

## 1. What this is

When a sales rep logs a call with a recording, this service transcribes it,
identifies who spoke, evaluates the conversation against ScaleSerum's six-stage
sales framework, and returns a scored, evidence-backed report.

**What was built:**

| Capability | Detail |
|---|---|
| Transcription | Deepgram, pre-recorded audio |
| Speaker diarization | Any number of speakers — not limited to two |
| Speaker identification | Maps voices to `sales_rep` / `customer` / `participant`, only on evidence |
| Framework evaluation | 6 stages, 32 criteria, evaluated by Gemini |
| Deterministic scoring | Criterion → stage → overall, computed in Python |
| Evidence verification | Every claim checked against the transcript |
| Context awareness | Brand Brain, customer, product, region, price band |
| Persistence | MongoDB, with idempotency and job status |
| APIs | 4 endpoints in the existing FastAPI service |

**Verified on a real 10-minute call:** 52 segments transcribed, 3 speakers
separated (rep, customer, receptionist), 64 evidence anchors — none dropped,
transcript confidence 0.96.

**238 automated tests**, run by CI on every deploy.

---

## 2. How it works

```
Rep logs a call with a recording
        │
        ▼
BACKEND    stores recording, mints a URL, POSTs to the AI service
        │
        ▼
AI SERVICE
   ├─ Deepgram ......... what was said, who spoke, when
   ├─ Speaker roles .... which voice is the rep / the customer
   ├─ Context .......... Brand Brain + lead + product + call metadata
   ├─ Gemini ........... needs, objections, signals, per-criterion ratings
   ├─ Evidence check ... every quote verified against the transcript
   ├─ Scoring .......... criterion → stage → overall, in Python
   └─ MongoDB .......... persisted
        │
        ▼
BACKEND    polls, stores the report, exposes it
        │
        ▼
FRONTEND   renders the scorecard
```

### Division of labour

| Component | Responsible for |
|---|---|
| **Deepgram** | What was said, who spoke, when |
| **Gemini** | Interpretation — needs, objections, buying signals, criterion ratings, evidence, prose |
| **Python** | Validation, evidence verification, **all scoring**, aggregation, persistence, idempotency |

**The AI model never produces a number.** The response schema has no score field,
the system instruction forbids one, and scoring is computed in code from ordinal
ratings plus a config file. Three consequences that matter to you:

- the same ratings always produce the same score
- scores can be **recalculated** when management sets weights, without re-analysing calls
- nothing said on a call can influence a score

---

## 3. Prerequisites — read before integrating

Three things must be true before this works in production. **None are code changes in the AI service.**

### 3.1 TLS certificate does not cover `api.scaleserum.com` 🔴

The certificate on the server is issued for `scaleserum.com` only. Any HTTPS call
to `https://api.scaleserum.com` fails with a trust error.

```
subject: CN=scaleserum.com
issuer:  Let's Encrypt
```

**Fix (server):** `certbot --nginx -d scaleserum.com -d api.scaleserum.com`

Until this is done the backend cannot call the AI service over HTTPS.

### 3.2 Server environment variables 🔴

`/root/Marketing_tool/.env` needs:

```
DEEPGRAM_API_KEY=<key>
DEEPGRAM_MODEL=nova-2
GEMINI_MODEL=gemini-flash-latest
```

Without the Deepgram key the feature deploys **silently dead** — audio requests
return `transcription_not_configured`.

Verify: `curl -s http://127.0.0.1:3001/health | jq .sales_call_analyzer`
→ expect `"transcription": "configured"`.

### 3.3 Recordings must be reachable by Deepgram 🟡

Deepgram fetches the audio URL **itself**. It must be publicly reachable or a
signed URL — a `localhost` path or a private bucket path will fail with
`audio_unreachable`.

A short-lived signed S3/GCS URL is ideal. The AI service reads it and never
stores it.

### 3.4 Deploy note 🟡

The backend's deploy workflow runs `pm2 restart all`, which restarts **every**
process on the server including the AI service (`marketing-tool`). That kills any
analysis in progress. Please target the backend process by name.

---

## 4. API reference

| | |
|---|---|
| **Base URL** | `https://api.scaleserum.com` (prod) · `http://127.0.0.1:3001` (local) |
| **Base path** | `/api/sales-calls` |
| **Auth** | `X-API-Key: <key>` on every request |
| **Processing** | **Asynchronous** — POST returns an id, then poll |
| **Interactive docs** | `{BASE_URL}/docs` |

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/sales-calls/analyze` | Submit a call. Returns `analysis_id` immediately |
| `GET` | `/api/sales-calls/analysis/{analysis_id}` | Poll status, then read the report |
| `GET` | `/api/sales-calls/analysis/by-call/{call_id}` | Latest analysis for a call |
| `POST` | `/api/sales-calls/analysis/{analysis_id}/rescore` | Recompute scores. No provider calls, no cost |

**Versioning** is carried in the payload (`framework_version`, `prompt_version`,
`transcript_version`), not the URL — matching the rest of this service. Response
fields are added, never repurposed.

---

### 4.1 `POST /api/sales-calls/analyze`

```json
{
  "call_id": "e0f1a2b3",
  "lead_id": "9c2a55d1-aefc-4fa3-874f-f98824dd21ab",
  "audio": {
    "url": "https://storage.example.com/rec.mp3?X-Amz-Signature=...",
    "mime_type": "audio/mpeg",
    "duration_seconds": 594
  },
  "call_metadata": {
    "direction": "outbound",
    "occurred_at": "2026-09-04T10:14:00Z",
    "duration_seconds": 594,
    "disposition": "SQL",
    "remarks": "Discussed EMI options",
    "provider": "Dialer",
    "recording_reference": "rec_8811"
  },
  "rep": { "id": "u_12", "name": "Devansh Sharma", "designation": "Counsellor" },
  "customer": {
    "id": "c_88", "name": "Sanjay",
    "email": "sanjay@example.com", "phone": "+919876543210",
    "region": "Karnataka", "language": "en-IN",
    "designation": "CXO", "industry": "Manufacturing",
    "awareness_level": "problem-aware",
    "previous_interactions": ["Webinar, 28 Aug 2026"]
  },
  "product": {
    "id": "career_accelerator", "name": "Career Accelerator",
    "price": 200000, "currency": "INR", "complexity": "high",
    "is_structured_programme": true, "sold_on_call": true
  },
  "options": { "force_reanalysis": false, "language_hint": "en-IN" }
}
```

**Only `call_id` is required**, plus **one of** `audio` or `transcript`. Send what
you have — missing context is reported, never guessed.

**Response (immediate, ~200 ms):**

```json
{
  "analysis_id": "51d5f3cef9e940238773d379d3f17af0",
  "call_id": "e0f1a2b3",
  "status": "queued",
  "created_at": "2026-09-06T07:29:13Z",
  "idempotent_hit": false,
  "poll_url": "/api/sales-calls/analysis/51d5f3ce...",
  "suggested_poll_interval_seconds": 5
}
```

**This is not the report.** Poll `poll_url`. Typical completion: **30–90 seconds**
for a 10-minute call.

#### Fields that change the analysis

| Field | If omitted |
|---|---|
| `rep.name` / `customer.name` | Speaker roles stay `unknown` — never guessed from turn order or talk time |
| `lead_id` | Brand Brain cannot be resolved; brand-dependent criteria become not-applicable |
| `customer.region`, `language`, `designation`, `awareness_level` | Fewer adaptation factors available |
| `product.price`, `complexity` | Pitch cannot be judged against the price band |
| `product.is_structured_programme` | "Explains programme structure" → **not applicable** |
| `product.sold_on_call` | "Explains payment steps" → **not applicable** |
| `customer.previous_interactions` | "References previous interaction" → **not applicable** |

`null` means *unknown* and is never scored as a failure.

#### Which input is used

| You send | What happens |
|---|---|
| Audio only | Deepgram transcribes + diarizes |
| **Structured** transcript (speakers + timings) | Deepgram **skipped** — you already have what it would produce |
| Plain-text transcript only | Degraded mode: no timings, no tone. Tone criteria become not-applicable |
| Audio **+** structured transcript | Transcript wins, Deepgram skipped |
| Audio **+** plain-text transcript | **Audio wins** — plain text has no speakers or timings |
| Neither | `status: "failed"`, `reason: "no_audio_or_transcript"`. Nothing spent |

---

### 4.2 `GET /api/sales-calls/analysis/{analysis_id}`

**While processing** — `queued` / `transcribing` / `analyzing` / `scoring`:

```json
{
  "analysis_id": "51d5f3ce...", "call_id": "e0f1a2b3",
  "status": "analyzing",
  "availability": { "available": true, "reason": null },
  "scores": null,
  "attempts": 1,
  "poll_url": "/api/sales-calls/analysis/51d5f3ce...",
  "suggested_poll_interval_seconds": 5
}
```

**When complete** — abridged; see §7 for every field:

```json
{
  "analysis_id": "51d5f3ce...", "call_id": "e0f1a2b3", "lead_id": "9c2a55d1...",
  "status": "completed",
  "availability": { "available": true, "reason": null, "message": null },

  "call": {
    "disposition": "SQL",
    "disposition_source": "rep_reported",
    "direction": "outbound",
    "duration_seconds": 593.64,
    "remarks": "Discussed EMI options",
    "recording_reference": "rec_8811",
    "rep": { "name": "Devansh Sharma", "designation": "Counsellor" },
    "customer": { "name": "Sanjay", "email": "...", "phone": "...", "region": "Karnataka" }
  },

  "scores": {
    "overall": 4.9, "overall_100": 49, "score_max": 10,
    "band": null, "band_reason": "thresholds_not_configured",
    "weighting": "equal_unweighted_placeholder",
    "rating_scale_mode": "uniform_placeholder",
    "stage_weights_confirmed": false, "criterion_weights_confirmed": false,
    "not_applicable_mode": "exclude",
    "framework_version": "sales_v1",
    "stages": [
      { "stage_id": "call_opening",       "name": "Call Opening",        "order": 1, "score": 5.34, "score_max": 10, "status": "scored" },
      { "stage_id": "purpose_of_call",    "name": "Purpose of the Call", "order": 2, "score": 3.33, "score_max": 10, "status": "scored" },
      { "stage_id": "probing",            "name": "Probing",             "order": 3, "score": 2.00, "score_max": 10, "status": "scored" },
      { "stage_id": "product_pitching",   "name": "Product Pitching",    "order": 4, "score": 6.67, "score_max": 10, "status": "scored" },
      { "stage_id": "objection_handling", "name": "Objection Handling",  "order": 5, "score": null, "score_max": 10, "status": "not_applicable" },
      { "stage_id": "closing",            "name": "Closing the Deal",    "order": 6, "score": 7.14, "score_max": 10, "status": "scored" }
    ],
    "stages_scored": 5, "stages_not_applicable": 1,
    "basis": "26 of 32 criteria scored, 6 not applicable, 0 unsupported; equal weighting (no business weights configured); ..."
  },

  "stage_evaluations": [ /* per stage: objective, kpis, assessment, criteria[], strengths, weaknesses, recommendations */ ],

  "summary": "Devansh Sharma called Sanjay to answer questions regarding the Career Accelerator programme...",
  "highlights":      [ { "text": "...", "type": "positive", "evidence_backed": true, "evidence": [...] } ],
  "strengths":       [ { "text": "...", "detail": "...", "stage_id": "product_pitching", "evidence": [...] } ],
  "weaknesses":      [ ... ],
  "recommendations": [ ... ],
  "customer_needs":  [ { "summary": "...", "kind": "pain_point", "addressed": true, "evidence": [...] } ],
  "objections":      [ { "summary": "...", "category": "price", "handled": "resolved", "evidence": [...] } ],
  "buying_signals":  [ { "type": "buying_intent", "label": "Buying intent", "strength": "strong", "evidence": [...] } ],
  "customer_signals":[ ... ],
  "rep_techniques":  [ { "type": "social_proof", "label": "Social proof", "effectiveness": "landed", "evidence": [...] } ],
  "pitch_structure":    { "structure": "feature_led", "fit": "partially_matched", "rationale": "..." },
  "context_adaptation": { "assessment": "partially_adapted", "affects_score": false, "factors_considered": [...] },

  "transcript": {
    "source": "deepgram",
    "diarization_available": true, "timestamps_available": true,
    "speaker_count": 3, "segment_count": 52, "duration_seconds": 593.64,
    "language_detected": "en", "multilingual": false,
    "speakers": [
      { "speaker_id": "speaker_1", "role": "sales_rep", "name": "Devansh Sharma",
        "role_basis": "crm_rep_self_introduction", "role_confidence": "high",
        "talk_time_seconds": 351.3, "turn_count": 23 },
      { "speaker_id": "speaker_2", "role": "customer", "name": "Sanjay",
        "role_basis": "crm_customer_name_addressed", "role_confidence": "medium" },
      { "speaker_id": "speaker_0", "role": "participant", "name": null,
        "role_basis": "unresolved", "role_confidence": "low" }
    ],
    "segments": [
      { "index": 1, "speaker_id": "speaker_1", "start": 6.08, "end": 18.55,
        "text": "Hi, mister Sanjay. Good afternoon. Devansh here from Directors Institute.",
        "confidence": 0.98 }
    ],
    "quality": { "mean_confidence": 0.9651, "usable": true, "warnings": [] }
  },

  "context_used": {
    "brand_brain": { "available": true, "brand_name": "Director's Institute",
                     "brand_brain_id": "8d49c0e9..." },
    "customer": { "name_known": true, "region": "Karnataka", "profile_fields": [] },
    "product":  { "name": "Career Accelerator", "price": 200000, "currency": "INR" },
    "missing": [],
    "unmet_requirements": {}
  },

  "analysis_quality": {
    "criteria_total": 32, "criteria_scored": 26,
    "criteria_not_applicable": 6, "criteria_unsupported": 0,
    "evidence_anchors_total": 64, "evidence_anchors_dropped": 0,
    "transcript_confidence": 0.9651, "degraded": false, "warnings": []
  },

  "processing": {
    "transcription_ms": 4967, "llm_ms": 15000, "total_ms": 20063,
    "transcription_provider": "deepgram", "transcription_model": "nova-2",
    "llm_model": "gemini-flash-latest",
    "prompt_version": "sales_call_v2", "framework_version": "sales_v1",
    "audio_seconds_submitted": 593.64,
    "llm_input_tokens": 7255, "llm_output_tokens": 5858
  },

  "fallback": false
}
```

---

### 4.3 `GET /api/sales-calls/analysis/by-call/{call_id}`

The most recent analysis for a call — for when you kept the `call_id` but not the
`analysis_id`. Same response shape. Returns 404 if none exists.

⚠️ Returns the **latest**, which may be a failed attempt. Prefer storing the
`analysis_id`.

---

### 4.4 `POST /api/sales-calls/analysis/{analysis_id}/rescore`

Recomputes scores from the **stored ratings** under the current framework
configuration. **No Deepgram call, no Gemini call, no cost.**

When management finalises weights or thresholds, every historical call can be
brought onto the new rules without re-analysing. Returns the updated report.
Returns 409 if the analysis is not `completed`.

---

## 5. Backend integration guide

### What to do when a rep saves a Log Call

```
1. Save the call record as you do today (disposition, remarks, recording)
2. Upload the recording; mint a signed URL Deepgram can fetch
3. POST /api/sales-calls/analyze
   → store the returned analysis_id on the call record
   → set analysis_status = "processing"
4. Poll GET /api/sales-calls/analysis/{analysis_id} every 5s
   (background job / queue — do NOT block the rep's request)
5. On "completed": store the report; set analysis_status = "completed"
   On "failed":    store availability.reason; set analysis_status = "failed"
6. Expose the report to the frontend through your own API
```

### Send `lead_id`, not `brand_brain_id`

`lead_id` lets the AI service resolve the Brand Brain automatically:

```
leads.brand_id → brands.brand_brain_id → MongoDB brand_brains
```

**Verified working** — a test lead resolved through to its Brand Brain
successfully. `brand_brain_id` exists only as a manual override for testing.

### Polling

- Poll every **5 seconds** (`suggested_poll_interval_seconds`)
- Give up after ~5 minutes and mark it failed; the report can be retried
- Statuses: `queued` → `transcribing` → `analyzing` → `scoring` → `completed`
- Terminal statuses: `completed`, `failed`, `skipped`

### Idempotency — you get this free

Resubmitting the same call returns the existing analysis with
`idempotent_hit: true` and **costs nothing**. Protects against double-clicks and
retries. Send `options.force_reanalysis: true` only when you deliberately want a
fresh run.

A transcript is **paid for once, ever** — a retry after a failed analysis reuses
the stored transcript and does not call Deepgram again.

### What to store

Minimum: `analysis_id`, `status`, and the full report JSON.
The report is self-contained — no joins needed to render it.

---

## 6. Frontend rendering guide

### Nine rules that matter

1. **`null` is not `0`.** A stage with `score: null` and `status: "not_applicable"`
   means the call gave no opportunity — the customer raised no objection, for
   example. Render "Not applicable", **never a zero bar**.
2. **`status: "unsupported"`** means it could not be verified. Also not a zero.
3. **Do not render a verdict.** `band` is `null` because management has not set
   thresholds. Do not invent red/amber/green from `overall_100`.
4. **Scores are provisional.** `weighting: "equal_unweighted_placeholder"` means
   no business weights exist yet. Consider surfacing this.
5. **`disposition` is the rep's own.** `disposition_source: "rep_reported"`. The
   AI never confirms, corrects or replaces it. There is no AI disposition field.
6. **Speaker roles may be `unknown` or `participant`.** Calls have more than two
   speakers. Render `speaker_0` as "Speaker 1" rather than forcing a label.
7. **`evidence_backed: false`** marks a claim about something that did *not*
   happen — it cannot be quoted. Show it, but without an evidence link.
8. **Evidence is clickable.** `segment_index` + `start` let you jump the audio
   player to the exact moment behind a score.
9. **Lists are not padded.** Three highlights means three were supported.

### Suggested screen layout

```
┌─ HEADER ────────────────────────────────────────────────┐
│ {call.customer.name}          {scores.overall_100}/100  │
│ {call.rep.name} · {duration} · {direction}              │
│ Disposition: {call.disposition}  (rep-reported)         │
└─────────────────────────────────────────────────────────┘

┌─ STAGE SCORES ── scores.stages[] ───────────────────────┐
│ Call Opening        ████████░░  5.3                     │
│ Purpose of Call     ██████░░░░  3.3                     │
│ Probing             ████░░░░░░  2.0                     │
│ Product Pitching    ████████░░  6.7                     │
│ Objection Handling  — Not applicable                    │  ← null, not 0
│ Closing the Deal    ████████░░  7.1                     │
└─────────────────────────────────────────────────────────┘

┌─ SUMMARY ── summary ────────────────────────────────────┐
┌─ HIGHLIGHTS ── highlights[] ────────────────────────────┐
┌─ STRENGTHS / WEAKNESSES / RECOMMENDATIONS ──────────────┐
┌─ CUSTOMER ── customer_needs[], objections[],            │
│              buying_signals[] ──────────────────────────┐
┌─ STAGE DETAIL (expandable) ── stage_evaluations[] ──────┐
│  each criterion: name · rating · score · observation    │
│  · evidence → click to jump the player                  │
└─────────────────────────────────────────────────────────┘
┌─ TRANSCRIPT ── transcript.segments[] ───────────────────┐
│  [speaker role + name] [mm:ss] text                     │
└─────────────────────────────────────────────────────────┘
```

### Field mapping for the existing Sales Calls screen

| UI element | Field |
|---|---|
| Overall score | `scores.overall_100` |
| Stage bars | `scores.stages[].score` + `.status` |
| Disposition chip | `call.disposition` |
| Duration | `call.duration_seconds` |
| Direction | `call.direction` |
| Rep name | `call.rep.name` |
| Lead name / email / phone | `call.customer.name` / `.email` / `.phone` |
| Remarks | `call.remarks` |
| Recording | `call.recording_reference` |
| Highlights | `highlights[]` |
| Transcript | `transcript.segments[]` + `transcript.speakers[]` |

---

## 7. Complete field reference

### `call`
`call_id` · `disposition` · `disposition_source` · `direction` · `occurred_at` ·
`duration_seconds` · `remarks` · `provider` · `recording_reference` ·
`rep{id,name,designation}` ·
`customer{id,name,email,phone,region,language,designation,industry,experience,awareness_level,profile,previous_interactions}`

### `scores`
`overall` · `overall_100` · `score_max` · `band` · `band_reason` · `weighting` ·
`rating_scale_mode` · `framework_version` · `stage_weights_confirmed` ·
`criterion_weights_confirmed` · `not_applicable_mode` ·
`stages[{stage_id,name,order,score,score_max,status}]` · `stages_scored` ·
`stages_not_applicable` · `basis`

### `stage_evaluations[]`
`stage_id` · `name` · `order` · `objective` · `kpis[]` · `score` · `score_max` ·
`status` · `not_applicable_reason` · `assessment` · `confidence` · `criteria[]` ·
`strengths[]` · `weaknesses[]` · `recommendations[]` · `criteria_scored` ·
`criteria_not_applicable` · `criteria_unsupported`

### `criteria[]`
`criterion_id` · `name` · `stage_id` · `applicable` · `not_applicable_reason` ·
`rating` · `score` · `score_max` · `confidence` · `status` · `observation` ·
`evidence_backed` · `missing_behaviour` · `recommendation` · `evidence[]`

**`rating`** ∈ `absent` · `weak` · `adequate` · `strong`
**`status`** ∈ `scored` · `not_applicable` · `unsupported`

### `evidence[]`
`segment_index` · `speaker_id` · `start` · `end` · `quote` · `verified`

Timestamps are copied from the transcript, never produced by the model.

### `transcript`
`source` (`deepgram` / `supplied_structured` / `supplied_text`) · `language` ·
`language_detected` · `multilingual` · `diarization_available` ·
`timestamps_available` · `speaker_count` · `segment_count` · `duration_seconds` ·
`word_count` · `speakers[]` · `segments[]` ·
`quality{mean_confidence,low_confidence_ratio,usable,warnings[]}`

**`speakers[]`** — `speaker_id` · `role` · `name` · `role_basis` ·
`role_confidence` · `talk_time_seconds` · `turn_count` · `word_count`
**`role`** ∈ `sales_rep` · `customer` · `participant` · `unknown`

**`segments[]`** — `index` · `speaker_id` · `start` · `end` · `text` · `confidence`

### Signal vocabularies (closed sets)

**`buying_signals[]` / `customer_signals[]`** — `type` ∈ `buying_intent` ·
`hesitation` · `price_sensitivity` · `trust` · `authority` · `objection_signal` ·
`engagement` · `interest` · `confusion` · `risk_concern` · `urgency` ·
`commitment` · `next_step_readiness` · `emotional_motivation` · `disengagement`

**`rep_techniques[]`** — `type` ∈ `social_proof` · `authority_positioning` ·
`scarcity_urgency` · `loss_aversion` · `value_framing` · `reciprocity` ·
`commitment_consistency` · `risk_reversal` · `personalisation` · `storytelling`
**`effectiveness`** ∈ `landed` · `neutral` · `backfired` · `insufficient_evidence`

**`pitch_structure.structure`** ∈ `consultative_discovery_first` ·
`problem_solution` · `story_led` · `roi_led` · `feature_led` · `social_proof_led` ·
`comparison_led` · `direct_offer` · `mixed` · `none_discernible`
**`.fit`** ∈ `well_matched` · `partially_matched` · `mismatched` · `insufficient_evidence`

**`objections[].category`** ∈ `price` · `timing` · `trust` · `authority` · `fit` ·
`competition` · `other`
**`.handled`** ∈ `resolved` · `partially_resolved` · `unresolved` · `ignored`

---

## 8. Failure handling

Always **HTTP 200** with `status: "failed"`, `availability.available: false`,
`scores: null`. Branch on the **code**, never the message.

| Reason | Meaning | Retry? |
|---|---|---|
| `no_audio_or_transcript` | Nothing to analyse | Only with input |
| `audio_unreachable` | Recording could not be fetched (expired URL?) | Yes, fresh URL |
| `audio_too_large` | Over the size limit | No |
| `transcription_not_configured` | No Deepgram key on the server | No — ops fix |
| `transcription_provider_error` | Deepgram failed after retries | Yes, later |
| `transcription_rate_limited` | Deepgram rate limit | Yes, later |
| `transcription_timeout` | Transcription too slow | Yes |
| `transcript_empty` | No speech found (voicemail, no answer) | No |
| `analysis_failed_after_transcription` | Gemini failed. **Transcript kept** — retry is cheap | Yes |
| `analysis_provider_error` | Gemini unreachable or quota exhausted | Yes, later |
| `processing_interrupted` | Job died mid-flight (usually a deploy restart) | Yes |
| `disposition_excluded_by_policy` | `status: "skipped"` — configured policy. Transcript produced, no scorecard | No |

**HTTP codes:** `401` bad/missing key · `404` unknown id · `409` rescoring an
incomplete analysis · `503` storage or analyzer not configured. Everything else
is `200`.

**A failed analysis never invents a scorecard.** `scores` is `null` — a sales
manager cannot tell a fabricated evaluation from a real one, so we don't produce
one.

---

## 9. Known limitations

**1. The overall score varies between runs (~±15) on ambiguous calls.**
Measured on a real recording. The scoring arithmetic is fully deterministic; the
variance comes from the AI's judgment on genuinely ambiguous cases — e.g. whether
a rep "had no opportunity to probe" or "had the opportunity and missed it".

Stage scores, evidence, transcript and the written analysis are stable and
reliable. **The single headline number is not yet precise enough to present as
exact.** Recommended: show a band once thresholds exist, rather than a number.

**2. Deployment restarts kill in-flight analyses.** They are marked
`processing_interrupted` and can be retried. Acceptable at current volume; a
proper job queue is the follow-up if volume grows.

**3. Tone criteria need audio.** A pasted text transcript carries no tone or
interruption information, so those criteria become not-applicable rather than
being guessed at.

**4. Multilingual accuracy is unverified.** Hindi-English code-switching is
supported and handled, but has not been measured against a large sample.

---

## 10. Awaiting management decisions

The system ships with **no invented business values**. These are all `null` in
`sales_call_analyzer/sales_framework.json`:

| Decision | Current placeholder |
|---|---|
| Stage weights | Equal |
| Criterion weights | Equal |
| Rating scale (ordinal → numeric) | Uniform spacing |
| Score bands / pass-fail thresholds | **None** — `band` is `null` |
| Not-applicable policy | Excluded from the average |
| Dispositions that skip evaluation | None |
| Minimum call length to evaluate | None |
| Does context adaptation affect the score? | No — reported only |

Every response states which values were still unconfirmed, so a placeholder can
never be mistaken for a decision.

**Setting them is a JSON edit plus a version bump — no code changes.** The
`/rescore` endpoint then brings existing analyses onto the new configuration at
no cost.

---

## Questions

- **API contract / behaviour** → this document, or `{BASE_URL}/docs`
- **Module internals** → `sales_call_analyzer/README.md`
- **Postman collection** → `postman/ScaleSerum-SalesCallAnalyzer.postman_collection.json`
