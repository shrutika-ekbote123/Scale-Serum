# Sales Call Analyzer — API Contract

Integration contract for the **backend** team. The frontend never talks to this
service directly, and never talks to Deepgram at all.

> Companion docs: `API_HANDOFF.md` (Brand Brain endpoints), `PURCHASE_PROBABILITY_API.md`.

---

## 1. What it does

```
Frontend  ──(Log Call form: outcome, remarks, recording, transcript)──▶  Backend
Backend   ──(stores recording, mints a signed URL)───────────────────▶  AI service
AI        ──(Deepgram: transcription + speaker diarization)──────────▶  AI
AI        ──(Gemini: framework evaluation + evidence)────────────────▶  AI
AI        ──(evidence verification, deterministic scoring in Python)─▶  AI
Backend   ──(polls, stores the report on its own call record)────────▶  Frontend
```

**Division of labour, and why it matters to you:**

| Component | Owns |
|---|---|
| **Deepgram** | What was said, who spoke, when |
| **Gemini** | Interpretation: needs, objections, buying signals, per-criterion ratings, evidence, prose |
| **Python** | Validation, evidence verification, **all scoring**, aggregation, persistence, idempotency |

The model never produces a score. Every number in the response is computed in
Python from ordinal ratings plus `sales_framework.json`. That is why a score is
reproducible and why calls can be rescored later without re-analysing them.

---

## 2. Basics

| | |
|---|---|
| **Base path** | `/api/sales-calls` |
| **Auth** | `X-API-Key: <key>` on every request |
| **Processing** | **Asynchronous.** `POST` returns an id; poll the `GET` |
| **Interactive docs** | `{BASE_URL}/docs` |
| **Health** | `GET /health` → includes `sales_call_analyzer.{available,transcription,storage}` |

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/sales-calls/analyze` | Submit a call. Returns `analysis_id` immediately |
| `GET` | `/api/sales-calls/analysis/{analysis_id}` | Poll status, then read the report |
| `GET` | `/api/sales-calls/analysis/by-call/{call_id}` | Latest analysis for a call |
| `POST` | `/api/sales-calls/analysis/{analysis_id}/rescore` | Recompute scores under the current framework. No provider calls, no cost |

**Versioning** is carried in the payload (`framework_version`, `prompt_version`,
`transcript_version`), not the URL — matching every other endpoint in this
service. Response fields are added, never repurposed.

---

## 3. Submit

`POST /api/sales-calls/analyze`

```json
{
  "call_id": "e0f1a2b3",
  "lead_id": "9c2a55d1",
  "audio": {
    "url": "https://storage.example.com/recordings/abc.mp3?X-Amz-Signature=...",
    "mime_type": "audio/mpeg",
    "duration_seconds": 1443,
    "expires_at": "2026-09-04T10:11:00Z"
  },
  "transcript": { "format": "text", "text": "..." },
  "call_metadata": {
    "direction": "outbound",
    "occurred_at": "2026-05-28T23:07:00Z",
    "duration_seconds": 1443,
    "disposition": "SQL",
    "remarks": "Discussed EMI options",
    "provider": "Dialer",
    "recording_reference": "rec_8811"
  },
  "rep": { "id": "u_12", "name": "Rajan Kumar", "designation": "Senior Counsellor" },
  "customer": {
    "name": "Meera Patel", "email": "meera@example.com", "phone": "+919876543210",
    "region": "Maharashtra", "language": "en-IN",
    "designation": "Operations Manager", "industry": "Logistics",
    "awareness_level": "problem-aware",
    "profile": { "seniority": "Manager" },
    "previous_interactions": ["Masterclass webinar, 6 Aug 2026"]
  },
  "product": {
    "id": "career_accelerator", "name": "Career Accelerator",
    "price": 200000, "currency": "INR", "complexity": "high",
    "is_structured_programme": true, "sold_on_call": false
  },
  "options": { "force_reanalysis": false, "language_hint": "en-IN" }
}
```

**Only `call_id` is required.** Send what you have — missing context is reported
in `context_used.missing`, never guessed at.

### Fields that change the analysis

| Field | Effect if omitted |
|---|---|
| `rep.name` / `customer.name` | Speaker roles stay `unknown`. Roles are **never** guessed from turn order or talk time |
| `customer.region`, `language`, `designation`, `awareness_level` | Adaptation is judged on fewer factors. Nothing is assumed about people from a region |
| `product.price`, `complexity` | The pitch cannot be judged against the price band |
| `product.is_structured_programme` | "Explains the programme structure" becomes **not applicable** — `null` means *unknown*, and unknown is never scored as a failure |
| `product.sold_on_call` | "Explains the payment steps" becomes **not applicable** |
| `customer.previous_interactions` | "References a previous interaction" becomes **not applicable** |
| `lead_id` | Brand Brain cannot be resolved; brand-dependent criteria become not applicable |

### Which input is used

| You send | What happens |
|---|---|
| Audio only | Deepgram transcribes + diarizes |
| **Structured** transcript (speakers + timings) | Deepgram is **skipped** — you already have what it would produce |
| Plain-text transcript only | Analysed in degraded mode: no timings, no tone. Tone/interruption criteria become not applicable |
| Audio **+** structured transcript | Transcript wins, Deepgram skipped |
| Audio **+** plain-text transcript | **Audio wins.** Plain text has no speakers or timings, so trusting it would degrade every evidence anchor to save one API call. The text is kept as a fallback if Deepgram fails |
| Neither | `status: "failed"`, `reason: "no_audio_or_transcript"`, nothing is spent |

### Response (immediate)

```json
{
  "analysis_id": "8f3c2b1a...",
  "call_id": "e0f1a2b3",
  "status": "queued",
  "created_at": "2026-09-04T09:11:04Z",
  "idempotent_hit": false,
  "poll_url": "/api/sales-calls/analysis/8f3c2b1a...",
  "suggested_poll_interval_seconds": 5
}
```

**This is not the report.** Poll `poll_url` every ~5s. Typical completion is
1–3 minutes for a 25-minute call.

---

## 4. Poll / read

`GET /api/sales-calls/analysis/{analysis_id}`

**While processing** — `status` is `queued` / `transcribing` / `analyzing` / `scoring`:

```json
{ "analysis_id": "8f3c...", "call_id": "e0f1a2b3", "status": "analyzing",
  "availability": { "available": true, "reason": null },
  "scores": null, "attempts": 1,
  "poll_url": "/api/sales-calls/analysis/8f3c...",
  "suggested_poll_interval_seconds": 5 }
```

**When complete** (abridged — see `/docs` for the full schema):

```json
{
  "analysis_id": "8f3c...", "call_id": "e0f1a2b3", "lead_id": "9c2a55d1",
  "status": "completed",
  "availability": { "available": true },

  "call": {
    "disposition": "SQL", "disposition_source": "rep_reported",
    "direction": "outbound", "duration_seconds": 1443,
    "remarks": "Discussed EMI options", "recording_reference": "rec_8811",
    "rep": { "name": "Rajan Kumar" },
    "customer": { "name": "Meera Patel", "email": "...", "phone": "..." }
  },

  "scores": {
    "overall": 7.0, "overall_100": 70, "score_max": 10,
    "band": null, "band_reason": "thresholds_not_configured",
    "weighting": "equal_unweighted_placeholder",
    "rating_scale_mode": "uniform_placeholder",
    "stage_weights_confirmed": false, "criterion_weights_confirmed": false,
    "not_applicable_mode": "exclude",
    "framework_version": "sales_v1",
    "stages": [
      { "stage_id": "call_opening", "name": "Call Opening", "order": 1,
        "score": 8.0, "score_max": 10, "status": "scored" }
    ],
    "stages_scored": 5, "stages_not_applicable": 1,
    "basis": "24 of 32 criteria scored, 6 not applicable, 2 unsupported; equal weighting (no business weights configured); ..."
  },

  "stage_evaluations": [
    {
      "stage_id": "probing", "name": "Probing", "order": 3,
      "objective": "Understand the customer's background, aspirations...",
      "kpis": ["Effective questioning", "Active listening", "..."],
      "score": 6.6, "score_max": 10, "status": "scored",
      "assessment": "Good discovery of the certification gap, but budget was never explored.",
      "confidence": "high",
      "criteria": [
        {
          "criterion_id": "probing_active_listening",
          "name": "Practises active listening and asks relevant follow-up questions",
          "applicable": true, "rating": "adequate", "score": 6.67, "score_max": 10,
          "status": "scored", "confidence": "high", "evidence_backed": true,
          "observation": "Followed up on the certification gap, but moved past the budget concern.",
          "missing_behaviour": "Did not probe the budget objection.",
          "recommendation": "Ask what budget has been approved before quoting.",
          "evidence": [
            { "segment_index": 42, "speaker_id": "speaker_1",
              "start": 512.3, "end": 519.8,
              "quote": "my main problem is I don't have a recognised certification",
              "verified": true }
          ]
        }
      ],
      "strengths": [], "weaknesses": [], "recommendations": [],
      "criteria_scored": 4, "criteria_not_applicable": 1, "criteria_unsupported": 0
    }
  ],

  "summary": "...",
  "highlights": [{ "text": "...", "type": "positive", "evidence_backed": true, "evidence": [...] }],
  "strengths": [...], "weaknesses": [...], "recommendations": [...],
  "customer_needs": [{ "summary": "...", "kind": "pain_point", "evidence": [...] }],
  "objections": [{ "summary": "...", "category": "price", "handled": "partially_resolved", "evidence": [...] }],
  "buying_signals": [{ "type": "buying_intent", "label": "Buying intent", "strength": "moderate", "evidence": [...] }],
  "customer_signals": [...],
  "rep_techniques": [{ "type": "social_proof", "effectiveness": "landed", "evidence": [...] }],
  "pitch_structure": { "structure": "consultative_discovery_first", "fit": "well_matched", "rationale": "..." },
  "context_adaptation": { "assessment": "well_adapted", "affects_score": false, "factors_considered": ["region", "price band"] },

  "transcript": {
    "source": "deepgram", "diarization_available": true, "timestamps_available": true,
    "speaker_count": 3, "segment_count": 118, "duration_seconds": 1443.2,
    "speakers": [
      { "speaker_id": "speaker_0", "role": "sales_rep", "name": "Rajan Kumar",
        "role_basis": "crm_rep_self_introduction", "role_confidence": "high",
        "talk_time_seconds": 812.4, "turn_count": 61 },
      { "speaker_id": "speaker_2", "role": "participant", "name": null,
        "role_basis": "unresolved", "role_confidence": "low" }
    ],
    "segments": [
      { "index": 0, "speaker_id": "speaker_0", "start": 0.0, "end": 4.2,
        "text": "...", "confidence": 0.94 }
    ],
    "quality": { "mean_confidence": 0.91, "usable": true, "warnings": [] }
  },

  "context_used": { "brand_brain": {...}, "customer": {...}, "product": {...},
                    "missing": ["no_product_context"], "unmet_requirements": {...} },
  "analysis_quality": { "criteria_total": 32, "criteria_scored": 24,
                        "criteria_not_applicable": 6, "criteria_unsupported": 2,
                        "evidence_anchors_dropped": 3, "degraded": false, "warnings": [] },
  "processing": { "transcription_ms": 41230, "llm_ms": 28110, "total_ms": 71980,
                  "llm_model": "...", "framework_version": "sales_v1", "attempts": 1 },
  "fallback": false
}
```

---

## 5. Rules the UI must follow

1. **`null` is not `0`.** A stage with `score: null` and `status: "not_applicable"`
   means the call gave no opportunity for it — the customer raised no objection,
   for example. Render it as "Not applicable", never as a zero bar.
2. **`status: "unsupported"`** means we could not verify it. Also not a zero.
3. **Do not render a verdict.** `band` is `null` because management has not set
   thresholds. Do not invent red/amber/green from `overall_100`.
4. **Scores are provisional.** `weighting: "equal_unweighted_placeholder"` and
   `rating_scale_mode: "uniform_placeholder"` mean no business weights exist yet.
   Consider surfacing this.
5. **`disposition` is the rep's own.** The AI does not confirm, correct or
   replace it. There is no AI disposition field.
6. **Speaker roles may be `unknown` or `participant`.** Calls have more than two
   speakers. Render `speaker_2` as "Speaker 3" rather than forcing a label.
7. **`evidence_backed: false`** marks a claim about something that did *not*
   happen — it cannot be quoted. Show it, but without an evidence link.
8. **Evidence is clickable.** `segment_index` + `start` let you jump the audio
   player to the moment behind a score.
9. **Highlights are not padded.** Three means three were supported.

---

## 6. Failure reasons

Always HTTP 200 with `status: "failed"`, `availability.available: false`,
`scores: null`. Stable codes — branch on these, do not parse the message.

| Reason | Meaning | Retry? |
|---|---|---|
| `no_audio_or_transcript` | Nothing to analyse | Only with input |
| `audio_unreachable` | Recording could not be fetched (expired URL?) | Yes, with a fresh URL |
| `audio_too_large` | Over the configured size limit | No |
| `transcription_not_configured` | No Deepgram key on the server | No — ops fix |
| `transcription_provider_error` / `_timeout` / `_rate_limited` | Deepgram failed after retries | Yes, later |
| `transcript_empty` | No speech found (voicemail, no answer) | No |
| `analysis_failed_after_transcription` | Gemini failed. **Transcript kept** — retry is cheap | Yes |
| `analysis_provider_error` | Gemini unreachable | Yes |
| `processing_interrupted` | Job died mid-flight (usually a deploy restart) | Yes |
| `disposition_excluded_by_policy` | `status: "skipped"` — configured policy. Transcript produced, no scorecard | No |

**HTTP codes:** `401` bad/missing key · `404` unknown id · `409` rescoring an
incomplete analysis · `503` storage or analyzer not configured. Everything else
is `200`.

---

## 7. Cost control

- **Idempotent.** Resubmitting the same call returns the existing analysis for
  free. The fingerprint covers inputs *and* versions, so a new framework version
  is correctly treated as a new analysis.
- **A transcript is paid for once, ever.** It is stored before the LLM runs, so a
  retry after an analysis failure does not call Deepgram again.
- **Rescoring is free.** No provider calls.
- Send `options.force_reanalysis: true` only when you mean it.

---

## 8. Security

- The Deepgram key is server-side only and never appears in a response, a log
  line or a URL. **Never put it in frontend code.**
- Signed audio URLs are treated as credentials: read, never stored, never
  logged (only the host is).
- `customer.email` / `phone` are returned in the report for the UI but are
  **never sent to the LLM**.
- Transcript content is treated as untrusted input. It is fenced in the prompt,
  and — more importantly — the model cannot produce a score at all, so nothing
  said on a call can move a number.
- Logs carry ids, counts and timings only. No transcript text, no contact details.

---

## 9. Not yet configured

These are business decisions, deliberately absent from the code:

- stage weights, criterion weights
- the rating scale (ordinal → numeric mapping)
- score bands / pass-fail thresholds
- whether not-applicable should be excluded (current default) or scored zero
- which dispositions skip evaluation (e.g. `No Answer`, `Invalid number`)
- minimum call length to evaluate
- whether context adaptation should affect the score (currently reported only)

All live in `sales_call_analyzer/sales_framework.json`. Setting them is a JSON
edit plus a `framework_version` bump — no code change — and historical calls can
be brought onto the new configuration with the rescore endpoint.
