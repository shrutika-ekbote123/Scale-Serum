# Vision Lab — Handoff for Backend & Frontend

**Status:** built and tested end to end on real ScaleSerum ads. **Not yet deployed** — the
server plumbing is the remaining step (see [§3.1](#31-it-is-not-deployed-yet-)).
`VL_ENABLED=false` on the server until it is signed off.
**Owner:** AI service (`Marketing_tool` repo) — same FastAPI service as Brand Brain and the
Sales Call Analyzer.
**Audience:** backend and frontend teams.

---

## Contents

1. [What this is](#1-what-this-is)
2. [How it works](#2-how-it-works)
3. [Prerequisites — read before integrating](#3-prerequisites--read-before-integrating)
4. [Conventions every call follows](#4-conventions-every-call-follows)
5. [API reference](#5-api-reference)
6. [Job lifecycle and polling](#6-job-lifecycle-and-polling)
7. [Backend integration guide](#7-backend-integration-guide)
8. [Frontend rendering guide](#8-frontend-rendering-guide)
9. [Complete field reference](#9-complete-field-reference)
10. [Reason codes and failure handling](#10-reason-codes-and-failure-handling)
11. [Limits](#11-limits)
12. [Testing it yourself](#12-testing-it-yourself)
13. [Known limitations](#13-known-limitations)
14. [Awaiting management decisions](#14-awaiting-management-decisions)
15. [Glossary](#15-glossary)

---

## 1. What this is

A marketer uploads a video ad (or an image). Vision Lab predicts where a viewer's eye goes,
frame by frame, and returns an **Attention Report**:

- a **heatmap** of the ad with numbered, labelled attention hotspots
- an **overall score out of 100** and a written **guidance** paragraph
- **six metric scores** — Attention, Focus, Cognitive Demand, Clarity, Brand Memory, Engagement
- **key moments** — the peak, the hero moment, the key message, the weak spots
- an **attention-over-time** curve with weak zones marked
- the **transcript**, line by line, each line with its attention level
- the **15 psychological triggers**, each present / weak / absent / not applicable
- **fix recommendations** — what is wrong, why it matters, what to change, which scores it hurts

> **It is AI-predicted attention, not eye-tracking.** No person watched the ad. A model
> trained on eye-tracking studies of other images predicts where eyes would go on this one.
> Every screen that shows these numbers must say so.

### What was built

| Capability | Detail |
|---|---|
| Frame sampling | ffmpeg. Up to 120 frames spread across the **whole** ad — a long ad gets a lower frame rate, never a cut-off ending. Shot cuts detected |
| Attention prediction | **UNISAL** saliency model (ONNX, runs on CPU). Chosen in a bake-off against hand-annotated ScaleSerum frames |
| On-screen detection | Text (OCR, **English + Hindi**), faces, brand wordmark, call to action, prices |
| Measurement | Where the predicted attention actually lands — on the text, the face, the brand, the CTA |
| Heatmap | Overlay of the most representative frame, top peaks numbered and named, plus a 6-frame strip. Stored in S3 |
| Timeline | Attention and engagement curves, weak zones, key moments |
| Transcript | Deepgram. Every line joined to the attention curve |
| Psychology | 15 triggers — 10 measured by counting, the rest judged by Gemini with verified evidence |
| Interpretation | Gemini writes the summary and the fix recommendations — **about problems Python measured**, never its own |
| Evidence verification | Every quote, timestamp and number Gemini writes is checked. Anything unsupported is dropped and logged |
| Scoring | All six metrics and the overall computed in Python. The AI never produces a number |
| Persistence | MongoDB. Per-frame measurements kept 180 days so scores can be recomputed |
| APIs | 7 endpoints on the existing FastAPI service |

**Verified on real ads:** an 82.5 s English ad and an 83.6 s Hinglish ad — 120 frames each,
100% of the runtime covered, transcript, 15 triggers and anchored recommendations — **45–100 s
per analysis** on a development laptop (CPU only).

**161 automated Vision Lab tests** (494 across the service), run by CI on every deploy.

---

## 2. How it works

### Two processes, one database as the queue

```
  Backend (server-to-server)            AI service
 ────────────────────────────  ┌───────────────────────────────────────────────┐
                               │                                               │
  POST /analyze  ────────────► │  API PROCESS  (app.py, port 3001)             │
  ◄─ analysis_id, instantly    │  accepts, validates, stores. NEVER analyses.  │
                               │           │                                   │
                               │           ▼  writes a job, status = queued    │
                               │   ┌───────────────────────┐                   │
                               │   │ MongoDB               │  the job document │
                               │   │ vision_lab_analyses   │  IS the queue     │
                               │   └───────────────────────┘                   │
                               │           ▲                                   │
                               │           │  claims it, works, writes back    │
                               │  WORKER PROCESS  (vision_lab/worker.py)       │
                               │  ffmpeg · UNISAL · OCR · Deepgram · Gemini    │
                               │           │                                   │
                               │           ▼  heatmap PNGs                     │
                               │   ┌───────────────────────┐                   │
  GET /analysis/{id} ────────► │   │ S3  vision-lab/...    │                   │
  ◄─ status, then full report  │   └───────────────────────┘                   │
                               └───────────────────────────────────────────────┘
```

**Why two processes.** One analysis is 45–100 s of solid CPU work. If the API process did it,
every other endpoint on this service — onboarding, Script Lab, sales calls — would freeze for
that time. The API only *accepts* the job; a separate worker does it.

**Why no Redis or Celery.** The MongoDB job document is the queue. The worker claims a job
with one atomic operation, so two workers can never take the same job.

### What the worker does, in order

The job's `status` changes at each stage, so a poll shows real progress:

```
probing           download the creative · read duration and size · sample frames
                  · find shot cuts · pull out the audio · delete the download
                  · erase the signed URL from the database (it is a credential)

analyzing_frames  for every frame:
                    UNISAL      where the eye goes — a map that sums to 100%
                    detection   text, faces, brand mark, CTA, prices
                    the join    how much of that 100% lands on each of them
                  then: heatmap → S3 · attention timeline · measured defects

transcribing      Deepgram on the audio, each line joined to the timeline

interpreting      15 triggers counted · Gemini reads the measurements and the
                  defect list and writes prose · every claim it made is checked

scoring           six scores and the overall, in Python
completed         the report is stored
```

### Division of labour

| Component | Responsible for | Never does |
|---|---|---|
| **Computer vision** | Measuring — where gaze goes, what is on screen, when, how big, how long | Interpreting |
| **Deepgram** | What is said, and when | — |
| **Gemini** | Interpreting — the guidance paragraph, the fix prose, ordinal ratings | **Producing a number** |
| **Python** | Scoring, validation, verifying every claim, persistence | Trusting the model |

**The AI model never produces a score.** It is never shown the scores, it may only return an
ordinal rating (`absent` / `weak` / `adequate` / `strong`) — a numeric one is rejected
outright — and it may only write about defects Python already found. Three consequences
that matter to you:

- the same measurements always produce the same score
- scores can be **recalculated** when management sets weights, without re-analysing any ad
- nothing written in an ad can move its own score

---

## 3. Prerequisites — read before integrating

### 3.1 It is not deployed yet 🔴

The code is finished; the **server plumbing is not**:

- `ecosystem.config.js` runs only `app.py`. The worker is a **second pm2 process**
  (`python -m vision_lab.worker`) and is not in it yet.
- `deploy/deploy.sh` does not restart the worker and does not check for ffmpeg.

Until both land, anything submitted on the server stays `queued` — there is no worker to pick
it up. This is Milestone E in `VISION_LAB_BUILD_STEPS.md`. Build against a local instance
(see [§12](#12-testing-it-yourself)) until then.

### 3.2 Server environment variables 🔴

| Variable | Why it is required |
|---|---|
| `API_KEY` | **If unset, every `/api/*` endpoint on the service is open to anyone.** Must be set in production |
| `MONGODB_URI` | The job queue and the reports |
| `AWS_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | Heatmaps, thumbnails and uploads |
| `VL_ENABLED=true` | Off by default. When false every Vision Lab endpoint answers 503 |
| `VL_ALLOWED_URL_HOSTS` | The S3 bucket host(s), comma-separated. Creative URLs on any other host are refused (`url_not_allowed`) |
| `VL_MODEL_DIR` | Folder holding `unisal.onnx` — **outside** the app directory, because `deploy/deploy.sh` runs `git reset --hard` |
| `DEEPGRAM_API_KEY` | Transcript. Without it reports come back with no transcript, stating why |
| `GEMINI_API_KEY` | Guidance and recommendations. Without it, measurements and scores still come back |
| `VL_OCR_LANGUAGES=eng+hin` | Hinglish creatives. English alone reads Devanagari as noise |

The full list, with defaults and explanations, is in `.env.example`.

### 3.3 System packages on the server 🔴

| Package | Without it |
|---|---|
| **ffmpeg** | The worker cannot decode video. It logs an error and falls back to **stub vision** — every score comes back null. Install: `apt install -y ffmpeg` |
| **tesseract-ocr** + **tesseract-ocr-hin** | No on-screen text: word counts are null, Cognitive Demand cannot be scored |
| **`unisal.onnx`** in `VL_MODEL_DIR` | sha256 `f18d76f49941ce1cad7eda4dce7d2dd71164edb108ccf18b142a3c01ac147d16`. A file with a different digest is **refused**, not loaded |
| **python-multipart** (in `requirements.txt`) | `POST /upload` is switched off; every other endpoint still works |

### 3.4 TLS 🟡

Vision Lab is served by the same host as the Sales Call Analyzer. The certificate issue in
`SALES_CALL_ANALYZER.md` §3.1 applies here unchanged.

### 3.5 S3 layout 🟡

| Prefix | Holds | Lifetime |
|---|---|---|
| `vision-lab/<analysis_id>/` | Heatmap and thumbnail PNGs | Removed by `DELETE /analysis/{id}` |
| `vision-lab-uploads/<date>/<id>/` | Files sent to `POST /upload` | **Not** removed by DELETE. Needs a bucket lifecycle rule — e.g. expire after 7 days |
| `vision-lab-test/` | The four reference test ads | Permanent |

### 3.6 Model licence 🟡

UNISAL's code and weights are Apache-2.0, but it was trained on datasets that include film
footage whose licensing for commercial use is not settled. Get legal sign-off before the
report is shown to customers. Attribution is recorded in `NOTICE.md`. This is one reason
`VL_ENABLED` stays false in production for now.

---

## 4. Conventions every call follows

### Authentication

Send `X-API-Key: <API_KEY>` on every request.

```json
HTTP 401
{ "detail": "Invalid or missing API key. Send it in the 'X-API-Key' header." }
```

**The key is server-side only.** The browser must never call Vision Lab directly — shipping
the key to the browser publishes it, and it grants Gemini spend across *every* endpoint on
this service, not just Vision Lab. The frontend calls the **backend**; the backend calls us.

### Failures are `200` with a stable `reason`

A problem with the *analysis* — an expired link, an unsupported file, a silent video — is a
normal `200` response carrying `reason` and `message`. HTTP error codes are reserved for
problems with the *request itself*:

| Code | When |
|---|---|
| `200` | Everything else, **including failed analyses** — check `status` and `reason` |
| `401` | Missing or wrong `X-API-Key` |
| `404` | Unknown `analysis_id`, or no analysis for that `creative_id` |
| `409` | Rescoring an analysis that is not `completed`, or whose measurements have expired |
| `413` | `POST /upload` file over the size limit — refused before it is received |
| `422` | Malformed request — invalid JSON (e.g. a line break pasted inside a URL), or `/upload` sent with no file |
| `503` | Vision Lab disabled, not installed, or its database not configured |

**Branch on `reason`, never on `message`.** Reason codes are stable identifiers; messages are
human copy and may be reworded. The full list is in [§10](#10-reason-codes-and-failure-handling).

### Units

| Kind | Format |
|---|---|
| Times inside the ad (`t`, `time`, `t_start`, `t_end`, `frame_time`) | Seconds from the start, float |
| Scores | Integer 0–100, or `null` with a `reason` |
| Attention values on the timeline | 0–100 index, float |
| Boxes (`box`) | **Fractional** `[x0, y0, x1, y1]`, each 0–1 — multiply by the displayed image size |
| Shares (`share`) | Fraction 0–1 of the frame's predicted attention — `0.4155` prints as "42%" |
| Dates (`created_at`, `updated_at`) | ISO 8601 **UTC** — see the timezone note in [§8](#8-frontend-rendering-guide) |

---

## 5. API reference

| # | Method | Path | Purpose |
|---|---|---|---|
| 1 | `POST` | `/api/vision-lab/analyze` | Submit a creative by URL. Returns `analysis_id` immediately |
| 2 | `POST` | `/api/vision-lab/upload` | Upload a file directly (form-data). **For testing** |
| 3 | `GET` | `/api/vision-lab/analysis/{analysis_id}` | Poll status, then read the **full report** |
| 4 | `GET` | `/api/vision-lab/analysis/by-creative/{creative_id}` | Latest analysis for a creative |
| 5 | `GET` | `/api/vision-lab/history` | Past analyses, newest first |
| 6 | `POST` | `/api/vision-lab/analysis/{analysis_id}/rescore` | Recompute scores under current config. No cost |
| 7 | `DELETE` | `/api/vision-lab/analysis/{analysis_id}` | Remove an analysis and its images |
| — | `GET` | `/health` | Includes a `vision_lab` block — is it configured, is the worker alive |

**Every screen of the prototype comes from #3.** One call returns the whole report; the
frontend does not stitch several endpoints together to draw one page.

---

### 5.1 `POST /api/vision-lab/analyze`

Submit a creative that already lives in S3. **Returns immediately** — about 20 ms, or ~2 s
for the first request after a restart — because it only queues the job. Poll #3 for the
result.

**Request** — `Content-Type: application/json`

```json
{
  "creative_id": "cre_8f21",
  "division": "directors_institute",
  "ad_number": "042",
  "creative": {
    "url": "https://aife-media-prod.s3.amazonaws.com/creatives/ad.mp4?X-Amz-Algorithm=...&X-Amz-Signature=...",
    "kind": "video",
    "mime_type": "video/mp4"
  },
  "brand_assets": {
    "brand_names": ["Director's Institute", "DI"],
    "wordmark_url": null
  },
  "options": { "force_reanalysis": false }
}
```

| Field | Type | Required | What it does |
|---|---|---|---|
| `creative_id` | string | **yes** | Your stable id for the creative |
| `creative.url` | string | **yes** in practice | Presigned **https** GET link. Its host must be in `VL_ALLOWED_URL_HOSTS` |
| `creative.kind` | `"video"` \| `"image"` | no | A hint. The file itself decides |
| `creative.mime_type` | string | no | A hint. A declared non-media type (e.g. `application/pdf`) is refused |
| `creative.duration_seconds`, `width`, `height`, `size_bytes`, `expires_at` | | no | Hints and diagnostics. `size_bytes` is part of the idempotency key |
| `ad_number` | string | no | Groups versions of one ad. Powers the History tab |
| `division` | string | no | The Division selector. Stored and filterable |
| `brand_assets.brand_names` | string[] | recommended | Brand detection matches these in on-screen text. **Without this or a wordmark, Brand Memory cannot be scored** |
| `brand_assets.wordmark_url` | string | no | Logo image, matched against every frame. **Read the security note in [§13](#13-known-limitations) first** |
| `brand_brain_id` | string | no | Accepted and stored. **Not used yet** — see [§13](#13-known-limitations) |
| `brand_id`, `campaign_id`, `funnel_stage` | string | no | Stored with the analysis for your own records. Do not change the result |
| `options.force_reanalysis` | bool | no | `true` = analyse again even if an identical request was already analysed |
| `options.sample_fps` | float | no | Override the frame rate (default 2.0). Leave unset |

**Response — accepted**

```json
{
  "analysis_id": "2a1178aea026451b93450daf80b44f7b",
  "creative_id": "cre_8f21",
  "status": "queued",
  "created_at": "2026-09-10T06:06:33.732890Z",
  "poll_url": "/api/vision-lab/analysis/2a1178aea026451b93450daf80b44f7b",
  "suggested_poll_interval_seconds": 5,
  "idempotent_hit": false,
  "thumbnail_url": null,
  "availability": { "available": true, "reason": null, "message": null },
  "reason": null,
  "message": null
}
```

`thumbnail_url` is **`null` for a new submission** — nothing has been downloaded yet, so
there is no frame to show. When you resubmit a creative that is **already analysed**, you get
the existing analysis back (`idempotent_hit: true`) and, once it has completed, a signed link
to a plain frame of the ad, valid for **1 hour**:

```json
{
  "analysis_id": "284efbfbd52f4398a65779b7b30907c9",
  "creative_id": "cre_ads1_postman",
  "status": "completed",
  "idempotent_hit": true,
  "thumbnail_url": "https://aife-media-prod.s3.amazonaws.com/vision-lab/284efbfb…/poster_55674.png?X-Amz-Signature=…",
  "…": "the rest as above"
}
```

The image is the report's **poster** — the heatmap's frame without the overlay, which is chosen
to skip black openings and fades ([§9](#9-complete-field-reference)). Analyses made before
posters existed return the strip frame nearest that moment instead.

**Response — refused** (still `200`; a failed analysis row is recorded so the refusal is
auditable)

```json
{
  "analysis_id": "…",
  "status": "failed",
  "availability": { "available": false, "reason": "url_not_allowed",
                    "message": "The creative URL is not on the allowed host list." },
  "reason": "url_not_allowed",
  "message": "The creative URL is not on the allowed host list."
}
```

Refused at submit: `no_creative` (no URL), `url_not_allowed` (host not allowlisted, or not
`https://`). Everything else about the file — format, size, duration, whether it downloads —
is decided by the worker and reported on the poll.

**Idempotency — free.** An identical request returns the existing analysis with
`idempotent_hit: true` and costs nothing. "Identical" means the same `creative_id`, the same
URL **ignoring its query string** (a fresh signature for the same file is the same file), the
same `size_bytes` and `ad_number`, and the same model and framework versions.

| Previous analysis is… | Resubmitting… |
|---|---|
| `completed` | returns it (`idempotent_hit: true`) |
| still running | returns it — no second job is started |
| started, but its worker went quiet > 30 min | starts a new one |
| `failed` | **starts a new one** — resubmitting is how you retry |
| any, with `force_reanalysis: true` | starts a new one |

---

### 5.2 `POST /api/vision-lab/upload`

Upload a file straight to Vision Lab. `Content-Type: multipart/form-data`.

> **For testing, and for callers with a file but no bucket access.** In production the
> file should go browser → S3 directly (see [§7](#7-backend-integration-guide)). Sending a
> 150 MB video through the AI service ties up a connection on it for the whole transfer.

| Form field | Type | Required | What it does |
|---|---|---|---|
| `file` | file | **yes** | `.mp4 .mov .webm .m4v .jpg .jpeg .png .webp`, up to 400 MB |
| `analyze` | `true` / `false` | no | `true` also queues the analysis in this same call. Default `false` |
| `creative_id` | text | no | Defaults to a readable id such as `cre_ad3_1a2b3c4d` |
| `ad_number`, `division` | text | no | As in 5.1 |
| `brand_names` | text | no | Comma-separated: `Director's Institute,DI` |

**Response** — real, with the signature redacted

```json
{
  "uploaded": true,
  "reason": null,
  "message": null,
  "upload": {
    "object_key": "vision-lab-uploads/2026-09-10/cbaf90eb884c4dbeb011cc2221f873e2/upload-test-clip.mp4",
    "filename": "Upload Test Clip.mp4",
    "size_bytes": 878642,
    "mime_type": "video/mp4",
    "kind": "video",
    "creative_url": "https://aife-media-prod.s3.amazonaws.com/vision-lab-uploads/…?X-Amz-Signature=…",
    "creative_url_expires_in_seconds": 21600
  },
  "analysis": {
    "analysis_id": "a0d5f9c342824cbd86cc66748c9fbd1e",
    "creative_id": "cre_upload_live",
    "status": "queued",
    "poll_url": "/api/vision-lab/analysis/a0d5f9c342824cbd86cc66748c9fbd1e",
    "suggested_poll_interval_seconds": 5,
    "idempotent_hit": false,
    "…": "exactly what POST /analyze returns"
  }
}
```

- `upload.creative_url` is the link `/analyze` takes. Valid **6 hours** — longer than a
  heatmap link, because the worker only downloads the file when it reaches the job.
- `analysis` is `null` unless `analyze=true`.

| Outcome | Response |
|---|---|
| Stored | `200`, `uploaded: true` |
| Not a supported format, or the name and declared type disagree | `200`, `reason: unsupported_format` |
| Empty file | `200`, `reason: creative_unusable` |
| S3 not configured | `200`, `reason: object_storage_not_configured` |
| S3 write failed | `200`, `reason: upload_failed` — retry |
| Over 400 MB | **`413`**, `reason: creative_too_large` — refused before the body is accepted |
| No file in the request | **`422`** — `file`: "Field required" |

---

### 5.3 `GET /api/vision-lab/analysis/{analysis_id}`

Poll this. It returns one of three shapes.

**While running** — a small envelope. `scores` is always `null` until the analysis completes;
a half-finished analysis never shows numbers.

```json
{
  "analysis_id": "2a1178aea026451b93450daf80b44f7b",
  "creative_id": "cre_8f21",
  "ad_number": "042",
  "division": "directors_institute",
  "status": "analyzing_frames",
  "availability": { "available": true, "reason": null, "message": null },
  "reason": null,
  "message": null,
  "media": { "kind": "video", "duration_seconds": 82.48, "frames_analyzed": 120, "…": "…" },
  "scores": null,
  "created_at": "2026-09-10T06:06:33.732000",
  "updated_at": "2026-09-10T06:06:40.118000",
  "attempts": 1,
  "poll_url": "/api/vision-lab/analysis/2a1178aea026451b93450daf80b44f7b",
  "suggested_poll_interval_seconds": 5,
  "fallback": false
}
```

`media` fills in once the video has been read (from `analyzing_frames` onwards) — you can
show the duration while the rest is processing.

**When completed** — the **full report**, documented field by field in
[§9](#9-complete-field-reference). A real one: `tests/fixtures/vision_lab/example_report.json`.

**When failed** — the envelope above with `status: "failed"`,
`availability.available: false`, a `reason`, `scores: null` and `fallback: true`. A failed
analysis never invents a scorecard.

**Image links are signed fresh on every call** and are valid for **1 hour**. An analysis
opened three weeks later still returns working images — as long as you fetch the report again
rather than reuse an old URL.

`404` — unknown `analysis_id`.

---

### 5.4 `GET /api/vision-lab/analysis/by-creative/{creative_id}`

The most recent analysis for a creative — for when you kept the `creative_id` but not the
`analysis_id`. Same three shapes as 5.3.

`404` — `{"detail": "no analysis exists for that creative_id"}`.

---

### 5.5 `GET /api/vision-lab/history`

The History tab. Newest first.

| Query param | Default | Notes |
|---|---|---|
| `ad_number` | — | The version history of one ad |
| `division` | — | |
| `creative_id` | — | |
| `limit` | `25` | 1–100 |

```json
{
  "items": [
    {
      "analysis_id": "2a1178aea026451b93450daf80b44f7b",
      "creative_id": "cre_8f21",
      "ad_number": "042",
      "division": "directors_institute",
      "status": "completed",
      "overall_score": 62,
      "thumbnail_url": "https://aife-media-prod.s3.amazonaws.com/vision-lab/2a1178ae…/poster_55674.png?X-Amz-Signature=…",
      "created_at": "2026-09-10T06:06:33.732000"
    },
    {
      "analysis_id": "252df957b0ad424c90c86fd7bd6b940b",
      "creative_id": "cre_8f21",
      "ad_number": "042",
      "division": "directors_institute",
      "status": "failed",
      "overall_score": null,
      "thumbnail_url": null,
      "created_at": "2026-09-10T03:48:42.245000"
    }
  ],
  "count": 2
}
```

`overall_score` and `thumbnail_url` are `null` for anything not completed. `thumbnail_url` is a signed plain frame of the ad, valid 1 hour — enough to draw a History list with previews without fetching every report. Failed analyses are included — hide or
grey them as you see fit.

---

### 5.6 `POST /api/vision-lab/analysis/{analysis_id}/rescore`

Recompute every score from the stored per-frame measurements under the **current**
configuration. No ffmpeg, no model, **no Gemini call, no cost** — about 0.05 s.

Use it when management sets real metric weights or score bands: call it for existing analyses
and they move onto the new configuration without re-processing a single frame.

- Returns the full report, recomputed.
- Gemini's stored ratings are reused, so Clarity keeps its blend and nothing is re-billed.
- Recommendations are re-anchored to the recomputed defects; one whose defect no longer exists
  under the new configuration is dropped.

| Code | When |
|---|---|
| `404` | Unknown `analysis_id` |
| `409` | `"Only a completed analysis can be rescored."` |
| `409` | Measurements expired (kept 180 days) — re-run the analysis instead |

---

### 5.7 `DELETE /api/vision-lab/analysis/{analysis_id}`

Call this when a customer deletes a creative.

```json
{ "deleted": true, "analysis_id": "2a1178aea026451b93450daf80b44f7b", "images_removed": 7 }
```

Removes the analysis, its stored measurements and its heatmap/thumbnail images.
**Does not touch the creative itself** — that is yours — and does not remove a file sent to
`/upload` (a bucket lifecycle rule handles those). `404` if unknown.

---

### 5.8 `GET /health` — the `vision_lab` block

```json
"vision_lab": {
  "available": true,
  "enabled": true,
  "storage": "configured",
  "worker": { "seen_seconds_ago": 3.2, "healthy": true, "in_flight": 1 },
  "queue": { "queued": 3, "oldest_waiting_seconds": 142.0 },
  "model": "configured",
  "framework_version": "vision_v1",
  "transcription": "configured",
  "interpretation": "configured"
}
```

| Field | Meaning |
|---|---|
| `available` | The code loaded |
| `enabled` | `VL_ENABLED` is true |
| `storage` | MongoDB is configured |
| `worker.healthy` | `true` — a job in flight has a recent heartbeat. `false` — a job is in flight and the worker has gone quiet. **`null` — nothing in flight, so unknown, not broken** |
| `worker.in_flight` | Jobs a worker is processing right now. **Queued jobs are not counted** |
| `worker.reason` | `jobs_waiting_unclaimed` — jobs have waited over a minute and none is being processed: **no worker is running**. A running worker that is idle picks a job up within seconds |
| `queue.queued` | Jobs waiting for a worker |
| `queue.oldest_waiting_seconds` | How long the oldest has waited — your backlog, in seconds |
| `model` | A saliency model is configured |
| `transcription` / `interpretation` | Deepgram / Gemini keys present. `"not_configured"` means reports will come back without a transcript / without written guidance — **not** that the ad had nothing wrong with it |

Worth a monitor: `healthy: false`. With `in_flight > 0` the worker died mid-job; with `reason: jobs_waiting_unclaimed` no worker is running at all. A large `oldest_waiting_seconds` with a healthy worker is not a failure — it is load, and the fix is more workers (§13, #11).

---

## 6. Job lifecycle and polling

| `status` | Terminal | Meaning | Suggested progress copy |
|---|---|---|---|
| `queued` | | Waiting for the worker | "Waiting to start…" |
| `probing` | | Downloading and reading the file | "Reading your video…" |
| `analyzing_frames` | | Predicting attention frame by frame — **the longest stage** | "Predicting where viewers look…" |
| `transcribing` | | Deepgram | "Transcribing the voiceover…" |
| `interpreting` | | Triggers, Gemini, evidence checks | "Writing recommendations…" |
| `scoring` | | Computing the six scores | "Scoring…" |
| `completed` | ✓ | Full report available | — |
| `failed` | ✓ | See `reason` | — |
| `skipped` | ✓ | Reserved for "analysed but deliberately not scored" (e.g. a minimum duration). Not produced with the current configuration | — |

**Do not assume you will see every status.** Stages can be quick enough to fall between two
polls, and a silent video has nothing to transcribe.

### Polling rules

1. Wait `suggested_poll_interval_seconds` (currently **5**) between polls.
2. Stop on `completed`, `failed` or `skipped`.
3. While running, `scores` is `null` — render progress, not numbers.
4. A job a worker has **started** and then stops reporting on for **30 minutes** is marked
   `failed` with `reason: processing_interrupted` — usually a server restart mid-job.
   Resubmit to retry. **A job still waiting in the queue is never failed for waiting**,
   however long the queue: it stays `queued` and is analysed when its turn comes.
5. Give up client-side after ~10 minutes and show a "still working, check back" state; the
   analysis continues regardless.

### How long it takes

| Creative | Measured |
|---|---|
| ~80 s video ad | 45–100 s |

Measured on real ScaleSerum ads, one job at a time, on a laptop CPU — including the Deepgram
and Gemini calls. A slow connection to S3 adds to it. A shorter video samples fewer frames and
finishes sooner.

The worker processes **one job at a time**, so with three jobs queued, the third waits for the
first two.

---

## 7. Backend integration guide

### The call sequence

```
Browser  ──▶  POST /api/creatives/:id/vision-lab     (your route, your session auth)
Backend  ──▶  POST /api/vision-lab/analyze           (X-API-Key, server to server)
```

1. **Frontend** asks the backend for an upload target.
2. **Backend** presigns an S3 **PUT** and returns it.
3. **Frontend** PUTs the file directly to S3. The progress bar lives here — we report no
   upload progress.
4. **Backend** presigns a **GET** for the same object and calls `POST /analyze` with it.
   Stores the returned `analysis_id` on the creative record and returns it to the frontend.
5. **Frontend** polls the **backend** every ~5 s; the backend forwards to
   `GET /api/vision-lab/analysis/{id}`.
6. On a terminal status, the frontend renders.

### Rules

| Do | Why |
|---|---|
| Sign the GET URL for **at least 6 hours** | The worker downloads the file when it reaches the job, not when you submit. With a queue ahead, a 1-hour link can expire first → `creative_unreachable` |
| Use `https://` and a bucket host in `VL_ALLOWED_URL_HOSTS` | Anything else is refused as `url_not_allowed` |
| Always send `brand_assets.brand_names` | Without a name or a wordmark, Brand Memory cannot be scored |
| Store `analysis_id` on the creative | `by-creative` exists as a fallback, not the primary lookup |
| Send `ad_number` for re-uploads of the same ad | It groups versions in History |
| Retry a failure by submitting again | Failed analyses are never reused |
| Call `DELETE` when a creative is deleted | Removes our images and measurements |
| Pass the report through **unmodified** | Every field is part of the contract; dropping one breaks a screen |
| **Re-fetch** the report to show images | Image URLs expire after 1 hour. Never cache `image_url` |

| Do not | Why |
|---|---|
| Expose `API_KEY` to the browser | It grants access to every endpoint on the service |
| Send `wordmark_url` built from user input | See [§13](#13-known-limitations) — it is not yet allowlist-checked |
| Log a presigned URL | It is a credential until it expires |
| Use `POST /upload` from the frontend in production | It exists for testing (see 5.2) |

### What to store on your side

| Field | Where from |
|---|---|
| `analysis_id` | `POST /analyze` response |
| `status` | Latest poll — so the UI can show "analysing" without calling us |
| `overall.score` | The completed report — for list views, so History does not need a round trip per row |

Store the rest by fetching the report when it is viewed, not by copying it — image links in a
copy go stale within the hour.

### When management changes the scoring

Scoring configuration lives in `vision_lab/vision_framework.json`. After it changes, call
`POST /rescore` for each analysis you want on the new configuration. Nothing about the
response shape changes — `overall.band`, currently `null`, simply starts carrying a value once
bands are set.

---

## 8. Frontend rendering guide

### Rules that matter

1. **Say "AI-predicted attention, not eye-tracking"** on every screen that shows these
   numbers. The prototype already does; keep it.
2. **Cognitive Demand is lower-is-better.** A 21 is *good*. Read `scores.<id>.direction` —
   `"lower_better"` — and invert the colour scale for it. Do not hard-code which metric it is.
3. **A null score is not zero.** Show "—" with the `reason` explained (table in §10), and
   never average a null in. `overall.metrics_scored` says how many of the six the overall is
   built from; say so if it is fewer than six.
4. **No score bands exist yet.** `overall.band` is `null` with `band_reason:
   "thresholds_not_configured"`. Show the number; **do not invent** Good / Average / Poor
   cut-offs in the UI — management has not set them.
5. **Heatmap markers come from fractional boxes.** Centre of a marker =
   `((x0 + x1) / 2 × displayed width, (y0 + y1) / 2 × displayed height)`. The rendered PNG
   already carries the numbers; draw your own markers only if you want them interactive.
6. **Label markers from `element` / `element_text`** — "① 42% of attention · face",
   "② 17% · brand mark". `element: null` means nothing we detect sits under that hotspot
   (product footage, b-roll): show the marker, without a label.
7. **Re-fetch for images.** `image_url` expires after 1 hour. If an image 403s, fetch the
   report again for a fresh link.
8. **The timeline is an index, not a percentage of viewers.** `timeline.basis` is
   `"derived_composite"`. Label the axis "Attention index (0–100)" — never "% watching".
9. **`not_applicable` ≠ `absent` ≠ `unsupported`** for triggers. The ad had no opportunity /
   the ad did not do it / we could not stand behind the claim. Three different visuals.
10. **A trigger with `reason: "awaiting_interpretation"` was not assessed** — Gemini did not
    run. Show it as "not assessed", not as absent.
11. **Choice Overload is the one trigger where `present` is bad news.** Read `polarity` —
    `"defect"` — rather than hard-coding the trigger id.
12. **Absent blocks say why.** `transcript.available: false` → hide the transcript panel and
    show `transcript.message`. `interpretation.available: false` → no guidance or
    recommendations; show the measured `defects` list instead, which is always there.
13. **Timezone.** `created_at` / `updated_at` on `GET` responses are UTC **without a trailing
    `Z`** (`"2026-09-10T06:06:33.732000"`). JavaScript's `new Date()` reads that as *local*
    time — 5 h 30 min off in India. Append `Z` before parsing.
14. **`stub` is always `false`.** A leftover from development; ignore it.

### Prototype screen → field

| Screen element | Field |
|---|---|
| Attention Report image | `heatmap.image_url` (the frame is at `heatmap.frame_time` seconds) |
| Numbered hotspots ①②③ | `heatmap.peaks[]` → `rank`, `box`, `share`, `element`, `element_text` |
| Frame strip | `thumbnails[]` → `t`, `image_url` |
| Preview image (History list, upload card) | `thumbnail_url` on `/history` and `/analyze`, or `poster.image_url` in the report |
| Score out of 100 | `overall.score` |
| Guidance paragraph | `summary` |
| Six metric bars | `scores.attention`, `.focus`, `.cognitive_demand`, `.clarity`, `.brand_memory`, `.engagement` → `score`, `label`, `direction` |
| Key Moments strip | `key_moments[]` → `time`, `label`, `attention`, `note` |
| Attention over time graph | `timeline.points[]` → `t`, `attention`, `engagement` |
| Red bands on the graph | `timeline.weak_zones[]` → `t_start`, `t_end` |
| Pins on the graph | `timeline.markers[]` → `t`, `severity`, `text` |
| Transcript, time-stamped | `transcript.segments[]` → `t`, `end`, `text`, `attention`, `in_weak_zone` (the ⚠ icon) |
| Fix recommendations | `recommendations[]` → `title`, `why`, `fix`, `scores_impacted`, `anchor.t_start`–`t_end`, `severity` |
| Psychological triggers | `psychology.triggers[]`, totals in `psychology.coverage` |
| "The ad is built around…" | `key_message.element`, `key_message.quote` |
| Other observations (optional) | `observations[]` — things Gemini noticed with **no measured defect** behind them. Show below the recommendations, visibly lighter |

### Suggested layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Attention Report · Ad #042          AI-predicted attention, not eye-tracking │
├──────────────────────────────┬───────────────────────────────────────────┤
│  heatmap.image_url           │  67 / 100          (overall.score)          │
│  with ①②③ from peaks[]      │  summary …                                  │
│                              │  Attention ████████░░ 84                    │
│  thumbnails[] strip          │  Focus     █████░░░░░ 46  ⓘ reason          │
│                              │  Cognitive ██░░░░░░░░ 21  lower is better   │
│                              │  Clarity · Brand Memory · Engagement        │
├──────────────────────────────┴───────────────────────────────────────────┤
│  Key moments   KEY 0.0s · PEAK 74.2s · …                                   │
│  Attention over time   ── attention  ── engagement   ▒ weak zones          │
├──────────────────────────────┬───────────────────────────────────────────┤
│  Transcript                  │  Fix recommendations                        │
│  1.2s  73  Aspiring…         │  1 · MEDIUM · 8.9s–12.4s                    │
│  2.9s  64  did you know…  ⚠  │    title / why / fix / scores impacted      │
├──────────────────────────────┴───────────────────────────────────────────┤
│  Psychological triggers  (4 present · 2 weak · 5 absent · 4 n/a)            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Complete field reference

The completed report, top to bottom. Every field is present in every completed report; values
may be `null`, with a reason alongside.

### Top level

| Field | Type | Meaning |
|---|---|---|
| `analysis_id`, `creative_id`, `ad_number`, `division` | string | Identity, echoed from the request |
| `status` | string | `"completed"` |
| `availability` | object | `{available, reason, message}` |
| `media` | object | The file as analysed |
| `vision` | object | Which model produced the numbers |
| `overall` | object | Overall score |
| `summary` | string | The guidance paragraph. `""` when interpretation did not run |
| `scores` | object | The six metrics |
| `measurements` | object | Raw counts behind the scores |
| `heatmap`, `thumbnails` | object, array | The Attention Report images |
| `poster` | object | A plain frame of the ad for previews — `{frame_time, object_key, image_url}` |
| `key_moments` | array | Peak / hero / key / weak |
| `timeline` | object | The attention-over-time curve. **`null` for an image** |
| `transcript` | object | Time-stamped lines, or why there are none |
| `key_message` | object | What the ad is built around. `null` when interpretation did not run |
| `psychology` | object | The 15 triggers |
| `defects` | array | Problems Python measured, worst first |
| `recommendations` | array | Written fixes, one per defect Gemini wrote about |
| `observations` | array of string | Gemini's notes with no measured defect behind them |
| `interpretation` | object | Whether the Gemini pass ran, and what it tried to say that was refused |
| `notes` | array of string | Context reason codes — e.g. `no_brand_brain`, `no_audio_stream` |
| `versions` | object | Framework, prompt and model versions |
| `config_disclosure` | object | Which business values are still placeholders |
| `created_at`, `updated_at` | string | UTC, see rule 13 in §8 |
| `stub` | bool | Always `false` |

### `media`

| Field | Meaning |
|---|---|
| `kind` | `"video"` or `"image"` |
| `duration_seconds`, `width`, `height`, `codec`, `source_fps`, `size_bytes` | Read from the file |
| `frames_analyzed` | Frames sampled (max 120) |
| `sample_fps` | Rate actually used |
| `requested_sample_fps`, `sample_fps_thinned` | A long ad gets a lower rate so the **whole** ad is covered — `true` when that happened |
| `coverage_seconds`, `coverage_fraction` | How much of the ad was analysed. Should be `1.0` |
| `shot_count` | Cuts detected |
| `has_audio` | `false` → no transcript |

### `vision`

`saliency_method` (`"onnx"`), `saliency_model` (`"UNISAL"`), `saliency_trained` (`true` —
`false` would mean a classical fallback, **not suitable to show a client**), `saliency_licence`.

### `overall`

```json
{ "score": 67, "band": null, "band_reason": "thresholds_not_configured",
  "weighting": "equal_unweighted_placeholder", "metrics_scored": 6, "metrics_missing": [] }
```

The mean of the six metrics with **Cognitive Demand inverted** (100 − score) first, currently
equally weighted. A metric that could not be scored is **excluded**, not counted as zero —
`metrics_missing` names it. When nothing could be scored: `score: null` with a `score_reason`.

### `scores.<metric>`

```json
{ "score": 46, "label": "attention on the key message", "direction": "higher_better",
  "basis": "measured_partial", "reason": "key_message_not_on_screen",
  "signals": { "copy_attention_vs_chance": 1.92, "mean_competing_peaks": 1.98 } }
```

| Metric | Label | Direction | What drives it |
|---|---|---|---|
| `attention` | captures initial gaze | higher better | How concentrated predicted gaze is in the **first 3 seconds** (the hook), confirmed by the strongest hotspot's share |
| `focus` | attention on the key message | higher better | Gaze on the on-screen copy **relative to how much of the frame it covers** (1.0 = chance, 3.0 = strongly drawn). Measured at the key message's moment when that message is written on screen. Penalised when many hotspots compete |
| `cognitive_demand` | effort to understand | **lower better** | Share of shots holding more words than can be read at **4 words/second** in their screen time, how badly the worst overruns, cut rate, text density |
| `clarity` | message + CTA are unambiguous | higher better | CTA legibility, seconds on screen, weak wording — **blended 50/50 with Gemini's ordinal rating** when available (`basis: "hybrid"`) |
| `brand_memory` | likelihood brand is remembered | higher better | When the brand first appears, how long it is shown, and whether it is shown **while attention is high**. Needs the brand to be detected |
| `engagement` | pull to keep watching | higher better | The engagement curve's mean, the hook's strength, time spent in dead zones |

| `basis` | Meaning |
|---|---|
| `measured` | Fully computed from measurements |
| `measured_partial` | Computed, but from less than the full signal — `reason` says what was missing |
| `hybrid` | Measurements blended with an ordinal AI rating (Clarity only) |

`signals` are the raw inputs, for tooltips. **Treat the key set as open** — do not hard-code
it; it can grow, and a key is omitted when its value is null.

| Metric | `signals` keys |
|---|---|
| `attention` | `hook_seconds` — the hook window (3.0) · `mean_concentration` — how tightly attention is concentrated over the hook · `strongest_region_share` — the top hotspot's share |
| `focus` | `copy_attention_vs_chance` — gaze on the copy ÷ the area it covers (1.0 = chance) · `mean_competing_peaks` · and when measured at the key message: `key_message_t`, `key_message_element`, `frames_at_key_message` |
| `cognitive_demand` | `reading_speed_words_per_second` (4.0) · `overloaded_shots` — how many · `worst_overrun_ratio` — seconds needed ÷ seconds available for the worst shot · `cut_rate_per_minute` |
| `clarity` | `cta_seconds_on_screen` · with the AI rating: `llm_rating` (`absent` … `strong`), `llm_blend` (0.5) · without it: `cta_weak` |
| `brand_memory` | `first_appearance_seconds` · `first_appearance_fraction` · `exposure_seconds` · `attention_peak_overlap` — share of the brand's screen time that falls in high-attention moments · `detected` |
| `engagement` | `mean_engagement` · `hook_strength` — how strongly the first 3 s hold · `dead_zone_seconds` — time spent in weak zones · `weak_zones` — how many |

### `measurements`

Raw counts: `frames`, `duration_seconds`, `shot_count`, `cut_rate_per_minute`,
`ocr_operational_fraction`, `frames_with_text_fraction`, `mean_concentration`,
`max_words_on_screen`, `reading_speed_words_per_second`, `prices_detected`,
`faces_detected`, `sample_fps`, plus:

- `overloaded_shots[]` — `{shot, words, seconds_available, seconds_needed, t_start, t_end}`
- `brand` — `{detected, first_appearance_seconds, first_appearance_fraction, exposure_frames, exposure_seconds}`
- `cta` — `{detected, weak}`

### `heatmap` and `thumbnails`

```json
"heatmap": {
  "frame_time": 55.674,
  "object_key": "vision-lab/07dc…/heat_55674.png",
  "image_url": "https://…?X-Amz-Signature=…",
  "peaks": [
    { "rank": 1, "box": [0.4047, 0.3281, 0.6141, 0.5042], "share": 0.4155,
      "element": "face", "element_text": null },
    { "rank": 2, "box": [0.225, 0.6646, 0.3937, 0.7531], "share": 0.112,
      "element": null, "element_text": null }
  ]
},
"thumbnails": [ { "t": 0.0, "object_key": "vision-lab/07dc…/thumb_0.png", "image_url": "https://…" } ]
```

- The frame is the **most representative** one: blank frames, fades and white flashes between
  shots are excluded, so the picture always shows actual ad content.
- Up to 3 peaks, strongest first. `share` is that hotspot's fraction of the frame's attention.
- `element` is one of `brand_mark`, `call_to_action`, `face`, `on_screen_text`, or `null`.
  Brand mark and CTA win over plain text on the same pixels. `element_text` carries the words
  (text, CTA) or the brand name.
- `image_url` is `null` if S3 is not configured — the report still carries `peaks`.
- `thumbnails`: up to 6 frames across the ad.
- `poster`: the heatmap's frame **without** the overlay — the image `thumbnail_url` points to. Signed on every read like the rest. `object_key` is `null` for analyses made before posters existed, and for a server without S3.

### `key_moments[]`

`{time, label, attention, note}`. `label` is one of:

| Label | Rule |
|---|---|
| `peak` | Highest point of the attention index |
| `hero` | Strongest moment in the second half |
| `key` | First frame where on-screen copy takes the majority of gaze |
| `weak` | Inside a weak zone |

Blank frames are never chosen as `peak` or `hero`. Not every label appears in every report.

### `timeline`

| Field | Meaning |
|---|---|
| `points[]` | `{t, attention, engagement}` — one per sampled frame (up to 120). Draw both as lines |
| `unit` | `"index_0_100"` |
| `basis` | `"derived_composite"` — see rule 8 |
| `basis_note` | Why: a saliency map always sums to 100%, so it cannot say whether a frame held the viewer. The curve is a composite of six measured signals — concentration, motion, faces, gaze stability, text load, novelty |
| `weak_zones[]` | `{t_start, t_end, seconds, min_attention, mean_attention}` — spans where attention stayed low |
| `weak_zone_rule` | `{threshold: 35.0, min_seconds: 1.0, configured: false}` — a placeholder rule |
| `markers[]` | One per weak zone: `{t, severity, text}` — e.g. `"Attention falls to 26 for 2.5s"` |
| `coefficients`, `calibration` | `"uniform_placeholder"`, `"provisional_absolute"` — values are comparable across ads, but not yet calibrated against campaign results |

### `transcript`

```json
{
  "available": true, "reason": null, "message": null, "language": "en",
  "stats": { "lines": 27, "words": 171, "seconds_of_speech": 64.45,
             "words_per_second": 2.65, "mean_line_attention": 64.9,
             "lines_in_weak_zones": 0 },
  "segments": [
    { "t": 1.2, "end": 2.34, "text": "Aspiring directors,", "attention": 73,
      "in_weak_zone": false }
  ]
}
```

- `attention` per line is the attention index averaged over that line's own time span — a
  join, not a second prediction.
- `in_weak_zone: true` → the line overlaps a weak zone → the ⚠ icon.
- When there is none: `available: false`, `segments: []`, and a `reason` —
  `no_audio_stream` (silent ad — normal), `transcript_empty`, `transcription_not_configured`,
  `transcription_provider_error`.
- Hinglish comes back in mixed script, e.g. `"और दूसरा client बोलता है"`. Use a font that
  renders Devanagari.

### `key_message`

```json
{ "element": "Transforming professional seniority into boardroom credentials",
  "t": 70.785, "quote": "has been helping them turn their years of experience",
  "carrier": "voiceover", "verified": true }
```

`carrier` — `on_screen_text`, `voiceover`, `both` or `visual` — which channel carries the
claim. When it is spoken, there is nothing on screen for the eye to land on, so Focus is
measured across the whole ad instead and says `key_message_not_on_screen`.
`model_quote_rejected: true` (when present) — Gemini mistyped the quote and the real
transcript line was substituted. Common on Hinglish ads; not an error.

### `psychology`

```json
{
  "triggers": [
    { "id": "halo_effect", "number": 1, "name": "The Halo Effect",
      "subtitle": "The Power of First Impressions", "status": "present",
      "rating": "strong", "polarity": "positive", "detection": "hybrid",
      "scored": true, "reason": null, "evidence": [],
      "measured": { "hook_attention": 76.2, "rest_of_ad_attention": 63.3, "hook_seconds": 3.0 },
      "note": "the first 3s score 76 against 63 for the rest" }
  ],
  "coverage": { "present": 4, "weak": 2, "absent": 5, "not_applicable": 4, "measured": 6 },
  "unsupported": 0,
  "version": "triggers_v1",
  "affects_overall_score": false
}
```

| Field | Meaning |
|---|---|
| `status` | `present` · `weak` · `absent` · `not_applicable` · `unsupported` |
| `rating` | `strong` · `adequate` · `weak` · `absent` · `null` — how strongly the trigger **shows** |
| `polarity` | `positive`, or `defect` (Choice Overload — showing it strongly is bad) |
| `detection` | `measured` (counted), `judged` (Gemini), `hybrid` |
| `measured` | The counts behind a measured trigger |
| `evidence[]` | `{t, quote, source}` — every quote is checked against the transcript |
| `reason` | Why there is no verdict — see below |
| `judged_by` | `"llm"` when Gemini rated it |
| `scored` | `false` only for Blind-Spot Bias — reported, never scored |
| `affects_overall_score` | `false` — triggers do not change `overall.score` today |

| # | Trigger | Detection | Not applicable when |
|---|---|---|---|
| 1 | The Halo Effect — first impressions | hybrid | |
| 2 | The Serial Position Effect — first and last matter most | measured | |
| 3 | The Recency Effect — recent info carries more weight | measured | |
| 4 | The Mere Exposure Effect — familiarity breeds likability | measured | no brand detected |
| 5 | Loss Aversion — fear of missing out | hybrid | |
| 6 | The Compromise Effect — offering 3 choices | measured | no options presented |
| 7 | Anchoring — setting expectations with price | hybrid | no price shown |
| 8 | Choice Overload — less is more *(polarity: defect)* | measured | |
| 9 | The Framing Effect — positioning your message | judged | |
| 10 | The IKEA Effect — value increases with involvement | judged | |
| 11 | The Pygmalion Effect — high expectations | judged | |
| 12 | Confirmation Bias — reinforcing existing beliefs | judged | no Brand Brain |
| 13 | The Peltzman Effect — lowering perceived risk | hybrid | |
| 14 | The Bandwagon Effect — people follow the crowd | measured | |
| 15 | Blind-Spot Bias — biases that go unnoticed *(unscored)* | judged | no Brand Brain |

What each measured trigger counted, in `measured`:

| Trigger | `measured` keys |
|---|---|
| Halo Effect | `hook_attention`, `rest_of_ad_attention`, `hook_seconds` |
| Serial Position | `opening_words`, `closing_words`, `opening_close_similarity`, `threshold` |
| Recency | `opening_close_similarity` — how much of the opening's wording the close reuses |
| Mere Exposure | `first_appearance_seconds`, `exposure_seconds`, `share_of_runtime` |
| Loss Aversion | `deadline_phrases`, `scarcity_phrases` — the phrases matched |
| Compromise Effect | `max_text_blocks_on_a_frame` |
| Anchoring | `prices_detected` — the prices read on screen |
| Choice Overload | `distinct_ctas`, `mean_competing_peaks` |
| Peltzman Effect | `risk_reversal_phrases` — guarantees, trials, refunds found |
| Bandwagon | `adoption_numerals`, `spelled_quantities` (e.g. "four thousand two hundred"), `social_proof_phrases` |

Judged triggers — Framing, IKEA, Pygmalion, Confirmation Bias, Blind-Spot Bias — have an
empty `measured`; their support is in `evidence`.

Trigger `reason` values: `awaiting_interpretation` (not assessed — Gemini did not run),
`no_brand_detected`, `no_options_presented`, `no_price_shown`, `no_brand_brain`,
`report_level_observation_only` (Blind-Spot Bias), `evidence_failed_verification`,
`no_evidence_cited`, `no_timeline`, `no_transcript_or_on_screen_text`.

### `defects[]` — what Python measured

```json
{ "defect_id": "overloaded_slide", "severity": "medium",
  "t_start": 8.94, "t_end": 12.37, "scores_impacted": ["cognitive_demand", "focus"],
  "measured": { "words": 24, "seconds_available": 4.12, "seconds_needed": 6.0,
                "seconds_short": 1.88, "reading_speed_words_per_second": 4.0 },
  "note": "24 words need 6.0s at 4 words/second but are held for 4.1s - …" }
```

Sorted worst first (severity, then duration). `t_start`/`t_end` are `null` for a
whole-ad finding.

| `defect_id` | Meaning | Scores hurt | Severity |
|---|---|---|---|
| `dead_zone` | Attention stays low for a sustained span | engagement, attention | high ≥ 2 s · medium ≥ 1 s · else low |
| `overloaded_slide` | A shot holds more words than can be read in its screen time | cognitive_demand, focus | by how far it overruns |
| `late_brand_appearance` | The brand first appears late in the ad | brand_memory | high if after 60% of the runtime |
| `brand_never_attended` | The brand is on screen, but never while attention is high | brand_memory | medium |
| `brand_not_detected` | No brand mark or name found anywhere | brand_memory | high |
| `weak_cta` | The call to action uses weak wording ("learn more", "submit") | clarity | medium |
| `cta_not_detected` | No call to action on screen | clarity | high |
| `competing_peaks` | Too many attention centres compete per frame | focus, cognitive_demand | medium |
| `flat_hook` | The first 3 s hold attention no better than the rest of the ad | attention, engagement | high if well below the rest |

### `recommendations[]` — what to fix

```json
{
  "rank": 2,
  "defect_id": "overloaded_slide",
  "title": "Shot 2 shows 24 words for 4.12s - 1.88s short of readable time",
  "why": "Between 8.94s and 12.37s, shot 2 displays 24 words. At the standard baseline of 4.0 words per second, this quantity of copy requires 6.0s to comprehend, leaving viewers 1.88s short…",
  "fix": "Reduce on-screen copy on shot 2 from 8.94s to 12.37s to a concise graphic card of no more than 16 words…",
  "scores_impacted": ["cognitive_demand", "focus"],
  "severity": "medium",
  "verified": true,
  "anchor": { "defect_id": "overloaded_slide", "t_start": 8.94, "t_end": 12.37,
              "measured": { "words": 24, "seconds_available": 4.12, "seconds_needed": 6.0 } }
}
```

| Prototype label | Field |
|---|---|
| Main recommendation | `title` |
| Why it matters | `why` |
| Fix | `fix` |
| Scores impacted | `scores_impacted` |
| Timestamp chip | `anchor.t_start`–`anchor.t_end` (`null` → whole ad) |

**Every recommendation is written about an entry in `defects`**, and carries that defect's
timestamps and numbers in `anchor`. There is no recommendation without a measured problem
behind it — Gemini is not allowed to invent one. Numbers in `title` and `why` are checked
against the measurements; numbers in `fix` are *targets* ("cut it to 16 words") and are not.

`recommendations` can be shorter than `defects`: a recommendation that failed the evidence
check is dropped, and the defect is still listed.

### `interpretation`

```json
{
  "available": true, "reason": null, "message": null,
  "prompt_version": "vl_interpret_2026_09",
  "evidence_audit": {
    "checked": 9, "dropped": 1,
    "unsupported": [ { "where": "trigger[mere_exposure]", "reason": "no_evidence_cited",
                       "detail": "rated weak with nothing cited" } ]
  },
  "dropped_invented_recommendations": []
}
```

- `available: false` with `reason` (`analysis_not_configured`, `analysis_provider_error`,
  `analysis_invalid_output`) → `summary` is `""`, `recommendations` and `observations` are
  empty, `key_message` is `null`, and Focus/Clarity report `llm_rating_not_available`. The six
  scores, timeline and defects are unaffected.
- `evidence_audit` — what Gemini tried to say and could not support. Not for customers; worth
  tracking over time.
- `dropped_invented_recommendations` — defects Gemini nominated itself. Always refused.

### `versions` and `config_disclosure`

`versions` — `framework_version`, `psychology_version`, `prompt_version`,
`interpret_prompt_version`, `saliency_model`. Store them if you compare analyses over time: a
version change is a different analysis.

`config_disclosure` — which business values are still placeholders. Nothing to render; it is
there so no one mistakes a placeholder for a decision.

| Key | Current value | Meaning |
|---|---|---|
| `weighting` | `equal_unweighted_placeholder` | All six metrics count equally |
| `bands` | `thresholds_not_configured` | No Good / Average / Poor cut-offs |
| `rating_scale` | `uniform_placeholder` | How an AI rating converts to a number |
| `weak_zone_rule` | `placeholder` | The "attention is low" threshold |
| `timeline_coefficients` | `uniform_placeholder` | How six signals combine into the attention curve |
| `calibration` | `provisional_absolute` | Comparable across ads; not yet tied to campaign results |
| `triggers_affect_overall_score` | `false` | Triggers are reported, not scored |
| `reading_speed_words_per_second` | `4.0` | Drives the overloaded-slide check |
| `unconfirmed` | a list | Every business value still awaiting a decision — see §14 |

---

## 10. Reason codes and failure handling

### Why an analysis failed — `status: "failed"`, top-level `reason`

| Reason | Meaning | What to show / do |
|---|---|---|
| `no_creative` | No URL, or no file | Ask for a file |
| `url_not_allowed` | Host not allowlisted, or not https | Backend bug — the link must be our bucket |
| `creative_unreachable` | Download failed — usually an **expired link** | Resubmit with a fresh link |
| `creative_unusable` | Not a readable image or video, or empty | Ask for a different file |
| `creative_too_large` | Over 400 MB | Ask for a smaller export |
| `unsupported_format` | Not JPG, PNG, WebP, MP4, MOV, WebM | Ask for a supported format |
| `duration_too_long` | Over 180 s | Ask for a shorter cut |
| `decode_failed` | The video could not be decoded | Re-export and resubmit |
| `no_video_stream` | The file has no video track | Ask for a different file |
| `ffmpeg_unavailable` | Server misconfigured | Ops |
| `saliency_model_unavailable` | Server misconfigured | Ops |
| `inference_failed` | The attention step failed | Resubmit; if it repeats, report it |
| `processing_interrupted` | The worker stopped mid-job (a restart) | Resubmit |
| `storage_not_configured`, `object_storage_not_configured` | Server misconfigured | Ops |
| `upload_failed` | `/upload` could not write to S3 | Retry the upload |

### Not failures — context in `notes[]` or on a block

| Code | Where | Meaning |
|---|---|---|
| `no_audio_stream` | `notes`, `transcript.reason` | Silent ad — no transcript. Normal |
| `transcript_empty` | `transcript.reason` | Audio, but no speech |
| `transcription_not_configured`, `transcription_provider_error` | `transcript.reason` | No transcript this time |
| `analysis_not_configured`, `analysis_provider_error`, `analysis_invalid_output` | `interpretation.reason` | No guidance or recommendations this time |
| `ocr_unavailable` | `notes`, metric `reason` | On-screen text could not be read |
| `no_brand_brain` | `notes` | No Brand Brain context was used |
| `no_brand_assets` | `notes` | No wordmark — brand found by name only |

### Why one metric is null or partial — `scores.<id>.reason`

| Reason | Meaning |
|---|---|
| `no_measurements` | Nothing could be measured for it |
| `ocr_unavailable` | Needs on-screen text, which could not be read |
| `no_brand_detected` | Brand Memory — no brand found. Send `brand_names` |
| `no_cta_detected` | Clarity — no call to action found |
| `no_timeline` | Needs a timeline — images have none |
| `llm_rating_not_available` | Focus / Clarity computed without the AI pass |
| `key_message_not_on_screen` | Focus — the key claim is spoken, so it is measured across the whole ad |
| `key_message_outside_sampled_frames` | Focus — the key moment was between samples |

### Defined but never returned

`analysis_not_found` and `vision_lab_disabled` are in the code's list of reasons, but no
endpoint returns them. Those two cases are HTTP errors instead: an unknown id is **`404`**
(`{"detail": "analysis_id not found"}`), and a disabled or unconfigured Vision Lab is
**`503`**. Branch on the status code for them, not on a `reason`.

---

## 11. Limits

| Limit | Value | Setting |
|---|---|---|
| File size | 400 MB | `VL_MAX_DOWNLOAD_MB` / `VL_MAX_UPLOAD_MB` |
| Video duration | 180 s | `VL_MAX_DURATION_SECONDS` |
| Frames analysed | 120, spread over the whole ad | `VL_MAX_FRAMES` |
| Frame rate | 2 fps, lowered automatically for long ads | `VL_SAMPLE_FPS` |
| Formats | `.mp4 .mov .webm .m4v .jpg .jpeg .png .webp` | — |
| On-screen text languages | English, Hindi | `VL_OCR_LANGUAGES` |
| Jobs processed at once | 1 | `VL_MAX_CONCURRENT_JOBS` |
| A started job reported stalled after | 30 min — never applies to a waiting job | `VL_JOB_STALE_SECONDS` |
| No worker running flagged in `/health` after | 60 s of jobs waiting unclaimed | `VL_QUEUE_UNCLAIMED_ALERT_SECONDS` |
| Measurements kept (for rescore) | 180 days | `VL_MEASUREMENTS_TTL_DAYS` |
| Image link validity | 1 hour, re-signed on every GET | `AWS_S3_URL_TTL_SECONDS` |
| Upload link validity | 6 hours | `VL_UPLOAD_URL_TTL_SECONDS` |
| History page size | 25, max 100 | `limit` |

---

## 12. Testing it yourself

| Resource | What it is |
|---|---|
| `postman/ScaleSerum-VisionLab.postman_collection.json` | 20 requests: health, the full happy path (upload → analyse → poll → report → rescore → delete), 7 failure paths, and a regression check on the rest of the service |
| `VISION_LAB_POSTMAN_TESTING.md` | Step-by-step guide to running it, and what to check by eye |
| `postman/example-responses/` | The real response of every endpoint, captured from a live run. Signatures redacted |
| `tests/fixtures/vision_lab/example_report.json` | A real, complete report — **build the frontend against this**. A test fails if a field ever disappears from it |
| `tests/test_vision_lab_*.py` | 161 tests — `python -m pytest tests/ -k vision_lab` |
| `scripts/presign.py` | Generate a presigned link for a file in the bucket |
| `scripts/upload_test_ads.py` | Upload the reference ads in `Ads_Video/` to `vision-lab-test/` |

### Quick start (local)

```powershell
# Terminal 1 - the API
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 3001

# Terminal 2 - the worker. Without it every analysis stays "queued".
.venv\Scripts\python.exe -m vision_lab.worker
```

In Postman: import the collection (choose **Replace** if an older copy exists), set
`api_key` in Collection → Variables → Current value, open **00 Upload a video**, pick a file
in its Body tab, send. Then **01** → **02** → **03**.

After any change under `vision_lab/`, **restart the worker**, not only the API — the analysis
runs in the worker process.

---

## 13. Known limitations

| # | Limitation | Impact | Workaround |
|---|---|---|---|
| 1 | 🔴 **Not deployed** — the worker is not in `ecosystem.config.js` and `deploy/deploy.sh` does not handle it | Nothing processes on the server | Milestone E. Integrate against a local instance meanwhile |
| 2 | 🔴 **`brand_assets.wordmark_url` is not allowlist-checked.** `creative.url` is; the wordmark link is fetched by the worker without the same check | Anyone holding the API key could make the server fetch an arbitrary address | Until fixed: only ever send a wordmark URL that the backend generated for our own bucket — **never** one taken from user input |
| 3 | 🟡 **Brand Brain is not wired in.** `brand_brain_id` is accepted and stored, but no Brand Brain is loaded | Confirmation Bias and Blind-Spot Bias are always `not_applicable`; Gemini writes without persona context. **Sending a `brand_brain_id` also suppresses the `no_brand_brain` note**, so the report does not say the context was missing | Do not tell users the analysis is persona-aware yet |
| 4 | 🟡 **`GET` timestamps have no timezone suffix** | Parsed as local time in JavaScript | Append `Z` before parsing (rule 13, §8) |
| 5 | AI-predicted, not eye-tracking | Directional, not a measurement of real viewers | Say so on every screen |
| 6 | Saliency models are weak on small logos | — | Handled: Brand Memory uses detection to find the logo and saliency only to ask whether that moment was attended |
| 7 | Scores are provisional | Absolute and comparable across ads, but not yet calibrated against campaign performance | — |
| 8 | Brand detection needs `brand_names` or a wordmark | Brand Memory is null without them | Always send `brand_names` |
| 9 | OCR reads English and Hindi only; heavily stylised type may be misread | Word counts can undercount | — |
| 10 | Gemini's wording and ratings vary between runs | Two separate analyses of the same ad can differ slightly in Clarity and prose | Identical resubmissions return the stored result; rescoring reuses the stored rating |
| 11 | One job at a time per worker, CPU only | A queue of N ads takes roughly N × 60 s | **Run more worker processes** — they share the MongoDB queue safely, and no job is ever taken twice. Raising `VL_MAX_CONCURRENT_JOBS` barely helps: inside one process the CPU-heavy steps run one after another |
| 12 | `DELETE` does not remove files sent to `/upload` | They accumulate in `vision-lab-uploads/` | A bucket lifecycle rule |
| 13 | No upload progress from Vision Lab | — | The browser → S3 PUT carries the progress bar |
| 14 | Model licence for commercial use not yet signed off | Blocks customer-facing launch | Legal review — see §3.6 |

---

## 14. Awaiting management decisions

Every one of these is a placeholder today, reported in every response under
`config_disclosure`. Setting them is a configuration change plus `POST /rescore` —
**no API change for either team**.

| Decision | Current placeholder | Effect when set |
|---|---|---|
| Metric weights | All six equal | `overall.score` changes; `overall.weighting` reports it |
| Score bands (Good / Average / Poor) | None | `overall.band` starts carrying a value |
| Weak-zone threshold and minimum length | Below 35 for 1 s | Weak zones, markers and `dead_zone` defects |
| Timeline coefficients | Equal | The attention curve's shape |
| Rating scale for AI ratings | Uniform | Clarity's blend |
| Whether triggers count towards the overall | No | `psychology.affects_overall_score` |
| Trigger weights | None | Only if the above is yes |
| Brand appearance targets | None | Brand Memory |
| Minimum creative duration | None | Very short videos would become `skipped` |

Also outstanding: legal sign-off on the model licence (§3.6), a lifecycle rule for
`vision-lab-uploads/` (§3.5), and whether Brand Brain context should be wired in (§13 #3).

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Saliency map** | A per-pixel prediction of where the eye lands, summing to 100% across the frame |
| **Peak / hotspot** | A region holding a large share of that 100% — the numbered circles |
| **Attention index** | The 0–100 curve over time — a composite of six measured signals, not a saliency output |
| **Weak zone** | A span where the attention index stays low |
| **Hook** | The first 3 seconds |
| **Defect** | A problem Python measured, with timestamps and numbers |
| **Recommendation** | Gemini's written fix for one defect |
| **Carrier** | Whether the key claim is written, spoken, both, or shown |
| **Rescore** | Recomputing scores from stored measurements, at no cost |
| **Fingerprint** | The identity of an analysis — the file, its identifiers and every version — used for idempotency |

---

## Related documents

| Document | For |
|---|---|
| `VISION_LAB_PLAN.md` | The design, the metric formulas, the report contract |
| `VISION_LAB_DELIVERY.md` | The service boundary and deployment |
| `VISION_LAB_BUILD_STEPS.md` | How it was built, and every defect found by running it on real ads |
| `VISION_LAB_POSTMAN_TESTING.md` | Testing every endpoint in Postman |
| `NOTICE.md` | Third-party licences and attribution |
| `SALES_CALL_ANALYZER.md` | The sibling feature on the same service — shared prerequisites |

## Questions

Raise them with the AI service owner. When reporting a problem with a specific analysis,
include its `analysis_id` — every log line and stored record is keyed by it.
