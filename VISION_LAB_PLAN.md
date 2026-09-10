# Vision Lab — implementation plan

Status: **plan, nothing built yet.** This document is the design agreed before code. When
the feature ships it becomes `VISION_LAB.md`, alongside `SALES_CALL_ANALYZER.md` and
`PURCHASE_PROBABILITY_API.md`.

> **Companion:** [`VISION_LAB_DELIVERY.md`](VISION_LAB_DELIVERY.md) covers how this
> ships — the microservice boundary, the test layers, the deploy-pipeline changes a
> second pm2 process forces, and the integration contract for the backend and
> frontend teams. This document is *what* we build; that one is *how it lands*.
>
> **To start coding:** [`VISION_LAB_BUILD_STEPS.md`](VISION_LAB_BUILD_STEPS.md) — the
> 24 steps in dependency order, each with a verify command and a definition of done.

Vision Lab takes an uploaded creative — a static image or a video ad — and returns an
**AI-predicted attention report**: a heatmap, six creative scores, key moments, a
per-second attention timeline, a transcript, and ranked fix recommendations, all read
through the lens of 15 psychological marketing triggers.

---

## 0. Read this first — three corrections to the brief

These change the plan materially, so they are at the top rather than buried in a risks
section.

### 0.1 You cannot train on MIT300

MIT300 is a **held-out test set of 300 images whose fixation data is not public**. It
exists only as a leaderboard: you submit predictions to the MIT/Tuebingen Saliency
Benchmark and receive AUC-Judd / sAUC / NSS / CC / KLD / SIM back. Training on it is
impossible, and claiming it as training data would be wrong on a spec sheet.

The correct split of the three names in the brief:

| Dataset | What it is | Our use |
|---|---|---|
| **SALICON** | ~10k train / 5k val images, mouse-contingent pseudo-fixations over COCO images. The standard **pre-training** corpus. | Already baked into every pretrained checkpoint we would use. We do not retrain it. |
| **MIT1003** | 1003 images, real eye-tracking, **public fixations**. | Local validation, and fine-tuning if Phase 5 justifies it. |
| **MIT300** | 300 images, fixations withheld. Benchmark only. | A one-off external **validation** submission if we ever want a published number to quote in sales material. |

The prototype's own caption is already the honest one and must survive into production:
*"Attention predicted by a model trained on eye-tracking from a large human panel
(SALICON) — not measured on live viewers."*

### 0.2 We are not training a saliency model — we are selecting one and calibrating it

Training a competitive saliency model from scratch needs a multi-GPU box and weeks of
work. The live service is a **single CPU VPS running one pm2 fork process**
(`/root/Marketing_tool`, pm2 name `marketing-tool`) shared with onboarding, Script Lab,
purchase probability and the Sales Call Analyzer. There is no GPU anywhere in this stack.

What we actually do:

1. **Bake-off** three or four pretrained checkpoints on MIT1003 + CAT2000 + ~100 of our
   own ad frames, measure NSS/CC/AUC, pick one (§5.2).
2. **Calibrate** its raw outputs into 0–100 scores against a percentile reference built
   from our own analysed ads — the mechanism
   `purchase_probability_model/percentile_reference.json` already uses (§7.5).
3. Only then, if the bake-off shows a real gap on ad-like content, consider fine-tuning.
   That is a Phase 5 decision with a measurement behind it, not a Phase 1 assumption.

### 0.3 A saliency model predicts *where* people look, not *whether* they keep watching

This is the largest correctness risk in the prototype. A saliency map is a spatial
probability distribution that **sums to 1 on every frame** — a frame of grey mush and a
frame with a gripping face both sum to 1. It cannot, on its own, produce "attention falls
to 26/100 at 8 s".

So the timeline number is a **derived composite index** over spatial concentration, motion
energy, face presence, saliency stability, cut rate and text load (§7.2) — not a model
output. It is labelled as an index everywhere it appears
(`"basis": "derived_composite"`), and Phase 5 exists to calibrate it against the only real
ground truth available to us: platform retention curves (Meta / YouTube 3 s and
25/50/75/100 % view-through) for creatives we have actually run.

> **When Phase 5 arrives, do not join creatives to conversion outcomes from `scrumdb`.**
> That database's conversion labels are pervasively backfilled at payment time — a model
> trained on them reads the receipt and scores ~0.95 AUC on nothing. Platform retention
> data is clean; scrumdb conversions are not.

---

## 1. The house rules this feature inherits

Vision Lab is the fourth feature in this service and must look like the other three.

**Division of labour — a correctness requirement, not a style preference.**

| Layer | Owns |
|---|---|
| **CV** (saliency, OCR, detection, ffmpeg) | *Measures*: where gaze goes, what is on screen, when, how big, how long. |
| **Deepgram** | *Measures*: what is said, and when. |
| **Gemini** | *Interprets*: what the creative is trying to do, ordinal ratings, trigger evidence, prose. |
| **Python** | *Validates, scores, aggregates, persists, orchestrates.* |

**Gemini never produces a number.** It returns an ordinal rating per criterion plus the
evidence for it; `vision_lab/scoring.py` turns ratings and measurements into scores using
`vision_framework.json`. Consequences, all deliberate: the same inputs always produce the
same score; historical creatives can be rescored under new weights with **no GPU, no
ffmpeg and no API call**; and nothing written in the ad copy can talk the model into a
better number.

**No invented business values.** Weights, bands, the rating scale, the weak-zone
threshold, the reading speed — every one is `null` in config until management sets it, and
every response reports which values were still placeholders. Do **not** copy Script Lab's
90/70/50 bands here; they carry no authority over creative attention.

**Every failure is a stated failure.** Always-200 envelope, stable `reason` codes, a
`fallback: true` flag, `scores: null` on failure. A failed analysis never invents a
scorecard.

**Nothing is invented about the creative either.** Every fix recommendation is anchored to
a *measured* defect with timestamps — the model writes prose for a defect Python found, it
does not get to nominate defects. Same rule as `sales_call_analyzer/evidence.py`.

---

## 2. Architecture

```
                 main app / frontend
                        |  (1) presigned PUT direct to object storage
                        v
                 object storage (S3 / R2 / GCS)
                        |  creative_url (signed, short TTL)
                        v
   POST /api/vision-lab/analyze  --->  Mongo: vision_lab_analyses (status=queued)
   (returns analysis_id instantly)              |
                                                | claimed by
                                                v
                                   +------------------------------+
                                   | vision-worker (2nd pm2 proc) |
                                   |  ffmpeg      -> frames       |
                                   |  saliency    -> heatmaps     |
                                   |  OCR/face/brand detection    |
                                   |  Deepgram    -> transcript   |
                                   |  Gemini      -> ratings+prose|
                                   |  Python      -> scores       |
                                   +------------------------------+
                                                | report + heatmap URLs
                                                v
   GET /api/vision-lab/analysis/{id}  <--- polled every 5 s by the frontend
```

### 2.1 Why a separate worker process (this is not optional)

The Sales Call Analyzer runs its jobs in a FastAPI `BackgroundTask`, which is correct
*there* — Deepgram and Gemini are network waits, so the event loop stays free. Vision Lab
is **CPU-bound**: ffmpeg decode, saliency inference and OCR saturate a core for 30–120 s
per ad. Running that in the API process would stall onboarding, Script Lab and every
sales-call poll for the duration.

So: a second pm2 process, `vision-worker`, in the same repo and venv. It claims jobs from
Mongo with an atomic `find_one_and_update({status: "queued"}, {$set: {status: "processing",
worker_id, heartbeat}})` loop. **No Redis, no Celery, no new infrastructure** — the job
document *is* the queue, it is the same document the API already polls, and the stale-job
reaper pattern (`STALE_AFTER_SECONDS` in `sales_call_analyzer/store.py`) becomes the
recovery mechanism for a worker killed mid-deploy.

Concurrency gate: **1** on the current box, as `VL_MAX_CONCURRENT_JOBS`.

### 2.2 Uploads never touch this API

`deploy/nginx.conf.example` sets `client_max_body_size 1m`. The prototype accepts images
to 15 MB and video to 200 MB. Raising that limit would push 200 MB through nginx, through
uvicorn, and into a process that also serves onboarding — don't.

The main app presigns an upload straight to **the existing AWS S3 bucket** ScaleSerum
already uses for ad creatives, and sends us a **URL**. We read it and forget it, the same
rule the Sales Call Analyzer applies to call recordings: the asset belongs to the backend,
and a signed URL is a credential with an expiry. No new bucket is needed — see
`VISION_LAB_DELIVERY.md` §4.8 for the IAM, region and egress details.

Heatmap PNGs go the other way: the worker writes them to the same bucket under a TTL
prefix and the report carries URLs. Never base64 in the document — a 48-frame report would
be tens of megabytes.

### 2.3 Timing budget (24 s ad, 2 fps, CPU)

Rough, to be replaced by measurement in Phase 2. The prototype's own footer —
*"Analyzed 48 frames sampled across the video"* on a 24 s ad — is exactly 2 fps, so that
is the default.

| Step | Estimate |
|---|---|
| Fetch + ffmpeg decode to 48 frames | 3–8 s |
| Saliency, 48 frames @ ~384×224 | 10–30 s |
| OCR, 48 frames | 15–50 s ← **the expensive part** |
| Heatmap render + upload | 3–6 s |
| Deepgram (audio) | 5–20 s |
| Gemini (ratings + fixes) | 10–25 s |
| **Total** | **~1–2.5 min** |

Two levers if that is too slow, both config rather than code: run OCR only on frames after
a detected shot change (typically ~4× fewer), and drop saliency input to 320×180.

---

## 3. Module layout

Mirrors `sales_call_analyzer/` deliberately — same shapes, same names, so anyone who has
read one package can read the other.

```
vision_lab/
  __init__.py            statuses, reason codes, package docstring (the contract)
  models.py              Pydantic request/response — the API contract
  media.py               fetch, probe, ffmpeg frame sampling, shot-change detection
  saliency.py            the attention model: load, infer, batch. ONNX Runtime.
  regions.py             OCR (text boxes + word counts), faces, brand mark, CTA detection
  measure.py             per-frame + per-second measurements -> the measurement record
  timeline.py            attention index, drop-off zones, key-moment labelling
  heatmap.py             overlay render (colormap + numbered peaks) -> PNG -> storage
  psychology.py          the 15 triggers: measured detectors + prompt rendering
  analyzer.py            Gemini: ratings, trigger evidence, fix prose. PROMPT_VERSION.
  evidence.py            verify every claim against transcript / OCR / measurements
  scoring.py             DETERMINISTIC: measurements + ratings -> 6 scores + overall
  report.py              assemble the response. Never rescores.
  store.py               MongoDB. Injected collections, fingerprint idempotency, TTLs.
  pipeline.py            orchestration, status transitions, heartbeat
  worker.py              pm2 entrypoint: claim loop, concurrency gate, graceful stop
  vision_framework.json       metric definitions, weights, thresholds — business values null
  psychology_framework.json   the 15 triggers, definitions, applicability, weights null
```

`app.py` gains one section, imported defensively exactly like the Sales Call Analyzer, so
a broken import or a missing model file degrades **this feature only** and everything else
still boots.

**One refactor first:** `sales_call_analyzer/deepgram_client.py` is needed by both
features. Promote it to a top-level `transcription/deepgram_client.py` and re-export from
`sales_call_analyzer` for compatibility. A `vision_lab -> sales_call_analyzer` import
would be the wrong dependency edge and would spread on the next feature.

---

## 4. API contract

Same path style, same auth (`X-API-Key`), same always-200 discipline as the rest of the
file. Versioning lives in the payload (`framework_version`, `prompt_version`,
`saliency_model`), not in the URL.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/vision-lab/analyze` | Submit a creative. Returns `analysis_id` immediately. Idempotent by fingerprint; `options.force_reanalysis` overrides. |
| `POST` | `/api/vision-lab/upload` | **For testing.** `multipart/form-data` with a `file` field. Stores it under `vision-lab-uploads/<date>/<id>/` and returns the presigned link `/analyze` consumes; `analyze=true` also queues the analysis and returns exactly what `/analyze` would. A file over the size cap is **413**, refused before its body is read. The frontend should keep uploading to S3 itself — see VISION_LAB_DELIVERY.md §6.2. |
| `GET` | `/api/vision-lab/analysis/{analysis_id}` | Poll status, then read the report. |
| `GET` | `/api/vision-lab/analysis/by-creative/{creative_id}` | Latest analysis for a creative. |
| `GET` | `/api/vision-lab/history` | The **History** tab: `?division=&ad_number=&limit=` -> the version list for an ad number. |
| `POST` | `/api/vision-lab/analysis/{id}/rescore` | Recompute all six scores from stored measurements under the current config. **No ffmpeg, no inference, no Gemini, no cost.** |
| `DELETE` | `/api/vision-lab/analysis/{id}` | Remove the analysis, its stored measurements and its heatmap images. Does **not** remove a video uploaded through `/upload` — expire `vision-lab-uploads/` with a bucket lifecycle rule. |

The rescore endpoint is the payoff for storing measurements rather than only numbers: when
management sets weights and bands, every historical creative comes onto the new
configuration without re-processing a single frame.

### 4.1 Request

```jsonc
{
  "creative_id": "cre_8f21",              // caller's id, required
  "brand_brain_id": "bb_2091",            // optional; persona, offer, funnel stage
  "division": "directors_institute",      // the prototype's Division selector
  "ad_number": "042",                     // optional — this is what enables versioned history
  "creative": {
    "url": "https://…signed…/di_board_seat_v4_final.mp4",
    "kind": "video",                      // "video" | "image"
    "mime_type": "video/mp4",
    "duration_seconds": 24,               // a hint; we probe and trust the probe
    "width": 1920, "height": 1080
  },
  "brand_assets": {                       // optional; sharply improves Brand Memory
    "wordmark_url": "…/di_wordmark.png",
    "brand_names": ["Director's Institute", "DI"]
  },
  "options": { "force_reanalysis": false, "sample_fps": null }
}
```

### 4.2 Response (completed) — one field per prototype element

```jsonc
{
  "analysis_id": "…", "creative_id": "cre_8f21", "ad_number": "042",
  "status": "completed",
  "availability": { "available": true, "reason": null, "message": null },

  "media": { "kind": "video", "duration_seconds": 24.0, "width": 1920, "height": 1080,
             "frames_analyzed": 48, "sample_fps": 2.0 },

  "overall": { "score": 63, "band": null, "band_reason": "thresholds_not_configured" },
  "summary": "A strong hook and a genuinely arresting product reveal, wrapped around a soft middle…",

  "scores": {                              // the six metric bars
    "attention":        { "score": 96, "label": "captures initial gaze",         "direction": "higher_better", "basis": "measured" },
    // `reason` is present whenever a metric could not be scored in full. Focus
    // reports `key_message_not_on_screen` when the central claim is spoken
    // rather than written — that is not a low score, it is a different answer.
    "focus":            { "score": 61, "label": "attention on the key message",  "direction": "higher_better", "basis": "measured", "reason": null },
    "cognitive_demand": { "score": 44, "label": "effort to understand",          "direction": "lower_better",  "basis": "measured" },
    "clarity":          { "score": 66, "label": "message + CTA are unambiguous", "direction": "higher_better", "basis": "hybrid"   },
    "brand_memory":     { "score": 38, "label": "likelihood brand is remembered","direction": "higher_better", "basis": "measured" },
    "engagement":       { "score": 57, "label": "pull to keep watching",         "direction": "higher_better", "basis": "measured" }
  },

  "heatmap": {                             // the Attention Report image
    "frame_time": 4.124,                   // which second of the ad this frame is
    "object_key": "vision-lab/<analysis_id>/heat_4124.png",
    // Signed FRESH on every GET, never stored. An analysis opened three weeks
    // later still returns a live link rather than a dead one.
    "image_url": "https://…/heat_4124.png?X-Amz-Signature=…",
    "peaks": [
      // `box` is FRACTIONAL [x0,y0,x1,y1], not pixels, so the frontend can place
      // the numbered marker at any display size without knowing our working
      // resolution. `share` is a real fraction of predicted attention - the
      // saliency map sums to 1 across the frame - so 0.341 prints as "34%".
      { "rank": 1, "box": [0.38,0.28,0.62,0.45], "share": 0.341,
        "element": "face",           "element_text": null },
      { "rank": 2, "box": [0.03,0.02,0.16,0.11], "share": 0.170,
        "element": "brand_mark",     "element_text": "Director's Institute" },
      { "rank": 3, "box": [0.10,0.12,0.90,0.20], "share": 0.120,
        "element": "on_screen_text", "element_text": "Did You Know That Next Year" }
    ]
  },
  // `element` is one of: brand_mark | call_to_action | face | on_screen_text,
  // or NULL. Null means nothing we detect sits under that hotspot - product
  // footage, b-roll, or a person the face detector missed. It is a more honest
  // answer than naming the peak after a caption that merely clipped its corner,
  // and the frontend should render the marker without a label rather than
  // hiding it. `element_text` carries the words for a text/CTA peak and the
  // brand name for a brand peak, so the report can say "the eye went to BOARD
  // READINESS" rather than "to text".

  "key_moments": [                          // the PEAK / KEY / WEAK / HERO strip
    { "time": 1.0, "label": "peak", "thumbnail_url": "…", "attention": 93, "note": "…" },
    { "time": 8.0, "label": "weak", "thumbnail_url": "…", "attention": 26, "note": "…" }
  ],

  "timeline": {                             // Attention over time
    "unit": "index_0_100", "basis": "derived_composite",
    "points": [ { "t": 0.0, "attention": 88, "engagement": 84 } ],
    "markers": [ { "t": 8.0,  "severity": "high",   "text": "Attention flatlines for 2.5 s — three text bullets, no motion" },
                 { "t": 12.5, "severity": "medium", "text": "Stat card holds 1.5 s past the point the eye has read it" },
                 { "t": 19.0, "severity": "low",    "text": "Silent logo hold — exposure alone does not build brand recall" } ]
  },

  "transcript": {                           // Deepgram, server-side (not in-browser)
    "available": true, "reason": null, "language": "en",
    "stats": { "lines": 27, "words": 171, "seconds_of_speech": 64.45,
               "words_per_second": 2.65, "mean_line_attention": 64.9,
               "lines_in_weak_zones": 0 },
    // `attention` is a JOIN over the timeline, not a second prediction.
    // `in_weak_zone` is the warning icon beside a line in the transcript panel.
    "segments": [ { "t": 0.0, "end": 3.0, "text": "Most senior executives never get offered a board seat.",
                    "attention": 88, "in_weak_zone": false } ]
  },
  // Absent instead: { "available": false, "reason": "no_audio_stream", "segments": [] }

  "key_message": {                          // what completes the Focus metric
    "element": "The headline promise of a board seat within 12 months",
    "t": 8.0, "quote": "…", "verified": true,
    // WHICH CHANNEL carries the claim. Focus measures where GAZE went, so a
    // claim delivered in voice has nothing on screen to measure against and
    // Focus reports `key_message_not_on_screen` rather than a low score.
    "carrier": "on_screen_text",
    // true when the model's wording failed verification and the transcript
    // line's own text was substituted. The published quote is always ours.
    "model_quote_rejected": false
  },

  "psychology": {                           // the 15 triggers
    "triggers": [
      { "id": "bandwagon", "name": "The Bandwagon Effect", "status": "present",
        "rating": "strong", "detection": "measured", "polarity": "positive",
        "measured": { "adoption_numerals": ["4200"] },
        "evidence": [ { "t": 10.0, "quote": "Four thousand two hundred directors have been certified through us.", "source": "transcript" } ],
        "note": "A specific adoption number at 10 s, inside a rising attention window." },
      { "id": "anchoring", "name": "Anchoring", "status": "not_applicable",
        "reason": "no_price_shown", "rating": null, "evidence": [] },
      // The model rated it but could not support it. NOT the same as `absent`:
      // we do not know how the creative did, which differs from knowing it did
      // badly. `rating` is null and nothing is published as its evidence.
      { "id": "ikea_effect", "name": "The IKEA Effect", "status": "unsupported",
        "reason": "no_evidence_cited", "rating": null, "evidence": [] },
      // Choice Overload is the ONE trigger where showing it strongly is bad
      // news. `rating` says how strongly it shows; `polarity` says which
      // direction is good.
      { "id": "choice_overload", "name": "Choice Overload", "status": "absent",
        "rating": "absent", "polarity": "defect", "evidence": [] }
    ],
    "coverage": { "present": 6, "weak": 3, "absent": 4, "not_applicable": 2 },
    "unsupported": 1
  },

  "recommendations": [                      // Fix recommendations
    { "rank": 1,
      "title": "The 7–10 s text slide is a dead zone — attention falls to 26/100",
      "why": "Three static bullets ask a 40–55 year-old senior professional to read while a voice-over is already speaking…",
      "fix": "Cut the bullet slide entirely. Hold the presenter through 7–10 s and let the VO carry governance / fiduciary duty / board readiness…",
      "scores_impacted": ["engagement", "cognitive_demand", "focus"],
      "trigger_ids": ["choice_overload"],
      "severity": "high", "verified": true,
      "anchor": { "defect_id": "dead_zone", "t_start": 7.0, "t_end": 10.0,
                  "measured": { "attention_min": 26, "words_on_screen": 34, "reading_seconds_needed": 8.5, "seconds_available": 2.5 } } }
  ],
  // Every recommendation is written ABOUT an entry in `defects` and carries its
  // timestamp and counted numbers. There is no recommendation without a
  // measured defect behind it. Numbers in `title`/`why` are CLAIMS and are
  // verified; numbers in `fix` are TARGETS for the editor and are not.
  "observations": [ "The close assumes the viewer already knows the brand." ],

  "interpretation": {                       // did the LLM pass run, and what did it try to slip through
    "available": true, "reason": null, "prompt_version": "vl_interpret_2026_09",
    "evidence_audit": { "checked": 9, "dropped": 1,
                        "unsupported": [ { "where": "trigger[ikea_effect]",
                                           "reason": "no_evidence_cited",
                                           "detail": "rated strong with nothing cited" } ] },
    "dropped_invented_recommendations": ["the_soundtrack_is_tired"]
  },
  // Absent instead: { "available": false, "reason": "analysis_not_configured" }.
  // The six scores, the timeline and the defect list do not depend on this pass
  // and are unaffected by its absence.

  "versions": { "framework_version": "vision_v1", "psychology_version": "triggers_v1",
                "prompt_version": "vl_2026_09", "saliency_model": "…", "llm_model": "gemini-flash-latest" },
  "config_disclosure": { "weighting": "equal_unweighted_placeholder",
                         "bands": "not_configured",
                         "calibration": "provisional_absolute",
                         "unconfirmed": ["metric weights", "score bands", "weak-zone threshold"] },
  "created_at": "…", "updated_at": "…"
}
```

While processing, the same envelope with `status` in `queued | probing | analyzing_frames |
transcribing | interpreting | scoring`, `scores: null`, plus `poll_url` and
`suggested_poll_interval_seconds`.

### 4.3 Reason codes (stable; the UI may branch on them)

```
input:      no_creative, creative_unreachable, creative_unusable, creative_too_large,
            unsupported_format, duration_too_long
media:      decode_failed, no_video_stream, no_audio_stream (not fatal)
vision:     saliency_model_unavailable, ocr_unavailable (not fatal), inference_failed
transcript: transcription_not_configured, transcription_provider_error, transcript_empty
analysis:   analysis_not_configured, analysis_provider_error, analysis_invalid_output
metrics:    llm_rating_not_available, key_message_not_on_screen,
            key_message_outside_sampled_frames, no_text_read, no_cta_detected,
            no_brand_detected, no_measurements, no_timeline
            (why ONE metric could not be scored - never why the analysis failed)
evidence:   quote_not_in_transcript, timestamp_outside_creative,
            number_not_measured, no_matching_defect, no_evidence_cited
            (inside interpretation.evidence_audit; a claim we would not publish)
lifecycle:  storage_not_configured, processing_interrupted, analysis_not_found
context:    no_brand_brain, no_brand_assets   (reported, never fatal)
```

---

## 5. The vision layer

### 5.1 Frame sampling

`ffmpeg` (a **new server dependency** — `apt install ffmpeg`, and add it to
`deploy/README.md` and `deploy.sh`'s preflight) samples at `sample_fps` (default 2.0),
capped at `VL_MAX_FRAMES` (default 120, so a 60 s ad does not become a 5-minute job).
`ffprobe` gives duration, dimensions, and whether an audio stream exists at all. A
scene-change filter marks shot boundaries, which both the OCR optimisation and the
"novelty" term in Engagement use.

Static creatives skip all of this: one frame, image mode, no timeline, no transcript, no
key moments. The report shape stays identical with those fields null — one contract, not
two.

### 5.2 The saliency model — a bake-off, not a guess

Candidates to evaluate (all published, all with released weights):

| Model | Why it is a candidate |
|---|---|
| **UNISAL** | One small unified model (~4M params) that handles **both image and video** saliency — matches our two input kinds with one artefact. Fast on CPU. |
| **TranSalNet** | Transformer-based, strong current image numbers on SALICON/MIT. Heavier. |
| **MSI-Net / DeepGaze IIE** | Well-established image baselines, useful as the reference floor. |
| **TASED-Net / ViNet** | Video-specific, stronger temporal modelling than UNISAL if video quality is the bottleneck. |
| **Graphic-design importance models** (e.g. the UMSI / Imp1k line of work) | Trained on *designed* images — ads, posters, infographics — rather than natural photos. Natural-image saliency systematically under-weights text blocks and logos, which is precisely what an ad report is about. Evaluate this even if it loses on MIT1003. |

**Gate tasks before any of these is chosen (do not skip):**

1. **Licence review.** Several saliency checkpoints are research-only / non-commercial.
   ScaleSerum is commercial. A model that fails licence review is disqualified regardless
   of its NSS. Record the licence in `vision_framework.json` next to the model name.
2. **Bake-off protocol.** Score every candidate on MIT1003 and CAT2000 with AUC-Judd,
   sAUC, NSS, CC, KLD, SIM, plus **wall-clock CPU ms/frame at 384×224 on the actual VPS**.
   A model that is 4 % better and 5× slower loses.
3. **Ad-relevance check.** Hand-annotate ~100 frames from our own ads with "what the
   client believes should draw the eye" (headline, CTA, product, logo) and measure how
   much predicted saliency mass lands there. This is the number that matters for us; the
   academic benchmarks are the sanity check, not the decision.
4. **Export to ONNX** and run through **ONNX Runtime CPU**, not PyTorch. It removes a
   ~2 GB torch dependency from a small VPS, and is typically 1.5–3× faster on CPU.

Vendored weights do not belong in git. Ship them to the server as a build artefact into
`VL_MODEL_DIR`, and have `saliency.py` report `saliency_model_unavailable` (a stated
degradation, not a crash) when the directory is empty — the same defensive posture the
purchase-probability model already uses for its `.pkl` files.

### 5.3 Region detection — what the eye landed *on*

A heatmap is decoration until you can say which element it fell on. Per sampled frame:

- **Text**: OCR (PaddleOCR or Tesseract; benchmark both, OCR is the CPU bottleneck) gives
  boxes, text and per-box word counts. Feeds Cognitive Demand and Clarity.
- **Faces**: a light detector (OpenCV DNN / MediaPipe). Faces are the strongest known
  saliency attractor; the Engagement index needs the term.
- **Brand**: template match of `brand_assets.wordmark_url` at several scales, **plus** OCR
  text matched against `brand_names`. Either hit counts as a brand appearance. Without
  brand assets we fall back to OCR-only and report `no_brand_assets` — Brand Memory is
  then measured on a weaker basis and the response says so.
- **CTA**: OCR text matched against a CTA lexicon in config (`apply`, `book`, `register`,
  `enrol`, `download`, plus the weak-CTA list Script Lab already flags).

Each region carries its saliency mass share, so every peak in the heatmap gets an
`element` label — `product_ui`, `wordmark`, `stat_card`, `face`, `headline`, `cta`,
`unattributed`.

### 5.4 Heatmap rendering

`heatmap.py` upsamples the saliency map to frame size, applies a colormap, alpha-composites
over the frame, and numbers the top-3 peaks — the prototype's "Attention Report · Ad #042"
image. Rendered for: the hero frame, plus every key moment thumbnail, plus any frame the
frontend requests via the timeline's "click a point to view that frame's attention map".

To keep storage bounded: full-resolution overlays only for the hero frame and key moments;
everything else on demand, cached in the bucket under the analysis id with a TTL.

---

## 6. Transcript

Reuse Deepgram (already configured, already paid for) rather than the prototype's
in-browser transcription: it is more accurate, it gives word-level timestamps we can join
to the timeline, and it keeps the API key server-side. `media.py` extracts the audio track
with ffmpeg and hands Deepgram bytes; a video with no audio stream reports
`no_audio_stream` and the report simply omits the transcript block.

The per-line attention numbers in the prototype's transcript panel are a **join**, not a
second model: each segment takes the mean timeline attention over its own time span, and
the warning icon appears when a segment overlaps a detected weak zone.

---

## 7. The six scores — exact definitions

Every score is 0–100, computed in `scoring.py`. Thresholds, weights and coefficients live
in `vision_framework.json`; the formulas live in code. **Nothing below is a placeholder
number invented here** — where a business value is needed it is listed in §11 as
unconfirmed.

Notation: `S_t` = saliency map at frame `t`, normalised to sum 1. `mass(R, t)` = summed
saliency inside region `R`. `conc_t` = fraction of mass in the top 5 % of pixels (spatial
concentration — high means the eye is pulled to one place, low means the frame is
visually flat or contested).

### 7.1 The six

| Metric | Direction | Measured from |
|---|---|---|
| **Attention** — captures initial gaze | higher better | First `hook_seconds` (default 3.0): mean `conc_t`, the peak mass share of the single strongest region, and whether that region is the intended focal element. Pure CV, no LLM. |
| **Focus** — attention on the key message | higher better | Mean over the ad of `mass(key_message_region, t)`, penalised by the mean count of competing peaks. The *key message region* per frame = the element Gemini names as carrying the message, matched by Python to an actual detected box; if no match, the frame is excluded rather than scored zero. |
| **Cognitive Demand** — effort to understand | **lower better** | Purely measured. Per shot: `reading_load = words_on_screen / (reading_speed_wps × shot_seconds)`; plus a penalty when on-screen copy runs under a live voice-over (the two compete for the same channel), plus competing-peak count, cut rate, and text area share. `reading_speed_wps` defaults to **4** — the prototype's own stated basis — and is configurable. |
| **Clarity** — message and CTA unambiguous | higher better | Hybrid. Gemini's ordinal rating (evidence-verified) blended with measured CTA presence, legibility (text height vs frame height), and on-screen duration. Blend weight in config. |
| **Brand Memory** — likelihood brand is remembered | higher better | Measured: first brand appearance as a fraction of runtime; total brand on-screen seconds; `mass(brand_region)` *at* appearance; **whether the brand appears during top-quartile attention moments** (the mechanism the prototype's fix #2 is built on); and the count of distinct exposures (which is also the Mere Exposure trigger). |
| **Engagement** — pull to keep watching | higher better | Shape of the attention timeline: area under the curve, depth and duration of drop-offs, hook slope over the first 3 s, recovery after a dip, and novelty rate (shot changes, motion energy). |

**Overall** = weighted mean of the six with Cognitive Demand inverted. Weights are `null`
until management sets them, so the placeholder is equal weighting and every response
reports `weighting: "equal_unweighted_placeholder"`. **Bands do not exist until
configured** — `band: null`, `band_reason: "thresholds_not_configured"`.

### 7.2 The attention timeline index

```
attention_t = 100 × sigmoid( Σ_i  w_i · z_i(t) )

  z1  spatial concentration          conc_t
  z2  motion energy                  mean |frame_t − frame_{t−1}|, luma
  z3  face presence                  max face-region saliency mass
  z4  saliency stability             1 − EMD(S_t, S_{t−1})   (a settled gaze holds)
  z5  text load                      −reading_load_t          (negative term)
  z6  novelty                        recency of the last shot change
```

Each `z_i` is standardised against the percentile reference (§7.5). The weights `w_i` are
the **one genuinely fitted part of this feature** and are `null` until Phase 5 fits them
against platform retention data; until then the config carries a documented uniform
placeholder and every response says `basis: "derived_composite"`,
`calibration: "provisional_absolute"`.

`engagement_t` (the prototype's second line) is the same index with the novelty and motion
terms up-weighted and concentration down-weighted — one function, two coefficient sets, so
the two lines cannot drift apart in meaning.

### 7.3 Weak zones and key moments — deterministic labels

```
weak zone   attention_t < weak_threshold for ≥ min_weak_seconds   (both config)
peak        global maximum of attention_t
hero        maximum attention in the second half of the runtime
key         first frame where the key-message region takes majority saliency mass
```

These produce the PEAK / KEY / WEAK / HERO strip and the timeline markers. They are rules
over measurements, not model output, which is why the marker text can quote exact numbers.

### 7.4 Fix recommendations

The pipeline emits a **defect list** from measurements alone — `dead_zone`,
`late_brand_appearance`, `overloaded_slide`, `unclosed_promise` (opening claim not restated
at the close, from transcript similarity), `silent_logo_hold`, `stat_overhold`,
`weak_cta`, `competing_peaks`. Gemini then writes the `title` / `why` / `fix` prose **for
the defects it was handed**, and `evidence.py` rejects any recommendation whose timestamps
or quoted numbers do not match the measurement record. `scores_impacted` is derived from
which measured signal produced the defect — not from the model's opinion.

This is what makes the prototype's cards trustworthy: "34 words on screen for 2.5 s" is a
counted fact, and "average adult reading speed is ~4 words/second" is a config value the
report can cite.

### 7.5 Calibration — how raw signals become 0–100

Raw saliency and CV statistics have no natural 0–100 meaning. Build
`vision_lab/percentile_reference.json` the same way `purchase_probability_model` does:
run the corpus of analysed ads, store the empirical distribution of each raw signal, and
map a new creative's raw value to its percentile.

Until the corpus reaches `VL_CALIBRATION_MIN_N` (suggest 200 creatives), report
`calibration: "provisional_absolute"` and use documented absolute anchors. **A score of 63
must never imply "63rd percentile" while the reference set is 12 ads.** When the corpus is
large enough the mode flips to `percentile`, and every historical creative is re-derived
through the rescore endpoint — no reprocessing.

---

## 8. The 15 psychological triggers

Config-driven, exactly like `sales_call_analyzer/sales_framework.json`. Each trigger in
`psychology_framework.json`:

```jsonc
{
  "id": "loss_aversion",
  "name": "Loss Aversion — The Fear of Missing Out",
  "definition": "…one paragraph, the definition we are actually judging against…",
  "detection": "hybrid",                     // "measured" | "judged" | "hybrid"
  "measured_signals": ["deadline_phrase", "scarcity_numeral", "cohort_close_date"],
  "applies_when": { "requires": [] },        // e.g. anchoring requires a price on screen
  "not_applicable_reason": null,
  "rating_levels": ["absent", "weak", "adequate", "strong"],
  "weight": null,                            // management has not decided
  "where_it_should_land": ["close"]          // which part of the runtime it belongs in
}
```

**The important design choice: several triggers are measured, not opined.** That is what
stops this becoming a horoscope.

| Trigger | Detection | How |
|---|---|---|
| 1. Halo Effect | hybrid | First 3 s: measured attention capture + Gemini's read of first-impression quality. |
| 2. Serial Position | **measured** | Does the core claim occupy the first *and* last 3 s? Transcript + OCR against the claim Gemini identifies. |
| 3. Recency | **measured** | Is the opening promise restated at the close? Text similarity between the first and last segments. This is exactly the prototype's fix #3. |
| 4. Mere Exposure | **measured** | Count and total duration of distinct brand exposures. Shares its signal with Brand Memory. |
| 5. Loss Aversion | hybrid | Deadline / scarcity lexicon in transcript and OCR, then Gemini judges whether it is real or decorative. |
| 6. Compromise Effect | **measured** | Are exactly three options presented? Counted from OCR option blocks. |
| 7. Anchoring | hybrid | Price detected by OCR regex; Gemini judges whether an anchor precedes it. **Not applicable when no price is shown** — reported, not scored zero. |
| 8. Choice Overload | **measured** | Count of distinct CTAs and on-screen options; also feeds Cognitive Demand. |
| 9. Framing Effect | judged | Gain vs loss framing of the core claim. |
| 10. IKEA Effect | judged | Does the creative invite participation (assessment, quiz, application) rather than passive consumption? |
| 11. Pygmalion Effect | judged | Does it address the viewer as already capable of the outcome? |
| 12. Confirmation Bias | judged | Does it open from a belief the persona already holds? Requires Brand Brain persona; reports `no_brand_brain` when absent. |
| 13. Peltzman Effect | hybrid | Risk-reversal lexicon (money-back, free trial, cancel anytime, no obligation) + Gemini's read of perceived risk. |
| 14. Bandwagon Effect | **measured** | Adoption numerals and social-proof phrasing. The sample ad's "Four thousand two hundred directors have been certified" is a textbook hit. |
| 15. Blind-Spot Bias | **report-level, not scored** | See below. |

**Trigger 15 is different and should not be forced into the scorecard.** Blind-Spot Bias
is about *the marketer* failing to see their own assumptions — it is not an artefact
present in the frames. Model it as a single report-level observation ("this creative
assumes the viewer already accepts X") with no score and no weight. Scoring it would mean
inventing a measurement, which breaks the rule this whole design rests on.

Two more rules, both inherited from the sales framework:

- **Not applicable is not zero.** A trigger the creative had no opportunity for
  (Anchoring with no price, Compromise with no options) is excluded from the denominator
  and reported as such. Marking a creative down for not showing a price it was never
  supposed to show is an artefact, not a finding.
- **Unsupported is not zero either.** A trigger whose evidence fails verification against
  the transcript and OCR is excluded and reported as `unsupported`.

Triggers connect to the report in two places: the `psychology` block (the audit) and
`recommendations[].trigger_ids` (the trigger a fix would strengthen). They do **not** get
their own score bar in the UI unless management asks for one.

---

## 9. Storage

| Collection | Contents | TTL |
|---|---|---|
| `vision_lab_analyses` | The job document and the assembled report. One read serves a GET. | none |
| `vision_lab_measurements` | Per-frame measurements: saliency stats, regions, OCR boxes, word counts. This is the durable artefact rescore reads. ~48 small records per ad. | 180 d (config) |
| `vision_lab_frames_raw` | Raw saliency arrays, off by default (`VL_STORE_RAW_MAPS`) — megabytes with no consumer beyond debugging. | 14 d |
| Object storage | Sampled frames, heatmap overlays, thumbnails, under `vision-lab/{analysis_id}/`. | 90 d (bucket lifecycle rule) |

**Not stored:** the creative itself, and signed URLs. Same rule as the call recordings.

Idempotency fingerprint over `creative.url` content hash (or `creative_id` + byte length
when the hash is unavailable) **plus** `framework_version`, `psychology_version`,
`prompt_version`, `saliency_model`, `sample_fps`. A re-upload of the same file costs
nothing; a new framework version is genuinely a different analysis and is allowed to run.

Indexes: `{fingerprint}`, `{creative_id, created_at}`, `{division, ad_number, created_at}`
(the History tab), `{status, created_at}` (the worker's claim query).

---

## 10. Configuration

New environment variables, all documented in `.env.example` in the existing style:

```bash
# --- Vision Lab -------------------------------------------------------------
VL_MODEL_DIR=/root/models/vision_lab      # ONNX saliency weights; empty = feature degrades
VL_SALIENCY_MODEL=                        # chosen in the Phase 2 bake-off
VL_SAMPLE_FPS=2.0
VL_MAX_FRAMES=120
VL_MAX_DURATION_SECONDS=180
VL_MAX_CONCURRENT_JOBS=1                  # CPU-bound; keep at 1 on the current box
VL_JOB_STALE_SECONDS=1800
VL_OCR_ENGINE=paddle                      # paddle | tesseract | none
VL_OCR_ON_SHOT_CHANGE_ONLY=true           # the main speed lever
VL_STORE_RAW_MAPS=false
VL_CALIBRATION_MIN_N=200
VL_POLL_INTERVAL_SECONDS=5

# AWS S3 — the bucket ScaleSerum already uses for ad creatives. We read the
# creative from it and write heatmaps back under our own prefix.
VL_S3_BUCKET=
VL_S3_REGION=
VL_S3_PREFIX=vision-lab/
AWS_ACCESS_KEY_ID=                        # a dedicated IAM user, scoped to the prefix
AWS_SECRET_ACCESS_KEY=
VL_SIGNED_URL_TTL_SECONDS=3600            # for the heatmap URLs we hand back
```

Server-side, new and required: **ffmpeg**. Add it to `deploy/README.md`'s provisioning
list and to `deploy.sh`'s preflight so a missing binary fails the deploy loudly rather
than every job silently.

`ecosystem.config.js` gains a second app:

```js
{ name: "vision-worker", script: "vision_lab/worker.py",
  interpreter: "./venv/bin/python", cwd: "/root/Marketing_tool",
  max_memory_restart: "1200M" }
```

---

## 11. Business values that must be decided (all `null` until then)

Every one of these is a decision management makes, not a number engineering invents. They
sit `null` in `vision_framework.json` and every response lists which were unconfirmed —
the same discipline as `sales_framework.json`.

1. **Metric weights** for the overall score (six values).
2. **Score bands** / pass-fail thresholds. *(Explicitly not Script Lab's 90/70/50.)*
3. **Weak-zone threshold** and minimum duration (what counts as a dead zone).
4. **Reading speed** — defaults to 4 words/second, cited in the report; confirm or replace.
5. **Brand Memory targets** — how early is early enough, how many seconds of exposure.
6. **Trigger weights**, and whether triggers affect the overall score at all or stay an
   audit alongside it.
7. **Timeline index coefficients** — placeholder until fitted in Phase 5.
8. **Minimum creative duration** below which a video is not scored.

---

## 12. Delivery phases

| Phase | Scope | Ships | Est. |
|---|---|---|---|
| **0. Contract freeze** | `models.py`, both framework JSONs, reason codes, this document reviewed. No ML. Frontend can build against the schema immediately. | The contract | 2–3 d |
| **1. Skeleton service** | Endpoints, Mongo store, fingerprint idempotency, worker claim loop, heartbeat, stale reaping, always-200 envelope, History tab. Returns a real report from **stub** vision numbers. | Working end-to-end flow, frontend unblocked | 4–5 d |
| **2. Vision core** | ffmpeg sampling, saliency bake-off + ONNX export, OCR, face/brand/CTA detection, heatmap render, storage. | Real heatmaps and real measurements | 2–3 wk (the bake-off is most of it) |
| **3. Timeline + deterministic scoring** | Attention index, weak zones, key moments, the six scores, defect list, `/rescore`. | The full Attention Report, no LLM | 1–1.5 wk |
| **4. Interpretation** | Deepgram transcript + join, Gemini ratings, 15 triggers, evidence verification, fix prose. | The complete prototype | 1–1.5 wk |
| **5. Calibration** | Percentile reference over the corpus; fit the timeline coefficients against platform retention data; optional MIT300 benchmark submission for a quotable number. | Scores that mean something | ongoing |

Phases 1 and 2 can run in parallel by two people — the stub boundary in `saliency.py` is
the seam.

---

## 13. Tests

Mirroring `tests/test_sales_call_*.py`:

- `test_vision_lab_scoring.py` — synthetic saliency maps and measurement records in,
  exact scores out. Every formula in §7 pinned. Null weights produce equal weighting and
  say so; null bands produce `band: null`.
- `test_vision_lab_timeline.py` — a hand-built attention curve produces the expected weak
  zones, peak, hero and key labels.
- `test_vision_lab_psychology.py` — each measured trigger fires on a crafted
  transcript/OCR pair and stays silent otherwise; `not_applicable` never becomes zero.
- `test_vision_lab_evidence.py` — a recommendation quoting a timestamp or number absent
  from the measurement record is rejected.
- `test_vision_lab_api.py` — idempotent resubmit, poll envelope, failure envelope,
  rescore-after-config-change, 503 when the package or storage is unconfigured.
- `test_vision_lab_pipeline.py` — a killed worker leaves a job that the reaper turns into
  `processing_interrupted`, not a row spinning forever.

A golden-file test over 3–5 real ads guards against a silent model or config regression:
scores must stay within a tolerance, and any change must be an explicit re-baseline.

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| **The timeline index is a proxy, and clients will read it as measurement.** | Label it everywhere (`basis: "derived_composite"`, the "AI-predicted, not eye-tracking" caption). Phase 5 calibration against retention data. Never call it "attention measured". |
| **CPU inference makes analysis slow enough to feel broken.** | Async job + poll from day one (already the plan). Shot-change-gated OCR. If it is still too slow, one small GPU box for the worker only — the architecture already isolates it. |
| **Model licence blocks commercial use.** | Licence review is a **gate** in Phase 2, before integration work, not after. |
| **OCR quality on stylised ad typography.** | Benchmark two engines on our own frames in Phase 2; Cognitive Demand degrades gracefully with `ocr_unavailable` rather than producing a wrong word count. |
| **Brand detection without brand assets.** | OCR-name fallback, `no_brand_assets` reported, and Brand Memory says which basis it used. Ask for the wordmark in the Brand Brain onboarding. |
| **200 MB uploads.** | Presigned direct-to-bucket; the API never sees bytes. |
| **A second pm2 process on a small VPS.** | `max_memory_restart`, concurrency 1, and ONNX Runtime instead of torch to keep the footprint small. |
