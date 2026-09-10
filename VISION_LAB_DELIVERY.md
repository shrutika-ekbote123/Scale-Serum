# Vision Lab — build, test, deploy, integrate

Companion to [`VISION_LAB_PLAN.md`](VISION_LAB_PLAN.md). That document is *what we
build*; this one is *how it ships* — the service boundary, the test layers, the
deploy changes it forces, and the handoff pack the backend and frontend teams need.

This repo is **one microservice**: the ScaleSerum AI Service. It owns Gemini,
Deepgram, the purchase-probability model and now the vision pipeline. It does not
own users, brands, billing, ad accounts or the UI. Everything below follows from
that.

---

## 0. Two facts that shape this entire plan

**`main` deploys itself.** `.github/workflows/deploy.yml` triggers on every push to
`main`, SSHs to the VPS, pulls the commit, reinstalls deps, restarts pm2 and
health-checks with automatic rollback. There is no staging branch. So a
half-finished Vision Lab merged to `main` is *in production the same minute*.

Consequence: **every phase must be safe to merge.** Two mechanisms, both already
house patterns —

1. **Defensive import.** `app.py` imports the package in a try/except exactly as it
   does for `sales_call_analyzer` and `purchase_probability_model`. A broken or
   incomplete package degrades Vision Lab only; onboarding, Script Lab, purchase
   probability and sales calls keep working.
2. **`VL_ENABLED` flag**, default `false` on the server until Phase 4 is signed off.
   Merged code is inert; the endpoints answer `503 vision_lab_disabled`. Turning the
   feature on is then a one-line `.env` edit and a `pm2 restart` — not a deploy.

**The frontend must never hold `X-API-Key`.** Every `/api/*` route on this service
is gated by `require_api_key`. The older `FRONTEND_HANDOFF.md` says "no auth
headers" — that doc predates the key and is now wrong. See §6.1: the backend
proxies, the browser never sees the key.

---

## 1. The service boundary

```
   ┌─────────────┐        ┌──────────────────┐        ┌────────────────────┐
   │  Frontend   │──────▶ │  ScaleSerum      │──────▶ │  AI Service        │
   │  (React)    │ ◀──────│  Backend (Node)  │◀────── │  (this repo)       │
   └─────────────┘        └──────────────────┘        └────────────────────┘
         │                        │                            │
         │  presigned PUT         │  presigns the upload       │  reads the URL,
         ▼                        ▼                            ▼  forgets it
   ┌──────────────────────────────────────────────────────────────────────┐
   │                          Object storage                              │
   └──────────────────────────────────────────────────────────────────────┘
```

| Concern | Owner | Notes |
|---|---|---|
| The creative file | **Backend** | Presigns the upload; owns the bucket and its lifecycle rules |
| Calling Vision Lab | **Backend** | Holds `X-API-Key`. Proxies submit and poll for the frontend. |
| The analysis + report | **AI Service** | Our Mongo, our collections. A microservice owns its own data. |
| `analysis_id` ↔ creative/campaign | **Backend** | Stores our `analysis_id` on its own record. It should not mirror our report. |
| Rendering the report | **Frontend** | Reads the JSON contract in §6.3 |
| Heatmap PNGs | **AI Service writes, backend serves** | We write to the bucket, the report carries URLs |
| Retention / deletion | **Backend** | It owns the customer relationship; we honour a delete call (§6.6) |

**What the backend must *not* do:** re-implement scoring, cache our report and
serve a stale copy, or read our Mongo directly. If it needs a number we do not
return, that is a contract change here, not a query there.

---

## 2. Phase plan as shippable increments

Each row is one PR onto `main`. Each is independently deployable and inert until
`VL_ENABLED=true`.

| PR | Contents | Merge-safe because | Unblocks |
|---|---|---|---|
| **1. Contract** | `vision_lab/models.py`, `__init__.py` (statuses + reason codes), `vision_framework.json`, `psychology_framework.json`. No routes. | Nothing imports it yet | Nothing yet — but the schema is now reviewable |
| **2. Skeleton service** | Store, worker claim loop, all five endpoints, defensive import in `app.py`, `VL_ENABLED`, URL allowlist, health fields. **Vision layer stubbed** — returns a fixed, obviously-fake report. | Flag off; defensive import | The whole contract becomes testable by hand from Postman (§3.5), and the handover docs can be written and reviewed against a running service rather than a schema. |
| **3. Deploy plumbing** | `ecosystem.config.js` second app, `deploy.sh` two-process restart, ffmpeg preflight, worker health. | Deploy scripts change, app behaviour does not | Worker runs in production, doing nothing |
| **4. Vision core** | ffmpeg, saliency (post-bake-off), OCR, detectors, heatmap render. | Behind the same flag | Real measurements |
| **5. Scoring** | Timeline, key moments, six scores, defect list, `/rescore` | Same | The real Attention Report |
| **6. Interpretation** | Deepgram join, Gemini ratings, 15 triggers, evidence verification, fix prose | Same | Feature complete → flip `VL_ENABLED=true` |
| **7. Calibration** | Percentile reference, fitted timeline coefficients | Config only, rescore-able | Scores that mean something |

> **PR 2 is the one that matters for the other two teams.** They get real
> endpoints, real status transitions, real error envelopes and a real report shape
> weeks before the ML exists. The stub report must be *obviously* stubbed —
> `"stub": true` at the top level and scores of `0` — so nobody demos it by
> accident.

Branch naming follows the existing history (`plain-language-purchase-probability-factors`):
`vision-lab-contract`, `vision-lab-skeleton`, `vision-lab-deploy`, and so on. PR to
`main`, squash or merge commit as the repo already does.

---

## 3. Testing

### 3.1 The rule CI enforces

**Every test that runs in CI must be offline** — no network, no MongoDB, no
Deepgram, no Gemini, **no ffmpeg and no ONNX model**. The existing sales-call suite
already proves this is achievable: Deepgram is an `httpx.MockTransport`, Gemini is a
`FakeClient`, MongoDB is a `FakeCollection`.

Vision Lab needs one more seam: **the CV layer is injected, not imported**. The
pipeline takes a `VisionDeps` object (mirroring `PipelineDeps` in
`sales_call_analyzer/pipeline.py`) carrying `sample_frames`, `predict_saliency`,
`detect_regions`. In CI those are fakes reading a checked-in fixture; in production
they are the real thing.

```
tests/fixtures/vision_lab/
  ad_042_measurements.json     48 frames of real measurements, captured once
  ad_042_expected_scores.json  the golden scorecard
  frame_0075_regions.json      one frame's OCR + detector output
```

Capturing those fixtures needs ffmpeg and the model **once, locally**. After that
the whole scoring, timeline, trigger and API surface is testable on a bare runner.

### 3.2 Test layers

| Layer | File(s) | Runs in CI | What it proves |
|---|---|---|---|
| **Unit — scoring** | `test_vision_lab_scoring.py` | ✅ | Synthetic measurements in, exact scores out. Every formula in `VISION_LAB_PLAN.md` §7 pinned. Null weights → equal weighting *and* say so; null bands → `band: null`. |
| **Unit — timeline** | `test_vision_lab_timeline.py` | ✅ | A hand-built attention curve produces the expected weak zones, PEAK / KEY / WEAK / HERO labels. |
| **Unit — triggers** | `test_vision_lab_psychology.py` | ✅ | Each measured trigger fires on a crafted transcript/OCR pair and stays silent otherwise. `not_applicable` never becomes zero. |
| **Unit — evidence** | `test_vision_lab_evidence.py` | ✅ | A recommendation quoting a timestamp or number absent from the measurement record is rejected. |
| **Pipeline** | `test_vision_lab_pipeline.py` | ✅ | Status transitions, heartbeat, a killed worker becoming `processing_interrupted`, every failure path producing a stated reason. |
| **API contract** | `test_vision_lab_api.py` | ✅ | Driven through the real app with `TestClient`, exactly like `test_sales_call_api.py`: idempotent resubmit, poll envelope, failure envelope, `/rescore` after a config change, 503 when disabled or unconfigured, **and that the existing endpoints are untouched**. |
| **Golden ads** | `test_vision_lab_golden.py` | ❌ `@pytest.mark.local` | 3–5 real ads end-to-end. Scores must stay within tolerance; any change is a deliberate re-baseline. Needs ffmpeg + weights. |
| **Smoke (HTTP)** | Postman / Newman | ❌ manual + post-deploy | A running instance answers correctly over real HTTP with real auth. |

### 3.3 CI change

One line in `.github/workflows/deploy.yml`:

```diff
-        run: python -m pytest tests/ -q -k "sales_call"
+        run: python -m pytest tests/ -q -k "sales_call or vision_lab"
```

Keep the comment above it accurate — it currently explains why purchase-probability
tests are excluded (they read the real PostgreSQL); add why the golden-ad tests are
excluded (they need ffmpeg and model weights the runner does not have).

The existing **"Import the app"** step is the cheapest and most valuable check we
have: it catches a syntax error, a bad import, or a malformed framework JSON before
anything touches the server. Make sure `vision_lab`'s framework files are loaded at
import time (`_vl_framework.load_framework()` next to the sales-call equivalents) so
a typo in `vision_framework.json` fails CI, not the first request.

### 3.4 Postman collection

Add `postman/ScaleSerum-VisionLab.postman_collection.json`, matching the two that
exist. It must cover, in order, as a runnable folder:

1. `GET /health` — asserts `vision_lab.available === true`
2. `POST /api/vision-lab/analyze` — saves `analysis_id` to a collection variable
3. `GET /api/vision-lab/analysis/{{analysis_id}}` — polls until terminal, asserts the
   status vocabulary
4. The completed report — asserts every field in the §6.3 contract exists
5. `POST /api/vision-lab/analyze` again with the same body — asserts
   `idempotent_hit: true` and that it cost nothing
6. `POST .../rescore` — asserts scores present, no new provider calls
7. Failure cases: bad URL → `creative_unreachable`; missing key → 401; flag off → 503

Variables: `base_url`, `api_key`, `creative_url`, `analysis_id`. Ship it with the
**key blank** — never commit a real one.

### 3.5 Testing the API by hand, with no frontend and no backend

The whole feature is testable from Postman or curl long before either team is wired
in, because the only thing we accept is a URL. Two ways to produce one from the
existing S3 bucket:

```sh
# A time-limited GET link for an object already in the bucket. One line, no code.
aws s3 presign s3://scaleserum-creatives/ads/di_board_seat_v4_final.mp4 --expires-in 3600

# Or upload a test ad first, then presign it.
aws s3 cp ./di_board_seat_v4_final.mp4 s3://scaleserum-creatives/test/
```

Paste the result into the Postman collection's `creative_url` variable and run the
folder. Nothing else about the flow needs to exist — no presign endpoint on the
backend, no upload widget, no React.

Keep a small set of committed test creatives in the bucket under a `test/` prefix:
a good 24 s video, a static JPG, a video with no audio track, a corrupt file, and
one over the size cap. Those five cover most of the reason codes in §6.4, and they
make the Postman failure cases reproducible instead of anecdotal.

> **Accepting arbitrary URLs is server-side request forgery unless it is bounded.**
> In production, validate that the URL is `https`, resolves to a public address, and
> matches an allowlisted host (the S3 bucket domain, plus CloudFront if it is added).
> A dev-only `VL_ALLOW_ANY_URL=true` keeps local testing convenient without shipping
> that hole. Add this at PR 2, not later.

### 3.6 What I test manually before calling a phase done

- `/docs` renders and the new endpoints are executable there.
- A real 24 s ad end-to-end on the server, timed, with `pm2 logs vision-worker`.
- Kill the worker mid-job (`pm2 stop vision-worker`) and confirm the job is reported
  `processing_interrupted` rather than spinning in `analyzing` forever.
- Submit while `MONGODB_URI` is unset → clean 503, service still healthy.

---

## 4. Deploy changes this feature forces

The current pipeline handles **one** pm2 process and checks **one** health URL.
Vision Lab breaks both assumptions. These are real edits, not documentation.

### 4.1 `deploy.sh` must restart two processes

Today it restarts `$PM2_NAME` and rolls back if `/health` fails. With a worker, a
green deploy can still leave every job stuck in `queued`.

- Introduce `PM2_NAMES="marketing-tool vision-worker"` (keep `PM2_NAME` working as a
  fallback so nothing else breaks).
- Restart both; if either is missing, start from `ecosystem.config.js` — the existing
  fallback logic already does this for one name, extend it to the list.
- **Rollback must restart both too.** A rollback that reverts the code but leaves the
  new worker running is the worst of both states.

### 4.2 The health check must cover the worker

Extend `/health` in the existing style — booleans and names only:

```jsonc
"vision_lab": {
  "available": true,
  "enabled": true,
  "storage": "configured",
  "model": "configured",              // VL_MODEL_DIR non-empty
  "ffmpeg": "present",
  "worker": {
    "seen_seconds_ago": 4,            // from the worker's heartbeat document
    "healthy": true,
    "in_flight": 1
  }
}
```

The worker writes a heartbeat document to Mongo every few seconds; `/health` reads
its age. `deploy.sh` keeps reading `ok` (unchanged), and gains an **optional**
second check — `WORKER_HEALTH_REQUIRED=true` — so the worker being down can fail a
deploy once we trust it, without changing behaviour for the other features today.

### 4.3 ffmpeg preflight

`ffmpeg` is a new system binary and `git reset --hard` will not install it. Add to
`deploy.sh`, before the pm2 restart:

```sh
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg is not installed — apt install -y ffmpeg" >&2; exit 1; }
```

Fail the deploy loudly rather than have every job fail quietly at runtime. Add
`ffmpeg` to the provisioning list in `deploy/README.md` §"Rebuilding from scratch".

### 4.4 Model weights live outside the app directory

`deploy.sh` runs `git reset --hard` on every deploy — anything inside
`/root/Marketing_tool` that is not committed is wiped. Weights are ~40 MB of binary
and do not belong in git.

So: `VL_MODEL_DIR=/root/models/vision_lab`, provisioned **once** by hand (or by a
separate `workflow_dispatch` job), never touched by a normal deploy. `saliency.py`
reports `saliency_model_unavailable` when the directory is empty — a stated
degradation, exactly as `purchase_probability_model` handles a missing `.pkl`.

Record the model's SHA-256 in `vision_framework.json` and have the worker log a
warning at boot when the file on disk does not match. A silently swapped model that
changes every score is otherwise invisible.

### 4.5 Dependency weight

`onnxruntime` + `opencv-python-headless` + an OCR engine is a large install on a
small VPS, and `deploy.sh` runs `pip install -r requirements.txt` on **every**
deploy. Before PR 4:

- Check free disk (`df -h`) and pip cache size on the box.
- Pin exact versions — an unpinned `onnxruntime` picking up a new major mid-deploy is
  a production incident with no code change to blame.
- Use `opencv-python-**headless**` (no GUI/X11 deps) and `paddleocr`'s CPU wheel.
- Confirm `pip install` still finishes inside the deploy's patience; if not, split
  the heavy deps into `requirements-vision.txt` installed only when changed.

### 4.6 Server `.env`

`.env` is gitignored and never touched by a deploy — adding a variable is a manual
server-side edit. So the **first** Vision Lab deploy needs a planned `.env` update
(the block in `VISION_LAB_PLAN.md` §10) *before* the flag is flipped. Everything
must default sensibly when absent, so the order of operations can never brick a
deploy.

### 4.7 nginx

No change needed for uploads — they bypass us. The existing 60 s
`proxy_read_timeout` is fine because Vision Lab's POST returns immediately and the
GET is a single Mongo read. **This is a reason to keep the async design even if a
short ad could finish synchronously.**

### 4.8 AWS S3 — the bucket already exists

ScaleSerum already stores ad creatives in S3, so **no new bucket is needed**. We read
the creative from wherever the backend put it, and write our own output under our own
prefix in the same bucket:

```
s3://<existing-bucket>/
    ads/…                          the backend's, we only ever GET
    vision-lab/{analysis_id}/      ours: frames, heatmap overlays, thumbnails
```

**IAM — a dedicated user for this service, scoped to those two paths.** Not `s3:*`,
not bucket-wide:

| Action | Resource | Why |
|---|---|---|
| `s3:GetObject` | `arn:aws:s3:::<bucket>/ads/*` | read the creative to analyse |
| `s3:PutObject` | `arn:aws:s3:::<bucket>/vision-lab/*` | write heatmaps and thumbnails |
| `s3:DeleteObject` | `arn:aws:s3:::<bucket>/vision-lab/*` | the delete endpoint in §6.6 |

Credentials go in the server's `.env` only — never in git, never in the frontend,
never in a log line. If the VPS is ever moved into AWS, replace the keys with an
instance role and delete the user.

**Store S3 keys in Mongo, sign at read time.** Do *not* persist presigned URLs in the
report document: they expire, and a stored report would then serve dead image links
weeks later. The report holds object keys; the GET endpoint signs them fresh on every
read with `VL_SIGNED_URL_TTL_SECONDS`. This keeps the heatmaps private without ever
going stale.

**Two things to check before PR 4:**

- **Region vs. the VPS.** The box is a Vultr VPS, not an EC2 instance, so every
  analysis pulls the full video out of AWS over the public internet — that is billed
  egress (~$0.09/GB, so under a cent for a 60 MB ad — negligible) but it is also
  *latency*. If the bucket sits in `us-east-1` and the server sits in India, a 200 MB
  download may be the slowest step in the entire job. Measure it before assuming the
  timing budget in `VISION_LAB_PLAN.md` §2.3 holds. Fingerprint idempotency limits the
  damage: a given creative is downloaded once, not once per re-poll.
- **Lifecycle rules.** If the bucket already expires objects, our `vision-lab/` prefix
  will be caught by it and old reports will lose their heatmaps while the analysis
  still exists in Mongo. Either exclude our prefix, or accept it and have the frontend
  render a report whose `heatmap.image_url` is null — a decision to make, not to
  discover in six months.

**Bucket CORS** is the backend's concern for the browser's direct PUT. It matters for
us only if the frontend loads heatmaps through `fetch` or draws them to a canvas;
plain `<img>` tags need no CORS rule.

---

## 5. The API submission pack

Vision Lab is built, tested and deployed **complete**, and the API is then submitted to
the backend and frontend teams in one pack. This is what that pack contains — nothing in
it should be written on the last day, all of it is a by-product of the phases above.

| # | Artefact | Audience | Modelled on |
|---|---|---|---|
| 1 | `VISION_LAB.md` — full contract: every endpoint, every field, every reason code, poll semantics, worked request/response examples | Backend | `SALES_CALL_ANALYZER.md` |
| 2 | An **Integration** section inside it — when to call what, what to render per status, what to store | Frontend | `FRONTEND_HANDOFF.md` |
| 3 | `postman/ScaleSerum-VisionLab.postman_collection.json`, runnable against the live server | Both | The two existing collections |
| 4 | `/docs` (Swagger UI) on the deployed service | Both | Automatic |
| 5 | ✅ `tests/fixtures/vision_lab/example_report.json` — one **real completed report**, an actual analysed ad, not a hand-written example | Frontend | New; the frontend builds its rendering against this, and `test_the_report_still_matches_the_example_handed_to_the_frontend_team` fails the build if a field leaves the payload |
| 6 | The reason-code table as a flat list the UI can branch on | Frontend | `VISION_LAB_PLAN.md` §4.3 |
| 7 | A 30-minute walkthrough call: submit → poll → report, live | Both | — |

Item 5 is the one that saves the most time downstream. A frontend developer building
the Attention Report screen needs a real payload with real key moments, real timeline
points and real recommendation text — a schema alone leaves them guessing at array
lengths, null cases and string lengths.

### 5.1 One thing worth doing early, at no cost to this sequence

Handing over the **working API** at the end is a reasonable call. But the **contract**
— items 1, 2 and 6 — is finished at PR 1, weeks before the ML is. Sending just those
documents early costs nothing, changes no deadline, and lets the backend team design
their proxy route and their creative-record schema while we build. If they discover a
field they need, we learn it while the contract is still cheap to change rather than
after it is frozen.

If that is not how the teams prefer to work, nothing above changes — the pack ships
whole at the end either way.

---

## 6. What the backend and frontend need to know

This section is the actual integration contract. It is written to be lifted into
`VISION_LAB.md` at PR 2.

### 6.1 Auth — the backend proxies, always

The browser must never hold `X-API-Key`. A key in frontend JavaScript is a key
published to every visitor, and it would grant access to Gemini spend across *every*
endpoint on this service, not just Vision Lab.

```
Browser  ──▶  POST /api/creatives/:id/vision-lab      (backend's own route, session auth)
Backend  ──▶  POST /api/vision-lab/analyze            (X-API-Key, server to server)
```

The frontend polls the **backend's** route; the backend forwards to ours. This also
gives the backend the natural place to enforce per-brand quotas, which we do not
model.

> If the frontend must call us directly for latency reasons, that needs a *separate*
> short-lived, scoped token minted per analysis — a design change, not a config
> change. Do not solve it by putting the shared key in the browser.

### 6.2 The call sequence

> **`POST /api/vision-lab/upload` exists, and this sequence still does not use it.** It
> is for testing, and for callers holding a file with no bucket access. In production
> the file goes browser → S3 directly (steps 1–3): routing a 150 MB video through the
> AI service would tie up a connection on it for the whole transfer, and the backend
> already owns the bucket.

1. **Frontend** asks the backend for an upload target.
2. **Backend** presigns a PUT to the bucket and returns it.
3. **Frontend** PUTs the file directly. Progress bar lives here — we have no part in
   it and report no upload progress.
4. **Backend** calls `POST /api/vision-lab/analyze` with the object URL, gets
   `analysis_id` + `poll_url` + `suggested_poll_interval_seconds`, stores
   `analysis_id` on its creative record, returns it to the frontend.
5. **Frontend** polls the backend every ~5 s; backend forwards to
   `GET /api/vision-lab/analysis/{id}`.
6. On a terminal status, render.

**Poll rules for whoever writes the loop:** honour
`suggested_poll_interval_seconds`; stop on any status in
`completed | failed | skipped`; give up after ~5 minutes and show the
`processing_interrupted` copy; **never poll faster than 2 s** — it costs a Mongo read
per call and buys nothing, since the worker updates at most once per stage.

### 6.3 Statuses the UI must handle

`queued` · `probing` · `analyzing_frames` · `transcribing` · `interpreting` ·
`scoring` · `completed` · `failed` · `skipped`

Map them to the prototype's Processing screen — they are deliberately narrated so
the progress copy can be specific ("reading 48 frames" rather than a spinner). The
UI must render an unknown status gracefully; new stages may be added.

### 6.4 Errors

Every response is **HTTP 200** with `availability.available=false`, a stable
`reason`, a human `message`, and `scores: null`. The frontend branches on `reason`,
never on the message text. The full list is in `VISION_LAB_PLAN.md` §4.3 and will be
frozen at PR 1.

The only non-200s are: `401` (bad or missing key — a backend bug, never shown to a
user), `404` (unknown `analysis_id`), `409` (rescore of a non-completed analysis),
`503` (feature disabled or storage unconfigured).

### 6.5 What the backend stores

Just `analysis_id`, `status`, `created_at`, and optionally the overall score for list
views. **Not** the report — poll us for it. If list-view latency needs the six
scores denormalised, we add a deliberately small `GET /api/vision-lab/summaries`
endpoint rather than the backend copying our documents.

### 6.6 Deletion

When a customer deletes a creative, the backend calls
`DELETE /api/vision-lab/analysis/{id}` (to be added at PR 2). We remove the analysis,
the measurements and the bucket objects under that analysis prefix. We do not touch
the creative itself — the backend owns it.

---

## 7. Joint integration test — before go-live

Run on the real server, with the real backend, against a real ad. Everyone in the
room.

1. Upload a 24 s MP4 through the real UI → object appears in the bucket.
2. Backend submits → `analysis_id` returned in under 500 ms.
3. Poll shows the status progression, not a jump from `queued` to `completed`.
4. Report renders: heatmap image loads from the bucket, all six scores, key moments,
   timeline, transcript, recommendations.
5. Submit the same file again → `idempotent_hit: true`, no new worker job, no cost.
6. Submit a corrupt file → `creative_unusable`, the UI shows a real message.
7. `pm2 stop vision-worker` mid-job → the job reports `processing_interrupted`; the
   UI offers a retry.
8. `pm2 restart marketing-tool` during a poll → the frontend recovers.
9. Change a weight in `vision_framework.json`, redeploy, call `/rescore` → scores
   move, no ffmpeg or Gemini call in the logs.
10. Check `pm2 logs` for anything that should not be there: no keys, no signed URLs,
    no transcript text (the logging rule in `app.py`).

---

## 8. Go-live checklist

- [ ] `ffmpeg` installed; `deploy/README.md` provisioning list updated
- [ ] `VL_MODEL_DIR` provisioned outside the app dir; SHA recorded and verified at boot
- [ ] Server `.env` has the full Vision Lab block
- [ ] IAM user created and scoped to the two prefixes; keys in the server `.env` only
- [ ] `vision-lab/` prefix confirmed against the bucket's existing lifecycle rules
- [ ] Bucket region vs VPS download time measured on a real 100 MB+ ad
- [ ] URL-host allowlist on, `VL_ALLOW_ANY_URL` off in production
- [ ] Five test creatives committed to the bucket's `test/` prefix
- [ ] `deploy.sh` restarts and rolls back **both** pm2 processes
- [ ] `/health` reports the worker heartbeat; deploy fails when the worker is down
- [ ] `pm2 save --force` run so both apps survive a reboot
- [ ] CI runs `-k "sales_call or vision_lab"` and it is green
- [ ] Postman collection committed, run green against the server
- [ ] `VISION_LAB.md` handed to backend and frontend, `/docs` verified
- [ ] Golden-ad baseline captured and committed
- [ ] `VL_ENABLED=true`, then the joint test in §7 re-run once on production

---

## 9. Risks specific to this delivery model

| Risk | Mitigation |
|---|---|
| **A partial merge auto-deploys to production** | Defensive import + `VL_ENABLED=false`. Every PR in §2 is inert until the flag flips. |
| **Green deploy, dead worker** | Worker heartbeat in `/health`; `WORKER_HEALTH_REQUIRED` gate in `deploy.sh`. |
| **Backend and frontend blocked waiting on ML** | PR 2 ships the full contract with stubbed vision numbers. They integrate for weeks against real endpoints. |
| **The stub gets demoed as real** | `"stub": true` at the top level and zeroed scores. Remove the flag only when Phase 6 lands. |
| **Heavy pip install breaks a deploy for the other three features** | Pin versions, headless OpenCV, check disk first, split into `requirements-vision.txt` if the install time is unacceptable. |
| **Contract drift after handoff** | The contract is frozen at PR 1 and versioned in the payload (`framework_version`, `prompt_version`). Additive changes only; anything breaking is a new field, never a changed meaning. |
| **API key ends up in the browser** | §6.1 is non-negotiable, and `FRONTEND_HANDOFF.md`'s stale "no auth headers" line gets corrected in the same PR. |
