# Vision Lab — testing every API in Postman

The gate before `git push`. A push to `main` triggers `.github/workflows/deploy.yml`,
which deploys to the VPS — so this run is the last check before production, not a
post-deploy smoke test.

Pytest is not a substitute: it runs on `FakeCollection`, fake providers and a
`TestClient`, and never exercises real HTTP, real `X-API-Key` handling, real JSON
serialization or CORS. A route can be green in pytest and broken over the wire.

Collection: `postman/ScaleSerum-VisionLab.postman_collection.json` — 18 requests in
4 folders.

---

## Step 1 · Configure the service for local testing

Edit `.env` (not `.env.example`) and add:

```bash
VL_ENABLED=true
VL_ALLOW_ANY_URL=true          # local only — skips the host allowlist
VL_STUB_DURATION_SECONDS=24
```

`MONGODB_URI`, `MONGODB_DB` and `API_KEY` are already set.

> `VL_ALLOW_ANY_URL=true` is a development convenience. It must be **false** on the
> server, with the real bucket host in `VL_ALLOWED_URL_HOSTS` — otherwise anyone who
> can call the API can make the server issue GETs to arbitrary hosts, including
> private addresses on its own network.

## Step 2 · Start both processes

Vision Lab is two processes. Starting only the API is the single most common reason
a poll never leaves `queued`.

**Terminal 1 — the API:**
```powershell
.\.venv\Scripts\python.exe app.py
```
Expect: `Uvicorn running on http://0.0.0.0:3001`.

**Terminal 2 — the worker:**
```powershell
.\.venv\Scripts\python.exe -m vision_lab.worker
```
Expect: `vision-worker started [id=... concurrency=1 enabled=True]`.

If it says `VL_ENABLED is false - the worker will idle without claiming jobs`, go
back to Step 1. If it exits with `MONGODB_URI is not set`, your `.env` is not being
read from the directory you launched from.

## Step 3 · Import and configure the collection

1. Postman → **Import** → `postman/ScaleSerum-VisionLab.postman_collection.json`
2. Collection → **Variables** tab, set **Current value** (not Initial value — that
   is what gets committed):

| Variable | Value |
|---|---|
| `base_url` | `http://localhost:3001` |
| `api_key` | the `API_KEY` from `.env` |
| `creative_url` | see Step 4 |

**Never save a real key into Initial value.** The collection is committed with
`api_key` blank and must stay that way.

## Step 4 · Get a creative URL

### Where AWS credentials do and do not belong

This trips people up, so it is worth being precise. There are three places
credentials could live, and only one of them needs any:

| Place | AWS credentials? | Why |
|---|---|---|
| **Your laptop**, to run `aws s3 presign` | **Yes** | Signing is done locally with your own IAM keys |
| **Postman** | **No** | The signature travels in the URL's query string |
| **The Vision Lab service** (`.env`) | **Not yet** | It only fetches a URL. It needs its own IAM user at Step 12, when it starts *writing* heatmaps back to S3 |

A presigned URL is **self-authenticating**: `aws s3 presign` does not call AWS at
all, it computes an HMAC signature locally and embeds it in the query string.
Whoever holds the resulting URL can GET that one object, for as long as the expiry
allows, with no credentials of their own. That is exactly why the backend can hand
us a URL instead of giving us access to its bucket.

So: **do not add `AWS_ACCESS_KEY_ID` to `.env` for this.** Nothing in Milestone A
reads it.

### Easiest — upload the file through the API (`00 Upload a video`)

Open **1 Happy path → 00 Upload a video → Body**, click **Select Files** in the `file`
row and pick your video. Send. The file goes to
`s3://aife-media-prod/vision-lab-uploads/<date>/<id>/`, and the returned link is written
into `creative_url` for you — no S3 console, no presign script, no copy-paste, and no line
break smuggled in from a wrapped terminal.

Then run **01** as normal. Or set `analyze` to `true` in the Body to queue the analysis in
the same call, and go straight to **02**.

Up to 400 MB; `.mp4 .mov .webm .m4v .jpg .jpeg .png .webp`. An oversized file gets **413**
before the body is accepted. Anything else refused comes back **200** with a `reason` —
`unsupported_format`, `creative_unusable` (an empty file) or `upload_failed`.

**This is for testing.** The ScaleSerum frontend should keep uploading to S3 itself and
sending the link to `/analyze`: a 150 MB video through the AI service ties up a
connection on it for the whole transfer, and duplicates what S3 already does.

### Option A — no credentials needed (Milestone A)

The stub never downloads anything, so **any well-formed `https://` URL exercises the
entire flow**, including one that points at nothing:

```
https://bucket.s3.amazonaws.com/test/di_board_seat_v4_final.mp4
```

Use this to test the API surface today. Come back to Option B before Step 7 of the
build, when `media.py` starts actually fetching.

### Option B — a real presigned URL, from the credentials already in `.env`

The AWS CLI reads `~/.aws/credentials` or real environment variables. **It does not
read this repo's `.env`**, so credentials sitting there do nothing for
`aws s3 presign`. `scripts/presign.py` closes that gap: it loads `.env` exactly as
`app.py` does and signs with boto3, so there is one place credentials live.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # once

.\.venv\Scripts\python.exe scripts\presign.py --list                # what is in the bucket
.\.venv\Scripts\python.exe scripts\presign.py test/di_board_seat_v4_final.mp4
```

It prints the object, whether it actually exists, the expiry, the host to put in
`VL_ALLOWED_URL_HOSTS`, and the URL to paste into Postman.

Two things it guards against, both of which otherwise cost an afternoon:

- **A region mismatch.** A presign signed against the wrong region produces a URL
  that fails with a signature mismatch rather than anything naming the real problem.
  The script compares `AWS_REGION` against the bucket's actual location and warns.
- **A key that does not exist.** Presigning never checks — it will happily sign a
  URL for something that was never uploaded, and you find out with a 404 when
  something finally fetches it. The script does a `head_object` and tells you.

### Option C — no local setup at all

- **S3 console** → the object → **Object actions** → **Share with a presigned URL**.
- **Ask the backend team.** They already presign uploads for the main app, and it
  doubles as a check that both sides agree on the flow.

### There is no test video in the bucket yet

`aife-media-prod` holds 604 objects, all images under `admin/` — it is the main
app's product-media bucket. Nothing in it is an MP4.

That is fine for Milestone A, because the stub never downloads. **Before Step 7**,
upload a real test creative under a prefix of its own:

```powershell
aws s3 cp .\di_board_seat_v4_final.mp4 s3://aife-media-prod/vision-lab-test/
```

Keep the five fixtures from `VISION_LAB_DELIVERY.md` §3.5 there — a good 24 s video,
a static JPG, a video with no audio, a corrupt file, and one over the size cap.
Those five reproduce most of the reason codes in Folder 2.

> Presigning the same object twice produces two different URLs, because the
> signature and expiry change. The fingerprint deliberately ignores the query
> string — request **04** proves it, and without that behaviour every poll from a
> re-presigning backend would start a brand-new job.

## Step 5 · Run the folders, in order

Use the **Collection Runner** (not one-by-one) so the polling loop works.

**Runner settings that matter:**

| Setting | Value | Why |
|---|---|---|
| **Delay** | `3000` ms | This is what paces the polling loop. Postman's sandbox does not block on `setTimeout`, so the delay *must* come from the Runner. |
| Iterations | 1 | |
| Keep variable values | ✅ on | `analysis_id` is passed between requests |

### Folder 0 · Health

One request. It must show:

```
vision_lab.available      : true
vision_lab.enabled        : true
vision_lab.storage        : "configured"
vision_lab.model          : "configured"
vision_lab.transcription  : "configured"     ← Deepgram key present
vision_lab.interpretation : "configured"     ← Gemini key present
```

The last two are the honest answer to a question you will otherwise ask by opening a
report: a server without them still produces every measurement and every score, but the
transcript and the written recommendations come back stating why they are absent. Seeing
`"not_configured"` here tells you *this server cannot interpret* rather than *this ad had
nothing wrong with it*.

Also asserts no `mongodb+srv` or `amazonaws` string appears in the health body —
health reports names and booleans, never credentials.

The console line `worker: {...}` shows the worker heartbeat. With nothing in flight,
`healthy` is `null` — that means *unknown*, not *broken*: an idle worker writes no
heartbeat because there is nothing to beat about.

### Folder 1 · Happy path (9 requests)

| # | Request | What it proves |
|---|---|---|
| 00 | Upload a video | *Optional.* Stores your file and fills `creative_url` automatically. **Pick the file in its Body tab first** — Postman cannot remember a local file path inside a shared collection, so it ships with the row empty. |
| 01 | Submit | Returns an `analysis_id` in under 1.5 s. **If this takes 30 s, the work is happening in the API process and the architecture is broken.** |
| 02 | Poll until finished | Re-runs itself until terminal. Asserts `scores: null` while active — a half-finished analysis must never show numbers. |
| 03 | The report contract | Every field the frontend renders; six metrics; `cognitive_demand.direction == "lower_better"`; `band: null` with `band_reason`; `config_disclosure.weighting`; **and that `X-Amz-Signature` appears nowhere in the response.** |
| 04 | Resubmit | `idempotent_hit: true`, same id, no second job. |
| 05 | By creative | Recovers the analysis from `creative_id` alone. |
| 06 | History | Filters by `ad_number`, newest first. |
| 07 | Rescore | 200, still completed, **under 2 s**. Watch Terminal 2 while it runs — *nothing should appear in the worker log*. That silence is the whole point of storing measurements. From Milestone D it must also cost **no Gemini call**: the stored ratings are reused, so `scores.clarity.basis` stays `"hybrid"` and `recommendations` survive, re-anchored to the recomputed defects |
| 08 | Delete | Removes the analysis and its measurements. |
| 09 | Deleted is gone | 404. |

**From Milestone D, `stub` is `false`** and request 03 should see a finished report.
Nothing in the payload is a placeholder; anything absent says why it is absent. Check
these by eye once on a real ad — they are the things a passing assertion will not tell
you:

| Look at | What a correct report shows |
|---|---|
| `transcript.available` | `true` with per-line `attention` and `in_weak_zone`, or `false` with a reason. A silent cutdown is `no_audio_stream` — not a failure |
| `interpretation.available` | `true`, plus `evidence_audit`. **Read the audit.** It lists what the model tried to say and could not support — that is the number you watch over time |
| `recommendations` | One per defect it was given, each with an `anchor` carrying the defect's timestamp and its counted numbers. **A recommendation with no anchor is a bug** |
| `interpretation.dropped_invented_recommendations` | Usually empty. A name here means the model nominated a defect of its own and was refused |
| `psychology.triggers` | Exactly 15, each with a `status`. `not_applicable` ≠ `absent` ≠ `unsupported` — the creative had no opportunity, the creative did not do it, we could not stand behind the claim |
| `scores.*.reason` | Present wherever a score is partial. `key_message_not_on_screen` on Focus means the central claim is **spoken**, so there was no on-screen element to measure gaze against |
| `key_message.model_quote_rejected` | `true` means the model mistyped the quote and we substituted the real transcript line. Common on Hinglish creatives, and not an error |

**A report with zero recommendations and `interpretation.available: true` is worth a
second look.** It is legitimate when no defect was found — check `defects` is empty too.
If `defects` has entries and `recommendations` does not, read `evidence_audit`: something
the model wrote was refused, and the audit says what.

### Folder 2 · Failure paths (7 requests)

This is the half that catches regressions — the happy path rarely breaks quietly.

| Request | Expected |
|---|---|
| Missing API key | `401` |
| URL off the allowlist (`169.254.169.254` — the cloud metadata address) | `200` + `reason: url_not_allowed` |
| Plain `http://` | `200` + `reason: url_not_allowed` |
| No creative URL | `200` + `reason: no_creative` |
| Unsupported format (`application/pdf`) | `200`, then the worker fails it with `unsupported_format` |
| Unknown `analysis_id` | `404` |
| Rescore an unfinished analysis | `409` |

Note the shape: **failures are `200` with a stable `reason`**, not 4xx. The frontend
branches on `reason`, never on the message text. Only genuine protocol errors —
missing key, unknown id, wrong state — are status codes.

Set `VL_ALLOW_ANY_URL=false` and restart the API before running this folder, or the
two allowlist tests will pass a URL they should refuse.

### Folder 3 · The rest of the service

One request against `/api/brand-brain/rewrite-persona`. Vision Lab must not be able
to break onboarding, Script Lab, purchase probability or sales calls. Run it after
every Vision Lab change.

---

## Step 6 · Four checks Postman cannot make for you

**1. A killed worker is reported, not left spinning.**
Submit a job, then `Ctrl+C` Terminal 2 while it runs. The document sits in an active
status until its heartbeat goes stale, then the next poll — or the worker's reaper on
restart — turns it into `processing_interrupted`. To see it without waiting 30
minutes, set `VL_JOB_STALE_SECONDS=10` and restart.

*Why this matters: pm2 restarts the worker on every deploy. Without this, every
deploy would strand whatever was in flight.*

**2. Two workers never claim the same job.**
Start a second worker in a third terminal, submit several jobs, and watch the
`claimed [analysis_id=...]` lines. No id appears in both terminals.

**3. The feature flag really disables it.**
Set `VL_ENABLED=false`, restart the API, resubmit → `503` with a clear message, while
`/health` still returns `ok: true` and the other features still answer. This is what
makes merging unfinished work onto a self-deploying `main` safe.

**4. Nothing sensitive reaches the logs.**
Scroll both terminals. There must be no API key, no `mongodb+srv://` string, and no
presigned URL — ids, counts and timings only, per the logging rule in `app.py`.

---

## Step 7 · Automate it (optional, but it pays for itself)

```powershell
npm install -g newman
newman run postman/ScaleSerum-VisionLab.postman_collection.json `
  --env-var "base_url=http://localhost:3001" `
  --env-var "api_key=$env:API_KEY" `
  --env-var "creative_url=https://bucket.s3.amazonaws.com/test/ad.mp4" `
  --delay-request 3000
```

Same assertions, one command, exit code 0 or 1. This is the form to run against the
server after a deploy, and eventually the form to add to CI.

---

## Step 8 · The push checklist

```
[ ] pytest tests/ -q -k "sales_call or vision_lab"     all green
[ ] python -c "import app"                             imports clean
[ ] Folder 0 Health                                    all green
[ ] Folder 1 Happy path                                all green
[ ] Folder 2 Failure paths                             all green (VL_ALLOW_ANY_URL=false)
[ ] Folder 3 Existing endpoints                        all green
[ ] Step 6 manual checks                               done
[ ] api_key blank in the committed collection          confirmed
[ ] .env not staged                                    git status
```

Then push — and remember the push deploys. Re-run Folder 0 and Folder 1 against the
server once it lands, with `base_url` pointed at the real host.

---

## When a poll never leaves `queued`

In order of likelihood:

1. **The worker is not running.** Terminal 2. This is nearly always it.
2. **`VL_ENABLED=false` in the worker's environment.** The API and the worker read
   `.env` separately — the API can be enabled while the worker is not.
3. **The worker is pointed at a different database.** Check `MONGODB_DB` matches.
4. **The worker crashed on boot.** Scroll up in Terminal 2; a malformed
   `vision_framework.json` fails loudly there rather than per job.
