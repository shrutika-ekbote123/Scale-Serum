# Vision Lab — step-by-step build

The executable sequence. [`VISION_LAB_PLAN.md`](VISION_LAB_PLAN.md) is the design and
[`VISION_LAB_DELIVERY.md`](VISION_LAB_DELIVERY.md) is how it ships; **this** is the order
to actually write it in.

**Scope of this document:** what *we* build and test in this repo. The S3 bucket, the
presigned upload, the backend proxy route and the UI are the backend and frontend teams'
work. From our side a creative is a URL that arrives in a request body — nothing more.

Every step has a verify command. **Do not move to the next step until the current one
verifies**, because each later step assumes the previous one is trustworthy.

---

## The gate before every push

> **Nothing is pushed to GitHub until every endpoint it touches has passed a Postman run
> against a locally running instance.** Not after the deploy — before the `git push`.

This is not belt-and-braces. `.github/workflows/deploy.yml` fires on every push to `main`:
SSH to the VPS, pull, reinstall, `pm2 restart`, health-check. There is no staging branch,
so **a push is a production deploy**, and the only thing standing between a bad commit and
the live service is what you checked before it left your machine.

Pytest is not a substitute. The offline suite runs on `FakeCollection`, `FakeClient` and
`httpx.MockTransport` — it never exercises real HTTP, real `X-API-Key` handling, real JSON
serialization of a Pydantic model, or CORS. A route can be green in pytest and broken over
the wire.

So the order at every step is fixed:

```
   write code
      ↓
   pytest -k "vision_lab"        offline, fast, catches logic
      ↓
   run the service locally       .\.venv\Scripts\python.exe app.py
      ↓
   Postman collection run        real HTTP, real auth, real payloads
      ↓
   git push                      ← which deploys
      ↓
   Postman again, against the server
```

Practical consequences for this build:

- **The Postman collection is built at Step 6, not at Step 23.** It grows one folder per
  step from then on, and every new endpoint or field arrives with its request in the same
  commit.
- **Add the failure cases as you go**, not at the end — bad URL, missing key, feature
  disabled, unknown `analysis_id`, rescore of an incomplete analysis. Those are the
  requests that catch regressions; the happy path rarely breaks quietly.
- **Commit the collection with the API key blank.** It lives in `postman/` beside the two
  that already exist.
- When a step below says *Verify*, the pytest command is the first half. If the step
  touched an endpoint, a Postman run is the second half.

---

## Milestone map

| Milestone | Steps | You can demo |
|---|---|---|
| **A — the service works** | 0–6 | Submit a URL, poll, get a report shape back. Vision numbers are fake. |
| **B — real vision** | 7–12 | Real heatmaps and real measurements from a real ad. ✅ **DONE** — trained model installed. |
| **C — real scores** | 13–15 | The complete Attention Report. No LLM anywhere in it. ✅ **DONE** |
| **D — feature complete** | 16–20 | Transcript, 15 triggers, fix recommendations — the prototype. ✅ **DONE** |
| **E — shipped** | 21–24 | Live on the server, docs handed over. |

Milestone A is worth reaching fast. Once it exists, everything after it is a swap of one
stubbed function for a real one, and you always have a running service to test against.

---

# Milestone A — the service works

### Step 0 · Set up

**Do:**
```powershell
git checkout -b vision-lab-contract
.\.venv\Scripts\python.exe -m pytest tests/ -q -k "sales_call"   # must be green before you start
winget install Gyan.FFmpeg                                        # needed from Step 7, install now
```
Then close and reopen the terminal so `ffmpeg` lands on `PATH`.

**Verify:** `ffmpeg -version` prints, and the sales-call suite passes.

**Done when:** you have a green baseline. If the existing tests are red, fix that first —
you cannot tell your own breakage from pre-existing breakage otherwise.

---

### Step 1 · The contract

**Create:**
- `vision_lab/__init__.py` — the package docstring (the division-of-labour contract),
  statuses, reason codes. Copy the *shape* of `sales_call_analyzer/__init__.py`.
- `vision_lab/models.py` — Pydantic request/response models from
  `VISION_LAB_PLAN.md` §4.1–4.2.
- `vision_lab/vision_framework.json` — metric definitions, thresholds, weights.
  **Every business value `null`**, with a `how_to_configure` note, exactly like
  `sales_call_analyzer/sales_framework.json`.
- `vision_lab/psychology_framework.json` — the 15 triggers from §8.
- `vision_lab/framework.py` — loads and validates both JSONs, exposes
  `config_disclosure()`.

**Verify:**
```powershell
.\.venv\Scripts\python.exe -c "import vision_lab as vl; from vision_lab import framework as f; print(f.load_framework()['framework_version'], len(f.load_psychology()['triggers']))"
```

**Done when:** it prints `vision_v1 15`, and a deliberately broken JSON raises a clear
`FrameworkConfigError` rather than a `KeyError` at request time.

> This is the step to be slow and careful on. Everything downstream — and both other
> teams — is shaped by these files. Changing a field name here later is cheap; changing
> it after handover is not.

---

### Step 2 · Storage

**Create:** `vision_lab/store.py` — `AnalysisStore` taking injected collections, with
`create` / `get` / `update_status` / `heartbeat` / `complete` / `fail` /
`find_by_fingerprint` / `latest_for_creative` / `history` / `claim_next` /
`ensure_indexes`, plus `compute_fingerprint`, `new_analysis_id`, `is_stale`.

Read `sales_call_analyzer/store.py` first and follow it — the fingerprint, the stale-job
reaper and the injected-collection rule all transfer directly.

**Create:** `tests/test_vision_lab_store.py` with a `FakeCollection` (lift the one from
`tests/test_sales_call_pipeline.py`).

**Verify:** `pytest tests/ -q -k "vision_lab_store"`

**Done when:** an identical submission returns the same fingerprint; a changed
`framework_version` returns a different one; a document whose heartbeat is older than
`VL_JOB_STALE_SECONDS` reports stale.

---

### Step 3 · Pipeline skeleton

**Create:** `vision_lab/pipeline.py` with a `VisionDeps` dataclass — the seam that makes
everything testable:

```python
@dataclass
class VisionDeps:
    store: AnalysisStore
    sample_frames: Callable      # media.py       — Step 7
    predict_saliency: Callable   # saliency.py    — Step 9
    detect_regions: Callable     # regions.py     — Step 10
    render_heatmap: Callable     # heatmap.py     — Step 12
    transcribe: Callable         # Deepgram       — Step 16
    llm_client: Any              # Gemini         — Step 18
    put_object: Callable         # S3             — Step 12
```

For now every callable is a stub returning fixture data. `run_analysis()` walks the
status transitions, writes a heartbeat at each, and persists a report marked
`"stub": true` with zeroed scores.

**Verify:** `pytest tests/ -q -k "vision_lab_pipeline"`

**Done when:** a job goes `queued → probing → analyzing_frames → scoring → completed`,
each transition is persisted, and any raised exception lands as a `failed` status with a
stated reason — never an unhandled crash.

---

### Step 4 · The worker

**Create:** `vision_lab/worker.py` — the pm2 entrypoint. A loop that claims one job with
`find_one_and_update({status: "queued"}, ...)`, runs the pipeline, heartbeats while it
works, and stops cleanly on SIGTERM (pm2 sends it on every restart).

**Verify:** run it against a local Mongo, insert a queued document by hand, watch it get
claimed and completed.

**Done when:** two workers started at once never claim the same job, and `Ctrl+C` mid-job
leaves a document that the stale reaper later marks `processing_interrupted` rather than
one stuck in `analyzing_frames` forever.

---

### Step 5 · The API

**Edit `app.py`.** One new section, placed after the Sales Call Analyzer section and
built the same way:

1. Defensive import in a `try/except`, with stand-in `BaseModel` classes in the `except`
   so a failed import cannot take the whole service down at `def` time.
2. `VL_ENABLED` flag (default `false`) and `_require_vision_lab()`.
3. The five endpoints from `VISION_LAB_PLAN.md` §4.
4. URL validation: `https` only, host allowlist, `VL_ALLOW_ANY_URL` for dev.
5. Extend `/health` with the `vision_lab` block from `VISION_LAB_DELIVERY.md` §4.2.
6. `lifespan` calls `vision_store.ensure_indexes()`.

**Verify:**
```powershell
.\.venv\Scripts\python.exe -c "import app; print(len(app.app.routes), 'routes')"
.\.venv\Scripts\python.exe app.py     # then open http://localhost:3001/docs
```

**Done when:** the new endpoints appear in Swagger and are executable there; with
`VL_ENABLED=false` they return a clean 503; with no `MONGODB_URI` the service still boots
and the other three features work untouched.

---

### Step 6 · Prove Milestone A

**Create:** `tests/test_vision_lab_api.py`, modelled on `tests/test_sales_call_api.py` —
same approach: import the real app offline, swap in a fake store, stub the deps.

**Create:** `postman/ScaleSerum-VisionLab.postman_collection.json` (contents listed in
`VISION_LAB_DELIVERY.md` §3.4), with the key blank.

**Verify:**
```powershell
pytest tests/ -q -k "vision_lab"
```
Then, against a locally running service, get a real URL and run the collection:
```powershell
aws s3 presign s3://<bucket>/test/di_board_seat_v4_final.mp4 --expires-in 3600
```

**Done when:** submit → poll → stub report works over real HTTP, a duplicate submit
returns `idempotent_hit: true`, and the existing endpoints still pass their own tests.

> **🏁 Milestone A.** Green pytest, then a green Postman run locally, **then** push — and
> remember that the push deploys. Re-run the collection against the server once it lands.
> From here the service is real and every remaining step is replacing one stub.

---

# Milestone B — real vision

### Step 7 · Frames

**Create:** `vision_lab/media.py` — `probe()` (ffprobe: duration, dimensions, audio
stream present?), `sample_frames()` (ffmpeg at `VL_SAMPLE_FPS`, capped at
`VL_MAX_FRAMES`), `detect_shots()` (scene filter), `extract_audio()`.

**Verify:** run it on the real DI ad; you should get 48 frames from 24 seconds. Open a
few and check they are not black or duplicated.

**Done when:** a corrupt file returns `creative_unusable` rather than raising, and a video
with no audio track returns `no_audio_stream` as a fact rather than an error.

---

### Step 8 · The saliency bake-off · ✅ **DONE — UNISAL selected and installed**

| | |
|---|---|
| **Winner** | UNISAL, exported to ONNX (12.2 MB), installed in `VL_MODEL_DIR` |
| **Decision metric** | mean NSS on 99 hand-annotated frames from our own four ads |
| **Result** | UNISAL **3.017** · MSI-Net 2.268 · spectral 1.127 · centre control 0.598 |
| **Same on reviewed frames only** | 2.658 / 1.963 / 1.001 / 0.476 — identical ranking |
| **CPU cost** | 86 ms/frame single-threaded → ~10 s for a 120-frame ad |
| **Licence** | Apache-2.0 repo, weights committed within it. Hollywood-2 in the training mix is recorded in `NOTICE.md` and unresolved. |

Full results in [`vision_lab/BAKEOFF.md`](vision_lab/BAKEOFF.md); licences in
[`vision_lab/BAKEOFF_LICENCES.md`](vision_lab/BAKEOFF_LICENCES.md).

**Ruled out:** UMSI (explicitly non-commercial), DeepGaze IIE (ships no LICENSE
at all — `license='MIT'` is *commented out* in its setup.py, which is an intent
rather than a grant).

**A finding that shapes Milestone C:** neither trained model is good at brand
marks — UNISAL 1.18, MSI-Net **0.49 (below chance)**, against spectral's 1.81.
Logos are small high-contrast graphics that models trained on photographs and
film have no reason to prioritise. **Brand Memory must not lean on saliency
alone**; use the detected brand box for *where it is* and saliency only for *was
this moment attended*.

### The bake-off harness (retained for the next model)

The harness lives in `scripts/bakeoff/`:

| Script | Status |
|---|---|
| `fetch_models.py` | ✅ done — clones candidates, reads their licences, writes `vision_lab/BAKEOFF_LICENCES.md` and `NOTICE.md` |
| `provenance.py` | ✅ done — weights licences and training-data provenance, separately from the repo licence |
| `extract_ad_frames.py` | ✅ done — 100 frames pulled from the four creatives |
| `annotate_template.html` | ✅ done — run `extract_ad_frames.py --serve` to mark them |
| `fetch_datasets.py` | ✅ done — MIT1003 + CAT2000, mirrors verified before downloading |
| `evaluate.py` | ⬜ not built |
| `benchmark_cpu.py` | ⬜ not built |
| `export_onnx.py` | ⬜ not built |

**Licence outcome:** UNISAL (Apache-2.0), MSI-Net (MIT) and TranSalNet (MIT) are
usable on their code licence. DeepGaze IIE ships no LICENSE file at all and UMSI
is explicitly non-commercial — both are out. On *weights* provenance MSI-Net is
the cleanest: SALICON-only, with MIT declared against the model itself on
HuggingFace. UNISAL's weights were trained partly on Hollywood-2, which is
feature-film footage.

**Blocked on:** the 100 frames being annotated. That is the decision metric and
nobody else can supply it.

> **This is the one piece of Milestone B still outstanding.** Everything else is
> built and verified against real ads. Until it is done, `saliency.py` runs a
> classical Spectral Residual baseline, every report carries
> `vision.saliency_trained: false`, and the worker logs a warning at boot.
> Measurements are real, but they are not model-predicted attention and must not
> be shown to a client as such.

### The bake-off itself

This step needs PyTorch and several GB of datasets. It does **not** belong in
`Marketing_tool` — work in a scratch directory. Only two things come back.

**Do:**
1. Pull the candidate checkpoints from `VISION_LAB_PLAN.md` §5.2.
2. **Licence review first.** Anything research-only is out before you spend time on it.
   This is a gate, not a footnote.
3. Score each on MIT1003 and CAT2000: NSS, CC, AUC-Judd, sAUC, SIM, KLD — *and* the
   centre-Gaussian baseline alongside, or you cannot tell whether a model is doing
   anything.
4. Hand-annotate ~100 frames from real ScaleSerum ads with the boxes the client thinks
   should draw the eye. **Mean NSS on those frames is the decision metric.**
5. Time each model on the actual VPS at 384×224. A model 4 % better and 5× slower loses.
6. Export the winner to ONNX and check the ONNX output matches PyTorch's within tolerance.

**Comes back:** one `.onnx` file (to `VL_MODEL_DIR` on the server, never into git) and
`vision_lab/BAKEOFF.md` — the results table, the chosen model, its licence, and its
SHA-256.

**Done when:** you can state in one sentence why this model and not the others, with
numbers.

> Step 8 is the long pole — budget two to three weeks. **Steps 9–15 can be built in
> parallel against a placeholder ONNX model**, because `saliency.py` only needs *a*
> model with the right input/output shape to develop against.

---

### Step 9 · Saliency inference

**Create:** `vision_lab/saliency.py` — load the ONNX session once at worker boot (never
per frame), preprocess, batch, upsample to frame size, **normalise so the map sums to 1**,
and return per-region mass and concentration. Verify the model file's SHA against
`vision_framework.json` at boot and log a warning on mismatch.

**Verify:** every returned map sums to 1.0 ± 1e-5. Assert this in a test; a normalisation
bug silently corrupts every score downstream and is invisible in a heatmap image.

**Done when:** an empty `VL_MODEL_DIR` reports `saliency_model_unavailable` and the
service still boots.

---

### Step 10 · Region detection

**Create:** `vision_lab/regions.py` — OCR (boxes, strings, word counts), faces, brand
(multi-scale template match + OCR name match), CTA (lexicon match).

Benchmark PaddleOCR against Tesseract on 20 real ad frames before committing to one. OCR
is the CPU bottleneck, not saliency.

**Done when:** on the DI ad's 7.5 s frame, OCR returns 34 words in 3 boxes. That is the
number the whole dead-zone finding rests on.

---

### Step 11 · Measurement + fixtures

**Create:** `vision_lab/measure.py` — combines Steps 7, 9 and 10 into one measurement
record per frame, which is the durable artefact `/rescore` later reads.

**Then capture the fixtures.** Run it once on a real ad and commit:
```
tests/fixtures/vision_lab/ad_042_measurements.json
tests/fixtures/vision_lab/frame_0075_regions.json
```

**Done when:** those files exist. **This is the step that unblocks CI** — from here every
scoring, timeline and trigger test runs on a bare runner with no ffmpeg and no model.

---

### Step 12 · Heatmaps

**Create:** `vision_lab/heatmap.py` — colormap, alpha-composite over the frame, number the
top-3 peaks; plus the S3 put. **Store object keys in Mongo, sign URLs at read time** —
never persist a presigned URL in the report.

**Done when:** the rendered overlay for the 14 s frame looks like the prototype's
Attention Report image: hot on the product UI, warm on the wordmark and the stat card.

**✅ Done.** Three defects here were invisible in the code and obvious the moment the
rendered PNG was actually opened. Anything that produces a picture has to be checked by
looking at the picture.

| What the image showed | Cause | Fix |
|---|---|---|
| The ad washed out to flat lavender — presenter, office and certificate all invisible under their own report | `cv2.addWeighted` applied ONE flat 55% alpha to the whole frame, so JET's cold end (dark blue) was painted over every unattended area | Alpha now follows the attention: `weight = intensity × ALPHA` per pixel. An ignored region keeps the ad's real pixels; the gradient between hot and cold is itself the finding |
| The hero frame was a **white flash between two shots** — a near-blank page with one marker | A flash has black letterbox bars over a blank body, so its luma std is HIGH and it sailed through `is_degenerate`. With nothing else to look at it also scored the best saliency concentration in the ad | New `frame_detail` (mean absolute Laplacian) and `is_representative`. Calibrated on two real ads: flash 0.029, its neighbours 0.063 and 0.078, fade-to-black 0.005 — threshold 0.04 |
| Markers were placed but not named | `saliency.peaks()` only ever sees the saliency array, so it has coordinates and no idea what is under them | `regions.label_peaks()` joins each peak box to the detected brand mark, CTA, face or text. The hero frame's regions are re-detected for that ONE frame, because `measure.py` stores `text_boxes` as a count |

`is_degenerate` was deliberately left alone rather than tightened — `timeline.py` uses it,
and moving it would move scores. A frame can be perfectly sound to measure and still be
the wrong one to put at the top of a report, so hero selection got its own predicate.

Peaks now carry `element` (`brand_mark` | `call_to_action` | `face` | `on_screen_text` |
null) and `element_text`, so the report can say *"41% of attention went to her face"* or
*"the eye went to BOARD READINESS"* rather than naming a rectangle. Null is a real answer:
product footage and b-roll are not things we detect, and labelling a hotspot after a
caption that merely clipped its corner would be a guess.

> **🏁 Milestone B.** Real measurements from real pixels.

---

# Milestone C — real scores

### Step 13 · Timeline

**Create:** `vision_lab/timeline.py` — the six-term index from `VISION_LAB_PLAN.md` §7.2,
weak-zone detection, and the PEAK / KEY / WEAK / HERO labels.

**Verify:** `pytest -k "vision_lab_timeline"` with a hand-built curve.

**Done when:** on the real DI ad, 7–10 s comes out as a weak zone and ~14 s comes out as
the hero frame. If it doesn't, the index weights are wrong — fix them before scoring.

---

### Step 14 · Scoring

**Create:** `vision_lab/scoring.py` — the six scores, the overall, and `config_disclosure`.

**The rule:** every number is computed here from measurements and JSON config. If you find
yourself reading an LLM response in this file, stop — that belongs in Step 18.

**Done when:** null weights produce equal weighting *and* the response says so; null bands
produce `band: null` with `band_reason: "thresholds_not_configured"`; and the same
measurement fixture always produces byte-identical scores.

---

### Step 15 · Rescore

**Create:** `POST /api/vision-lab/analysis/{id}/rescore`.

**Verify:** change a weight in `vision_framework.json`, restart, call rescore. Scores move.
Watch the logs: **no ffmpeg, no ONNX, no network.**

**Done when:** that holds. This is the payoff for storing measurements instead of numbers,
and it is the proof that Step 14 stayed pure.

> **🏁 Milestone C.** The full Attention Report, with no LLM in it anywhere.

---

# Milestone D — feature complete ✅ DONE

Verified end to end against two real ads from `Ads_Video/` (an English 82 s ad and a
Hinglish 84 s ad), 45–65 s per analysis on the dev box. Seven real defects were found by
running it rather than by reading it — each is recorded under the step it belongs to and
pinned by a regression test in `tests/test_vision_lab_interpretation.py`.

### Step 16 · Transcript

**Do:** promote `sales_call_analyzer/deepgram_client.py` to `transcription/deepgram_client.py`
and re-export from the old path so nothing breaks. Then wire it in and join each segment
to the timeline by time span.

**Done when:** the transcript panel shows per-line attention numbers, and the sales-call
tests are still green after the move.

**✅ Done.** `transcription/` is a top-level package; `sales_call_analyzer.deepgram_client`
is a true module alias (`sys.modules[__name__] = _module`) — a `from ... import *`
re-export broke 11 tests that monkeypatch module state, because you cannot proxy a
rebound module attribute that way.

`vision_lab/transcript.py` posts the WAV **bytes** ffmpeg already extracted rather than
handing Deepgram a URL: the creative has been downloaded once and deleted, and a second
150 MB fetch of the same file is a cost and a dependency on Deepgram reaching our bucket.
`media.sample_frames` therefore pulls the audio before its `finally` removes the workdir.

**Found by running it:** a closing line that *starts* before the last 3 s but is still
being spoken over them was excluded from "the close", so two triggers reported "nothing is
said in the final seconds" about an ad that ends on its call to action. The window now
matches on overlap.

---

### Step 17 · Measured triggers

**Create:** `vision_lab/psychology.py` — detectors for the eight measured triggers
(Serial Position, Recency, Mere Exposure, Loss Aversion lexicon, Compromise, Anchoring
price detection, Choice Overload, Bandwagon).

**Done when:** Bandwagon fires on *"Four thousand two hundred directors have been
certified"*, and Anchoring returns `not_applicable` with `reason: "no_price_shown"` —
**not** a zero.

**✅ Done.** Ten detectors are measured, not judged. `apply_interpretation()` folds the
model's ratings back on afterwards under three rules: a measured trigger is never
overwritten, a `not_applicable` one stays that way, and a rating whose evidence failed
verification is `unsupported` — not `absent`.

**Found by running it:**
- OCR read `$8` on five frames of an ad showing no price (`$` is a routine misread of
  `S`), which made Anchoring applicable and invited the model to judge a price nobody
  had shown. A price now needs two digits.
- Choice Overload — the one trigger where showing it strongly is a **defect** — returned
  rating `weak` beside status `absent`, so an ad that asked for exactly one thing read as
  having done badly. Ratings now say how strongly a trigger *shows*; the new `polarity`
  field says which direction is good news.
- `regions.text_boxes` is a **count** in a stored record, and the detector called `len()`
  on it. Nothing caught it because the stub record omitted the key, so `or []` supplied a
  list. The stub now carries every key `measure.py` emits, with the same type.

---

### Step 18 · Gemini

**Create:** `vision_lab/analyzer.py` with a `PROMPT_VERSION`. It receives the measurements
and the defect list and returns ordinal ratings, trigger evidence and fix prose.

**The prompt must forbid inventing defects.** It writes about what Python found; it does
not nominate problems of its own.

**Done when:** a response containing a numeric score is rejected by the parser. The model
does not get to produce numbers, and the code should enforce that rather than trust it.

**✅ Done.** Both rules are enforced in `validate()`, not merely asked for: a non-ordinal
rating raises `AnalyzerError`, a recommendation whose `defect_id` was not in the list we
sent is dropped, and `SCORE_PATTERN` strips a score smuggled into prose. The context sent
to the model deliberately contains **no scores** — if it could see them, the easiest thing
it could do is restate them, and the separation would exist only on paper.

`key_message.carrier` was added after the first real run: the model correctly named the
central claim at 70.8 s of a real ad, where the **voiceover** states it. Focus then
measured gaze against on-screen copy that was not there and scored **0** for an ad whose
copy was in fact being read. The model now says which channel carries the claim, and
`scoring.py` checks the frames as well — a claim delivered in voice is not a Focus
failure, and `focus.reason` says `key_message_not_on_screen`.

---

### Step 19 · Evidence verification

**Create:** `vision_lab/evidence.py` — every quoted timestamp, word count and transcript
line is checked against the measurement record. Anything unverifiable is dropped and
reported as `unsupported`.

**Done when:** a hand-corrupted LLM response quoting a timestamp that does not exist is
rejected in a test.

**✅ Done** — `test_a_timestamp_that_does_not_exist_in_the_creative_is_rejected` is that
test: 34 s cited in a 20 s creative, evidence dropped, trigger reported `unsupported`.

Four things this got wrong on real ads, all now fixed and pinned:

| What happened | Why it was wrong | The rule now |
|---|---|---|
| **Every** recommendation dropped on a real ad | `fix` prose was checked for unmeasured numbers, but "cut this to 16 words" is a **target**, not a claim. Both were sound arguments about real defects | Only `title` and `why` are checked. A prescription describes an ad that does not exist yet |
| A timeline point at 34.8 s — a number **we** showed the model — rejected as invented | The allowed set was assembled by naming keys by hand and had drifted from the context actually sent | The set is walked from the context. What we showed it, it may quote back |
| A verbatim Hinglish quote rejected | Punctuation normalises to a space, leaving `है, सर` as two spaces where a word-by-word compare gives one | Whitespace is collapsed. Half of ScaleSerum's creatives are Hinglish |
| A trigger rated `absent` reported as `unsupported` for citing nothing | There is nothing to quote for a thing the ad did not do | Only a **positive** rating needs evidence |

**Citations are pointers, not transcriptions.** Across two runs of the same Hinglish ad
the model reproduced one line correctly and then as `दहले डेरे लषआ इसकव`. Devanagari does
not survive being retyped by a model reliably, so when a quote fails but its **timestamp**
lands on a real line, that line's own text becomes the quote and the report says
`model_quote_rejected: true`. This is stricter than trusting the model's copy — the
published words are now always ours — and a repair that would duplicate a line already
cited is dropped rather than counted twice.

---

### Step 20 · Assembly

**Create:** `vision_lab/report.py` — assembles the response. **It never rescores**; it
reads what Step 14 produced.

**Done when:** the payload matches `VISION_LAB_PLAN.md` §4.2 field for field, and a real
ad produces a report you would show a client.

**✅ Done.** `vision_lab/report.py` assembles and never computes — if it ever starts
calculating, a report and a `/rescore` of the same analysis can disagree and no reader can
tell which is wrong. Every block states whether it is there: a missing transcript, an
unavailable interpretation and an unjudgeable trigger each carry a reason rather than
being omitted, because an absent key and a null are the same thing to a frontend and very
different things to a marketer. `stub` is now `false`.

`/rescore` reuses the stored ratings and key message, so Clarity keeps its blend and Focus
keeps its window **without a second Gemini call**, and recommendations are re-anchored
against the recomputed defects — one whose defect no longer exists under new config is
dropped rather than left pointing at a finding that has stopped existing.

> **🏁 Milestone D.** ✅ Saved as `tests/fixtures/vision_lab/example_report.json` — a real
> report from a real ad, item 5 of the submission pack. Its shape is pinned by
> `test_the_report_still_matches_the_example_handed_to_the_frontend_team`, so a field
> quietly leaving the payload fails the build instead of the frontend's integration.
>
> **The first version of it was generated with S3 upload switched off**, so the Attention
> Report — the screen the frontend most needs a reference for — came back as
> `peaks: []`, `image_url: null`, no thumbnails. The contract test now asserts the example
> actually demonstrates a rendered heatmap with named, fractional-boxed peaks and a
> key-moment strip, so it cannot regress to a reference file that documents everything
> except the picture.
>
> It also asserts no live presigned signature is committed. The generator signs the images
> and fetches them to prove the links resolve, then redacts the signature before writing —
> a signed URL is a credential, and this file goes to another team.

---

# Milestone E — ship

### Step 21 · Deploy plumbing

**Edit:**
- `ecosystem.config.js` — add the `vision-worker` app.
- `deploy/deploy.sh` — restart **both** pm2 processes, roll back **both**, and add the
  ffmpeg preflight.
- `.github/workflows/deploy.yml` — change the test filter to
  `-k "sales_call or vision_lab"`.
- `requirements.txt` — the new deps, **pinned**, in the existing commented style.
- `.env.example` — the full Vision Lab block.
- `deploy/README.md` — ffmpeg in the provisioning list, `VL_MODEL_DIR` provisioning.

**Done when:** a deploy to the server restarts both processes and `/health` reports a live
worker heartbeat.

---

### Step 22 · Server preparation

On the box, once, by hand: `apt install ffmpeg`, create `/root/models/vision_lab` and put
the ONNX file there, add the Vision Lab block to `.env` (still `VL_ENABLED=false`),
`pm2 save --force`.

**Done when:** `pm2 list` shows both processes online after a reboot.

---

### Step 23 · The submission pack

Write `VISION_LAB.md` (the contract, modelled on `SALES_CALL_ANALYZER.md`) with the
Integration section for the frontend, finish the Postman collection against the live
server, and collect the seven items in `VISION_LAB_DELIVERY.md` §5.

**Done when:** someone who has never seen the code can integrate from those documents
alone.

---

### Step 24 · Go live

Work through the checklist in `VISION_LAB_DELIVERY.md` §8, set `VL_ENABLED=true`,
`pm2 restart` both, and run the joint test in §7 once on production.

---

## What to do first, concretely

1. Step 0 — half an hour.
2. Step 1 — take a full day. The contract is the thing you will regret rushing.
3. Steps 2–6 — about a week. That gets you a running service.
4. **Start Step 8 in parallel the moment Step 1 is done.** It is the long pole and it
   needs no code from this repo.

Ping me at Milestone A and I'll review the contract and the API tests before you build on
top of them.
