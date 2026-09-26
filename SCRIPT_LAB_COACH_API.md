# Script Lab — Creative Coach API

The **Creative Coach** panel on Script Lab → Script Tester: the chat beside the score card
that explains a verdict, diagnoses what is weak, and rewrites copy against the brand's
Brand Brain and the ad's real performance.

> **Status: BUILT AND RUNNING LOCALLY — NOT YET DEPLOYED.**
> All five endpoints are implemented in this repo and answer real questions against live
> data. The Postman collection's examples are **captured from the running service**, and
> its 79 assertions pass against it (§9). The contract below is unchanged from the frozen
> version the backend and frontend were given, and nothing changes without a version bump
> and a note in §11.
>
> **Still to do before it is live:** deploy, plus the backend and frontend work in §2.

| You are… | Read |
|---|---|
| Anyone new to this | [1. Overview](#1-overview) |
| **Backend (Node)**: calling the AI Service and storing the thread | [2. For the backend team](#2-for-the-backend-team) |
| **Frontend**: building the panel | [3. The contract](#3-the-contract) and [6. For the frontend team](#6-for-the-frontend-team) |
| **AI Service**: building the coach | [4](#4-intents) → [8](#8-known-data-issues) |

---

## 1. Overview

```
Browser (Script Lab → Script Tester)
   │   no API key in the browser, ever
   ▼
Node backend  ──  X-API-Key + X-User-Id / X-User-Role / X-User-Permissions  ──▶
   ▼
AI Service (Python, this repo)
   │   reads  sl_script_lab_tests, meta_insights, mi_ad_creative_analyses,
   │          mi_competitor_ads, brands        (PostgreSQL, READ-ONLY)
   │   reads  brand_brains                     (MongoDB)
   │   returns ONE coach turn. It stores nothing.
   ▼
Node appends the turn to sl_script_lab_tests.coach_thread
```

Three rules that shape everything below:

1. **The AI Service never writes PostgreSQL.** It has no write path to `scrumdb` and will
   not grow one. The `coach_thread` column is written by Node, exactly as it is today.
2. **Python computes every number; the model only writes the sentences.** A figure the
   model produces that is not in the computed fact set is rejected, and the turn is
   repaired or replaced. This is why the coach cannot invent a CTR.
3. **The panel never sees an error.** Every failure is an HTTP 200 turn with
   `fallback: true` and usable text — the convention `POST /api/script-lab/test-script`
   already follows.

---

## 2. For the backend team

### 2.1 What you send

| Header | Value | Required |
|---|---|---|
| `X-API-Key` | The AI Service key. **Server-side only** | Yes |
| `X-User-Id` | `users.id` of the person chatting | Yes — threads are per user |
| `X-User-Role` | e.g. `superAdmin`, `user` | Yes |
| `X-User-Permissions` | Comma-separated `role.permissions` | Yes |

A request **without** the `X-User-*` headers is treated as a trusted service call and is not
permission-checked. Never let a browser reach this service directly.

**Permission:** `script_lab` opens the panel. Without it every call returns **403**, except
for `superAdmin`. Please add it to the role editor if it does not exist.

### 2.2 What you store

The AI Service returns one turn. **You append it to `sl_script_lab_tests.coach_thread`**,
alongside the user's own turn, in the shape already in that column:

```json
[
  {"role": "coach", "text": "…", "intent": "intro"},
  {"role": "user",  "text": "How do I improve the hook?"},
  {"role": "coach", "text": "…", "intent": "diagnose", "…": "the rest of the turn"}
]
```

The response is a **superset** of the three keys already stored, so existing threads stay
readable and the extra keys can be persisted or dropped as you prefer. Persisting them is
recommended: `evidence` and `facts_used` are what make a past answer auditable.

**You pass the thread back** on the next call. The AI Service is stateless and has no memory
between turns.

### 2.3 Endpoints

Base URL: local `http://127.0.0.1:3011` · production `https://api.scaleserum.com`

| Method + path | Use |
|---|---|
| `POST /api/script-lab/coach/chat` | One question → one coach turn |
| `GET /api/script-lab/coach/starters?test_id=` | The opening turn + the three chips. **No LLM call**, instant |
| `POST /api/script-lab/coach/feedback` | The thumbs on a coach bubble: `{request_id, rating: "up"\|"down", reason?}` |
| `GET /api/script-lab/coach/flagged?brand_id=&limit=` | Turns users marked down, newest first |
| `GET /api/script-lab/coach/usage?brand_id=&from=&to=` | Tokens, estimated spend, **health and alerts** over a date range |

### 2.4 Timeouts

Target p50 **< 3 s**, p95 **< 8 s**; the service caps its own upstream call at 25 s. Set the
reverse-proxy read timeout to **≥ 60 s** — it is shared with `test-script`, which legitimately
takes 15–25 s.

---

## 3. The contract

### 3.1 `POST /api/script-lab/coach/chat`

**Request**

```json
{
  "test_id": "05cc1f26-3527-48b4-99bc-c748963d44ab",
  "message": "How do I improve the hook?",
  "intent": null,
  "thread": [
    {"role": "coach", "text": "Tested \"…\" — score 5/100. …", "intent": "intro"}
  ],
  "session": {
    "user_id": "u_88",
    "brand_id": "bee79bff-d5f6-4220-a7d4-7a04bf173e59",
    "request_id": "c7f1e0c4-…"
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `test_id` | string, **required** | A row in `sl_script_lab_tests`. The service reads it itself |
| `message` | string, **required** | The user's question. Max **1000** chars |
| `intent` | string \| null | Send the intent when the user **clicked a chip** — it skips classification and is faster and cheaper. `null` for free text |
| `thread` | array | The existing `coach_thread`. Omit or `[]` for a new conversation. Only the last 12 turns are used |
| `session.user_id` | string | Also taken from `X-User-Id`; the header wins |
| `session.brand_id` | string | **Checked against the test's brand.** A mismatch is a 403 |
| `session.request_id` | string | Optional; echoed back for tracing. Generated if absent |

**Response — 200**

```json
{
  "role": "coach",
  "intent": "diagnose",
  "text": "Your hook scored 1/10, and it is the only thing worth fixing right now…",
  "facts_used": [
    {"key": "hook_score", "value": 1, "unit": "/10"},
    {"key": "overall_score", "value": 5, "unit": "/100"}
  ],
  "evidence": [],
  "suggested_rewrite": "How many hours did your team spend drafting repetitive legal agreements this week?",
  "follow_ups": [
    "Rewrite it for a colder audience",
    "Does this still fit the Original angle?"
  ],
  "confidence": "high",
  "brand_brain": {"tier": "A", "used": true, "missing": []},
  "grounded": true,
  "fallback": false,
  "usage": {"input_tokens": 2310, "output_tokens": 240, "total_tokens": 2550},
  "model": "gemini-2.5-flash",
  "prompt_version": "coach-v1",
  "request_id": "c7f1e0c4-…",
  "created_at": "2026-09-24T09:12:03Z"
}
```

| Field | Meaning |
|---|---|
| `role`, `text`, `intent` | The three keys already stored in `coach_thread` |
| `facts_used` | Every figure the answer is allowed to contain, computed in Python. If `text` contains a number that is not here, the turn was rejected before you saw it |
| `evidence` | Provenance for any claim drawn from outside the test — ad performance, a past winner, a competitor hook. Empty when the answer is grounded only in the score card |
| `suggested_rewrite` | Copy block for the UI's **Copy** button. `null` when the intent produced no rewrite |
| `follow_ups` | Two or three next questions. **Replace the starter chips with these** |
| `confidence` | `high` \| `medium` \| `low`. `low` means the answer rests on thin data and the UI should not present it as settled |
| `brand_brain.tier` | `A` complete · `B` partial · `C` absent. See §5 |
| `grounded` | `false` means validation failed twice and `text` is the safe fallback wording |
| `fallback` | `true` means no model answer was produced at all |
| `usage`, `model`, `prompt_version` | Cost accounting and reproducibility |

An `evidence` entry:

```json
{"type": "ad_performance", "ad_id": "120251070761150057", "metric": "ctr",
 "value": 0.0141, "window": "30d", "source": "meta_insights"}
```

`type` is one of `ad_performance`, `past_version`, `brand_creative`, `competitor_ad`,
`brand_brain`.

### 3.2 `GET /api/script-lab/coach/starters?test_id=…`

The opening turn and the chips, **computed in Python — no model call, no token cost**. Call
it when the panel opens on a fresh test.

```json
{
  "intro": {
    "role": "coach",
    "intent": "intro",
    "text": "Tested \"1411_PGLI_Dubai AI startup\" — score 5/100, rewrite required. The hook is the weakest part at 1/10. Ask me why, or how to push the score higher.",
    "fallback": false
  },
  "starters": [
    {"label": "Why is the Hook only 1/10?", "intent": "explain"},
    {"label": "What should I change first?", "intent": "prioritize"},
    {"label": "Will this scale?", "intent": "scale"}
  ],
  "brand_brain": {"tier": "A", "used": true, "missing": []},
  "capabilities": {"performance": true, "compare": false, "brand_fit": true}
}
```

`capabilities` tells the UI which chips are worth offering: `compare: false` means this is
the ad's first version, so there is nothing to compare against.

### 3.3 `GET /api/script-lab/coach/usage?brand_id=&from=&to=`

Tokens and estimated Gemini spend, priced at the rates in force when each turn ran — the
same rule the Sales Call Analyzer and AI Briefings bills follow. Everything is an
**estimate**; `notes` names every caveat that applies.

---

## 4. Intents

| Intent | The question it answers | Extra data it reads |
|---|---|---|
| `explain` | "Why did I get this score?" | — |
| `diagnose` | "What is wrong with it?" | — |
| `prioritize` | "What should I change first?" | — |
| `improve` | "How do I make this better?" | Brand creatives, competitor hooks |
| `rewrite` | "Rewrite the hook / CTA" | Brand Brain voice |
| `compare` | "Is this better than the last version?" | Past versions of the same ad |
| `performance` | "How is this ad actually doing?" | Ad-level `meta_insights` |
| `scale` | "Should I put more budget behind it?" | Ad performance + score |
| `brand_fit` | "Does this sound like us?" | Brand Brain |
| `out_of_scope` | Anything not about this script | — (one-line redirect) |

`intro` also appears in stored threads; it is produced by `/starters`, never by `/chat`.

---

## 5. Degradation — what the UI must handle

The coach is **never** unavailable. It degrades, and it says so. Every row below is a
specified behaviour, not an error.

| Situation | Response | What the UI shows |
|---|---|---|
| **No Brand Brain** for the brand | `brand_brain.tier: "C"`, `used: false` | Normal answer. The coach says once: *"No Brand Brain connected — I'm coaching on craft and performance. Connect it and I can judge brand fit too."* |
| **Partial Brand Brain** | `tier: "B"`, `missing: ["competitors","salesCycle"]` | Normal answer; the coach names what it is missing rather than guessing |
| **`brand_fit` asked at tier C** | `intent: "brand_fit"`, `confidence: "low"` | The coach explains it cannot judge brand fit and offers craft feedback instead |
| **Script never published** (no `source_ad_id`) | `capabilities.performance: false` | `performance`/`scale` answer with no numbers and a clear reason |
| **Ad live < 7 days** | `confidence: "low"`, evidence present | "Not enough data yet" plus what to watch |
| **First version of an ad** | `capabilities.compare: false` | The `compare` chip is hidden. A `compare` question still gets an answer: the coach says in one line that there is no earlier version, then compares the script against this brand's **best analysed creative** instead, which carries `evidence` of type `brand_creative` |
| **The test itself failed** (`ai_fallback = true`) | `confidence: "low"` | The coach **does not defend the neutral 50s** — it says the review did not complete and offers a re-run |
| **Validation failed twice** | `grounded: false` | Safe wording; no numbers |
| **The coach claimed something in words it had no data for** — "this ad is performing well" with no delivery data, "better than your last version" with no earlier version, "your brand voice is…" with no Brand Brain | `grounded: false`, `fallback_reason` one of `unbacked_performance`, `unbacked_comparison`, `unbacked_brand_claim` | The deterministic stand-in. A claim with no figure in it is still a claim, and it is rejected like an invented figure |
| **Model outage** | `fallback: true` | Heuristic answer from the score card. Still useful |
| **Off-topic question** | `intent: "out_of_scope"` | One-line redirect to the script |
| **Thread at 40 turns** | `409`, `detail: "thread_limit"` | Offer "Start a new conversation" |

---

## 6. For the frontend team

- Call `/starters` when a test is opened; render `intro` as the first bubble and `starters`
  as the three chips.
- Call `/chat` on send **or** on a chip click. On a chip, pass its `intent` — noticeably faster.
- After each answer, **replace the chips with `follow_ups`**.
- Render `suggested_rewrite` as a distinct block with a **Copy** button, not inside the bubble.
- Show `evidence` behind a "why?" affordance — it is what makes the coach credible.
- `confidence: "low"` should read visually softer. `fallback: true` may show a Retry.
- Disable the input until a test exists — the empty state in the prototype is correct.
- Keep the whole `coach_thread` and send it back; the service remembers nothing.

---

## 7. Errors

| Code | When |
|---|---|
| **401** | Missing or wrong `X-API-Key` |
| **403** | No `script_lab` permission, or `session.brand_id` does not match the test's brand |
| **404** | `test_id` not found |
| **409** | Thread limit reached (`detail: "thread_limit"`) |
| **422** | `message` empty or > 1000 chars, `test_id` not a UUID, unknown `intent` |

Anything else — model outage, DB timeout, validation failure — is a **200** with
`fallback: true`. The panel must never show a red error for those.

---

## 8. Known data issues

Found while freezing this contract, against live data on 24 Sep 2026. These sit **upstream
of the coach** and need owners.

1. **DI's Brand Brain contains Lawttorney's content.** Brand Brain `c6f11180…`
   (`brandName: "DI"`) describes *"project lawyers and legal professionals burdened by
   manual document drafting"* — that is LawTorney.ai's persona, not Director's Institute's.
   **Consequence: 20 of DI's 53 script reviews (38%) discuss the wrong brand.** DI is 82% of
   all Script Lab usage. The coach will inherit this and repeat it confidently.
   **This is the highest-value fix in Script Lab, and it is a data fix, not a code fix.**
2. **`ad_number` is not the ad number the UI shows.** In `sl_script_lab_tests`, `ad_number`
   is a small per-brand sequence (1, 2, 53…). The Meta ad id shown in the UI's **AD NUMBER**
   field is `source_ad_id`. Please confirm which one the UI writes where.
3. **Some `source_ad_id` values are junk** (`mi_named_0`, `202`, `31241551`). The coach
   validates the join before making any performance claim and degrades silently, but the
   write path should stop storing these.
4. **`version` is always 1, for every test in the system.** Nothing increments it, and each
   test also takes a fresh `ad_number`, so **not one of the 198 stored tests has an earlier
   version** — checked directly. "Scored against its history", which the Script Tester tip
   banner promises, has no history to read at all.
   The coach works around it (§5: it compares against the brand's best analysed creative
   instead), but that is a substitute, not the feature. **To make version history real, the
   write path must reuse `ad_number` and increment `version` when the same ad is retested.**
   That is a one-line decision on the backend and it unlocks the whole learning loop.
5. **Four `brand_id`s in `sl_script_lab_tests` (133 rows) do not exist in `brands`.** Assumed
   to be seed data; they have no brand row and no Brand Brain.
6. **JSON columns are camelCase, the API is snake_case.** `context_alignment` stores
   `brandVoiceFit`; improvements store `whyItMatters` / `suggestedRewrite`. The coach
   normalises on read. New writes should pick one convention and stay with it.
7. **68 of 196 stored tests (35%) have `ai_fallback = true`** — the AI review did not
   complete and a neutral 50/100 placeholder was stored. One in three users is therefore
   looking at a score that means nothing. The coach handles this (§5) by refusing to defend
   those numbers, but the failure rate itself needs investigating: it is a bigger quality
   problem than anything the coach can compensate for.
8. **No brand currently sits at Brand Brain tier B.** Both linked Brand Brains are complete
   (9 of 9 answers), so the partial-context path has no live example and must be tested
   against a synthesised fixture.

---

## 9. Postman

`postman/ScaleSerum-ScriptLabCoach.postman_collection.json` — 28 requests, each with a
response **captured from the running service**, and 79 assertions.

```bash
PORT=3011 python app.py            # in one terminal
newman run postman/ScaleSerum-ScriptLabCoach.postman_collection.json \
  --env-var baseUrl=http://127.0.0.1:3011 --env-var apiKey=$API_KEY --timeout-request 90000
```

Last run: **79/79 assertions passed, 28 requests, 0 failures**, average response 1.45 s
(max 3.9 s). Verified to fail — 65 assertions and exit 1 — when the service misbehaves, so
it is a gate and not a document.

The assertions are the contract in executable form: the three keys `coach_thread` already
stores, the grounding block (`facts_used`, `evidence`, `grounded`, `fallback`,
`request_id`, `brand_brain.tier`), that no placeholder text reaches the user, that the
refusal cases actually refuse, and that a rewrite request returns copy. A field that
quietly disappears fails the run.

Run the folder **in order**: requests 21–24 rate the turn created by 4–14 through the
`{{lastRequestId}}` variable. Requests 8–14 are the degraded paths; a coach that only
handles 4–7 is not finished.

The collection can still back a Postman **mock server** for frontend work before deploy —
the examples are now real responses rather than specimens, so the mock is more faithful
than it was.

---

## 10. Monitoring

`GET /api/script-lab/coach/usage` is the operational view as well as the bill. The health
counters travel **with** the cost on purpose: a month that got cheaper because half its
turns fell back to the stand-in is not a saving, and a report that hides the cause is a
misleading one.

```json
{
  "status": "ok | warn | alert",
  "alerts": [{"signal": "fallback_rate", "value": 0.07, "threshold": 0.02,
              "severity": "alert",
              "note": "turns produced no AI answer at all - users are reading the stand-in"}],
  "health": {"turns": 412, "fallback_rate": 0.01, "ungrounded_rate": 0.0,
             "latency_p50_ms": 2600, "latency_p95_ms": 7400,
             "thumbs_down_rate": 0.04, "model_routed_rate": 0.11},
  "feedback": {"up": 22, "down": 3, "rated": 25, "unrated": 387, "down_reasons": [...]},
  "fallback_reasons": {"foreign_brand": 2, "llm_unavailable": 1},
  "per_route": {"supplied": {...}, "matched": {...}, "model": {...}},
  "per_brand_brain_tier": {"A": {...}, "C": {...}}
}
```

| Signal | Fires above | Why it matters |
|---|---|---|
| `ungrounded_rate` | **0** | An answer was rejected for citing a figure nothing supports. The test suite gates this at 100%; production should not hold itself to a lower bar |
| `fallback_rate` | 2% | No AI answer at all — users are reading the deterministic stand-in while the page looks normal |
| `latency_p95_ms` | 12 s | Past this it stops feeling like a chat |
| `thumbs_down_rate` | 15% of **rated** turns | People rarely bother to complain; when they do it matters. Rated, not total — dividing by every turn would hide it |
| `model_routed_rate` | 40% | Most questions suddenly need the classifier: the phrase patterns have stopped fitting what people ask. A product signal before it is a cost one |

**Nothing fires below 20 turns.** A quiet day with one fallback is not a 100% fallback
rate, and an alert that cries wolf is an alert people mute. Every threshold is overridable
(`COACH_ALERT_UNGROUNDED`, `COACH_ALERT_FALLBACK`, `COACH_ALERT_P95_MS`,
`COACH_ALERT_THUMBS_DOWN`, `COACH_ALERT_MODEL_ROUTED`, `COACH_ALERT_MIN_TURNS`).

**`per_brand_brain_tier` is a product metric, not a technical one.** It says how much of
your coaching is running without a complete Brand Brain, which is the cheapest quality
improvement available to you — it needs onboarding, not engineering.

### What the frontend should send

Put 👍/👎 on each coach bubble and post the turn's `request_id` with it. The **reason** is
the valuable half: a rate tells you something is wrong, a reason tells you what. Those
turns become evaluation cases via `python -m evals.from_feedback`, which is how a real
complaint becomes a permanent test.

### When an alert fires

| Alert | First thing to check |
|---|---|
| `ungrounded_rate` | `fallback_reasons` — `unsupported_number` means the model is inventing figures; `foreign_brand` means a Brand Brain contains another company's content (see §8.1) |
| `fallback_rate` | `fallback_reasons`. `llm_unavailable` is a provider problem; a gate reason is a prompt problem |
| `latency_p95_ms` | Whether `model_routed_rate` rose at the same time — a classifier call on every turn adds a second |
| `thumbs_down_rate` | `GET /coach/flagged` and read what people actually wrote |

---

## 11. Versioning

`prompt_version` identifies the wording rules; this document's contract version is
**`coach-api-1`**. Additive fields may appear without a bump. A removed or re-typed field
requires a new version and a note here.
