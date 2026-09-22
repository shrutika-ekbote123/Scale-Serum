# AI Briefings — API guide

The **Briefings** page: a daily AI morning briefing with five tabs (**All**, **Sales Team**,
**Ads & Marketing**, **WhatsApp**, **Leads**), the Sales Team **Consolidated Score** card,
**Past Briefings**, the **Read more** modal, and the briefing card on the **Dashboard**.

| You are… | Read |
|---|---|
| Anyone new to this | [1. Overview](#1-overview) |
| **Backend (Node)**: calling the AI Service | [2. For the backend team](#2-for-the-backend-team) |
| **Frontend**: building the page | [3. For the frontend team](#3-for-the-frontend-team) |
| AI Service: changing numbers or rules | [4. How it works](#4-how-it-works-ai-service-team) |

---

## 1. Overview

```
Every morning (07:00 in each brand's timezone)
  briefing-worker ──reads──▶ scrumdb (read-only) + call analyses (Mongo)
        │   computes every number and Watch item in Python
        │   Gemini writes the sentences (numbers are checked)
        ▼
  MongoDB: 5 briefings per brand per day  (all, sales, ads, whatsapp, leads)
        ▲
  GET /api/ai-briefings/...  ← the page only READS. Instant, no AI call.
```

- **Briefings are generated ahead of time**, once per brand per day, for the brand's
  **yesterday**. Opening the page never waits on Gemini or scrumdb.
- **Python computes every number; Gemini only writes the words.** Any number in Gemini's
  text that isn't in the computed facts is rejected, and the fixed template wording is
  stored instead (`fallback: true`). The numbers are identical either way.
- **The All tab is built from the other four**, so its numbers always match theirs.
- **Past briefings never change.** Each day is stored as it was written.

What the numbers mean (settled 22 Sep 2026):

| Figure | Definition |
|---|---|
| Revenue | Real payments received (non-backfilled `touchpoint_events` payments), not what Meta/Google report |
| Blended ROAS | Payments received ÷ total ad spend, for the day |
| Campaign ROAS | **7-day Meta-reported** (Meta's own attribution). Payments carry no campaign, so real money can't be split by campaign. Always labelled as Meta-reported |
| Spend, leads, CPL | Ad platforms, **campaign level only** (the same spend is also stored per ad set and per ad) |
| Hot leads | The purchase-probability model's **High** band, among open leads from the last 7 days with no call or WhatsApp for 24h. Not `leads.score`, which is written at purchase time |
| Qualified | Leads marked **SQL on a sales call** (`sales_calls.disposition`) |
| Consolidated Score | Mean Sales Call Analyzer score (0–100) over the last **7 days**; trend is vs the 7 days before |
| Closures | A payment is credited to the **last rep who called that lead in the 30 days before it was paid**. Otherwise it's "unassigned" |
| Conversion | Credited closures ÷ calls, over 7 days |

---

## 2. For the backend team

### 2.1 Who calls what

```
 Browser ──▶ Node backend (your auth, knows the user)
               │  GET /api/ai-briefings/{brandId}/today?section=ads
               │  X-API-Key:           <AI service key>        ← server-side only
               │  X-User-Id:           <users.id>
               │  X-User-Role:         <users.role>            e.g. superAdmin | user
               │  X-User-Permissions:  <role.permissions, comma-separated>
               ▼
 AI Service (this repo)
```

**The browser must never call the AI Service directly**, because it would expose the
API key. Always forward the three `X-User-*` headers. They decide which tabs and rep
rows the user gets. A request **without** them is treated as a trusted service call
and sees everything.

### 2.2 Who sees what

Access comes from the role's **permissions**, not its name (role names are custom per
company):

| Permission | Gives |
|---|---|
| `briefings` | The page itself. Without it every call returns **403** (except for `superAdmin`) |
| `sales_calls` or `crm` | Sales Team tab |
| `meta_ads`, `google_ads`, `linkedin_ads` or `all_accounts` | Ads & Marketing tab |
| `whatsapp_business` | WhatsApp tab |
| `crm`, `lead_search`, `lead_journey` or `lead_analytics` | Leads tab |
| `users_roles` or **`briefings_team`** | **Team view:** every rep's row. Without it, a user sees the team numbers and **only their own rep row** |

- `superAdmin` sees everything.
- **All** is always shown. For a user without every tab, it's built from their tabs only.
- **Please add a `briefings_team` permission** to the role editor. A "Sales Head" who
  can't manage users still needs to see the whole team.

### 2.3 Endpoints

Base URL: local `http://127.0.0.1:3011` · production `https://api.scaleserum.com`

| Method + path | Use |
|---|---|
| `GET /api/ai-briefings/{brandId}/today?section=all\|sales\|ads\|whatsapp\|leads&date=YYYY-MM-DD` | One tab's card. `date` omitted = the brand's yesterday |
| `GET /api/ai-briefings/{brandId}/score?date=` | Consolidated Score card + per-rep table |
| `GET /api/ai-briefings/{brandId}/history?section=all&limit=10&before=` | Past Briefings, newest first. `limit` is in **days**; pass `next_before` for the next page |
| `GET /api/ai-briefings/briefing/{briefingId}` | One briefing in full (the Read more modal) |
| `GET /api/ai-briefings/{brandId}/latest` | Compact All briefing for the Dashboard header card |
| `POST /api/ai-briefings/{brandId}/generate?date=&days=1` | **Super admin only.** (Re)generate now; `days=30` backfills Past Briefings for a new brand. Returns **202** with queued runs |
| `GET /api/ai-briefings/{brandId}/runs?limit=20` | Generation runs and their status (for the admin action above) |

### 2.4 Response: `today` (real, DI, Ads tab, trimmed)

```json
{
  "briefing_id": "bee79bff-…:2026-09-21:ads",
  "date": "2026-09-21",
  "date_label": "Monday, 21 September 2026",
  "section": "ads",
  "section_label": "Ads & Marketing",
  "available": true, "reason": null, "message": null,
  "stale": false,
  "summary": {
    "label": "Ads & Marketing",
    "text": "Spend reached ₹2.4L across Meta (+5% on the day before), delivering a blended ROAS of 0.5× on ₹1.2L of payments and 519 leads at ₹464 each. Creative analysis flagged 0 to Scale and 81 to Pause.",
    "top_label": "Top",
    "top": "FG Cold Conv Webinar ICDP India 26thAugust2026 NewLp SS Paused (16.4× 7-day Meta-reported ROAS)",
    "watch_label": "Watch",
    "watch": "Review FG Cold Conv Webinar ICDP India 5thAug2026 OldLp Sunidhi Paused (0.3× Meta-reported ROAS on ₹2.2L over 7 days) …"
  },
  "watch": [
    {"type": "roas_below_threshold", "severity": "high",
     "entity": "FG Cold Conv Webinar ICDP India 5thAug2026 OldLp Sunidhi Paused",
     "text": "… returned 0.3× Meta-reported ROAS on ₹2.2L over 7 days (threshold 3.0×)."}
  ],
  "kpis": {
    "spend":        {"value": 240711.05, "display": "₹2.4L", "previous": 228721.35, "delta": "+5%"},
    "revenue":      {"value": 120598.0,  "display": "₹1.2L", "basis": "payments received"},
    "blended_roas": {"value": 0.501,     "display": "0.5×"}
  },
  "blocks": [
    {"key": "ad_performance", "title": "Ad performance", "bullets": ["Spend reached ₹2.4L …", "Blended ROAS was 0.5× …"]},
    {"key": "creatives", "title": "Creative analysis", "bullets": ["Out of 122 ads analysed: 0 Scale, 41 Watch, and 81 Pause."]}
  ],
  "score_card": null,
  "tabs": [
    {"section": "all", "label": "All sections", "available": true, "reason": null},
    {"section": "sales", "label": "Sales Team", "available": true, "reason": null},
    {"section": "ads", "label": "Ads & Marketing", "available": true, "reason": null},
    {"section": "whatsapp", "label": "WhatsApp", "available": true, "reason": null},
    {"section": "leads", "label": "Leads", "available": true, "reason": null}
  ],
  "data_as_of": {"meta": "2026-09-22", "google": "2026-09-18", "linkedin": "2026-08-19"},
  "generated_at": "2026-09-22T09:26:14+00:00",
  "fallback": false,
  "wording": {"source": "llm", "fallback_reason": null},
  "viewer": {"team_view": true, "sections": ["sales", "ads", "whatsapp", "leads"], "service": true}
}
```

`facts` (every computed number, raw and formatted) is also returned. It's useful for
extra UI, and it's what the AI was allowed to quote.

### 2.5 `score`

```json
{
  "date": "2026-09-17", "available": true, "stale": false, "team_view": true,
  "score_card": {
    "score": 50, "score_max": 100, "trend_points": null, "trend_display": null, "window_days": 7,
    "calls_yesterday": 3, "calls_window": 9,
    "closures": {"count": 0, "revenue": 0.0, "display": "0 · ₹0"},
    "unassigned_payments": {"count": 0, "revenue": 0.0},
    "conversion": 0.0, "conversion_display": "0%",
    "basis": {"score": "mean analysed call score over 7 days", "closures": "…", "conversion": "…"}
  },
  "reps": [
    {"rep": "Aaditya WDC", "user_id": "d946255a-…", "calls_yesterday": 0, "calls": 5, "analysed_calls": 4,
     "score": 51, "score_change": null, "closures": 0, "closures_revenue": "₹0", "sqls": 1,
     "missed_callbacks": 0,
     "good": "Stated call purpose early and framed value around cost savings",
     "watch": "Omitted permission check on a cold outbound call"}
  ]
}
```

### 2.6 Status codes

| Code | When |
|---|---|
| 200 | Always for reads, **including "nothing to show"**. Check `available` and `reason` |
| 202 | `generate` accepted (runs are queued) |
| 401 | Missing or wrong `X-API-Key` |
| 403 | User lacks `briefings`, lacks the requested tab, or isn't super admin for `generate` |
| 404 | Unknown `briefingId`, or unknown brand on `generate` |
| 422 | Bad `section`, bad date, brand id not a UUID, or `generate` for a day that isn't over |
| 503 | Feature or MongoDB not configured on this server |

`reason` values when `available: false`: `not_generated` (no briefing yet for this brand),
`whatsapp_not_connected`, `no_sales_calls`, `no_ad_accounts`, `no_leads`, `no_data`,
`section_error`.

### 2.7 Timing and caching

- **Reads** are Mongo lookups: typically 100–500 ms. Cache per brand + day + tab if you
  like; a day's briefing only changes if someone regenerates it.
- **Generation** takes about 15–40 s per brand-day (scrumdb queries, hot-lead scoring,
  5 Gemini calls). It's always in the background. Never call `generate` on page load.

### 2.8 Testing

Postman: `postman/ScaleSerum-AIBriefings.postman_collection.json`. Paste your API key
into `apiKey` and run it in order. Request 2 generates DI's yesterday for real, and the
rest read it back. It covers the viewer rules and every error case: 25 requests,
94 assertions, all passing on 22 Sep 2026.

---

## 3. For the frontend team

### 3.1 Page layout → endpoint

| UI piece | Call | Fields |
|---|---|---|
| Tabs | `today` (any tab) | `tabs[]`. Hide tabs the user doesn't have; grey out `available: false` with `reason` |
| Today's Briefing card | `today?section=…` | `date_label`, `section_label` (badge), `summary.label` (bold), `summary.text`, then `summary.top_label: summary.top` and `summary.watch_label: summary.watch` in colour |
| Consolidated Score card | `score` (or `today`'s `score_card`) | ring = `score`/`score_max`; rows: Calls (yesterday) `calls_yesterday`, Closures `closures.display`, Conversion `conversion_display`, Trend `trend_display` |
| Per-Rep table | `score` | `reps[]`: `rep`, `calls`, `score`, `good` (green ✓), `watch` (⚠) |
| Past Briefings | `history` | `items[]`: `short_label`, `section_label` chip, `summary`; "Read →" opens `briefing/{briefing_id}` |
| Read more modal | `briefing/{id}` | `section_label` badge + `summary.text`, then each `blocks[]` as a heading with bullets |
| Dashboard header card | `latest` | `summary.text`, `top_watch.text`, "View past briefings →" |

### 3.2 States

| Response | Show |
|---|---|
| `available: true`, `stale: false` | The briefing |
| `available: true`, `stale: true` | The briefing, with a note: "Latest briefing: {date_label}. Today's is being prepared." |
| `available: false`, `reason: not_generated` | "Your first briefing will be ready tomorrow morning." |
| `available: false`, other reason | `message` as-is, e.g. "WhatsApp is not connected for this brand yet." |
| `summary.top` or `summary.watch` is `""` | Hide that callout, including its label |

### 3.3 Small things

- The badge on the card should show the **tab's** `section_label` ("Ads & Marketing"). The
  prototype shows "All sections" on every tab.
- Show `data_as_of` when a source is behind, for example "LinkedIn data as of 19 Aug".
  It's also a Watch item.
- `kpis.*.display` and `delta` strings are already formatted (₹14.2L, 5.9×, ▲6). Don't
  reformat them.
- `fallback: true` means template wording. It's still accurate, so no warning is needed.

---

## 4. How it works (AI Service team)

### 4.1 Pipeline (`ai_briefings/`)

```
worker.py / POST generate
  └─ service.generate_day(brand, day)
       ├─ timezones.resolve(brands.timezone)     "IST – UTC+5:30 (…)", "Asia/Kolkata", "UTC"
       ├─ data.load_day(...)                      scrumdb, read-only, windows in brand-local time
       ├─ hot_leads.score(untouched)              purchase-probability model, one batch
       ├─ loaders.analyses_loader(...)            sales_call_analyses (Mongo) via sales_calls.analysis_id
       ├─ sections/{sales,ads,whatsapp,leads}.py  facts + Watch + template wording (pure functions)
       ├─ sections/overall.py                     All, from the four
       ├─ writer.write(...)                       Gemini; numbers checked; template fallback
       └─ store.save(...) × 5, finish_run(...)    ai_briefings, ai_briefing_runs
```

### 4.2 Configuration: `ai_briefings/briefing_config.json`

- Generation hour, retries and backfill limit.
- Thresholds: ROAS 3×, CPL spike +25%, overspend 125%, minimum campaign spend ₹2,000,
  7-day window, 30-day closure credit, hot-lead lookback.
- Watch-item sentence templates.
- `demo_data.sales_call_rep_names`: the July seed reps. Remove them once scrumdb is cleaned.

Environment: `BRIEFING_WORKER_INTERVAL_SECONDS` (300), `BRIEFING_STUCK_MINUTES` (30),
`BRIEFING_LLM_TIMEOUT_MS` (30000), `BRIEFING_LLM_TEMPERATURE` (0.2).

### 4.3 Deploying

- `ecosystem.config.js` has a new pm2 app, **`briefing-worker`**. `deploy.sh` restarts it
  on every deploy, but **it doesn't start it the first time**. Once on the server:
  `pm2 start ecosystem.config.js --only briefing-worker && pm2 save`.
- To fill Past Briefings for existing brands, call `POST /generate?days=30` once per
  brand as a super admin.

### 4.4 Known data limitations (22 Sep 2026)

- **Payments carry no campaign.** No payment has a `utm_campaign`, so ROAS by campaign
  can only be Meta's own figure.
- **Some brands have the wrong timezone.** DI's `brands.timezone` is `UTC`, so its "yesterday"
  runs 00:00–24:00 UTC, not IST. It should be fixed in the brand settings, not here.
- **Few sales calls are logged.** DI logged none in the week to 21 Sep, so its Sales tab
  shows a "no calls" Watch item and all 33 payments that week are unassigned.
- **Google `leads` is always 0,** so Google contributes spend and clicks only. **LinkedIn data
  stops on 19 Aug.**
- **`leads.funnel` is empty on most recent leads,** so "Best funnel" often doesn't appear.
- **Creative verdicts are overwritten daily,** so a backfilled day only shows the verdicts
  that happen to still be dated that day.
