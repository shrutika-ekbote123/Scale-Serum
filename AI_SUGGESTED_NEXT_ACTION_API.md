# AI Suggested Next Action — API guide

One card on the **Lead Journey** page: what the sales team should do next with a lead, how
urgent it is, and why.

| You are… | Read |
|---|---|
| Anyone new to this | [1. Overview](#1-overview) |
| **Backend (Node)**: calling the AI Service | [2. For the backend team](#2-for-the-backend-team) |
| **Frontend**: building the card | [3. For the frontend team](#3-for-the-frontend-team) |
| AI Service: changing how decisions are made | [4. How it works](#4-how-it-works-ai-service-team) |

---

## 1. Overview

```
AI Suggested Next Action
├── Urgency          → how soon to act               (HIGH / MEDIUM / LOW URGENCY)
├── Recommendation   → what the sales team should do
└── Reason / Context → why, citing the lead's own activity
```

- The card is built from the lead's **touchpoints** (forms, payments), **sales calls** and
  **WhatsApp** state, plus the brand's own sales history.
- **Fixed rules decide the action and the urgency.** The same data always gives the same
  answer. **AI (Gemini) only writes the sentences**, and it can't change the decision.
- It knows the difference between a lead who paid **₹299 for a webinar ticket** and one who
  paid **₹30,000 for the programme**. Payments are placed on the brand's price ladder
  (entry → core → premium), so "Converted" alone never decides the advice.

Examples from real Lawtorney leads (11 Sep 2026):

| Lead | Card |
|---|---|
| 5 webinar forms in 15 days, never contacted, not paid | **HIGH** · Call within 24 hours to find out what is stopping them |
| Paid ₹299 (webinar ticket) today | **HIGH** · Call within 24 hours to pitch the ₹30,000 programme |
| Paid ₹30,000 three days ago | **MEDIUM** · Call within 3 days to complete onboarding |
| Sales-qualified on a call 4 days ago, no payment | **MEDIUM** · Call within 3 days to address the objections from that call |
| One form 110 days ago, nothing since | **LOW** · Add to a WhatsApp re-engagement campaign |

---

## 2. For the backend team

### 2.1 Who calls what

```
 Browser (app.scaleserum.com)
      │  GET /leads/:id/next-action          ← your Node route, your auth
      ▼
 Node backend
      │  GET /api/ai-suggested-next-action/:leadId
      │  Header X-API-Key: <AI service key>  ← server-side only
      ▼
 AI Service (this repo)  ──reads──▶ Postgres (read-only), MongoDB
```

**The browser must never call the AI Service directly.** The `X-API-Key` would then be
visible to anyone. The Node backend calls it and passes the card on.

### 2.2 Base URLs

| Environment | AI Service base URL |
|---|---|
| Local | `http://127.0.0.1:3002` |
| Production | `https://api.scaleserum.com` |

### 2.3 Get the card

```
GET /api/ai-suggested-next-action/{lead_id}
Header: X-API-Key: <API_KEY>
```

| Query param | Type | Default | Use |
|---|---|---|---|
| `refresh` | bool | `false` | Regenerate the wording (for example, a "Regenerate" button). The decision is recomputed on every call anyway. |
| `brand_brain_id` | string | from the lead's brand | Rarely needed. Overrides the Brand Brain used for tone. |

```bash
curl -H "X-API-Key: $API_KEY" \
  https://api.scaleserum.com/api/ai-suggested-next-action/b510926c-84d5-4cd5-964d-55fd8b80857e
```

### 2.4 Response (real, from the lead with 5 forms)

```json
{
  "lead_id": "b510926c-84d5-4cd5-964d-55fd8b80857e",
  "brand_id": "bee79bff-d5f6-4220-a7d4-7a04bf173e59",
  "brand_name": "Lawtorney",
  "ai_suggested_next_action": {
    "urgency": {
      "level": "high",
      "label": "HIGH URGENCY",
      "act_within_hours": 24,
      "due_by": "2026-09-12T12:04+00:00",
      "basis": "Last active 8 hours ago and half of this brand's buyers pay within 4 days of their first form."
    },
    "recommendation": {
      "action_type": "call_probe_repeat",
      "channel": "phone",
      "title": "Call within 24 hours to identify barriers to enrolling",
      "text": "Call the lead to find out what is stopping them from signing up after multiple webinar form submissions."
    },
    "reason": {
      "headline": "Submitted 5 webinar registration forms across recent weeks",
      "text": "The lead submitted 5 forms between 27 Aug 2026 and 11 Sep 2026 and has not been contacted by sales.",
      "evidence": [
        {"type": "form_submit", "count": 5, "first": "27 Aug 2026", "last": "11 Sep 2026",
         "sources": ["ICDP Webinar Registration New", "ICDP Webinar Registration Testing"]},
        {"type": "sales_call", "count": 0, "last_date": null, "last_disposition": null},
        {"type": "whatsapp", "conversations": 0, "last_direction": null, "last_message_date": null}
      ]
    }
  },
  "lead_state": {"stage": "new_uncontacted", "converted": false, "contacted": false,
                 "highest_tier": null, "payment_count": 0, "total_paid": null},
  "tiers": {"source": "inferred", "items": ["… see 2.7 …"]},
  "conversion_window": {"source": "brand_history", "converters": 259, "p50_hours": 98.0, "p75_hours": 365.9},
  "confidence": "medium",
  "data_flags": ["multiple_phone_numbers", "brand_currency_invalid"],
  "wording": {"source": "cache", "fallback_reason": null, "call_analysis_used": false},
  "versions": {"framework_version": "na_v1", "prompt_version": "na_p1", "llm_model": "gemini-flash-latest"},
  "generated_at": "2026-09-11T12:04:43+00:00",
  "cached": true,
  "availability": {"available": true, "reason": null, "message": null},
  "fallback": false
}
```

### 2.5 Field reference

Fields marked **UI** are what the frontend needs; pass at least those through.

| Field | Type | Null? | UI | Meaning |
|---|---|---|---|---|
| `lead_id` | string (UUID) | no | | Echo of the request |
| `brand_id`, `brand_name` | string | when unavailable | | The lead's brand |
| `ai_suggested_next_action` | object | **when unavailable** | ✔ | The card. Everything below it is filled when present. |
| `…urgency.level` | `"high"` \| `"medium"` \| `"low"` | no | ✔ | Drives the colour |
| `…urgency.label` | string | no | ✔ | `HIGH URGENCY` etc. Show as-is. |
| `…urgency.act_within_hours` | int | no | | 24 / 72 / 168 |
| `…urgency.due_by` | ISO 8601 with offset | no | ✔ | When to act by. For `callback` it is the promised callback time. |
| `…urgency.basis` | string | no | optional | One sentence explaining the timing |
| `…recommendation.action_type` | enum (see 2.6) | no | ✔ | Stable code. Use it for icons and logic, never parse the text. |
| `…recommendation.channel` | `"phone"` \| `"whatsapp"` \| `"none"` | no | ✔ | Which button to show |
| `…recommendation.title` | string ≤ 90 chars | no | ✔ | Bold line |
| `…recommendation.text` | string ≤ 400 chars | no | ✔ | Body |
| `…reason.headline` | string ≤ 90 chars | no | ✔ | Short reason |
| `…reason.text` | string ≤ 400 chars | no | ✔ | Grey context line |
| `…reason.evidence` | array | no | optional | Facts behind the reason (see 3.6) |
| `lead_state` | object | when unavailable | | `stage`, `converted`, `contacted`, `highest_tier`, `payment_count`, `total_paid` |
| `confidence` | `"high"` \| `"medium"` \| `"low"` | when unavailable | ✔ | How solid the underlying data is |
| `data_flags` | string[] | no (may be empty) | optional | Data problems (see 3.7) |
| `wording.source` | `"llm"` \| `"cache"` \| `"template"` | when unavailable | | Where the sentences came from |
| `fallback` | bool | no | | `true` means fixed template wording (still a valid card) |
| `cached` | bool | no | | Wording reused from cache |
| `availability.available` | bool | no | ✔ | **Check this first** |
| `availability.reason` | string | when available | ✔ | See 2.8 |
| `generated_at` | ISO 8601 | when feature is off | | Server time of the decision |
| `tiers`, `conversion_window`, `versions` | object | when unavailable | | Diagnostics, not for the UI |

### 2.6 `action_type` values

| `action_type` | `channel` | Meaning | Default urgency |
|---|---|---|---|
| `call_now` | phone | New lead, not contacted yet, still in the buying window | high |
| `call_probe_repeat` | phone | Registered 3+ times without buying. Find out why. | high |
| `callback` | phone | The lead asked for a callback | high |
| `reply_whatsapp` | whatsapp | The lead's WhatsApp message is unanswered | high |
| `pitch_core_offer` | phone | Bought the cheap entry offer. Pitch the main one. | high → medium → low over 30 days |
| `follow_up_objection` | phone | Qualified on a call but hasn't paid | high / medium |
| `retry_contact` | whatsapp | The last call went unanswered | medium |
| `follow_up` | whatsapp | Interest cooling, or contacted with no outcome | medium |
| `onboard` | phone | Bought the main offer within the last 7 days | medium |
| `nurture_next_tier` | phone | Existing customer. Retention or upgrade. | low |
| `re_engage` | whatsapp | Gone quiet past the buying window | low |
| `disqualify` | none | Lost or irrelevant. No action. | low |
| `insufficient_data` | none | Nothing reliable to go on | low |

New values may be added. **Treat an unknown `action_type` as a generic card.** Don't fail on it.

### 2.7 Status codes and errors

The card endpoint **always returns HTTP 200** when the key is valid. Problems are reported
in the body:

| HTTP | When | What to do |
|---|---|---|
| 200 + `availability.available: true` | Normal | Pass the card on |
| 200 + `availability.reason: "lead_not_found"` | Bad or unknown lead id | Hide the card. Don't retry. |
| 200 + `availability.reason: "database_unavailable"` | Postgres unreachable | Retry once after about 2 s, then show the error state |
| 200 + `availability.reason: "feature_unavailable"` | Feature failed to start on the server | Hide the card and alert the AI Service team |
| 401 | Missing or wrong `X-API-Key` | Configuration bug on the Node side |
| 5xx / network error | AI Service down | Retry once, then show the error state |

### 2.8 Timeouts, retries, caching

| | |
|---|---|
| Typical latency | **0.1–0.5 s** when the wording is cached; **2–6 s** when the AI writes new wording |
| Worst case | About **30 s** (AI call capped at 15 s, plus one repair retry) |
| Recommended Node timeout | **35 s**. Load the card **asynchronously** so it never blocks the Lead Journey page. |
| Caching in Node | **Don't cache it for longer than a page view.** The AI Service already caches the wording, and urgency changes as time passes (a lead goes from high to medium on its own). |
| When to call | When the Lead Journey page opens, and again after an action that changes the lead (a call is logged, a payment arrives). |

### 2.9 Product tier override (brand settings)

A brand's payments are placed into price tiers automatically. A brand can correct this
(for example, "₹299 is the webinar ticket, ₹30,000 is the programme").

```
GET    /api/ai-suggested-next-action/tiers/{brand_id}
PUT    /api/ai-suggested-next-action/tiers/{brand_id}
DELETE /api/ai-suggested-next-action/tiers/{brand_id}
```

**GET** (real, Lawtorney):

```json
{
  "brand_id": "bee79bff-d5f6-4220-a7d4-7a04bf173e59",
  "active_source": "inferred",
  "override": null,
  "override_updated_at": null,
  "inferred": [
    {"name": "Tier 1", "kind": "entry", "min_amount": 100.0, "max_amount": 499.0,
     "payment_count": 927, "typical_amount": 299.0, "product_label": null,
     "product_codes": ["CrossroadstoBoardroom", "riseandfallofboards399", "AIFE149"], "source": "inferred"},
    {"name": "Tier 2", "kind": "core", "min_amount": 6999.0, "max_amount": 150000.0,
     "payment_count": 265, "typical_amount": 30000.0, "product_label": null,
     "product_codes": ["TheComprehensiveNonExecut", "CNEDP10000", "2year10999"], "source": "inferred"}
  ]
}
```

`active_source` is `override`, `inferred`, or `none` (the brand has no payments yet).

**PUT** body. This replaces the whole override:

```json
{
  "tiers": [
    {"kind": "entry", "min_amount": 0,    "max_amount": 999,    "product_label": "Crossroads to Boardroom webinar"},
    {"kind": "core",  "min_amount": 1000, "max_amount": 200000, "product_label": "Comprehensive Non-Executive Director programme"}
  ]
}
```

| Rule | Error |
|---|---|
| `kind` is `entry`, `core` or `premium` | 422 |
| `0 ≤ min_amount ≤ max_amount`, ranges don't overlap, at most 10 tiers | 422 (`detail` is a readable message you can show) |
| `brand_id` is a UUID | 422 |
| MongoDB not configured on the server | 503 |

**DELETE** returns `{"brand_id": "...", "deleted": true|false}`, and the brand goes back to
inferred tiers. Cards pick up a change on their next request. Nothing needs to be flushed.

### 2.10 Testing

Postman collection: [postman/ScaleSerum-AISuggestedNextAction.postman_collection.json](postman/ScaleSerum-AISuggestedNextAction.postman_collection.json).
It has 13 requests covering auth, both screenshot leads, the cache, the not-found paths
and the full override round trip. Set `apiKey` in the collection variables, then run it
in order.

---

## 3. For the frontend team

You receive the fields marked **UI** in 2.5 from the Node backend. The card sits in the
right-hand column of the Lead Journey page, under **Funnel**.

### 3.1 Card states — mockups

**A. High urgency** (`level: "high"`)

```
┌──────────────────────────────────────────────────────┐  ← red border
│ AI Suggested Next Action              [HIGH URGENCY] │  ← red title + red pill
│                                                      │
│ Recommendation: Call within 24 hours to identify     │  ← bold: recommendation.title
│ barriers to enrolling                                │
│ Call the lead to find out what is stopping them      │  ← recommendation.text
│ from signing up after multiple webinar submissions.  │
│                                                      │
│ The lead submitted 5 forms between 27 Aug 2026 and   │  ← grey: reason.text
│ 11 Sep 2026 and has not been contacted by sales.     │
│                                                      │
│ ⏱ Act by 12 Sep, 5:34 pm · in 22 h        [ 📞 Call ]│  ← due_by + channel button
│ Why? ▸                                               │  ← optional evidence expander
└──────────────────────────────────────────────────────┘
```

**B. Medium urgency** (`level: "medium"`). Same layout with amber accents.

```
┌──────────────────────────────────────────────────────┐  ← amber border
│ AI Suggested Next Action            [MEDIUM URGENCY] │
│                                                      │
│ Recommendation: Call within 3 days to complete       │
│ onboarding                                           │
│ Welcome the lead to the Comprehensive Non-Executive  │
│ programme and make sure onboarding is complete.      │
│                                                      │
│ The lead paid ₹30,000 on 10 Sep 2026.                │
│                                                      │
│ ⏱ Act by 14 Sep, 5:34 pm · in 3 days      [ 📞 Call ]│
└──────────────────────────────────────────────────────┘
```

**C. Low urgency** (`level: "low"`). Blue accents, as in the current prototype.

```
┌──────────────────────────────────────────────────────┐  ← blue border
│ AI Suggested Next Action               [LOW URGENCY] │
│                                                      │
│ Recommendation: Add to a WhatsApp re-engagement      │
│ campaign                                             │
│ Send nurture content about Crossroads to Boardroom   │
│ instead of calling.                                  │
│                                                      │
│ The lead submitted one form on 24 May 2026 and has   │
│ been inactive for 110 days.                          │
│                                                      │
│ ⏱ Act by 18 Sep · in 7 days       [ 💬 Open WhatsApp ]│
└──────────────────────────────────────────────────────┘
```

**D. No action** (`action_type: "disqualify"` or `"insufficient_data"`, `channel: "none"`).
Muted grey, no button, no deadline.

```
┌──────────────────────────────────────────────────────┐  ← grey border
│ AI Suggested Next Action               [LOW URGENCY] │  ← grey pill
│                                                      │
│ Recommendation: No action needed                     │
│ Deprioritise this lead unless they get back in touch.│
│                                                      │
│ The last call on 11 Sep 2026 was marked irrelevant.  │
└──────────────────────────────────────────────────────┘
```

**E. Loading.** The request can take up to about 6 s when the AI writes new wording.

```
┌──────────────────────────────────────────────────────┐
│ AI Suggested Next Action                  [░░░░░░░░] │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                 │  ← shimmer skeleton
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░                         │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░            │
│ Analysing this lead's activity…                      │
└──────────────────────────────────────────────────────┘
```

**F. Error** (`database_unavailable`, network error, or timeout)

```
┌──────────────────────────────────────────────────────┐
│ AI Suggested Next Action                             │
│                                                      │
│ Couldn't load a suggestion right now.    [ Retry ]   │
└──────────────────────────────────────────────────────┘
```

**G. Low confidence** (`confidence: "low"`). Render the normal card for its urgency and add
one line at the bottom:

```
│ ⓘ Based on limited data for this lead.               │
```

**H. Not found / feature off** (`lead_not_found`, `feature_unavailable`). **Hide the card
completely.** Don't show an error.

### 3.2 Which state to show

```
request pending                              → E  Loading
request failed / timed out                   → F  Error
availability.available == false
    reason == "database_unavailable"         → F  Error
    anything else                            → H  Hide
action_type in (disqualify, insufficient_data) → D  No action
otherwise, by urgency.level                  → A / B / C
    + confidence == "low"                    → add the G line
```

`fallback: true` needs **no special UI**. The wording is plainer, but it's still a
correct card.

### 3.3 Colours

| `level` | Border / title / pill | Pill text |
|---|---|---|
| `high` | red, e.g. `#EF4444` (pill background at ~15% opacity) | `urgency.label` |
| `medium` | amber, e.g. `#F59E0B` | `urgency.label` |
| `low` | blue, e.g. `#3B82F6` (the current prototype) | `urgency.label` |
| no-action (state D) | neutral grey | `urgency.label` |

Use the design system's red/amber/blue tokens if they exist. The hex values are only
guidance.

### 3.4 Text

| Element | Field | Notes |
|---|---|---|
| "Recommendation:" + bold line | `recommendation.title` | Always present, ≤ 90 chars |
| Body | `recommendation.text` | ≤ 400 chars. Wrap, don't truncate. |
| Grey context line | `reason.text` | Use `reason.headline` instead on a compact card |
| Tooltip on the pill (optional) | `urgency.basis` | Explains the timing |

All strings are plain text. **Don't render them as HTML.**

### 3.5 "Act by" and the channel button

- **`due_by`** is ISO 8601 with a timezone offset. Show it in the **viewer's local time**:
  `Act by 12 Sep, 5:34 pm`, plus a relative hint (`in 22 h`, `in 3 days`). If it's in the
  past, show `Overdue` in the urgency colour. Hide the line for state D.
- **Button by `channel`:**

| `channel` | Button | Action |
|---|---|---|
| `phone` | 📞 **Call** | Open the existing call / log-call flow for this lead |
| `whatsapp` | 💬 **Open WhatsApp** | Open this lead's conversation in WhatsApp Business |
| `none` | no button | |

This API returns **no phone numbers or emails**. Take contact details from the CRM data
the page already has.

- An optional **Regenerate** link can call the Node route with `refresh=true`. The action
  and urgency won't change; only the wording is rewritten.

### 3.6 "Why?" expander (optional)

`reason.evidence` is a list of facts. Suggested one-line rendering per `type`:

| `type` | Render as |
|---|---|
| `form_submit` | `📝 {count} form(s) · {first} → {last}` |
| `payment` | `💳 {count} payment(s) · latest {latest.amount} on {latest.date}` |
| `sales_call` | `📞 {count} call(s)` + ` · last {last_date} ({last_disposition})` when `count > 0` |
| `whatsapp` | `💬 {conversations} conversation(s)` + ` · last message {last_message_date}` when present |
| anything else | `{type}: {count}` |

The `sales_call` and `whatsapp` lines appear even with a zero count, on purpose: "nobody
has contacted them yet" is part of the reason.

### 3.7 Data warnings (optional)

If you surface `data_flags`, map them to text. Ignore unknown flags.

| Flag | Text |
|---|---|
| `multiple_phone_numbers` | This lead has more than one phone number on record |
| `synthetic_touchpoints_ignored` | Some test activity on this lead was ignored |
| `default_conversion_window` | Timing is an estimate: this brand has little sales history |
| `brand_brain_missing` | Complete the Brand Brain for better suggestions |
| `brand_currency_invalid` | Check this brand's currency setting |

### 3.8 Brand settings: product tiers (optional screen)

If brands should correct their tiers in the UI, the backend exposes 2.9. Suggested screen:

```
┌ Product tiers ───────────────────────────────────────────────────┐
│ Used by AI Suggested Next Action to tell a ticket from a sale.   │
│ Currently: Detected automatically                    [ Edit ]    │
│                                                                  │
│  Tier      Price range            Product                Sales   │
│  Entry     ₹100 – ₹499            Crossroads to Boardroom  927   │
│  Core      ₹6,999 – ₹1,50,000     Comprehensive Non-Exec…  265   │
└──────────────────────────────────────────────────────────────────┘

Edit mode: a row per tier with [Kind ▾] [Min ₹] [Max ₹] [Product name], + Add tier,
[ Save ] → PUT, [ Reset to automatic ] → DELETE. Show the 422 `detail` under the form.
```

---

## 4. How it works (AI Service team)

Code: [ai_suggested_next_action/](ai_suggested_next_action/) · route: `app.py` · prompt:
`prompts.py` (`NEXT_ACTION_SYSTEM_INSTRUCTION`) · rulebook:
[next_action_framework.json](ai_suggested_next_action/next_action_framework.json).

### 4.1 Pipeline

1. **Load** (`data.py`, Postgres read-only) the lead, all touchpoints, sales calls, the
   latest WhatsApp conversation, and the count of distinct phone numbers. Email and phone
   values are never selected.
2. **Tiers** (`tiers.py`): the brand's override, or inferred from brand payments. Amounts
   are split into bands wherever the next amount is more than 3× the previous one, and
   thin bands are merged, so one outlier never becomes a tier. Cached for 15 min.
3. **Buying window**: this brand's first-form-to-first-payment p50/p75. If fewer than 20
   leads have paid, the Brand Brain `salesCycle` answer is used, and failing that, the
   default (96 h / 360 h). Cached for 15 min.
4. **Decide** (`rules.py`). Pure and deterministic, first match wins:
   1. no usable activity → `insufficient_data`
   2. lost, or last call irrelevant/non-SQL with nothing since → `disqualify`
   3. inbound WhatsApp unanswered for ≤ 72 h → `reply_whatsapp`
   4. last call disposition `callback` → `callback`
   5. bought core/premium → `onboard` (≤ 7 d) / `nurture_next_tier`
   6. bought entry only → `pitch_core_offer` (or `follow_up_objection` if qualified on a later call)
   7. called, not paid → `follow_up_objection` (SQL/MQL) / `retry_contact` / `follow_up`
   8. never contacted → `call_now` / `call_probe_repeat` (≤ p50) · `follow_up` (≤ p75) · `re_engage`
5. **Word** (`writer.py`, Gemini). The model receives the decision, the facts, the
   evidence, the brand voice and language, and the latest analysed call's objections and
   buying signals. Output is **rejected** (one repair retry, then the template) when a
   field is missing or too long, when a number isn't present in the facts, or when it
   uses a gendered pronoun.
6. **Cache** (`store.py`, MongoDB `ai_suggested_next_actions`, keyed by lead) the wording,
   under a fingerprint of the decision, the facts and the versions. The same fingerprint
   means Gemini isn't called. Template wording isn't cached, so the next request tries
   the model again.

### 4.2 Deliberately not used

- `leads.score`, `temperature`, `status` (except `lost`) and `touchpoint_count`. These
  are rewritten at payment time.
- Touchpoints whose payload has `backfill`, which are seeded test rows. They are ignored
  and reported as `synthetic_touchpoints_ignored`.
- The Brand Brain's ideal-customer and journey text, in the wording. When it's wrong
  (Lawtorney's describes a legal-drafting SaaS) the model repeats it to reps as if it
  were known about the lead. The Brand Brain still sets the tone and the buying-window
  fallback.

### 4.3 Configuration

| env | default | |
|---|---|---|
| `NA_LLM_TIMEOUT_MS` | `15000` | Cap per wording call |
| `NA_LLM_TEMPERATURE` | `0.2` | |
| `GEMINI_MODEL`, `DB_*`, `MONGODB_URI`, `API_KEY` | — | Shared with the rest of the service |

MongoDB collections: `ai_suggested_next_actions`, `ai_suggested_next_action_tiers`. Both
are keyed by `_id`, so there are no indexes to create.

Changing a threshold or template in the rulebook: edit the JSON and **bump
`framework_version`**. That changes the fingerprint, so every cached wording regenerates.
Changing the prompt: bump `PROMPT_VERSION` in `writer.py`.

`/health` reports `ai_suggested_next_action.available` and `.storage`.

### 4.4 Known data limitations (11 Sep 2026)

- **Webinar attendance isn't recorded**, so `pitch_core_offer` can't tell whether an entry
  buyer attended.
- Only 8 of 64 WhatsApp conversations carry a `lead_id`, and none match a lead by phone
  number. Most leads show zero WhatsApp activity.
- `sales_calls.callback_at` has never been set, so callbacks default to "within 24 hours".
- Lawtorney's `brands.timezone` is `UTC` and `brands.currency` is `1234567`. Dates follow
  the brand setting, and the currency comes from the payments.
- Cashfree truncates product codes (`TheComprehensiveNonExecut`). A tier override with a
  `product_label` fixes the name on every card.
