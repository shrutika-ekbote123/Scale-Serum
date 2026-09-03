# Purchase Probability API

**Score one lead:** how likely they are to buy, what they have actually done since they
arrived, and how well they match the brand that owns them.

| | |
|---|---|
| **Model** | `purchase_probability` |
| **Status** | `baseline_mvp` |
| **Feature version** | `V1` (frozen) |
| **Signal version** | `heuristic_v1` |
| **Last updated** | 1 September 2026 |

---

## Table of contents

1. [Purpose — what this API is for](#1-purpose--what-this-api-is-for)
2. [Base URL, endpoint and HTTP method](#2-base-url-endpoint-and-http-method)
3. [Request (payload)](#3-request-payload)
4. [Three numbers, three jobs](#4-three-numbers-three-jobs)
5. [Response fields](#5-response-fields)
6. [Example response](#6-example-response)
7. [When a lead cannot be scored](#7-when-a-lead-cannot-be-scored)
8. [Integration notes](#8-integration-notes)

---

## 1. Purpose — what this API is for

A sales team cannot call every lead. This endpoint tells them **who to call first**, and
shows its working so they can trust the order.

It answers three questions about a single lead, from three different kinds of evidence:

- **How likely is this person to buy?**
  A frozen logistic-regression model, calibrated on out-of-fold predictions, reads the lead's
  original form submission.
- **What have they done since?**
  Click actions, high-intent page visits, return sessions, and how long ago the last of it
  happened.
- **Do they look like this brand's customer?**
  The brand's own onboarding answers — its ideal customer, the channels it buys, its language
  and its sales cycle — read from the Brand Brain document in MongoDB.

The three are kept deliberately separate in the response. **The measured probability is never
moved by the other two**; they produce their own ranking number that is honest about being a
prior rather than a measurement. [Section 4](#4-three-numbers-three-jobs) is the part to read
before building any UI on this.

### Data sources

| Store | Used for | Access |
|---|---|---|
| PostgreSQL (`scrumdb`) | The lead, its touchpoints, its on-site tracking events | **Read-only** |
| MongoDB (`brand_brains`) | The brand's onboarding answers | **Read-only** |

This service never writes to either.

---

## 2. Base URL, endpoint and HTTP method

### HTTP method

```
GET
```

### Base URL

| Environment | Base URL | Notes |
|---|---|---|
| Local development | `http://localhost:3001` | Port comes from `PORT` in `.env` (currently `3001`). |
| Production | `https://<your-api-host>` | Same host as every other `/api/*` route in this service. Set by your deployment. |

### Endpoint

```
/api/purchase-probability/{lead_id}
```

### Full URL

```
GET http://localhost:3001/api/purchase-probability/a376f7eb-70ac-4e98-a735-fc7ef2bcd9fc
```

### At a glance

| | |
|---|---|
| **Method** | `GET` |
| **Auth** | `X-API-Key: <API_KEY>` header — required on every `/api/*` route (`/health` is open) |
| **Content-Type** | Not required (no request body) |
| **Returns** | `application/json` |
| **Status codes** | Always **200**, including for leads that cannot be scored. **401** for a missing or wrong API key — the only non-200 this endpoint produces. |

---

## 3. Request (payload)

> **There is no request body.** This is a `GET` — everything it needs travels in the path,
> the query string, and one header.

### Path parameter

| Name | Type | Required | Description |
|---|---|---|---|
| `lead_id` | `uuid` | **yes** | The lead to score — `leads.id` in PostgreSQL. |

### Query parameter

| Name | Type | Required | Description |
|---|---|---|---|
| `brand_brain_id` | `string` | no | **Normally omit this.** The endpoint resolves the Brand Brain itself: `leads.brand_id` → `brands.brand_brain_id` → `brand_brains._id`. Pass it only to override that resolution — for testing, or for a brand whose record is not linked yet. |

### Headers

| Header | Required | Description |
|---|---|---|
| `X-API-Key` | **yes** | Your `API_KEY`. A missing or wrong key returns **401**. |

### Request examples

**curl — standard call (brand context resolved automatically)**

```bash
curl -H "X-API-Key: $API_KEY" \
  http://localhost:3001/api/purchase-probability/a376f7eb-70ac-4e98-a735-fc7ef2bcd9fc
```

**curl — with an explicit Brand Brain override**

```bash
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:3001/api/purchase-probability/a376f7eb-70ac-4e98-a735-fc7ef2bcd9fc?brand_brain_id=4624afab35ce40dba54582a9f0a88488"
```

**JavaScript**

```js
const res = await fetch(
  `${API_BASE}/api/purchase-probability/${leadId}`,
  { headers: { 'X-API-Key': API_KEY } }
);
const lead = await res.json();

lead.purchase_probability;   // 2.68  -> display as "2.68%"
lead.lead_priority.score;    // 99    -> sort your list on this
```

**Python**

```python
import requests

r = requests.get(
    f"{API_BASE}/api/purchase-probability/{lead_id}",
    headers={"X-API-Key": API_KEY},
    timeout=30,
)
lead = r.json()
```

---

## 4. Three numbers, three jobs

The response carries three quantities that are easy to confuse and **must not be used
interchangeably**. The values below come from one real lead, scored live.

| Field | Example | What it is | Use it for |
|---|---|---|---|
| `purchase_probability` | `2.68` | **Measured.** The calibrated model output as a percentage. Base rate is 1.09%, so real values sit roughly between 0.3% and 4%. Never rescaled, never moved by the layers. | Quoting a probability |
| `percentile` / `decile` / `priority` | `99` / `10` / `High` | **Measured.** Position against the frozen out-of-fold reference population of 8,445 leads. The model's demonstrated strength is ranking (lift@10% ≈ 2.10×). | The base model's own ranking |
| `lead_priority.score` | `99` | **Prior-adjusted.** The ranking after click behaviour and brand fit are applied. Reports `calibrated: false` about itself. | **Sorting the lead list** |

### And one number that is not a model output at all

| Field | Example | What it is | Use it for |
|---|---|---|---|
| `lead_score` | `25` | **CRM state.** The lead score the CRM maintains on the `leads` row, out of 100. It is written and mutated by the CRM at payment time, which is exactly why the model never reads it. | Showing the CRM's own badge |

⚠️ **`lead_score` is not a percentage and must never be rendered as one.** A lead card that
shows "25%" next to "Lead score 25/100" has substituted the CRM score for the prediction.
The two are unrelated: this lead's calibrated probability is `1.4`, i.e. 1.4%. Removing that
substitution is the reason this endpoint exists.

### Why they are kept apart

The engagement and brand-fit weights are **hand-set priors, not trained coefficients**.
Folding an unfitted prior into a calibrated probability destroys the calibration — it turns a
measured 2.68% into a number that means nothing.

So `purchase_probability` stays frozen and `lead_priority` carries the adjustment. **Passing a
Brand Brain changes the ranking, never the probability.**

### Provenance markers

Throughout the response, every factor carries a `basis` and a `status`:

| Marker | Meaning |
|---|---|
| `basis: "calibrated_model"` | Comes from the calibrated model |
| `basis: "heuristic"` | A bounded prior adjustment |
| `status: "observed"` | The signal was found and read |
| `status: "no_data"` | Looked for, not found — **contributes exactly `0.0`** |
| `status: "not_applicable"` | The factor does not apply to this brand |

**Absent signal is never a penalty.** A lead you know nothing about must not rank below a lead
you know something mildly bad about.

---

## 5. Response fields

Every key below is present on **every** response, including fallbacks. Branch on
`availability.available` and each layer's `available` — never on whether a key exists.

### Calibrated model output

| Field | Type | Description |
|---|---|---|
| `lead_id` | `string` | The lead that was scored. |
| `purchase_probability` | `float \| null` | Calibrated probability as a percentage. `2.68` means 2.68%. |
| `purchase_probability_percent` | `int \| null` | The same value rounded to a whole number. Reads 1% or 2% for most leads — prefer the ranking fields in UI. |
| `probability` | `float \| null` | The raw 0–1 form of the same number. |
| `percentile` | `int \| null` | 0–100 against the out-of-fold reference population. |
| `decile` | `int \| null` | 1–10. |
| `priority` | `string` | `High` (decile 10) · `Medium` (8–9) · `Low` (1–7) · `Unavailable`. |
| `top_factors` | `array` | **The lead-card list.** Plain-language factors from all three layers — base model, touchpoint history, brand fit — ranked by absolute contribution. Every row carries `affects`; see below. |
| `why` | `object \| null` | The header for the factor panel: the base rate the score starts from, the result, and the caveat about summing rows. `null` when the lead is unscorable. |
| `model_factors` | `array` | The calibrated explanation on its own — exactly the `f_*` factors that sum, in log-odds, to `probability`. **Use this, not `top_factors`, for any arithmetic.** |
| `model_features` | `object \| null` | Exactly what the model consumed: the 7 V1 features, their values, and the form submission they came from. |
| `model` | `object` | Model name, version, status, feature version, description. |
| `scored_at` | `string` | ISO timestamp. One clock is used for every time-based factor in the response. |

### Lead history (display only)

| Field | Type | Description |
|---|---|---|
| `touchpoints` | `array` | The lead's real, unfiltered history in chronological order — **including payments**. Shown to users; never a model input. |
| `touchpoint_count` | `int` | Length of that array. |

### CRM lead card — `lead_summary`

The lead-detail card: the CRM's own score and temperature, what the lead has actually paid,
and what it is worth. Read from the `leads` row and this lead's payment touchpoints — **not
from the model**. It therefore stays populated on a lead the model cannot score: when
`availability.available` is `false`, the probability goes null and this block does not.

Every column here is written or mutated by the CRM at payment time, so it leaks the outcome.
It is safe to **show** and is never **learned from** — it is read by a separate query from the
model's own, so a card column cannot reach the feature builder by accident.

| Field | Type | Description |
|---|---|---|
| `lead_summary.available` | `bool` | `false` only when the lead row could not be read at all. |
| `lead_summary.reason` / `.message` | `string \| null` | Why, when it could not. |
| `lead_summary.lead_score` | `int \| null` | The CRM's engagement-quality score. **Not a probability.** |
| `lead_summary.lead_score_max` | `int` | `100`. The denominator, so the UI never assumes one. |
| `lead_summary.temperature` | `string \| null` | `cold` · `warm` · `hot`. The CRM's badge. |
| `lead_summary.status` / `.stage` / `.source` | `string \| null` | CRM pipeline state and acquisition source. |
| `lead_summary.created_at` | `string \| null` | ISO timestamp the lead row was written. |
| `lead_summary.touchpoint_count` | `int \| null` | The CRM's own counter (`leads.touchpoint_count`). May legitimately differ from the top-level `touchpoint_count`, which is the length of the displayed history. |
| `lead_summary.total_revenue` | `float \| null` | Money **actually received** from this lead. `0.0` means nothing yet; `null` means not known. Render both as a dash if you like, but do not collapse them. |
| `lead_summary.currency` | `string \| null` | Currency of the payments, e.g. `INR`. |
| `lead_summary.payment_count` | `int` | Number of payment touchpoints. `0` → "No payments yet." |
| `lead_summary.converted` | `bool \| null` | Whether the lead has converted. |
| `lead_summary.converted_at` | `string \| null` | ISO timestamp of conversion. |
| `lead_summary.days_to_convert` | `int \| null` | `leads.time_to_convert_days`. **`null` when the lead has not converted — do not render that as `0`.** Zero days is a same-day purchase, which is a completely different fact. |
| `lead_summary.basis` | `string` | Provenance sentence, safe to display. |

#### `lead_summary.lifetime_value`

What the lead is worth, in money. Three numbers, because a value tile can honestly mean any
of them — pick one and label it:

| Field | Type | Description |
|---|---|---|
| `amount` | `float \| null` | **The headline.** Recorded revenue once the lead has paid, otherwise the probability-weighted estimate. Mirrored as the top-level `lifetime_value`. |
| `estimated` | `bool` | `false` when `amount` is money actually received, `true` when it is a forecast. |
| `expected_amount` | `float \| null` | Always the probability-weighted estimate, so a converted lead can still be compared against an open one. |
| `potential_amount` | `float \| null` | The order value undiscounted — what the lead would be worth **if** it converted. Independent of the model, so it survives an unscorable lead. |
| `potential_basis` | `string \| null` | How `potential_amount` was derived. |
| `available` | `bool` | Whether `amount` has a number. |
| `reason` | `string \| null` | `probability_unavailable` · `no_paid_orders_for_brand`, or the scoring reason. |
| `basis` | `string \| null` | Plain-English derivation of `amount`. |
| `probability_used` | `float \| null` | The 0–1 probability that went into the estimate. |
| `order_value_used` | `float \| null` | The order value that went into it. |
| `order_value_basis` | `string \| null` | `median paid order for this brand`. |
| `order_count` | `int \| null` | How many paid orders that median is drawn from. |
| `average_order_value` | `float \| null` | The brand mean, reported but **not used** — order values are long-tailed and one outsized order drags the mean away from what a typical lead is worth. |
| `median_order_value` | `float \| null` | The brand median, which is what the estimate uses. |

The estimate is `probability × median paid order for the brand`. It is **null, never a
substitute number**, when the lead has not paid and there is either no calibrated probability
or no brand order history to price against. Brand order statistics are cached in-process for
15 minutes (`PP_ORDER_STATS_TTL_SECONDS`).

### Card fields mirrored at the top level

For convenience, the five scalars a lead card puts in its header tiles are repeated at the
top level. `lead_summary` remains authoritative and carries the provenance.

| Field | Type | Mirrors |
|---|---|---|
| `lead_score` | `int \| null` | `lead_summary.lead_score` |
| `temperature` | `string \| null` | `lead_summary.temperature` |
| `total_revenue` | `float \| null` | `lead_summary.total_revenue` |
| `days_to_convert` | `int \| null` | `lead_summary.days_to_convert` |
| `lifetime_value` | `float \| null` | `lead_summary.lifetime_value.amount` |

### Engagement layer — clicks and timeline

| Field | Type | Description |
|---|---|---|
| `engagement.available` | `bool` | Whether any admissible behaviour was found. |
| `engagement.reason` | `string \| null` | `no_behavioural_data` · `only_form_submission` · `base_model_unavailable`. |
| `engagement.message` | `string \| null` | Human-readable explanation, safe to display. |
| `engagement.observed` | `object \| null` | See sub-table below. |
| `engagement.timeline` | `array` | Chronological admissible events. **Payments are excluded outright** — a payment is the outcome being predicted. |
| `engagement.channel` | `string \| null` | Normalised acquisition channel (`google`, `meta`, `youtube`, `linkedin`, …). |
| `engagement.window` | `object` | `{ start, end, end_reason }`. `end_reason` is `now` or `first_payment_write`. |
| `engagement.sources` | `object` | Row counts per source: `tracking_events`, `touchpoint_events`. |
| `engagement.recency_half_life_hours` | `object` | `{ hours, derived_from, sales_cycle_days }`. `derived_from` is `brand_sales_cycle` or `default`. |
| `engagement.factors` | `array` | Seven factors, each with a contribution and a `status`. |
| `engagement.total_raw` | `float` | Sum of all factor contributions, before clamping. |
| `engagement.total_applied` | `float` | Log-odds actually applied, after clamping to `bounds`. |
| `engagement.clamped` | `bool` | Whether the clamp was hit. |
| `engagement.bounds` | `array` | `[-0.75, 1.75]` — the layer's hard limits. |

**`engagement.observed` fields**

| Field | Type | Description |
|---|---|---|
| `clicks` | `int` | Click actions recorded. |
| `page_views` | `int` | Page views recorded. |
| `form_submits` | `int` | Form submissions. |
| `high_intent_hits` | `int` | Hits on pricing / demo / checkout / booking pages. |
| `sessions` | `int` | Distinct on-site sessions. |
| `events_total` | `int` | Total admissible events. |
| `active_days` | `int` | Distinct days with activity. |
| `returning` | `bool` | Came back after the first day. |
| `ad_click_arrival` | `bool` | Arrived carrying `fbclid` / `gclid` / `li_fat_id`. |
| `first_seen_at` | `string` | ISO timestamp of first activity. |
| `last_seen_at` | `string` | ISO timestamp of last activity. |
| `recency_hours` | `float` | Hours since last activity. |
| `span_hours` | `float` | Hours from first to last activity. |
| `hours_to_first_activity` | `float` | Hours from lead creation to first activity. |
| `events_per_active_day` | `float` | Activity density. |

**The seven engagement factors**

| Factor | Fires when |
|---|---|
| `e_ad_click_arrival` | The lead arrived through a paid ad click |
| `e_click_volume` | Click actions are recorded |
| `e_high_intent_pages` | High-intent pages were visited |
| `e_page_view_volume` | Pages were viewed |
| `e_session_return` | The lead came back after the first visit |
| `e_recency` | **The only engagement factor that can go negative** — once we have seen a lead act, silence since then is genuine evidence rather than missing data |
| `e_engagement_velocity` | Activity per active day |

### Brand Brain layer — brand fit

| Field | Type | Description |
|---|---|---|
| `brand_brain.available` | `bool` | Whether a Brand Brain was loaded and applied. |
| `brand_brain.reason` | `string \| null` | `no_brand_brain` · `base_model_unavailable` · `signal_config_unavailable`. |
| `brand_brain.message` | `string \| null` | Human-readable explanation, safe to display. |
| `brand_brain.brand_id` | `string \| null` | The brand that owns this lead. |
| `brand_brain.brand_name` | `string \| null` | That brand's name — shown even when it has no Brand Brain, so the UI can explain the gap. |
| `brand_brain.resolved_brand_brain_id` | `string \| null` | Which document was used. `null` means the brand has not completed onboarding. |
| `brand_brain.brand_brain_store` | `string` | `configured` or `not_configured` — whether MongoDB is wired up at all. |
| `brand_brain.profile` | `object \| null` | What was parsed from the Brand Brain. See sub-table below. |
| `brand_brain.factors` | `array` | Five factors. |
| `brand_brain.total_raw` / `total_applied` / `clamped` / `bounds` | mixed | As per the engagement layer. Bounds are `[-0.75, 1.25]`. |

**`brand_brain.profile` fields**

| Field | Type | Description |
|---|---|---|
| `brand_name` | `string \| null` | Brand name from the Brand Brain document. |
| `business_model` | `string \| null` | `b2b`, `b2c`, or `null` if not stated. |
| `business_type_raw` | `string \| null` | The brand's own words. |
| `industry` | `string \| null` | Industry from onboarding. |
| `target_seniority` | `string \| null` | `founder_c_level` · `vp_director` · `manager` · `individual_contributor` · `other`. |
| `target_seniority_matches` | `array` | All seniority levels detected in the brand's text. |
| `declared_channels` | `array` | Normalised channels the brand says it buys. |
| `language` / `language_prefix` | `string \| null` | e.g. `"English only"` → `"en"`. |
| `sales_cycle_raw` | `string \| null` | The brand's own words, e.g. `"Same day"`. |
| `sales_cycle_days` | `float \| null` | Parsed to days. `null` when the brand answered non-committally — the factor is then skipped rather than applied against an invented number. |
| `marketing_goal` | `string \| null` | The brand's stated goal. |

**The five brand-fit factors**

| Factor | Reads | Fires when |
|---|---|---|
| `b_icp_seniority_fit` | `idealCustomer`, `audienceShort`, `journey` | The brand described a target seniority |
| `b_channel_fit` | `trafficChannels`, `context.channels` | Both brand channels and the lead's channel are known |
| `b_business_type_email_fit` | `businessType`, `industry` | The brand is B2B (for B2C a personal email carries no signal) |
| `b_language_locale_fit` | `language` | Brand language and lead locale are both known |
| `b_sales_cycle_timing` | `salesCycle` | The brand declared a sales cycle |

**Sales-cycle parsing.** Free text is parsed, not guessed:

| Brand answer | Parsed as |
|---|---|
| `"Same day"` | 1 day |
| `"1-4 weeks"` | 28 days (the pessimistic end of a quoted range) |
| `"1-3 months"` | 90 days |
| `"Less than a week"` | 7 days |
| `"6+ months"` | 180 days |
| `"varies"`, `"not sure"`, blank | `null` — factor skipped |

### Ranking output

| Field | Type | Description |
|---|---|---|
| `lead_priority.score` | `int \| null` | 0–100. **The field to sort on.** |
| `lead_priority.probability` | `float \| null` | Adjusted probability, 0–1. Not calibrated. |
| `lead_priority.probability_percent` | `float \| null` | The same, as a percentage. |
| `lead_priority.percentile` | `int \| null` | Ranking percentile after adjustment. |
| `lead_priority.decile` | `int \| null` | 1–10 after adjustment. |
| `lead_priority.priority` | `string` | Band after adjustment. |
| `lead_priority.calibrated` | `bool` | Always `false`. **Do not present this number as a probability.** |
| `lead_priority.layers_applied` | `array` | Which layers actually contributed, e.g. `["engagement","brand_brain"]`. |
| `lead_priority.basis` | `string` | Plain-English description of how the number was produced. |
| `lead_priority.log_odds` | `object` | `base` + `engagement` + `brand_brain` = `total`. The full audit trail. |
| `lead_priority.percentile_basis` | `string` | Caveat on how to read the adjusted percentile. |
| `ranking_factors` | `array` | All factors from all three layers merged and ranked by absolute contribution, each tagged with `layer` and `basis`. |
| `signal_version` | `string \| null` | Version of the signal weights, e.g. `heuristic_v1`. |

### Availability

| Field | Type | Description |
|---|---|---|
| `availability` | `object` | `{ available, reason, message }`. `message` is written for humans and safe to display. |
| `fallback` | `bool` | `true` when the lead could not be scored. |

### Factor object

Every entry in `top_factors`, `ranking_factors` and each layer's `factors` has the same shape:

```json
{
  "feature":      "b_icp_seniority_fit",
  "label":        "Fit with the brand's ideal customer",
  "layer":        "brand_brain",
  "basis":        "heuristic",
  "status":       "observed",
  "value":        { "lead": "founder_c_level", "brand_target": "founder_c_level" },
  "contribution": 0.45,
  "direction":    "positive",
  "explanation":  "Contributed positively: the lead's seniority meets the brand's stated target."
}
```

| Field | Description |
|---|---|
| `feature` | Stable identifier. `f_*` = base model, `e_*` = engagement, `b_*` = brand fit. |
| `label` | **The line to print.** `"<what was observed> - <why that matters>"`, e.g. `"Founder or C-level contact - senior decision-maker"`. It answers "why did this move the score" on its own, with no tooltip. It used to name the model input (`"Lead locale"`), which told a salesperson nothing; that is a regression a test now blocks. |
| `layer` | `model` · `engagement` · `brand_brain`. |
| `basis` | `calibrated_model` · `heuristic`. |
| `status` | `observed` · `no_data` · `not_applicable`. |
| `value` | The observed value. Shape varies by factor. |
| `contribution` | Signed, in log-odds. |
| `direction` | `positive` · `negative` · `neutral`. |
| `explanation` | Deliberately **non-causal** — "contributed to", never "caused". Safe to show to a salesperson verbatim. |

### Plain-language fields (rendering `top_factors`)

Entries in `top_factors` carry these extra fields. They are presentation only —
no number here is new, and `contribution` is passed through untouched.

**The frontend needs no change to benefit from this** — `label` already carries
the full reason, so an existing card that renders `label` becomes readable on
deploy. The fields below are extras for a richer panel.

| Field | Description |
|---|---|
| `title` | The observed half of `label` on its own: `"Founder or C-level contact"`. Use when you want the reason on a second line. |
| `detail` | One sentence a salesperson can read. For touchpoint factors it leads with what was measured — `"Last active 22 hours ago."` |
| `impact` | `Strong positive` · `Moderate positive` · `Slight positive` and the negative equivalents. Use this, not the raw log-odds, in the UI. |
| `odds_multiplier` | `exp(contribution)`. `1.65` reads as "about 1.65× the odds". The same quantity as `contribution`, in a unit people understand. |
| `source` | `Form submission` · `Touchpoint history` · `Brand profile`. Where the evidence came from. |
| `affects` | **Do not drop this.** `purchase_probability` or `lead_priority` — see below. |

#### ⚠️ `affects` — why the rows do not add up

`top_factors` mixes two kinds of row:

- **`affects: "purchase_probability"`** — base-model factors. These sum, in
  log-odds, to the calibrated score on screen.
- **`affects: "lead_priority"`** — engagement and brand-fit factors. These are
  bounded, **unfitted** priors. They move the ranking score, **not** the
  probability.

A user who sums every row expecting the displayed percentage will be wrong. Show
`source` or `affects` on the row, or group the two kinds under separate
subheadings. `why.reading_note` carries this caveat in prose.

Factors that contributed nothing are dropped rather than shown as zero — an
absent signal is reported by its layer's `available: false`, not as a `0` row.

### The `why` block

```json
{
  "starting_point": { "label": "Base rate - all leads", "percent": 1.09,
                      "detail": "1.09% of leads purchase within 21 days." },
  "result":         { "label": "Purchase probability", "percent": 2.68 },
  "reading_note":   "Factors marked purchase_probability sum, in log-odds, ..."
}
```

`starting_point.percent` is the **measured** conversion rate of the training
snapshot (160 / 14,692), not a design constant. Do not replace it with a
round number to make the panel add up more neatly.

### Editing the copy

All wording lives in `purchase_probability_model/factor_language.json`. Changing
a title or a sentence needs no code change and cannot alter a score. The one
rule: sentences describe **associations**, never causes — a logistic regression
on observational data cannot support "caused", "because" or "drives", and
`tests/test_purchase_probability_explanations.py` fails the build if one appears.

---

## 6. Example response

A real lead, scored live, with both layers active. Long arrays are truncated where marked;
nothing else is edited.

```json
{
  "lead_id": "a376f7eb-70ac-4e98-a735-fc7ef2bcd9fc",

  // ---- calibrated model output - never moved by the layers below ----
  "purchase_probability": 2.68,
  "purchase_probability_percent": 3,
  "probability": 0.02682699,
  "percentile": 99,
  "decile": 10,
  "priority": "High",
  "top_factors": [
    { "feature": "f_locale", "label": "Lead locale", "value": "en",
      "direction": "positive", "contribution": 0.965592,
      "explanation": "Contributed positively to this prediction." },
    { "feature": "f_email_class", "label": "Email address type", "value": "other_freemail",
      "direction": "positive", "contribution": 0.668886,
      "explanation": "Contributed positively to this prediction." }
    /* ... 3 more ... */
  ],
  "model_features": {
    "feature_version": "V1",
    "values": {
      "f_seniority": "founder_c_level",
      "f_email_class": "other_freemail",
      "f_locale": "en",
      "f_company_len": 13.0,
      "f_company_is_placeholder": 0.0,
      "f_hour_sin": 0.0,
      "f_hour_cos": -1.0
    },
    "source": "first admissible form_submit (written at lead creation, not backdated)",
    "form_occurred_at": "2026-05-24T12:52:54.709000+00:00",
    "form_created_at":  "2026-05-24T12:53:00.241000+00:00"
  },
  "model": {
    "name": "purchase_probability",
    "version": "baseline_mvp",
    "status": "baseline_mvp",
    "feature_version": "V1",
    "description": "Baseline/model MVP trained on currently available real data..."
  },

  // ---- display history: real, unfiltered, includes payments ----
  "touchpoint_count": 98,
  "touchpoints": [
    { "type": "form_submit",
      "occurred_at": "2026-05-24T12:52:54.709000+00:00",
      "created_at":  "2026-05-24T12:53:00.241000+00:00",
      "channel": "website", "provider": "wix",
      "source": "ICDP Webinar Registration Testing" }
    /* ... 97 more ... */
  ],

  "availability": { "available": true, "reason": null, "message": null },
  "fallback": false,
  "scored_at": "2026-09-01T11:48:11.035513+00:00",

  // ---- ENGAGEMENT: what the lead actually did ----
  "engagement": {
    "available": true,
    "reason": null,
    "message": null,
    "observed": {
      "clicks": 0,
      "page_views": 0,
      "form_submits": 98,
      "high_intent_hits": 0,
      "sessions": 0,
      "events_total": 98,
      "active_days": 58,
      "returning": true,
      "ad_click_arrival": false,
      "first_seen_at": "2026-05-24T12:52:54.709000+00:00",
      "last_seen_at":  "2026-08-28T15:49:37.487000+00:00",
      "recency_hours": 91.976,
      "span_hours": 2306.945,
      "hours_to_first_activity": -0.002,
      "events_per_active_day": 1.69
    },
    "channel": null,
    "channel_raw": "website",
    "window": {
      "start": "2026-02-23T12:53:00.229000+00:00",
      "end":   "2026-09-01T11:48:11.035513+00:00",
      "end_reason": "now"
    },
    "sources": { "tracking_events": 0, "touchpoint_events": 98 },

    // the brand's sales cycle sets how fast interest goes stale
    "recency_half_life_hours": {
      "hours": 6.0,
      "derived_from": "brand_sales_cycle",
      "sales_cycle_days": 1.0
    },
    "factors": [
      { "feature": "e_session_return", "status": "observed", "contribution": 0.30,
        "explanation": "Contributed positively: the lead came back after the first visit." },
      { "feature": "e_recency", "status": "observed", "contribution": -0.60,
        "explanation": "Contributed negatively: the lead has been quiet for longer than this brand's pace allows." },
      { "feature": "e_click_volume", "status": "no_data", "contribution": 0.0,
        "explanation": "No recorded click activity on record, so this factor contributed nothing." }
      /* ... 4 more ... */
    ],
    "total_raw": -0.177004,
    "total_applied": -0.177004,
    "clamped": false,
    "bounds": [-0.75, 1.75]
  },

  // ---- BRAND BRAIN: what the brand said it wants ----
  "brand_brain": {
    "available": true,
    "reason": null,
    "message": null,
    "brand_id": "bee79bff-d5f6-4220-a7d4-7a04bf173e59",
    "brand_name": "Lawtorney",
    "resolved_brand_brain_id": "4624afab35ce40dba54582a9f0a88488",
    "brand_brain_store": "configured",
    "profile": {
      "brand_name": "DI",
      "business_model": "b2b",
      "business_type_raw": "Education / Coaching / Consulting",
      "industry": "Education / Coaching",
      "target_seniority": "founder_c_level",
      "target_seniority_matches": ["founder_c_level"],
      "declared_channels": ["google", "meta", "youtube"],
      "language": "English only",
      "language_prefix": "en",
      "sales_cycle_raw": "Same day",
      "sales_cycle_days": 1.0,
      "marketing_goal": "Lead quality (higher intent)"
    },
    "factors": [
      { "feature": "b_icp_seniority_fit",       "status": "observed", "contribution":  0.45 },
      { "feature": "b_language_locale_fit",     "status": "observed", "contribution":  0.20 },
      { "feature": "b_business_type_email_fit", "status": "observed", "contribution": -0.125 },
      { "feature": "b_sales_cycle_timing",      "status": "observed", "contribution": -0.3491 },
      { "feature": "b_channel_fit",             "status": "no_data",  "contribution":  0.0 }
    ],
    "total_raw": 0.175911,
    "total_applied": 0.175911,
    "clamped": false,
    "bounds": [-0.75, 1.25]
  },

  // ---- the number to sort on ----
  "lead_priority": {
    "probability": 0.02679847,
    "probability_percent": 2.68,
    "score": 99,
    "percentile": 99,
    "decile": 10,
    "priority": "High",
    "calibrated": false,
    "layers_applied": ["engagement", "brand_brain"],
    "basis": "calibrated base model, adjusted by bounded heuristic engagement and brand-fit priors",
    "log_odds": {
      "base":        -3.591153,
      "engagement":  -0.177004,
      "brand_brain":  0.175911,
      "total":       -3.592246
    },
    "percentile_basis": "base-model out-of-fold reference distribution; the mapping is monotone, so ranking holds, but the absolute percentile is approximate once adjusted"
  },

  "ranking_factors": [ /* all three layers merged, ranked by |contribution| */ ],
  "signal_version": "heuristic_v1",

  "lead_summary": {
    "available": true,
    "reason": null,
    "message": null,
    "lead_score": 25,
    "lead_score_max": 100,
    "temperature": "cold",
    "status": "new",
    "stage": null,
    "source": "ICDP Webinar Registration New",
    "created_at": "2026-09-01T11:05:45.942000+00:00",
    "touchpoint_count": 1,
    "total_revenue": 0.0,
    "currency": "INR",
    "payment_count": 0,
    "converted": false,
    "converted_at": null,
    "days_to_convert": null,
    "lifetime_value": {
      "amount": 4.18,
      "currency": "INR",
      "estimated": true,
      "available": true,
      "reason": null,
      "basis": "calibrated purchase probability x the brand's median paid order value - an expected value, not recorded revenue",
      "expected_amount": 4.18,
      "potential_amount": 299.0,
      "potential_basis": "the brand's median paid order - what this lead would be worth if they converted, not discounted by how likely that is",
      "probability_used": 0.0139649,
      "order_value_used": 299.0,
      "order_value_basis": "median paid order for this brand",
      "order_count": 1137,
      "average_order_value": 49441.2,
      "median_order_value": 299.0
    },
    "basis": "leads table plus this lead's payment touchpoints - CRM display state, mutated at payment time and therefore never a model input"
  },

  "lead_score": 25,
  "temperature": "cold",
  "total_revenue": 0.0,
  "days_to_convert": null,
  "lifetime_value": 4.18
}
```

> **The two 25s are not the same 25.** `lead_score: 25` is the CRM's badge; the probability
> for this same lead is `1.4`, i.e. 1.4%. Any card showing "25%" as the purchase probability
> is reading the wrong field.

> **Note on this example.** It was captured using an explicit `brand_brain_id` override, which
> is why `brand_name` (`"Lawtorney"` — the brand that owns the lead) differs from
> `profile.brand_name` (`"DI"` — the Brand Brain that was forced on). Without the override the
> two always agree.

### How to read the arithmetic

```
base log-odds        -3.591153   (calibrated model)
+ engagement         -0.177004   (quiet for 92h against a 6h half-life)
+ brand fit          +0.175911   (ICP match, offset by lead age vs a same-day cycle)
= total              -3.592246
sigmoid(total)     =  0.0267985  ->  lead_priority.probability
```

---

## 7. When a lead cannot be scored

The endpoint returns **HTTP 200 even then**, with `fallback: true`,
`availability.available: false` and a stable reason.

**`purchase_probability` is `null` — never `0`.** Rendering an unscorable lead as 0% tells a
salesperson something false.

### Scoring reasons — `availability.reason`

| Reason | Meaning |
|---|---|
| `lead_not_found` | No lead with that id. |
| `no_admissible_form_payload` | No form submission was recorded at lead-creation time, so the model has no point-in-time input. Common and expected for CRM-imported cohorts. |
| `model_artefacts_unavailable` | The model package is missing on this server. |
| `database_unavailable` | PostgreSQL could not be reached. Also returned for a malformed lead id. |
| `invalid_lead_data` | The lead's data could not be interpreted safely. |

### Layer reasons — `engagement.reason`, `brand_brain.reason`

| Reason | Meaning |
|---|---|
| `no_behavioural_data` | No admissible click or on-site activity for this lead. |
| `only_form_submission` | Just the original form submit — already scored by the base model, so not counted twice as engagement. |
| `no_brand_brain` | The lead's brand has no Brand Brain document. Onboarding is incomplete. |
| `base_model_unavailable` | Scoring stopped before the layers ran. **Not** a statement about the brand — check `resolved_brand_brain_id` to tell the two apart. |
| `signal_config_unavailable` | Signal configuration missing. The base model still scores normally. |

### Fallback response shape

```json
{
  "lead_id": "00000000-0000-0000-0000-000000000001",
  "purchase_probability": null,
  "purchase_probability_percent": null,
  "probability": null,
  "percentile": null,
  "decile": null,
  "priority": "Unavailable",
  "top_factors": [],
  "model_features": null,
  "touchpoint_count": 0,
  "touchpoints": [],
  "model": { "name": "purchase_probability", "version": "baseline_mvp", "status": "baseline_mvp" },
  "availability": {
    "available": false,
    "reason": "lead_not_found",
    "message": "Lead not found."
  },
  "fallback": true,
  "reason": "lead_not_found",
  "engagement":  { "available": false, "reason": "base_model_unavailable", "factors": [], "total_applied": 0.0 },
  "brand_brain": { "available": false, "reason": "base_model_unavailable", "factors": [], "total_applied": 0.0 },
  "lead_priority": {
    "probability": null,
    "score": null,
    "priority": "Unavailable",
    "calibrated": false,
    "layers_applied": []
  },
  "ranking_factors": [],
  "signal_version": null,
  "lead_summary": {
    "available": false,
    "reason": "lead_not_found",
    "message": "Lead not found.",
    "lead_score": null, "lead_score_max": 100, "temperature": null,
    "status": null, "stage": null, "source": null, "created_at": null,
    "touchpoint_count": null, "total_revenue": null, "currency": null,
    "payment_count": 0, "converted": null, "converted_at": null,
    "days_to_convert": null,
    "lifetime_value": { "amount": null, "available": false, "estimated": true,
                        "reason": "lead_not_found", "expected_amount": null,
                        "potential_amount": null },
    "basis": "lead row could not be read"
  },
  "lead_score": null,
  "temperature": null,
  "total_revenue": null,
  "days_to_convert": null,
  "lifetime_value": null
}
```

### The card survives most fallbacks

`lead_not_found` is the case above: there is no lead row, so there is nothing to show. Every
other reason still fills the card, because the lead row was read — only the model had nothing
to say:

```json
{
  "purchase_probability": null,
  "availability": { "available": false, "reason": "no_admissible_form_payload" },
  "lead_summary": {
    "available": true,
    "lead_score": 66, "temperature": "warm", "status": "new",
    "touchpoint_count": 5, "total_revenue": 0.0, "payment_count": 0,
    "converted": false, "days_to_convert": null,
    "lifetime_value": {
      "amount": null, "available": false, "estimated": true,
      "reason": "no_admissible_form_payload",
      "expected_amount": null,
      "potential_amount": 84729.5,
      "order_count": 220
    }
  },
  "lead_score": 66, "temperature": "warm", "total_revenue": 0.0,
  "days_to_convert": null, "lifetime_value": null
}
```

The card shows the lead's real history and the CRM's own score; the probability tile shows
**Unavailable**. Do not fill that tile from `lead_score`.

### ⚠️ Current data coverage — read before demoing this

- `tracking_events` and `tracking_visitors` are **empty (0 rows)**. The on-site collector is
  deployed but capturing nothing, so the engagement layer reports `no_behavioural_data` for
  most leads today.
- Every `ad_click` row in the database is **backdated to payment time** (234 of 234) and is
  therefore rejected by the anti-backfill filter.
- Only **one brand of nineteen** currently has a `brand_brain_id`, so the brand-fit layer is
  inert for the rest.

The wiring is complete and correct — **the signal is what is missing.** Both layers light up
with no code change once the data arrives.

---

## 8. Integration notes

- **Sort on `lead_priority.score`, display `purchase_probability`.** Never sort on the raw
  percentage and never present `lead_priority` as a probability.
- **Don't round the probability for display.** `purchase_probability_percent` reads 1% or 2%
  for nearly every lead. Show one decimal, or lead with the priority band.
- **Branch on `available`, not on key existence.** Every key is present on every response.
- **`availability.message` is written for humans** and can be shown to users verbatim. So can
  every factor `label` and `explanation`.
- **Scores move over time.** Both layers read the clock, so a lead's ranking decays as it ages
  past its brand's sales cycle. Two calls minutes apart can differ slightly — that is by
  design. `purchase_probability` does not move.
- **One call per lead.** There is no batch endpoint yet. Typical latency is ~320 ms p50,
  ~675 ms p95 at 12 concurrent requests.
- **The only non-200 is 401.** Handle a bad API key; everything else arrives as a 200 with a
  reason.

---

## Related documentation

| Document | Covers |
|---|---|
| `purchase_probability_production.md` | Model provenance, calibration, benchmark metrics, and §16 on the signal layers |
| `purchase_probability_model/README.md` | The inference package: files, usage, limitations |
| `postman/ScaleSerum-PurchaseProbability.postman_collection.json` | Importable Postman collection with assertions |
| `PURCHASE_PROBABILITY_API.html` | This document as a formatted page |
