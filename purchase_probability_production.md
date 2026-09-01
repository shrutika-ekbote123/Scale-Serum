# Purchase Probability — Production MVP (Track 1)

**Status: implemented and tested. Frontend integration pending — see §12.**

> **Baseline/model MVP trained on currently available real data. Not yet
> production-optimized due to limited behavioural data and temporal instability.**

---

## 1. What was built

```
Lead (PostgreSQL, read-only)
  ↓  first ADMISSIBLE form_submit  (written at lead creation, not backdated)
V1 feature construction  (7 features)
  ↓
Frozen logistic-regression baseline
  ↓  Platt calibration (fitted on pooled OOF only)
Calibrated probability  →  percentile / decile / priority
  +  signed per-feature contributions
  +  real touchpoint history (display only)
  ↓
GET /api/purchase-probability/{lead_id}
```

## 2. Model

| Property | Value |
|---|---|
| Product model name | `purchase_probability` |
| Status / version | `baseline_mvp` |
| Internal feature version | V1 (frozen) |
| Estimator | `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight=None, random_state=42)` |
| Design matrix | 14 columns + intercept |
| Fitted on | All 11,753 TRAIN rows / 135 positives |
| Snapshot | `final_trainset_v3.sql`, pinned at `2026-08-31T00:00:00+00:00` |
| **TEST usage** | **NONE** — split off and deleted before any fit, calibration or reference |

**Build-time gate:** the builder aborts unless V1 reproduces exactly. Verified at build:
**PR-AUC 0.01943**, **ROC-AUC 0.60934** — match to five decimals.

### Frozen benchmark (unchanged, for reference only)

| Metric | Value |
|---|---|
| Pooled OOF PR-AUC | 0.01943 (no-skill 0.01184) |
| Pooled OOF ROC-AUC | 0.60934 |
| Pooled OOF lift@10% | 2.10× [1.30, 2.90] |
| Honest nested ECE | 0.0046 |
| Honest calibration slope | 0.69 |

Experiment 2 was evaluated and is **not promoted**. It remains research evidence.

## 3. Features (V1, frozen — nothing added)

`f_seniority` · `f_email_class` · `f_locale` · `f_company_len` ·
`f_company_is_placeholder` · `f_hour_sin` · `f_hour_cos`

Sourced point-in-time from the immutable form payload
(`touchpoint_events.payload → data → contact`), with `leads.email` as the only fallback —
verified unmutated on 157/157 positives during the audit.

### A real bug found and fixed during implementation

PostgreSQL's ARE dialect treats `\b` as **backspace**, not a word boundary. The training
query's `^vp\b` and `^head\b` alternatives therefore **never matched** — titles like `'VP'`
and `'Head Admin & IR'` were trained as `individual_contributor`. Python's `re` treats `\b`
as a word boundary, so a naive port would have silently reclassified them as `vp_director`
and produced scores the frozen model never learned.

`feature_schema.json` stores Python-dialect regexes that reproduce what PostgreSQL actually
did. Verified against 400 distinct job titles: **0 disagreements**. Live feature parity
against 200 real TRAIN leads: **0 mismatches across all 7 features**.

## 4. Calibration

Platt (sigmoid) on log-odds, fitted on the **pooled out-of-fold predictions** from the
rolling-origin CV inside TRAIN. Out-of-sample by construction; TEST never involved.

Parameters: intercept −2.18416, slope +0.49087.
Calibrated OOF mean 0.011849 against an observed rate of 0.011841.
Calibrated range across OOF: **0.29% – 3.86%**.

Honest nested-calibration slope is **0.69**, i.e. probabilities remain over-dispersed. This
is documented, not hidden. The probability is a **baseline estimate**, not a proven
production-calibrated probability.

## 5. Probability representation — read this before designing the UI

The base rate is **1.09%**. Observed calibrated output on a 400-lead real sample:

| | Value |
|---|---|
| Minimum | 0.52% |
| Median | 1.10% |
| Maximum | 2.80% |

**`purchase_probability` is the real calibrated probability as a percentage.** `1.28` means
1.28%. It is never inflated, rescaled or cosmetically boosted. A single-digit number here is
the honest answer, not a bug.

`purchase_probability_percent` is the same number rounded to an integer, which means it will
read **1%** or **2%** for almost every lead. **Use the ranking fields as the primary UI
signal**, not the integer percent.

## 6. Percentile / decile / priority

- **Reference population:** the 8,445 pooled out-of-fold calibrated probabilities (100
  positives). Out-of-sample, TRAIN-only, **TEST never used**, no future outcomes involved.
- Stored as a 1,001-point quantile grid in `percentile_reference.json`.
- **`percentile`** = position of this lead's calibrated probability in that reference.
- **`decile`** = `ceil(percentile / 10)`, clamped to 1–10.
- **`priority`** = decile 10 → `High`; deciles 8–9 → `Medium`; deciles 1–7 → `Low`.

Bands follow the evidence: the demonstrated lift is concentrated in the top decile
(lift@10% ≈ 2.10×), so only decile 10 is called High.

## 7. Top contributing factors

Contribution = **transformed feature value × model coefficient** (signed, in log-odds).
One-hot columns of a feature are summed. `f_hour_sin` and `f_hour_cos` are summed into a
single "Submission time pattern" factor — they are one feature in two columns and are
meaningless read apart. Factors are ranked by absolute contribution; negatives are returned
too.

Language is deliberately non-causal: *"Contributed positively to this prediction."* Never
*"caused"*. A test asserts the word `caused` never appears.

| Feature | Label |
|---|---|
| `f_seniority` | Senior decision-maker role |
| `f_email_class` | Email address type |
| `f_locale` | Lead locale |
| `f_company_len` | Company information detail |
| `f_company_is_placeholder` | Company information quality |
| `f_hour_time_pattern` | Submission time pattern |

## 8. Touchpoints — display vs model, kept strictly apart

**Display** (`touchpoints`): every real touchpoint for the lead from `touchpoint_events`,
chronological, all types including payment. Only fields that genuinely exist are returned;
nulls are omitted rather than invented. Nothing is fabricated.

**Model** (`model_features`): the **first admissible `form_submit`** only —
`created_at <= lead.created_at + 1h` **AND** `created_at − occurred_at <= 1h`.

Payment / `ad_click` / `call` are **never** features. The V1 audit established that 100% of
`ad_click` and 99% of `call` rows are written at or after the payment with a backdated
`occurred_at` (median write lag 45 and 36 days).

The example in §10 shows exactly this: a lead that later paid ₹30,000 is displayed with its
payment event, and was still scored 1.28% from its form submission alone.

## 9. API

```
GET /api/purchase-probability/{lead_id}
Header: X-API-Key: <API_KEY>
```

Follows the existing service conventions: `Depends(require_api_key)`, and the house rule of
**always HTTP 200** with `fallback: true` on any failure. Inference is synchronous
(psycopg + sklearn) so it runs via `run_in_threadpool` and never blocks the event loop.

`401` for a missing or wrong key (app.py:103). `/health` remains open.

## 10. Real response

Real lead, real data, unedited:

```json
{
  "lead_id": "02183a5e-e51c-4928-b2be-ac9797936fdb",
  "purchase_probability": 1.28,
  "purchase_probability_percent": 1,
  "probability": 0.01279036,
  "percentile": 66,
  "decile": 7,
  "priority": "Low",
  "top_factors": [
    { "feature": "f_locale", "label": "Lead locale", "value": "en-US",
      "direction": "positive", "contribution": 0.500586,
      "explanation": "Contributed positively to this prediction." },
    { "feature": "f_seniority", "label": "Senior decision-maker role",
      "value": "founder_c_level", "direction": "positive", "contribution": 0.421617,
      "explanation": "Contributed positively to this prediction." },
    { "feature": "f_hour_time_pattern", "label": "Submission time pattern",
      "value": null, "direction": "negative", "contribution": -0.164086,
      "explanation": "Contributed negatively to this prediction." },
    { "feature": "f_company_len", "label": "Company information detail",
      "value": 23.0, "direction": "positive", "contribution": 0.097566,
      "explanation": "Contributed positively to this prediction." }
  ],
  "model_features": {
    "feature_version": "V1",
    "values": {
      "f_seniority": "founder_c_level", "f_email_class": "gmail",
      "f_locale": "en-US", "f_company_len": 23.0,
      "f_company_is_placeholder": 0.0,
      "f_hour_sin": 0.707107, "f_hour_cos": 0.707107
    },
    "source": "first admissible form_submit (written at lead creation, not backdated)",
    "form_occurred_at": "2026-06-23T03:36:06.843000+00:00",
    "form_created_at": "2026-06-23T03:36:11.794000+00:00"
  },
  "touchpoint_count": 2,
  "touchpoints": [
    { "type": "form_submit", "occurred_at": "2026-06-23T03:36:06.843000+00:00",
      "created_at": "2026-06-23T03:36:11.794000+00:00", "channel": "website",
      "provider": "wix", "source": "ICDP Webinar Registration Testing" },
    { "type": "payment", "occurred_at": "2026-07-07T12:14:50+00:00",
      "created_at": "2026-07-07T12:15:07.007000+00:00", "channel": "payment",
      "provider": "cashfree", "source": "00::Success",
      "value": 30000.0, "currency": "INR" }
  ],
  "model": { "name": "purchase_probability", "version": "baseline_mvp",
             "status": "baseline_mvp", "feature_version": "V1" },
  "availability": { "available": true, "reason": null, "message": null },
  "fallback": false
}
```

Unavailable case, also real:

```json
{
  "lead_id": "00004b77-a7b1-4137-a7cf-6318295b1415",
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
  "model": { "name": "purchase_probability", "version": "baseline_mvp",
             "status": "baseline_mvp" },
  "availability": {
    "available": false,
    "reason": "no_admissible_form_payload",
    "message": "No admissible form submission was recorded for this lead, so the model has no point-in-time input to score."
  },
  "fallback": true,
  "reason": "no_admissible_form_payload"
}
```

### Unavailable reasons

`lead_not_found` · `no_admissible_form_payload` · `model_artefacts_unavailable` ·
`database_unavailable` · `invalid_lead_data`

**`null` and `0` mean different things.** Never render an unavailable result as 0%.

## 11. UI guidance

```
Purchase Probability
1.3%                      ← the real calibrated number, never inflated
Priority: Low  ·  62nd percentile  ·  decile 7

Why this prediction
  + Lead locale                    contributed positively
  + Senior decision-maker role     contributed positively
  − Submission time pattern        contributed negatively

Touchpoints (2)
  form_submit   23 Jun 2026 03:36   website / wix
  payment       07 Jul 2026 12:14   ₹30,000
```

Rules:
- Show one decimal. `1.3%` is honest; `1%` throws away the only resolution there is.
- **Lead the eye with priority/percentile**, not the percent.
- Never derive this from Lead Score, "Hot", or any business rule.
- On `available: false` show **"Unavailable"** and the message — never a number.

## 12. Frontend integration status

**API ready; frontend integration requires the main application repository.**

The lead-detail UI that currently renders Purchase Probability / Lead Score / Qualified /
Touchpoints / LTV / Total Revenue is **not in this workspace**. `c:\Marketing_tool` is the
Brand Brain AI sidecar; it contains one onboarding-wizard component and an integration
example, no app shell, router or lead pages. The main React app and the Node/Sequelize
service that owns `scrumdb` are separate repositories.

No frontend files were invented. The API contract above is complete and ready to consume.

## 13. Deployment

1. `pip install -r requirements.txt` — adds scikit-learn, pandas, numpy, joblib, psycopg.
2. Ensure `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` are set. The
   connection is opened `read_only`. **A read-only database role is strongly recommended** —
   the application-level flag is a safety net, not a permission boundary.
3. Deploy `purchase_probability_model/` alongside `app.py` (the `.pkl` files must ship).
4. Restart via the existing pm2 process (`marketing-tool`) or the systemd unit.
5. Smoke test:
   `curl -H "X-API-Key: $API_KEY" https://<host>/api/purchase-probability/<lead_id>`

If the ML dependencies are missing the service still boots; the endpoint returns
`model_artefacts_unavailable` rather than 500.

## 14. Guarantees

- **No dummy data.** No synthetic leads, conversions, touchpoints or training rows.
- **No database writes.** Read-only connection; no DDL, DML or schema change.
- **TEST never used** for fitting, calibration, the percentile reference, or any decision.
- **No V2 features.** Asserted in tests against the schema and the live response.
- **No fabricated probability.** Unavailable returns `null`.
- **V1 remains the frozen benchmark**; Experiment 2 is not promoted.
- Frozen artefacts untouched: `final_trainset_v3.sql`, `purchase_probability_ml_spec_v1.md`,
  `experiment_v1/`, `experiment_v2/`, all V1/V2 reports.

## 15. Honest limitations

1. Ranking, not probability, is the usable signal. Plan a pilot around **1.2×–2.0×** lift at
   the top decile, not the 2.10× point estimate.
2. Temporal stability is unproven — signal concentrates in June; fold 4 ROC-AUC was 0.475.
3. Calibration slope 0.69 — over-dispersed. Refit the intercept monthly on matured labels.
4. No behavioural features exist: `tracking_events` = 0 rows (Track 2).
5. Kaizen CRM leads have no touchpoint rows and will return `no_admissible_form_payload`
   by design, not by accident.
6. This is **not** a proven production-optimized model and must not be described as one.
