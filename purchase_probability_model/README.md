# Purchase Probability — production model package

Frozen logistic-regression baseline. One product-level model: **`purchase_probability`**,
status **`baseline_mvp`**. Research experiment history lives outside this package.

> **Baseline/model MVP trained on currently available real data. Not yet
> production-optimized due to limited behavioural data and temporal instability.**

## Contents

| File | What it is |
|---|---|
| `model.pkl` | sklearn `Pipeline` — OneHotEncoder + StandardScaler + LogisticRegression, fitted on all 11,753 TRAIN rows |
| `calibration.pkl` | Platt (sigmoid) calibrator, fitted on pooled out-of-fold predictions only |
| `feature_schema.json` | Feature list, reference levels, coefficients, frozen constants, regexes, human labels |
| `percentile_reference.json` | Quantile grid of calibrated OOF probabilities — the ranking reference |
| `model_metadata.json` | Training provenance, frozen benchmark metrics, known limitations |
| `inference.py` | Scoring, feature construction, contributions, touchpoint fetch, layer assembly |
| `signal_config.json` | Weights, regexes and bounds for the two signal layers — **hand-set priors, not fitted** |
| `behavioural.py` | Engagement layer: admissible click actions + journey timeline |
| `brand_fit.py` | Brand Brain layer: parses the MongoDB document, scores brand fit |
| `blend.py` | Bounded log-odds combination → `lead_priority` |

## Usage

```python
from purchase_probability_model import predict_for_lead, resolve_brand_brain_ref

result = predict_for_lead("02183a5e-e51c-4928-b2be-ac9797936fdb")
result["purchase_probability"]   # 1.28  -> 1.28%   (calibrated, never adjusted)
result["percentile"]             # 66
result["priority"]               # "Low"

# With brand context. The document comes from MongoDB `brand_brains`; this package
# deliberately does not open a Mongo connection — the caller supplies the doc.
ref = resolve_brand_brain_ref(lead_id)          # -> {"brand_brain_id": "...", ...}
result = predict_for_lead(lead_id, brand_brain=doc)
result["lead_priority"]["score"]     # 0-100, the number to sort a lead list on
result["engagement"]["observed"]     # clicks, sessions, recency, active days
result["brand_brain"]["factors"]     # ICP fit, channel fit, sales-cycle timing

# The CRM lead card. Read from the leads row, not the model, so it is populated
# even when the model could not score the lead.
result["lead_summary"]["lead_score"]    # 25/100 - the CRM's badge, NOT a probability
result["lead_summary"]["temperature"]   # "cold"
result["lead_summary"]["total_revenue"] # money actually received
result["lifetime_value"]                # what the lead is worth (see below)
```

Pass an existing read-only `psycopg` connection as `conn=` to avoid per-call connects.
Pass `now=` to fix the clock — every time-based factor in one response reads it, so
recency and lead-age can never disagree, and results become reproducible in tests.

## Three numbers, three jobs

**`purchase_probability`** is the real calibrated model output as a percentage. The base
rate is 1.09%, so genuine values sit roughly between **0.3% and 4%**. It is never rescaled.
A lead showing 2.8% is genuinely near the top of the distribution. **The layers below never
modify it** — passing a Brand Brain changes the ranking, not the probability.

**`percentile` / `decile` / `priority`** are the base model's relative ranking. The model's
demonstrated value is ranking (lift@10% ≈ 2.10×), not absolute probability.

**`lead_priority`** is the ranking signal after the engagement and brand-fit layers are
applied. Sort a lead list on `lead_priority.score`; quote `purchase_probability` as the
probability. It reports `calibrated: false` about itself, because it is: the adjustments are
documented priors, not fitted coefficients.

## The lead card — and the number that is not ours

`lead_summary` is CRM state, not model output: the lead score and temperature the CRM keeps
on the `leads` row, the revenue actually received, the payment count, and days-to-convert.
It is read by its own query — `_SQL_LEAD_CARD`, deliberately separate from `_SQL_LEAD` —
because every column in it is written or mutated at payment time and therefore leaks the
outcome. Safe to **show**, never **learned from**, and structurally unable to reach
`build_features`.

**`lead_summary.lead_score` is not `purchase_probability`.** A CRM badge reading `cold 25`
is a score of 25 out of 100. It is not a 25% chance of purchase, and the two differ by more
than an order of magnitude on real leads.

`lifetime_value` is the one derived number here and says so. Once a lead has paid, `amount`
is its recorded revenue and `estimated` is `false`. Before that, `amount` is
`probability × the brand's median paid order` — the median, not the mean, because order
values are long-tailed and one outsized order drags the mean away from what a typical lead is
worth. `expected_amount` always holds the probability-weighted figure and `potential_amount`
holds the undiscounted order value, so a caller can show deal size instead. When there is no
probability and no payment, `amount` is `null` — not a substitute number.

Because the card comes from the lead row, it stays populated on every fallback except
`lead_not_found`: the probability goes null, the history does not.

## The two signal layers

`engagement` reads what the lead has **done** — click actions, high-intent page hits, return
sessions, and how long ago the last of it happened. `brand_brain` reads what the brand said
it **wants** — ideal-customer seniority, the channels it buys, its selling language, and its
sales cycle.

The two meet at recency: the brand's declared sales cycle sets the half-life at which
engagement goes stale. A brand that closes same-day treats a two-day-old click as cold; a
brand with a six-month cycle does not. That is `engagement.recency_half_life_hours`, and its
`derived_from` field says whether it came from the brand or from the default.

Each layer's total is **clamped** (`bounds` in the response). An unfitted prior must not be
able to overwhelm a measured model, so it structurally cannot.

## Absent signal is not negative signal

A factor with no data contributes **exactly 0.0** and reports `status: "no_data"`. A lead we
know nothing about must never sink below a lead we know something mildly bad about. The same
rule governs whole layers: a brand with no Brand Brain gets `available: false`, not a penalty.

## Why the layers are priors and not trained features

There is nothing to train them on, and the obvious training data is poisoned:

- `tracking_events` / `tracking_visitors` are **empty** — the on-site collector is deployed
  but capturing nothing. This is the real blocker on behavioural modelling.
- All 234 `ad_click` rows are **backdated**: written at or after their lead's payment. Zero
  survive the admissibility filter. A model trained on them reads the receipt.

So the layers are applied as bounded, documented, separately-reported adjustments. When the
collector starts producing events the wiring is already in place, and at that point these
weights should be replaced by fitted ones.

## Three touchpoint concepts, never conflated

`touchpoints` is the lead's **real display history**, unfiltered — including payments.

`model_features` reports what the base model consumed: the **first admissible `form_submit`**
only — a row written at lead-creation time and not backdated. Payment, `ad_click` and `call`
events are never base-model features; the audit showed they are written at or after
conversion with a backdated `occurred_at`.

`engagement.timeline` is a third thing again: rows that passed the anti-backfill filter
(`created_at - occurred_at <= 1h`) with `payment` excluded outright, inside a window that
closes at the first payment **write** time. It feeds `lead_priority` and nothing else.

Likewise `top_factors` stays base-model-only. The merged, provenance-tagged list is
`ranking_factors`, where every entry carries `layer` (`model` / `engagement` /
`brand_brain`) and `basis` (`calibrated_model` / `heuristic`).

## Unavailable is not zero

If the lead is missing, has no admissible form payload, or the artefacts are absent, the
result carries `availability.available: false`, `fallback: true` and a stable `reason`.
`purchase_probability` is `null` — **never `0`**.

Layer keys are present on every response, including fallbacks, so the UI branches on
`available` rather than on whether a key exists. Stable layer reasons:

| Reason | Meaning |
|---|---|
| `no_behavioural_data` | No admissible click or on-site activity for this lead |
| `only_form_submission` | Just the original form submit — already scored by the base model, so not re-counted as engagement |
| `no_brand_brain` | The lead's brand has no Brand Brain document (onboarding incomplete) |
| `base_model_unavailable` | The base model could not score the lead, so the layers never ran |
| `signal_config_unavailable` | `signal_config.json` missing; the base model still scores |

A layer reason describes **the layer**, never the base model's problem. `no_brand_brain` and
`base_model_unavailable` are different facts: the first says the brand has no document, the
second says we never got as far as looking. `brand_brain.resolved_brand_brain_id` tells you
which it was.

## Regenerating

Requires the frozen training query and a read-only database. The builder aborts unless V1
reproduces **PR-AUC 0.01943 / ROC-AUC 0.60934** exactly.

## Known limitations

- Signal is stronger in June and weakens in July (fold 4 ROC-AUC 0.475).
- Honest nested calibration slope 0.69 — probabilities remain over-dispersed.
- `tracking_events` has 0 rows, so no behavioural features exist **in the trained model**.
  The engagement layer reads it live and will light up the moment the collector reports.
- Trained on website-channel Wix-form leads; other acquisition paths are out of scope.
- Layer weights in `signal_config.json` are priors, not measurements. They are bounded and
  reported separately for exactly that reason. Replace them with fitted values once there is
  behavioural data to fit on.
- Only one of nineteen brands currently has a `brand_brain_id`, so the brand-fit layer is
  inert for the rest — reported as `no_brand_brain`, never as a penalty.
- `f_seniority`, `f_email_class` and `f_locale` are inputs to *both* the base model and the
  brand-fit layer (which asks a different, brand-specific question of them). The brand
  weights are deliberately small to limit the resulting double count.
