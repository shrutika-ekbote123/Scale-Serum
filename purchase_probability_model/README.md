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
| `inference.py` | Scoring, feature construction, contributions, touchpoint fetch |

## Usage

```python
from purchase_probability_model import predict_for_lead

result = predict_for_lead("02183a5e-e51c-4928-b2be-ac9797936fdb")
result["purchase_probability"]   # 1.28  -> 1.28%
result["percentile"]             # 66
result["priority"]               # "Low"
```

Pass an existing read-only `psycopg` connection as `conn=` to avoid per-call connects.

## Two numbers, two jobs

**`purchase_probability`** is the real calibrated model output as a percentage. The base
rate is 1.09%, so genuine values sit roughly between **0.3% and 4%**. It is never rescaled.
A lead showing 2.8% is genuinely near the top of the distribution.

**`percentile` / `decile` / `priority`** are the relative ranking. This is what a sales team
should sort by — the model's demonstrated value is ranking (lift@10% ≈ 2.10×), not absolute
probability.

## Two touchpoint concepts, never conflated

`touchpoints` is the lead's **real display history**, unfiltered. `model_features` reports
what the model actually consumed: the **first admissible `form_submit`** only — a row written
at lead-creation time and not backdated. Payment, `ad_click` and `call` events are never
features; the audit showed they are written at or after conversion with a backdated
`occurred_at`.

## Unavailable is not zero

If the lead is missing, has no admissible form payload, or the artefacts are absent, the
result carries `availability.available: false`, `fallback: true` and a stable `reason`.
`purchase_probability` is `null` — **never `0`**.

## Regenerating

Requires the frozen training query and a read-only database. The builder aborts unless V1
reproduces **PR-AUC 0.01943 / ROC-AUC 0.60934** exactly.

## Known limitations

- Signal is stronger in June and weakens in July (fold 4 ROC-AUC 0.475).
- Honest nested calibration slope 0.69 — probabilities remain over-dispersed.
- `tracking_events` has 0 rows, so no behavioural features exist.
- Trained on website-channel Wix-form leads; other acquisition paths are out of scope.
