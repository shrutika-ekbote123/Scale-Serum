# Sales Call Analyzer

Transcribes and diarizes a recorded sales call, evaluates it against the
six-stage sales framework, verifies every claim against the transcript, and
scores it deterministically.

API contract for the backend team: **`../SALES_CALL_ANALYZER.md`**.

## Contents

| File | Responsibility |
|---|---|
| `framework.py` | Loads and validates `sales_framework.json`; derives the neutral placeholders |
| `models.py` | Request/response and internal shapes (pydantic) |
| `deepgram_client.py` | Pre-recorded transcription + diarization over plain HTTP |
| `transcript.py` | Normalises any source into one indexed, speaker-attributed shape |
| `speakers.py` | `speaker_id` → role, only where evidence supports it |
| `context.py` | Brand Brain + customer + product + call metadata assembly |
| `analyzer.py` | The one Gemini call: schema, prompt, structural validation |
| `evidence.py` | Verifies every quote against the transcript; drops what does not hold |
| `scoring.py` | criterion → stage → overall, in Python only |
| `report.py` | Assembles the response. Renames and arranges; never rescores |
| `store.py` | MongoDB persistence, idempotency, stale-job handling |
| `pipeline.py` | Orchestration and the status machine |
| `sales_framework.json` | The six stages, their criteria, and every business value (all null) |
| `signals_config.json` | Closed vocabularies for signals, techniques and pitch structures |

## Division of labour

These are correctness requirements, not style preferences.

    Deepgram  what was said, who spoke, when
    Gemini    interpretation: needs, objections, signals, evidence, ratings, prose
    Python    validation, evidence verification, scoring, persistence, orchestration

**The model never produces a number.** The response schema has no numeric score
field, the system instruction forbids one, and `scoring.py` computes every score
from ordinal ratings plus the config. Three consequences, all deliberate:

* the same ratings always produce the same score;
* historical calls can be rescored under new weights with no provider call;
* nothing said on a call can move a number, because the component that reads the
  call never touches the arithmetic.

## No invented business values

Weights, the rating scale, score bands, the not-applicable policy and the
disposition policy are all `null` in `sales_framework.json`. Management has not
set them. Placeholders are neutral — equal weighting, uniform rating spacing —
and every response reports which values were still unconfirmed
(`weighting`, `rating_scale_mode`, `*_confirmed: false`, `band_reason`).

Setting a real value is a JSON edit plus a `framework_version` bump. No Python
changes, and the rescore endpoint brings existing analyses onto the new config.

## Three things that are not zero

* **Not applicable** — the call gave no opportunity (no objection was raised).
  Excluded from the denominator; `score: null`.
* **Unsupported** — evidence did not verify. We do not know how the rep did,
  which is different from knowing they did badly. Also excluded.
* **Unavailable** — the analysis failed. `scores: null` and a stated reason; a
  failed analysis never becomes a neutral scorecard.

## Evidence

Every claim cites `segment_index` + `speaker_id` + a verbatim quote.
`evidence.py` resolves it against the real transcript and drops anything that
does not match; timestamps are copied from the segment, so they cannot be
invented.

Claims about **absence** are the exception — there is no line where a rep failed
to ask about budget. The lowest rating level, weaknesses and recommendations may
stand without an anchor and are marked `evidence_backed: false`. Nothing
unsupported passes silently: it either dies or is labelled.

## Speakers

`speaker_id` is the primary identity and is never renamed. Roles are annotations
with a stated basis, and `unknown` is a valid outcome. Elimination ("the other
one must be the customer") applies only to two-speaker calls. Roles are never
inferred from turn order or talk time — "the first speaker is the rep" is false
for inbound calls, and "whoever talks most is the rep" is the exact bias the
analysis would then be scoring. Names come from the CRM record or nowhere.

## Context, not stereotypes

Region, language, price band and customer profile are supplied to the model as
facts about **this call**. They inform interpretation and recommendations. They
do not change a weight, and there are no per-region or per-profile behaviour
rules anywhere in this package — adding any would need a written basis from the
business. A scenario test asserts that identical ratings produce an identical
score whatever the region.

## Cost

* Idempotent by input fingerprint (which includes the versions that shape output).
* The transcript is persisted **before** the LLM runs, so a failed analysis never
  re-transcribes on retry.
* Rescoring costs nothing.
* Raw provider output is not stored unless `SCA_STORE_RAW_TRANSCRIPT=true`.

## Configuration

See `../.env.example` for every setting. Nothing here is required for the app to
boot: an unset `DEEPGRAM_API_KEY` degrades transcription and says so.

## Tests

    python -m pytest tests/ -k sales_call

231 tests, all offline — no network, no database, no API keys. Deepgram is an
`httpx.MockTransport`, Gemini is a fake client, MongoDB is a fake collection.
`tests/test_sales_call_scenarios.py` maps the 22 required scenarios.
