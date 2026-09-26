# Creative Coach — evaluation suite

Built **before** the prompts, so every change to the coach answers with a number
instead of an impression.

```bash
python -m evals.run --adapter fallback --offline   # CI: no database, no key, no spend
python -m evals.run --adapter coach                # the real coach, live data, costs money
python -m evals.run --adapter baseline             # the coach that ships today
python -m evals.extract                            # rebuild production.jsonl from live threads
python -m evals.fixtures                           # re-snapshot the offline fixtures
python -m evals.run --adapter coach --json report.json --baseline evals/baseline-offline.json
```

## Two modes, and what each one can tell you

| Mode | Needs | Proves | Cannot tell you |
|---|---|---|---|
| `--offline` | nothing | The machinery: every hard gate, the router, fact coverage, and that the deterministic stand-in is itself grounded | Anything about the model's answers — none is called |
| `--adapter coach` | scrumdb, Mongo, a Gemini key, real money | Answer quality, refusals, rewrites, fallback rate | — |

**CI runs the offline mode on every push** (see `.github/workflows/deploy.yml`). The live
mode is for a human, or a schedule. A build that needs a database and a paid API call to
go green is a build people learn to bypass.

Exit code is 1 when a **hard gate** fails, so it can gate CI and the Postman run.
Soft checks are printed and tracked but do not fail the run unless you pass
`--strict`: at temperature 0.5 they move a few points between runs, and a build
that fails at random is one people rerun until it is green - which quietly
disables the hard gates riding along with it. Use `--strict` for a release check,
plain for CI.

## What is in the suite

| File | Cases | Where it comes from |
|---|---|---|
| `production.jsonl` | 12 | Generated from real `coach_thread` conversations. 50 real user turns, but only **12 distinct questions** — volume is not variety |
| `adversarial.jsonl` | 25 | Hand-written, one per failure mode in `SCRIPT_LAB_COACH_API.md` §5 |

A case may carry a `fixture` (currently `{"brand_brain_tier": "B"}`), which trims a real
Brand Brain down to that tier before the turn runs. It exists because **no brand in live
data sits at tier B** — both linked Brand Brains are complete — so the partial-context
path would otherwise ship untested. A fixture case is scored against the same trimmed
context it was answered from.

`production.jsonl` is generated; edit the labels in `extract.py`, not the file.
`adversarial.jsonl` is hand-written; edit it directly.

## Hard gates — a failure fails the run, whatever the totals say

| Gate | Why it is absolute |
|---|---|
| `grounded` | An invented figure is a checkable false claim about the user's own money |
| `no_placeholder` | "undefined" reached real users 22 times in 246 turns. Never again |
| `no_leak` | Naming another company to this brand's user is the worst thing this feature can do — and DI's Brand Brain already contains Lawttorney's content |
| `right_test` | An answer about a different test is worthless and invisible |
| `answered` | — |

Soft checks (`intent`, `refused`, `covers_required_facts`, `gave_a_rewrite`) have
targets in `run.py` and are qualities to improve rather than absolutes.

## Where the two coaches stand, 24 Sep 2026

The coach that **ships today**, on the 12 real questions:

```
1/12 cases fully passed
HARD  grounded 12/12 · no_leak 12/12 · no_placeholder 3/12   <- FAILS
SOFT  covers_required_facts 92% · refused 0% · gave_a_rewrite 0%
```

It never invents a number - it is a template, so it cannot - but it prints
`undefined` at users in **9 of 12** answers, never declines an off-topic question,
and never produces the concrete copy people ask it for.

The **new coach**, on all 37 cases (phase 6):

```
32/37 cases fully passed
HARD  answered 37/37 · grounded 37/37 · no_leak 37/37
      no_placeholder 37/37 · right_test 37/37          ALL PASS
SOFT  answered_with_ai 100% · covers_required_facts 97%
      gave_a_rewrite 100% · intent 97% · refused 83%   <- under target
```

**Expect run-to-run variation.** The coach runs at temperature 0.5, so the soft
numbers move by a few points between runs and `gave_a_rewrite` has ranged from
75% to 100% on an unchanged build. Judge a change by several runs, or drop the
temperature first; a single run is a reading, not a verdict. The hard gates have
held at 32/32 on every run so far.

## Regression detection

`--baseline <report.json>` compares this run with a stored one and **fails when a case
that used to pass stops passing**. Rate drift is printed but does not fail: rates move a
few points between runs at this temperature, while a named case flipping is a specific,
reproducible claim about a specific question — the kind worth blocking a build for.

`evals/baseline-offline.json` is the committed offline baseline. Refresh it deliberately,
never to make a red build green:

```bash
python -m evals.run --adapter fallback --offline --json evals/baseline-offline.json
```

## Offline fixtures

`evals/fixtures/*.json` are frozen **inputs** to four real turns — a failing script with a
stale ad, a script that was never published, a brand with no Brand Brain, and a review that
never completed. They hold what the coach was given, never what any model said, so a
fixture cannot quietly become the expected answer. Re-snapshot with `python -m evals.fixtures`
when the underlying test rows change.

## Adding a case

Append to `adversarial.jsonl`:

```json
{"id": "adv-021", "source": "adversarial", "scenario": "short_name",
 "test_id": "<a real test id>", "brand_id": "<its brand, or null>",
 "question": "...", "expect_intent": "explain",
 "must_mention": ["hook_score"], "must_refuse": false,
 "forbid_brands": ["Other Brand"], "note": "why this case exists"}
```

`must_mention` names **fact keys** (see `script_lab_coach/facts.py`), not words.
A case is satisfied by engaging with the fact's value *or* its subject, so a case
does not fail merely because the coach phrased something well.

## Caveats, honestly

- **12 real questions is a small sample**, and four of them are greetings. The
  adversarial half carries most of the coverage until real usage grows. Re-run
  `extract.py` as it does.
- **The baseline adapter cannot be asked new questions.** The 20 adversarial
  cases are skipped for it, because the shipping coach was never asked them and
  scoring silence as a pass would flatter it.
- **No judged quality score yet.** Tone and usefulness need a rated sample and
  belong on top of these mechanical checks, never in place of them.
