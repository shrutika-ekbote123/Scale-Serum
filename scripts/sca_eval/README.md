# Sales Call Analyzer — accuracy harness

Measures transcription, speaker-role and language-routing accuracy on real and
labelled audio, so every change to the transcription path answers with a
number. It runs the product's own code (`build_params`, `language_for`,
`from_deepgram`, `resolve_roles`), not a copy of it.

Everything it downloads or writes goes to `workspace/`, which is **gitignored**:
real customer recordings, their transcripts and CRM names. Never commit it.

```bash
python scripts/sca_eval/fetch_datasets.py real          # 28 recordings from scrumdb (read-only)
python scripts/sca_eval/fetch_datasets.py fleurs        # 30 utterances x 8 Indian languages
python scripts/sca_eval/make_synthetic_calls.py         # 16 two-speaker calls, exact truth (~$0.50)
python scripts/sca_eval/run_eval.py                     # all sets x baseline/phase0/oracle (~$3)
python scripts/sca_eval/run_eval.py --rescore           # re-score cached responses, no spend
python scripts/sca_eval/make_label_drafts.py            # drafts for human labelling
# then open scripts/sca_eval/label_tool.html in a browser
```

Needs `DEEPGRAM_API_KEY`, `GEMINI_API_KEY` and the `DB_*` settings from `.env`,
plus ffmpeg (`SCA_FFMPEG_DIR` or PATH).

## Three sets, three questions

| Set | What it is | Answers | Cannot answer |
|---|---|---|---|
| `real` | 28 ScaleSerum calls (142 min, 3 brands), from `sales_calls.recording_url` | Coverage, words/min, roles resolved, language chosen — and everything else once labelled | Accuracy, until a human labels them |
| `fleurs` | Google FLEURS test split (CC-BY-4.0), hi/mr/kn/pa/ta/te/gu/bn, degraded to 8 kHz mu-law | Word/character error per language | Diarization (one speaker per file) |
| `synthetic` | Gemini-written code-switched sales calls, one fixed TTS voice per role, turns joined with real gaps, 8 kHz mu-law | Diarization, role accuracy, backchannels, per-speaker loss — with exact truth | Absolute error rates: TTS is cleaner than people |

Synthetic calls deliberately contain what breaks the pipeline: 3–4 one-word
customer backchannels, a rep self-introduction with the brand, the customer
addressed by name, a language switch, an objection and a price. Half use two
voices of the same gender.

## Configurations

| Config | Language sent | Keyterms |
|---|---|---|
| `baseline` | `multi` (production today) | none |
| `phase0` | from language identification + the English-only rule (`SCA_LANGUAGE_ID=on`) | brand, product, rep, customer |
| `oracle` | the true language | same |

`oracle` is the ceiling for any one-language-per-call routing. Where it is still
poor, better detection cannot help; per-segment routing (Phase 2) is needed.

## Reading the numbers

- **roman WER/CER** compares after romanising both sides, so "demo" vs "डेमो"
  is not an error. Read it for mixed-language calls. **native** is for
  single-language references.
- **CER** matters more than WER for Marathi/Kannada, where one "wrong word" is
  often one wrong suffix.
- **words kept** (length ratio) far below 1.0 means speech was dropped.
- **role accuracy** is per word: the product labelled the speaker of that word
  with the right role. Unknown counts as wrong. This is the ">95% speaker role"
  number.
- **diarization accuracy** is the same, before role labels: were the voices
  split right? Low diarization + high role accuracy on the rest separates the
  two failure modes.
- **backchannel recall** — short customer turns ("haan", "okay") attributed to
  the customer.

## Labelling the real calls

`make_label_drafts.py` writes `workspace/labels/<id>.draft.json` from the
machine's own output. Open `label_tool.html` (from disk, no server), load the
draft and the audio from `workspace/real/`, then:

1. set a role for each voice, and split turns where two people were merged;
2. correct the words — Devanagari for Hindi/Marathi, Latin for English words;
3. set each turn's language and any clear tone tag;
4. tick *I listened to the whole call*, save, and move the file into
   `workspace/labels/` as `<id>.json`.

Only files with `"reviewed": true` are scored; a draft scored against itself
would measure nothing. Tone tags are collected now for the Phase 3 tone work.

## Baseline — 2026-09-28

`run_eval.py`, all sets. Real calls: 13 unique recordings (28 rows; the same
audio is attached to up to 7 rows). Full report in `workspace/reports/`.

**FLEURS, one language per clip, phone quality — roman WER**

| | bn | gu | hi | kn | mr | pa | ta | te |
|---|---|---|---|---|---|---|---|---|
| baseline (`multi`) | 101% | 90% | 19% | 122% | 86% | 82% | 117% | 112% |
| phase0 (detected) | **24%** | **27%** | 19% | **29%** | **31%** | **37%** | **44%** | **29%** |
| oracle | 18% | 23% | 16% | 24% | 30% | 36% | 31% | 29% |

**Synthetic two-speaker calls (16)**

| | role acc | diarization | backchannels to customer | rep WER | customer WER |
|---|---|---|---|---|---|
| baseline | 78.1% | 87.1% | 21.9% | 9.1% | 31.4% |
| phase0 | 78.0% | 87.1% | 20.3% | **6.6%** | 30.8% |
| oracle (regional code for the whole call) | 60.6% | 85.5% | 18.2% | 35.0% | 28.4% |

The oracle row is why `SCA_REGIONAL_MIN_SHARE` exists: even with the right
language known, one code per call hurts mixed calls. Speaker role accuracy is
held back by diarization, not by labelling. Same-gender pairs collapse to one
voice (4/16), a language switch splits one person into two, and short customer
replies land on the rep.

**Real calls (label-free)**

| | coverage (median) | rep identified | language sent |
|---|---|---|---|
| baseline | 0.89 | 0 / 13 | multi ×13 |
| phase0 | 0.89 | 3 / 13 | en ×7, multi ×5, mr ×1 |

Rep identification on real calls is capped by the CRM data: most rows name a
rep who is not the person on the recording. Labelling fixes the measurement.

## Phase 1 — speakers, 2026-09-28

New configs (no extra Deepgram spend — they reuse phase0's cached responses):

| Config | What it adds |
|---|---|
| `phase1` | speaker refinement, voices compared only with each other |
| `phase1_vp` | refinement with the rep's voiceprint, enrolled through `voiceprints.build()` from OTHER calls (`SCA_EVAL_ENROL_CALLS`, default 1) |
| `prep` | phase0 with the audio high-passed, loudness-normalised and sent as FLAC |

`embedding_bakeoff.py` chose the embedding model (ERes2Net) before any of this.

**Synthetic two-speaker calls (16)**

| | role acc | diarization | backchannels to customer | correct / swapped |
|---|---|---|---|---|
| phase0 | 78.0% | 87.1% | 20.3% | 8 / 3 |
| phase1 (no voiceprint) | 77.9% | 89.5% | 26.0% | 9 / 2 |
| phase1_vp, enrolled from 1 call | 85.4% | 91.9% | 62.6% | 13 / 3 |
| **phase1_vp, enrolled from 2 calls** | **90.4%** | **92.6%** | **67.3%** | **15 / 1** |

What moved it, each measured before it was kept:

- **Voiceprint, stretch by stretch.** At 0.40 against the voiceprint, a stretch is the rep.
  Below 0.35 it is not; in between, Deepgram's label stands. This fixed every same-gender
  collapse (1 speaker → 2) and every language split.
- **Edge words.** A customer's "Okay." sits in the same stretch as the rep's next sentence,
  because Deepgram leaves no gap. The first and last word of each rep stretch are scored alone
  and split off below 0.10. At 0.35, 56% of rep edge words would have been wrongly split.
- **Folding stray voices back.** A new voice with under 4 s of speech rejoins the nearest one.
  Without this, two-person calls came back with up to five speakers.
- **Two enrolment calls.** The embedding shifts with language. A rep enrolled from English
  scored 0.73 on their English and 0.34 on their Hindi, close to the customer's 0.18.

Tried and dropped: a relative second pass (81% against 83%), lower voiceprint thresholds
(0.35/0.30 scored 87%, but over-fits 16 TTS calls), and **audio preprocessing**. `prep` moved
FLEURS WER by ±1 point in most languages, made synthetic customer WER worse (30.8% → 32.3%)
and lowered real-call words per minute. Deepgram keeps getting the original recording.

Without a voiceprint, refinement barely moves the numbers, and that is deliberate. The same
rep across a language switch scored 0.48–0.51 against themself, while two different
same-gender people scored up to 0.74. Merging or splitting on that would guess.

Still wrong: en_only_3, whose customer voice scores 0.53 against the rep's voiceprint. No
threshold separates that pair. Cost: about 27 s of CPU per 10-minute call.

## Phase 2 — the customer's language, 2026-09-28

`segment_pass_experiment.py` decided the approach before any product code: the customer's
turns of the 8 regional synthetic calls, each candidate scored against the truth.

| Candidate for the customer's turns | WER | CER | words kept |
|---|---|---|---|
| today (`multi`) | 57.0% | 47.3% | 0.70 |
| Deepgram regional model, customer turns only | 62.6% | 37.8% | 0.80 |
| **Gemini, one clip per turn** | **23.4%** | **17.4%** | **0.89** |
| best of the three per turn (ceiling) | 22.6% | 17.2% | 0.89 |

Gemini sits almost at the ceiling, so no per-turn selection rule is needed. On the real
Marathi call, `multi` produced Hindi-sounding nonsense. Deepgram's `mr` model garbled the
English and dropped an exchange. Gemini gave correct Marathi with the English intact.
No Sarvam key was available, so Sarvam was not compared.

In the pipeline (`phase2_vp`: phase1_vp + segment pass), customer WER (roman) was:

| | kn | mr | pa | ta | te | all 16 calls |
|---|---|---|---|---|---|---|
| phase1_vp | 51.7% | 61.4% | 51.1% | 54.2% | 47.3% | 30.8% |
| phase2_vp, prompt v2, 3 runs | 23–26% | 30–31% | 33–35% | 9–12% | 14–15% | 16.3–17.3% |

The rep's WER was unchanged (6.6% → 6.8%). English and Hindi calls are not touched.

**Gemini varies between runs.** Two runs of prompt v1 gave Telugu WER of 16% and 49%.
The words were mostly right, but in the bad run English was written in Telugu script
("కాస్ట్" for "cost"). With thinking off, one run also moved clip 2's words into clip 1.
Prompt v2 states each clip's duration, forbids moving words between clips, and gives a
script example. Three v2 runs then agreed within about a point, with no clip refused.
Thinking stays on: `thinking_budget=0` halved the cost but raised overall WER to 20.6%
(Telugu 61%).

Cost: about 3,800 input and 1,700 output+thinking tokens for a 142 s call, roughly $0.017.
Across the synthetic calls it averaged about $0.009 per call.

**Caveat.** The synthetic calls were written and voiced by Gemini, so Gemini transcribing
them may be flattered. The real Marathi call agrees, but it is one call. Labelled real
regional calls are what settles it.

## Phase 3 — tone, 2026-09-28

`tone_eval.py` scores tone on sentences whose words carry no tone, so reading the
transcript cannot beat chance:

- `make_tone_set.py`: 12 sales phrases × 7 tones × 3 voices, Gemini TTS (252 clips).
- `fetch_datasets.py cremad`: CREMA-D (ODbL), 20 real actors, 12 neutral sentences,
  5 emotions (200 clips). These map to anger→frustrated, happy→interested, fear→hesitant,
  sad→disengaged, neutral→neutral.

| | toneset (7 tones) | CREMA-D (5 emotions) |
|---|---|---|
| chance | 14% | 20% |
| text only (words) | 14% | 20% (everything "neutral") |
| prosody only (model-free, leave-one-out) | 46% | 45% |
| **Gemini listening, prompt v2** | 100% ⚠ | **48%** (48% on each half of the actors) |

**The toneset's 100% is not a real-world number.** It stayed at 100% after shuffling the
batches, so Gemini is very likely recognising its own acted TTS renditions. That set
proves only that the audio is being used. CREMA-D is the measure. Its authors report
human listeners at about 41% from the voice alone.

What moved it, and what did not:

- **Prompt v1 → v2: 44% → 48%.** v1 heard how intense a voice was but not whether it
  was positive or negative. Anger came back "urgent" (13/40) or "confident" (9/40). v2
  narrows "urgent" to time pressure and "confident" to calm. Frustrated recall rose from
  30% to 47%.
- **Measured delivery in the prompt: no gain** (48.5% without). It stays out of the
  prompt.
- **Measured delivery as a check.** When it contradicts Gemini's tone, Gemini was right
  23% of the time (10/44), against 53–57% otherwise. The pipeline lowers those to `low`
  confidence.
- **Weakest:** fear/nervousness heard as hesitant, 12%.

On a real 7.6-minute ScaleSerum call, the 12 moments cost about $0.02 and took 18 s.
They were plausible: an impatient customer "Yeah. Yeah. Tell me. I can hear you." heard as
frustrated, and a rep's "what do you mean by that?" heard as confused. That led to one
change: turns under four words are no longer chosen.

### Phase 2 follow-up — the rep code-switches too, 2026-09-29

The real Marathi call (`testaudio/mycall4.mp3`) showed the rep switching into Marathi, not
only the customer. The synthetic set had the rep in English only, so it could not show this.
Three calls where both sides code-switch were added (`both_mr_1`, `both_kn_1`, `both_pa_1`),
and the segment pass got a scope setting. `phase2_vp`, regional calls:

| Scope | Rep WER (both switch, n=3) | Customer WER | Rep WER (rep English, n=8) | Customer WER | cost/call |
|---|---|---|---|---|---|
| no pass | 33.2% | 49.0% | 6.5% | 53.7% | — |
| customer | 33.3% | 17.9% | 6.8% | 28.4% | ~$0.007 |
| mixed | 25.9% | 15.8% | 6.8% | 28.4% | ~$0.013 |
| **all** (default) | **12.3%** | 16.3% | 9.4% | 24.4% | ~$0.02 |

`mixed` falls short because Deepgram often garbles regional speech into Latin letters
("Tamurai" for "त्यामुळे"), which the Indian-script check cannot see.

Clips now read up to 1 s into the silence after a turn, stopping 0.25 s before the next
turn. Right up to the next turn, the clip caught its first syllable ("आप-"). Gemini now
gets the call's names too. On the real call that fixed "ScaleCRM" → ScaleSerum, with no
names forced onto other words.
