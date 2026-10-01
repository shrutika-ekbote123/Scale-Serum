"""
Score Sarvam AI speech-to-text on the evaluation sets, against the same truth
and metrics as the current pipeline.

    python scripts/sca_eval/sarvam_eval.py                   # synthetic + fleurs
    python scripts/sca_eval/sarvam_eval.py --sets synthetic

Sarvam is run the way it was tested by hand: saaras:v3, mode codemix,
language auto-detected ("unknown"), diarization with 2 speakers, Batch API
(<=20 files per job). Responses are cached under workspace/runs/sarvam/.

Two Sarvam variants are scored:
  sarvam         the diarized transcript as returned
  sarvam_dedup   with its duplicated fragments removed - short entries whose
                 words also appear, at the same moment, in the other speaker's
                 entry (seen 5-26 times per call on the real test calls)

Sarvam returns chunk-level timestamps, not word-level ones, so for the speaker
metrics each entry's words are spread evenly across its time span.
Roles come from the analyzer's own speakers.resolve_roles, run on Sarvam's
output with the call's CRM names - the same evidence the pipeline gets.

Nothing here touches the analyzer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from _common import (FLEURS_DIR, REPORTS_DIR, RUNS_DIR, SYNTH_DIR, read_json, read_jsonl,
                     write_json)
import metrics as M

from sales_call_analyzer import speakers as sp
from sales_call_analyzer import transcript as tr
from sales_call_analyzer.models import CustomerInfo, RepInfo

CACHE = RUNS_DIR / "sarvam"
SETS = {"synthetic": SYNTH_DIR, "fleurs": FLEURS_DIR}


# --------------------------------------------------------------------------- #
# Running Sarvam
# --------------------------------------------------------------------------- #
def _cache_path(set_name, item_id):
    return CACHE / set_name / (str(item_id).replace("/", "__") + ".json")


def run_sarvam(set_name, items, speakers=2):
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
    todo = [x for x in items if not _cache_path(set_name, x["id"]).exists()]
    print(f"  {set_name}: {len(items) - len(todo)} cached, {len(todo)} to transcribe")
    out_dir = CACHE / set_name / "_download"
    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        job = client.speech_to_text_job.create_job(
            model="saaras:v3", mode="codemix", language_code="unknown",
            with_diarization=set_name != "fleurs", num_speakers=speakers)
        job.upload_files(file_paths=[x["path"] for x in batch])
        job.start()
        job.wait_until_complete(poll_interval=5, timeout=1800)
        out_dir.mkdir(parents=True, exist_ok=True)
        job.download_outputs(output_dir=str(out_dir))
        for x in batch:
            got = out_dir / (Path(x["path"]).name + ".json")
            if got.exists():
                target = _cache_path(set_name, x["id"])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(got.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                print(f"    no output for {x['id']}")
        print(f"    job {i // 20 + 1}: {len(batch)} files done")


# --------------------------------------------------------------------------- #
# Turning Sarvam output into what the metrics read
# --------------------------------------------------------------------------- #
_norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()  # noqa: E731


def entries(resp):
    return sorted((resp.get("diarized_transcript") or {}).get("entries") or [],
                  key=lambda e: e["start_time_seconds"])


def is_duplicate(e, all_entries):
    t = _norm(e["transcript"])
    return bool(t) and len(t.split()) <= 6 and any(
        o is not e and o["speaker_id"] != e["speaker_id"]
        and o["start_time_seconds"] <= e["start_time_seconds"] <= o["end_time_seconds"]
        and t in _norm(o["transcript"]) for o in all_entries)


def as_deepgram(ents):
    """A Deepgram-shaped response: words spread across each entry's span, and
    one utterance per entry - so transcript.py and metrics.py read it unchanged."""
    words, utterances = [], []
    for e in ents:
        toks = e["transcript"].split()
        a, b = float(e["start_time_seconds"]), float(e["end_time_seconds"])
        step = (b - a) / max(len(toks), 1)
        spk = int(e["speaker_id"]) if str(e["speaker_id"]).isdigit() else 0
        for k, tok in enumerate(toks):
            words.append({"word": tok, "punctuated_word": tok, "start": a + k * step,
                          "end": a + (k + 1) * step - 0.001, "speaker": spk, "confidence": 1.0})
        utterances.append({"speaker": spk, "start": a, "end": b, "transcript": e["transcript"],
                           "confidence": 1.0})
    return {"metadata": {"duration": max((e["end_time_seconds"] for e in ents), default=0)},
            "results": {"channels": [{"alternatives": [{"transcript": " ".join(e["transcript"] for e in ents),
                                                        "words": words}]}],
                        "utterances": utterances}}


def score_call(item, truth, ents):
    raw = as_deepgram(ents)
    t = tr.from_deepgram(raw)
    t = sp.resolve_roles(t, rep=RepInfo(name=item.get("rep_name")),
                         customer=CustomerInfo(name=item.get("customer_name")),
                         brand_name=item.get("brand_name"))
    roles = {s.speaker_id: s.role for s in t.speakers}
    turns = truth["turns"]
    placed = M.place_words(M.deepgram_words(raw), turns)
    return {"text": M.text_scores(" ".join(x["text"] for x in turns), " ".join(e["transcript"] for e in ents)),
            "text_by_role": M.per_role_text(placed, turns),
            "speakers": M.speaker_scores(placed, turns, roles),
            "language": truth.get("customer_language"), "pattern": truth.get("pattern"),
            "same_gender_voices": truth.get("same_gender_voices")}


# --------------------------------------------------------------------------- #
def synthetic(results):
    items = []
    for row in read_jsonl(SYNTH_DIR / "manifest.jsonl"):
        row["path"] = str(SYNTH_DIR / row["audio"])
        items.append(row)
    run_sarvam("synthetic", items)
    for row in items:
        path = _cache_path("synthetic", row["id"])
        if not path.exists():
            continue
        resp, truth = read_json(path), read_json(SYNTH_DIR / row["truth"])
        ents = entries(resp)
        dups = [e for e in ents if is_duplicate(e, ents)]
        for config, use in (("sarvam", ents), ("sarvam_dedup", [e for e in ents if e not in dups])):
            r = score_call(row, truth, use)
            r.update({"id": row["id"], "set": "synthetic", "config": config,
                      "language_detected": resp.get("language_code"), "duplicates": len(dups)})
            results.append(r)


def fleurs(results):
    items = []
    for row in read_jsonl(FLEURS_DIR / "manifest.jsonl"):
        row["path"] = str(FLEURS_DIR / row["audio"])
        items.append(row)
    run_sarvam("fleurs", items)
    for row in items:
        path = _cache_path("fleurs", row["id"])
        if not path.exists():
            continue
        resp = read_json(path)
        results.append({"id": row["id"], "set": "fleurs", "config": "sarvam", "language": row["language"],
                        "language_detected": resp.get("language_code"),
                        "text": M.text_scores(row.get("reference") or "", resp.get("transcript") or "")})


def current_pipeline(which):
    """The newest run_eval report holding each config."""
    wanted = {"synthetic": ("phase0", "phase2"), "fleurs": ("phase0",)}
    found = {}
    for path in sorted(REPORTS_DIR.glob("eval-*.json"), reverse=True):
        rep = read_json(path)
        for r in rep.get("results", []):
            key = (r["set"], r["config"])
            if r["set"] in which and r["config"] in wanted.get(r["set"], ()) and key not in found:
                found[key] = path.name
        if all((s, c) in found for s in which for c in wanted[s]):
            break
    out = []
    for (s, c), name in found.items():
        out += [r for r in read_json(REPORTS_DIR / name)["results"] if r["set"] == s and r["config"] == c]
    return out, found


def summarise(results):
    table = {}
    syn = defaultdict(list)
    for r in results:
        if r["set"] == "synthetic" and "speakers" in r:
            syn[r["config"]].append(r)
    if syn:
        print("\nSYNTHETIC (19 calls)   rep WER | cust WER | all WER | role acc | diar acc | backchannel | correct/swapped")
    for config in ("phase0", "phase2", "sarvam", "sarvam_dedup"):
        rows = syn.get(config) or []
        if not rows:
            continue
        rep, cust, allc = {}, {}, {}
        for r in rows:
            M.add_counts(rep, (r["text_by_role"].get("rep") or {}).get("roman") or {})
            M.add_counts(cust, (r["text_by_role"].get("customer") or {}).get("roman") or {})
            M.add_counts(allc, r["text"]["roman"])
        words = sum(r["speakers"]["words_scored"] for r in rows)
        role = sum(r["speakers"]["role_accuracy"] * r["speakers"]["words_scored"] for r in rows) / max(words, 1)
        diar = sum(r["speakers"]["diarization_accuracy"] * r["speakers"]["words_scored"] for r in rows) / max(words, 1)
        bc = M.summarise(r["speakers"].get("backchannel_recall") for r in rows).get("mean")
        out = Counter(r["speakers"]["role_outcome"] for r in rows)
        table[config] = {"n": len(rows), "rep_wer": M.rates(rep)["wer"], "customer_wer": M.rates(cust)["wer"],
                         "all_wer": M.rates(allc)["wer"], "role_accuracy": round(role, 4),
                         "diarization_accuracy": round(diar, 4), "backchannel": bc, "outcomes": dict(out)}
        print(f"  {config:13} {M.rates(rep)['wer']:.3f}   {M.rates(cust)['wer']:.3f}    {M.rates(allc)['wer']:.3f}   "
              f"{role:.3f}    {diar:.3f}     {bc if bc is None else round(bc, 3)}      {dict(out)}")

    by_lang = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r["set"] == "synthetic" and "text_by_role" in r and r["config"] in ("phase2", "sarvam", "sarvam_dedup"):
            M.add_counts(by_lang[r["language"]][r["config"] + "_rep"], (r["text_by_role"].get("rep") or {}).get("roman") or {})
            M.add_counts(by_lang[r["language"]][r["config"] + "_cust"], (r["text_by_role"].get("customer") or {}).get("roman") or {})
    if by_lang:
        print("\nSYNTHETIC by customer language (roman WER)   current(phase2) rep/cust | sarvam_dedup rep/cust")
        for lang in sorted(by_lang):
            d = by_lang[lang]
            f = lambda k: M.rates(d[k])["wer"] if d.get(k) else None  # noqa: E731
            print(f"  {lang}   {f('phase2_rep')} / {f('phase2_cust')}   |   {f('sarvam_dedup_rep')} / {f('sarvam_dedup_cust')}")
            table.setdefault("by_language", {})[lang] = {k: f(k) for k in d}

    fl = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r["set"] == "fleurs":
            M.add_counts(fl[r["language"]][r["config"]], r["text"]["roman"])
            M.add_counts(fl[r["language"]][r["config"] + "_native"], r["text"]["native"])
    if fl:
        print("\nFLEURS, real human speech (WER native / roman)   current(phase0) | sarvam")
        for lang in sorted(fl):
            d = fl[lang]
            g = lambda k: M.rates(d[k])["wer"] if d.get(k) else None  # noqa: E731
            print(f"  {lang}   {g('phase0_native')} / {g('phase0')}   |   {g('sarvam_native')} / {g('sarvam')}")
            table.setdefault("fleurs", {})[lang] = {"current_native": g("phase0_native"), "current_roman": g("phase0"),
                                                    "sarvam_native": g("sarvam_native"), "sarvam_roman": g("sarvam")}
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sets", nargs="+", default=["synthetic", "fleurs"], choices=list(SETS))
    args = parser.parse_args()
    started = time.time()
    results = []
    if "synthetic" in args.sets:
        synthetic(results)
    if "fleurs" in args.sets:
        fleurs(results)
    current, sources = current_pipeline(args.sets)
    print(f"current-pipeline results from: {sources}")
    table = summarise(results + current)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    write_json(REPORTS_DIR / f"sarvam-vs-current-{stamp}.json",
               {"summary": table, "sarvam_results": results, "current_sources": {f"{k[0]}/{k[1]}": v for k, v in sources.items()}})
    print(f"\n{len(results)} Sarvam results in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
