"""
Test the Sarvam clean-up (transcription/sarvam_cleanup.py) and the optional
Gemini re-check (transcription/sarvam_recheck.py) - outside the analyzer.

    python scripts/sca_eval/sarvam_cleanup_test.py                 # testaudio calls, free steps only
    python scripts/sca_eval/sarvam_cleanup_test.py --recheck       # + Gemini on the flagged turns
    python scripts/sca_eval/sarvam_cleanup_test.py --synthetic     # regression check on the 19 calls with truth
    python scripts/sca_eval/sarvam_cleanup_test.py --calls mycall8.mp3 --recheck

The testaudio calls: Sarvam's response is cached under
workspace/runs/sarvam/testaudio/ (one Sarvam job per missing call, ~Rs 0.75/min).
For each call this writes to workspace/sarvam_cleanup/:
    <call>.before.txt   Sarvam as returned (roles from the analyzer's resolve_roles)
    <call>.after.txt    after the clean-up (and the re-check, with --recheck)
    <call>.report.json  every change made, every flag, tokens used
and prints a summary. Nothing here touches the analyzer: its role resolver is
only called to label speakers in the text files.

--synthetic scores Sarvam before and after the clean-up against exact truth
(same metrics as sarvam_eval.py). The clean-up must not make any number worse.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from _common import (REPO, SYNTH_DIR, WORKSPACE, decode_pcm, ffprobe_path, read_json, read_jsonl,
                     write_json)
import metrics as M
import sarvam_eval as SE

from sales_call_analyzer import speakers as sp
from sales_call_analyzer import transcript as tr
from sales_call_analyzer.models import CustomerInfo, RepInfo
from transcription import sarvam_cleanup as sc
from transcription import sarvam_client
from transcription import sarvam_recheck as rc

AUDIO_DIR = REPO / "testaudio"
CACHE = WORKSPACE / "runs" / "sarvam" / "testaudio"
OUT = WORKSPACE / "sarvam_cleanup"

# What the CRM / Brand Brain would supply. Names are the ones actually said on
# each recording (the CRM rows attached to them carry placeholders).
LAWTORNEY_VOCAB = ["ChatGPT", "trial version", "refund", "prompt book", "our tool"]
CALLS = {
    "mycall.mp3": {"terms": ["Directors Institute"], "rep": "Devansh", "customer": "Sanjay"},
    "mycall1.mp3": {"terms": ["WDC"], "rep": "Shrutika", "customer": "Geeta"},
    "mycall2.mp3": {"terms": ["ABC Educational Institute"]},
    "mycall3.mp3": {"terms": ["ScaleSerum"], "rep": "Aniket"},
    "mycall4.mp3": {"terms": ["ScaleSerum"], "rep": "Rahul", "customer": "Amit"},
    "mycall5.mp3": {"terms": ["ABC Educational Institute"]},
    "mycall6.wav": {"terms": []},
    "mycall7.mp3": {"terms": ["Lawtorney", "Lawtorney AI"], "vocabulary": LAWTORNEY_VOCAB, "rep": "Shruti"},
    "mycall8.mp3": {"terms": ["Lawtorney", "Lawtorney AI"], "vocabulary": LAWTORNEY_VOCAB,
                    "rep": "Shruti", "customer": "Rupesh"},
    "mycall9.mp3": {"terms": ["Lawtorney", "Lawtorney AI"], "vocabulary": LAWTORNEY_VOCAB},
}
GEMINI_INR_PER_M_INPUT = 0.75 * 88
GEMINI_INR_PER_M_OUTPUT = 3.75 * 88


# --------------------------------------------------------------------------- #
def sarvam_response(name: str, fresh: bool = False) -> tuple[dict, Optional[int]]:
    """Sarvam's response for a call, and how long the job took (None if cached).
    `fresh` sends the call to Sarvam again through transcription/sarvam_client.py;
    the previous cached response is kept as <name>.prev.json."""
    path = CACHE / f"{name}.json"
    if path.exists() and not fresh:
        return read_json(path), None
    print(f"  {name}: running Sarvam ...", flush=True)
    result = asyncio.run(sarvam_client.transcribe((AUDIO_DIR / name).read_bytes(), filename=name))
    if not result.ok:
        raise SystemExit(f"Sarvam failed for {name}: {result.reason} {result.error or ''}")
    if path.exists():
        path.replace(CACHE / f"{name}.prev.json")
    write_json(path, result.response)
    return result.response, result.ms


def audio_seconds(name: str) -> float:
    out = subprocess.run([ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(AUDIO_DIR / name)], capture_output=True, text=True)
    return float(out.stdout.strip())


def render(entries: list[sc.Entry], cfg: dict, title: str) -> str:
    t = tr.from_deepgram(sc.to_deepgram_shape(entries))
    t = sp.resolve_roles(t, rep=RepInfo(name=cfg.get("rep")), customer=CustomerInfo(name=cfg.get("customer")),
                         brand_name=(cfg["terms"] or [None])[0])
    label = {"sales_rep": "Rep", "customer": "Customer"}
    roles = {s.speaker_id: label.get(s.role, s.role) for s in t.speakers}
    lines = [title]
    for e in entries:
        who = f"{roles.get(f'speaker_{e.speaker_id}', '?')} ({e.speaker_id})"
        lines.append(f"[{e.start:6.1f}-{e.end:6.1f}] {who}: {e.text}")
    return "\n".join(lines) + "\n"


async def recheck_audio(path: Path, result: sc.CleanupResult, terms: list[str],
                        vocabulary: list[str]) -> rc.RecheckResult:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    samples = decode_pcm(path)[:, 0]
    return await rc.recheck(client, os.environ.get("GEMINI_MODEL", "gemini-flash-latest"), samples,
                            result.entries, result.suspect_indices, terms + vocabulary, brand_terms=terms)


def testaudio(names: list[str], do_recheck: bool, fresh: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, costs, combined = [], [], []
    for name in names:
        cfg = CALLS[name]
        response, sarvam_ms = sarvam_response(name, fresh)
        seconds = audio_seconds(name)
        before = sc.entries_from_sarvam(response)
        result = sc.clean(response, cfg["terms"], cfg.get("vocabulary"))
        report = result.report()
        recheck = None
        if do_recheck and result.suspect_indices:
            recheck = asyncio.run(recheck_audio(AUDIO_DIR / name, result, cfg["terms"], cfg.get("vocabulary", [])))
            report["recheck"] = recheck.report()
        stem = Path(name).stem
        lang = response.get("language_code")
        (OUT / f"{stem}.before.txt").write_text(
            render(before, cfg, f"{name} - Sarvam as returned - {lang}"), encoding="utf-8")
        (OUT / f"{stem}.after.txt").write_text(
            render(result.entries, cfg, f"{name} - after clean-up{' + re-check' if recheck else ''} - {lang}"),
            encoding="utf-8")
        c = report["counts"]
        cost = None
        if recheck and recheck.input_tokens is not None:
            cost = ((recheck.input_tokens or 0) * GEMINI_INR_PER_M_INPUT
                    + ((recheck.output_tokens or 0) + (recheck.thinking_tokens or 0)) * GEMINI_INR_PER_M_OUTPUT) / 1e6
        sarvam_inr = sarvam_client.estimated_cost_inr(seconds)
        costs.append({"call": name, "language": lang, "minutes": round(seconds / 60, 2),
                      "sarvam_seconds_to_transcribe": round(sarvam_ms / 1000) if sarvam_ms else None,
                      "sarvam_inr": round(sarvam_inr, 2), "gemini_inr": round(cost or 0.0, 2),
                      "gemini_tokens": {"input": recheck.input_tokens, "output": recheck.output_tokens,
                                        "thinking": recheck.thinking_tokens} if recheck else None,
                      "total_inr": round(sarvam_inr + (cost or 0.0), 2)})
        report["cost"] = costs[-1]
        write_json(OUT / f"{stem}.report.json", report)
        combined.append((OUT / f"{stem}.after.txt").read_text(encoding="utf-8"))
        rows.append((name, lang, c["duplicates_removed"], c["script_fixes"], c["term_fixes"],
                     c["suspect_entries"], len(recheck.changes) if recheck else "-",
                     len(recheck.rejected) if recheck else "-", f"{cost:.2f}" if cost is not None else "-"))
        print(f"\n== {name} ({lang})")
        for d in result.duplicates_removed:
            print(f"   echo removed   [{d['start']:6.1f}] spk {d['speaker_id']}: {d['text']}")
        for d in result.script_fixes:
            print(f"   script         {d['before']!r} -> {d['after']!r}")
        for d in result.term_fixes:
            print(f"   term ({d['tier']:7}) {d['heard']!r} -> {d['written']!r}")
        for d in result.suspects:
            print(f"   flagged        {d['heard']!r} ~ {d['term']}")
        if recheck:
            for d in recheck.changes:
                print(f"   re-check       [{d['start']:6.1f}] {d['before']!r}\n                  -> {d['after']!r}")
            for i, why in recheck.rejected.items():
                print(f"   re-check refused for entry {i}: {why}")
            if recheck.reason:
                print(f"   re-check: {recheck.reason}")
    print("\ncall          lang    echoes  script  terms  flagged  rechecked  refused  Gemini Rs")
    for r in rows:
        print(f"{r[0]:13} {str(r[1]):7} {r[2]:6}  {r[3]:6}  {r[4]:5}  {r[5]:7}  {str(r[6]):9}  {str(r[7]):7}  {r[8]}")
    print("\ncall          minutes  Sarvam Rs  Gemini Rs  total Rs  Sarvam job s")
    for c in costs:
        print(f"{c['call']:13} {c['minutes']:7.2f}  {c['sarvam_inr']:9.2f}  {c['gemini_inr']:9.2f}  "
              f"{c['total_inr']:8.2f}  {c['sarvam_seconds_to_transcribe'] or '-'}")
    total = {k: round(sum(c[k] for c in costs), 2) for k in ("minutes", "sarvam_inr", "gemini_inr", "total_inr")}
    print(f"{'TOTAL':13} {total['minutes']:7.2f}  {total['sarvam_inr']:9.2f}  {total['gemini_inr']:9.2f}  "
          f"{total['total_inr']:8.2f}")
    write_json(OUT / "costs.json", {"calls": costs, "total": total, "rates": {
        "sarvam_inr_per_hour_diarized": sarvam_client.INR_PER_HOUR_DIARIZED,
        "gemini_inr_per_m_input": GEMINI_INR_PER_M_INPUT,
        "gemini_inr_per_m_output_and_thinking": GEMINI_INR_PER_M_OUTPUT}})
    (OUT / "all_transcripts.txt").write_text(("\n" + "=" * 100 + "\n").join(combined), encoding="utf-8")
    print(f"\nwritten to {OUT}")


# --------------------------------------------------------------------------- #
def as_sarvam_entries(entries: list[sc.Entry]) -> list[dict]:
    return [{"transcript": e.text, "start_time_seconds": e.start, "end_time_seconds": e.end,
             "speaker_id": e.speaker_id} for e in entries]


def synthetic(do_recheck: bool) -> None:
    results = {"sarvam": [], "sarvam_clean": []}
    if do_recheck:
        results["sarvam_clean_recheck"] = []
    changes = Counter()
    tokens = Counter()
    for row in read_jsonl(SYNTH_DIR / "manifest.jsonl"):
        path = SE._cache_path("synthetic", row["id"])
        if not path.exists():
            continue
        resp, truth = read_json(path), read_json(SYNTH_DIR / row["truth"])
        results["sarvam"].append(SE.score_call(row, truth, SE.entries(resp)))
        cleaned = sc.clean(resp, [row["brand_name"]])
        for k, v in cleaned.report()["counts"].items():
            changes[k] += v
        for d in cleaned.term_fixes:
            print(f"  {row['id']:14} term {d['heard']!r} -> {d['written']!r}")
        for d in cleaned.duplicates_removed:
            print(f"  {row['id']:14} echo removed {d['text']!r}")
        results["sarvam_clean"].append(SE.score_call(row, truth, as_sarvam_entries(cleaned.entries)))
        if do_recheck:
            if cleaned.suspect_indices:
                done = asyncio.run(recheck_audio(SYNTH_DIR / row["audio"], cleaned, [row["brand_name"]], []))
                for d in done.changes:
                    print(f"  {row['id']:14} re-check {d['before']!r}\n  {'':14}       -> {d['after']!r}")
                for i, why in done.rejected.items():
                    print(f"  {row['id']:14} re-check refused for entry {i}: {why}")
                changes["rechecked"] += len(done.changes)
                changes["recheck_refused"] += len(done.rejected)
                tokens["input"] += done.input_tokens or 0
                tokens["output"] += (done.output_tokens or 0) + (done.thinking_tokens or 0)
            results["sarvam_clean_recheck"].append(
                SE.score_call(row, truth, as_sarvam_entries(cleaned.entries)))
    print(f"\nSYNTHETIC ({len(results['sarvam'])} calls)   rep WER | cust WER | all WER | role acc | diar acc | backchannel")
    table = {}
    for config, rows in results.items():
        rep, cust, allc = {}, {}, {}
        for r in rows:
            M.add_counts(rep, (r["text_by_role"].get("rep") or {}).get("roman") or {})
            M.add_counts(cust, (r["text_by_role"].get("customer") or {}).get("roman") or {})
            M.add_counts(allc, r["text"]["roman"])
        words = sum(r["speakers"]["words_scored"] for r in rows)
        role = sum(r["speakers"]["role_accuracy"] * r["speakers"]["words_scored"] for r in rows) / max(words, 1)
        diar = sum(r["speakers"]["diarization_accuracy"] * r["speakers"]["words_scored"] for r in rows) / max(words, 1)
        bc = M.summarise(r["speakers"].get("backchannel_recall") for r in rows).get("mean")
        table[config] = {"rep_wer": M.rates(rep)["wer"], "customer_wer": M.rates(cust)["wer"],
                         "all_wer": M.rates(allc)["wer"], "role_accuracy": round(role, 4),
                         "diarization_accuracy": round(diar, 4), "backchannel": bc}
        print(f"  {config:13} {M.rates(rep)['wer']:.4f}  {M.rates(cust)['wer']:.4f}   {M.rates(allc)['wer']:.4f}  "
              f"{role:.4f}   {diar:.4f}   {bc:.4f}")
    print(f"  changes made: {dict(changes)}")
    if tokens:
        inr = (tokens["input"] * GEMINI_INR_PER_M_INPUT + tokens["output"] * GEMINI_INR_PER_M_OUTPUT) / 1e6
        print(f"  Gemini re-check: {dict(tokens)} tokens, about Rs {inr:.2f} for all calls")
    write_json(OUT / "synthetic_regression.json", {"summary": table, "changes": dict(changes),
                                                   "recheck_tokens": dict(tokens)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calls", nargs="+", default=list(CALLS), choices=list(CALLS))
    parser.add_argument("--recheck", action="store_true", help="let Gemini re-check the flagged turns")
    parser.add_argument("--fresh", action="store_true",
                        help="send every call to Sarvam again (~Rs 0.75/min) instead of using the cache")
    parser.add_argument("--synthetic", action="store_true",
                        help="regression check on the synthetic set (with --recheck: Gemini too)")
    args = parser.parse_args()
    started = time.time()
    if args.synthetic:
        OUT.mkdir(parents=True, exist_ok=True)
        synthetic(args.recheck)
    else:
        testaudio(args.calls, args.recheck, args.fresh)
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
