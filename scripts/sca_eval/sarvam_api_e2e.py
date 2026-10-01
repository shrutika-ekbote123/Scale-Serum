"""
End-to-end: the testaudio calls through the running analyzer API with
SCA_TRANSCRIBER=sarvam, exactly as the backend (or Postman) would send them.

    python -m http.server 8766 --bind 127.0.0.1 --directory testaudio   # audio
    uvicorn app:app --host 127.0.0.1 --port 3001                       # analyzer
    python scripts/sca_eval/sarvam_api_e2e.py [--calls mycall8.mp3 ...]

Each call gets a fresh call_id. Writes to workspace/sarvam_e2e/:
    <call>.txt            the transcript the analysis used, Rep / Customer labelled
    all_transcripts.txt   all of them
    costs.json            per call: Sarvam Rs, Gemini Rs (re-check / analysis / other)
and prints a summary. Costs a real Sarvam job and a real analysis per call.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx
from _common import WORKSPACE, write_json

API = os.environ.get("SCA_E2E_API", "http://127.0.0.1:3001")
AUDIO = os.environ.get("SCA_E2E_AUDIO", "http://127.0.0.1:8766")
OUT = WORKSPACE / "sarvam_e2e"
KEY = os.environ.get("API_KEY", "")
GEMINI_INR_PER_M_INPUT = 0.75 * 88
GEMINI_INR_PER_M_OUTPUT = 3.75 * 88

LAW_TERMS = ["ChatGPT", "trial version", "refund", "prompt book", "our tool"]
# What the CRM would send. Names are the ones said on each recording (the CRM
# rows attached to these test calls carry placeholders).
CALLS = {
    "mycall.mp3": {"brand_name": "Directors Institute", "rep": {"name": "Devansh"},
                   "customer": {"name": "Sanjay"}},
    "mycall1.mp3": {"brand_name": "WDC", "rep": {"name": "Shrutika"}, "customer": {"name": "Geeta"}},
    "mycall2.mp3": {"brand_name": "ABC Educational Institute"},
    "mycall3.mp3": {"brand_name": "ScaleSerum", "rep": {"name": "Aniket"}},
    "mycall4.mp3": {"brand_name": "ScaleSerum", "rep": {"name": "Rahul"}, "customer": {"name": "Amit"}},
    "mycall5.mp3": {"brand_name": "ABC Educational Institute"},
    "mycall6.wav": {"customer": {"name": "Shubhojit Mondal"}},
    "mycall7.mp3": {"brand_name": "Lawtorney", "rep": {"id": "rep_shruti", "name": "Shruti"},
                    "product": {"name": "Lawtorney AI", "terms": LAW_TERMS}},
    "mycall8.mp3": {"brand_name": "Lawtorney", "rep": {"id": "rep_shruti", "name": "Shruti"},
                    "customer": {"name": "Rupesh"},
                    "product": {"name": "Lawtorney AI", "terms": LAW_TERMS}},
    "mycall9.mp3": {"brand_name": "Lawtorney", "rep": {"id": "rep_shruti", "name": "Shruti"},
                    "product": {"name": "Lawtorney AI", "terms": LAW_TERMS}},
}
TERMINAL = {"completed", "failed", "skipped"}


def inr(tokens_in, tokens_out) -> float:
    return ((tokens_in or 0) * GEMINI_INR_PER_M_INPUT + (tokens_out or 0) * GEMINI_INR_PER_M_OUTPUT) / 1e6


def submit(http: httpx.Client, name: str, stamp: str) -> str:
    cfg = CALLS[name]
    body = {"call_id": f"sarvam_e2e_{stamp}_{Path(name).stem}",
            "audio": {"url": f"{AUDIO}/{name}",
                      "mime_type": "audio/wav" if name.endswith(".wav") else "audio/mpeg"},
            "call_metadata": {"direction": "outbound"},
            "options": {"force_reanalysis": True},
            **{k: v for k, v in cfg.items()}}
    r = http.post(f"{API}/api/sales-calls/analyze", json=body)
    r.raise_for_status()
    return r.json()["analysis_id"]


def wait(http: httpx.Client, ids: dict[str, str]) -> dict[str, dict]:
    done: dict[str, dict] = {}
    while len(done) < len(ids):
        for name, aid in ids.items():
            if name in done:
                continue
            r = http.get(f"{API}/api/sales-calls/analysis/{aid}")
            if r.status_code == 200 and r.json().get("status") in TERMINAL:
                done[name] = r.json()
                print(f"  {name}: {done[name]['status']}", flush=True)
        if len(done) < len(ids):
            time.sleep(5)
    return done


def _db():
    from pymongo import MongoClient
    return MongoClient(os.environ["MONGODB_URI"])[os.environ.get("MONGODB_DB", "scaleserum")]


def latest(name: str) -> dict:
    """The newest analysis of this call from any e2e run: completed if there is
    one, otherwise the newest attempt (so a failure is reported, not hidden)."""
    stem = Path(name).stem
    query = {"call_id": {"$regex": rf"^sarvam_e2e_\d+_{stem}$"}}
    col = _db().sales_call_analyses
    return (col.find_one({**query, "status": "completed"}, sort=[("created_at", -1)])
            or col.find_one(query, sort=[("created_at", -1)]) or {})


def render(name: str, doc: dict) -> str:
    """Lines labelled Speaker 1, Speaker 2, ... in the order they first speak.
    The role each speaker was given (and why) is in the header."""
    t = doc.get("transcript") or {}
    p = doc.get("processing") or {}
    label = {"sales_rep": "Rep", "customer": "Customer", "participant": "Participant"}
    roles = {s["speaker_id"]: (label.get(s.get("role"), "role unknown"), s.get("role_basis"))
             for s in t.get("speakers") or []}
    number: dict[str, int] = {}
    for seg in t.get("segments") or []:
        number.setdefault(seg["speaker_id"], len(number) + 1)
    sv = p.get("sarvam") or {}
    head = (f"{name} - transcribed by {p.get('transcription_provider')} "
            f"({p.get('transcription_model')}), language {sv.get('language_code')}, "
            f"status {doc.get('status')}")
    who = "; ".join(f"Speaker {n} = {roles.get(sid, ('role unknown', None))[0]}"
                    f" [{roles.get(sid, (None, 'unresolved'))[1]}]"
                    for sid, n in sorted(number.items(), key=lambda kv: kv[1]))
    lines = [head, f"speakers: {who}"]
    for seg in t.get("segments") or []:
        lines.append(f"[{seg['start']:6.1f}-{seg['end']:6.1f}] Speaker {number[seg['speaker_id']]}: "
                     f"{seg['text']}")
    return "\n".join(lines) + "\n"


def costs(name: str, doc: dict) -> dict:
    p = doc.get("processing") or {}
    cost = p.get("cost") or {}
    sv = cost.get("sarvam") or {}
    recheck = inr(p.get("sarvam_recheck_input_tokens"),
                  (p.get("sarvam_recheck_output_tokens") or 0) + (p.get("sarvam_recheck_thinking_tokens") or 0))
    analysis = inr(p.get("llm_input_tokens"),
                   (p.get("llm_output_tokens") or 0) + (p.get("llm_thinking_tokens") or 0))
    gemini_total = round((cost.get("gemini") or {}).get("usd", 0) * (cost.get("usd_to_inr") or 88), 2)
    return {"call": name, "status": doc.get("status"),
            "minutes": round((p.get("audio_seconds_submitted") or 0) / 60, 2),
            "provider": p.get("transcription_provider"),
            "sarvam_inr": round(sv.get("inr") or 0, 2),
            "deepgram_inr": round(((cost.get("deepgram") or {}).get("usd") or 0) * 88, 2),
            "gemini_recheck_inr": round(recheck, 2),
            "gemini_analysis_inr": round(analysis, 2),
            "gemini_other_inr": round(gemini_total - round(recheck, 2) - round(analysis, 2), 2),
            "gemini_inr": gemini_total,
            "total_inr": cost.get("total_inr"),
            "seconds_end_to_end": round((p.get("total_ms") or 0) / 1000),
            "cleanup": ((p.get("sarvam") or {}).get("cleanup") or {}).get("counts"),
            "recheck_changes": len(((p.get("sarvam") or {}).get("recheck") or {}).get("changes") or []),
            "sarvam_reason": (p.get("sarvam") or {}).get("reason")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calls", nargs="+", default=list(CALLS), choices=list(CALLS),
                        help="calls to (re)submit")
    parser.add_argument("--collect-only", action="store_true",
                        help="submit nothing; report the latest analysis of every call")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.collect_only:
        stamp = time.strftime("%Y%m%d%H%M%S")
        with httpx.Client(headers={"X-API-Key": KEY}, timeout=60) as http:
            ids = {name: submit(http, name, stamp) for name in args.calls}
            print(f"submitted {len(ids)} calls; waiting", flush=True)
            wait(http, ids)
    # The report always covers every call: the latest analysis of each.
    rows, texts = [], []
    for name in CALLS:
        doc = latest(name)
        if not doc:
            continue
        text = render(name, doc)
        (OUT / f"{Path(name).stem}.txt").write_text(text, encoding="utf-8")
        texts.append(text)
        rows.append({**costs(name, doc), "analysis_id": doc["_id"], "reason": doc.get("reason")})
    (OUT / "all_transcripts.txt").write_text(("\n" + "=" * 100 + "\n").join(texts), encoding="utf-8")
    total = {k: round(sum(r[k] or 0 for r in rows), 2)
             for k in ("minutes", "sarvam_inr", "gemini_recheck_inr", "gemini_analysis_inr",
                       "gemini_other_inr", "gemini_inr", "total_inr")}
    write_json(OUT / "costs.json", {"calls": rows, "total": total})
    print("\ncall          status     min   Sarvam  Gemini(recheck/analysis/other)   Gemini   Total")
    for r in rows:
        print(f"{r['call']:13} {r['status']:9} {r['minutes']:5.2f}  {r['sarvam_inr']:6.2f}  "
              f"{r['gemini_recheck_inr']:5.2f} / {r['gemini_analysis_inr']:5.2f} / {r['gemini_other_inr']:5.2f}"
              f"        {r['gemini_inr']:6.2f}  {r['total_inr'] or 0:6.2f}  {r['reason'] or ''}")
    print(f"{'TOTAL':13} {'':9} {total['minutes']:5.2f}  {total['sarvam_inr']:6.2f}  "
          f"{total['gemini_recheck_inr']:5.2f} / {total['gemini_analysis_inr']:5.2f} / "
          f"{total['gemini_other_inr']:5.2f}        {total['gemini_inr']:6.2f}  {total['total_inr']:6.2f}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
