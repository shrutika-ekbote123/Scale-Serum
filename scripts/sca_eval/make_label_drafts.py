"""
Pre-fill a reference transcript for each real call, for a human to correct.

    python scripts/sca_eval/make_label_drafts.py            # every real call with a cached run
    python scripts/sca_eval/make_label_drafts.py --only <sales_call_id>

Writes workspace/labels/<id>.draft.json from the best cached Deepgram run
(phase0 if present). Open scripts/sca_eval/label_tool.html in a browser, load
the draft and its audio, correct it, and save as workspace/labels/<id>.json.

A DRAFT IS NOT A REFERENCE
    It is the machine's own output. Scoring the machine against it would
    measure nothing, so run_eval.py only reads files named <id>.json with
    "reviewed": true - which only the label tool writes, and only when the
    labeller ticks that they listened to the whole call.

WHAT A LABELLER CORRECTS
    1. Who is who: one role per diarized voice (rep / customer / participant),
       and splitting a turn where the machine merged two people.
    2. The words, in the script the speaker would write them: Devanagari for
       Hindi and Marathi, Latin for English words even mid-sentence.
    3. The language of each turn, and tone tags where they are clear.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _common import LABELS_DIR, REAL_DIR, RUNS_DIR, read_json, read_jsonl, write_json

from sales_call_analyzer import speakers as sp  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from sales_call_analyzer.models import CustomerInfo, RepInfo  # noqa: E402

ROLE_TO_LABEL = {"sales_rep": "rep", "customer": "customer", "participant": "participant"}


def _best_run(call_id: str):
    runs = sorted((RUNS_DIR / "raw" / "real" / call_id).glob("*.json"))
    if not runs:
        return None
    # Prefer a run that sent keyterms: that is the phase0 configuration.
    runs.sort(key=lambda p: "keyterm" not in read_json(p)["params"])
    return read_json(runs[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    written, seen = 0, set()
    # One draft per recording: the same audio sits on several sales_calls rows.
    # Rows whose CRM record has both names first, as in run_eval.py.
    items = sorted(read_jsonl(REAL_DIR / "manifest.jsonl"),
                   key=lambda r: not (r.get("rep_name") and r.get("customer_name")))
    for item in items:
        if args.only and item["id"] not in args.only:
            continue
        digest = hashlib.sha1((REAL_DIR / item["audio"]).read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        if (LABELS_DIR / f"{item['id']}.json").exists():
            continue            # never overwrite a human's work
        run = _best_run(item["id"])
        if not run:
            continue
        t = tr.from_deepgram(run["response"], merge_gap_seconds=0.0)
        t = sp.resolve_roles(t, rep=RepInfo(name=item.get("rep_name")),
                             customer=CustomerInfo(name=item.get("customer_name")),
                             brand_name=item.get("brand_name"))
        roles = {s.speaker_id: ROLE_TO_LABEL.get(s.role, "") for s in t.speakers}
        write_json(LABELS_DIR / f"{item['id']}.draft.json", {
            "id": item["id"], "audio": item["audio"], "reviewed": False,
            "brand_name": item.get("brand_name"), "rep_name": item.get("rep_name"),
            "customer_name": item.get("customer_name"),
            "draft_from": {"language": run["params"].get("language"),
                           "keyterms": bool(run["params"].get("keyterm"))},
            "speaker_roles": roles,
            "turns": [{"speaker": s.speaker_id, "role": roles.get(s.speaker_id, ""),
                       "lang": "", "tone": [], "text": s.text,
                       "start": s.start, "end": s.end} for s in t.segments],
        })
        written += 1
    print(f"{written} drafts in {LABELS_DIR}")
    print(f"label them with {Path(__file__).with_name('label_tool.html')}")


if __name__ == "__main__":
    main()
