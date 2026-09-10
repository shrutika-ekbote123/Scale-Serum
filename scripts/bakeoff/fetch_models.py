"""
Step 8, part 1 - fetch the candidate models and read their licences.

    python scripts/bakeoff/fetch_models.py
    python scripts/bakeoff/fetch_models.py --only unisal,msinet
    python scripts/bakeoff/fetch_models.py --licences-only    (no clone, just report)

THIS IS THE GATE
    Run this BEFORE downloading 5 GB of datasets or spending an afternoon on
    annotation. Several published saliency checkpoints are research-only or
    non-commercial. ScaleSerum is a commercial product, so a model that fails
    the licence check is out of the running however well it scores - and finding
    that out first costs fifteen minutes instead of a day.

WHAT IT DOES NOT DO
    It does not decide. It reads the LICENSE file each repository actually ships
    and puts the text in front of you, flagging the phrases that usually mean
    "not for commercial use". Accepting a licence is a business decision, and
    this script deliberately stops short of making it.

    It also does not fetch weights that live behind Google Drive. Those need a
    human, and the report says which ones and where to look, rather than
    appearing to have succeeded.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from candidates import CANDIDATES, CENTRE_BASELINE, CURRENT_BASELINE, MANUAL, PIP  # noqa: E402
import provenance as prov  # noqa: E402

WORKSPACE = os.environ.get("VL_BAKEOFF_DIR", os.path.join(HERE, "workspace"))
CLONES = os.path.join(WORKSPACE, "models")
REPORT = os.path.join(REPO, "vision_lab", "BAKEOFF_LICENCES.md")
NOTICE = os.path.join(REPO, "NOTICE.md")

LICENCE_FILENAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE",
                     "LICENCE.md", "LICENCE.txt", "COPYING", "COPYING.txt")

# Phrases that mean "stop and read this properly". Deliberately broad: a false
# alarm costs a minute of reading, a missed one costs a licence violation.
RESTRICTIVE = (
    "non-commercial", "noncommercial", "not for commercial", "research purposes only",
    "research only", "academic purposes only", "academic use only",
    "personal and non-commercial", "cc by-nc", "creativecommons.org/licenses/by-nc",
    "may not be used for commercial", "educational purposes only",
)

# Well-known licences, identified by wording only they use.
#
# Order matters - the first match wins, so the more specific variant is listed
# before the family it belongs to (BSD-3 before BSD-2, NC before plain CC-BY).
#
# Each entry lists the phrases that must ALL be present. A licence identified by
# a single distinctive phrase gets a one-item tuple; requiring several
# alternatives to co-occur is how MIT-licensed repos end up reported as
# "unrecognised" and sent for pointless manual review.
KNOWN = [
    ("CC-BY-NC-4.0", ("creativecommons.org/licenses/by-nc",)),
    ("CC-BY-4.0", ("creativecommons.org/licenses/by/4.0",)),
    ("Apache-2.0", ("apache license",)),
    ("GPL-3.0", ("gnu general public license",)),
    ("MIT", ("permission is hereby granted, free of charge",)),
    ("BSD-3-Clause", ("redistributions of source code must retain",
                      "neither the name")),
    ("BSD-2-Clause", ("redistributions of source code must retain",)),
]


def run(args: list[str], cwd: str = None, timeout: int = 300):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, check=False)


def clone(candidate: dict) -> tuple[str, str]:
    """Shallow-clone the repo. Returns (path, status)."""
    target = os.path.join(CLONES, candidate["id"])
    if os.path.isdir(os.path.join(target, ".git")):
        return target, "already cloned"
    if not candidate.get("repo"):
        return "", "no repository"

    os.makedirs(CLONES, exist_ok=True)
    # Depth 1: we want the licence and the code, not the history. Retried because
    # a transient DNS blip otherwise reads as "this model is unavailable", which
    # is a far stronger claim than the evidence supports.
    detail = "unknown error"
    for attempt in range(3):
        result = run(["git", "clone", "--depth", "1", candidate["repo"], target],
                     timeout=600)
        if result.returncode == 0:
            return target, "cloned"
        shutil.rmtree(target, ignore_errors=True)
        lines = (result.stderr or "").strip().splitlines()
        detail = lines[-1] if lines else "unknown error"
        transient = any(sign in detail.lower() for sign in
                        ("could not resolve", "failed to connect", "timed out",
                         "connection reset", "unable to access"))
        if not transient:
            break
        if attempt < 2:
            time.sleep(3 * (attempt + 1))
    return "", f"clone failed: {detail}"


def find_licence(path: str) -> tuple[str, str]:
    """(filename, text). Falls back to a licence section in the README."""
    if not path:
        return "", ""
    for name in LICENCE_FILENAMES:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8", errors="replace") as handle:
                return name, handle.read()

    # Plenty of research repos state their terms in the README and ship no
    # LICENSE file at all. No licence file is itself a finding: with no licence,
    # default copyright applies and the answer is "no".
    for name in ("README.md", "README.rst", "README.txt", "readme.md"):
        candidate = os.path.join(path, name)
        if not os.path.isfile(candidate):
            continue
        with open(candidate, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        match = re.search(r"#+\s*licen[cs]e(.{0,1200})", text,
                          re.IGNORECASE | re.DOTALL)
        if match:
            return f"{name} (licence section)", match.group(1).strip()
    return "", ""


def classify(text: str) -> tuple[str, list[str]]:
    """(best guess at the licence, phrases that need a human to read them)."""
    if not text:
        return "NONE FOUND", []
    lowered = text.lower()

    flags = sorted({phrase for phrase in RESTRICTIVE if phrase in lowered})
    name = "unrecognised"
    for licence, markers in KNOWN:
        if all(marker in lowered for marker in markers):
            name = licence
            break
    return name, flags


def verdict(licence: str, flags: list[str], fetched: bool = True) -> str:
    """What this licence means for a commercial product.

    "We could not fetch the repo" and "we fetched it and it ships no licence" are
    completely different findings. Reporting the first as the second would
    wrongly discard a candidate that may be perfectly usable, so an unfetched
    repo gets its own verdict and never a BLOCKED.
    """
    if not fetched:
        return "UNKNOWN - repo not fetched, licence never checked"
    if flags:
        return "REVIEW - restrictive wording found"
    if licence in ("MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "CC-BY-4.0"):
        return "likely OK for commercial use - confirm"
    if licence == "GPL-3.0":
        return "REVIEW - copyleft, check what it obliges us to publish"
    if licence == "NONE FOUND":
        return ("BLOCKED - no licence file: default copyright applies, so no "
                "permission is granted. An intent stated elsewhere (a commented-out "
                "license= in setup.py, a line in a paper) is not a grant - ask the "
                "author to add a LICENSE file before relying on it.")
    return "REVIEW - licence not recognised, read it"


def weights_note(candidate: dict, path: str) -> str:
    delivery = candidate.get("weights")
    hint = candidate.get("weights_hint", "")
    if delivery == MANUAL:
        return f"MANUAL DOWNLOAD NEEDED - {hint}"
    if delivery == PIP:
        return f"fetched by the package on first use - {hint}"
    if not path:
        return "not checked - repo unavailable"

    found = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".pth", ".pt", ".pkl", ".h5", ".onnx", ".ckpt")):
                full = os.path.join(root, name)
                size = os.path.getsize(full) / 1048576
                found.append(f"{os.path.relpath(full, path)} ({size:.0f} MB)")
    if found:
        return "in repo: " + "; ".join(sorted(found)[:3])
    return f"not in the clone - {hint}" if hint else "not found in the clone"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated candidate ids")
    parser.add_argument("--licences-only", action="store_true",
                        help="report on what is already cloned; fetch nothing")
    args = parser.parse_args()

    if not shutil.which("git"):
        sys.exit("git is not on PATH.")

    wanted = CANDIDATES
    if args.only:
        ids = {value.strip() for value in args.only.split(",")}
        wanted = [c for c in CANDIDATES if c["id"] in ids]

    os.makedirs(WORKSPACE, exist_ok=True)
    rows = []

    print(f"\nFetching {len(wanted)} candidate(s) into {CLONES}\n")
    for candidate in wanted:
        print(f"  {candidate['name']:<26} ", end="", flush=True)
        if args.licences_only:
            path = os.path.join(CLONES, candidate["id"])
            path = path if os.path.isdir(path) else ""
            status = "already cloned" if path else "not cloned"
        else:
            path, status = clone(candidate)

        filename, text = find_licence(path)
        licence, flags = classify(text)
        row = {
            **candidate,
            "status": status,
            "path": path,
            "licence_file": filename,
            "licence_text": text,
            "licence": licence,
            "flags": flags,
            "verdict": verdict(licence, flags, fetched=bool(path)),
            "weights_note": weights_note(candidate, path),
            "provenance": prov.summarise(candidate),
        }
        rows.append(row)
        marker = {prov.GREEN: "", prov.AMBER: "  [weights: check]",
                  prov.RED: "  [WEIGHTS: likely disqualifying]"}
        print(f"{status:<16} {licence:<16} {row['verdict'].split(' - ')[0]}"
              f"{marker[row['provenance']['level']]}")

    write_report(rows)
    prov.write_notice(NOTICE)
    print(f"\nLicence report: {os.path.relpath(REPORT, REPO)}")
    print(f"NOTICE:         {os.path.relpath(NOTICE, REPO)}")

    unknown = [r for r in rows if r["verdict"].startswith("UNKNOWN")]
    blocked = [r for r in rows if r["verdict"].startswith(("BLOCKED", "REVIEW"))]
    clear = [r for r in rows if r["verdict"].startswith("likely OK")]

    if clear:
        print(f"\n{len(clear)} look usable (read the text yourself before relying on it):")
        for row in clear:
            print(f"  + {row['name']}: {row['licence']}")
    if blocked:
        print(f"\n{len(blocked)} need a human decision before any evaluation work:")
        for row in blocked:
            print(f"  - {row['name']}: {row['verdict']}")
            for flag in row["flags"]:
                print(f"      found: {flag}")
    if unknown:
        print(f"\n{len(unknown)} could not be checked at all - re-run to retry:")
        for row in unknown:
            print(f"  ? {row['name']}: {row['status']}")
    print("\nRead the report, decide which candidates are usable, THEN run "
          "fetch_datasets.py.\n")
    return 0


def write_report(rows: list[dict]) -> None:
    lines = [
        "# Vision Lab - saliency model licences",
        "",
        f"Generated by `scripts/bakeoff/fetch_models.py` on "
        f"{datetime.now(timezone.utc):%Y-%m-%d}.",
        "",
        "**This is a gate, not a recommendation.** ScaleSerum is a commercial",
        "product. A model whose licence forbids commercial use is out of the",
        "running however well it scores, and that decision is management's to",
        "make - this file only puts the actual licence text in front of them.",
        "",
        "No licence file at all is the worst outcome, not the most permissive:",
        "with no licence, default copyright applies and no permission is granted.",
        "",
        "| Model | Licence | Verdict | Weights | Repo |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        repo = f"[link]({row['repo']})" if row.get("repo") else "-"
        lines.append(f"| {row['name']} | {row['licence']} | {row['verdict']} | "
                     f"{row['weights_note']} | {repo} |")

    for control in (CURRENT_BASELINE, CENTRE_BASELINE):
        lines.append(f"| {control['name']} | {control['licence']} | "
                     f"no licence question | generated in code | - |")

    lines += [
        "", "---", "",
        "## Weights provenance - a SEPARATE question from the repo licence",
        "",
        "A LICENSE file grants rights over **code**. The trained weights are a",
        "different artefact, produced from datasets with terms of their own, and",
        "whether those terms follow the weights is unsettled law.",
        "",
        "Checking only the repo licence made UNISAL look like the cleanest",
        "candidate here. Its weights were trained on Hollywood-2 - clips from",
        "commercial feature films. No LICENSE file in a model repo can answer that.",
        "",
        "| Model | Weights licence | Trained on | Signal |",
        "|---|---|---|---|",
    ]
    for row in rows:
        p = row["provenance"]
        declared = p["weights_licence_declared"] or "**not declared**"
        if p["weights_licence_url"]:
            declared = f"[{declared}]({p['weights_licence_url']})"
        lines.append(f"| {p['name']} | {declared} | {', '.join(p['trained_on'])} | "
                     f"{p['level']} |")
    lines.append("")
    for row in rows:
        p = row["provenance"]
        if p["level"] == prov.GREEN:
            continue
        lines += [f"**{p['name']}** - {p['level']}:", ""]
        lines += [f"- {reason}" for reason in p["reasons"]]
        lines.append("")

    lines += ["", "---", "", "## Why each one is on the list", ""]
    for row in rows:
        lines += [f"### {row['name']}", "",
                  f"- **Paper:** {row.get('paper', '-')}",
                  f"- **Repo:** {row.get('repo', '-')}",
                  f"- **Clone:** {row['status']}",
                  f"- **Licence file:** {row['licence_file'] or 'NONE FOUND'}",
                  f"- **Verdict:** {row['verdict']}", ""]
        if row["flags"]:
            lines += ["**Restrictive wording found - read this properly:**", ""]
            lines += [f"- `{flag}`" for flag in row["flags"]]
            lines.append("")
        lines += [f"**Why considered:** {row['why']}", ""]
        if row.get("watch"):
            lines += [f"**Watch out for:** {row['watch']}", ""]
        if row["licence_text"]:
            excerpt = row["licence_text"].strip()
            excerpt = excerpt[:1500] + ("\n\n[...truncated - read the full file in "
                                        "the clone]" if len(excerpt) > 1500 else "")
            lines += ["<details><summary>Licence text</summary>", "",
                      "```", excerpt, "```", "", "</details>", ""]

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
