"""
Step 8, part 3 - the public eye-tracking datasets, for the sanity check.

    python scripts/bakeoff/fetch_datasets.py --check      (verify the URLs, download nothing)
    python scripts/bakeoff/fetch_datasets.py --only mit1003
    python scripts/bakeoff/fetch_datasets.py

WHAT THESE ARE FOR, AND WHAT THEY ARE NOT
    MIT1003 and CAT2000 are photographs with real human fixation data. They tell
    us whether a candidate model is fundamentally sound - whether it predicts
    human gaze at all.

    They do NOT decide the bake-off. Our creatives are designed images: headline
    text, logos, buttons over video. A model can score beautifully on natural
    photographs and still ignore a headline. The decision metric is mean NSS on
    our own hand-annotated ad frames; these are the check that a model is not
    simply broken.

    So this download is NOT on the critical path. extract_ad_frames.py and the
    annotation tool need none of it, and that work can proceed while this runs.

EVALUATION INPUTS, NEVER PRODUCT COMPONENTS
    Downloaded into the gitignored bake-off workspace. Never deployed, never
    redistributed, never bundled into anything a customer receives. That
    distinction is recorded in NOTICE.md, because "we only used it to test" is a
    much easier position to hold if it is written down beforehand.

DOWNLOADS RESUME
    These servers drop long transfers - measured, not guessed. Every download is
    resumable: the partial file is kept and the next attempt asks for a byte
    range from where it stopped. Re-running the script continues rather than
    starting again.

URLS ROT
    Academic dataset hosting moves, and these particular files have moved
    before. Every URL below is CHECKED before anything is downloaded, and a dead
    one is reported with the page to go and look at - rather than a stack trace
    forty minutes into a download.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from typing import Optional

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

WORKSPACE = os.environ.get("VL_BAKEOFF_DIR", os.path.join(HERE, "workspace"))
DATA_DIR = os.path.join(WORKSPACE, "datasets")

# Each entry lists every mirror we know of. They are tried in order.
DATASETS = {
    "mit1003": {
        "name": "MIT1003",
        "what": "1003 images, real eye-tracking from 15 observers",
        "approx_mb": 1000,
        "homepage": "https://people.csail.mit.edu/tjudd/WherePeopleLook/",
        "files": [
            {"name": "ALLSTIMULI.zip", "urls": [
                "http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip",
                "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip",
            ]},
            {"name": "ALLFIXATIONMAPS.zip", "urls": [
                "http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip",
                "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip",
            ]},
        ],
    },
    "cat2000": {
        "name": "CAT2000 (training split)",
        "what": "2000 images across 20 scene categories",
        "approx_mb": 4000,
        "homepage": "https://saliency.tuebingen.ai/datasets.html",
        "files": [
            {"name": "trainSet.zip", "urls": [
                "http://saliency.mit.edu/trainSet.zip",
                "https://saliency.tuebingen.ai/datasets/CAT2000/trainSet.zip",
            ]},
        ],
    },
}


def probe(url: str, timeout: int = 20) -> tuple[bool, str]:
    """Is this URL alive, and how big is the file?

    HEAD first, then a 1-byte ranged GET - plenty of academic servers refuse
    HEAD but serve GET perfectly well, and treating that as "dead" would send
    someone hunting for a mirror they do not need.
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.head(url)
            if response.status_code >= 400:
                response = client.get(url, headers={"Range": "bytes=0-0"})
            if response.status_code >= 400:
                return False, f"HTTP {response.status_code}"
            size = response.headers.get("content-range", "").split("/")[-1] \
                or response.headers.get("content-length", "")
            mb = f"{int(size) / 1048576:.0f} MB" if size.isdigit() else "size unknown"
            return True, mb
    except Exception as err:  # noqa: BLE001
        return False, type(err).__name__


def download(url: str, dest: str) -> bool:
    tmp = dest + ".part"
    done = 0
    # RESUMABLE, because these servers drop long transfers.
    #
    # Measured: ALLSTIMULI died at 66 MB of 235, CAT2000 at 34 MB of 660, both
    # with the peer closing the connection. A downloader that discards the
    # partial file and restarts will never finish a 660 MB file from a host that
    # behaves like this - it just burns bandwidth in a loop.
    #
    # So the .part file is KEPT between attempts and each retry asks for a byte
    # range starting where the last one stopped. A server that ignores Range
    # (answers 200 instead of 206) is handled too, by starting over rather than
    # appending a second copy of the file onto the first.
    attempts = int(os.environ.get("VL_DOWNLOAD_ATTEMPTS", 8))
    expected = 0

    for attempt in range(1, attempts + 1):
        done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        mode = "ab" if done else "wb"
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=120) as response:
                if done and response.status_code == 200:
                    # Range ignored - the body is the whole file, so restart.
                    done, mode = 0, "wb"
                elif response.status_code not in (200, 206):
                    response.raise_for_status()

                length = int(response.headers.get("content-length") or 0)
                expected = expected or (done + length)

                with open(tmp, mode) as handle:
                    for chunk in response.iter_bytes(1024 * 512):
                        handle.write(chunk)
                        done += len(chunk)
                        if expected:
                            sys.stdout.write(
                                f"\r    {done / 1048576:7.0f} / "
                                f"{expected / 1048576:.0f} MB "
                                f"({done / expected * 100:5.1f}%)"
                                + (f"  [resume {attempt}]" if attempt > 1 else ""))
                        else:
                            sys.stdout.write(f"\r    {done / 1048576:7.0f} MB")
                        sys.stdout.flush()

            if expected and done < expected:
                raise OSError(f"short read: {done} of {expected} bytes")
            os.replace(tmp, dest)
            print()
            return True

        except Exception as err:  # noqa: BLE001
            got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            print(f"\n    attempt {attempt}/{attempts} stopped at "
                  f"{got / 1048576:.0f} MB: {err}")
            if attempt == attempts:
                print(f"    keeping the partial file - re-run to carry on from here")
                return False
            time.sleep(min(30, 3 * attempt))
    return False


def unpack(archive: str, into: str) -> Optional[str]:
    target = os.path.join(into, os.path.splitext(os.path.basename(archive))[0])
    if os.path.isdir(target):
        return target
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        return target
    except zipfile.BadZipFile:
        print(f"    not a valid zip - the server probably returned an error page")
        return None


def handle(key: str, spec: dict, check_only: bool) -> dict:
    print(f"\n{spec['name']}  (~{spec['approx_mb']} MB)")
    print(f"  {spec['what']}")
    folder = os.path.join(DATA_DIR, key)
    os.makedirs(folder, exist_ok=True)
    results = []

    for entry in spec["files"]:
        dest = os.path.join(folder, entry["name"])
        if os.path.isfile(dest):
            print(f"  {entry['name']:<24} already downloaded "
                  f"({os.path.getsize(dest) / 1048576:.0f} MB)")
            results.append({"file": entry["name"], "ok": True})
            continue

        live = None
        for url in entry["urls"]:
            ok, detail = probe(url)
            print(f"  {entry['name']:<24} {'OK  ' if ok else 'dead'}  {detail}  "
                  f"{url.split('//')[1][:44]}")
            if ok:
                live = url
                break

        if not live:
            print(f"    NO WORKING MIRROR. Fetch it by hand from "
                  f"{spec['homepage']}\n    and drop it in {folder}")
            results.append({"file": entry["name"], "ok": False})
            continue

        if check_only:
            results.append({"file": entry["name"], "ok": True})
            continue

        if download(live, dest):
            unpack(dest, folder)
            results.append({"file": entry["name"], "ok": True})
        else:
            results.append({"file": entry["name"], "ok": False})

    return {"key": key, "files": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated dataset keys")
    parser.add_argument("--check", action="store_true",
                        help="verify the URLs are alive, download nothing")
    args = parser.parse_args()

    keys = list(DATASETS)
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip() in DATASETS]

    os.makedirs(DATA_DIR, exist_ok=True)
    if args.check:
        print("\nChecking mirrors only - nothing will be downloaded.")

    outcomes = [handle(key, DATASETS[key], args.check) for key in keys]

    missing = [f"{o['key']}/{f['file']}" for o in outcomes
               for f in o["files"] if not f["ok"]]
    print("\n" + "-" * 62)
    if missing:
        print(f"{len(missing)} file(s) could not be fetched automatically:")
        for item in missing:
            print(f"  - {item}")
        print("\nThese are the SANITY CHECK, not the decision metric. The "
              "annotation\nwork does not depend on them - carry on with "
              "extract_ad_frames.py.")
    else:
        print("All datasets present." if not args.check
              else "All mirrors alive. Re-run without --check to download.")
    print(f"\nLocation: {os.path.relpath(DATA_DIR, REPO)}")
    print("Evaluation inputs only - never deployed, never redistributed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
