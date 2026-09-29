"""
Fetch the evaluation audio for the Sales Call Analyzer accuracy harness.

    python scripts/sca_eval/fetch_datasets.py real              # recordings from scrumdb
    python scripts/sca_eval/fetch_datasets.py fleurs            # labelled Indian read speech
    python scripts/sca_eval/fetch_datasets.py fleurs --per-language 40 --languages hi_in mr_in

THREE SOURCES, THREE DIFFERENT QUESTIONS
    real       28 genuine ScaleSerum calls. The only source that sounds like
               production. No reference transcript exists until a human writes
               one (label_tool.html), so before that they can only answer the
               label-free questions: coverage, words per minute, speakers found,
               roles resolved.
    fleurs     Google FLEURS test split, CC-BY-4.0. Read sentences with exact
               references in 8 Indian languages. Answers "how accurate is the
               engine on this language", degraded to 8 kHz mu-law so the number
               is not flattered by studio audio. One speaker per file, so it
               says nothing about diarization.
    synthetic  make_synthetic_calls.py. Two fixed voices, code-switched sales
               dialogue, exact turn timings. Answers the diarization and role
               questions. TTS is cleaner than people, so trust its RELATIVE
               comparisons more than its absolute word error rates.

PRIVACY
    Real recordings and the CRM names that go with them land in the gitignored
    workspace and nowhere else. scrumdb is opened read-only. Recording URLs are
    never printed: a URL to a customer's call is as sensitive as the call.
"""
from __future__ import annotations

import argparse
import csv
import io
import tarfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from _common import (FLEURS_DIR, REAL_DIR, WORKSPACE, decode_pcm, ffprobe_path, pg_connect,
                     to_telephony, write_jsonl)

FLEURS_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
# Hindi is the only Indian language inside Deepgram's `multi`; the rest are the
# ones the analyzer is losing today.
FLEURS_LANGUAGES = ["hi_in", "mr_in", "kn_in", "pa_in", "ta_in", "te_in", "gu_in", "bn_in"]


# --------------------------------------------------------------------------- #
# Real calls
# --------------------------------------------------------------------------- #
def _stereo_profile(path: Path) -> dict:
    """Is a stereo file really split by speaker, or the same mix twice?

    Measured on the test calls: "stereo" recordings had L/R correlation 0.93 and
    both channels active together 80% of the time - no speaker separation at
    all. A genuinely dual-channel call shows low correlation and mostly one
    channel active at a time.
    """
    import json
    import subprocess

    import numpy as np

    probe = ffprobe_path()
    info = {}
    if probe:
        out = subprocess.run([probe, "-v", "error", "-show_streams", "-select_streams", "a:0",
                              "-of", "json", str(path)], capture_output=True, text=True).stdout
        streams = (json.loads(out or "{}").get("streams") or [{}])
        info = streams[0]
    channels = int(info.get("channels") or 1)
    profile = {"channels": channels, "sample_rate": int(info.get("sample_rate") or 0),
               "codec": info.get("codec_name")}

    audio = decode_pcm(path, rate=16000, channels=min(channels, 2))
    profile["duration_seconds"] = round(len(audio) / 16000, 2)

    # Effective bandwidth: share of energy above 4 kHz. ~0 means phone-line audio.
    mono = audio.mean(axis=1)
    n = 4096
    frames = mono[: len(mono) // n * n].reshape(-1, n)
    if len(frames):
        spectrum = (np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2).mean(0)
        freqs = np.fft.rfftfreq(n, 1 / 16000)
        profile["energy_above_4khz"] = round(float(spectrum[freqs >= 4000].sum() / spectrum.sum()), 5)

    if channels >= 2:
        left, right = audio[:, 0], audio[:, 1]
        profile["lr_correlation"] = round(float(np.corrcoef(left, right)[0, 1]), 3)
        hop = 1600  # 100 ms
        k = len(left) // hop
        e_l = np.sqrt((left[: k * hop].reshape(k, hop) ** 2).mean(1))
        e_r = np.sqrt((right[: k * hop].reshape(k, hop) ** 2).mean(1))
        threshold = max(e_l.max(), e_r.max()) * 0.05
        a_l, a_r = e_l > threshold, e_r > threshold
        both = float((a_l & a_r).mean())
        exclusive = float((a_l ^ a_r).mean())
        profile["both_active_ratio"] = round(both, 3)
        profile["one_active_ratio"] = round(exclusive, 3)
        profile["speaker_split_stereo"] = bool(profile["lr_correlation"] < 0.5 and exclusive > both)
    else:
        profile["speaker_split_stereo"] = False
    return profile


def fetch_real() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    with pg_connect() as conn:
        rows = conn.execute("""
            select c.id::text, c.brand_id::text, b.name, c.direction::text, c.provider,
                   c.duration_seconds, c.rep_name, l.full_name, c.recording_url,
                   c.transcript is not null and length(c.transcript) > 50,
                   c.analysis_id
              from sales_calls c
              left join brands b on b.id = c.brand_id
              left join leads l on l.id = c.lead_id
             where c.recording_url is not null
             order by c.created_at""").fetchall()

    manifest, fetched, skipped = [], 0, 0
    with httpx.Client(timeout=httpx.Timeout(300, connect=20), follow_redirects=True) as client:
        for (call_id, brand_id, brand, direction, provider, duration, rep, customer,
             url, has_crm_transcript, analysis_id) in rows:
            host = urlsplit(url).netloc
            if host.endswith("example.com"):
                skipped += 1
                continue
            ext = Path(urlsplit(url).path).suffix.lower() or ".mp3"
            target = REAL_DIR / f"{call_id}{ext}"
            if not target.exists():
                response = client.get(url)
                if response.status_code != 200 or not response.content:
                    print(f"  {call_id}: HTTP {response.status_code} from {host}, skipped")
                    skipped += 1
                    continue
                target.write_bytes(response.content)
                fetched += 1
            try:
                profile = _stereo_profile(target)
            except Exception as err:  # noqa: BLE001 - one bad file must not stop the set
                print(f"  {call_id}: unreadable audio ({err}), skipped")
                skipped += 1
                continue
            manifest.append({
                "id": call_id, "source": "real", "audio": target.name,
                "brand_id": brand_id, "brand_name": brand, "direction": direction,
                "provider": provider, "crm_duration_seconds": duration,
                "rep_name": rep, "customer_name": customer,
                "has_crm_transcript": bool(has_crm_transcript),
                "production_analysis_id": analysis_id, **profile,
            })
    count = write_jsonl(REAL_DIR / "manifest.jsonl", manifest)
    split = sum(1 for m in manifest if m.get("speaker_split_stereo"))
    narrow = sum(1 for m in manifest if m.get("energy_above_4khz") is not None and m["energy_above_4khz"] < 0.001)
    print(f"real: {count} calls in manifest ({fetched} downloaded now, {skipped} skipped). "
          f"speaker-split stereo: {split}. phone-line bandwidth: {narrow}.")


# --------------------------------------------------------------------------- #
# FLEURS
# --------------------------------------------------------------------------- #
def fetch_fleurs(languages: list[str], per_language: int) -> None:
    rows = []
    with httpx.Client(timeout=httpx.Timeout(600, connect=30), follow_redirects=True) as client:
        for lang in languages:
            out_dir = FLEURS_DIR / lang
            out_dir.mkdir(parents=True, exist_ok=True)
            tsv = client.get(f"{FLEURS_BASE}/{lang}/test.tsv").text
            refs = {}
            for rec in csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(rec) >= 4:
                    # rec: id, file, raw transcription, normalised transcription, ...
                    refs[rec[1]] = {"sentence_id": rec[0], "raw": rec[2], "text": rec[3],
                                    "gender": rec[-1]}

            have = sorted(p.name for p in out_dir.glob("*.wav") if not p.name.endswith(".tel.wav"))
            if len(have) < per_language:
                # Stream the tarball and stop once we have enough; the full test
                # archive is 100-400 MB per language and we need a few percent.
                with client.stream("GET", f"{FLEURS_BASE}/{lang}/audio/test.tar.gz") as resp:
                    resp.raise_for_status()
                    stream = _IterStream(resp.iter_bytes())
                    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                        for member in tar:
                            name = Path(member.name).name
                            if not member.isfile() or name not in refs:
                                continue
                            data = tar.extractfile(member).read()
                            (out_dir / name).write_bytes(data)
                            have.append(name)
                            if len(have) >= per_language:
                                break

            for name in sorted(have)[:per_language]:
                src = out_dir / name
                tel = src.with_suffix(".tel.wav")
                if not tel.exists():
                    to_telephony(src, tel)
                ref = refs.get(name, {})
                rows.append({"id": f"{lang}/{src.stem}", "source": "fleurs",
                             "language": lang.split("_")[0], "audio": f"{lang}/{tel.name}",
                             "clean_audio": f"{lang}/{name}", "reference": ref.get("raw"),
                             "reference_normalised": ref.get("text"),
                             "gender": ref.get("gender"), "sentence_id": ref.get("sentence_id")})
            print(f"fleurs {lang}: {min(len(have), per_language)} utterances")
    write_jsonl(FLEURS_DIR / "manifest.jsonl", rows)
    print(f"fleurs: {len(rows)} utterances in manifest")


# --------------------------------------------------------------------------- #
# CREMA-D: real human voices, same sentence, different emotion
# --------------------------------------------------------------------------- #
CREMAD_URL = "https://huggingface.co/datasets/myleslinder/crema-d/resolve/main/data/crema_d.tar.gz"
CREMAD_DIR = WORKSPACE / "cremad"
# CREMA-D emotion codes. Disgust is left out: it has no counterpart in a sales call.
CREMAD_EMOTIONS = {"ANG": "anger", "HAP": "happy", "FEA": "fear", "SAD": "sad", "NEU": "neutral"}
CREMAD_SENTENCES = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.", "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}


def fetch_cremad(per_emotion: int, per_actor: int) -> None:
    """CREMA-D (ODbL): 91 actors saying 12 neutral sentences in 6 emotions.

    The point is that the WORDS carry no emotion - "It's eleven o'clock" said
    angrily and happily is the same transcript. A tone detector that reads text
    must score at chance here; only one that hears the voice can do better.
    Degraded to 8 kHz mu-law like every other set.
    """
    CREMAD_DIR.mkdir(parents=True, exist_ok=True)
    counts = {e: 0 for e in CREMAD_EMOTIONS}
    per = {}
    rows = []
    archive = CREMAD_DIR / "crema_d.tar.gz"
    _download_resumable(CREMAD_URL, archive)
    if True:
        with archive.open("rb") as fh:
            with tarfile.open(fileobj=fh, mode="r|gz") as tar:
                for member in tar:
                    name = Path(member.name).name
                    if not member.isfile() or not name.endswith(".wav"):
                        continue
                    parts = name[:-4].split("_")
                    if len(parts) < 3 or parts[2] not in CREMAD_EMOTIONS:
                        continue
                    actor, sentence, emo = parts[0], parts[1], parts[2]
                    if counts[emo] >= per_emotion or per.get((actor, emo), 0) >= per_actor:
                        continue
                    src = CREMAD_DIR / name
                    src.write_bytes(tar.extractfile(member).read())
                    tel = src.with_suffix(".tel.wav")
                    to_telephony(src, tel)
                    counts[emo] += 1
                    per[(actor, emo)] = per.get((actor, emo), 0) + 1
                    rows.append({"id": name[:-4], "source": "cremad", "audio": tel.name,
                                 "actor": actor, "emotion": CREMAD_EMOTIONS[emo],
                                 "text": CREMAD_SENTENCES.get(sentence, "")})
                    if all(c >= per_emotion for c in counts.values()):
                        break
    write_jsonl(CREMAD_DIR / "manifest.jsonl", rows)
    print(f"cremad: {len(rows)} clips, {counts}, {len({r['actor'] for r in rows})} actors")


def _download_resumable(url: str, target: Path, attempts: int = 30) -> None:
    """This host drops long transfers (measured: cut at 23 MB of 471). Each
    retry asks for the bytes from where the last one stopped."""
    import time
    with httpx.Client(timeout=httpx.Timeout(120, connect=30), follow_redirects=True) as client:
        total = int(client.head(url).headers.get("content-length", 0))
        for attempt in range(attempts):
            have = target.stat().st_size if target.exists() else 0
            if total and have >= total:
                return
            try:
                with client.stream("GET", url, headers={"Range": f"bytes={have}-"}) as resp:
                    if resp.status_code not in (200, 206):
                        raise httpx.HTTPError(f"HTTP {resp.status_code}")
                    mode = "ab" if resp.status_code == 206 else "wb"
                    with target.open(mode) as out:
                        for chunk in resp.iter_bytes():
                            out.write(chunk)
            except httpx.HTTPError as err:
                size = target.stat().st_size if target.exists() else 0
                print(f"  download interrupted at {size / 1e6:.0f} of {total / 1e6:.0f} MB "
                      f"({type(err).__name__}), resuming")
                time.sleep(min(2 * (attempt + 1), 20))
        raise SystemExit(f"could not download {url}")


class _IterStream(io.RawIOBase):
    """A file-like view of an httpx byte iterator, for tarfile's stream mode."""

    def __init__(self, iterator):
        self._it = iterator
        self._buf = b""

    def readable(self) -> bool:
        return True

    def readinto(self, target) -> int:
        while not self._buf:
            try:
                self._buf = next(self._it)
            except StopIteration:
                return 0
        n = min(len(target), len(self._buf))
        target[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="what", required=True)
    sub.add_parser("real")
    fl = sub.add_parser("fleurs")
    fl.add_argument("--languages", nargs="+", default=FLEURS_LANGUAGES)
    fl.add_argument("--per-language", type=int, default=30)
    cr = sub.add_parser("cremad")
    cr.add_argument("--per-emotion", type=int, default=40)
    cr.add_argument("--per-actor", type=int, default=2)
    args = parser.parse_args()
    if args.what == "real":
        fetch_real()
    elif args.what == "cremad":
        fetch_cremad(args.per_emotion, args.per_actor)
    else:
        fetch_fleurs(args.languages, args.per_language)


if __name__ == "__main__":
    main()
