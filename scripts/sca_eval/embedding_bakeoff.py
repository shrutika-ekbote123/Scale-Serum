"""
Choose the speaker-embedding model for Phase 1, on calls with exact truth.

    python scripts/sca_eval/embedding_bakeoff.py

Every candidate is a sherpa-onnx speaker model in SCA_MODEL_DIR. Scored on the
synthetic calls (8 kHz mu-law, code-switched, half same-gender voices):

  within_eer       equal error rate telling two turns of the same call apart
                   by speaker. What cluster merging and splitting need.
  same_gender_eer  the same, on same-gender calls only - where Deepgram
                   collapsed to one voice.
  short_turn_acc   a turn of <= 1.5 s assigned to the nearer of the two
                   speakers' centroids (built from their long turns). What
                   rescuing "haan"/"okay" backchannels needs.
  enrolled_eer     rep voice enrolled from OTHER calls, then rep-vs-customer
                   turns in this call scored against it. The voiceprint case.
  rtf              compute seconds per audio second, single thread.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import SYNTH_DIR, decode_pcm, read_json, read_jsonl

MODEL_DIR = Path(os.environ.get("SCA_MODEL_DIR") or Path(__file__).resolve().parents[2] / ".models" / "sca")
CANDIDATES = [
    "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
    "wespeaker_en_voxceleb_resnet34_LM.onnx",
    "nemo_en_titanet_small.onnx",
]
RATE = 16000


def eer(genuine: list[float], impostor: list[float]) -> float:
    if not genuine or not impostor:
        return float("nan")
    scores = np.array(genuine + impostor)
    labels = np.array([1] * len(genuine) + [0] * len(impostor))
    best = 1.0
    for t in np.unique(scores):
        far = ((scores >= t) & (labels == 0)).sum() / len(impostor)
        frr = ((scores < t) & (labels == 1)).sum() / len(genuine)
        best = min(best, max(far, frr))
    return float(best)


def main() -> None:
    import sherpa_onnx

    calls = []
    for row in read_jsonl(SYNTH_DIR / "manifest.jsonl"):
        truth = read_json(SYNTH_DIR / row["truth"])
        audio = decode_pcm(SYNTH_DIR / row["audio"], rate=RATE)[:, 0]
        calls.append((row, truth, audio))

    print(f"{len(calls)} synthetic calls\n")
    print(f"{'model':52} within  same-g  short   enrolled  rtf")
    for name in CANDIDATES:
        path = MODEL_DIR / name
        if not path.exists():
            print(f"{name:52} missing")
            continue
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=1))

        def embed(samples):
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=RATE, waveform=samples.astype(np.float32))
            stream.input_finished()
            v = np.array(extractor.compute(stream), dtype=np.float32)
            return v / (np.linalg.norm(v) + 1e-9)

        started, audio_seconds = time.perf_counter(), 0.0
        per_call = []
        for row, truth, audio in calls:
            turns = []
            for t in truth["turns"]:
                a, b = int(t["start"] * RATE), int(t["end"] * RATE)
                if b - a < int(0.3 * RATE):
                    continue
                turns.append({**t, "dur": t["end"] - t["start"], "emb": embed(audio[a:b])})
                audio_seconds += t["end"] - t["start"]
            per_call.append((row, truth, turns))
        rtf = (time.perf_counter() - started) / max(audio_seconds, 1e-9)

        gen, imp, gen_sg, imp_sg = [], [], [], []
        short_ok = short_n = 0
        voice_turns = defaultdict(list)          # voice -> [(call id, emb)]
        for row, truth, turns in per_call:
            long_turns = [t for t in turns if t["dur"] >= 2.0]
            for i in range(len(long_turns)):
                for j in range(i + 1, len(long_turns)):
                    s = float(long_turns[i]["emb"] @ long_turns[j]["emb"])
                    same = long_turns[i]["role"] == long_turns[j]["role"]
                    (gen if same else imp).append(s)
                    if row["same_gender_voices"]:
                        (gen_sg if same else imp_sg).append(s)
            cent = {}
            for role in ("rep", "customer"):
                embs = [t["emb"] for t in long_turns if t["role"] == role]
                if embs:
                    c = np.mean(embs, axis=0)
                    cent[role] = c / np.linalg.norm(c)
            if len(cent) == 2:
                for t in turns:
                    if t["dur"] <= 1.5:
                        guess = max(cent, key=lambda r: float(t["emb"] @ cent[r]))
                        short_ok += guess == t["role"]
                        short_n += 1
            for t in long_turns:
                voice_turns[truth["voices"][t["role"]]].append((row["id"], t["emb"]))

        e_gen, e_imp = [], []
        for row, truth, turns in per_call:
            rep_voice = truth["voices"]["rep"]
            others = [e for cid, e in voice_turns[rep_voice] if cid != row["id"]]
            if len(others) < 3:
                continue
            print_ = np.mean(others, axis=0)
            print_ /= np.linalg.norm(print_)
            for t in turns:
                if t["dur"] < 1.0:
                    continue
                (e_gen if t["role"] == "rep" else e_imp).append(float(t["emb"] @ print_))

        print(f"{name:52} {eer(gen, imp):6.3f}  {eer(gen_sg, imp_sg):6.3f}  "
              f"{short_ok / max(short_n, 1):6.3f}  {eer(e_gen, e_imp):6.3f}   {rtf:.3f}"
              f"   (short n={short_n}, enrolled pairs {len(e_gen)}/{len(e_imp)})")


if __name__ == "__main__":
    main()
