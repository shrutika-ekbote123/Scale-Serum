"""
Scoring for the accuracy harness. Pure functions, no I/O.

TWO SPELLINGS OF EVERY ERROR RATE
    native   Both sides lowercased, punctuation removed, script untouched. The
             honest number for a single-language reference such as FLEURS.
    roman    Both sides passed through transcription.romanize first. A
             code-switched call can be written "demo" or "डेमो" and both are
             right; comparing across scripts would count that as an error. This
             is the number to read for mixed-language calls.

    CER is reported beside WER because Indic word boundaries are unstable:
    Marathi and Kannada glue suffixes on, so one "wrong word" is often one
    wrong letter.

SPEAKERS
    Every transcribed word is placed against the truth by the midpoint of its
    timing. From that:
      role_accuracy        share of words whose speaker the PRODUCT labelled
                           with the right role (sales_rep / customer). Unknown
                           and participant count as wrong. This is the
                           end-to-end ">95% speaker role accuracy" number.
      diarization_accuracy share of words whose diarized speaker, mapped to the
                           truth speaker it mostly overlaps, is right. Separates
                           "the voices were split wrongly" from "the voices were
                           split right but labelled wrongly".
      backchannel_recall   short customer turns ("haan", "ji") attributed to
                           the customer - the case that bleeds into the rep.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable, Optional

from rapidfuzz.distance import Levenshtein

from transcription.romanize import romanize

_SPACE = re.compile(r"\s+")


def normalise_native(text: str) -> str:
    """Lowercase, punctuation and symbols removed, Indic vowel signs KEPT
    (they are combining marks, not punctuation)."""
    text = unicodedata.normalize("NFC", text or "").lower()
    kept = [c if unicodedata.category(c)[0] in ("L", "M", "N") else " " for c in text]
    return _SPACE.sub(" ", "".join(kept)).strip()


def normalise_roman(text: str) -> str:
    return romanize(text or "")


def edits(reference: str, hypothesis: str) -> dict:
    """Word and character edit counts, to be summed across a corpus."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    ref_chars, hyp_chars = reference.replace(" ", ""), hypothesis.replace(" ", "")
    return {
        "word_edits": Levenshtein.distance(ref_words, hyp_words),
        "ref_words": len(ref_words), "hyp_words": len(hyp_words),
        "char_edits": Levenshtein.distance(ref_chars, hyp_chars),
        "ref_chars": len(ref_chars),
    }


def text_scores(reference: str, hypothesis: str) -> dict:
    """Both spellings of WER/CER for one reference/hypothesis pair."""
    return {"native": edits(normalise_native(reference), normalise_native(hypothesis)),
            "roman": edits(normalise_roman(reference), normalise_roman(hypothesis))}


def rates(counts: dict) -> dict:
    """Corpus rates from summed edit counts. Summing edits then dividing is the
    standard corpus WER; averaging per-file WERs over-weights short files."""
    ref_w, ref_c = counts.get("ref_words", 0), counts.get("ref_chars", 0)
    return {
        "wer": round(counts["word_edits"] / ref_w, 4) if ref_w else None,
        "cer": round(counts["char_edits"] / ref_c, 4) if ref_c else None,
        # Words kept against words said. Far below 1.0 means speech was dropped,
        # which is the failure confidence cannot see.
        "length_ratio": round(counts["hyp_words"] / ref_w, 3) if ref_w else None,
    }


def add_counts(total: dict, part: dict) -> dict:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value
    return total


# --------------------------------------------------------------------------- #
# Speakers
# --------------------------------------------------------------------------- #
def deepgram_words(raw: dict) -> list[dict]:
    """Word list with timings and diarized speaker, from a raw Deepgram response."""
    try:
        words = raw["results"]["channels"][0]["alternatives"][0].get("words") or []
    except (KeyError, IndexError, TypeError):
        return []
    out = []
    for w in words:
        if w.get("start") is None or w.get("end") is None:
            continue
        speaker = w.get("speaker")
        out.append({"start": float(w["start"]), "end": float(w["end"]),
                    "speaker_id": f"speaker_{speaker}" if speaker is not None else "speaker_unattributed",
                    "text": w.get("punctuated_word") or w.get("word") or ""})
    return out


def _truth_turn_index(turns: list[dict], t: float, slack: float = 0.5) -> Optional[int]:
    best, best_gap = None, slack
    for i, turn in enumerate(turns):
        if turn["start"] <= t <= turn["end"]:
            return i
        gap = min(abs(t - turn["start"]), abs(t - turn["end"]))
        if gap < best_gap:
            best, best_gap = i, gap
    return best


def place_words(words: list[dict], turns: list[dict]) -> list[dict]:
    """Attach the truth turn (and its role) each hypothesis word falls in."""
    placed = []
    for w in words:
        idx = _truth_turn_index(turns, (w["start"] + w["end"]) / 2)
        placed.append({**w, "turn": idx, "truth_role": turns[idx]["role"] if idx is not None else None})
    return placed


def per_role_text(placed: list[dict], turns: list[dict]) -> dict:
    """Reference and hypothesis text per truth role, for per-speaker WER. Shows
    whether it is the customer's half that is being lost."""
    out = {}
    for role in sorted({t["role"] for t in turns}):
        ref = " ".join(t["text"] for t in turns if t["role"] == role)
        hyp = " ".join(w["text"] for w in placed if w["truth_role"] == role)
        out[role] = text_scores(ref, hyp)
    return out


def speaker_scores(placed: list[dict], turns: list[dict],
                   product_roles: dict[str, str]) -> dict:
    """product_roles: speaker_id -> role the product assigned
    ('sales_rep' / 'customer' / 'participant' / 'unknown')."""
    role_name = {"rep": "sales_rep", "customer": "customer", "participant": "participant"}
    scored = [w for w in placed if w["truth_role"] is not None]
    if not scored:
        return {"words_scored": 0}

    by_speaker: dict[str, Counter] = defaultdict(Counter)
    for w in scored:
        by_speaker[w["speaker_id"]][w["truth_role"]] += 1
    mapping = {sid: counts.most_common(1)[0][0] for sid, counts in by_speaker.items()}

    diarization_ok = sum(1 for w in scored if mapping[w["speaker_id"]] == w["truth_role"])
    role_ok = sum(1 for w in scored
                  if product_roles.get(w["speaker_id"]) == role_name.get(w["truth_role"]))

    rep_speakers = [sid for sid, role in product_roles.items() if role == "sales_rep"]
    cust_speakers = [sid for sid, role in product_roles.items() if role == "customer"]
    rep_truth = [mapping.get(s) for s in rep_speakers]
    outcome = ("unresolved" if not rep_speakers and not cust_speakers
               else "swapped" if "customer" in rep_truth or any(mapping.get(s) == "rep" for s in cust_speakers)
               else "correct" if rep_truth and all(r == "rep" for r in rep_truth) and cust_speakers
               else "partial")

    backchannels = [i for i, t in enumerate(turns)
                    if t["role"] == "customer" and len(t["text"].split()) <= 2]
    bc_ok = bc_missed = 0
    for i in backchannels:
        inside = [w for w in scored if w["turn"] == i]
        if not inside:
            bc_missed += 1
        elif Counter(product_roles.get(w["speaker_id"]) for w in inside).most_common(1)[0][0] == "customer":
            bc_ok += 1

    truth_talk = Counter()
    for t in turns:
        truth_talk[t["role"]] += t["end"] - t["start"]
    hyp_talk = Counter()
    for w in scored:
        hyp_talk[product_roles.get(w["speaker_id"])] += w["end"] - w["start"]
    truth_share = truth_talk["rep"] / max(sum(truth_talk.values()), 1e-9)
    hyp_total = sum(hyp_talk.values())
    hyp_share = hyp_talk["sales_rep"] / hyp_total if hyp_total else None

    return {
        "words_scored": len(scored),
        "speakers_found": len(by_speaker),
        "role_accuracy": round(role_ok / len(scored), 4),
        "diarization_accuracy": round(diarization_ok / len(scored), 4),
        "role_outcome": outcome,
        "backchannels": len(backchannels),
        "backchannel_recall": round(bc_ok / len(backchannels), 4) if backchannels else None,
        "backchannels_missed": bc_missed,
        "truth_rep_talk_share": round(truth_share, 3),
        "product_rep_talk_share": round(hyp_share, 3) if hyp_share is not None else None,
    }


# --------------------------------------------------------------------------- #
# Real calls, before anyone has labelled them
# --------------------------------------------------------------------------- #
def energy_speech_seconds(samples, rate: int = 16000) -> float:
    """Seconds of likely speech by a plain energy detector.

    A floor for the coverage check, not a VAD: it counts noise bursts as speech
    and a quiet customer as silence. Phase 1 replaces it with Silero VAD.
    """
    import numpy as np
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    hop = int(0.03 * rate)
    n = len(mono) // hop
    if n == 0:
        return 0.0
    frames = mono[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    floor = np.percentile(db, 10)
    active = db > floor + 12
    # 300 ms hangover so pauses between words do not count as silence
    hang = int(0.3 / 0.03)
    smoothed = active.copy()
    last = -hang - 1
    for i, on in enumerate(active):
        if on:
            last = i
        elif i - last <= hang:
            smoothed[i] = True
    return float(smoothed.sum() * 0.03)


def summarise(values: Iterable[Optional[float]]) -> dict:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": round(sum(vals) / len(vals), 4),
            "min": round(vals[0], 4), "median": round(vals[len(vals) // 2], 4),
            "max": round(vals[-1], 4)}
