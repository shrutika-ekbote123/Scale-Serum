"""
Speaker refinement - checking Deepgram's "who spoke when" against the voices.

WHAT GOES WRONG WITHOUT IT (scripts/sca_eval, synthetic calls with exact truth)
    * Two same-gender voices collapse into ONE speaker (4 of 16 calls).
    * One person becomes TWO speakers after switching English -> Hindi.
    * One-word customer replies ("okay", "haan") are credited to the rep ~80%
      of the time.
    Every one of these corrupts talk time, objection attribution and the role
    each speaker is given.

HOW
    The words Deepgram returned are cut into short stretches at pauses, each
    stretch is embedded (transcription/voice.py), and then:
      1. merge   speakers whose voices are the same person
      2. split   a speaker whose stretches form two distinct voices
      3. move    a stretch to the voice it clearly sounds like, when that is not
                 the speaker Deepgram gave it - the backchannel case
      4. rep     with the rep's enrolled voiceprint, which voice is the rep
    The output is a Deepgram-shaped response with words relabelled and
    utterances rebuilt, so transcript.py and everything after it is unchanged.

WHAT IT NEVER DOES
    * Change a word, a timing, or the order of anything. Only speaker labels.
    * Decide a role. It reports a voice score; speakers.py decides.
    * Fail an analysis. Any error returns the input untouched with a reason.

THRESHOLDS
    Cosine similarities between ERes2Net embeddings, calibrated on the
    synthetic set with scripts/sca_eval/run_eval.py. Each is an env setting so a
    recalibration on labelled real calls needs no code change.
"""
from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("sales_call_analyzer.diarization")

RATE = 16000

# Stretches: consecutive words of one Deepgram speaker, cut at a pause this long
# or when they reach MAX_STRETCH_SECONDS.
PAUSE_SECONDS = float(os.environ.get("SCA_REFINE_PAUSE_SECONDS", 0.25))
MAX_STRETCH_SECONDS = float(os.environ.get("SCA_REFINE_MAX_STRETCH_SECONDS", 6.0))
# Stretches at least this long define what a speaker sounds like.
ANCHOR_SECONDS = float(os.environ.get("SCA_REFINE_ANCHOR_SECONDS", 1.5))
# Only the middle this-many seconds of a stretch is embedded. Cost is linear in
# audio embedded: a 10-minute call took 42 s of CPU uncropped (single thread).
EMBED_MAX_SECONDS = float(os.environ.get("SCA_REFINE_EMBED_MAX_SECONDS", 3.0))

# WITHOUT a voiceprint, voices are compared only with each other - and that is
# weak. Measured on the synthetic set: the SAME rep before and after switching
# to Hindi scored 0.48-0.51, while two DIFFERENT same-gender speakers scored up
# to 0.74. No threshold separates those, so merging and splitting stay
# conservative (only near-certain cases) and the main gain is moving stretches.
MERGE_AT = float(os.environ.get("SCA_REFINE_MERGE_AT", 0.80))
SPLIT_BELOW = float(os.environ.get("SCA_REFINE_SPLIT_BELOW", 0.25))
SPLIT_MIN_SECONDS = float(os.environ.get("SCA_REFINE_SPLIT_MIN_SECONDS", 20))
SPLIT_MIN_SHARE = float(os.environ.get("SCA_REFINE_SPLIT_MIN_SHARE", 0.15))
MOVE_MARGIN = float(os.environ.get("SCA_REFINE_MOVE_MARGIN", 0.10))

# WITH the rep's enrolled voiceprint, each stretch is scored against it directly,
# which is far stronger. Measured (voiceprint enrolled from OTHER calls): rep
# turns scored 0.50-0.86 (5th-95th percentile), customer turns -0.02-0.45, and
# customer one-word turns below 0.30. At 0.45, 98.7% of rep turns are kept and
# 97.3% of customer turns rejected. Between the two thresholds a stretch is
# "unsure" and keeps Deepgram's label.
REP_MATCH_AT = float(os.environ.get("SCA_VOICEPRINT_MATCH_AT", 0.40))
REP_REJECT_BELOW = float(os.environ.get("SCA_VOICEPRINT_REJECT_BELOW", 0.35))
# A non-rep stretch this unlike every other voice on the call starts a new one:
# Deepgram had merged the customer into the rep.
NEW_VOICE_BELOW = float(os.environ.get("SCA_REFINE_NEW_VOICE_BELOW", 0.30))
# One word is far weaker evidence than a stretch: a rep's own first/last words
# scored a median of only 0.33 against their voiceprint (n=444). Splitting an
# edge word off below 0.10 caught 46% of customer one-word replies stuck to a
# rep stretch, and wrongly split 3.6% of rep edge words; at 0.35 it would have
# wrongly split 56%. Measured on the synthetic set, 2026-09-28.
EDGE_REJECT_BELOW = float(os.environ.get("SCA_REFINE_EDGE_REJECT_BELOW", 0.10))
NEW_VOICE_MIN_SECONDS = float(os.environ.get("SCA_REFINE_NEW_VOICE_MIN_SECONDS", 4.0))

# Refinement is skipped past this length: the audio is held in memory.
MAX_SECONDS = float(os.environ.get("SCA_REFINE_MAX_SECONDS", 3600))

REASON_NO_WORDS = "no_word_timings"
REASON_TOO_LONG = "recording_too_long"
REASON_TOO_LITTLE_SPEECH = "too_little_speech_to_compare"
REASON_ERROR = "refinement_error"


@dataclass
class Stretch:
    words: list[int]
    start: float
    end: float
    speaker: int
    emb: Optional[np.ndarray] = None

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass
class Refinement:
    """What refinement found and did. Stored on the analysis; no embeddings."""
    applied: bool = False
    reason: Optional[str] = None
    model: Optional[str] = None
    speakers_before: int = 0
    speakers_after: int = 0
    merged: list[list[str]] = field(default_factory=list)
    split: list[str] = field(default_factory=list)
    moved_words: int = 0
    moved_seconds: float = 0.0
    words_relabelled: int = 0
    rep_speaker_id: Optional[str] = None
    voice_scores: dict[str, float] = field(default_factory=dict)
    ms: Optional[int] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _words(raw: dict) -> list[dict]:
    try:
        return raw["results"]["channels"][0]["alternatives"][0].get("words") or []
    except (KeyError, IndexError, TypeError):
        return []


def _stretches(words: list[dict]) -> list[Stretch]:
    out: list[Stretch] = []
    for i, w in enumerate(words):
        speaker = int(w.get("speaker") or 0)
        start, end = float(w["start"]), float(w["end"])
        cur = out[-1] if out else None
        if (cur is not None and cur.speaker == speaker and start - cur.end < PAUSE_SECONDS
                and end - cur.start <= MAX_STRETCH_SECONDS):
            cur.words.append(i)
            cur.end = end
        else:
            out.append(Stretch(words=[i], start=start, end=end, speaker=speaker))
    return out


def _centroids(stretches: list[Stretch]) -> dict[int, np.ndarray]:
    groups: dict[int, list[np.ndarray]] = {}
    fallback: dict[int, list[np.ndarray]] = {}
    for s in stretches:
        if s.emb is None:
            continue
        fallback.setdefault(s.speaker, []).append(s.emb)
        if s.seconds >= ANCHOR_SECONDS:
            groups.setdefault(s.speaker, []).append(s.emb)
    out = {}
    for spk, embs in fallback.items():
        chosen = groups.get(spk) or embs
        out[spk] = _unit(np.mean(chosen, axis=0))
    return out


def _speech(stretches: list[Stretch], speaker: int) -> float:
    return sum(s.seconds for s in stretches if s.speaker == speaker)


def _two_means(vectors: np.ndarray, seed: int = 0, restarts: int = 5):
    """Spherical 2-means. Returns (labels, centroid_a, centroid_b)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        a = vectors[rng.integers(len(vectors))]
        # k-means++: the second seed is the vector least like the first
        b = vectors[int(np.argmin(vectors @ a))]
        for _ in range(20):
            labels = (vectors @ b > vectors @ a).astype(int)
            if labels.min() == labels.max():
                break
            a_new = _unit(vectors[labels == 0].mean(0))
            b_new = _unit(vectors[labels == 1].mean(0))
            if np.allclose(a_new, a) and np.allclose(b_new, b):
                break
            a, b = a_new, b_new
        score = float(np.sum(np.maximum(vectors @ a, vectors @ b)))
        if best is None or score > best[0]:
            best = (score, labels.copy(), a, b)
    return best[1], best[2], best[3]


def refine(raw: dict, samples: np.ndarray,
           embed: Callable[[np.ndarray], Optional[np.ndarray]],
           rep_voiceprint: Optional[np.ndarray] = None,
           model: Optional[str] = None) -> tuple[dict, Refinement]:
    """(refined raw Deepgram response, what happened). Never raises."""
    report = Refinement(model=model)
    started = time.monotonic()
    try:
        refined = _refine(raw, samples, embed, rep_voiceprint, report)
    except Exception as err:  # noqa: BLE001 - refinement is an improvement, never a dependency
        logger.warning("speaker refinement failed: %s", type(err).__name__)
        report.applied, report.reason = False, REASON_ERROR
        refined = raw
    report.ms = int((time.monotonic() - started) * 1000)
    return refined, report


def _refine(raw, samples, embed, rep_voiceprint, report: Refinement) -> dict:
    words = _words(raw)
    words = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    if not words:
        report.reason = REASON_NO_WORDS
        return raw
    if len(samples) / RATE > MAX_SECONDS:
        report.reason = REASON_TOO_LONG
        return raw

    stretches = _stretches(words)
    original = [int(w.get("speaker") or 0) for w in words]
    report.speakers_before = len(set(original))

    pad = int(0.05 * RATE)
    crop = int(EMBED_MAX_SECONDS * RATE)
    for s in stretches:
        a = max(int(s.start * RATE) - pad, 0)
        b = min(int(s.end * RATE) + pad, len(samples))
        if b - a > crop:            # the middle of a long stretch says who it is
            a += (b - a - crop) // 2
            b = a + crop
        s.emb = embed(samples[a:b]) if b > a else None
    if sum(1 for s in stretches if s.emb is not None and s.seconds >= ANCHOR_SECONDS) < 2:
        report.reason = REASON_TOO_LITTLE_SPEECH
        return raw

    if rep_voiceprint is not None:
        stretches = _split_edges(stretches, words, samples, embed, rep_voiceprint)
        _by_voiceprint(stretches, original, rep_voiceprint, report)
    else:
        _by_comparison(stretches, original, report)

    labels = list(original)
    for s in stretches:
        for i in s.words:
            labels[i] = s.speaker
    report.words_relabelled = sum(1 for a, b in zip(original, labels) if a != b)
    report.speakers_after = len(set(labels))
    report.moved_seconds = round(report.moved_seconds, 2)
    if rep_voiceprint is not None:
        report.voice_scores = {f"speaker_{k}": round(float(v @ rep_voiceprint), 4)
                               for k, v in _centroids(stretches).items()}
    report.applied = True
    return _rebuild(raw, words, labels)


def _split_edges(stretches: list[Stretch], words: list[dict], samples: np.ndarray,
                 embed, voiceprint: np.ndarray) -> list[Stretch]:
    """Split a customer's one-word reply off the front or back of a rep stretch.

    Deepgram's word timings leave almost no gap between a customer's "Okay."
    and the rep's next sentence, so pause-cutting joins them: on the synthetic
    set the reply was the FIRST WORD of a rep stretch ("Okay. We offer a
    curated twelve week...") in most lost backchannels. For each stretch that
    sounds like the rep, its first and last word are scored on their own.
    """
    pad = int(0.05 * RATE)

    def score(i: int) -> Optional[float]:
        w = words[i]
        a = max(int(float(w["start"]) * RATE) - pad, 0)
        b = min(int(float(w["end"]) * RATE) + pad, len(samples))
        v = embed(samples[a:b]) if b > a else None
        return None if v is None else float(v @ voiceprint)

    out: list[Stretch] = []
    for s in stretches:
        if (s.emb is None or len(s.words) < 3 or float(s.emb @ voiceprint) < REP_MATCH_AT):
            out.append(s)
            continue
        head, tail = [], []
        first, last = s.words[0], s.words[-1]
        if (sc := score(first)) is not None and sc < EDGE_REJECT_BELOW:
            head = [first]
        if (sc := score(last)) is not None and sc < EDGE_REJECT_BELOW:
            tail = [last]
        if not head and not tail:
            out.append(s)
            continue
        middle = s.words[len(head):len(s.words) - len(tail)]
        for part in (head, middle, tail):
            if not part:
                continue
            a, b = float(words[part[0]]["start"]), float(words[part[-1]]["end"])
            piece = Stretch(words=part, start=a, end=b, speaker=s.speaker)
            lo, hi = max(int(a * RATE) - pad, 0), min(int(b * RATE) + pad, len(samples))
            piece.emb = s.emb if part is middle else (embed(samples[lo:hi]) if hi > lo else None)
            out.append(piece)
    return out


def _by_voiceprint(stretches: list[Stretch], original: list[int],
                   voiceprint: np.ndarray, report: Refinement) -> None:
    """Every stretch scored against the rep's own voice.

    rep      -> the rep's speaker, whatever Deepgram said (fixes the rep split
                in two by a language switch, and customer speech that Deepgram
                put under the rep's label is freed below)
    not rep  -> if Deepgram had it under the rep's label, the nearest other
                voice, or a new voice when none is close (fixes the collapse)
    unsure   -> Deepgram's label stands
    """
    scores = {id(s): float(s.emb @ voiceprint) for s in stretches if s.emb is not None}
    rep_like = [s for s in stretches if scores.get(id(s), -1) >= REP_MATCH_AT]
    if not rep_like:
        return          # the enrolled rep is not on this call, or not audibly

    # The rep's label: the Deepgram speaker holding most rep-like speech.
    weight: dict[int, float] = {}
    for s in rep_like:
        weight[s.speaker] = weight.get(s.speaker, 0.0) + s.seconds
    rep = max(weight, key=weight.get)
    report.rep_speaker_id = f"speaker_{rep}"

    before = {id(s): s.speaker for s in stretches}
    for s in rep_like:
        if s.speaker != rep:
            report.merged.append([f"speaker_{s.speaker}", f"speaker_{rep}"])
        s.speaker = rep

    others = [s for s in stretches if s.speaker != rep and s.emb is not None
              and scores.get(id(s), 1) < REP_REJECT_BELOW]
    cents = _centroids(others)
    new_id = max(max(original), max(s.speaker for s in stretches)) + 1
    for s in stretches:
        if s.speaker != rep or s.emb is None or scores[id(s)] >= REP_REJECT_BELOW:
            continue
        best = max(cents, key=lambda k: float(s.emb @ cents[k])) if cents else None
        # A short stretch is too little evidence to declare a new person: it
        # joins the nearest other voice. Only a long one, unlike every voice
        # already on the call, starts a new one.
        if best is not None and (float(s.emb @ cents[best]) >= NEW_VOICE_BELOW
                                 or s.seconds < ANCHOR_SECONDS):
            s.speaker = best
        else:
            # Deepgram merged someone into the rep. They become a voice of their
            # own, and later stretches like them join it.
            s.speaker = new_id
            cents[new_id] = s.emb
            report.split.append(f"speaker_{rep}")
            new_id += 1

    # Several stretches may have opened new voices for the same person.
    report.split = sorted(set(report.split))
    report.merged = [list(p) for p in sorted({tuple(p) for p in report.merged})]
    _merge_new_voices(stretches, original, rep)
    for s in stretches:
        if before[id(s)] != s.speaker:
            report.moved_words += len(s.words)
            report.moved_seconds += s.seconds


def _merge_new_voices(stretches: list[Stretch], original: list[int],
                      rep: Optional[int] = None) -> None:
    """Tidy the voices opened one stretch at a time.

    New voices that sound alike become one. A new voice with under
    NEW_VOICE_MIN_SECONDS of speech is not a person - usually one odd stretch -
    and joins the nearest non-rep voice. On the synthetic set, without this a
    two-person call came back with up to five speakers.
    """
    first_new = max(original) + 1
    while True:
        cents = _centroids([s for s in stretches if s.speaker >= first_new])
        ids = sorted(cents)
        pair = next(((a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
                     if float(cents[a] @ cents[b]) >= NEW_VOICE_BELOW), None)
        if pair is None:
            break
        for s in stretches:
            if s.speaker == pair[1]:
                s.speaker = pair[0]

    while True:
        new = [k for k in {s.speaker for s in stretches} if k >= first_new]
        small = sorted((k for k in new if _speech(stretches, k) < NEW_VOICE_MIN_SECONDS),
                       key=lambda k: _speech(stretches, k))
        if not small:
            return
        victim = small[0]
        cents = _centroids([s for s in stretches if s.speaker not in (victim, rep)])
        if not cents:
            return      # the only other voice there is: keep it
        mine = _centroids([s for s in stretches if s.speaker == victim]).get(victim)
        target = max(cents, key=lambda k: float(mine @ cents[k])) if mine is not None else next(iter(cents))
        for s in stretches:
            if s.speaker == victim:
                s.speaker = target


def _by_comparison(stretches: list[Stretch], original: list[int],
                   report: Refinement) -> None:
    """No voiceprint: voices compared only with each other, conservatively."""
    # ---- 1. merge: same voice under two labels ------------------------------
    while True:
        cents = _centroids(stretches)
        ids = sorted(cents)
        best = None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = float(cents[ids[i]] @ cents[ids[j]])
                if sim >= MERGE_AT and (best is None or sim > best[0]):
                    best = (sim, ids[i], ids[j])
        if best is None:
            break
        _, keep, drop = best
        # Keep the label that spoke more, so speaker_0 stays the first voice where it can.
        if _speech(stretches, drop) > _speech(stretches, keep):
            keep, drop = drop, keep
        for s in stretches:
            if s.speaker == drop:
                s.speaker = keep
        report.merged.append([f"speaker_{drop}", f"speaker_{keep}"])

    # ---- 2. split: two voices under one label -------------------------------
    next_id = max(max(original), max(s.speaker for s in stretches)) + 1
    for spk in sorted({s.speaker for s in stretches}):
        mine = [s for s in stretches if s.speaker == spk and s.emb is not None]
        anchors = [s for s in mine if s.seconds >= ANCHOR_SECONDS]
        total = sum(s.seconds for s in mine)
        if total < SPLIT_MIN_SECONDS or len(anchors) < 6:
            continue
        labels, ca, cb = _two_means(np.stack([s.emb for s in anchors]))
        share_b = sum(s.seconds for s, l in zip(anchors, labels) if l == 1) / max(
            sum(s.seconds for s in anchors), 1e-9)
        if float(ca @ cb) >= SPLIT_BELOW or min(share_b, 1 - share_b) < SPLIT_MIN_SHARE:
            continue
        for s in mine:
            if float(s.emb @ cb) > float(s.emb @ ca):
                s.speaker = next_id
        report.split.append(f"speaker_{spk}")
        next_id += 1

    # ---- 3. move stretches to the voice they sound like ----------------------
    cents = _centroids(stretches)
    if len(cents) >= 2:
        for s in stretches:
            if s.emb is None:
                continue
            own = float(s.emb @ cents[s.speaker]) if s.speaker in cents else -1.0
            best_spk = max(cents, key=lambda k: float(s.emb @ cents[k]))
            if best_spk != s.speaker and float(s.emb @ cents[best_spk]) - own >= MOVE_MARGIN:
                s.speaker = best_spk
                report.moved_words += len(s.words)
                report.moved_seconds += s.seconds


def _rebuild(raw: dict, words: list[dict], labels: list[int]) -> dict:
    """The response with each word's speaker replaced and utterances regrouped."""
    out = copy.deepcopy(raw)
    new_words = []
    for w, spk in zip(words, labels):
        nw = dict(w)
        nw["speaker"] = spk
        new_words.append(nw)
    alt = out["results"]["channels"][0]["alternatives"][0]
    alt["words"] = new_words

    utterances = []
    for w in new_words:
        text = w.get("punctuated_word") or w.get("word") or ""
        cur = utterances[-1] if utterances else None
        if cur is not None and cur["speaker"] == w["speaker"] and float(w["start"]) - cur["end"] <= 1.0:
            cur["transcript"] += " " + text
            cur["end"] = float(w["end"])
            cur["_conf"].append(w.get("confidence"))
            cur["words"].append(w)
        else:
            utterances.append({"speaker": w["speaker"], "start": float(w["start"]),
                               "end": float(w["end"]), "transcript": text,
                               "_conf": [w.get("confidence")], "words": [w], "channel": 0})
    for u in utterances:
        confs = [c for c in u.pop("_conf") if isinstance(c, (int, float))]
        u["confidence"] = round(sum(confs) / len(confs), 4) if confs else None
    out["results"]["utterances"] = utterances
    return out


# =========================================================================== #
# Dual-channel recordings: one speaker per channel
# =========================================================================== #
# A dialer that records each leg on its own channel settles "who spoke when"
# exactly - no voice comparison needed. None of the 28 test recordings is one:
# their "stereo" files carry the same mix twice (L/R correlation 0.92-1.0), so
# this only acts when the channels really differ.
SPLIT_MAX_CORRELATION = float(os.environ.get("SCA_STEREO_SPLIT_MAX_CORRELATION", 0.5))
# A word belongs to a channel carrying this much more energy (6 dB = 4x power).
CHANNEL_DOMINANCE_DB = float(os.environ.get("SCA_STEREO_DOMINANCE_DB", 6.0))


def stereo_profile(stereo: np.ndarray) -> dict:
    """Are the two channels two different speakers, or the same mix twice?"""
    if stereo is None or stereo.ndim != 2 or stereo.shape[1] < 2 or len(stereo) < RATE:
        return {"channels": 1 if stereo is None or stereo.ndim == 1 else int(stereo.shape[1]),
                "speaker_split": False}
    left, right = stereo[:, 0], stereo[:, 1]
    corr = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else 1.0
    hop = int(0.1 * RATE)
    k = len(left) // hop
    e_l = np.sqrt((left[: k * hop].reshape(k, hop) ** 2).mean(1))
    e_r = np.sqrt((right[: k * hop].reshape(k, hop) ** 2).mean(1))
    floor = max(float(e_l.max()), float(e_r.max())) * 0.05
    a_l, a_r = e_l > floor, e_r > floor
    both, one = float((a_l & a_r).mean()), float((a_l ^ a_r).mean())
    return {"channels": 2, "lr_correlation": round(corr, 3),
            "both_active_ratio": round(both, 3), "one_active_ratio": round(one, 3),
            "speaker_split": bool(corr < SPLIT_MAX_CORRELATION and one > both)}


def attribute_by_channel(raw: dict, stereo: np.ndarray) -> tuple[dict, Refinement]:
    """Relabel every word with the channel it was spoken on: speaker_0 = left,
    speaker_1 = right. A word loud on both (crosstalk) keeps Deepgram's label."""
    report = Refinement(model="channels")
    started = time.monotonic()
    words = [w for w in _words(raw) if w.get("start") is not None and w.get("end") is not None]
    if not words:
        report.reason = REASON_NO_WORDS
        return raw, report
    original = [int(w.get("speaker") or 0) for w in words]
    report.speakers_before = len(set(original))
    ratio = 10 ** (CHANNEL_DOMINANCE_DB / 20)
    labels = list(original)
    for i, w in enumerate(words):
        a, b = int(float(w["start"]) * RATE), max(int(float(w["end"]) * RATE), int(float(w["start"]) * RATE) + 1)
        seg = stereo[a:b]
        if not len(seg):
            continue
        rms_l = float(np.sqrt((seg[:, 0] ** 2).mean()))
        rms_r = float(np.sqrt((seg[:, 1] ** 2).mean()))
        if rms_l >= rms_r * ratio:
            labels[i] = 0
        elif rms_r >= rms_l * ratio:
            labels[i] = 1
    report.words_relabelled = sum(1 for a, b in zip(original, labels) if a != b)
    report.speakers_after = len(set(labels))
    report.applied = True
    report.ms = int((time.monotonic() - started) * 1000)
    return _rebuild(raw, words, labels), report


def score_speakers(raw: dict, samples: np.ndarray,
                   embed: Callable[[np.ndarray], Optional[np.ndarray]],
                   voiceprint: np.ndarray) -> dict[str, float]:
    """Each speaker's similarity to the rep's voiceprint, changing nothing.
    For dual-channel calls, where the channels already say who spoke when."""
    stretches = _stretches([w for w in _words(raw)
                            if w.get("start") is not None and w.get("end") is not None])
    crop = int(EMBED_MAX_SECONDS * RATE)
    for s in stretches:
        if s.seconds < ANCHOR_SECONDS:
            continue
        a = int(s.start * RATE)
        b = min(int(s.end * RATE), len(samples))
        if b - a > crop:
            a += (b - a - crop) // 2
            b = a + crop
        s.emb = embed(samples[a:b])
    return {f"speaker_{k}": round(float(v @ voiceprint), 4)
            for k, v in _centroids(stretches).items()}
