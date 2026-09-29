"""
How each turn was SAID - pace, pauses, loudness, pitch, timing - from the audio.

WHY
    The analysis read only words. "I don't think this works for me" is the same
    transcript whether it is curious, hesitant or angry, and the criteria that
    ask about tone ("Maintains a calm tone", "Listens without interrupting")
    were rated from text and timestamps. These measurements are what the voice
    adds, and what tone.py's listening step is checked against.

WHAT IS MEASURED, per segment
    rate_wps        words per second of the segment
    pause_ratio     share of the segment that is silence between words
    loudness_db     mean level of the voiced frames
    pitch_hz        median fundamental frequency of voiced frames
    pitch_range_st  spread of pitch (10th-90th percentile) in semitones - flat
                    delivery is narrow, animated or agitated delivery wide
    reply_latency   seconds between the other speaker stopping and this turn
                    starting (negative = started while they were talking)
    interruption    started over, or within a breath of, an unfinished turn
    fillers         "um", "uh", "matlab", ... in the text
  and each is also given relative to THAT SPEAKER's own average (z-score):
  absolute pitch says nothing - a deep voice is not a calm one.

Pure numpy, no model, a few milliseconds per turn. Never raises: a turn it
cannot measure simply has no values.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

RATE = 16000
FRAME = int(0.04 * RATE)        # 40 ms analysis frames
HOP = int(0.01 * RATE)          # 10 ms hop
F0_MIN, F0_MAX = 70.0, 400.0
VOICING_THRESHOLD = 0.45        # normalised autocorrelation peak
INTERRUPT_OVERLAP = 0.2         # started this long before the other stopped
CUT_IN_GAP = 0.15               # or this soon after an unfinished sentence
VOICED_RANGE_DB = 25.0
SILENCE_DB = -60.0              # full scale is 0 dB

_FILLERS = re.compile(
    r"\b(um+|uh+|umm+|uhh+|hmm+|er+|ah+|matlab|मतलब|वो|अं|umm|you know)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.?!।]\s*$")


@dataclass
class SegmentProsody:
    index: int
    speaker_id: str
    rate_wps: Optional[float] = None
    pause_ratio: Optional[float] = None
    loudness_db: Optional[float] = None
    pitch_hz: Optional[float] = None
    pitch_range_st: Optional[float] = None
    reply_latency: Optional[float] = None
    interruption: bool = False
    fillers: int = 0
    z: dict = field(default_factory=dict)       # measure -> z-score within speaker

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class SpeakerProsody:
    speaker_id: str
    talk_share: Optional[float] = None
    rate_wpm: Optional[float] = None
    median_reply_latency: Optional[float] = None
    interruptions_made: int = 0
    pitch_hz: Optional[float] = None
    pitch_range_st: Optional[float] = None
    loudness_db: Optional[float] = None
    fillers_per_min: Optional[float] = None

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        return np.zeros((0, FRAME), np.float32)
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def _pitch(frames: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """F0 per voiced frame by normalised autocorrelation. Crude, consistent,
    and good enough for "higher / wider than this speaker usually is"."""
    lo, hi = int(RATE / F0_MAX), int(RATE / F0_MIN)
    out = []
    window = np.hanning(FRAME)
    for f in frames[voiced]:
        f = (f - f.mean()) * window
        energy = float(f @ f)
        if energy <= 0:
            continue
        ac = np.correlate(f, f, mode="full")[FRAME - 1:]
        ac = ac / ac[0]
        seg = ac[lo:hi]
        if not len(seg):
            continue
        lag = int(np.argmax(seg)) + lo
        if ac[lag] >= VOICING_THRESHOLD:
            out.append(RATE / lag)
    return np.array(out)


def measure_segment(samples: np.ndarray, start: float, end: float, text: str) -> dict:
    """Acoustic measures of one stretch of audio. Empty dict if too short."""
    a, b = max(int(start * RATE), 0), min(int(end * RATE), len(samples))
    x = samples[a:b].astype(np.float32)
    if len(x) < FRAME * 3:
        return {}
    frames = _frames(x)
    rms = np.sqrt((frames ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(rms)
    # Voiced = within VOICED_RANGE_DB of the loudest frame and above absolute
    # silence. (Relative to the segment's own quietest frames, a turn with no
    # pause in it had no voiced frames at all.)
    voiced = (db > db.max() - VOICED_RANGE_DB) & (db > SILENCE_DB)
    out: dict = {}
    words = len(text.split())
    seconds = end - start
    if seconds > 0:
        out["rate_wps"] = words / seconds
    out["pause_ratio"] = float(1 - voiced.mean())
    if voiced.any():
        out["loudness_db"] = float(db[voiced].mean())
        f0 = _pitch(frames, voiced)
        if len(f0) >= 5:
            out["pitch_hz"] = float(np.median(f0))
            p10, p90 = np.percentile(f0, [10, 90])
            out["pitch_range_st"] = float(12 * math.log2(p90 / p10)) if p10 > 0 else None
    out["fillers"] = len(_FILLERS.findall(text or ""))
    return out


def analyse(transcript, samples: np.ndarray) -> tuple[list[SegmentProsody], list[SpeakerProsody]]:
    """Per-segment and per-speaker measures for a NormalizedTranscript."""
    segs = [s for s in transcript.segments if s.start is not None and s.end is not None]
    results: list[SegmentProsody] = []
    previous = None
    for s in segs:
        m = measure_segment(samples, s.start, s.end, s.text)
        sp = SegmentProsody(index=s.index, speaker_id=s.speaker_id,
                            rate_wps=m.get("rate_wps"), pause_ratio=m.get("pause_ratio"),
                            loudness_db=m.get("loudness_db"), pitch_hz=m.get("pitch_hz"),
                            pitch_range_st=m.get("pitch_range_st"), fillers=m.get("fillers", 0))
        if previous is not None and previous.speaker_id != s.speaker_id:
            sp.reply_latency = s.start - previous.end
            unfinished = not _SENTENCE_END.search(previous.text or "")
            sp.interruption = (sp.reply_latency < -INTERRUPT_OVERLAP
                               or (unfinished and sp.reply_latency < CUT_IN_GAP))
        results.append(sp)
        previous = s

    # Relative to each speaker's own delivery.
    for measure in ("rate_wps", "pause_ratio", "loudness_db", "pitch_hz", "pitch_range_st"):
        by_speaker: dict[str, list[float]] = {}
        for r in results:
            v = getattr(r, measure)
            if v is not None:
                by_speaker.setdefault(r.speaker_id, []).append(v)
        for r in results:
            v, values = getattr(r, measure), by_speaker.get(r.speaker_id, [])
            if v is not None and len(values) >= 3:
                sd = float(np.std(values))
                if sd > 0:
                    r.z[measure] = round((v - float(np.mean(values))) / sd, 2)

    total_talk = sum(s.end - s.start for s in segs) or 1.0
    speakers = []
    for sid in dict.fromkeys(s.speaker_id for s in segs):
        mine = [s for s in segs if s.speaker_id == sid]
        rows = [r for r in results if r.speaker_id == sid]
        talk = sum(s.end - s.start for s in mine)
        words = sum(len(s.text.split()) for s in mine)
        latencies = [r.reply_latency for r in rows if r.reply_latency is not None]

        def med(attr):
            vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
            return float(np.median(vals)) if vals else None

        speakers.append(SpeakerProsody(
            speaker_id=sid, talk_share=talk / total_talk,
            rate_wpm=(words / talk * 60) if talk > 0 else None,
            median_reply_latency=float(np.median(latencies)) if latencies else None,
            interruptions_made=sum(1 for r in rows if r.interruption),
            pitch_hz=med("pitch_hz"), pitch_range_st=med("pitch_range_st"),
            loudness_db=med("loudness_db"),
            fillers_per_min=(sum(r.fillers for r in rows) / talk * 60) if talk > 0 else None))
    return results, speakers
