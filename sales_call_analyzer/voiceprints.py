"""
Rep voiceprints - enrolling a rep's voice so calls can recognise them by it.

WHY
    Measured on synthetic calls with exact truth (scripts/sca_eval): with the
    rep's voiceprint, role accuracy rose from 78% to 93% and every one of 16
    calls had both roles right, including the four where Deepgram had merged
    two same-gender voices into one speaker. Without one, voices can only be
    compared with each other, which could not reliably tell a rep who switched
    language from a second person.

    It also makes the rep independent of the CRM: on the real test calls the
    rep named on the record was often not the person who made the call.

WHAT IS STORED
    One 192-number vector per rep, the model that produced it, how much speech
    it came from, and who confirmed consent and when. Never audio, never a
    transcript, never anything about a customer. A voiceprint is biometric
    data: it is created only with consent recorded, can be read back only as
    status (never the vector), and is deleted on request.

ENROLMENT QUALITY
    A voiceprint built from the wrong voice would mislabel every future call,
    so a sample is refused when it is:
      * too short         - under MIN_SECONDS of speech
      * not one voice     - its windows split into two clearly different voices
      * inconsistent      - its windows do not agree with their own average
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from . import diarization as diar

logger = logging.getLogger("sales_call_analyzer.voiceprints")

RATE = 16000
MIN_SECONDS = float(os.environ.get("SCA_VOICEPRINT_MIN_SECONDS", 20))
WINDOW_SECONDS = 3.0
MIN_WINDOW_SECONDS = 1.0
MAX_WINDOWS = int(os.environ.get("SCA_VOICEPRINT_MAX_WINDOWS", 60))
# Measured on 16 synthetic calls, enrolling from the rep's turns only (clean) and
# from the whole call (mixed: rep + customer), 2026-09-28:
#                         clean          mixed
#   two-voice similarity  0.46 - 0.88    -0.04 - 0.50
#   consistency           0.665 - 0.79   0.54 - 0.68
# Rejecting below 0.45 similarity (with a real minority) or 0.65 consistency
# accepted all 16 clean samples and refused 14 of 16 mixed ones. The margins are
# thin, so prefer explicit segments or an analysis speaker over a whole call.
TWO_VOICES_BELOW = float(os.environ.get("SCA_VOICEPRINT_TWO_VOICES_BELOW", 0.45))
TWO_VOICES_MIN_SHARE = 0.15
MIN_CONSISTENCY = float(os.environ.get("SCA_VOICEPRINT_MIN_CONSISTENCY", 0.65))

REJECT_TOO_SHORT = "voiceprint_sample_too_short"
REJECT_TWO_VOICES = "voiceprint_sample_has_more_than_one_voice"
REJECT_INCONSISTENT = "voiceprint_sample_inconsistent"
REJECT_NO_SPEECH = "voiceprint_sample_has_no_speech"


class EnrolmentRejected(Exception):
    def __init__(self, reason: str, message: str, stats: Optional[dict] = None):
        super().__init__(message)
        self.reason, self.message, self.stats = reason, message, stats or {}


def _windows(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for start, end in spans:
        t = start
        while end - t >= MIN_WINDOW_SECONDS:
            out.append((t, min(t + WINDOW_SECONDS, end)))
            t += WINDOW_SECONDS
    return out


def build(samples: np.ndarray, spans: list[tuple[float, float]],
          embed: Callable[[np.ndarray], Optional[np.ndarray]]) -> tuple[np.ndarray, dict]:
    """(unit voiceprint, stats) from the rep's speech in `spans`, or raise
    EnrolmentRejected. Blocking CPU work."""
    windows = _windows(sorted(spans))
    speech = sum(b - a for a, b in windows)
    if speech < MIN_SECONDS:
        raise EnrolmentRejected(
            REJECT_TOO_SHORT,
            f"The sample holds {speech:.0f} s of usable speech; {MIN_SECONDS:.0f} s is needed.",
            {"speech_seconds": round(speech, 1)})
    if len(windows) > MAX_WINDOWS:          # evenly spread, bounded cost
        step = len(windows) / MAX_WINDOWS
        windows = [windows[int(i * step)] for i in range(MAX_WINDOWS)]

    vectors = []
    for a, b in windows:
        v = embed(samples[int(a * RATE):int(b * RATE)])
        if v is not None:
            vectors.append(v)
    if len(vectors) < 3:
        raise EnrolmentRejected(REJECT_NO_SPEECH, "No speech could be measured in the sample.")

    matrix = np.stack(vectors)
    labels, ca, cb = diar._two_means(matrix)
    minority = min(labels.mean(), 1 - labels.mean())
    if float(ca @ cb) < TWO_VOICES_BELOW and minority >= TWO_VOICES_MIN_SHARE:
        raise EnrolmentRejected(
            REJECT_TWO_VOICES,
            "The sample contains more than one voice. Use a recording, or a part of "
            "one, where only this rep speaks.",
            {"voice_similarity": round(float(ca @ cb), 3), "minority_share": round(float(minority), 3)})

    mean = matrix.mean(0)
    voiceprint = mean / np.linalg.norm(mean)
    consistency = float(np.mean(matrix @ voiceprint))
    stats = {"speech_seconds": round(speech, 1), "windows": len(vectors),
             "consistency": round(consistency, 3)}
    if consistency < MIN_CONSISTENCY:
        raise EnrolmentRejected(
            REJECT_INCONSISTENT,
            "The sample's speech does not sound like one consistent voice.", stats)
    return voiceprint.astype(np.float32), stats


def combine(old: np.ndarray, old_seconds: float, new: np.ndarray,
            new_seconds: float) -> np.ndarray:
    """One voiceprint from two samples, weighted by their speech.

    WHY ENROL FROM MORE THAN ONE CALL
        The embedding model shifts with language. Measured on a synthetic rep
        enrolled from English speech: their English stretches on a new call
        scored 0.73 against the voiceprint, their Hindi stretches 0.34 - close
        to the customer's 0.18. A rep who sells in Hindi and English should be
        enrolled from speech in both.
    """
    mixed = old * max(old_seconds, 1e-6) + new * max(new_seconds, 1e-6)
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


class VoiceprintStore:
    """`sca_rep_voiceprints`, one document per rep, keyed by the CRM rep id."""

    def __init__(self, collection):
        self.collection = collection

    async def save(self, *, rep_id: str, vector: np.ndarray, model: str, stats: dict,
                   consent: dict, source: dict, brand_id: Optional[str] = None,
                   rep_name: Optional[str] = None, add_to_existing: bool = False) -> dict:
        """Create or replace a voiceprint - or, with add_to_existing, fold this
        sample into the one already enrolled under the same model."""
        now = datetime.now(timezone.utc)
        sources = [source]
        existing = await self.get(rep_id) if add_to_existing else None
        if existing and existing.get("model") == model and existing.get("vector"):
            old_seconds = float((existing.get("stats") or {}).get("speech_seconds") or 0)
            vector = combine(np.asarray(existing["vector"], dtype=np.float32), old_seconds,
                             vector, float(stats.get("speech_seconds") or 0))
            stats = {**stats, "speech_seconds": round(old_seconds + float(stats.get("speech_seconds") or 0), 1),
                     "samples": int((existing.get("stats") or {}).get("samples") or 1) + 1}
            sources = list(existing.get("sources") or [existing.get("source")]) + [source]
        else:
            stats = {**stats, "samples": 1}
        doc = {
            "_id": rep_id, "rep_id": rep_id, "brand_id": brand_id, "rep_name": rep_name,
            "model": model, "dims": int(vector.shape[0]),
            "vector": [round(float(x), 6) for x in vector],
            "stats": stats, "source": source, "sources": sources[-20:],
            "consent": {**consent, "recorded_at": now},
            "updated_at": now,
        }
        await self.collection.update_one({"_id": rep_id},
                                         {"$set": doc, "$setOnInsert": {"created_at": now}},
                                         upsert=True)
        return status(doc)

    async def get(self, rep_id: str) -> Optional[dict]:
        return await self.collection.find_one({"_id": rep_id})

    async def vector_for(self, rep_id: Optional[str], model: str) -> Optional[np.ndarray]:
        """The rep's voiceprint, if enrolled with the model in use. One from a
        different model is not comparable and is ignored, not guessed at."""
        if not rep_id:
            return None
        doc = await self.get(rep_id)
        if not doc or doc.get("model") != model or not doc.get("vector"):
            return None
        return np.asarray(doc["vector"], dtype=np.float32)

    async def delete(self, rep_id: str) -> bool:
        result = await self.collection.delete_one({"_id": rep_id})
        return bool(getattr(result, "deleted_count", 0))


def status(doc: Optional[dict], current_model: Optional[str] = None) -> dict:
    """What an API caller may see: never the vector."""
    if not doc:
        return {"enrolled": False}
    out = {
        "enrolled": True, "rep_id": doc.get("rep_id"), "brand_id": doc.get("brand_id"),
        "rep_name": doc.get("rep_name"), "model": doc.get("model"),
        "stats": doc.get("stats"), "source": doc.get("source"),
        "sources": doc.get("sources"),
        "consent": doc.get("consent"),
        "created_at": doc.get("created_at"), "updated_at": doc.get("updated_at"),
    }
    if current_model is not None:
        # Enrolled under a model since replaced: present, but not used.
        out["usable"] = doc.get("model") == current_model
    return out
