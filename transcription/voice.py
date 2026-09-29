"""
Speaker embeddings and voice activity - who a stretch of audio sounds like, and
where the speech is.

WHY
    Deepgram's diarization clusters voices on its own and cannot be told who the
    rep is. Measured 2026-09-28 on synthetic calls with exact truth
    (scripts/sca_eval): two same-gender voices collapsed into ONE speaker on 4 of
    16 calls, a rep switching English->Hindi became a SECOND speaker, and
    one-word customer replies went to the rep ~80% of the time. An embedding per
    stretch of speech lets us check and correct that, and a rep's enrolled
    voiceprint identifies the rep regardless of what they said.

THE MODEL - chosen by scripts/sca_eval/embedding_bakeoff.py
    3D-Speaker ERes2Net (VoxCeleb), via sherpa-onnx. Apache-2.0, CPU, ~26 MB,
    ~0.06 s of compute per second of audio. Against WeSpeaker ResNet34, CAM++
    and TitaNet-small it was best on short turns (92% of <=1.5 s turns placed
    with the right speaker, the backchannel case) and had 4.3% EER for a rep
    enrolled from other calls.

OPTIONAL, ALWAYS
    sherpa-onnx or the model file missing -> available() says why and every
    caller carries on without voice evidence. An analysis never fails here.

PRIVACY
    An embedding is biometric data. This module never persists anything; only
    the rep voiceprint store does, for reps who consented. Customer embeddings
    live in memory for the length of one analysis.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger("transcription.voice")

RATE = 16000

MODEL_DIR = os.environ.get("SCA_MODEL_DIR", "")
SPEAKER_MODEL = os.environ.get("SCA_SPEAKER_MODEL",
                               "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx")
VAD_MODEL = os.environ.get("SCA_VAD_MODEL", "silero_vad.onnx")
THREADS = int(os.environ.get("SCA_VOICE_THREADS", 1))

# Shorter than this carries too little voice to embed reliably.
MIN_EMBED_SECONDS = 0.3

REASON_NO_LIBRARY = "voice_library_not_installed"
REASON_NO_MODEL = "voice_model_not_installed"

_lock = threading.Lock()
_extractor = None
_extractor_error: Optional[str] = None


def _model_path(name: str) -> Optional[str]:
    if not MODEL_DIR:
        return None
    path = os.path.join(MODEL_DIR, name)
    return path if os.path.isfile(path) else None


def availability() -> tuple[bool, Optional[str]]:
    """(usable, reason when not)."""
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False, REASON_NO_LIBRARY
    if not _model_path(SPEAKER_MODEL):
        return False, REASON_NO_MODEL
    return True, None


def model_id() -> str:
    """What produced an embedding. Voiceprints from another model are not
    comparable and must be re-enrolled."""
    return SPEAKER_MODEL.rsplit(".", 1)[0]


def _get_extractor():
    global _extractor, _extractor_error
    if _extractor is not None:
        return _extractor
    with _lock:
        if _extractor is None:
            import sherpa_onnx
            path = _model_path(SPEAKER_MODEL)
            if not path:
                raise RuntimeError(REASON_NO_MODEL)
            _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=path, num_threads=THREADS))
    return _extractor


def embed(samples: np.ndarray) -> Optional[np.ndarray]:
    """Unit-length embedding of 16 kHz mono float samples, or None if too short.

    Blocking CPU work: call through asyncio.to_thread from async code.
    """
    if samples is None or len(samples) < int(MIN_EMBED_SECONDS * RATE):
        return None
    extractor = _get_extractor()
    with _lock:     # the extractor is not documented as thread-safe
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=RATE, waveform=samples.astype(np.float32))
        stream.input_finished()
        vector = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def centroid(vectors) -> Optional[np.ndarray]:
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else None


def speech_regions(samples: np.ndarray) -> Optional[list[tuple[float, float]]]:
    """(start, end) seconds of speech by Silero VAD, or None if unavailable.

    A far better denominator for "how much speech did the transcript keep" than
    the recording length, which counts hold music and silence.
    """
    path = _model_path(VAD_MODEL)
    if not path or samples is None or not len(samples):
        return None
    try:
        import sherpa_onnx
    except ImportError:
        return None
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = path
    config.silero_vad.min_silence_duration = 0.25
    config.silero_vad.min_speech_duration = 0.2
    config.sample_rate = RATE
    config.num_threads = THREADS
    window = config.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=max(
        60, int(len(samples) / RATE) + 5))
    regions: list[tuple[float, float]] = []
    data = samples.astype(np.float32)
    for i in range(0, len(data) - window + 1, window):
        vad.accept_waveform(data[i:i + window])
        while not vad.empty():
            seg = vad.front
            regions.append((seg.start / RATE, (seg.start + len(seg.samples)) / RATE))
            vad.pop()
    vad.flush()
    while not vad.empty():
        seg = vad.front
        regions.append((seg.start / RATE, (seg.start + len(seg.samples)) / RATE))
        vad.pop()
    return regions
