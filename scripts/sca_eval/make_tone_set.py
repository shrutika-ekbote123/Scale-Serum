"""
A tone test set where the WORDS carry no tone - only the voice does.

    python scripts/sca_eval/make_tone_set.py          # resumable

Every phrase is voiced in every tone, by several voices. "I don't think this
works for me" appears as frustrated, hesitant, confident, interested... so a
detector that reads the transcript can score no better than chance, and any
accuracy above chance is the voice being heard.

Gemini TTS takes a style instruction ("Say in a frustrated, irritated tone:").
Whether it really conveys the tone is checked independently of Gemini:
tone_eval.py measures pitch, loudness and pace per tone - acted frustration
that is not louder or tenser than neutral would show up there.

CREMA-D (fetch_datasets.py cremad) is the human counterpart: real actors,
same sentence, different emotion.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import random
import wave
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from _common import WORKSPACE, to_telephony, write_jsonl

TONE_DIR = WORKSPACE / "toneset"
TTS_MODEL = os.environ.get("SCA_EVAL_TTS_MODEL", "gemini-3.8-flash-tts")
RATE = 24000

TONES = {
    "neutral": "Say in a flat, neutral, matter-of-fact tone",
    "interested": "Say with genuine interest and warmth, sounding engaged and positive",
    "frustrated": "Say in a frustrated, irritated, impatient tone",
    "hesitant": "Say hesitantly and unsure, with uncertainty and small pauses in your voice",
    "confused": "Say in a confused, puzzled tone, as if you did not understand",
    "urgent": "Say urgently and fast, as if there is no time to lose",
    "confident": "Say confidently and assertively, calm and certain",
}

PHRASES = [
    ("en", "I don't think this works for me."),
    ("en", "Okay, send me the details."),
    ("en", "What is the price for this?"),
    ("en", "I will think about it and get back to you."),
    ("en", "Can you explain that part again?"),
    ("en", "When can we start?"),
    ("en", "We are already using another tool."),
    ("en", "Fine, let's do the demo tomorrow."),
    ("hi", "मुझे नहीं लगता ये मेरे लिए काम करेगा।"),
    ("hi", "ठीक है, details भेज दीजिए।"),
    ("hi", "इसका price क्या है?"),
    ("hi", "मैं सोच के बताता हूँ।"),
]
VOICES = ["Puck", "Kore", "Charon"]


def _to_float(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        frames, rate = w.readframes(w.getnframes()), w.getframerate()
    audio = np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0
    if rate != RATE:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * RATE / rate))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    level = np.abs(audio)
    loud = np.where(level > max(level.max() * 0.02, 1e-4))[0]
    return audio[max(loud[0] - 720, 0): loud[-1] + 720] if len(loud) else audio


async def _one(client, sem, pid, lang, text, tone, voice) -> dict | None:
    cid = f"p{pid:02d}_{tone}_{voice}"
    clean = TONE_DIR / f"{cid}.wav"
    tel = TONE_DIR / f"{cid}.tel.wav"
    row = {"id": cid, "source": "toneset", "audio": tel.name, "phrase_id": pid,
           "language": lang, "text": text, "tone": tone, "voice": voice}
    if tel.exists():
        return row
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    for attempt in range(5):
        try:
            async with sem:
                r = await client.aio.models.generate_content(
                    model=TTS_MODEL, contents=f"{TONES[tone]}: {text}", config=config)
            audio = _to_float(r.candidates[0].content.parts[0].inline_data.data)
            break
        except Exception as err:  # noqa: BLE001 - preview TTS rate-limits
            if attempt == 4:
                print(f"  {cid}: failed {type(err).__name__}")
                return None
            await asyncio.sleep(4 * (attempt + 1) + random.random())
    with wave.open(str(clean), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    to_telephony(clean, tel)
    row["seconds"] = round(len(audio) / RATE, 2)
    return row


async def main_async(concurrency: int) -> None:
    TONE_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sem = asyncio.Semaphore(concurrency)
    jobs = [_one(client, sem, pid, lang, text, tone, voice)
            for pid, (lang, text) in enumerate(PHRASES)
            for tone in TONES for voice in VOICES]
    rows = [r for r in await asyncio.gather(*jobs) if r]
    write_jsonl(TONE_DIR / "manifest.jsonl", rows)
    print(f"toneset: {len(rows)} clips ({len(PHRASES)} phrases x {len(TONES)} tones x {len(VOICES)} voices)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(parser.parse_args().concurrency))


if __name__ == "__main__":
    main()
