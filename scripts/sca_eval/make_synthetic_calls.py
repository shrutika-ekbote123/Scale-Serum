"""
Generate two-speaker sales calls with EXACT ground truth, for diarization and
role accuracy.

    python scripts/sca_eval/make_synthetic_calls.py            # all scenarios, resumes
    python scripts/sca_eval/make_synthetic_calls.py --only en_kn_1 hinglish_1

WHY SYNTHETIC
    Diarization and role assignment need to know, for every second, who really
    spoke. No public Indian dataset available to us has two-party code-switched
    sales calls with speaker labels, and our 28 real calls have no labels until a
    human writes them. So: Gemini writes a realistic code-switched dialogue, and
    Gemini TTS speaks each turn with ONE fixed voice per role. Turns are joined
    with realistic gaps, so every turn's start, end, speaker, role, language and
    reference text is known exactly. Then it is degraded to 8 kHz mu-law.

    Each call deliberately contains what breaks today's pipeline: short
    customer backchannels ("haan", "ji"), a rep self-introduction and brand line,
    the customer addressed by name, a language switch, an objection and a price.
    Half the scenarios use two voices of the same gender, the hardest case.

WHAT IT CANNOT TELL YOU
    TTS is cleaner and more regular than a person on a phone. Word error rates
    here are optimistic; use them to compare configurations, and use FLEURS and
    the labelled real calls for absolute accuracy.

COST
    About $0.03 of TTS per two-minute call plus a fraction of a cent for the
    script. The full set is well under a dollar. Resumable: finished calls are
    skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import random
import wave
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from _common import SYNTH_DIR, read_json, to_telephony, write_json, write_jsonl

SCRIPT_MODEL = os.environ.get("SCA_EVAL_SCRIPT_MODEL", os.environ.get("GEMINI_MODEL", "gemini-flash-latest"))
TTS_MODEL = os.environ.get("SCA_EVAL_TTS_MODEL", "gemini-3.8-flash-tts")
RATE = 24000

LANG_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi", "kn": "Kannada",
              "pa": "Punjabi", "ta": "Tamil", "te": "Telugu"}
SCRIPTS = {"hi": "Devanagari", "mr": "Devanagari", "kn": "Kannada script",
           "pa": "Gurmukhi", "ta": "Tamil script", "te": "Telugu script"}

MALE = ["Puck", "Charon", "Fenrir", "Orus"]
FEMALE = ["Kore", "Aoede", "Leda", "Zephyr"]

# (id, customer language, pattern, same-gender voices?)
SCENARIOS = [
    ("en_only_1", "en", "english_only", False),
    ("en_only_2", "en", "english_only", True),
    ("en_only_3", "en", "english_only", True),
    ("en_to_hi_1", "hi", "english_then_switch", False),
    ("en_to_hi_2", "hi", "english_then_switch", True),
    ("en_to_hi_3", "hi", "english_then_switch", False),
    ("hinglish_1", "hi", "hinglish_throughout", True),
    ("hinglish_2", "hi", "hinglish_throughout", False),
    ("en_mr_1", "mr", "rep_english_customer_regional", False),
    ("en_mr_2", "mr", "rep_english_customer_regional", True),
    ("en_kn_1", "kn", "rep_english_customer_regional", False),
    ("en_kn_2", "kn", "rep_english_customer_regional", True),
    ("en_pa_1", "pa", "rep_english_customer_regional", False),
    ("en_pa_2", "pa", "rep_english_customer_regional", True),
    ("en_ta_1", "ta", "rep_english_customer_regional", False),
    ("en_te_1", "te", "rep_english_customer_regional", True),
    # Both sides code-switch - the real mycall4.mp3 pattern, which the calls
    # above (rep in English only) could not show. Added 2026-09-29.
    ("both_mr_1", "mr", "both_code_switch", False),
    ("both_kn_1", "kn", "both_code_switch", True),
    ("both_pa_1", "pa", "both_code_switch", False),
]

PATTERNS = {
    "english_only": "The whole call is in Indian English. Neither person uses Hindi.",
    "english_then_switch": ("The rep opens in English. The customer answers in English at first, "
                            "then switches to {lang} (with the usual English loanwords) about a "
                            "third of the way in, and the rep follows into {lang} mixed with English."),
    "hinglish_throughout": ("Both speak Hinglish throughout: Hindi grammar with many English words, "
                            "switching mid-sentence, as Indian sales calls really sound."),
    "both_code_switch": ("The rep opens in English and then, like the customer, speaks mostly {lang} "
                         "mixed with English - {lang} for explanations and small talk, English for "
                         "product terms, numbers and some whole sentences. Both switch mid-sentence."),
    "rep_english_customer_regional": ("The rep speaks English throughout. The customer mostly speaks "
                                      "{lang}, with English loanwords, and a few English sentences."),
}

SCRIPT_PROMPT = """Write a realistic Indian outbound sales phone call for evaluation data.

Company: {brand}. Product: a {product}.
Sales representative: {rep}. Customer: {customer}.
Language pattern: {pattern}

Requirements:
- 16 to 22 turns, alternating speakers, about two minutes when spoken.
- Turn 1 is the rep greeting the customer BY NAME. In turn 1 or 3 the rep introduces
  themself by name and says they are calling from {brand}.
- Include 3 or 4 customer turns that are ONLY a short backchannel of one or two words
  (for example "haan", "ji", "okay", "hmm", "achha", or the equivalent in {lang}).
- Include one clear objection from the customer (price, time or trust), a price being
  stated, and a next step agreed or declined at the end.
- Write {lang} words in {script}. Write English words in Latin letters, even inside a
  {lang} sentence. Do not add translations, stage directions or speaker names inside text.
- lang for each turn is the main language of that turn: one of en, {code}, or "mixed".
"""

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {"turns": {"type": "array", "items": {
        "type": "object",
        "properties": {"role": {"type": "string", "enum": ["rep", "customer"]},
                       "lang": {"type": "string"},
                       "text": {"type": "string"}},
        "required": ["role", "lang", "text"]}}},
    "required": ["turns"],
}

REPS = ["Rohan Mehta", "Priya Nair", "Arjun Kulkarni", "Sneha Iyer", "Vikram Singh", "Kavya Rao"]
CUSTOMERS = {"en": ["Sanjay Kapoor", "Meera Joshi"], "hi": ["Rakesh Sharma", "Pooja Verma"],
             "mr": ["Sachin Patil", "Aarti Deshmukh"], "kn": ["Manjunath Gowda", "Shwetha Hegde"],
             "pa": ["Gurpreet Singh", "Harleen Kaur"], "ta": ["Karthik Subramanian"],
             "te": ["Srinivas Reddy"]}
PRODUCTS = ["12-week online leadership programme for company directors",
            "marketing analytics subscription for small businesses",
            "certificate course in corporate law for working professionals"]


def _wav_to_float(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        assert w.getsampwidth() == 2, "expected 16-bit TTS audio"
        frames = w.readframes(w.getnframes())
        rate, channels = w.getframerate(), w.getnchannels()
    audio = np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(1)
    if rate != RATE:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * RATE / rate))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _trim(audio: np.ndarray) -> np.ndarray:
    """TTS pads with silence; trim it so the truth timings mean speech."""
    level = np.abs(audio)
    loud = np.where(level > max(level.max() * 0.02, 1e-4))[0]
    if not len(loud):
        return audio
    pad = int(0.03 * RATE)
    return audio[max(loud[0] - pad, 0): loud[-1] + pad]


async def _script(client, scenario, rng) -> dict:
    sid, code, pattern, _ = scenario
    lang = LANG_NAMES[code]
    rep = rng.choice(REPS)
    customer = rng.choice(CUSTOMERS[code])
    product = rng.choice(PRODUCTS)
    prompt = SCRIPT_PROMPT.format(
        brand="ScaleSerum", product=product, rep=rep, customer=customer,
        pattern=PATTERNS[pattern].format(lang=lang), lang=lang,
        script=SCRIPTS.get(code, "Latin letters"), code=code)
    response = await client.aio.models.generate_content(
        model=SCRIPT_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.9, response_mime_type="application/json",
                                           response_schema=SCRIPT_SCHEMA))
    turns = [t for t in json.loads(response.text)["turns"] if (t.get("text") or "").strip()]
    return {"rep_name": rep, "customer_name": customer, "brand_name": "ScaleSerum",
            "product": product, "turns": turns}


async def _speak(client, text: str, voice: str, sem: asyncio.Semaphore) -> np.ndarray:
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    for attempt in range(5):
        try:
            async with sem:
                response = await client.aio.models.generate_content(
                    model=TTS_MODEL, contents=text, config=config)
            data = response.candidates[0].content.parts[0].inline_data.data
            return _trim(_wav_to_float(data))
        except Exception as err:  # noqa: BLE001 - preview TTS rate-limits often
            if attempt == 4:
                raise
            await asyncio.sleep(4 * (attempt + 1) + random.random())
            print(f"    tts retry {attempt + 1}: {type(err).__name__}")
    raise RuntimeError("unreachable")


async def _make(client, scenario, sem) -> None:
    sid, code, pattern, same_gender = scenario
    truth_path = SYNTH_DIR / f"{sid}.truth.json"
    if truth_path.exists() and (SYNTH_DIR / f"{sid}.tel.wav").exists():
        print(f"  {sid}: exists")
        return
    rng = random.Random(sid)
    script_path = SYNTH_DIR / f"{sid}.script.json"
    script = read_json(script_path) if script_path.exists() else await _script(client, scenario, rng)
    write_json(script_path, script)

    rep_voice = rng.choice(MALE + FEMALE)
    pool = (MALE if rep_voice in MALE else FEMALE) if same_gender else (FEMALE if rep_voice in MALE else MALE)
    customer_voice = rng.choice([v for v in pool if v != rep_voice])
    voices = {"rep": rep_voice, "customer": customer_voice}

    clips = await asyncio.gather(*[_speak(client, t["text"], voices[t["role"]], sem)
                                   for t in script["turns"]])

    pieces, turns, cursor = [np.zeros(int(0.8 * RATE), np.float32)], [], 0.8
    previous_role = None
    for turn, clip in zip(script["turns"], clips):
        if previous_role is not None:
            # Same speaker continuing: a breath. Speaker change: a real reply gap.
            gap = rng.uniform(0.15, 0.4) if turn["role"] == previous_role else rng.uniform(0.25, 0.9)
            pieces.append(np.zeros(int(gap * RATE), np.float32))
            cursor += gap
        start = cursor
        pieces.append(clip)
        cursor += len(clip) / RATE
        turns.append({"role": turn["role"], "speaker": turn["role"], "lang": turn["lang"],
                      "text": turn["text"], "start": round(start, 3), "end": round(cursor, 3)})
        previous_role = turn["role"]
    pieces.append(np.zeros(int(0.8 * RATE), np.float32))
    audio = np.concatenate(pieces)
    audio = audio / max(np.abs(audio).max(), 1e-6) * 0.8

    clean = SYNTH_DIR / f"{sid}.wav"
    with wave.open(str(clean), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    to_telephony(clean, SYNTH_DIR / f"{sid}.tel.wav")

    write_json(truth_path, {
        "id": sid, "customer_language": code, "pattern": pattern, "same_gender_voices": same_gender,
        "voices": voices, "rep_name": script["rep_name"], "customer_name": script["customer_name"],
        "brand_name": script["brand_name"], "product": script["product"],
        "duration_seconds": round(len(audio) / RATE, 3), "turns": turns})
    print(f"  {sid}: {len(turns)} turns, {len(audio) / RATE:.0f}s, voices {voices}")


async def main_async(only: list[str], concurrency: int) -> None:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sem = asyncio.Semaphore(concurrency)
    chosen = [s for s in SCENARIOS if not only or s[0] in only]
    for scenario in chosen:        # one call at a time; turns within it in parallel
        try:
            await _make(client, scenario, sem)
        except Exception as err:  # noqa: BLE001 - keep going, rerun resumes
            print(f"  {scenario[0]}: FAILED {type(err).__name__}: {str(err)[:200]}")

    rows = []
    for sid, code, pattern, same in SCENARIOS:
        truth = SYNTH_DIR / f"{sid}.truth.json"
        if truth.exists():
            t = read_json(truth)
            rows.append({"id": sid, "source": "synthetic", "audio": f"{sid}.tel.wav",
                         "truth": truth.name, "language": code, "pattern": pattern,
                         "same_gender_voices": same, "rep_name": t["rep_name"],
                         "customer_name": t["customer_name"], "brand_name": t["brand_name"],
                         "product": t["product"], "duration_seconds": t["duration_seconds"]})
    write_jsonl(SYNTH_DIR / "manifest.jsonl", rows)
    print(f"synthetic: {len(rows)} calls in manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(main_async(args.only, args.concurrency))


if __name__ == "__main__":
    main()
