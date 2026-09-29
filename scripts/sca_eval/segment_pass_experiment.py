"""
Phase 2 experiment: re-transcribing the customer's turns in their own language.

    python scripts/sca_eval/segment_pass_experiment.py

THE PROBLEM (Phase 0 baseline)
    On a call where the rep speaks English and the customer speaks Kannada,
    Marathi, Punjabi, Tamil or Telugu, `multi` keeps only 50-80% of the
    customer's words and gets about half of those wrong. Sending the regional
    code for the WHOLE call wrecks the rep's English instead.

WHAT IS COMPARED, per customer segment of the 9 regional synthetic calls
    pass1     today's transcript (nova-3 multi + keyterms)
    regional  the non-rep segments cut out, joined with silence into ONE audio
              file, sent once with the regional language code, words mapped
              back to segments by their time offset
    gemini    the same clips sent to Gemini as separate numbered audio parts,
              one transcription per part - Deepgram supplies the timing, which
              is what Gemini could not be trusted with
    best      per segment, whichever candidate is closest to the truth: the
              ceiling any selection rule can reach

Segments are chosen WITHOUT the truth: those whose speaker the product did not
label sales_rep (all of them when no rep was found).
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import wave
from collections import defaultdict

import httpx
import numpy as np

from _common import RUNS_DIR, SYNTH_DIR, decode_pcm, read_json, read_jsonl, write_json
import metrics as M

from sales_call_analyzer import speakers as sp
from sales_call_analyzer import transcript as tr
from sales_call_analyzer.models import CustomerInfo, RepInfo
from transcription import deepgram_client as dg
from transcription import voice
from sales_call_analyzer import diarization as diar
os.environ.setdefault("SCA_EVAL_ENROL_CALLS", "2")
import run_eval as E  # noqa: E402  (voiceprint enrolment + embedding cache)

RATE = 16000
GAP = 0.6
PAD = 0.15
GEMINI_MODEL = os.environ.get("SCA_EVAL_GEMINI_MODEL", os.environ.get("GEMINI_MODEL", "gemini-flash-latest"))

GEMINI_PROMPT = """You are given {n} numbered audio clips from one phone sales call in India.
The speaker mostly uses {lang}, mixed with English words.

Transcribe each clip VERBATIM.
- Write {lang} words in {script}. Write English words in Latin letters, even inside a {lang} sentence.
- Do not translate, summarise, correct grammar or add words that are not spoken.
- If a clip has no intelligible speech, return an empty string for it.
Return JSON: {{"clips": [{{"n": 1, "text": "..."}}, ...]}} with one entry per clip, in order."""

LANG = {"kn": ("Kannada", "Kannada script"), "mr": ("Marathi", "Devanagari"),
        "pa": ("Punjabi", "Gurmukhi"), "ta": ("Tamil", "Tamil script"),
        "te": ("Telugu", "Telugu script"), "hi": ("Hindi", "Devanagari")}


def _wav(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _cache(kind: str, key: str):
    return RUNS_DIR / "segpass" / kind / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


async def deepgram_joined(clips, language, keyterms, http):
    """One request for all clips. Returns per-clip text."""
    pieces, offsets, t = [], [], 0.0
    for c in clips:
        offsets.append((t, t + len(c) / RATE))
        pieces += [c, np.zeros(int(GAP * RATE), np.float32)]
        t += len(c) / RATE + GAP
    audio = _wav(np.concatenate(pieces))
    params = dg.build_params(language, keyterms)
    path = _cache("deepgram", hashlib.sha1(audio).hexdigest() + json.dumps(params, sort_keys=True))
    if path.exists():
        raw = read_json(path)
    else:
        for attempt in range(5):
            try:
                r = await http.post(dg.DEEPGRAM_ENDPOINT, params=params, content=audio,
                                    headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
                                             "Content-Type": "audio/wav"})
                if r.status_code == 200:
                    raw = r.json()
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(4 * (attempt + 1))
        else:
            raise RuntimeError("deepgram failed")
        write_json(path, raw)
    words = M.deepgram_words(raw)
    out = [[] for _ in clips]
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        for i, (a, b) in enumerate(offsets):
            if a - 0.1 <= mid <= b + 0.1:
                out[i].append(w["text"])
                break
    confs = [[] for _ in clips]
    for w in (raw["results"]["channels"][0]["alternatives"][0].get("words") or []):
        mid = (w["start"] + w["end"]) / 2
        for i, (a, b) in enumerate(offsets):
            if a - 0.1 <= mid <= b + 0.1:
                confs[i].append(w.get("confidence") or 0)
                break
    return [" ".join(x) for x in out], [float(np.mean(c)) if c else 0.0 for c in confs]


async def gemini_clips(clips, lang_code, client, sem):
    from google.genai import types
    name, script = LANG.get(lang_code, ("the customer's language", "its own script"))
    prompt = GEMINI_PROMPT.format(n=len(clips), lang=name, script=script)
    parts = []
    for i, c in enumerate(clips, 1):
        parts += [f"Clip {i}:", types.Part.from_bytes(data=_wav(c), mime_type="audio/wav")]
    key = hashlib.sha1(b"".join(_wav(c) for c in clips) + prompt.encode() + GEMINI_MODEL.encode()).hexdigest()
    path = _cache("gemini", key)
    if path.exists():
        return read_json(path)
    schema = {"type": "object", "properties": {"clips": {"type": "array", "items": {
        "type": "object", "properties": {"n": {"type": "integer"}, "text": {"type": "string"}},
        "required": ["n", "text"]}}}, "required": ["clips"]}
    for attempt in range(5):
        try:
            async with sem:
                r = await client.aio.models.generate_content(
                    model=GEMINI_MODEL, contents=parts + [prompt],
                    config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                                       response_schema=schema))
            got = {c["n"]: c["text"] for c in json.loads(r.text)["clips"]}
            texts = [got.get(i, "") for i in range(1, len(clips) + 1)]
            u = r.usage_metadata
            result = {"texts": texts, "input_tokens": u.prompt_token_count,
                      "output_tokens": u.candidates_token_count, "thinking": u.thoughts_token_count}
            write_json(path, result)
            return result
        except Exception as err:  # noqa: BLE001
            print("   gemini retry", type(err).__name__)
            await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError("gemini failed")


def _ref_for(seg, turns):
    """All truth words whose turn midpoint falls in the segment."""
    return " ".join(t["text"] for t in turns
                    if seg.start - 0.3 <= (t["start"] + t["end"]) / 2 <= seg.end + 0.3)


async def main() -> None:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sem = asyncio.Semaphore(2)
    totals = defaultdict(dict)
    per_lang = defaultdict(lambda: defaultdict(dict))
    selections = defaultdict(list)
    tokens = []
    async with httpx.AsyncClient(timeout=300) as http:
        for row in read_jsonl(SYNTH_DIR / "manifest.jsonl"):
            code = row["language"]
            if code in ("en", "hi"):
                continue
            truth = read_json(SYNTH_DIR / row["truth"])
            raws = [read_json(p) for p in (RUNS_DIR / "raw" / "synthetic" / row["id"]).glob("*.json")]
            raw = next(r["response"] for r in raws
                       if r["params"].get("keyterm") and r["params"].get("language") == "multi")
            keyterms = raws[0]["params"].get("keyterm") or []
            # Phase 1 first, as production will: speaker refinement with the
            # rep's voiceprint (two enrolment calls), so collapsed calls have a
            # customer to re-transcribe.
            samples = decode_pcm(SYNTH_DIR / row["audio"])[:, 0]
            item = {"set": "synthetic", "id": row["id"], "truth": row["truth"],
                    "path": str(SYNTH_DIR / row["audio"])}
            voiceprint = E.enrolled_voiceprint(item)
            raw, _ = diar.refine(raw, samples, E.cached_embedder(item), voiceprint, voice.model_id())
            t = tr.from_deepgram(raw)
            t = sp.resolve_roles(t, rep=RepInfo(name=row["rep_name"]),
                                 customer=CustomerInfo(name=row["customer_name"]),
                                 brand_name=row["brand_name"])
            reps = {s.speaker_id for s in t.speakers if s.role == "sales_rep"}
            segs = [s for s in t.segments if s.speaker_id not in reps and s.end and s.end - s.start >= 0.3]
            if not segs:
                print(f"  {row['id']:9} no customer segments")
                continue
            clips = [samples[max(int((s.start - PAD) * RATE), 0):int((s.end + PAD) * RATE)] for s in segs]

            regional, confs = await deepgram_joined(clips, code, keyterms, http)
            multi_joined, confs_multi = await deepgram_joined(clips, "multi", keyterms, http)
            gem = await gemini_clips(clips, code, client, sem)
            tokens.append(gem)

            for seg, r_text, m2_text, g_text, c_r, c_m in zip(segs, regional, multi_joined,
                                                              gem["texts"], confs, confs_multi):
                ref = _ref_for(seg, truth["turns"])
                if not ref.strip():
                    continue
                cands = {"pass1": seg.text, "regional": r_text, "gemini": g_text}
                scores = {k: M.text_scores(ref, v)["roman"] for k, v in cands.items()}
                best = min(scores, key=lambda k: scores[k]["word_edits"])
                choice = {**scores, "best": scores[best],
                          # deterministic rule: regional when its confidence beats multi's on the same clip
                          "conf_rule": scores["regional"] if c_r > c_m + 0.02 else scores["pass1"]}
                selections["best"].append(best)
                for k, v in choice.items():
                    M.add_counts(totals[k], v)
                    M.add_counts(per_lang[code][k], v)
            print(f"  {row['id']:9} {len(segs)} customer segments, "
                  f"{sum(len(c) for c in clips) / RATE:.0f}s of audio")

    print("\nCUSTOMER SEGMENTS, roman   WER   CER   words kept")
    for k in ("pass1", "regional", "gemini", "conf_rule", "best"):
        r = M.rates(totals[k])
        print(f"  {k:10} {r['wer']:.3f} {r['cer']:.3f} {r['length_ratio']:.2f}")
    print("\nby language (WER):", "  ".join(["pass1", "regional", "gemini", "best"]))
    for code, d in sorted(per_lang.items()):
        print(f"  {code}  " + "  ".join(f"{M.rates(d[k])['wer']:.3f}" for k in ("pass1", "regional", "gemini", "best")))
    from collections import Counter
    print("\nbest candidate per segment:", Counter(selections["best"]))
    it = sum(x["input_tokens"] or 0 for x in tokens)
    ot = sum((x["output_tokens"] or 0) + (x["thinking"] or 0) for x in tokens)
    print(f"gemini tokens: {it} in, {ot} out+thinking over {len(tokens)} calls")


if __name__ == "__main__":
    asyncio.run(main())
