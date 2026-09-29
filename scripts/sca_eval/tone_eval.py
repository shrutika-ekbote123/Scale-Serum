"""
Does the tone step hear the voice? Scored where the words cannot help.

    python scripts/sca_eval/tone_eval.py                  # both sets, all systems
    python scripts/sca_eval/tone_eval.py --sets cremad

SETS (same words, different delivery - text alone is at chance)
    toneset   make_tone_set.py: 12 sales phrases x 7 tones x 3 voices, Gemini TTS
    cremad    CREMA-D: real actors, 12 neutral sentences, 5 emotions, mapped to
              the product's tones: anger->frustrated, happy->interested,
              fear->hesitant, sad->disengaged, neutral->neutral

SYSTEMS
    text      Gemini given ONLY the words. What the analysis did before Phase 3.
    prosody   nearest-centroid on prosody.py's measures, normalised within each
              voice, cross-validated leaving one phrase (toneset) or actor
              (cremad) out. Model-free evidence that the tone is audible.
    audio     tone.classify_clips - Gemini hears the clip (the product path).

Also printed: each tone's mean measures relative to its voice. If acted
frustration is not louder, or hesitation not slower, than neutral, the
stimuli - not the detector - are the problem.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np

from _common import RUNS_DIR, WORKSPACE, decode_pcm, read_json, read_jsonl, write_json

from sales_call_analyzer import prosody as pr
from sales_call_analyzer import tone as tn

SETS = {"toneset": WORKSPACE / "toneset", "cremad": WORKSPACE / "cremad"}
CREMAD_MAP = {"anger": "frustrated", "happy": "interested", "fear": "hesitant",
              "sad": "disengaged", "neutral": "neutral"}
VALENCE = {"interested": "positive", "confident": "positive", "urgent": "positive",
           "neutral": "neutral",
           "frustrated": "negative", "hesitant": "negative", "confused": "negative",
           "disengaged": "negative"}
BATCH = 12
ZS: dict = {}
USE_MEASURED = os.environ.get("SCA_EVAL_TONE_MEASURED", "1") == "1"
MEASURES = ("rate_wps", "pause_ratio", "loudness_db", "pitch_hz", "pitch_range_st")
TEXT_PROMPT = """Below are {n} numbered things a customer said on a phone sales call.
You have only the words, not the audio. For each, give the single most likely tone of
delivery, one of {tones}. Return JSON: {{"clips": [{{"n": 1, "tone": "..."}}, ...]}}."""


def load(which):
    items = []
    for name in which:
        for r in read_jsonl(SETS[name] / "manifest.jsonl"):
            r["set"] = name
            r["path"] = str(SETS[name] / r["audio"])
            r["truth"] = r["tone"] if name == "toneset" else CREMAD_MAP[r["emotion"]]
            r["group"] = r["voice"] if name == "toneset" else r["actor"]
            r["fold"] = r["phrase_id"] if name == "toneset" else r["actor"]
            items.append(r)
    return items


def _cache(kind, key):
    return RUNS_DIR / "tone" / kind / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


async def run_audio(items, client, model, sem):
    out = {}
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        key = "|".join(x["id"] for x in batch) + model + tn.PROMPT_VERSION + str(USE_MEASURED)
        path = _cache("audio", key)
        if path.exists():
            labels = read_json(path)
        else:
            clips = [tn.ToneClip(decode_pcm(x["path"])[:, 0], "customer", x["text"],
                                 measured=(tn.describe(dict(zip(MEASURES, ZS[x["id"]])))
                                           if USE_MEASURED and x["id"] in ZS else ""))
                     for x in batch]
            for attempt in range(4):
                async with sem:
                    res = await tn.classify_clips(client, model, clips)
                if res.ok:
                    break
                await asyncio.sleep(5 * (attempt + 1))
            labels = [(lab or {}).get("tone") for lab in res.labels]
            if res.ok:
                write_json(path, labels)
        for x, lab in zip(batch, labels):
            out[x["id"]] = lab
    return out


async def run_text(items, client, model, sem):
    from google.genai import types
    out = {}
    schema = {"type": "object", "properties": {"clips": {"type": "array", "items": {
        "type": "object", "properties": {"n": {"type": "integer"},
                                         "tone": {"type": "string", "enum": list(tn.TONES)}},
        "required": ["n", "tone"]}}}, "required": ["clips"]}
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        key = "|".join(x["id"] for x in batch) + model
        path = _cache("text", key)
        if path.exists():
            labels = read_json(path)
        else:
            lines = "\n".join(f'{n}. "{x["text"]}"' for n, x in enumerate(batch, 1))
            prompt = TEXT_PROMPT.format(n=len(batch), tones=", ".join(tn.TONES)) + "\n\n" + lines
            async with sem:
                r = await client.aio.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                                       response_schema=schema))
            got = {c["n"]: c["tone"] for c in json.loads(r.text)["clips"]}
            labels = [got.get(n) for n in range(1, len(batch) + 1)]
            write_json(path, labels)
        for x, lab in zip(batch, labels):
            out[x["id"]] = lab
    return out


def prosody_features(items):
    feats = {}
    for x in items:
        s = decode_pcm(x["path"])[:, 0]
        m = pr.measure_segment(s, 0.0, len(s) / 16000, x["text"])
        feats[x["id"]] = [m.get(k) for k in MEASURES]
    # z-normalise within each voice / actor: tone is relative to the speaker
    by_group = defaultdict(list)
    for x in items:
        by_group[x["group"]].append(x["id"])
    z = {}
    for ids in by_group.values():
        mat = np.array([[np.nan if v is None else v for v in feats[i]] for i in ids], float)
        mu, sd = np.nanmean(mat, 0), np.nanstd(mat, 0)
        sd[sd == 0] = 1
        for i, row in zip(ids, (mat - mu) / sd):
            z[i] = np.nan_to_num(row)
    return z


def run_prosody(items, z):
    """Leave-one-fold-out nearest centroid."""
    out = {}
    for fold in {x["fold"] for x in items}:
        train = [x for x in items if x["fold"] != fold]
        cents = {}
        for tone in {x["truth"] for x in train}:
            rows = [z[x["id"]] for x in train if x["truth"] == tone]
            cents[tone] = np.mean(rows, 0)
        for x in items:
            if x["fold"] == fold:
                out[x["id"]] = min(cents, key=lambda t: float(np.sum((z[x["id"]] - cents[t]) ** 2)))
    return out


def score(items, preds, name):
    got = [(x["truth"], preds.get(x["id"])) for x in items if preds.get(x["id"])]
    if not got:
        return
    acc = np.mean([t == p for t, p in got])
    tones = sorted({t for t, _ in got})
    recall = {t: np.mean([p == t for tt, p in got if tt == t]) for t in tones}
    val = np.mean([VALENCE[t] == VALENCE.get(p) for t, p in got])
    chance = 1 / len(tones)
    print(f"  {name:8} accuracy {acc:.3f} (chance {chance:.3f})  macro recall {np.mean(list(recall.values())):.3f}"
          f"  valence {val:.3f}   n={len(got)}")
    print("           " + "  ".join(f"{t[:6]} {recall[t]:.2f}" for t in tones))
    return {"accuracy": float(acc), "recall": {k: float(v) for k, v in recall.items()},
            "valence": float(val), "n": len(got),
            "confusion": dict(Counter(f"{t}->{p}" for t, p in got if t != p).most_common(6))}


async def main_async(args):
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("SCA_TONE_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    sem = asyncio.Semaphore(2)
    summary = {}
    for name in args.sets:
        items = load([name])
        if not items:
            print(f"{name}: no manifest")
            continue
        print(f"\n{name.upper()}: {len(items)} clips, {len({x['truth'] for x in items})} tones")
        z = prosody_features(items)
        ZS.update(z)
        print("  mean measures per tone, z within voice:  " + "  ".join(m[:10] for m in MEASURES))
        for tone in sorted({x["truth"] for x in items}):
            rows = np.array([z[x["id"]] for x in items if x["truth"] == tone])
            print(f"    {tone:11} " + "  ".join(f"{v:+.2f}".rjust(10) for v in rows.mean(0)))
        res = {"prosody": run_prosody(items, z)}
        if "text" in args.systems:
            res["text"] = await run_text(items, client, model, sem)
        if "audio" in args.systems:
            # Shuffled first. In manifest order a batch held ONE phrase in several
            # tones in a fixed sequence - side-by-side contrast a real call never
            # offers - and scored 100%. A call's moments are different words.
            import random
            shuffled = list(items)
            random.Random(7).shuffle(shuffled)
            res["audio"] = await run_audio(shuffled, client, model, sem)
        summary[name] = {k: score(items, v, k) for k, v in res.items()}
        if name == "cremad":
            actors = sorted({x["actor"] for x in items})
            for half, group in (("first 10 actors", actors[:10]), ("last 10 actors", actors[10:])):
                sub = [x for x in items if x["actor"] in group]
                print(f"  -- {half}")
                for k, v in res.items():
                    score(sub, v, k)
    write_json(WORKSPACE / "reports" / "tone_eval.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--systems", nargs="+", default=["text", "audio"], choices=["text", "audio"])
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
