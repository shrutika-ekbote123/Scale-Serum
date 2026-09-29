"""
Run transcription configurations over the evaluation sets and score them.

    python scripts/sca_eval/run_eval.py                                  # everything
    python scripts/sca_eval/run_eval.py --sets synthetic --configs baseline phase0
    python scripts/sca_eval/run_eval.py --sets real --limit 5
    python scripts/sca_eval/run_eval.py --rescore        # no API calls, cached responses only

CONFIGURATIONS
    baseline   What production sends today: nova-3, language=multi, no keyterms.
    phase0     Language identification acting (SCA_LANGUAGE_ID=on) with the new
               English-only rule, plus keyterms (brand, product, rep, customer).
    oracle     The TRUE language (known for FLEURS and synthetic calls) plus
               keyterms. The ceiling for any one-language-per-call routing: if
               oracle is still poor on a set, better detection cannot fix it and
               per-segment routing (Phase 2) is needed.
    phase1     phase0 + speaker refinement (sales_call_analyzer/diarization.py),
               voices compared only with each other.
    phase2     phase1 + the segment pass: non-rep turns re-transcribed by Gemini
               in the detected regional language (pipeline._segment_pass).
    phase2_vp  phase1_vp + the segment pass. Synthetic calls only.
    phase1_vp  phase1 with the rep's voiceprint, enrolled from other calls
               (SCA_EVAL_ENROL_CALLS, default 1). Synthetic calls only.
    prep       phase0 on preprocessed audio (high-pass, loudness, FLAC).
               Measured and NOT adopted - see README.

IT RUNS THE PRODUCT'S OWN CODE
    Parameters come from transcription.deepgram_client.build_params, routing
    from language_for, keyterms from pipeline.transcription_keyterms, the
    transcript from transcript.from_deepgram and roles from
    speakers.resolve_roles. A change to any of those shows up here unchanged.

COST AND CACHING
    Every Deepgram response is cached under workspace/runs/raw, keyed by the
    exact request parameters, so configurations that send the same request
    share one call and --rescore re-scores everything for free. A full run over
    all three sets is roughly $3 of Deepgram plus cents of Gemini.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

from _common import (FLEURS_DIR, LABELS_DIR, REAL_DIR, REPORTS_DIR, RUNS_DIR, SYNTH_DIR,
                     decode_pcm, read_json, read_jsonl, write_json)
import metrics as M

from sales_call_analyzer import pipeline as pl  # noqa: E402
from sales_call_analyzer import speakers as sp  # noqa: E402
from sales_call_analyzer import transcript as tr  # noqa: E402
from sales_call_analyzer.models import (AnalyzeRequest, CustomerInfo, ProductInfo,  # noqa: E402
                                        RepInfo)
from transcription import audio_slice  # noqa: E402
from transcription import deepgram_client as dg  # noqa: E402
from transcription import language_id  # noqa: E402
from transcription.romanize import indic_share  # noqa: E402
from transcription import voice  # noqa: E402
from sales_call_analyzer import diarization as diar  # noqa: E402

CONFIGS = ("baseline", "phase0", "oracle", "phase1", "phase1_vp", "prep", "phase2", "phase2_vp")
SETS = ("fleurs", "synthetic", "real")
SET_DIRS = {"fleurs": FLEURS_DIR, "synthetic": SYNTH_DIR, "real": REAL_DIR}


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #
def load_items(which: list[str], limit: Optional[int]) -> list[dict]:
    items = []
    for name in which:
        rows = list(read_jsonl(SET_DIRS[name] / "manifest.jsonl"))
        if not rows:
            print(f"  {name}: no manifest - run fetch_datasets.py / make_synthetic_calls.py first")
        if name == "real":
            # 3-second stubs are not calls.
            rows = [r for r in rows if (r.get("duration_seconds") or 0) >= 15]
            # The same recording is attached to several sales_calls rows (15
            # unique of 28 on 2026-09-28, one of them 7 times). Score each once,
            # preferring the row whose CRM record carries both names.
            unique: dict[str, dict] = {}
            for r in sorted(rows, key=lambda r: not (r.get("rep_name") and r.get("customer_name"))):
                digest = hashlib.sha1((REAL_DIR / r["audio"]).read_bytes()).hexdigest()
                unique.setdefault(digest, r)
            print(f"  real: {len(unique)} unique recordings of {len(rows)} rows")
            rows = list(unique.values())
        for row in rows[:limit] if limit else rows:
            row["set"] = name
            row["path"] = str(SET_DIRS[name] / row["audio"])
            items.append(row)
    return items


def truth_for(item: dict) -> Optional[dict]:
    if item["set"] == "synthetic":
        return read_json(SYNTH_DIR / item["truth"])
    if item["set"] == "real":
        label = LABELS_DIR / f"{item['id']}.json"
        if label.exists():
            data = read_json(label)
            return data if data.get("reviewed") else None
    return None


def oracle_language(item: dict) -> Optional[str]:
    lang = item.get("language")
    if item["set"] == "fleurs":
        return lang
    if item["set"] == "synthetic":
        if lang == "en":
            return "en"
        if lang == "hi":
            return None           # multi: Hindi and English together
        return lang
    return None


def keyterms_for(item: dict) -> list[str]:
    request = AnalyzeRequest(call_id=str(item["id"]),
                             rep=RepInfo(name=item.get("rep_name")),
                             customer=CustomerInfo(name=item.get("customer_name")),
                             product=ProductInfo(name=None))
    return pl.transcription_keyterms(request, item.get("brand_name"))


# --------------------------------------------------------------------------- #
# Providers, cached
# --------------------------------------------------------------------------- #
def _safe(item_id: str) -> str:
    return str(item_id).replace("/", "__")


async def identify(item: dict, client, model: str, sem: asyncio.Semaphore,
                   offline: bool) -> Optional[dict]:
    path = RUNS_DIR / "langid" / item["set"] / f"{_safe(item['id'])}.json"
    if path.exists():
        return read_json(path)
    if offline:
        return None
    audio = Path(item["path"]).read_bytes()
    sample = await audio_slice.window(audio)
    if not sample:
        # Shorter than the window's 20 s start. The pipeline would send the
        # whole file; we send the whole file as MP3, because Gemini answers a
        # mu-law WAV with a server error (216 of 240 FLEURS clips, 2026-09-28).
        sample = await audio_slice.window(audio, start=0)
    if not sample:
        return None
    # Gemini answers bursts with 5xx (overloaded); one clip alone always worked.
    for attempt in range(6):
        async with sem:
            decision = await language_id.identify(client, model, sample, mime_type="audio/mpeg")
        if decision.ok:
            break
        await asyncio.sleep(5 * (attempt + 1))
    result = {"ok": decision.ok, "dominant_non_english": decision.dominant_non_english,
              "english_share": decision.english_share, "languages": decision.languages,
              "reason": decision.reason}
    if decision.ok:
        write_json(path, result)      # a failure is retried next run, never cached
    return result


async def deepgram(item: dict, params: dict, http: httpx.AsyncClient,
                   sem: asyncio.Semaphore, offline: bool) -> Optional[dict]:
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    path = RUNS_DIR / "raw" / item["set"] / _safe(item["id"]) / f"{key}.json"
    if path.exists():
        return read_json(path)["response"]
    if offline:
        return None
    audio = Path(item["path"]).read_bytes()
    mime = mimetypes.guess_type(item["path"])[0] or "application/octet-stream"
    headers = {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}", "Content-Type": mime}
    error = "not attempted"
    for attempt in range(6):
        resp = None
        async with sem:
            try:
                resp = await http.post(dg.DEEPGRAM_ENDPOINT, params=params, headers=headers,
                                       content=audio)
            except httpx.HTTPError as err:
                error = type(err).__name__
        if resp is not None and resp.status_code == 200:
            response = resp.json()
            write_json(path, {"params": params, "response": response})
            return response
        if resp is not None:
            error = f"HTTP {resp.status_code}: {resp.text[:160]}"
            if resp.status_code < 500 and resp.status_code not in (408, 429):
                break           # a rejected request does not fix itself
        # Connection drops, 408 SLOW_UPLOAD and 5xx are transient on long uploads.
        await asyncio.sleep(5 * (attempt + 1))
    print(f"    deepgram failed for {item['id']}: {error}")
    return None



# --------------------------------------------------------------------------- #
# Voices, cached: embeddings are pure functions of the audio slice
# --------------------------------------------------------------------------- #
_EMB_CACHE: dict[str, dict] = {}


def _emb_path(item: dict) -> Path:
    return RUNS_DIR / "emb" / voice.model_id() / item["set"] / f"{_safe(item['id'])}.npz"


def cached_embedder(item: dict):
    key = f"{item['set']}/{item['id']}"
    cache = _EMB_CACHE.get(key)
    if cache is None:
        cache = {}
        path = _emb_path(item)
        if path.exists():
            data = np.load(path)
            cache = {k: data[k] for k in data.files}
        _EMB_CACHE[key] = cache

    def embed(samples):
        digest = hashlib.sha1(np.ascontiguousarray(samples, dtype=np.float32).tobytes()).hexdigest()[:16]
        if digest not in cache:
            vector = voice.embed(samples)
            cache[digest] = vector if vector is not None else np.zeros(0, np.float32)
        vector = cache[digest]
        return vector if vector.size else None
    return embed


def save_embedding_cache() -> None:
    for key, cache in _EMB_CACHE.items():
        set_name, item_id = key.split("/", 1)
        path = _emb_path({"set": set_name, "id": item_id})
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **cache)


_VOICEPRINTS: dict[tuple, Optional[np.ndarray]] = {}
ENROL_CALLS = int(os.environ.get("SCA_EVAL_ENROL_CALLS", 1))


def enrolled_voiceprint(item: dict) -> Optional[np.ndarray]:
    """The rep's TTS voice, enrolled the way production enrols: through
    voiceprints.build(), from the rep-voice turns of ONE other synthetic call -
    never from the call being scored. (Pooling several calls made an unrealistically
    clean voiceprint and hid the "unsure" stretches a real one leaves.)"""
    if item["set"] != "synthetic":
        return None
    truth = truth_for(item)
    rep_voice = truth["voices"]["rep"]
    key = (rep_voice, item["id"])
    if key in _VOICEPRINTS:
        return _VOICEPRINTS[key]
    from sales_call_analyzer import voiceprints as vp_mod
    _VOICEPRINTS[key] = None
    # Prefer a call where the voice is the rep (more speech), then any.
    candidates = []
    for row in read_jsonl(SYNTH_DIR / "manifest.jsonl"):
        if row["id"] == item["id"]:
            continue
        other = read_json(SYNTH_DIR / row["truth"])
        roles = [r for r, v in other["voices"].items() if v == rep_voice]
        if roles:
            candidates.append((roles[0] != "rep", row, other, roles[0]))
    # ENROL_CALLS=1 is one sample; more folds further calls in with
    # voiceprints.combine(), as add_to_existing does in production.
    vector, seconds = None, 0.0
    used = 0
    for _, row, other, role in sorted(candidates, key=lambda c: (c[0], c[1]["id"])):
        if used >= ENROL_CALLS:
            break
        samples = decode_pcm(SYNTH_DIR / row["audio"])[:, 0]
        spans = [(t["start"], t["end"]) for t in other["turns"] if t["role"] == role]
        try:
            v, stats = vp_mod.build(samples, spans,
                                    cached_embedder({"set": "synthetic", "id": row["id"]}))
        except vp_mod.EnrolmentRejected:
            continue
        vector = v if vector is None else vp_mod.combine(vector, seconds, v, stats["speech_seconds"])
        seconds += stats["speech_seconds"]
        used += 1
    _VOICEPRINTS[key] = vector
    return _VOICEPRINTS[key]


# --------------------------------------------------------------------------- #
# Preprocessing experiment: does cleaning the audio help Deepgram?
# --------------------------------------------------------------------------- #
PREP_FILTER = os.environ.get("SCA_EVAL_PREP_FILTER", "highpass=f=80,dynaudnorm=f=150:g=15")


def prepared(item: dict) -> dict:
    """The item with its audio high-passed (hum), loudness-normalised per short
    window (a quiet customer leg) and sent as lossless FLAC - never upsampled:
    8 kHz phone audio stays 8 kHz. Cached under its own id, so its Deepgram
    responses never mix with the original audio's."""
    import subprocess
    from _common import WORKSPACE, ffprobe_path, run_ffmpeg
    out = WORKSPACE / "prep" / item["set"] / f"{_safe(item['id'])}.flac"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        rate = subprocess.run([ffprobe_path(), "-v", "error", "-select_streams", "a:0",
                               "-show_entries", "stream=sample_rate", "-of", "csv=p=0",
                               item["path"]], capture_output=True, text=True).stdout.strip()
        target = min(int(rate or 16000), 16000)
        run_ffmpeg(["-i", item["path"], "-af", PREP_FILTER, "-ac", "1", "-ar", str(target),
                    "-c:a", "flac", str(out)])
    return {**item, "id": f"{item['id']}@prep", "path": str(out)}


# --------------------------------------------------------------------------- #
# Phase 2: the product's segment pass, with Gemini replies cached
# --------------------------------------------------------------------------- #
def _cached_segment_transcribe(ctx):
    from transcription import segment_transcribe as seg_mod

    async def transcribe(clips, language, keyterms=None):
        key = hashlib.sha1(b"".join(np.asarray(c, np.float32).tobytes() for c in clips)
                           + language.encode() + ctx["segment_model"].encode()
                           + "|".join(keyterms or []).encode()
                           + str(seg_mod.THINKING_BUDGET).encode()
                           + getattr(seg_mod, "PROMPT", "").encode()
                           + os.environ.get("SCA_EVAL_SEED", "").encode()).hexdigest()[:20]
        path = RUNS_DIR / "segpass" / "pipeline" / f"{key}.json"
        if path.exists():
            data = read_json(path)
            return seg_mod.ClipResult(ok=data["ok"], texts=data["texts"],
                                      rejected={int(k): v for k, v in data["rejected"].items()},
                                      reason=data["reason"], requests=data["requests"],
                                      input_tokens=data["input_tokens"],
                                      output_tokens=data["output_tokens"],
                                      thinking_tokens=data["thinking_tokens"])
        if ctx["offline"]:
            return seg_mod.ClipResult(reason="offline")
        for attempt in range(4):
            async with ctx["gsem"]:
                result = await seg_mod.transcribe_clips(ctx["gemini"], ctx["segment_model"],
                                                        clips, language, keyterms=keyterms)
            if result.ok:
                break
            await asyncio.sleep(5 * (attempt + 1))
        if result.ok:
            write_json(path, {"ok": result.ok, "texts": result.texts, "rejected": result.rejected,
                              "reason": result.reason, "requests": result.requests,
                              "input_tokens": result.input_tokens,
                              "output_tokens": result.output_tokens,
                              "thinking_tokens": result.thinking_tokens})
        return result
    return transcribe


def _raw_with_segment_texts(raw: dict, transcript) -> dict:
    """Deepgram-shaped words for scoring: each re-transcribed segment's words are
    replaced by its new words, spread evenly across the segment's time."""
    import copy
    out = copy.deepcopy(raw)
    alt = out["results"]["channels"][0]["alternatives"][0]
    words = alt.get("words") or []
    changed = [s for s in transcript.segments if s.text_source]
    for seg in changed:
        speaker = int(seg.speaker_id.split("_")[-1]) if seg.speaker_id.split("_")[-1].isdigit() else 0
        words = [w for w in words if not (seg.start - 0.05 <= (w["start"] + w["end"]) / 2 <= seg.end + 0.05)]
        new = seg.text.split()
        step = (seg.end - seg.start) / max(len(new), 1)
        for k, token in enumerate(new):
            words.append({"word": token, "punctuated_word": token, "speaker": speaker,
                          "start": seg.start + k * step, "end": seg.start + (k + 1) * step - 0.01,
                          "confidence": 1.0})
    alt["words"] = sorted(words, key=lambda w: w["start"])
    return out


async def segment_pass(item, transcript, raw, samples, language, langid, rep_voice, ctx):
    from sales_call_analyzer.models import AnalyzeRequest, ProcessingInfo
    processing = ProcessingInfo(transcription_language_sent=dg.effective_language(language))
    if langid and langid.get("ok"):
        processing.language_detected = langid.get("dominant_non_english")
        processing.language_detection_shares = langid.get("languages")
    request = AnalyzeRequest(call_id=str(item["id"]), rep=RepInfo(name=item.get("rep_name")),
                             customer=CustomerInfo(name=item.get("customer_name")))
    deps = pl.PipelineDeps(store=None, llm_client=None, llm_model="", transcribe=None,
                           segment_transcribe=_cached_segment_transcribe(ctx),
                           segment_pass_mode="on", segment_pass_model=ctx["segment_model"],
                           segment_pass_scope=os.environ.get("SCA_SEGMENT_PASS_SCOPE"))
    changed = await pl._segment_pass(transcript, request, deps, processing, {"samples": samples},
                                     str(item["id"]), brand_name=item.get("brand_name"))
    if changed:
        transcript = sp.resolve_roles(transcript, rep=RepInfo(name=item.get("rep_name")),
                                      customer=CustomerInfo(name=item.get("customer_name")),
                                      brand_name=item.get("brand_name"), rep_voice=rep_voice)
        raw = _raw_with_segment_texts(raw, transcript)
    info = dict(processing.segment_pass or {})
    info["input_tokens"] = processing.segment_pass_input_tokens
    info["output_tokens"] = processing.segment_pass_output_tokens
    info["thinking_tokens"] = processing.segment_pass_thinking_tokens
    return transcript, raw, info

# --------------------------------------------------------------------------- #
# One item, one config
# --------------------------------------------------------------------------- #
async def run_one(item: dict, config: str, ctx: dict) -> Optional[dict]:
    language, why, langid = None, "server_default", None
    keyterms: list[str] = []
    if config != "baseline":
        keyterms = keyterms_for(item)
    if config in ("phase0", "phase1", "phase1_vp", "prep", "phase2", "phase2_vp"):
        langid = await identify(item, ctx["gemini"], ctx["langid_model"], ctx["gsem"], ctx["offline"])
        if langid and langid.get("ok"):
            dominant = langid.get("dominant_non_english")
            shares = {x["code"]: x.get("share_percent") for x in langid.get("languages") or []}
            language, why = dg.language_for(dominant, langid.get("english_share"),
                                            shares.get(dominant) if dominant else None)
    elif config == "oracle":
        language, why = oracle_language(item), "oracle"

    params = dg.build_params(language, keyterms)
    sent = prepared(item) if config == "prep" else item
    raw = await deepgram(sent, params, ctx["http"], ctx["dsem"], ctx["offline"])
    if raw is None:
        return None

    refinement, rep_voice = None, None
    samples = None
    if config.startswith(("phase1", "phase2")) and item["set"] != "fleurs":
        samples = decode_pcm(Path(item["path"]))[:, 0]
        voiceprint = enrolled_voiceprint(item) if config.endswith("_vp") else None
        raw, report_ = await asyncio.to_thread(
            diar.refine, raw, samples, cached_embedder(item), voiceprint, voice.model_id())
        refinement = report_.as_dict()
        if report_.rep_speaker_id:
            rep_voice = {"speaker_id": report_.rep_speaker_id,
                         "score": report_.voice_scores.get(report_.rep_speaker_id)}

    transcript = tr.from_deepgram(raw, language_hint=dg.effective_language(language))
    transcript = sp.resolve_roles(transcript, rep=RepInfo(name=item.get("rep_name")),
                                  customer=CustomerInfo(name=item.get("customer_name")),
                                  brand_name=item.get("brand_name"), rep_voice=rep_voice)
    segment_info = None
    if config.startswith("phase2") and samples is not None:
        transcript, raw, segment_info = await segment_pass(
            item, transcript, raw, samples, language, langid, rep_voice, ctx)
    roles = {s.speaker_id: s.role for s in transcript.speakers}
    hyp_text = " ".join(s.text for s in transcript.segments)
    result = {
        "id": item["id"], "set": item["set"], "config": config,
        "language_sent": params.get("language"), "language_why": why,
        "language_id": langid, "keyterms": len(keyterms),
        "speakers_found": transcript.speaker_count,
        "speech_coverage": transcript.quality.speech_coverage,
        "words_per_minute": transcript.quality.words_per_minute,
        "warnings": transcript.quality.warnings,
        "indic_share": round(indic_share(hyp_text), 3),
        "roles": {s.speaker_id: [s.role, s.role_basis, s.role_confidence] for s in transcript.speakers},
        "rep_resolved": any(s.role == "sales_rep" for s in transcript.speakers),
        "refinement": refinement,
        "segment_pass": segment_info,
        "customer_resolved": any(s.role == "customer" for s in transcript.speakers),
    }

    if item["set"] == "fleurs":
        result["text"] = M.text_scores(item.get("reference") or "", hyp_text)
        result["language"] = item["language"]
        return result

    truth = truth_for(item)
    if item["set"] == "real":
        speech = ctx["speech_seconds"].get(item["id"])
        covered = (transcript.quality.speech_coverage or 0) * (transcript.duration_seconds or 0)
        result["energy_speech_seconds"] = speech
        result["covered_vs_energy_speech"] = round(min(covered / speech, 1.5), 3) if speech else None
        result["duration_seconds"] = transcript.duration_seconds
    if truth:
        turns = truth["turns"]
        placed = M.place_words(M.deepgram_words(raw), turns)
        result["text"] = M.text_scores(" ".join(t["text"] for t in turns), hyp_text)
        result["text_by_role"] = M.per_role_text(placed, turns)
        result["speakers"] = M.speaker_scores(placed, turns, roles)
        result["language"] = truth.get("customer_language") or item.get("language")
        result["pattern"] = truth.get("pattern")
        result["same_gender_voices"] = truth.get("same_gender_voices")
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(v, pct=True):
    if v is None:
        return "   -  "
    return f"{v * 100:5.1f}%" if pct else f"{v:6.2f}"


def report(results: list[dict]) -> dict:
    summary: dict = {}
    by = defaultdict(list)
    for r in results:
        by[(r["set"], r["config"])].append(r)

    # FLEURS: per language
    fl = [r for r in results if r["set"] == "fleurs"]
    if fl:
        print("\nFLEURS (read speech, 8 kHz mu-law)   WER/CER native | WER/CER roman | words kept")
        langs = sorted({r["language"] for r in fl})
        for config in CONFIGS:
            rows = [r for r in fl if r["config"] == config]
            if not rows:
                continue
            for lang in langs:
                sub = [r for r in rows if r["language"] == lang]
                nat, rom = {}, {}
                for r in sub:
                    M.add_counts(nat, r["text"]["native"])
                    M.add_counts(rom, r["text"]["roman"])
                rn, rr = M.rates(nat), M.rates(rom)
                sent = Counter(r["language_sent"] for r in sub).most_common(1)[0][0]
                summary.setdefault("fleurs", {}).setdefault(config, {})[lang] = {
                    "native": rn, "roman": rr, "n": len(sub), "language_sent": sent}
                print(f"  {config:8} {lang}  sent={sent:5}  {_fmt(rn['wer'])} {_fmt(rn['cer'])} | "
                      f"{_fmt(rr['wer'])} {_fmt(rr['cer'])} | {_fmt(rr['length_ratio'], False)}")

    # Synthetic and labelled real calls: speakers and per-role text
    for set_name in ("synthetic", "real"):
        if any("speakers" in r for r in results if r["set"] == set_name):
            print(f"\n{set_name.upper()} WITH TRUTH   n | role acc | diar acc | "
                  f"backchannel | rep WER | cust WER (roman) | outcomes")
        for config in CONFIGS:
            rows = [r for r in by.get((set_name, config), []) if "speakers" in r]
            if not rows:
                continue
            text, rep, cust = {}, {}, {}
            for r in rows:
                M.add_counts(text, r["text"]["roman"])
                M.add_counts(rep, r["text_by_role"].get("rep", {}).get("roman", {"word_edits": 0, "ref_words": 0, "hyp_words": 0, "char_edits": 0, "ref_chars": 0}))
                M.add_counts(cust, r["text_by_role"].get("customer", {}).get("roman", {"word_edits": 0, "ref_words": 0, "hyp_words": 0, "char_edits": 0, "ref_chars": 0}))
            words = sum(r["speakers"]["words_scored"] for r in rows)
            role_acc = sum(r["speakers"]["role_accuracy"] * r["speakers"]["words_scored"] for r in rows) / max(words, 1)
            diar_acc = sum(r["speakers"]["diarization_accuracy"] * r["speakers"]["words_scored"] for r in rows) / max(words, 1)
            bc = M.summarise(r["speakers"].get("backchannel_recall") for r in rows)
            outcomes = Counter(r["speakers"]["role_outcome"] for r in rows)
            entry = {"n": len(rows), "role_accuracy": round(role_acc, 4),
                     "diarization_accuracy": round(diar_acc, 4), "backchannel_recall": bc,
                     "wer_roman": M.rates(text), "rep": M.rates(rep), "customer": M.rates(cust),
                     "outcomes": dict(outcomes)}
            summary.setdefault(f"{set_name}_truth", {})[config] = entry
            print(f"  {config:8} {len(rows):3} {_fmt(role_acc)}  {_fmt(diar_acc)}   {_fmt(bc.get('mean'))}   "
                  f"{_fmt(M.rates(rep)['wer'])}  {_fmt(M.rates(cust)['wer'])}   {dict(outcomes)}")

    syn = [r for r in results if r["set"] == "synthetic" and "speakers" in r]
    if syn:
        print("\nSYNTHETIC BY CUSTOMER LANGUAGE   role acc | customer WER roman | customer words kept")
        for config in CONFIGS:
            for lang in sorted({r["language"] for r in syn}):
                sub = [r for r in syn if r["config"] == config and r["language"] == lang]
                if not sub:
                    continue
                cust = {}
                for r in sub:
                    M.add_counts(cust, r["text_by_role"].get("customer", {}).get("roman", {}))
                words = sum(r["speakers"]["words_scored"] for r in sub)
                acc = sum(r["speakers"]["role_accuracy"] * r["speakers"]["words_scored"] for r in sub) / max(words, 1)
                cr = M.rates(cust) if cust else {}
                summary.setdefault("synthetic_by_language", {}).setdefault(config, {})[lang] = {
                    "role_accuracy": round(acc, 4), "customer": cr, "n": len(sub)}
                print(f"  {config:8} {lang}  {_fmt(acc)}  {_fmt(cr.get('wer'))}  {_fmt(cr.get('length_ratio'), False)}")

    real = [r for r in results if r["set"] == "real"]
    if real:
        print("\nREAL CALLS (label-free)   coverage med | transcribed/energy-speech med | wpm med | "
              "rep found | customer found | low-coverage warnings | language sent")
        for config in CONFIGS:
            rows = [r for r in real if r["config"] == config]
            if not rows:
                continue
            cov = M.summarise(r["speech_coverage"] for r in rows)
            ratio = M.summarise(r.get("covered_vs_energy_speech") for r in rows)
            wpm = M.summarise(r["words_per_minute"] for r in rows)
            entry = {"n": len(rows), "speech_coverage": cov, "covered_vs_energy_speech": ratio,
                     "words_per_minute": wpm,
                     "rep_resolved": round(sum(r["rep_resolved"] for r in rows) / len(rows), 3),
                     "customer_resolved": round(sum(r["customer_resolved"] for r in rows) / len(rows), 3),
                     "low_coverage_warnings": sum(tr.WARN_LOW_COVERAGE in r["warnings"] for r in rows),
                     "language_sent": dict(Counter(r["language_sent"] for r in rows)),
                     "role_confidence": dict(Counter(c for r in rows for (role, _b, c) in r["roles"].values()
                                                     if role in ("sales_rep", "customer")))}
            summary.setdefault("real", {})[config] = entry
            print(f"  {config:8} {_fmt(cov.get('median'))}  {_fmt(ratio.get('median'))}  "
                  f"{_fmt(wpm.get('median'), False)}  {_fmt(entry['rep_resolved'])}  "
                  f"{_fmt(entry['customer_resolved'])}  {entry['low_coverage_warnings']:3}  "
                  f"{entry['language_sent']}")
    return summary


# --------------------------------------------------------------------------- #
async def main_async(args) -> None:
    items = load_items(args.sets, args.limit)
    print(f"{len(items)} items, configs {args.configs}{' (offline rescore)' if args.rescore else ''}")

    gemini = None
    if not args.rescore:
        from google import genai
        gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    speech_seconds = {}
    for item in items:
        if item["set"] == "real":
            try:
                speech_seconds[item["id"]] = round(M.energy_speech_seconds(decode_pcm(Path(item["path"]))), 1)
            except Exception:  # noqa: BLE001
                speech_seconds[item["id"]] = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30)) as http:
        ctx = {"http": http, "gemini": gemini,
               "langid_model": os.environ.get("SCA_LANGUAGE_ID_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
               "dsem": asyncio.Semaphore(args.concurrency), "gsem": asyncio.Semaphore(2),
               "offline": args.rescore, "speech_seconds": speech_seconds,
               "segment_model": os.environ.get("SCA_SEGMENT_PASS_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")}
        started = time.monotonic()
        tasks = [run_one(item, config, ctx) for item in items for config in args.configs
                 if not (config in ("oracle", "phase1_vp", "phase2_vp") and item["set"] == "real")
                 and not (config.startswith(("phase1", "phase2")) and item["set"] == "fleurs")]
        results = [r for r in await asyncio.gather(*tasks) if r]
    save_embedding_cache()
    print(f"{len(results)} results in {time.monotonic() - started:.0f}s")

    summary = report(results)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORTS_DIR / f"eval-{stamp}.json"
    write_json(out, {"generated_at": stamp, "sets": args.sets, "configs": args.configs,
                     "english_only_min_share": dg.ENGLISH_ONLY_MIN_SHARE,
                     "min_speech_coverage": tr.MIN_SPEECH_COVERAGE,
                     "summary": summary, "results": results})
    print(f"\nreport: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rescore", action="store_true",
                        help="score cached responses only; make no API calls")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
