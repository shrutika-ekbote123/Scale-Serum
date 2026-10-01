"""
Cleaning a Sarvam AI transcript before anything reads it - deterministic, free.

STATUS: STANDALONE. Nothing in sales_call_analyzer imports this yet. It is
tested on its own (tests/test_sarvam_cleanup.py, scripts/sca_eval/
sarvam_cleanup_test.py) and wired into the analyzer only once that is approved.

WHY
    Sarvam (saaras:v3, codemix, diarized) separated speakers far better than
    Deepgram and wrote regional languages far better (scripts/sca_eval/README).
    On the real test calls it also showed four weaknesses (2026-09-30):

      1. Duplicated fragments - phone echo puts a few of the rep's words under
         the customer too (7, 23 and 5 on mycall7/8/9). Talk time and "what the
         customer said" are then wrong.
      2. Stray words in the wrong script - "ହଁ" (Odia), "હા" (Gujarati) on a
         Marathi/English call.
      3. The brand misheard - "Lot Earning", "lot army", "lot आणि AI" for
         Lawtorney; "Skyl Serum" for ScaleSerum. The analysis looks for the
         brand, and Sarvam's own keyterm option (saaras:v4) overwrote real words.
      4. Product terms misheard ("JTPT" for ChatGPT). Text alone cannot fix
         these safely; this module only FLAGS them, and sarvam_recheck.py lets
         Gemini listen to just those clips.

    Steps 1-3 run here, in that order (duplicates first so the term fixes and
    the flags see clean text). Every change is recorded in the report, so a
    reviewer can see exactly what was altered.

WHAT IS NEVER CHANGED
    Person names. The CRM names attached to calls are often placeholders (see
    the SCA eval findings), so rewriting "Divansh" to a CRM name could replace a
    right name with a wrong one. Role resolution already matches names fuzzily.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from rapidfuzz import fuzz

from .romanize import romanize

# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #


@dataclass
class Entry:
    """One diarized turn. `index` is its position in Sarvam's sorted output."""
    index: int
    speaker_id: str
    start: float
    end: float
    text: str


def entries_from_sarvam(response: dict) -> list[Entry]:
    """The diarized entries of a Sarvam batch response, in time order."""
    raw = ((response or {}).get("diarized_transcript") or {}).get("entries") or []
    raw = sorted(raw, key=lambda e: (float(e.get("start_time_seconds") or 0),
                                     float(e.get("end_time_seconds") or 0)))
    return [Entry(i, str(e.get("speaker_id")), float(e.get("start_time_seconds") or 0),
                  float(e.get("end_time_seconds") or 0), (e.get("transcript") or "").strip())
            for i, e in enumerate(raw)]


def _words(text: str) -> list[str]:
    return romanize(text).split()


# --------------------------------------------------------------------------- #
# 1. Duplicated fragments
# --------------------------------------------------------------------------- #
# Measured on the real calls: an echo is a short fragment (1-6 words) whose
# words are ALSO in a longer turn of the other speaker, at the same moment.
# "At the same moment" matters: a customer's "हो" during a long rep turn is
# usually a real backchannel, and the rep's turn often contains "हो" somewhere
# else. Sarvam gives no word timings, so a word's moment is estimated from its
# character position in the turn (speech time follows text length closely
# enough for a 1-2 s tolerance).
DUP_MAX_WORDS = 6
DUP_MIN_SCORE = 90.0           # rapidfuzz ratio of the romanised word windows
DUP_EXTRA_WORDS = 2            # the other turn must be at least this much longer
DUP_TOLERANCE_SECONDS = 2.0
DUP_TOLERANCE_ONE_WORD_SECONDS = 1.0
DUP_SPAN_SLACK_SECONDS = 0.5


def _moment_of(other: Entry, words: list[str], first: int, count: int) -> float:
    """Estimated time at the middle of words[first:first+count] of `other`."""
    lengths = [len(w) + 1 for w in words]
    total = sum(lengths) or 1
    middle = sum(lengths[:first]) + sum(lengths[first:first + count]) / 2
    return other.start + (other.end - other.start) * middle / total


def find_duplicate(entry: Entry, entries: list[Entry]) -> Optional[dict]:
    """The other speaker's turn this entry echoes, or None."""
    mine = _words(entry.text)
    if not mine or len(mine) > DUP_MAX_WORDS:
        return None
    needle = " ".join(mine)
    middle = (entry.start + entry.end) / 2
    one_word = len(mine) == 1
    tolerance = DUP_TOLERANCE_ONE_WORD_SECONDS if one_word else DUP_TOLERANCE_SECONDS
    # A lone "Hello" just before the other's "Hello, ..." is a greeting, not an
    # echo: one word must fall inside the other turn, not next to it.
    slack = 0.0 if one_word else DUP_SPAN_SLACK_SECONDS
    best = None
    for other in entries:
        if other is entry or other.speaker_id == entry.speaker_id:
            continue
        if not (other.start - slack <= middle <= other.end + slack):
            continue
        theirs = _words(other.text)
        if len(theirs) < len(mine) + DUP_EXTRA_WORDS:
            continue
        for first in range(len(theirs) - len(mine) + 1):
            score = fuzz.ratio(needle, " ".join(theirs[first:first + len(mine)]))
            if score < DUP_MIN_SCORE:
                continue
            gap = abs(_moment_of(other, theirs, first, len(mine)) - middle)
            if gap <= tolerance and (best is None or gap < best["gap_seconds"]):
                best = {"echo_of": other.index, "score": round(score, 1),
                        "gap_seconds": round(gap, 2)}
    return best


def remove_duplicates(entries: list[Entry]) -> tuple[list[Entry], list[dict]]:
    """Entries without echoed fragments, and what was removed."""
    removed, kept = [], []
    for entry in entries:
        hit = find_duplicate(entry, entries)
        if hit:
            removed.append({"index": entry.index, "speaker_id": entry.speaker_id,
                            "start": entry.start, "end": entry.end, "text": entry.text, **hit})
        else:
            kept.append(entry)
    return kept, removed


# --------------------------------------------------------------------------- #
# 2. Stray words in the wrong script
# --------------------------------------------------------------------------- #
_BLOCKS = [
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"), (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"), (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"), (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
]
SCRIPT_FOR_LANGUAGE = {
    "hi": "devanagari", "mr": "devanagari", "ne": "devanagari", "bn": "bengali",
    "as": "bengali", "pa": "gurmukhi", "gu": "gujarati", "od": "oriya", "or": "oriya",
    "ta": "tamil", "te": "telugu", "kn": "kannada", "ml": "malayalam",
}
# Hindi words turn up on calls in every Indian language, so Devanagari is
# always allowed next to Latin and the call's own script.
ALWAYS_ALLOWED = "devanagari"
DOMINANT_MIN_SHARE = 0.8
DOMINANT_MIN_LETTERS = 20
_DANDA = "।"


def script_of(ch: str) -> Optional[str]:
    code = ord(ch)
    for low, high, name in _BLOCKS:
        if low <= code <= high:
            return name
    return None


def main_script(entries: list[Entry], language_code: Optional[str]) -> str:
    """The call's own Indic script: from Sarvam's language code, else the
    script most of the Indic letters are in, else Devanagari."""
    code = (language_code or "").split("-")[0].lower()
    if code in SCRIPT_FOR_LANGUAGE:
        return SCRIPT_FOR_LANGUAGE[code]
    counts: dict[str, int] = {}
    for entry in entries:
        for ch in entry.text:
            name = script_of(ch)
            if name:
                counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    if total >= DOMINANT_MIN_LETTERS:
        name, count = max(counts.items(), key=lambda kv: kv[1])
        if count / total >= DOMINANT_MIN_SHARE:
            return name
    return ALWAYS_ALLOWED


def _transliterate(run: str, source: str, target: str) -> str:
    try:
        from indic_transliteration import sanscript
        return sanscript.transliterate(run, source, target)
    except Exception:  # noqa: BLE001 - library missing or a codepoint it rejects
        return run


def fix_scripts(entries: list[Entry], language_code: Optional[str]) -> list[dict]:
    """Rewrite letters in a foreign Indic script into the call's main script,
    in place. Also turns a Devanagari danda ending an all-Latin turn into "."."""
    target = main_script(entries, language_code)
    allowed = {target, ALWAYS_ALLOWED}
    changes = []
    for entry in entries:
        before = entry.text
        out, run, run_script = [], [], None

        def flush():
            if run:
                text = "".join(run)
                out.append(_transliterate(text, run_script, target)
                           if run_script not in allowed else text)
                run.clear()

        for ch in before:
            name = script_of(ch)
            # Signs shared by all scripts (danda) stay with the run they end.
            if name and ch != _DANDA:
                if run and name != run_script:
                    flush()
                run_script = name
                run.append(ch)
            else:
                flush()
                out.append(ch)
        flush()
        after = "".join(out)
        if _DANDA in after and not any(script_of(c) for c in after if c != _DANDA):
            after = after.replace(_DANDA, ".")
        if after != before:
            entry.text = after
            changes.append({"index": entry.index, "before": before, "after": after,
                            "target_script": target})
    return changes


# --------------------------------------------------------------------------- #
# 3. Brand and product names
# --------------------------------------------------------------------------- #
# Scores are rapidfuzz ratios (0-1) on the romanised, space-free window:
#   char   - the letters ("lawterney" vs "lawtorney" = 0.89)
#   sound  - a consonant skeleton ("lot earning" -> "ltrng" vs "ltrn" = 0.89)
# Calibrated on the real calls (2026-10-01):
#   replaced outright   Lawterney.ai, Law Attorney, Skyl Serum      sound >= .95, char >= .70
#   replaced in context Lot Earning, lot army, lotany.ai, lot आणि AI  sound >= .85 + "from"/".ai"/"मधून"...
#   flagged only        law training, lottery                        sound >= .80
#   left alone          sales team (.73), literally (.75)
# Context is what separates "from Lot Earning" from an ordinary word.
REPLACE_SOUND = 0.95
REPLACE_CHAR = 0.70
CONTEXT_SOUND = 0.85
CONTEXT_CHAR = 0.35
SUSPECT_SOUND = 0.80
SUSPECT_CHAR = 0.40
LENGTH_RATIO = (0.6, 1.6)
FUZZY_MIN_CHARS = 5           # shorter terms ("WDC") are only matched exactly
EXTRA_WINDOW_WORDS = 2

# Words around a brand in an introduction: "Shruti from X", "X मधून", "X से".
BEFORE_MARKERS = {"from", "frm"}
AFTER_MARKERS = {"se", "madhun", "madhuna", "kadun", "varun", "tarfe", "vale", "vali", "vala",
                 "wale", "wali", "wala"}
AI_WORDS = {"ai", "a i"}

_TOKEN = re.compile(r"^(?P<lead>[\"'“‘(\[]*)(?P<core>.*?)"
                    r"(?P<tail>(?:\.ai|-\S*)?[.,!?;:।\"”’)\]]*)$", re.IGNORECASE)
_VOWELS = re.compile(r"[aeiouyhv]")


def sound_key(key: str) -> str:
    """First letter plus consonants, with spelling variants folded together."""
    k = re.sub(r"[^a-z]", "", key.lower())
    for a, b in (("ph", "f"), ("sh", "s"), ("ck", "k"), ("c", "k"), ("q", "k"),
                 ("z", "j"), ("w", "v"), ("x", "ks")):
        k = k.replace(a, b)
    k = k[:1] + _VOWELS.sub("", k[1:])
    return re.sub(r"(.)\1+", r"\1", k)


def _key(text: str) -> str:
    return romanize(text).replace(" ", "")


def _plain(text: str) -> str:
    return re.sub(r"[^\w]", "", text.lower())


@dataclass
class _Token:
    raw: str
    lead: str
    core: str
    tail: str


def _tokens(text: str) -> list[_Token]:
    out = []
    for raw in text.split():
        m = _TOKEN.match(raw)
        out.append(_Token(raw, m["lead"], m["core"], m["tail"]) if m and m["core"]
                   else _Token(raw, "", raw, ""))
    return out


@dataclass
class Match:
    term: str
    first: int
    last: int                  # inclusive token index
    char: float
    sound: float
    context: bool
    tier: str                  # exact | replace | context | suspect

    @property
    def rank(self) -> tuple:
        order = {"exact": 3, "replace": 2, "context": 1, "suspect": 0}
        return (order[self.tier], self.char + self.sound, -(self.last - self.first))


def _context(tokens: list[_Token], first: int, last: int) -> bool:
    tail = tokens[last].tail.lower()
    if tail.startswith(".ai"):
        return True
    if tail.startswith("-") and _key(tail[1:]) in AFTER_MARKERS:     # "X-मधून"
        return True
    after = [_key(t.core) for t in tokens[last + 1:last + 3]]
    if after and (after[0] in AI_WORDS or after[0] in AFTER_MARKERS):
        return True
    if after[:2] == ["dot", "ai"]:
        return True
    before = [_key(t.core) for t in tokens[max(0, first - 2):first]]
    return any(b in BEFORE_MARKERS for b in before)


def find_terms(text: str, terms: list[str]) -> list[Match]:
    """Every window of `text` that is, or may be, one of `terms`."""
    tokens = _tokens(text)
    found: list[Match] = []
    for term in terms:
        tkey = _key(term)
        if not tkey:
            continue
        tsound = sound_key(tkey)
        width = len(term.split()) + EXTRA_WINDOW_WORDS
        # "Lawtorney AI" must not be written where only "Lot Earning" was said:
        # a short last word of a term ("AI", "Pro") has to be actually heard.
        short_tail = _key(term.split()[-1]) if len(term.split()) > 1 else ""
        short_tail = short_tail if 0 < len(short_tail) <= 3 else ""
        for first in range(len(tokens)):
            for last in range(first, min(first + width, len(tokens))):
                key = _key(" ".join(t.core for t in tokens[first:last + 1]))
                if not key:
                    continue
                if short_tail and _key(tokens[last].core) != short_tail:
                    continue
                if key == tkey:
                    found.append(Match(term, first, last, 1.0, 1.0, True, "exact"))
                    continue
                if len(tkey) < FUZZY_MIN_CHARS or key[0] != tkey[0]:
                    continue
                if not LENGTH_RATIO[0] <= len(key) / len(tkey) <= LENGTH_RATIO[1]:
                    continue
                char = fuzz.ratio(key, tkey) / 100
                sound = fuzz.ratio(sound_key(key), tsound) / 100
                in_context = _context(tokens, first, last)
                if sound >= REPLACE_SOUND and char >= REPLACE_CHAR:
                    tier = "replace"
                elif sound >= CONTEXT_SOUND and char >= CONTEXT_CHAR and in_context:
                    tier = "context"
                elif sound >= SUSPECT_SOUND and char >= SUSPECT_CHAR:
                    tier = "suspect"
                else:
                    continue
                found.append(Match(term, first, last, round(char, 2), round(sound, 2),
                                   in_context, tier))
    # Best first; overlapping weaker windows are dropped.
    chosen: list[Match] = []
    for match in sorted(found, key=lambda m: m.rank, reverse=True):
        if all(match.last < c.first or match.first > c.last for c in chosen):
            chosen.append(match)
    return sorted(chosen, key=lambda m: m.first)


def apply_terms(text: str, terms: list[str]) -> tuple[str, list[dict], list[dict]]:
    """Text with confident brand/product matches written as the term itself,
    the changes made, and the windows only flagged as suspect."""
    tokens = _tokens(text)
    matches = find_terms(text, terms)
    changes, suspects = [], []
    out: list[str] = []
    i = 0
    by_first = {m.first: m for m in matches}
    while i < len(tokens):
        match = by_first.get(i)
        if match is None:
            out.append(tokens[i].raw)
            i += 1
            continue
        span = tokens[match.first:match.last + 1]
        heard = " ".join(t.raw for t in span)
        info = {"heard": heard, "term": match.term, "tier": match.tier,
                "char": match.char, "sound": match.sound}
        if match.tier == "suspect":
            suspects.append(info)
            out.append(heard)
        else:
            # Already the term apart from case, spaces or punctuation ("Scale
            # Serum"): left as written.
            written = span[0].lead + match.term + span[-1].tail
            if _plain(heard) == _plain(written):
                out.append(heard)
            else:
                changes.append({**info, "written": written})
                out.append(written)
        i = match.last + 1
    return " ".join(out), changes, suspects


def fix_terms(entries: list[Entry], terms: list[str]) -> tuple[list[dict], list[dict]]:
    """Apply apply_terms to every entry, in place."""
    changes, suspects = [], []
    for entry in entries:
        after, changed, flagged = apply_terms(entry.text, terms)
        for c in changed:
            changes.append({"index": entry.index, "start": entry.start, **c})
        for s in flagged:
            suspects.append({"index": entry.index, "start": entry.start, **s})
        entry.text = after
    return changes, suspects


# --------------------------------------------------------------------------- #
# 4. Flags for product terms (fixed by sarvam_recheck.py, not here)
# --------------------------------------------------------------------------- #
# "JTPT" for ChatGPT, "free fund" for refund: a near-miss of a vocabulary term
# is only a reason to let Gemini listen again. The first-letter rule is not
# applied - mishearings change the first sound ("free fund" / "refund").
# A flag only costs one clip of Gemini listening, so these rules aim to drop
# the obvious noise, not to be exact. Each was added for a case on the real calls:
#   * a window that is part of the term, or contains it, is the term being
#     said ("prompt" for "prompt book", "refund" in "refunds")
#   * a word at the edge of the window must make the match better, or it is
#     just a neighbour ("fund के")
#   * when a window holds one of the term's own words exactly, its other words
#     must still resemble the term's other words ("Child version" for "trial
#     version", but not "काही version" or "prompt को")
#   * an all-capitals word of 2-5 letters is an acronym as heard ("JTPT" for
#     ChatGPT): its spelling says little, so only its sound is compared
VOCAB_MIN_CHAR = 0.60
VOCAB_MIN_SOUND = 0.60
VOCAB_OTHER_WORD_MIN_CHAR = 0.40
VOCAB_OTHER_WORD_MIN_SOUND = 0.30
_ACRONYM = re.compile(r"^[A-Z]{2,5}$")
ACRONYM_MIN_LENGTH_RATIO = 0.4


def _vocabulary_hit(cores: list[str], term: str) -> Optional[tuple[float, float]]:
    tkey = _key(term)
    term_words = [_key(w) for w in term.split()]
    # "prompts" is the word "prompt" being said, not a mishearing of it.
    heard = [h[:-1] if h.endswith("s") and h[:-1] in term_words else h
             for h in (_key(c) for c in cores)]
    key = "".join(heard)
    if not key or key in tkey or tkey in key:
        return None
    acronym = len(cores) == 1 and bool(_ACRONYM.match(cores[0]))
    low = ACRONYM_MIN_LENGTH_RATIO if acronym else LENGTH_RATIO[0]
    if not low <= len(key) / len(tkey) <= LENGTH_RATIO[1]:
        return None
    char = fuzz.ratio(key, tkey) / 100
    sound = fuzz.ratio(sound_key(key), sound_key(tkey)) / 100
    if sound < VOCAB_MIN_SOUND or (char < VOCAB_MIN_CHAR and not acronym):
        return None
    if len(cores) > 1:
        for trimmed in (heard[1:], heard[:-1]):
            tk = "".join(trimmed)
            if tk and fuzz.ratio(tk, tkey) / 100 >= char:
                return None
    if len(term_words) > 1 and any(h in term_words for h in heard):
        others = [h for h in heard if h not in term_words]
        rest = [w for w in term_words if w not in heard]
        if not others or not rest:
            return None
        for h in others:
            if not any(fuzz.ratio(h, w) / 100 >= VOCAB_OTHER_WORD_MIN_CHAR
                       and fuzz.ratio(sound_key(h), sound_key(w)) / 100 >= VOCAB_OTHER_WORD_MIN_SOUND
                       for w in rest):
                return None
    return round(char, 2), round(sound, 2)


def flag_vocabulary(entries: list[Entry], vocabulary: list[str]) -> list[dict]:
    flags = []
    for entry in entries:
        tokens = _tokens(entry.text)
        for term in vocabulary:
            if len(_key(term)) < 3:
                continue
            width = len(term.split()) + (1 if len(term.split()) == 1 else 0)
            for first in range(len(tokens)):
                for last in range(first, min(first + width, len(tokens))):
                    hit = _vocabulary_hit([t.core for t in tokens[first:last + 1]], term)
                    if hit:
                        flags.append({"index": entry.index, "start": entry.start,
                                      "heard": " ".join(t.raw for t in tokens[first:last + 1]),
                                      "term": term, "char": hit[0], "sound": hit[1],
                                      "tier": "vocabulary"})
    return flags


# --------------------------------------------------------------------------- #
# All of it
# --------------------------------------------------------------------------- #
@dataclass
class CleanupResult:
    entries: list[Entry]
    language_code: Optional[str]
    duplicates_removed: list[dict] = field(default_factory=list)
    script_fixes: list[dict] = field(default_factory=list)
    term_fixes: list[dict] = field(default_factory=list)
    suspects: list[dict] = field(default_factory=list)

    @property
    def suspect_indices(self) -> list[int]:
        return sorted({s["index"] for s in self.suspects})

    def report(self) -> dict:
        return {"language_code": self.language_code,
                "entries": len(self.entries),
                "counts": {"duplicates_removed": len(self.duplicates_removed),
                           "script_fixes": len(self.script_fixes),
                           "term_fixes": len(self.term_fixes),
                           "suspect_entries": len(self.suspect_indices)},
                "duplicates_removed": self.duplicates_removed,
                "script_fixes": self.script_fixes,
                "term_fixes": self.term_fixes,
                "suspects": self.suspects}


def clean(response: dict, terms: list[str], vocabulary: Optional[list[str]] = None,
          *, remove_echoes: bool = True) -> CleanupResult:
    """Steps 1-3 on a Sarvam batch response, plus the flags for step 4."""
    terms = [t for t in dict.fromkeys(t.strip() for t in terms if t and t.strip())]
    vocabulary = [v for v in dict.fromkeys(v.strip() for v in vocabulary or [] if v and v.strip())]
    entries = entries_from_sarvam(response)
    result = CleanupResult(entries=entries, language_code=(response or {}).get("language_code"))
    if remove_echoes:
        result.entries, result.duplicates_removed = remove_duplicates(entries)
    result.script_fixes = fix_scripts(result.entries, result.language_code)
    result.term_fixes, term_suspects = fix_terms(result.entries, terms)
    result.suspects = term_suspects + flag_vocabulary(result.entries, vocabulary)
    return result


# --------------------------------------------------------------------------- #
# The shape the rest of the pipeline reads
# --------------------------------------------------------------------------- #
def to_deepgram_shape(entries: list[Entry]) -> dict[str, Any]:
    """A Deepgram-shaped response: each entry's words spread evenly across its
    span (Sarvam gives no word timings) and one utterance per entry, so
    sales_call_analyzer/transcript.py could read it unchanged."""
    speakers: dict[str, int] = {}
    words, utterances = [], []
    for e in entries:
        spk = speakers.setdefault(e.speaker_id, int(e.speaker_id) if e.speaker_id.isdigit()
                                  else len(speakers))
        toks = e.text.split()
        step = (e.end - e.start) / max(len(toks), 1)
        for k, tok in enumerate(toks):
            words.append({"word": tok, "punctuated_word": tok, "start": e.start + k * step,
                          "end": e.start + (k + 1) * step - 0.001, "speaker": spk,
                          "confidence": 1.0})
        utterances.append({"speaker": spk, "start": e.start, "end": e.end,
                           "transcript": e.text, "confidence": 1.0})
    return {"metadata": {"duration": max((e.end for e in entries), default=0.0)},
            "results": {"channels": [{"alternatives": [{
                "transcript": " ".join(e.text for e in entries), "words": words}]}],
                "utterances": utterances}}


def entries_as_dicts(entries: list[Entry]) -> list[dict]:
    return [asdict(e) for e in entries]
