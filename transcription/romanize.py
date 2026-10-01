"""
Indic script -> plain Latin letters, for MATCHING, never for display.

WHY THIS EXISTS
    Deepgram writes Hindi in Devanagari, Kannada in Kannada script, and so on.
    The CRM stores "Rajan" and "ScaleSerum" in Latin letters. Every name and
    brand check in speakers.py was a Latin regex, so on a Hindi call "मेरा नाम
    राजन है" never matched and the rep stayed unidentified. Measured on
    mycall3.mp3 (2026-09-15).

    The output is deliberately lossy: diacritics gone, word-final inherent
    vowel dropped (राजन -> "rajan", not "rajana"), lowercase. It is a key for
    comparing names across scripts, not a transliteration anyone should read.
    The transcript itself is never rewritten.

    Also used by the accuracy harness to score a transcript against a reference
    written in a different script ("demo" vs "डेमो").
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Unicode blocks of the scripts our calls use, with the sanscript scheme name.
# Tamil has no sanscript mapping that survives its missing voiced stops, so it
# takes the generic path below.
_BLOCKS = [
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, None),          # Tamil -> generic
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
]

_WORD_FINAL_SCHWA = re.compile(r"(?<=[bcdfghjklmnpqrstvwxyzḍḥḷṁṃṅṇṛṣṭñś])a\b")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def _script_of(ch: str):
    code = ord(ch)
    for low, high, scheme in _BLOCKS:
        if low <= code <= high:
            return scheme or "generic"
    return None


def is_indic(ch: str) -> bool:
    return _script_of(ch) is not None


def indic_share(text: str) -> float:
    """Fraction of letters that are in an Indic script. 0.0 for empty text."""
    letters = [c for c in text or "" if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if is_indic(c)) / len(letters)


def _generic(run: str) -> str:
    """Romanise an abugida from Unicode character names.

    'TAMIL LETTER KA' -> 'ka', a vowel sign replaces the inherent 'a', the
    virama (Tamil pulli) removes it. Crude, but consistent, which is all a
    matching key needs.
    """
    out: list[str] = []
    pending_inherent = False
    for ch in run:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if " VOWEL SIGN " in name:
            if pending_inherent and out and out[-1].endswith("a"):
                out[-1] = out[-1][:-1]
            out.append(name.rsplit(" ", 1)[-1].lower())
            pending_inherent = False
        elif name.endswith(" SIGN VIRAMA") or name.endswith(" SIGN PULLI"):
            if pending_inherent and out and out[-1].endswith("a"):
                out[-1] = out[-1][:-1]
            pending_inherent = False
        elif " LETTER " in name:
            base = name.rsplit(" ", 1)[-1].lower()
            out.append(base)
            # Independent vowels (LETTER A, LETTER II) carry no inherent vowel.
            pending_inherent = base not in ("a", "aa", "i", "ii", "u", "uu", "e", "ee",
                                            "ai", "o", "oo", "au")
        elif ch.isspace():
            out.append(" ")
            pending_inherent = False
    return "".join(out)


# Malayalam chillu letters (a consonant with no vowel, written as one sign) are
# not in sanscript's table and came through untouched: "ഞാൻ" -> "naൻ". Each is
# its consonant + virama, which sanscript does know.
_CHILLU = str.maketrans({"ൺ": "ണ്", "ൻ": "ന്",
                         "ർ": "ര്", "ൽ": "ല്",
                         "ൾ": "ള്", "ൿ": "ക്"})


@lru_cache(maxsize=4096)
def _run_to_latin(run: str, scheme: str) -> str:
    if scheme == "generic":
        return _generic(run)
    if scheme == "malayalam":
        run = run.translate(_CHILLU)
    try:
        from indic_transliteration import sanscript
        return sanscript.transliterate(run, scheme, sanscript.IAST)
    except Exception:  # noqa: BLE001 - library missing or a codepoint it rejects
        return _generic(run)


def _fold(text: str) -> str:
    """Drop diacritics and the word-final inherent vowel; lowercase."""
    text = _WORD_FINAL_SCHWA.sub("", text)
    # Anusvara and candrabindu are written "n" in everyday Latin Hinglish:
    # मैं -> "main", हूँ -> "hun", ಮಂಜುನಾಥ್ -> "manjunath".
    text = text.replace("ṃ", "n").replace("ṁ", "n").replace("~", "n")
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower()


def romanize(text: str) -> str:
    """Latin matching key for any mix of Latin and Indic text.

    Latin runs pass through lowercased; Indic runs are transliterated and
    folded. Punctuation becomes a space.
    """
    if not text:
        return ""
    pieces: list[str] = []
    run, run_scheme = [], None

    def flush():
        if run:
            pieces.append(_fold(_run_to_latin("".join(run), run_scheme)))
            run.clear()

    for ch in text:
        scheme = _script_of(ch)
        if scheme is not None:
            if run and scheme != run_scheme:
                flush()
            run_scheme = scheme
            run.append(ch)
            continue
        if ch.isspace() and run:
            run.append(ch)        # keep words of one script in one run
            continue
        flush()
        pieces.append(ch.lower())
    flush()
    joined = _NON_WORD.sub(" ", "".join(pieces))
    return _SPACES.sub(" ", joined).strip()
