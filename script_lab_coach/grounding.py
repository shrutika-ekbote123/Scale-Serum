"""Is this answer supported by the facts it was given?

Pure functions, no I/O. The evaluation harness scores stored answers with them
today; the runtime validator (phase 5) rejects live turns with the same code, so
what the harness measures and what production enforces cannot drift apart.

THE NUMBER RULE
    Every figure in an answer must be one Python computed, or one that already
    appears in the material the model was shown - the script, the review's own
    comments, the brand's context. Anything else was invented, and an invented
    number in marketing coaching is worse than no answer: it is a confident,
    checkable claim that happens to be false.

    This mirrors AI Briefings, deliberately. Two features enforcing the same
    rule two different ways is two chances to get it wrong.

WHY REWRITES ARE CHECKED TOO
    A suggested rewrite is ad copy that a human may paste into a live campaign.
    "Save 12 hours a week" invented by a model becomes an advertising claim the
    brand cannot substantiate. Rewrites are held to the same rule as the prose:
    no figure that is not already established.

RHETORICAL NUMBERS
    Copywriting advice legitimately says "in the first three seconds" and "two
    things to fix". Those are idiom, not measurement. One, two and three are
    therefore allowed - but ONLY when they carry no unit. "3%" or "3/10" or
    "\u20b93" is a measurement and must be in the facts like any other.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# A number, with optional thousands separators and decimals.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# A unit immediately before or after a number turns idiom into measurement.
_UNIT_AFTER = re.compile(r"^\s*(%|/\s*\d+|x\b|x)")
_UNIT_BEFORE = re.compile(r"(\u20b9|\$|rs\.?|inr|usd)\s*$", re.IGNORECASE)

RHETORICAL = {"1", "2", "3"}

# Values a template renders when a field is missing. The coach that ships today
# puts "undefined" in front of users 22 times in 246 turns; this is here so that
# can never silently happen again.
PLACEHOLDERS = ("undefined", "nan", "null", "[object object]", "{{", "n/a")

# Matched on word boundaries: "nan" is a placeholder, but it is also inside
# "finance" and "Andromeda". Substring matching here produced false alarms in
# the first run of the evaluation suite, which is exactly how a gate gets
# switched off.
_PLACEHOLDER_RE = re.compile(
    r"(?<![a-z0-9])(?:undefined|nan|null|n/a|\[object object\]|\{\{)(?![a-z0-9])",
    re.IGNORECASE)


def _clean(token: str) -> str:
    token = token.replace(",", "").rstrip(".")
    if token.endswith(".0"):
        token = token[:-2]
    return token


def numbers(text: str) -> set:
    """Every numeric token in the text, normalised."""
    return {_clean(t) for t in _NUMBER.findall(text or "") if _clean(t)}


def _has_unit(text: str, match: re.Match) -> bool:
    after = text[match.end():match.end() + 6]
    before = text[max(0, match.start() - 5):match.start()]
    return bool(_UNIT_AFTER.match(after) or _UNIT_BEFORE.search(before))


def unsupported_numbers(text: str, allowed: Iterable) -> list:
    """Figures in `text` that nothing supports, in order of appearance.

    `allowed` may hold numbers or strings; both are normalised the same way."""
    permitted = {_clean(str(a)) for a in allowed}
    permitted |= {_clean(f"{float(a):g}") for a in allowed
                  if isinstance(a, (int, float))}
    out = []
    for match in _NUMBER.finditer(text or ""):
        token = _clean(match.group())
        if not token or token in permitted:
            continue
        if token in RHETORICAL and not _has_unit(text, match):
            continue
        out.append(token)
    return out


def placeholders(text: str) -> list:
    """Template leakage - the "undefined" class of bug, visible to the user."""
    return sorted({m.group().lower() for m in _PLACEHOLDER_RE.finditer(text or "")})


# Brand rows that are really channel names. Matching these would flag ordinary
# marketing English - "seen on Google Search", "the Meta feed" - and a gate that
# fires on normal sentences is one people switch off.
_NOT_REALLY_BRANDS = {"googleads", "google", "meta", "facebook", "instagram",
                      "linkedin", "youtube", "whatsapp", "ads", "test", "brand"}


def _squash(value: str) -> str:
    """Lowercase, drop punctuation and collapse doubled letters.

    This exists because of one live fact: the brands table spells it
    "Lawttorney" and the Brand Brain text spells it "Lawtorney". An exact match
    found neither in the other, so the leak check passed while the coach was
    writing another company's name into this brand's ad copy. Both spellings
    squash to "lawtorney"."""
    flat = re.sub(r"[^a-z0-9]+", "", (value or "").lower())
    return re.sub(r"(.)\1+", r"\1", flat)


def name_pattern(name: str):
    """A regex matching every spelling of a brand name that squashes the same.

    Used to REMOVE the name from material before the model sees it, so it has
    to tolerate the same variation `_squash` forgives: doubled letters, casing
    and punctuation. Detection and redaction share this, because a check that
    finds a name redaction cannot remove is a warning with no teeth."""
    squashed = _squash(name)
    if len(squashed) < 4:
        return None
    body = r"[^a-z0-9]*".join(f"{re.escape(c)}+" for c in squashed)
    # An optional trailing domain: "LawTorney.ai" is the same company.
    return re.compile(rf"\b{body}(?:\.[a-z]{{2,4}})?(?:'s)?", re.IGNORECASE)


def foreign_brands(text: str, *, own: Optional[str] = None,
                   others: Iterable = ()) -> list:
    """Other brands' names appearing in this brand's coaching.

    Not hypothetical: 38% of DI's stored reviews mention Lawttorney, because
    DI's Brand Brain holds Lawttorney's content. A coach reading that document
    will repeat it, so the leak is measured rather than assumed absent."""
    haystack = _squash(text)
    own_squashed = _squash(own)
    hits = []
    for name in others:
        name = (name or "").strip()
        if len(name) < 4:
            continue  # "DI", "T" and friends match everything
        squashed = _squash(name)
        if len(squashed) < 4 or squashed in _NOT_REALLY_BRANDS:
            continue
        if own_squashed and (squashed == own_squashed
                             or squashed in own_squashed or own_squashed in squashed):
            continue
        if squashed in haystack:
            hits.append(name)
    return sorted(set(hits))


def supported(text: str, allowed: Iterable) -> bool:
    return not unsupported_numbers(text, allowed)
