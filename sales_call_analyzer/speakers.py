"""
Speaker role resolution - deciding which diarized voice is the representative,
which is the customer, and which we simply do not know.

DIARIZATION IS NOT IDENTIFICATION
    Deepgram answers "who spoke when". It cannot answer "which real person is
    speaker_0". That second question is answered here, only from CRM facts that
    appear in the conversation, and it is left unanswered whenever the evidence
    is thin.

RULES THIS FILE ENFORCES
    * speaker_id is never renamed. Roles are an annotation with a stated basis.
    * Names come from the CRM record or nowhere. Nothing is invented.
    * Elimination ("the other one must be the customer") applies ONLY to
      two-speaker calls. On a three-way call the third voice is a participant,
      not a customer we guessed at.
    * Turn order and talk-time are never used to infer a role. "The first
      speaker is the rep" is false for inbound calls, and "whoever talks most is
      the rep" is exactly the bias the analysis would then be scoring.
    * "unknown" is a valid outcome and the UI must render it.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from . import (
    ROLE_BASIS_CUSTOMER_ADDRESSED,
    ROLE_BASIS_CUSTOMER_NAME,
    ROLE_BASIS_ELIMINATION,
    ROLE_BASIS_REP_SELF_INTRO,
    ROLE_BASIS_SUPPLIED,
    ROLE_BASIS_UNRESOLVED,
    ROLE_CUSTOMER,
    ROLE_PARTICIPANT,
    ROLE_SALES_REP,
    ROLE_UNKNOWN,
    SPEAKER_ROLES,
)
from .models import CustomerInfo, NormalizedTranscript, RepInfo

# How much of the call counts as "the opening", where introductions happen.
OPENING_TURNS = int(os.environ.get("SCA_OPENING_TURNS", 12))

# "this is Rajan", "my name is Rajan", "I am Rajan", "Rajan here", "Rajan speaking"
# Honorifics sit between the greeting and the name far more often than not on
# these calls ("Hi, mister Sanjay"), so every name pattern tolerates one.
_HONORIFIC = r"(?:mr|mrs|ms|miss|mister|sir|madam|ma'?am|shri|smt|dr|prof)\.?\s+"

_SELF_INTRO = (
    r"(?:this is|my name is|myself|i am|i'm|it's)\s+(?:" + _HONORIFIC + r")?{name}\b",
    r"\b{name}\s+(?:here|speaking|this side)\b",
)
# "calling from Lawtorney", "I'm from Lawtorney", "on behalf of Lawtorney"
_ORG_INTRO = r"(?:calling from|from|on behalf of|representing|with)\s+{name}\b"
# "Hi Meera", "Hi, mister Sanjay", "Am I speaking to Meera", "Is this Meera"
_ADDRESSES = (
    r"(?:hi|hello|hey|good morning|good afternoon|good evening|namaste)[,\s]+"
    r"(?:" + _HONORIFIC + r")?{name}\b",
    r"(?:speaking (?:to|with)|talking (?:to|with)|is this|am i speaking to)\s+"
    r"(?:" + _HONORIFIC + r")?{name}\b",
)

# How many turns after someone is addressed by name we will look for their reply.
ADDRESS_REPLY_WINDOW = int(os.environ.get("SCA_ADDRESS_REPLY_WINDOW", 3))


def _first_name(full: Optional[str]) -> Optional[str]:
    parts = [p for p in re.split(r"\s+", (full or "").strip()) if p]
    return parts[0] if parts else None


def _name_patterns(templates, name: str) -> list[re.Pattern]:
    escaped = re.escape(name)
    return [re.compile(t.format(name=escaped), re.IGNORECASE) for t in templates]


def _opening_text(transcript: NormalizedTranscript, speaker_id: str) -> str:
    """What this speaker said early in the call, where introductions live."""
    said = [s.text for s in transcript.segments[:OPENING_TURNS] if s.speaker_id == speaker_id]
    return " ".join(said)


def _all_text(transcript: NormalizedTranscript, speaker_id: str) -> str:
    return " ".join(s.text for s in transcript.segments if s.speaker_id == speaker_id)


def resolve_roles(transcript: NormalizedTranscript,
                  rep: Optional[RepInfo] = None,
                  customer: Optional[CustomerInfo] = None,
                  brand_name: Optional[str] = None) -> NormalizedTranscript:
    """Annotate transcript.speakers with roles, names, basis and confidence.

    Mutates and returns the transcript. Safe to call on a transcript with any
    number of speakers, including zero.
    """
    rep = rep or RepInfo()
    customer = customer or CustomerInfo()

    rep_first = _first_name(rep.name)
    customer_first = _first_name(customer.name)

    # A caller-supplied structured transcript may already carry roles. Trust it,
    # record that we did, and do not re-derive.
    #
    # The role_basis check is load-bearing: a transcript reused from an earlier
    # run also arrives with roles, but those were DERIVED, not supplied. Without
    # this test they would be mistaken for the caller's own assertion, mislabelled
    # `supplied_by_caller`, and - worse - frozen in, so corrected CRM names could
    # never re-resolve anyone.
    supplied = {s.speaker_id: s.role for s in transcript.speakers
                if s.role_basis == ROLE_BASIS_SUPPLIED
                and s.role in SPEAKER_ROLES and s.role != ROLE_UNKNOWN}

    # Clear every previously derived annotation so this run starts from the
    # transcript itself, not from what some earlier run concluded.
    for speaker in transcript.speakers:
        if speaker.speaker_id not in supplied:
            speaker.role = ROLE_UNKNOWN
            speaker.name = None
            speaker.role_basis = ROLE_BASIS_UNRESOLVED
            speaker.role_confidence = "none"

    rep_id: Optional[str] = None
    customer_id: Optional[str] = None
    basis: dict[str, str] = {}
    confidence: dict[str, str] = {}

    for speaker_id, role in supplied.items():
        basis[speaker_id] = ROLE_BASIS_SUPPLIED
        confidence[speaker_id] = "high"
        if role == ROLE_SALES_REP and rep_id is None:
            rep_id = speaker_id
        elif role == ROLE_CUSTOMER and customer_id is None:
            customer_id = speaker_id

    speaker_ids = [s.speaker_id for s in transcript.speakers]

    # ---- 1. the representative, from a self-introduction or the brand name ---
    if rep_id is None:
        for speaker_id in speaker_ids:
            opening = _opening_text(transcript, speaker_id)
            if not opening:
                continue
            if rep_first and any(p.search(opening) for p in _name_patterns(_SELF_INTRO, rep_first)):
                rep_id = speaker_id
                basis[speaker_id] = ROLE_BASIS_REP_SELF_INTRO
                confidence[speaker_id] = "high"
                break
            if brand_name and re.search(_ORG_INTRO.format(name=re.escape(brand_name)),
                                        opening, re.IGNORECASE):
                rep_id = speaker_id
                basis[speaker_id] = ROLE_BASIS_REP_SELF_INTRO
                confidence[speaker_id] = "medium"
                break

    # ---- 2. the customer, from a self-introduction --------------------------
    if customer_id is None and customer_first:
        patterns = _name_patterns(_SELF_INTRO, customer_first)
        for speaker_id in speaker_ids:
            if speaker_id == rep_id:
                continue
            if any(p.search(_all_text(transcript, speaker_id)) for p in patterns):
                customer_id = speaker_id
                basis[speaker_id] = ROLE_BASIS_CUSTOMER_NAME
                confidence[speaker_id] = "high"
                break

    # ---- 3. the customer, from being addressed by name and answering ---------
    # "Hi, mister Sanjay" followed by someone else speaking identifies the person
    # addressed. This is the only route that works on a call with three or more
    # voices where the customer never introduces themselves - which is the normal
    # shape of an outbound call that goes through a receptionist or a family
    # member. Elimination (step 4) cannot help there.
    if customer_id is None and customer_first:
        patterns = _name_patterns(_ADDRESSES, customer_first)
        for segment in transcript.segments[:OPENING_TURNS]:
            if not any(p.search(segment.text) for p in patterns):
                continue
            # The next different voice to speak is the person being addressed.
            window = transcript.segments[segment.index + 1:
                                         segment.index + 1 + ADDRESS_REPLY_WINDOW]
            replier = next((s.speaker_id for s in window
                            if s.speaker_id != segment.speaker_id
                            and s.speaker_id != rep_id), None)
            if replier:
                customer_id = replier
                basis[replier] = ROLE_BASIS_CUSTOMER_ADDRESSED
                confidence[replier] = "medium"
                break

    # ---- 3b. whoever addresses the customer by name is not the customer ------
    # Only useful for finding the rep, and only when nobody else has been found.
    if rep_id is None and customer_first:
        patterns = _name_patterns(_ADDRESSES, customer_first)
        for speaker_id in speaker_ids:
            if speaker_id == customer_id:
                continue
            if any(p.search(_opening_text(transcript, speaker_id)) for p in patterns):
                rep_id = speaker_id
                basis[speaker_id] = ROLE_BASIS_REP_SELF_INTRO
                confidence[speaker_id] = "low"
                break

    # ---- 4. elimination, two-speaker calls only -----------------------------
    named = [sid for sid in speaker_ids if sid != "speaker_unattributed"]
    if len(named) == 2:
        if rep_id and customer_id is None:
            other = next(sid for sid in named if sid != rep_id)
            customer_id = other
            basis[other] = ROLE_BASIS_ELIMINATION
            confidence[other] = "medium"
        elif customer_id and rep_id is None:
            other = next(sid for sid in named if sid != customer_id)
            rep_id = other
            basis[other] = ROLE_BASIS_ELIMINATION
            confidence[other] = "medium"

    # ---- 5. write the annotations back --------------------------------------
    for speaker in transcript.speakers:
        sid = speaker.speaker_id
        if sid == rep_id:
            speaker.role = ROLE_SALES_REP
            speaker.name = rep.name or None
        elif sid == customer_id:
            speaker.role = ROLE_CUSTOMER
            speaker.name = customer.name or None
        elif supplied.get(sid) in (ROLE_PARTICIPANT,):
            speaker.role = ROLE_PARTICIPANT
            speaker.name = None
        elif rep_id is not None:
            # We know who the rep is, so this voice is definitely someone else -
            # but on a 3+ speaker call we cannot say they are the customer.
            speaker.role = ROLE_PARTICIPANT
            speaker.name = None
            basis.setdefault(sid, ROLE_BASIS_UNRESOLVED)
            confidence.setdefault(sid, "low")
        else:
            speaker.role = ROLE_UNKNOWN
            speaker.name = None

        speaker.role_basis = basis.get(sid, ROLE_BASIS_UNRESOLVED)
        speaker.role_confidence = confidence.get(sid, "none")

    return transcript


def role_summary(transcript: NormalizedTranscript) -> dict:
    """Compact description of what was resolved, for the report and the logs."""
    by_role: dict[str, list[str]] = {}
    for speaker in transcript.speakers:
        by_role.setdefault(speaker.role, []).append(speaker.speaker_id)
    return {
        "speaker_count": transcript.speaker_count,
        "roles": by_role,
        "sales_rep_identified": ROLE_SALES_REP in by_role,
        "customer_identified": ROLE_CUSTOMER in by_role,
        "unresolved": by_role.get(ROLE_UNKNOWN, []) + by_role.get(ROLE_PARTICIPANT, []),
    }


def render_for_prompt(transcript: NormalizedTranscript) -> str:
    """Speaker table for the prompt. Unresolved roles are shown as unresolved so
    the model does not assume an attribution we did not make."""
    lines = []
    for speaker in transcript.speakers:
        name = speaker.name or "(name not known)"
        lines.append(
            f"- {speaker.speaker_id}: role={speaker.role}, name={name}, "
            f"basis={speaker.role_basis}, turns={speaker.turn_count}, "
            f"talk_time={speaker.talk_time_seconds:.0f}s")
    if not lines:
        return "(no speakers were identified)"
    lines.append("Attribute every piece of evidence to the speaker_id shown above. "
                 "Where a role is 'unknown' or 'participant', do not assume who that person is.")
    return "\n".join(lines)
