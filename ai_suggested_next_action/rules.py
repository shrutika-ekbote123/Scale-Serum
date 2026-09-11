"""The decision: which action, how urgent, and the evidence behind it.

Pure and deterministic - no I/O, no model, and the clock is a parameter. Given
the same lead context and `now`, `decide` always returns the same decision, which
is what makes it testable and what lets the wording be cached.

Rules are evaluated in priority order; the first that applies wins:

    1. no usable activity            -> insufficient_data     low
    2. lost / irrelevant             -> disqualify            low
    3. unanswered inbound WhatsApp   -> reply_whatsapp        high
    4. callback requested            -> callback              high
    5. bought the core/premium offer -> onboard (medium) / nurture_next_tier (low)
    6. bought the entry offer only   -> pitch_core_offer      by days since payment
    7. called, not paid              -> follow_up_objection / retry_contact / follow_up
    8. never contacted, not paid     -> call_now / call_probe_repeat / follow_up / re_engage
                                        by hours since last activity vs this brand's
                                        own form-to-payment window
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from .tiers import KIND_RANK, classify_amount, tier_product

DEFAULT_TZ = "Asia/Kolkata"

DISPOSITION_TEXT = {
    "SQL": "SQL (sales-qualified)", "MQL": "MQL (marketing-qualified)",
    "non_sql": "not sales-qualified", "irrelevant": "irrelevant",
    "callback": "callback requested", "no_answer": "no answer",
}

# Flags that describe missing or doubtful data and so lower confidence. Other
# flags are informational only.
MATERIAL_FLAGS = {
    "synthetic_touchpoints_ignored", "multiple_phone_numbers",
    "default_conversion_window", "brand_brain_missing", "brand_currency_invalid",
}


# --------------------------------------------------------------------------- helpers
def brand_zone(name: Optional[str]) -> ZoneInfo:
    try:
        return ZoneInfo(name) if name else ZoneInfo(DEFAULT_TZ)
    except Exception:  # noqa: BLE001 - unknown zone names are data, not bugs
        return ZoneInfo(DEFAULT_TZ)


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fmt_date(dt: Optional[datetime], tz: ZoneInfo) -> Optional[str]:
    if dt is None:
        return None
    local = _aware(dt).astimezone(tz)
    return f"{local.day} {local.strftime('%b %Y')}"


def _indian_grouping(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def fmt_money(amount: Optional[float], currency: Optional[str]) -> Optional[str]:
    if amount is None:
        return None
    whole = float(amount).is_integer()
    if (currency or "").upper() == "INR":
        return "₹" + (_indian_grouping(int(amount)) if whole else f"{amount:,.2f}")
    number = f"{int(amount):,}" if whole else f"{amount:,.2f}"
    return f"{number} {currency.upper()}" if currency else number


def _valid_currency(code: Optional[str]) -> Optional[str]:
    return code.upper() if code and re.fullmatch(r"[A-Za-z]{3}", code) else None


def parse_sales_cycle_hours(raw: Optional[str]) -> Optional[float]:
    """'3+ months' -> 2160h, '1-2 weeks' -> 336h, 'Same day' -> 24h. Takes the
    upper bound of a range. None when the answer carries no duration."""
    if not raw:
        return None
    text = str(raw).lower()
    if "same day" in text or "instant" in text:
        return 24.0
    unit_days = {"day": 1, "week": 7, "month": 30, "quarter": 90, "year": 365}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:[-–to]+\s*(\d+(?:\.\d+)?))?\s*\+?\s*"
                  r"(day|week|month|quarter|year)", text)
    if m:
        value = float(m.group(2) or m.group(1))
        return value * unit_days[m.group(3)] * 24
    m = re.search(r"\b(?:a|an|one)\s+(day|week|month|quarter|year)", text)
    if m:
        return unit_days[m.group(1)] * 24.0
    return None


def conversion_window(stats: dict, brand_brain: Optional[dict], cfg: dict) -> dict:
    """p50/p75 hours from first form to first payment. This brand's own history
    first, then its Brand Brain sales-cycle answer, then the rulebook default."""
    w = cfg["windows"]
    if (stats.get("converters") or 0) >= w["min_converters"] and stats.get("p75_hours"):
        return {"source": "brand_history", "converters": stats["converters"],
                "p50_hours": round(stats["p50_hours"], 1),
                "p75_hours": round(stats["p75_hours"], 1)}
    cycle = ((brand_brain or {}).get("answers") or {}).get("salesCycle")
    hours = parse_sales_cycle_hours(cycle)
    if hours:
        return {"source": "brand_brain_sales_cycle", "converters": stats.get("converters") or 0,
                "p50_hours": round(hours / 2, 1), "p75_hours": round(hours, 1)}
    return {"source": "default", "converters": stats.get("converters") or 0,
            "p50_hours": float(w["default_p50_hours"]),
            "p75_hours": float(w["default_p75_hours"])}


def _days_ago(days: int) -> str:
    """'today' / '1 day ago' / '5 days ago' - the basis is shown to reps."""
    if days <= 0:
        return "today"
    return "1 day ago" if days == 1 else f"{days} days ago"


def _hours_text(hours: float) -> str:
    if hours < 48:
        return f"{round(hours)} hours"
    return f"{round(hours / 24)} days"


# --------------------------------------------------------------------------- decide
def decide(ctx: dict, *, cfg: dict, tiers: list[dict], window: dict,
           brand_brain: Optional[dict], now: datetime) -> dict:
    """Decide the next action for one lead. See the module docstring for order."""
    now = _aware(now)
    th = cfg["thresholds"]
    tz = brand_zone(ctx["brand"].get("timezone"))
    brand_currency = _valid_currency(ctx["brand"].get("currency"))

    flags: list[str] = []
    all_tps = ctx.get("touchpoints") or []
    synthetic = [t for t in all_tps if t.get("synthetic")]
    tps = [t for t in all_tps if not t.get("synthetic") and t.get("occurred_at")]
    if synthetic:
        flags.append("synthetic_touchpoints_ignored")
    if (ctx.get("phone_count") or 0) >= 2:
        flags.append("multiple_phone_numbers")
    if window["source"] == "default":
        flags.append("default_conversion_window")
    if not brand_brain:
        flags.append("brand_brain_missing")
    if ctx["brand"].get("currency") and not brand_currency:
        flags.append("brand_currency_invalid")

    forms = [t for t in tps if t["type"] == "form_submit"]
    payments = [t for t in tps if t["type"] == "payment" and (t.get("value") or 0) > 0]
    calls = [c for c in (ctx.get("calls") or []) if c.get("occurred_at")]
    wa = ctx.get("whatsapp") or {}
    last_call = calls[-1] if calls else None
    last_touch = max((_aware(t["occurred_at"]) for t in tps), default=None)

    for p in payments:
        p["tier"] = classify_amount(p["value"], tiers)
        p["currency_display"] = _valid_currency(p.get("currency")) or brand_currency

    def money(p):
        return fmt_money(p["value"], p["currency_display"])

    core_tier = next((t for t in tiers if t["kind"] == "core"), None)
    entry_tier = next((t for t in tiers if t["kind"] == "entry"), None)
    # The currency to show money in: the brand's own when valid, otherwise the
    # one recorded on this lead's payments, otherwise none (a bare number) -
    # never a guessed symbol.
    display_currency = brand_currency or next(
        (p["currency_display"] for p in payments if p["currency_display"]), None)

    facts = {
        "form_count": len(forms),
        "first_touch_date": fmt_date(min((_aware(t["occurred_at"]) for t in tps), default=None), tz),
        "last_touch_date": fmt_date(last_touch, tz),
        # Omitted under a day: "0 days ago" is noise on the card.
        "days_since_last_touch": ((now - last_touch).days
                                  if last_touch and (now - last_touch).days >= 1 else None),
        "call_count": len(calls),
        "last_call_date": fmt_date(last_call["occurred_at"], tz) if last_call else None,
        "last_call_disposition": (DISPOSITION_TEXT.get(last_call["disposition"],
                                                       last_call["disposition"])
                                  if last_call and last_call.get("disposition") else None),
        "whatsapp_last_message_date": fmt_date(wa.get("last_message_at"), tz),
        "payments": [{"amount": money(p), "date": fmt_date(p["occurred_at"], tz),
                      "product": p.get("product_code"),
                      "tier": p["tier"]["kind"] if p.get("tier") else None}
                     for p in payments],
        "entry_product": tier_product(entry_tier),
        "core_product": tier_product(core_tier) or "the main offer",
        "core_typical_price": fmt_money(core_tier["typical_amount"], display_currency)
        if core_tier and core_tier.get("typical_amount") else None,
    }

    # What the sales team has already done.
    contacted = bool(calls) or any(t["type"] == "call" for t in tps) or (
        wa.get("conversations", 0) > 0 and wa.get("last_direction") == "outbound")

    paid_kinds = [p["tier"]["kind"] for p in payments if p.get("tier")]
    highest = max(paid_kinds, key=lambda k: KIND_RANK[k]) if paid_kinds else None

    def result(action, level, stage, *, due=None, basis):
        hours = cfg["urgency"][level]["act_within_hours"]
        due_by = _aware(due) if due else now + timedelta(hours=hours)
        return {
            "action_type": action, "urgency_level": level, "stage": stage,
            "due_by": due_by.astimezone(tz).isoformat(timespec="minutes"),
            "urgency_basis": basis,
            "facts": {k: v for k, v in facts.items() if v not in (None, [], "")},
            "evidence": _evidence(forms, payments, calls, wa, tps, tz, money),
            "flags": flags,
            "confidence": _confidence(flags, action),
            "lead_state": {
                "stage": stage, "converted": bool(payments), "contacted": contacted,
                "highest_tier": highest, "payment_count": len(payments),
                "total_paid": fmt_money(sum(p["value"] for p in payments), display_currency)
                if payments else None,
            },
        }

    # 1. Nothing to go on.
    if not tps and not calls:
        return result("insufficient_data", "low", "no_activity",
                      basis="No usable touchpoints or calls are recorded for this lead.")

    # 2. Lost or ruled out on a call, with nothing new since.
    after_last_call = [t for t in tps if last_call and
                       _aware(t["occurred_at"]) > _aware(last_call["occurred_at"])]
    if (ctx["lead"].get("status") == "lost" or
            (last_call and last_call.get("disposition") in ("irrelevant", "non_sql")
             and not after_last_call)):
        facts["disqualified_by"] = ("lost" if ctx["lead"].get("status") == "lost"
                                    else DISPOSITION_TEXT[last_call["disposition"]])
        return result("disqualify", "low", "disqualified",
                      basis="The lead has been ruled out and has not re-engaged since.")

    # 3. The lead is waiting on us in WhatsApp.
    wa_last = _aware(wa.get("last_message_at"))
    if (wa.get("last_direction") == "inbound" and wa_last and
            now - wa_last <= timedelta(hours=th["whatsapp_reply_window_hours"])):
        return result("reply_whatsapp", "high", "awaiting_reply",
                      basis="The lead's last WhatsApp message is unanswered.")

    # 4. A promised callback, unless they have paid since.
    paid_after_call = [p for p in payments if last_call and
                       _aware(p["occurred_at"]) > _aware(last_call["occurred_at"])]
    if last_call and last_call.get("disposition") == "callback" and not paid_after_call:
        cb = _aware(last_call.get("callback_at"))
        return result("callback", "high", "callback_due", due=cb if cb and cb > now else None,
                      basis=("Callback scheduled for " + fmt_date(cb, tz)) if cb
                      else "The lead asked for a callback on the last call.")

    # 5. Bought the main (or premium) offer.
    if highest in ("core", "premium"):
        last_core = [p for p in payments if p.get("tier") and p["tier"]["kind"] in ("core", "premium")][-1]
        facts["last_core_amount"] = money(last_core)
        facts["last_core_date"] = fmt_date(last_core["occurred_at"], tz)
        if last_core.get("product_code"):
            facts["core_product"] = last_core["product_code"]
        days = (now - _aware(last_core["occurred_at"])).days
        if days <= th["onboarding_medium_days"]:
            return result("onboard", "medium", "customer_core",
                          basis=f"Paid for the main offer {_days_ago(days)}; onboarding is time-sensitive.")
        return result("nurture_next_tier", "low", "customer_core",
                      basis=f"Paid for the main offer {_days_ago(days)}.")

    # 6. Bought the entry offer only - the upsell is the money.
    if highest == "entry":
        last_entry = [p for p in payments if p.get("tier") and p["tier"]["kind"] == "entry"][-1]
        facts["last_entry_amount"] = money(last_entry)
        facts["last_entry_date"] = fmt_date(last_entry["occurred_at"], tz)
        facts["entry_product"] = last_entry.get("product_code") or facts.get("entry_product") or "the entry offer"
        if not core_tier:
            return result("nurture_next_tier", "low", "customer_entry",
                          basis="Bought the entry offer; this brand has no higher tier on record.")
        calls_since = [c for c in calls if _aware(c["occurred_at"]) > _aware(last_entry["occurred_at"])]
        if calls_since and calls_since[-1].get("disposition") in ("SQL", "MQL"):
            level = "high" if calls_since[-1]["disposition"] == "SQL" else "medium"
            return result("follow_up_objection", level, "customer_entry",
                          basis="Qualified on a call after buying the entry offer.")
        days = (now - _aware(last_entry["occurred_at"])).days
        level = ("high" if days <= th["entry_pitch_high_days"] else
                 "medium" if days <= th["entry_pitch_medium_days"] else "low")
        return result("pitch_core_offer", level, "customer_entry",
                      basis=f"Bought the entry offer {_days_ago(days)} and has not been offered the main one.")

    # 7. Spoken to, not paid.
    if last_call:
        disposition = last_call.get("disposition")
        days = (now - _aware(last_call["occurred_at"])).days
        if disposition == "SQL":
            return result("follow_up_objection", "high" if days <= th["sql_follow_up_high_days"] else "medium",
                          "in_conversation", basis=f"Sales-qualified on a call {_days_ago(days)} with no payment since.")
        if disposition == "MQL":
            return result("follow_up_objection", "medium", "in_conversation",
                          basis=f"Marketing-qualified on a call {_days_ago(days)} with no payment since.")
        if disposition == "no_answer":
            return result("retry_contact", "medium", "in_conversation",
                          basis="The last call attempt was not answered.")
        return result("follow_up", "medium", "in_conversation",
                      basis=f"Last spoken to {_days_ago(days)} with no recorded outcome.")

    if contacted:
        return result("follow_up", "medium", "in_conversation",
                      basis="Contacted on WhatsApp with no reply or payment yet.")

    # 8. Never contacted, not paid: act inside this brand's buying window.
    hours = (now - last_touch).total_seconds() / 3600 if last_touch else float("inf")
    p50, p75 = window["p50_hours"], window["p75_hours"]
    basis_window = f"half of this brand's buyers pay within {_hours_text(p50)} of their first form"
    if window["source"] != "brand_history":
        basis_window = f"the expected buying window is about {_hours_text(p75)}"
    if hours <= p50:
        action = ("call_probe_repeat" if len(forms) >= th["repeat_registration_forms"]
                  else "call_now")
        return result(action, "high", "new_uncontacted",
                      basis=f"Last active {_hours_text(hours)} ago and {basis_window}.")
    if hours <= p75:
        return result("follow_up", "medium", "cooling",
                      basis=f"Last active {_hours_text(hours)} ago; {basis_window}.")
    return result("re_engage", "low", "dormant",
                  basis=f"Inactive for {_hours_text(hours)}, well past the usual buying window.")


def _confidence(flags: list[str], action: str) -> str:
    if action == "insufficient_data":
        return "low"
    material = sum(1 for f in flags if f in MATERIAL_FLAGS)
    return "high" if material == 0 else "medium" if material <= 2 else "low"


def _evidence(forms, payments, calls, wa, tps, tz, money) -> list[dict]:
    """The facts the card can cite, in a fixed order. Calls and WhatsApp always
    appear, with a zero count, because 'nobody has contacted them' is evidence."""
    out: list[dict] = []
    if forms:
        out.append({"type": "form_submit", "count": len(forms),
                    "first": fmt_date(forms[0]["occurred_at"], tz),
                    "last": fmt_date(forms[-1]["occurred_at"], tz),
                    "sources": sorted({f["source"] for f in forms if f.get("source")})[:5]})
    if payments:
        last = payments[-1]
        out.append({"type": "payment", "count": len(payments),
                    "latest": {"amount": money(last), "date": fmt_date(last["occurred_at"], tz),
                               "product": last.get("product_code"),
                               "tier": last["tier"]["kind"] if last.get("tier") else None}})
    out.append({"type": "sales_call", "count": len(calls),
                "last_date": fmt_date(calls[-1]["occurred_at"], tz) if calls else None,
                "last_disposition": calls[-1].get("disposition") if calls else None})
    out.append({"type": "whatsapp", "conversations": wa.get("conversations", 0),
                "last_direction": wa.get("last_direction"),
                "last_message_date": fmt_date(wa.get("last_message_at"), tz)})
    other: dict[str, int] = {}
    for t in tps:
        if t["type"] not in ("form_submit", "payment"):
            other[t["type"]] = other.get(t["type"], 0) + 1
    for kind, n in sorted(other.items()):
        out.append({"type": kind, "count": n})
    return out
