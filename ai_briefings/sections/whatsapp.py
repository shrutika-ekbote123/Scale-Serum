"""WhatsApp tab.

Unavailable until the brand has a connected WhatsApp Business account: the
broadcasts and conversations in scrumdb today belong to brands with none, and
are seed data. Once an account is connected, real rows start flowing.

  read rate     read / delivered (a message that never arrived cannot be read)
  replies->SQL  leads who messaged in the last N days and were then marked SQL
                on a sales call before the end of D
"""
from __future__ import annotations

from datetime import timedelta

from .. import fmt
from .common import DayContext, join_parts, kpi, result, unavailable, watch, watch_sentence, within


def build(raw: dict, ctx: DayContext) -> dict:
    cfg, wcfg = ctx.cfg, ctx.cfg["whatsapp"]
    wa = raw["whatsapp"]
    if not wa.get("accounts"):
        return unavailable("whatsapp", "whatsapp_not_connected",
                           "WhatsApp is not connected for this brand yet.")

    casts = wa.get("broadcasts") or []
    tot = {k: sum(b[k] for b in casts) for k in ("sent", "delivered", "read", "replied", "failed")}
    delivered_rate = fmt.safe_div(tot["delivered"], tot["sent"])
    read_rate = fmt.safe_div(tot["read"], tot["delivered"])
    best = None
    if casts:
        b = max(casts, key=lambda b: (fmt.safe_div(b["read"], b["delivered"]) or 0, b["sent"]))
        best = {"name": b["name"], "sent": b["sent"], "delivered_rate": fmt.pct(fmt.safe_div(b["delivered"], b["sent"])),
                "read_rate": fmt.pct(fmt.safe_div(b["read"], b["delivered"])), "replies": b["replied"]}

    inbound_today = [m for m in wa.get("inbound") or [] if within(m["at"], ctx.d)]
    first_msg = {}
    for m in sorted(wa.get("inbound") or [], key=lambda m: m["at"]):
        if m["lead_id"] and m["lead_id"] not in first_msg:
            first_msg[m["lead_id"]] = m["at"]
    sql_after = {c["lead_id"] for c in raw["calls"]
                 if c["lead_id"] in first_msg and c["disposition"] == "SQL"
                 and c["occurred_at"] and first_msg[c["lead_id"]] <= c["occurred_at"] < ctx.d.end}
    awaiting = wa.get("awaiting_reply", 0) if ctx.is_latest else None
    templates = wa.get("templates") or {}
    approved_today = (templates.get("approved") or {}).get("approved_today") or []

    items = []
    if awaiting:
        items.append(watch(cfg, "awaiting_reply", "medium", count=awaiting))
    if tot["failed"]:
        items.append(watch(cfg, "broadcast_failed", "medium", count=f"{tot['failed']:,}"))

    d = {
        "broadcasts": f"{len(casts):,}", "sent": f"{tot['sent']:,}",
        "delivered_rate": fmt.pct(delivered_rate), "read_rate": fmt.pct(read_rate),
        "replies": f"{tot['replied']:,}", "inbound_messages": f"{len(inbound_today):,}",
        "conversations": f"{len({m['conversation_id'] for m in inbound_today}):,}",
        "replies_to_sql": f"{len(sql_after):,}", "reply_to_sql_days": wcfg["reply_to_sql_days"],
        "awaiting_reply": f"{awaiting:,}" if awaiting is not None else None,
        "templates_approved": f"{(templates.get('approved') or {}).get('count', 0):,}",
        "templates_pending": f"{(templates.get('pending') or {}).get('count', 0):,}",
    }
    facts = {"window": "yesterday", "broadcasts": len(casts), **tot,
             "delivered_rate": delivered_rate, "read_rate": read_rate, "best_broadcast": best,
             "inbound_messages": len(inbound_today), "replies_to_sql": len(sql_after),
             "awaiting_reply_now": awaiting, "templates_approved_yesterday": approved_today,
             "display": d}
    kpis = {"sent": kpi(tot["sent"], d["sent"]),
            "read_rate": kpi(read_rate, d["read_rate"]),
            "replies": kpi(tot["replied"], d["replies"]),
            "replies_to_sql": kpi(len(sql_after), d["replies_to_sql"])}
    if awaiting is not None:
        kpis["awaiting_reply"] = kpi(awaiting, d["awaiting_reply"], basis="right now")

    if casts:
        summary = join_parts([
            f"{best['name']} broadcast - {d['sent']} sent" if len(casts) == 1
            else f"{d['broadcasts']} broadcasts - {d['sent']} sent",
            f"{d['delivered_rate']} delivered" if delivered_rate is not None else None,
            f"{d['read_rate']} read" if read_rate is not None else None,
            f"{d['replies']} replies",
        ]) + (f" → {d['replies_to_sql']} SQLs." if sql_after else ".")
    elif inbound_today:
        summary = f"No broadcasts; {d['inbound_messages']} inbound messages in {d['conversations']} conversations."
    else:
        summary = "No WhatsApp broadcasts or inbound messages yesterday."
    if approved_today:
        summary += f" Template{'s' if len(approved_today) > 1 else ''} approved by Meta: {', '.join(approved_today[:3])}."
    if awaiting:
        summary += f" {d['awaiting_reply']} conversations awaiting reply in Inbox."

    bullets = [b for b in [
        f"{d['broadcasts']} broadcasts: {d['sent']} sent, {d['delivered_rate']} delivered, "
        f"{d['read_rate']} read, {d['replies']} replies." if casts else None,
        f"{d['inbound_messages']} inbound messages across {d['conversations']} conversations.",
        f"{d['replies_to_sql']} leads who messaged in the last {wcfg['reply_to_sql_days']} "
        f"days were marked SQL on a call.",
        f"Templates: {d['templates_approved']} approved, {d['templates_pending']} pending.",
    ] if b]
    blocks = [{"key": "whatsapp", "title": "WhatsApp", "bullets": bullets}]

    out = result("whatsapp", facts=facts, kpis=kpis, watch_items=items, cfg=cfg,
                 template={"summary": summary, "top": "", "watch": "", "blocks": blocks})
    out["template"]["watch"] = watch_sentence(out["watch"])
    return out
