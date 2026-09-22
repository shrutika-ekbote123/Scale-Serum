"""Leads tab.

New leads are counted by leads.created_at, the one lead timestamp scrumdb never
rewrites. Buyers are first payments (non-backfilled) on D. "Qualified" comes
from sales-call dispositions, because leads.sql_status is filled on a handful
of rows. Hot leads are the purchase-probability model's High band among open
leads nobody has called or messaged.
"""
from __future__ import annotations

from collections import defaultdict

from .. import fmt
from .common import DayContext, join_parts, kpi, result, unavailable, watch, watch_sentence, within
from .money import first_time_buyers, payments_in, revenue


def build(raw: dict, ctx: DayContext) -> dict:
    cfg, lcfg, cur = ctx.cfg, ctx.cfg["leads"], ctx.currency
    leads = raw["leads"]
    if not leads and not raw["payments"]:
        return unavailable("leads", "no_leads",
                           "No leads have been captured for this brand recently.")

    def count(window):
        return sum(1 for l in leads if within(l["created_at"], window))

    new, new_prev, new_week_ago = count(ctx.d), count(ctx.day_n(1)), count(ctx.day_n(7))
    trail_avg = count(ctx.last_days(7, ending_days_ago=1)) / 7

    buyers = first_time_buyers(raw["payments"], ctx.d)
    paid_today = payments_in(raw["payments"], ctx.d)
    rev = revenue(raw["payments"], ctx.d)

    demo = {n.lower() for n in cfg["demo_data"]["sales_call_rep_names"]}
    calls = [c for c in raw["calls"] if (c["rep_name"] or "").lower() not in demo]
    sql_leads = {c["lead_id"] or c["id"] for c in calls
                 if within(c["occurred_at"], ctx.d) and c["disposition"] == "SQL"}
    mql_leads = {c["lead_id"] or c["id"] for c in calls
                 if within(c["occurred_at"], ctx.d) and c["disposition"] == "MQL"}

    # Funnels: leads created in the lookback, and how many have paid by the end of D.
    lookback = ctx.last_days(lcfg["funnel_lookback_days"])
    payers = {p["lead_id"] for p in raw["payments"] if p["lead_id"]}
    by_funnel = defaultdict(lambda: [0, 0])
    for l in leads:
        if l["funnel"] and within(l["created_at"], lookback):
            by_funnel[l["funnel"]][0] += 1
            by_funnel[l["funnel"]][1] += l["id"] in payers
    funnels = sorted(
        ({"funnel": f, "leads": n, "converted": k, "rate": k / n,
          "rate_display": fmt.pct(k / n)}
         for f, (n, k) in by_funnel.items() if n >= lcfg["funnel_min_leads"]),
        key=lambda f: (f["rate"], f["leads"]), reverse=True)
    untagged = sum(1 for l in leads if not l["funnel"] and within(l["created_at"], lookback))
    in_lookback = sum(1 for l in leads if within(l["created_at"], lookback))
    best = funnels[0] if funnels and funnels[0]["converted"] else None

    hot = ctx.hot or {"available": False, "reason": "not_scored", "hot": []}
    hot_ids = [h["lead_id"] for h in hot["hot"]]
    open_qualified = raw.get("open_qualified") if ctx.is_latest else None

    items = []
    if hot["available"] and hot_ids:
        items.append(watch(cfg, "hot_untouched", "high", count=len(hot_ids),
                           hours=lcfg["untouched_hours"]))
    if (new_week_ago and new < lcfg["drop_ratio"] * new_week_ago
            and new < lcfg["drop_ratio"] * trail_avg):
        items.append(watch(cfg, "leads_drop", "medium", count=new, previous=new_week_ago))

    d = {
        "new_leads": f"{new:,}", "new_leads_change": fmt.delta_count(new, new_prev),
        "vs_same_day_last_week": fmt.delta_count(new, new_week_ago),
        "buyers": f"{len(buyers):,}", "payments": f"{len(paid_today):,}",
        "revenue": fmt.money(rev, cur), "sqls": f"{len(sql_leads):,}",
        "mqls": f"{len(mql_leads):,}", "hot_untouched": f"{len(hot_ids):,}",
        "open_qualified": f"{open_qualified:,}" if open_qualified is not None else None,
        "best_funnel": best["funnel"] if best else None,
        "best_funnel_rate": best["rate_display"] if best else None,
        "funnel_lookback_days": lcfg["funnel_lookback_days"],
        "untagged_share": fmt.pct(fmt.safe_div(untagged, in_lookback), 0),
    }
    facts = {
        "window": "yesterday", "new_leads": new, "new_leads_previous_day": new_prev,
        "new_leads_same_day_last_week": new_week_ago,
        "new_leads_7d_average": round(trail_avg, 1),
        "buyers": len(buyers), "payments": len(paid_today), "revenue": rev,
        "sqls_from_calls": len(sql_leads), "mqls_from_calls": len(mql_leads),
        "open_qualified_now": open_qualified,
        "funnels": funnels[:5], "best_funnel": best,
        "leads_without_funnel_share": d["untagged_share"],
        "hot_untouched": {"available": hot["available"], "reason": hot.get("reason"),
                          "count": len(hot_ids),
                          "lead_ids": hot_ids[:lcfg["hot_lead_ids_returned"]],
                          "scored": hot.get("scored"),
                          "untouched_hours": lcfg["untouched_hours"]},
        "display": d,
    }
    kpis = {
        "new_leads": kpi(new, d["new_leads"], new_prev, d["new_leads_change"]),
        "buyers": kpi(len(buyers), d["buyers"]),
        "revenue": kpi(rev, d["revenue"], basis="payments received"),
        "sqls": kpi(len(sql_leads), d["sqls"], basis="sales-call dispositions"),
        "hot_untouched": kpi(len(hot_ids), d["hot_untouched"]),
    }
    if open_qualified is not None:
        kpis["open_qualified"] = kpi(open_qualified, d["open_qualified"], basis="right now")

    summary = join_parts([
        f"{d['new_leads']} new" + (f" ({d['new_leads_change']})" if d["new_leads_change"] else ""),
        f"{d['sqls']} qualified on calls",
        f"{d['buyers']} new buyers ({d['revenue']})",
    ]) + "."
    if best:
        summary += f" Best funnel: {best['funnel']} ({best['rate_display']})."
    if open_qualified is not None:
        summary += f" {d['open_qualified']} qualified leads are open."
    if hot_ids:
        summary += (f" {d['hot_untouched']} hot leads untouched for over "
                    f"{lcfg['untouched_hours']} hours - assign today.")

    bullets = [f"{d['new_leads']} new leads"
               + (f" ({d['new_leads_change']} on the day before, "
                  f"{d['vs_same_day_last_week']} vs the same day last week)."
                  if d["new_leads_change"] and d["vs_same_day_last_week"] else "."),
               f"{d['payments']} payments totalling {d['revenue']}, "
               f"{d['buyers']} of them from first-time buyers.",
               f"{d['sqls']} leads marked SQL and {d['mqls']} MQL on sales calls."]
    if hot["available"]:
        bullets.append(f"{d['hot_untouched']} high-intent leads have had no call or "
                       f"WhatsApp for over {lcfg['untouched_hours']} hours.")
    blocks = [{"key": "leads_sales", "title": "Leads & sales", "bullets": bullets}]
    if funnels:
        blocks.append({"key": "funnels", "title": "Funnels", "bullets": [
            f"{f['funnel']}: {f['rate_display']} of {f['leads']:,} leads converted"
            f" (last {lcfg['funnel_lookback_days']} days)." for f in funnels[:3]]})

    out = result("leads", facts=facts, kpis=kpis, watch_items=items, cfg=cfg,
                 template={"summary": summary, "top": "", "watch": "", "blocks": blocks})
    out["template"]["watch"] = watch_sentence(out["watch"])
    return out
