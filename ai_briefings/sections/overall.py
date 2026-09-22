"""The "All" tab, composed from the four section results.

It reads no data of its own, so its numbers are the section numbers by
construction: the All card can never say ROAS 5.9x while the Ads tab says 2.1x.
Watch items are the most severe across every available section.
"""
from __future__ import annotations

from .common import DayContext, join_parts, result, unavailable

MAX_WATCH = 3


def build(sections: dict, ctx: DayContext) -> dict:
    cfg = ctx.cfg
    live = {k: s for k, s in sections.items() if s and s["available"]}
    if not live:
        return unavailable("all", "no_data", "There is no data to brief on for this brand yet.")

    ads, leads, sales, wa = (live.get(k) for k in ("ads", "leads", "sales", "whatsapp"))
    parts, facts = [], {"sections": sorted(live)}
    if leads:
        ld = leads["facts"]["display"]
        parts.append(f"{ld['new_leads']} leads" + (f" ({ld['new_leads_change']})"
                                                   if ld["new_leads_change"] else ""))
        facts["leads"] = {k: leads["facts"][k] for k in ("new_leads", "buyers", "revenue")}
    if ads:
        ad = ads["facts"]["display"]
        if not leads:
            parts.append(f"{ad['payments_revenue']} revenue")
        else:
            parts.append(f"{leads['facts']['display']['revenue']} revenue")
        if ads["facts"]["blended_roas"] is not None:
            parts.append(f"blended ROAS {ad['blended_roas']}")
        facts["ads"] = {k: ads["facts"][k] for k in ("spend", "blended_roas", "top_campaign")}
    elif leads:
        parts.append(f"{leads['facts']['display']['revenue']} revenue")
    summary = join_parts(parts) + "." if parts else ""
    if sales:
        sd = sales["facts"]["display"]
        if sd["score"]:
            summary += (f" Sales team scored {sd['score']}"
                        + (f" ({sd['trend']})" if sd["trend"] else "")
                        + f" with {sd['closures']} closures ({sd['closures_revenue']}).")
        else:
            summary += f" Sales team: {sd['calls_window']} calls in {sd['window_days']} days."
        facts["sales"] = {k: sales["facts"][k] for k in ("score", "trend_points", "closures")}
    if wa and wa["facts"].get("best_broadcast"):
        b = wa["facts"]["best_broadcast"]
        summary += f" WhatsApp {b['name']} broadcast hit {b['read_rate']} read rate."
        facts["whatsapp"] = {"best_broadcast": b}

    rank = cfg["severity_rank"]
    merged = []
    for key in ("ads", "leads", "sales", "whatsapp"):
        for w in (live.get(key) or {}).get("watch", []):
            merged.append({**w, "section": key})
    merged.sort(key=lambda w: rank.get(w["severity"], 9))
    # The most severe item from each section first, so one noisy section (six
    # campaigns under threshold) cannot crowd out an untouched hot-lead queue.
    firsts, seen = [], set()
    for w in merged:
        if w["section"] not in seen:
            seen.add(w["section"])
            firsts.append(w)
    rest = [w for w in merged if w not in firsts]
    top_watch = sorted(firsts, key=lambda w: rank.get(w["severity"], 9))[:MAX_WATCH]
    top_watch += rest[:MAX_WATCH - len(top_watch)]

    blocks = []
    titles = {"ads": ("ad_performance", "Ad performance"), "leads": ("leads_sales", "Leads & sales"),
              "sales": ("sales_team", "Sales team"), "whatsapp": ("whatsapp", "WhatsApp")}
    for key in ("ads", "leads", "sales", "whatsapp"):
        if key in live:
            bkey, title = titles[key]
            blocks.append({"key": bkey, "title": title,
                           "bullets": [live[key]["template"]["summary"]]})
    if top_watch:
        blocks.append({"key": "notable_changes", "title": "Notable changes",
                       "bullets": [w["text"] for w in top_watch]})

    kpis = {}
    for key in ("leads", "ads", "sales"):
        if key in live:
            for name, value in live[key]["kpis"].items():
                kpis.setdefault(name, value)
    facts["display"] = {"summary_parts": parts}
    return result("all", facts=facts, kpis=kpis, watch_items=top_watch, cfg=cfg,
                  template={"summary": summary,
                            "top": (ads or {}).get("template", {}).get("top", ""),
                            "watch": " ".join(w["text"] for w in top_watch[:2]),
                            "blocks": blocks},
                  extras={"score_card": (sales or {}).get("score_card")})
