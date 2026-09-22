"""Ads & Marketing tab.

Spend, leads and CPL come from the ad platforms at CAMPAIGN level only (the
same spend is also stored per ad set and per ad; adding levels double counts).

Two different ROAS figures, never mixed:
  * blended ROAS  = payments actually received on D / total ad spend on D.
                    The headline number. Payments carry no campaign, so this
                    is the only ROAS that uses real money.
  * platform ROAS = the revenue Meta itself attributes to a campaign / its
                    spend. Labelled "Meta-reported" wherever it is shown, and
                    only computed for campaigns that report revenue at all -
                    a lead-gen campaign has no ROAS, not a ROAS of 0x.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from .. import fmt
from .common import DayContext, join_parts, kpi, result, unavailable, watch, watch_sentence
from .money import revenue

PLATFORM_LABEL = {"meta": "Meta", "google": "Google", "linkedin": "LinkedIn"}


def _sum(rows, key):
    return float(sum(r[key] for r in rows))


def build(raw: dict, ctx: DayContext) -> dict:
    cfg, acfg, cur = ctx.cfg, ctx.cfg["ads"], ctx.currency
    day = ctx.day
    connected = [f for f in raw["freshness"] if f["accounts"] > 0]
    insights = raw["insights"]
    if not connected and not insights:
        return unavailable("ads", "no_ad_accounts",
                           "No ad accounts are connected for this brand.")

    today = [r for r in insights if r["date"] == day]
    prev = [r for r in insights if r["date"] == day - timedelta(days=1)]
    trail = [r for r in insights if day - timedelta(days=7) <= r["date"] <= day - timedelta(days=1)]
    week = [r for r in insights if day - timedelta(days=6) <= r["date"] <= day]

    spend, spend_prev = _sum(today, "spend"), _sum(prev, "spend")
    leads, leads_prev = _sum(today, "leads"), _sum(prev, "leads")
    paid = revenue(raw["payments"], ctx.d)
    paid_7d = revenue(raw["payments"], ctx.last_days(7))
    blended = fmt.safe_div(paid, spend)
    blended_7d = fmt.safe_div(paid_7d, _sum(week, "spend"))
    cpl = fmt.safe_div(spend, leads)
    cpl_trail = fmt.safe_div(_sum(trail, "spend"), _sum(trail, "leads"))

    by_platform = []
    for platform in ("meta", "google", "linkedin"):
        rows = [r for r in today if r["platform"] == platform]
        if not rows:
            continue
        p_spend = _sum(rows, "spend")
        p_prev = _sum([r for r in prev if r["platform"] == platform], "spend")
        by_platform.append({
            "platform": platform, "label": PLATFORM_LABEL[platform],
            "spend": p_spend, "spend_display": fmt.money(p_spend, cur),
            "spend_change": fmt.delta_pct(p_spend, p_prev),
            "leads": _sum(rows, "leads"), "clicks": _sum(rows, "clicks"),
            "platform_revenue": _sum(rows, "revenue"),
            "platform_roas": fmt.ratio(fmt.safe_div(_sum(rows, "revenue"), p_spend))
            if _sum(rows, "revenue") else None,
        })

    # ---- per campaign
    series = defaultdict(list)
    for r in insights:
        series[(r["platform"], r["campaign_id"])].append(r)
    campaigns = []
    for (platform, cid), rows in series.items():
        t = [r for r in rows if r["date"] == day]
        if not t:
            continue
        tr = [r for r in rows if day - timedelta(days=7) <= r["date"] <= day - timedelta(days=1)]
        wk = [r for r in rows if day - timedelta(days=6) <= r["date"] <= day]
        meta = raw["campaigns"].get(f"{platform}:{cid}", {})
        c_spend, c_leads, c_rev = _sum(t, "spend"), _sum(t, "leads"), _sum(t, "revenue")
        # Webinar purchases land days after the click, so one day's platform ROAS
        # swings between 0x and 9x. ROAS is judged over 7 days; spend and CPL daily.
        reports_revenue = any(r["revenue"] > 0 for r in rows)
        campaigns.append({
            "platform": platform, "campaign_id": cid,
            "name": fmt.campaign_name(meta.get("name")), "status": meta.get("status"),
            "spend": c_spend, "leads": c_leads, "purchases": _sum(t, "purchases"),
            "platform_revenue": c_rev, "spend_7d": _sum(wk, "spend"),
            "platform_roas": fmt.safe_div(_sum(wk, "revenue"), _sum(wk, "spend"))
            if reports_revenue else None,
            "cpl": fmt.safe_div(c_spend, c_leads),
            "cpl_7d": fmt.safe_div(_sum(tr, "spend"), _sum(tr, "leads")),
            "had_leads_7d": _sum(tr, "leads") > 0,
            "daily_budget": meta.get("daily_budget"), "budget_level": meta.get("budget_level"),
        })
    campaigns.sort(key=lambda c: c["spend"], reverse=True)
    material = [c for c in campaigns if c["spend"] >= acfg["min_campaign_spend"]]

    top = None
    with_roas = [c for c in material if c["platform_roas"]]
    if with_roas:
        best = max(with_roas, key=lambda c: c["platform_roas"])
        top = {"metric": "platform_roas", "name": best["name"], "platform": best["platform"],
               "value": best["platform_roas"], "display": fmt.ratio(best["platform_roas"]),
               "basis": f"7-day {PLATFORM_LABEL[best['platform']]}-reported ROAS",
               "spend": fmt.money(best["spend"], cur)}
    else:
        with_cpl = [c for c in material if c["leads"] >= acfg["cpl_min_leads"]]
        if with_cpl:
            best = min(with_cpl, key=lambda c: c["cpl"])
            top = {"metric": "cpl", "name": best["name"], "platform": best["platform"],
                   "value": best["cpl"], "display": fmt.money(best["cpl"], cur),
                   "basis": "lowest cost per lead", "spend": fmt.money(best["spend"], cur)}

    # ---- watch
    items = []
    for c in material:
        if c["platform_roas"] is not None and c["platform_roas"] < acfg["roas_threshold"]:
            items.append(watch(cfg, "roas_below_threshold", "high", c["name"],
                               campaign=c["name"], roas=fmt.ratio(c["platform_roas"]),
                               spend=fmt.money(c["spend_7d"], cur),
                               threshold=fmt.ratio(acfg["roas_threshold"])))
        if (c["leads"] >= acfg["cpl_min_leads"] and c["cpl_7d"]
                and c["cpl"] >= acfg["cpl_spike_ratio"] * c["cpl_7d"]):
            items.append(watch(cfg, "cpl_spike", "medium", c["name"], campaign=c["name"],
                               cpl=fmt.money(c["cpl"], cur),
                               change=fmt.delta_pct(c["cpl"], c["cpl_7d"]),
                               cpl_avg=fmt.money(c["cpl_7d"], cur)))
        if c["leads"] == 0 and c["had_leads_7d"]:
            items.append(watch(cfg, "spend_no_leads", "medium", c["name"], campaign=c["name"],
                               spend=fmt.money(c["spend"], cur)))
        if c["daily_budget"] and c["spend"] > acfg["overspend_ratio"] * c["daily_budget"]:
            items.append(watch(cfg, "overspend", "medium", c["name"], campaign=c["name"],
                               spend=fmt.money(c["spend"], cur),
                               budget=fmt.money(c["daily_budget"], cur)))
    stale = []
    for f in connected:
        if f["last_date"] is not None and f["last_date"] < day:
            label = PLATFORM_LABEL[f["platform"]]
            stale.append(label)
            items.append(watch(cfg, "source_stale", "low", label, platform=label,
                               last_date=f"{f['last_date'].day} {f['last_date'].strftime('%b')}"))
    # Several campaigns can trip the same rule; keep the costliest few of each.
    per_type = defaultdict(int)
    kept = []
    for w in items:
        per_type[w["type"]] += 1
        if per_type[w["type"]] <= acfg["top_campaigns"]:
            kept.append(w)

    # ---- creatives
    verdicts = defaultdict(list)
    for c in raw["creatives"]:
        verdicts[c["verdict"]].append(c)
    creatives = {
        "analysed": len(raw["creatives"]),
        "scale": len(verdicts.get("scale", [])), "watch": len(verdicts.get("watch", [])),
        "pause": len(verdicts.get("pause", [])),
        "scale_examples": [c["ad_name"] for c in sorted(verdicts.get("scale", []),
                           key=lambda c: -(c["score"] or 0))[:2]],
    }

    d = {
        "spend": fmt.money(spend, cur), "spend_change": fmt.delta_pct(spend, spend_prev),
        "payments_revenue": fmt.money(paid, cur), "blended_roas": fmt.ratio(blended),
        "blended_roas_7d": fmt.ratio(blended_7d), "platform_leads": f"{leads:,.0f}",
        "platform_leads_change": fmt.delta_count(leads, leads_prev),
        "cpl": fmt.money(cpl, cur), "cpl_7d": fmt.money(cpl_trail, cur),
        "cpl_change": fmt.delta_pct(cpl, cpl_trail),
    }
    facts = {
        "window": "yesterday", "spend": spend, "spend_previous_day": spend_prev,
        "payments_revenue": paid, "blended_roas": blended, "blended_roas_7d": blended_7d,
        "platform_leads": leads, "cpl": cpl, "cpl_7d_average": cpl_trail,
        "by_platform": by_platform, "top_campaign": top, "creatives": creatives,
        "stale_sources": stale, "roas_threshold": fmt.ratio(acfg["roas_threshold"]),
        "campaigns_with_spend": len([c for c in campaigns if c["spend"] > 0]),
        "campaigns": [{k: c[k] for k in ("name", "platform", "spend", "spend_7d", "leads",
                                         "cpl", "platform_roas")} for c in campaigns[:8]],
        "platform_roas_basis": "7-day, as reported by the ad platform",
        "display": d,
    }
    kpis = {
        "spend": kpi(spend, d["spend"], spend_prev, d["spend_change"]),
        "revenue": kpi(paid, d["payments_revenue"], basis="payments received"),
        "blended_roas": kpi(blended, d["blended_roas"]),
        "leads": kpi(leads, d["platform_leads"], leads_prev, d["platform_leads_change"],
                     basis="ad platform leads"),
        "cpl": kpi(cpl, d["cpl"], cpl_trail, d["cpl_change"]),
    }

    # ---- template wording
    if spend:
        platforms = " and ".join(p["label"] for p in by_platform)
        summary = join_parts([
            f"{d['spend']} spend across {platforms}"
            + (f" ({d['spend_change']} on the day before)" if d["spend_change"] else ""),
            f"blended ROAS {d['blended_roas']} on {d['payments_revenue']} of payments"
            if blended is not None else None,
            f"{d['platform_leads']} leads at {d['cpl']} each" if cpl else None,
        ]) + "."
        if creatives["analysed"]:
            summary += (f" Creative analysis: {creatives['scale']} flagged Scale, "
                        f"{creatives['pause']} Pause.")
    elif stale:
        summary = ("No ad spend was recorded yesterday: "
                   + " ".join(w["text"] for w in items if w["type"] == "source_stale"))
    else:
        summary = "No ad spend was recorded yesterday."
    top_text = (f"{top['name']} ({top['display']} {top['basis']})" if top else "")
    blocks = [{"key": "ad_performance", "title": "Ad performance", "bullets": [
        b for b in [
            f"Spend {d['spend']}" + (f", {d['spend_change']} on the day before" if d["spend_change"] else "")
            + "; " + ", ".join(f"{p['label']} {p['spend_display']}" for p in by_platform) + "."
            if by_platform else None,
            f"Blended ROAS {d['blended_roas']} ({d['payments_revenue']} received); "
            f"{d['blended_roas_7d']} over 7 days." if blended is not None else None,
            f"{d['platform_leads']} platform leads at {d['cpl']} each"
            + (f" ({d['cpl_change']} vs the 7-day average)." if d["cpl_change"] else ".")
            if cpl else None,
            f"Top: {top_text}." if top else None,
        ] if b]}]
    if creatives["analysed"]:
        blocks.append({"key": "creatives", "title": "Creative analysis", "bullets": [
            f"{creatives['analysed']} ads analysed: {creatives['scale']} Scale, "
            f"{creatives['watch']} Watch, {creatives['pause']} Pause."]
            + ([f"Scale candidates: {', '.join(creatives['scale_examples'])}."]
               if creatives["scale_examples"] else [])})
    if kept:
        blocks.append({"key": "notable_changes", "title": "Notable changes",
                       "bullets": [w["text"] for w in kept[:4]]})

    out = result("ads", facts=facts, kpis=kpis, watch_items=kept, cfg=cfg,
                 template={"summary": summary, "top": top_text, "watch": "", "blocks": blocks})
    out["template"]["watch"] = watch_sentence(out["watch"])
    return out
