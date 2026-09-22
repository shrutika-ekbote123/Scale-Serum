"""Sales Team tab and the Consolidated Score card.

Most days a rep logs 0-2 calls, so the score and the per-rep table use a
rolling window (7 days by default) and the card also shows yesterday's calls.

  score        mean of the Sales Call Analyzer's overall_100 over the window's
               analysed calls (a call-weighted average of the reps' scores).
               Falls back to sales_calls.score for a call with no analysis.
  closures     a payment is credited to the LAST rep who called that lead in
               the 30 days before it was paid. A payment with no such call is
               "unassigned" (self-serve), never given to anyone.
  good / watch per rep: the analyzer's own strengths and weaknesses for that
               rep's calls - selected here, not re-written.

Rows written by the seeding scripts (config demo_data) are excluded.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from statistics import mean
from typing import Optional

from .. import fmt
from .common import DayContext, join_parts, kpi, result, unavailable, watch, watch_sentence, within
from .money import payments_in


def _call_score(call: dict, analyses: dict) -> Optional[float]:
    report = analyses.get(call["analysis_id"]) if call.get("analysis_id") else None
    scores = (report or {}).get("scores") or {}
    if scores.get("overall_100") is not None:
        return float(scores["overall_100"])
    return call.get("score")


def _rep_key(call: dict) -> str:
    return call["rep_user_id"] or (call["rep_name"] or "Unknown rep").strip().lower()


def credit_closures(payments: list[dict], calls: list[dict], days: int) -> list[dict]:
    """Each payment with the rep credited for it, or rep None (unassigned)."""
    by_lead = defaultdict(list)
    for c in calls:
        if c["lead_id"] and c["occurred_at"]:
            by_lead[c["lead_id"]].append(c)
    out = []
    for p in payments:
        credited = None
        if p["lead_id"]:
            eligible = [c for c in by_lead.get(p["lead_id"], [])
                        if p["occurred_at"] - timedelta(days=days) <= c["occurred_at"] <= p["occurred_at"]]
            if eligible:
                credited = max(eligible, key=lambda c: c["occurred_at"])
        out.append({**p, "rep_key": _rep_key(credited) if credited else None,
                    "rep_name": credited["rep_name"] if credited else None})
    return out


def missed_callbacks(calls: list[dict], window_start, until, grace: timedelta) -> list[dict]:
    """Callbacks promised for [window_start, until - grace) with no later call to
    the same lead."""
    by_lead = defaultdict(list)
    for c in calls:
        if c["lead_id"]:
            by_lead[c["lead_id"]].append(c["occurred_at"])
    missed = []
    for c in calls:
        cb = c.get("callback_at")
        if not cb or not c["lead_id"] or not (window_start <= cb < until - grace):
            continue
        if not any(t >= cb - grace for t in by_lead[c["lead_id"]] if t > c["occurred_at"]):
            missed.append(c)
    return missed


def build(raw: dict, ctx: DayContext) -> dict:
    cfg, scfg, cur = ctx.cfg, ctx.cfg["sales"], ctx.currency
    demo = {n.lower() for n in cfg["demo_data"]["sales_call_rep_names"]}
    calls = [c for c in raw["calls"]
             if c["occurred_at"] and (c["rep_name"] or "").lower() not in demo]
    if not calls:
        return unavailable("sales", "no_sales_calls",
                           "No sales calls have been logged for this brand recently.")

    days = scfg["window_days"]
    week, prev_week = ctx.last_days(days), ctx.last_days(days, ending_days_ago=days)
    w_calls = [c for c in calls if within(c["occurred_at"], week)]
    p_calls = [c for c in calls if within(c["occurred_at"], prev_week)]
    d_calls = [c for c in calls if within(c["occurred_at"], ctx.d)]

    def scores(cs):
        return [s for s in (_call_score(c, ctx.analyses) for c in cs) if s is not None]

    w_scores, p_scores = scores(w_calls), scores(p_calls)
    score = round(mean(w_scores)) if w_scores else None
    prev_score = round(mean(p_scores)) if p_scores else None
    trend = score - prev_score if score is not None and prev_score is not None else None

    credited = credit_closures(payments_in(raw["payments"], week), calls,
                               scfg["closure_credit_days"])
    closed = [p for p in credited if p["rep_key"]]
    unassigned = [p for p in credited if not p["rep_key"]]
    closures_rev = float(sum(p["value"] for p in closed))
    conversion = fmt.safe_div(len(closed), len(w_calls))
    sqls = sum(1 for c in w_calls if c["disposition"] == "SQL")
    until = min(ctx.d.end, ctx.now)
    missed = missed_callbacks(calls, week.start, until,
                              timedelta(hours=scfg["callback_grace_hours"]))

    # ---- per rep
    reps = {}
    for c in calls:
        key = _rep_key(c)
        rep = reps.setdefault(key, {"key": key, "name": c["rep_name"] or "Unknown rep",
                                    "user_id": c["rep_user_id"], "calls": [], "prev": []})
        if within(c["occurred_at"], week):
            rep["calls"].append(c)
        elif within(c["occurred_at"], prev_week):
            rep["prev"].append(c)
    rows = []
    for rep in reps.values():
        if not rep["calls"]:
            continue
        cs = sorted(rep["calls"], key=lambda c: c["occurred_at"], reverse=True)
        sc, pc = scores(cs), scores(rep["prev"])
        avg = round(mean(sc)) if sc else None
        prev_avg = round(mean(pc)) if pc else None
        reports = [ctx.analyses[c["analysis_id"]] for c in cs
                   if c.get("analysis_id") in ctx.analyses]
        good = next((s.get("text") for r in reports for s in (r.get("strengths") or [])
                     if s.get("text")), None)
        weak_stages = Counter(w.get("stage_id") for r in reports
                              for w in (r.get("weaknesses") or []) if w.get("stage_id"))
        watch_text = None
        if weak_stages:
            stage = weak_stages.most_common(1)[0][0]
            watch_text = next((w.get("text") for r in reports for w in (r.get("weaknesses") or [])
                               if w.get("stage_id") == stage and w.get("text")), None)
        if good is None and avg is not None and prev_avg is not None and avg - prev_avg >= 5:
            good = f"Score up {avg - prev_avg} points on the week before"
        rep_missed = [m for m in missed if _rep_key(m) == rep["key"]]
        if rep_missed:
            extra = f"missed {len(rep_missed)} callback{'s' if len(rep_missed) > 1 else ''}"
            watch_text = f"{watch_text}; {extra}" if watch_text else extra.capitalize()
        rep_closed = [p for p in closed if p["rep_key"] == rep["key"]]
        rows.append({
            "rep": rep["name"], "user_id": rep["user_id"],
            "calls_yesterday": sum(1 for c in cs if within(c["occurred_at"], ctx.d)),
            "calls": len(cs), "analysed_calls": len(sc),
            "score": avg, "score_previous": prev_avg,
            "score_change": fmt.delta_count(avg, prev_avg),
            "closures": len(rep_closed),
            "closures_revenue": fmt.money(sum(p["value"] for p in rep_closed), cur),
            "sqls": sum(1 for c in cs if c["disposition"] == "SQL"),
            "missed_callbacks": len(rep_missed),
            "good": good, "watch": watch_text,
        })
    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0), -r["calls"]))
    ranked = [r for r in rows if r["analysed_calls"] >= scfg["min_calls_for_ranking"]
              and r["score"] is not None]
    top = ranked[0] if ranked else None
    coach = [r for r in rows if r["watch"]]
    coach.sort(key=lambda r: (r["score"] if r["score"] is not None else 101))

    items = []
    if missed:
        names = ", ".join(sorted({m["rep_name"] or "Unknown rep" for m in missed}))
        items.append(watch(cfg, "missed_callbacks", "medium", count=len(missed), reps=names))
    if trend is not None and trend <= -scfg["score_drop_points"]:
        items.append(watch(cfg, "score_drop", "medium", points=abs(trend), score=score))
    new_leads = sum(1 for l in raw["leads"] if within(l["created_at"], week))
    if not w_calls and new_leads:
        items.append(watch(cfg, "no_calls", "high", days=days, leads=f"{new_leads:,}"))

    d = {
        "calls_yesterday": f"{len(d_calls):,}", "calls_window": f"{len(w_calls):,}",
        "window_days": days, "score": f"{score}/100" if score is not None else None,
        "trend": (f"{fmt.UP if trend > 0 else fmt.DOWN}{abs(trend)} pts WoW" if trend
                  else ("no change WoW" if trend == 0 else None)),
        "closures": f"{len(closed):,}", "closures_revenue": fmt.money(closures_rev, cur),
        "unassigned_payments": f"{len(unassigned):,}",
        "conversion": fmt.pct(conversion), "sqls": f"{sqls:,}",
    }
    score_card = {
        "score": score, "score_max": 100, "score_previous": prev_score, "trend_points": trend,
        "trend_display": d["trend"], "window_days": days,
        "calls_yesterday": len(d_calls), "calls_window": len(w_calls),
        "analysed_calls_window": len(w_scores),
        "closures": {"count": len(closed), "revenue": closures_rev,
                     "display": f"{len(closed)} · {d['closures_revenue']}"},
        "unassigned_payments": {"count": len(unassigned),
                                "revenue": float(sum(p["value"] for p in unassigned))},
        "conversion": conversion, "conversion_display": d["conversion"],
        "basis": {"score": f"mean analysed call score over {days} days",
                  "closures": f"payments credited to the last rep who called the lead "
                              f"within {scfg['closure_credit_days']} days before paying",
                  "conversion": f"credited closures / calls over {days} days"},
    }
    facts = {"window_days": days, **{k: v for k, v in score_card.items() if k != "basis"},
             "sqls_window": sqls, "missed_callbacks": len(missed),
             "top_rep": {k: top[k] for k in ("rep", "score", "closures", "closures_revenue")}
             if top else None,
             "coach": [{"rep": r["rep"], "score": r["score"], "watch": r["watch"]}
                       for r in coach[:3]],
             "display": d}
    kpis = {
        "score": kpi(score, d["score"], prev_score, d["trend"]),
        "calls_yesterday": kpi(len(d_calls), d["calls_yesterday"]),
        "calls_window": kpi(len(w_calls), d["calls_window"], window_days=days),
        "closures": kpi(len(closed), score_card["closures"]["display"]),
        "conversion": kpi(conversion, d["conversion"]),
    }

    summary = join_parts([
        f"{d['calls_window']} calls in {days} days ({d['calls_yesterday']} yesterday)",
        f"{d['closures']} closures ({d['closures_revenue']})" if closed else "no credited closures",
        f"{d['conversion']} conversion" if conversion is not None else None,
        f"avg score {d['score']}" + (f" ({d['trend']})" if d["trend"] else "")
        if score is not None else None,
    ]) + "."
    top_text = (f"{top['rep']} ({top['score']}"
                + (f", {top['closures_revenue']}" if top["closures"] else "") + ")") if top else ""
    coach_text = "; ".join(f"{r['rep']}: {r['watch']}" for r in coach[:2])
    public_summary = summary     # no rep names: what a rep without team view sees
    blocks = [{"key": "team", "title": "Team performance", "bullets": [b for b in [
        f"{d['calls_window']} calls over {days} days, {d['sqls']} marked SQL.",
        f"Average call score {d['score']}" + (f" ({d['trend']})." if d["trend"] else ".")
        if score is not None else None,
        f"{d['closures']} payments credited to reps ({d['closures_revenue']}); "
        f"{d['unassigned_payments']} came in with no call in the {scfg['closure_credit_days']} "
        f"days before.",
    ] if b]}]
    if rows:
        blocks.append({"key": "reps", "title": "Per rep", "bullets": [
            join_parts([f"{r['rep']}: {r['calls']} calls",
                        f"score {r['score']}" if r["score"] is not None else None,
                        f"good - {r['good']}" if r["good"] else None,
                        f"watch - {r['watch']}" if r["watch"] else None], "; ") + "."
            for r in rows[:5]]})

    out = result("sales", facts=facts, kpis=kpis, watch_items=items, cfg=cfg,
                 template={"summary": summary, "top": top_text, "watch": coach_text,
                           "blocks": blocks},
                 extras={"score_card": score_card, "reps": rows,
                         "public_summary": public_summary})
    if not coach_text:
        out["template"]["watch"] = watch_sentence(out["watch"])
    return out
