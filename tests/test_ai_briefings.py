"""AI Briefings: timezones, formatting, the four section builders, All, the
writer, access control, generation, the read views and the API.

No network and no database. Days are hand-built, modelled on DI's real data
(Meta campaigns named "_Paused" while ACTIVE, payments with no campaign, a
handful of logged calls); Gemini is a fake; MongoDB is the in-memory
FakeCollection.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.pop("MONGODB_URI", None)
os.environ["API_KEY"] = "test-api-key"

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
import ai_briefings as ab  # noqa: E402
from ai_briefings import access, fmt, timezones as tzs, views, writer  # noqa: E402
from ai_briefings.data import DataUnavailable  # noqa: E402
from ai_briefings.sections import BUILDERS, DayContext, overall  # noqa: E402
from ai_briefings.sections import sales as sales_mod  # noqa: E402
from test_sales_call_pipeline import FakeClient, FakeCollection, FakeResponse  # noqa: E402

HEADERS = {"X-API-Key": "test-api-key"}
BRAND = "bee79bff-d5f6-4220-a7d4-7a04bf173e59"
CFG = ab.load_config()
IST = tzs.resolve("Asia/Kolkata")
DAY = date(2026, 9, 21)
NOW = datetime(2026, 9, 22, 2, 0, tzinfo=timezone.utc)      # 07:30 IST on the 22nd


def run(coro):
    return asyncio.run(coro)


def at(day: date, hour: int = 12, zone=IST) -> datetime:
    """A moment on a local day, as the UTC timestamp scrumdb would return."""
    return datetime(day.year, day.month, day.day, hour, tzinfo=zone.tz).astimezone(timezone.utc)


# --------------------------------------------------------------------- fixtures
def insight(cid, day, spend, leads=0, revenue=0.0, platform="meta"):
    return {"platform": platform, "campaign_id": cid, "date": day, "spend": float(spend),
            "impressions": 1000.0, "clicks": 50.0, "leads": float(leads),
            "purchases": 1.0 if revenue else 0.0, "revenue": float(revenue),
            "conversions": 0.0}


def call(rep, day, hour=12, lead="lead-1", disposition="MQL", analysis_id=None, score=None,
         callback_at=None, user_id=None):
    return {"id": f"call-{rep}-{day}-{hour}-{lead}", "lead_id": lead, "rep_name": rep,
            "rep_user_id": user_id, "occurred_at": at(day, hour), "duration_seconds": 300.0,
            "disposition": disposition, "callback_at": callback_at,
            "analysis_id": analysis_id, "score": score}


def payment(lead, day, value, hour=15):
    return {"lead_id": lead, "occurred_at": at(day, hour), "value": float(value),
            "product_code": None}


def raw_day(**over):
    raw = {
        "insights": [], "campaigns": {},
        "freshness": [{"platform": "meta", "accounts": 1, "last_date": DAY, "last_synced_at": NOW},
                      {"platform": "google", "accounts": 0, "last_date": None, "last_synced_at": None},
                      {"platform": "linkedin", "accounts": 0, "last_date": None, "last_synced_at": None}],
        "creatives": [], "payments": [], "leads": [], "open_qualified": 0,
        "untouched": [], "calls": [], "whatsapp": {"accounts": 0},
    }
    raw.update(over)
    return raw


def ctx(**over):
    base = dict(day=DAY, zone=IST, cfg=CFG, currency="INR", now=NOW, analyses={},
                hot={"available": True, "reason": None, "scored": 0, "hot": []})
    base.update(over)
    return DayContext(**base)


def leads_on(day, n, funnel=None, prefix="l"):
    return [{"id": f"{prefix}-{day}-{i}", "created_at": at(day, 10), "funnel": funnel,
             "status": "new"} for i in range(n)]


# ------------------------------------------------------------------- timezones
@pytest.mark.parametrize("raw,name,source", [
    ("Asia/Kolkata", "Asia/Kolkata", "iana"),
    ("IST – UTC+5:30 (India Standard Time)", "UTC+05:30", "offset"),
    ("GMT – UTC+0 (Greenwich Mean Time)", "UTC", "offset"),
    ("UTC", "UTC", "iana"),
    ("IST", "Asia/Kolkata", "abbreviation"),
    ("", "Asia/Kolkata", "default"),
    ("not a zone", "Asia/Kolkata", "default"),
])
def test_timezone_every_shape_scrumdb_holds(raw, name, source):
    zone = tzs.resolve(raw)
    assert (zone.name, zone.source) == (name, source)


def test_ist_day_window_starts_at_1830_utc_the_day_before():
    w = tzs.day_window(DAY, IST)
    assert w.start == datetime(2026, 9, 20, 18, 30, tzinfo=timezone.utc)
    assert w.end - w.start == timedelta(days=1)


def test_yesterday_is_brand_local():
    # 20:00 UTC on the 21st is already the 22nd in India.
    now = datetime(2026, 9, 21, 20, 0, tzinfo=timezone.utc)
    assert tzs.yesterday(now, IST) == date(2026, 9, 21)
    assert tzs.yesterday(now, tzs.resolve("UTC")) == date(2026, 9, 20)


def test_date_labels_match_the_prototype():
    assert tzs.date_label(date(2026, 6, 3)) == "Wednesday, 3 June 2026"
    assert tzs.short_label(date(2026, 6, 2)) == "Tue, 2 Jun"


# ---------------------------------------------------------------------- format
@pytest.mark.parametrize("amount,text", [
    (950, "₹950"), (90300, "₹90.3K"), (190000, "₹1.9L"),
    (1420000, "₹14.2L"), (27000000, "₹2.7Cr"), (100000, "₹1L"),
])
def test_money_in_indian_units(amount, text):
    assert fmt.money(amount) == text


def test_deltas_and_ratios():
    assert fmt.delta_count(24, 18) == "▲6"
    assert fmt.delta_count(18, 24) == "▼6"
    assert fmt.delta_count(5, 5) == "no change"
    assert fmt.delta_count(5, None) is None
    assert fmt.ratio(5.94) == "5.9×"
    assert fmt.pct(0.128) == "12.8%"
    assert fmt.delta_pct(118, 100) == "+18%"
    assert fmt.delta_pct(1, 0) is None


def test_campaign_names_are_readable_but_not_renamed():
    assert fmt.campaign_name("FG_Cold_Webinar_ICDP") == "FG Cold Webinar ICDP"


# ------------------------------------------------------------------------- ads
def ads_raw(**over):
    week = [DAY - timedelta(days=i) for i in range(8)]
    insights = []
    for d in week:
        # A webinar campaign Meta attributes revenue to, doing badly this week.
        insights.append(insight("c1", d, 30000, leads=60, revenue=30000))
        # A lead-gen campaign with no revenue reporting at all: it has no ROAS.
        insights.append(insight("c2", d, 10000, leads=40))
    campaigns = {"meta:c1": {"name": "FG_Cold_Conv_Webinar_Paused", "status": "ACTIVE",
                             "daily_budget": 20000.0, "budget_level": "ad_set"},
                 "meta:c2": {"name": "FG_LeadGen", "status": "ACTIVE",
                             "daily_budget": None, "budget_level": None}}
    base = dict(insights=insights, campaigns=campaigns,
                payments=[payment("p1", DAY, 30000), payment("p2", DAY, 10000)])
    base.update(over)
    return raw_day(**base)


def test_ads_blended_roas_uses_payments_not_platform_revenue():
    out = BUILDERS["ads"](ads_raw(), ctx())
    f = out["facts"]
    assert f["spend"] == 40000 and f["payments_revenue"] == 40000
    assert f["display"]["blended_roas"] == "1.0×"
    assert out["kpis"]["revenue"]["basis"] == "payments received"


def test_ads_platform_roas_is_seven_day_and_only_where_reported():
    out = BUILDERS["ads"](ads_raw(), ctx())
    by_name = {c["name"]: c for c in out["facts"]["campaigns"]}
    assert by_name["FG Cold Conv Webinar Paused"]["platform_roas"] == pytest.approx(1.0)
    assert by_name["FG LeadGen"]["platform_roas"] is None      # no ROAS, not 0x
    kinds = [w["type"] for w in out["watch"]]
    assert kinds.count("roas_below_threshold") == 1
    assert "Meta-reported" in out["watch"][0]["text"]


def test_ads_overspend_against_ad_set_budget():
    out = BUILDERS["ads"](ads_raw(), ctx())
    over = [w for w in out["watch"] if w["type"] == "overspend"]
    assert over and "₹30K" in over[0]["text"] and "₹20K" in over[0]["text"]


def test_ads_spend_without_leads_and_stale_source():
    raw = ads_raw()
    raw["insights"] = [r for r in raw["insights"] if not (r["campaign_id"] == "c2" and r["date"] == DAY)]
    raw["insights"].append(insight("c2", DAY, 10000, leads=0))
    raw["freshness"][2] = {"platform": "linkedin", "accounts": 1,
                           "last_date": date(2026, 8, 19), "last_synced_at": NOW}
    out = BUILDERS["ads"](raw, ctx())
    kinds = {w["type"] for w in out["watch"]}
    assert {"spend_no_leads", "source_stale"} <= kinds
    assert "LinkedIn data has not updated since 19 Aug." in [w["text"] for w in out["watch"]]


def test_ads_counts_campaign_level_only_via_data_contract():
    # The builder sums what data.py hands it - campaign rows. Two campaigns: 40K.
    out = BUILDERS["ads"](ads_raw(), ctx())
    assert out["facts"]["by_platform"][0]["spend"] == 40000


def test_ads_unavailable_without_accounts():
    raw = raw_day(freshness=[{"platform": p, "accounts": 0, "last_date": None,
                              "last_synced_at": None} for p in ("meta", "google", "linkedin")])
    out = BUILDERS["ads"](raw, ctx())
    assert out["available"] is False and out["reason"] == "no_ad_accounts"


def test_ads_creatives_flagged():
    creatives = [{"ad_id": str(i), "ad_name": f"Ad {i}", "verdict": v, "score": 50.0,
                  "analyzed_at": NOW} for i, v in enumerate(["scale"] * 3 + ["pause"] * 4)]
    out = BUILDERS["ads"](ads_raw(creatives=creatives), ctx())
    assert "3 flagged Scale, 4 Pause" in out["template"]["summary"]


# ----------------------------------------------------------------------- leads
def test_leads_counted_by_brand_local_day():
    leads = leads_on(DAY, 24) + leads_on(DAY - timedelta(days=1), 18) \
        + leads_on(DAY - timedelta(days=7), 30)
    # 23:00 IST on the 20th is 17:30 UTC on the 20th: it belongs to the 20th.
    leads.append({"id": "late", "created_at": at(DAY - timedelta(days=1), 23), "funnel": None,
                  "status": "new"})
    out = BUILDERS["leads"](raw_day(leads=leads), ctx())
    assert out["facts"]["new_leads"] == 24
    assert out["facts"]["new_leads_previous_day"] == 19
    assert out["facts"]["display"]["new_leads_change"] == "▲5"


def test_leads_first_time_buyers_vs_repeat_payments():
    payments = [payment("a", DAY - timedelta(days=5), 299), payment("a", DAY, 30000),
                payment("b", DAY, 299)]
    out = BUILDERS["leads"](raw_day(payments=payments, leads=leads_on(DAY, 3)), ctx())
    assert out["facts"]["buyers"] == 1 and out["facts"]["payments"] == 2
    assert out["facts"]["revenue"] == 30299


def test_leads_hot_untouched_is_a_high_watch_item():
    hot = {"available": True, "reason": None, "scored": 40,
           "hot": [{"lead_id": f"h{i}", "probability": 0.05, "priority": "High"} for i in range(8)]}
    out = BUILDERS["leads"](raw_day(leads=leads_on(DAY, 5)), ctx(hot=hot))
    assert out["watch"][0]["type"] == "hot_untouched" and out["watch"][0]["severity"] == "high"
    assert out["facts"]["hot_untouched"]["count"] == 8
    assert "8 hot leads untouched" in out["template"]["summary"]


def test_leads_best_funnel_needs_volume():
    leads = leads_on(DAY - timedelta(days=3), 40, funnel="Masterclass", prefix="m") \
        + leads_on(DAY - timedelta(days=3), 10, funnel="Tiny", prefix="t")
    payments = [payment(f"m-{DAY - timedelta(days=3)}-{i}", DAY, 299) for i in range(7)] \
        + [payment(f"t-{DAY - timedelta(days=3)}-{i}", DAY, 299) for i in range(9)]
    out = BUILDERS["leads"](raw_day(leads=leads, payments=payments), ctx())
    assert out["facts"]["best_funnel"]["funnel"] == "Masterclass"
    assert out["facts"]["best_funnel"]["rate_display"] == "17.5%"


def test_leads_qualified_from_call_dispositions():
    calls = [call("Aaditya WDC", DAY, lead="x", disposition="SQL"),
             call("Priya Sharma", DAY, lead="y", disposition="SQL")]      # demo rep
    out = BUILDERS["leads"](raw_day(leads=leads_on(DAY, 2), calls=calls), ctx())
    assert out["facts"]["sqls_from_calls"] == 1


# ----------------------------------------------------------------------- sales
ANALYSIS = {"scores": {"overall_100": 80, "stages": []},
            "strengths": [{"text": "Strong discovery", "stage_id": "probing"}],
            "weaknesses": [{"text": "Rushing objection-handling on pricing",
                            "stage_id": "objection_handling"}]}


def test_sales_demo_reps_excluded_and_unavailable_without_calls():
    raw = raw_day(calls=[call("Priya Sharma", DAY), call("Rajan Kumar", DAY)])
    out = BUILDERS["sales"](raw, ctx())
    assert out["available"] is False and out["reason"] == "no_sales_calls"


def test_sales_score_from_analyses_and_trend():
    calls = [call("Aaditya WDC", DAY, analysis_id="a1", user_id="u1"),
             call("Aaditya WDC", DAY - timedelta(days=1), lead="lead-2", score=70, user_id="u1"),
             call("Aaditya WDC", DAY - timedelta(days=9), lead="lead-3", score=65, user_id="u1")]
    out = BUILDERS["sales"](raw_day(calls=calls), ctx(analyses={"a1": ANALYSIS}))
    card = out["score_card"]
    assert card["score"] == 75 and card["score_previous"] == 65 and card["trend_points"] == 10
    assert card["calls_yesterday"] == 1 and card["calls_window"] == 2
    rep = out["reps"][0]
    assert rep["good"] == "Strong discovery"
    assert rep["watch"] == "Rushing objection-handling on pricing"


def test_sales_closure_credited_to_last_rep_within_30_days():
    calls = [call("Appus Bayar", DAY - timedelta(days=20), lead="buyer", user_id="u2"),
             call("Aaditya WDC", DAY - timedelta(days=2), lead="buyer", user_id="u1"),
             call("Aaditya WDC", DAY - timedelta(days=40), lead="old", user_id="u1")]
    payments = [payment("buyer", DAY, 30000), payment("old", DAY, 299),
                payment("walk-in", DAY, 299)]
    out = BUILDERS["sales"](raw_day(calls=calls, payments=payments), ctx())
    card = out["score_card"]
    assert card["closures"]["count"] == 1 and card["closures"]["revenue"] == 30000
    assert card["unassigned_payments"]["count"] == 2
    by_rep = {r["rep"]: r for r in out["reps"]}
    assert by_rep["Aaditya WDC"]["closures"] == 1


def test_credit_ignores_calls_after_the_payment():
    calls = [call("Aaditya WDC", DAY, hour=18, lead="buyer")]
    credited = sales_mod.credit_closures([payment("buyer", DAY, 100, hour=9)], calls, 30)
    assert credited[0]["rep_key"] is None


def test_sales_missed_callback():
    promised = at(DAY - timedelta(days=2), 11)
    calls = [call("Aniket Arora", DAY - timedelta(days=3), lead="cb", callback_at=promised,
                  disposition="callback", user_id="u3")]
    out = BUILDERS["sales"](raw_day(calls=calls), ctx())
    assert any(w["type"] == "missed_callbacks" for w in out["watch"])
    assert "missed 1 callback" in out["reps"][0]["watch"].lower()


def test_sales_callback_kept_is_not_missed():
    promised = at(DAY - timedelta(days=2), 11)
    calls = [call("Aniket Arora", DAY - timedelta(days=3), lead="cb", callback_at=promised),
             call("Aniket Arora", DAY - timedelta(days=2), hour=12, lead="cb")]
    out = BUILDERS["sales"](raw_day(calls=calls), ctx())
    assert not any(w["type"] == "missed_callbacks" for w in out["watch"])


def test_sales_no_calls_this_week_is_flagged():
    calls = [call("Aaditya WDC", DAY - timedelta(days=20))]
    out = BUILDERS["sales"](raw_day(calls=calls, leads=leads_on(DAY, 50)), ctx())
    assert out["watch"][0]["type"] == "no_calls" and out["watch"][0]["severity"] == "high"


def test_sales_summary_names_no_rep():
    calls = [call("Aaditya WDC", DAY, analysis_id="a1"), call("Aaditya WDC", DAY, hour=14,
                                                              lead="l2", analysis_id="a1")]
    out = BUILDERS["sales"](raw_day(calls=calls), ctx(analyses={"a1": ANALYSIS}))
    assert "Aaditya" not in out["template"]["summary"]
    assert "Aaditya" in out["template"]["top"]


# -------------------------------------------------------------------- whatsapp
def test_whatsapp_not_connected():
    out = BUILDERS["whatsapp"](raw_day(), ctx())
    assert out["available"] is False and out["reason"] == "whatsapp_not_connected"


def test_whatsapp_rates_and_replies_to_sql():
    wa = {"accounts": 1, "broadcasts": [{
        "id": "b1", "name": "Career Accelerator May Cohort", "status": "sent", "recipients": 842,
        "sent": 842, "delivered": 821, "read": 597, "replied": 38, "failed": 21,
        "estimated_cost": 700.0, "sent_at": at(DAY, 10)}],
        "inbound": [{"conversation_id": "c1", "lead_id": "wl", "at": at(DAY, 11)}],
        "awaiting_reply": 4,
        "templates": {"approved": {"count": 3, "approved_today": ["Re-engagement Campaign"]}}}
    calls = [call("Aaditya WDC", DAY, hour=15, lead="wl", disposition="SQL")]
    out = BUILDERS["whatsapp"](raw_day(whatsapp=wa, calls=calls), ctx())
    d = out["facts"]["display"]
    assert d["delivered_rate"] == "97.5%" and d["read_rate"] == "72.7%"
    assert out["facts"]["replies_to_sql"] == 1
    assert "Re-engagement Campaign" in out["template"]["summary"]
    assert {w["type"] for w in out["watch"]} == {"awaiting_reply", "broadcast_failed"}


# ------------------------------------------------------------------------- all
def full_day():
    hot = {"available": True, "reason": None, "scored": 10,
           "hot": [{"lead_id": "h1", "probability": 0.05, "priority": "High"}]}
    c = ctx(hot=hot, analyses={"a1": ANALYSIS})
    raw = ads_raw(leads=leads_on(DAY, 24) + leads_on(DAY - timedelta(days=1), 18),
                  calls=[call("Aaditya WDC", DAY, analysis_id="a1", user_id="u1"),
                         call("Aniket Arora", DAY, hour=15, lead="l9", score=60, user_id="u3")])
    sections = {k: f(raw, c) for k, f in BUILDERS.items()}
    return sections, overall.build(sections, c), c


def test_all_quotes_the_section_numbers():
    sections, out, _ = full_day()
    assert sections["ads"]["facts"]["display"]["blended_roas"] in out["template"]["summary"]
    assert sections["leads"]["facts"]["display"]["new_leads"] in out["template"]["summary"]
    assert out["score_card"] == sections["sales"]["score_card"]


def test_all_watch_takes_one_item_per_section_first():
    _, out, _ = full_day()
    sections_in_watch = [w["section"] for w in out["watch"]]
    assert "leads" in sections_in_watch and "ads" in sections_in_watch


def test_all_unavailable_when_nothing_is():
    empty = {k: {"available": False} for k in ("ads", "leads", "sales", "whatsapp")}
    assert overall.build(empty, ctx())["available"] is False


# ---------------------------------------------------------------------- writer
def ads_payload():
    out = BUILDERS["ads"](ads_raw(), ctx())
    return writer.build_payload(out, section_label="Ads & Marketing",
                                date_label=tzs.date_label(DAY), brand={"name": "DI"})


def llm(payload, **over):
    w = writer.template_wording(payload)
    w.update(over)
    return FakeResponse(json.dumps(w, ensure_ascii=False))


def test_writer_accepts_grounded_output():
    payload = ads_payload()
    out = run(writer.write(FakeClient([llm(payload, summary="Spend held at ₹40K.")]), "m", payload))
    assert out["source"] == "llm" and out["wording"]["summary"] == "Spend held at ₹40K."


def test_writer_rejects_invented_number_then_repairs():
    payload = ads_payload()
    client = FakeClient([llm(payload, summary="ROAS recovered to 6.1×."), llm(payload)])
    out = run(writer.write(client, "m", payload))
    assert out["source"] == "llm" and client.models.calls == 2


def test_writer_falls_back_to_template():
    payload = ads_payload()
    client = FakeClient([llm(payload, summary="Up 99%."), llm(payload, summary="Up 98%.")])
    out = run(writer.write(client, "m", payload))
    assert out["source"] == "template" and out["fallback_reason"] == "unsupported_number"
    assert out["wording"]["summary"] == payload["draft"]["summary"]


def test_writer_rejects_missing_block_and_llm_errors():
    payload = ads_payload()
    bad = llm(payload, blocks=[])
    out = run(writer.write(FakeClient([bad, bad]), "m", payload))
    assert out["fallback_reason"] == "block_keys_mismatch"
    out = run(writer.write(FakeClient([RuntimeError("down")]), "m", payload))
    assert out["source"] == "template" and out["fallback_reason"] == "llm_error"
    out = run(writer.write(None, "m", payload))
    assert out["source"] == "template" and out["fallback_reason"] == "llm_unavailable"


def test_writer_rejects_rep_named_in_sales_summary():
    sections, _, _ = full_day()
    payload = writer.build_payload(sections["sales"], section_label="Sales Team",
                                   date_label="x", brand={})
    named = llm(payload, summary="Aaditya WDC carried the day.")
    out = run(writer.write(FakeClient([named, named]), "m", payload,
                           forbid_in_summary=("Aaditya WDC",)))
    assert out["fallback_reason"] == "rep_named_in_summary"


def test_writer_rejects_invented_watch():
    sections, _, _ = full_day()
    payload = writer.build_payload(sections["whatsapp"], section_label="WhatsApp",
                                   date_label="x", brand={})
    payload["draft"]["watch"] = ""
    payload["watch_items"] = []
    bad = llm(payload, watch="Reply to the backlog.")
    out = run(writer.write(FakeClient([bad, bad]), "m", payload))
    assert out["fallback_reason"] == "invented_watch"


# ---------------------------------------------------------------------- access
def test_access_by_permission_not_role_name():
    service = access.viewer_from(None, None, None)
    assert service.service and service.full_access and service.team_view
    rep = access.viewer_from("u1", "user", "crm,sales_calls,briefings")
    assert rep.sections() == ["sales", "leads"] and not rep.team_view
    head = access.viewer_from("u2", "user", "briefings,sales_calls,users_roles")
    assert head.team_view
    nopage = access.viewer_from("u3", "user", "crm,sales_calls")
    assert not nopage.can_open_page and nopage.sections() == []
    admin = access.viewer_from("u4", "superAdmin", "")
    assert admin.full_access and admin.team_view


# ------------------------------------------------------------- service + views
class RunCollection(FakeCollection):
    pass


def make_store():
    return ab.BriefingStore(FakeCollection(), RunCollection())


def make_deps(store, raw=None, llm_client=None, load_day=None):
    async def run_sync(fn, *args):
        return fn(*args)

    async def analyses(ids):
        return {"a1": ANALYSIS}

    return ab.BriefingDeps(
        run_sync=run_sync, store=store, llm_client=llm_client, llm_model="test-model",
        load_brand=lambda b: {"id": b, "name": "DI", "timezone": "Asia/Kolkata",
                              "currency": "INR", "brand_brain_id": None},
        load_day=load_day or (lambda *a: raw),
        score_hot=lambda cands, pr: {"available": True, "reason": None, "scored": 3,
                                     "hot": [{"lead_id": "h1", "probability": 0.1,
                                              "priority": "High"}]},
        load_analyses=analyses, now=lambda: NOW)


def generated_store():
    store = make_store()
    raw = ads_raw(leads=leads_on(DAY, 24),
                  calls=[call("Aaditya WDC", DAY, analysis_id="a1", user_id="u1"),
                         call("Aniket Arora", DAY, hour=15, lead="l9", score=60, user_id="u3")])
    result = run(ab.generate_day(BRAND, make_deps(store, raw), day=DAY))
    return store, result


def test_generate_day_stores_five_briefings_and_a_run():
    store, result = generated_store()
    assert result["status"] == "completed"
    ids = set(store.briefings.docs)
    assert ids == {ab.briefing_id(BRAND, "2026-09-21", s) for s in ab.ALL_SECTIONS}
    whatsapp = store.briefings.docs[ab.briefing_id(BRAND, "2026-09-21", "whatsapp")]
    assert whatsapp["available"] is False and whatsapp["reason"] == "whatsapp_not_connected"
    sales = store.briefings.docs[ab.briefing_id(BRAND, "2026-09-21", "sales")]
    assert sales["summary"]["label"] == "Sales team" and sales["score_card"]["score"] is not None
    assert sales["fallback"] is True                  # no Gemini client: template wording
    run_doc = next(iter(store.runs.docs.values()))
    assert run_doc["status"] == "completed" and run_doc["sections"]["ads"]["wording"] == "template"


def test_generate_day_uses_the_model_when_it_behaves():
    store = make_store()
    raw = ads_raw(leads=leads_on(DAY, 24))
    # The fake echoes each section's draft back, which always validates.
    class Echo:
        def __init__(self):
            self.aio = self
            self.models = self
            self.calls = 0

        async def generate_content(self, *, model, contents, config):
            self.calls += 1
            payload = json.loads(contents)
            return FakeResponse(json.dumps(writer.template_wording(payload), ensure_ascii=False))

    client = Echo()
    run(ab.generate_day(BRAND, make_deps(store, raw, llm_client=client), day=DAY))
    ads = store.briefings.docs[ab.briefing_id(BRAND, "2026-09-21", "ads")]
    assert ads["wording"]["source"] == "llm" and ads["fallback"] is False
    assert client.calls == 3          # ads, leads, all - sales and whatsapp are unavailable


def test_generate_day_records_database_failure():
    store = make_store()

    def down(*a):
        raise DataUnavailable("ConnectionTimeout")

    result = run(ab.generate_day(BRAND, make_deps(store, load_day=down), day=DAY))
    assert result["status"] == "failed" and result["error"].startswith("database_unavailable")
    assert not store.briefings.docs


def test_views_today_for_service_and_stale_fallback():
    store, _ = generated_store()
    service = access.viewer_from(None, None, None)
    out = run(views.today(store, BRAND, "ads", service, None, NOW))
    assert out["date"] == "2026-09-21" and out["stale"] is False and out["available"]
    assert [t["section"] for t in out["tabs"]] == ["all", "sales", "ads", "whatsapp", "leads"]
    later = run(views.today(store, BRAND, "ads", service, "2026-09-25", NOW))
    assert later["stale"] is True and later["date"] == "2026-09-21"


def test_views_rep_without_team_view_sees_only_own_row():
    store, _ = generated_store()
    rep = access.viewer_from("u3", "user", "briefings,sales_calls")
    out = run(views.today(store, BRAND, "sales", rep, None, NOW))
    assert [r["rep"] for r in out["reps"]] == ["Aniket Arora"]
    assert out["summary"]["top"] == ""
    assert all(b["key"] != "reps" for b in out["blocks"])
    with pytest.raises(views.Forbidden):
        run(views.today(store, BRAND, "ads", rep, None, NOW))


def test_views_restricted_all_is_composed_from_allowed_sections():
    store, _ = generated_store()
    marketer = access.viewer_from("u9", "user", "briefings,meta_ads")
    out = run(views.today(store, BRAND, "all", marketer, None, NOW))
    assert out["facts"]["sections"] == ["ads"]
    assert out["summary"]["text"].startswith("Ads & Marketing:")
    with pytest.raises(views.Forbidden):
        run(views.one(store, ab.briefing_id(BRAND, "2026-09-21", "all"), marketer))


def test_views_history_pages_by_day():
    store = make_store()
    raw = ads_raw(leads=leads_on(DAY, 5))
    deps = make_deps(store, raw)
    for n in range(3):
        run(ab.generate_day(BRAND, deps, day=DAY - timedelta(days=n)))
    service = access.viewer_from(None, None, None)
    page = run(views.history(store, BRAND, "all", service, None, 2))
    assert sorted({i["date"] for i in page["items"]}) == ["2026-09-20", "2026-09-21"]
    assert page["next_before"] == "2026-09-20"
    rest = run(views.history(store, BRAND, "all", service, page["next_before"], 2))
    assert {i["date"] for i in rest["items"]} == {"2026-09-19"} and rest["next_before"] is None


def test_views_not_generated():
    out = run(views.today(make_store(), BRAND, "all", access.viewer_from(None, None, None),
                          None, NOW))
    assert out["available"] is False and out["reason"] == "not_generated"


# ------------------------------------------------------------------------- API
@pytest.fixture
def api(monkeypatch):
    store, _ = generated_store()
    monkeypatch.setattr(app_module, "ai_briefing_store", store)
    monkeypatch.setattr(app_module, "AI_BRIEFINGS_AVAILABLE", True)
    return TestClient(app_module.app), store


def test_api_today_and_score(api):
    c, _ = api
    r = c.get(f"/api/ai-briefings/{BRAND}/today?section=leads&date=2026-09-21", headers=HEADERS)
    assert r.status_code == 200 and r.json()["section"] == "leads"
    r = c.get(f"/api/ai-briefings/{BRAND}/score?date=2026-09-21", headers=HEADERS)
    assert r.status_code == 200 and r.json()["score_card"]["score_max"] == 100


def test_api_requires_key_and_validates(api):
    c, _ = api
    assert c.get(f"/api/ai-briefings/{BRAND}/today").status_code == 401
    assert c.get(f"/api/ai-briefings/{BRAND}/today?section=nope", headers=HEADERS).status_code == 422
    assert c.get(f"/api/ai-briefings/{BRAND}/today?date=21-09-2026", headers=HEADERS).status_code == 422
    assert c.get("/api/ai-briefings/not-a-uuid/today", headers=HEADERS).status_code == 422
    assert c.get("/api/ai-briefings/briefing/nope", headers=HEADERS).status_code == 404


def test_api_permissions(api):
    c, _ = api
    rep = {**HEADERS, "X-User-Id": "u3", "X-User-Role": "user",
           "X-User-Permissions": "briefings,sales_calls"}
    assert c.get(f"/api/ai-briefings/{BRAND}/today?section=ads", headers=rep).status_code == 403
    assert c.get(f"/api/ai-briefings/{BRAND}/today?section=sales", headers=rep).status_code == 200
    nopage = {**HEADERS, "X-User-Id": "u3", "X-User-Role": "user", "X-User-Permissions": "crm"}
    assert c.get(f"/api/ai-briefings/{BRAND}/today", headers=nopage).status_code == 403
    assert c.post(f"/api/ai-briefings/{BRAND}/generate", headers=rep).status_code == 403


def test_api_history_latest_and_one(api):
    c, _ = api
    h = c.get(f"/api/ai-briefings/{BRAND}/history", headers=HEADERS).json()
    assert h["items"] and h["items"][0]["section"] == "all"
    one = c.get(f"/api/ai-briefings/briefing/{h['items'][0]['briefing_id']}", headers=HEADERS)
    assert one.status_code == 200 and one.json()["blocks"]
    latest = c.get(f"/api/ai-briefings/{BRAND}/latest", headers=HEADERS).json()
    assert latest["available"] and latest["summary"]["label"].startswith("Across all sections")


def test_api_generate_rejects_future_date(api, monkeypatch):
    c, _ = api
    monkeypatch.setattr(app_module._ab_data, "load_brand",
                        lambda b: {"id": b, "name": "DI", "timezone": "Asia/Kolkata",
                                   "currency": "INR", "brand_brain_id": None})
    r = c.post(f"/api/ai-briefings/{BRAND}/generate?date=2099-01-01", headers=HEADERS)
    assert r.status_code == 422


def test_writer_keeps_the_draft_callout_the_model_dropped():
    payload = ads_payload()
    assert payload["draft"]["top"] and payload["draft"]["watch"]
    out = run(writer.write(FakeClient([llm(payload, top="", watch="")]), "m", payload))
    assert out["source"] == "llm"
    assert out["wording"]["top"] == payload["draft"]["top"]
    assert out["wording"]["watch"] == payload["draft"]["watch"]


def test_ads_stale_source_explains_missing_spend():
    raw = raw_day(freshness=[{"platform": "meta", "accounts": 1, "last_date": date(2026, 7, 8),
                              "last_synced_at": NOW}])
    out = BUILDERS["ads"](raw, ctx())
    assert out["template"]["summary"] == ("No ad spend was recorded yesterday: "
                                          "Meta data has not updated since 8 Jul.")


# ----------------------------------------------------------------- token usage
def priced_deps(store, raw, client):
    deps = make_deps(store, raw, llm_client=client)
    from ai_briefings.loaders import usage_pricer
    deps.price_usage = usage_pricer("gemini-flash-latest")
    return deps


class Meter:
    """A Gemini fake that echoes the draft and reports token usage."""

    def __init__(self, tokens=(1000, 200, 100)):
        self.aio = self
        self.models = self
        self.calls = 0
        self.tokens = tokens

    async def generate_content(self, *, model, contents, config):
        self.calls += 1
        payload = json.loads(contents)
        response = FakeResponse(json.dumps(writer.template_wording(payload), ensure_ascii=False))

        class Usage:
            prompt_token_count, candidates_token_count, thoughts_token_count = self.tokens
            cached_content_token_count = None

        response.usage_metadata = Usage()
        return response


def priced_store():
    store = make_store()
    raw = ads_raw(leads=leads_on(DAY, 24))
    client = Meter()
    run(ab.generate_day(BRAND, priced_deps(store, raw, client), day=DAY))
    return store, client


def test_briefing_records_tokens_and_cost():
    store, client = priced_store()
    ads = store.briefings.docs[ab.briefing_id(BRAND, "2026-09-21", "ads")]
    assert ads["usage"] == {"input_tokens": 1000, "output_tokens": 200,
                            "thinking_tokens": 100, "total_tokens": 1300}
    cost = ads["cost"]
    assert cost["usd"] == pytest.approx(0.001875)       # 1000 in + 300 out at flash rates
    assert cost["inr"] == pytest.approx(0.165)          # fixed 88/USD from pricing.json
    assert cost["model_priced"] and cost["estimated"] is True
    assert ads["cost_usd"] == cost["usd"]
    # An unavailable tab makes no call, so it costs nothing.
    whatsapp = store.briefings.docs[ab.briefing_id(BRAND, "2026-09-21", "whatsapp")]
    assert whatsapp["usage"] == {} and whatsapp["cost"] is None


def test_run_totals_add_up_across_sections():
    store, client = priced_store()
    doc = next(iter(store.runs.docs.values()))
    assert doc["usage"]["input_tokens"] == 1000 * client.calls
    assert doc["usage"]["total_tokens"] == 1300 * client.calls
    assert doc["cost"]["llm_calls"] == client.calls
    assert doc["cost"]["usd"] == pytest.approx(0.001875 * client.calls)


def test_usage_summary_totals_and_breakdowns():
    store, client = priced_store()
    docs = run(store.usage_between(BRAND, "2026-09-01", "2026-09-30"))
    import billing
    out = ab.summarize_usage(docs, brand_id=BRAND, start="2026-09-01", end="2026-09-30",
                             pricing=billing.load_pricing())
    assert out["totals"]["briefings"] == client.calls
    assert out["totals"]["total_tokens"] == 1300 * client.calls
    assert out["totals"]["cost_usd"] == pytest.approx(0.001875 * client.calls)
    assert out["days_with_briefings"] == 1
    assert out["per_day"][0]["date"] == "2026-09-21"
    assert {s["section"] for s in out["per_section"]} == {"ads", "leads", "all"}
    assert out["averages"]["per_briefing_usd"] == pytest.approx(0.001875)
    assert out["averages"]["projected_30_days_usd"] == pytest.approx(
        0.001875 * client.calls * 30, abs=1e-4)      # the projection is rounded to 4 dp
    assert out["per_model"][0]["model"] == "test-model"
    assert out["pricing"]["usd_to_inr"] == 88
    assert out["estimated"] is True
    assert "fx_fixed_rate" in out["notes"]


def test_usage_summary_is_empty_without_briefings():
    out = ab.summarize_usage([], brand_id=BRAND, start="2026-09-01", end="2026-09-30")
    assert out["totals"]["briefings"] == 0 and out["totals"]["cost_usd"] == 0.0
    assert out["averages"]["per_briefing_usd"] is None and out["per_day"] == []


def test_usage_summary_counts_unpriced_briefings():
    docs = [{"date": "2026-09-21", "section": "ads", "usage": {"input_tokens": 10},
             "cost": {"usd": None, "attempts": 1}, "wording": {"source": "llm"},
             "versions": {"llm_model": "some-new-model"}}]
    out = ab.summarize_usage(docs, brand_id=BRAND, start="2026-09-01", end="2026-09-30")
    assert out["unpriced_briefings"] == 1 and "unpriced_briefings" in out["notes"]
    assert out["totals"]["total_tokens"] == 10 and out["totals"]["cost_usd"] == 0.0


def test_api_usage_endpoint(api):
    c, _ = api
    r = c.get(f"/api/ai-briefings/{BRAND}/usage?from=2026-09-01&to=2026-09-30", headers=HEADERS)
    assert r.status_code == 200
    b = r.json()
    assert b["from"] == "2026-09-01" and b["to"] == "2026-09-30"
    assert set(b) >= {"totals", "per_day", "per_section", "per_model", "averages", "pricing",
                      "notes", "estimated"}
    assert c.get(f"/api/ai-briefings/{BRAND}/usage?from=2026-09-30&to=2026-09-01",
                 headers=HEADERS).status_code == 422
    assert c.get(f"/api/ai-briefings/{BRAND}/usage?from=nope", headers=HEADERS).status_code == 422


def test_api_today_exposes_tokens_and_cost(api):
    c, _ = api
    b = c.get(f"/api/ai-briefings/{BRAND}/today?section=ads&date=2026-09-21",
              headers=HEADERS).json()
    assert "usage" in b and "cost" in b
