"""AI Suggested Next Action: rules, tiers, wording, service and API.

No network and no database. The rules run on hand-built lead contexts modelled
on real Lawtorney leads (the two from the Lead Journey screenshots among them),
Gemini is a fake, and MongoDB is the in-memory FakeCollection.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.pop("MONGODB_URI", None)
os.environ["API_KEY"] = "test-api-key"

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
import ai_suggested_next_action as na  # noqa: E402
from ai_suggested_next_action import data as na_data  # noqa: E402
from ai_suggested_next_action import rules, writer  # noqa: E402
from test_sales_call_pipeline import FakeClient, FakeCollection, FakeResponse  # noqa: E402

HEADERS = {"X-API-Key": "test-api-key"}
NOW = datetime(2026, 9, 11, 7, 30, tzinfo=timezone.utc)
LEAD = "b510926c-84d5-4cd5-964d-55fd8b80857e"
BRAND = "bee79bff-d5f6-4220-a7d4-7a04bf173e59"
CFG = na.load_framework()

# Lawtorney's real price ladder, in miniature.
LAWTORNEY_PAYMENTS = [(149, 70, "AIFE149"), (299, 600, "CrossroadstoBoardroom"),
                      (399, 240, "riseandfallofboards399"),
                      (30000, 200, "TheComprehensiveNonExecut"),
                      (60000, 29, "TheComprehensiveNonExecut")]
TIERS = na.infer_tiers(LAWTORNEY_PAYMENTS, CFG["tiers"])
WINDOW = {"source": "brand_history", "converters": 900, "p50_hours": 98.0, "p75_hours": 366.0}
BRAND_BRAIN = {"_id": "bb1", "answers": {"brandVoice": "Professional / Authoritative",
                                         "salesCycle": "3+ months"}}


class DeletableCollection(FakeCollection):
    async def delete_one(self, query):
        class R:
            deleted_count = 0
        result = R()
        for key, doc in list(self.docs.items()):
            if self._matches(doc, query):
                del self.docs[key]
                result.deleted_count = 1
                break
        return result


def tp(kind, hours_ago, value=None, code=None, synthetic=False, source="ICDP Webinar Registration New"):
    at = NOW - timedelta(hours=hours_ago)
    return {"type": kind, "occurred_at": at, "created_at": at, "source": source,
            "channel": "website", "value": value, "currency": "INR" if value else None,
            "product_code": code, "synthetic": synthetic}


def call(hours_ago, disposition, callback_at=None, analysis_id=None):
    return {"occurred_at": NOW - timedelta(hours=hours_ago), "disposition": disposition,
            "direction": "outbound", "callback_at": callback_at,
            "analysis_id": analysis_id, "analysis_status": "completed" if analysis_id else None}


def ctx(tps=(), calls=(), wa=None, phone_count=1, status="new", currency="INR"):
    return {
        "lead": {"id": LEAD, "brand_id": BRAND, "full_name": "sanjay verma", "status": status,
                 "created_at": NOW - timedelta(days=15), "designation": None},
        "brand": {"id": BRAND, "name": "Lawtorney", "timezone": "Asia/Kolkata",
                  "currency": currency, "brand_brain_id": "bb1"},
        "touchpoints": list(tps), "calls": list(calls),
        "whatsapp": wa or {"conversations": 0, "last_direction": None,
                           "last_message_at": None, "unread_count": 0},
        "phone_count": phone_count,
    }


def decide(c, window=WINDOW, tiers=TIERS, brand_brain=BRAND_BRAIN):
    return na.decide(c, cfg=CFG, tiers=tiers, window=window, brand_brain=brand_brain, now=NOW)


# =========================================================================== #
# Rules
# =========================================================================== #
def test_screenshot_sanjay_repeat_registrations_uncontacted_is_high_call():
    c = ctx([tp("form_submit", h) for h in (339, 290, 284, 27, 3)], phone_count=2)
    d = decide(c)
    assert d["action_type"] == "call_probe_repeat"
    assert d["urgency_level"] == "high"
    assert d["facts"]["form_count"] == 5
    assert "multiple_phone_numbers" in d["flags"]
    assert d["lead_state"]["contacted"] is False


def test_screenshot_praveena_entry_ticket_is_pitch_core_not_onboarding():
    c = ctx([tp("form_submit", 15), tp("payment", 1.5, 299, "CrossroadstoBoardroom")])
    d = decide(c)
    assert d["action_type"] == "pitch_core_offer"
    assert d["urgency_level"] == "high"
    assert d["lead_state"]["highest_tier"] == "entry"
    assert d["facts"]["last_entry_amount"] == "₹299"
    assert d["facts"]["core_product"] == "TheComprehensiveNonExecut"


def test_urgency_basis_reads_naturally():
    today = decide(ctx([tp("payment", 1.5, 299)]))
    assert "today" in today["urgency_basis"] and "0 days" not in today["urgency_basis"]
    yesterday = decide(ctx([tp("payment", 30, 299)]))
    assert "1 day ago" in yesterday["urgency_basis"]


def test_entry_ticket_urgency_decays_with_time():
    old = decide(ctx([tp("payment", 24 * 20, 299)]))
    older = decide(ctx([tp("payment", 24 * 60, 299)]))
    assert (old["urgency_level"], older["urgency_level"]) == ("medium", "low")


def test_core_buyer_recent_is_onboard_then_nurture():
    recent = decide(ctx([tp("payment", 24 * 3, 30000, "TheComprehensiveNonExecut")]))
    later = decide(ctx([tp("payment", 24 * 40, 30000)]))
    assert (recent["action_type"], recent["urgency_level"]) == ("onboard", "medium")
    assert (later["action_type"], later["urgency_level"]) == ("nurture_next_tier", "low")
    assert recent["facts"]["last_core_amount"] == "₹30,000"


def test_callback_uses_callback_time_as_due_by():
    cb = NOW + timedelta(hours=5)
    d = decide(ctx([tp("form_submit", 48)], [call(20, "callback", callback_at=cb)]))
    assert d["action_type"] == "callback" and d["urgency_level"] == "high"
    assert d["due_by"] == cb.astimezone(rules.brand_zone("Asia/Kolkata")).isoformat(timespec="minutes")


def test_irrelevant_call_disqualifies_unless_lead_came_back():
    assert decide(ctx([tp("form_submit", 48)], [call(20, "irrelevant")]))["action_type"] == "disqualify"
    came_back = decide(ctx([tp("form_submit", 48), tp("form_submit", 2)], [call(20, "irrelevant")]))
    assert came_back["action_type"] != "disqualify"


def test_unanswered_inbound_whatsapp_beats_everything_but_disqualify():
    wa = {"conversations": 1, "last_direction": "inbound",
          "last_message_at": NOW - timedelta(hours=5), "unread_count": 2}
    d = decide(ctx([tp("payment", 24 * 40, 30000)], wa=wa))
    assert (d["action_type"], d["urgency_level"]) == ("reply_whatsapp", "high")


def test_sql_call_without_payment_follows_up_on_objections():
    d = decide(ctx([tp("form_submit", 72)], [call(24, "SQL")]))
    assert (d["action_type"], d["urgency_level"]) == ("follow_up_objection", "high")
    assert decide(ctx([tp("form_submit", 300)], [call(24 * 6, "SQL")]))["urgency_level"] == "medium"


def test_no_answer_is_retry_contact():
    assert decide(ctx([tp("form_submit", 30)], [call(5, "no_answer")]))["action_type"] == "retry_contact"


def test_uncontacted_urgency_follows_brand_window():
    assert decide(ctx([tp("form_submit", 50)]))["action_type"] == "call_now"
    assert decide(ctx([tp("form_submit", 200)]))["action_type"] == "follow_up"
    stale = decide(ctx([tp("form_submit", 24 * 60)]))
    assert (stale["action_type"], stale["urgency_level"]) == ("re_engage", "low")


def test_synthetic_touchpoints_are_ignored_and_flagged():
    fake = [tp("custom", 3, synthetic=True, source="Webinar Attended"),
            tp("email", 2, synthetic=True, source="Follow-up Email Opened")]
    d = decide(ctx(fake))
    assert d["action_type"] == "insufficient_data"
    assert d["confidence"] == "low"
    assert "synthetic_touchpoints_ignored" in d["flags"]


def test_evidence_always_reports_calls_and_whatsapp():
    d = decide(ctx([tp("form_submit", 3)]))
    types_ = [e["type"] for e in d["evidence"]]
    assert types_[:3] == ["form_submit", "sales_call", "whatsapp"]
    assert d["evidence"][1]["count"] == 0


def test_invalid_brand_currency_never_guesses_a_symbol():
    c = ctx([tp("payment", 2, 299)], currency="1234567")
    c["touchpoints"][0]["currency"] = None
    d = decide(c)
    assert "brand_currency_invalid" in d["flags"]
    assert d["facts"]["last_entry_amount"] == "299"


def test_decision_is_deterministic():
    c1 = ctx([tp("form_submit", h) for h in (40, 3)])
    c2 = ctx([tp("form_submit", h) for h in (40, 3)])
    assert decide(c1) == decide(c2)


# =========================================================================== #
# Windows and money
# =========================================================================== #
def test_conversion_window_falls_back_to_brand_brain_then_default():
    thin = {"converters": 3, "p50_hours": 10.0, "p75_hours": 20.0}
    assert rules.conversion_window(thin, BRAND_BRAIN, CFG)["p75_hours"] == 2160.0
    assert rules.conversion_window(thin, None, CFG)["source"] == "default"
    rich = {"converters": 900, "p50_hours": 98.03, "p75_hours": 365.9}
    assert rules.conversion_window(rich, None, CFG)["source"] == "brand_history"


@pytest.mark.parametrize("raw,hours", [("3+ months", 2160.0), ("1-2 weeks", 336.0),
                                       ("Same day", 24.0), ("about a month", 720.0),
                                       ("it depends", None), (None, None)])
def test_parse_sales_cycle(raw, hours):
    assert rules.parse_sales_cycle_hours(raw) == hours


def test_money_uses_indian_grouping():
    assert rules.fmt_money(150000, "INR") == "₹1,50,000"
    assert rules.fmt_money(299, None) == "299"
    assert rules.fmt_money(1200.5, "USD") == "1,200.50 USD"


# =========================================================================== #
# Tiers
# =========================================================================== #
def test_lawtorney_prices_split_into_entry_and_core():
    assert [(t["kind"], t["min_amount"], t["max_amount"]) for t in TIERS] == [
        ("entry", 149.0, 399.0), ("core", 30000.0, 60000.0)]
    assert TIERS[0]["product_codes"][0] == "CrossroadstoBoardroom"


def test_single_outlier_does_not_become_a_tier():
    kaizen = [(76000, 25, None), (84302, 58, None), (95294, 15, None), (480000, 1, None)]
    tiers = na.infer_tiers(kaizen, CFG["tiers"])
    assert [t["kind"] for t in tiers] == ["core"]
    assert tiers[0]["max_amount"] == 480000.0


def test_classify_unseen_price_goes_to_nearest_tier():
    assert na.classify_amount(349, TIERS)["kind"] == "entry"
    assert na.classify_amount(45000, TIERS)["kind"] == "core"
    assert na.classify_amount(10000, TIERS)["kind"] == "core"
    assert na.classify_amount(0, TIERS) is None


def test_override_validation():
    ok = na.validate_override([{"kind": "core", "min_amount": 10000, "max_amount": 90000},
                               {"kind": "entry", "min_amount": 0, "max_amount": 999,
                                "product_label": "Webinar ticket"}])
    assert [t["kind"] for t in ok] == ["entry", "core"]
    assert ok[0]["source"] == "override"
    with pytest.raises(ValueError, match="overlap"):
        na.validate_override([{"kind": "entry", "min_amount": 0, "max_amount": 500},
                              {"kind": "core", "min_amount": 400, "max_amount": 900}])
    with pytest.raises(ValueError, match="kind"):
        na.validate_override([{"kind": "gold", "min_amount": 0, "max_amount": 5}])


# =========================================================================== #
# Writer
# =========================================================================== #
PAYLOAD = {"decision": {"act_within_hours": 24}, "facts": {"form_count": 5, "last_touch_date": "11 Sep 2026"}}
TEMPLATE = CFG["actions"]["call_probe_repeat"]


def good_wording(**over):
    w = {"title": "Call within 24 hours", "recommendation": "Ask what stopped them after 5 forms.",
         "reason_headline": "Keeps registering", "reason": "Last form on 11 Sep 2026."}
    w.update(over)
    return FakeResponse(json.dumps(w))


def neutral_wording():
    """Wording with no numbers, valid against any lead's facts - for service tests,
    where the facts differ from PAYLOAD."""
    return FakeResponse(json.dumps({
        "title": "Call the lead today", "recommendation": "Ask what they need to decide.",
        "reason_headline": "Not contacted yet", "reason": "Recent form activity, no calls."}))


def run(coro):
    return asyncio.run(coro)


def test_writer_accepts_grounded_output():
    out = run(writer.write(FakeClient([good_wording()]), "m", PAYLOAD, TEMPLATE))
    assert out["source"] == "llm" and out["wording"]["title"] == "Call within 24 hours"


def test_writer_rejects_invented_number_then_repairs():
    client = FakeClient([good_wording(recommendation="Offer 20% off."), good_wording()])
    out = run(writer.write(client, "m", PAYLOAD, TEMPLATE))
    assert out["source"] == "llm" and client.models.calls == 2


def test_writer_rejects_gendered_pronouns():
    client = FakeClient([good_wording(recommendation="Ask him what stopped the signup."),
                         good_wording()])
    out = run(writer.write(client, "m", PAYLOAD, TEMPLATE))
    assert out["source"] == "llm" and client.models.calls == 2
    assert writer.validate({"title": "t", "recommendation": "Ask what they need.",
                            "reason_headline": "h", "reason": "Heard back."}, PAYLOAD) is None


def test_writer_falls_back_to_template():
    bad = good_wording(recommendation="Offer 20% off.")
    out = run(writer.write(FakeClient([bad, bad]), "m", PAYLOAD, TEMPLATE))
    assert out["source"] == "template" and out["fallback_reason"] == "unsupported_number"
    assert out["wording"]["reason"].startswith("The lead submitted 5 forms")

    boom = run(writer.write(FakeClient([RuntimeError("timeout")]), "m", PAYLOAD, TEMPLATE))
    assert boom["source"] == "template" and boom["fallback_reason"] == "llm_error"


# =========================================================================== #
# Service
# =========================================================================== #
def make_deps(lead_ctx, llm, store=None, payments=LAWTORNEY_PAYMENTS, raise_db=False):
    async def run_sync(fn, *args):
        return fn(*args)

    def load_lead(_id):
        if raise_db:
            raise na_data.DataUnavailable("OperationalError")
        return lead_ctx

    async def brand_brain(_id):
        return BRAND_BRAIN

    async def analysis(_id):
        return {"report": {"summary": "Asked about fees.",
                           "objections": [{"summary": "The fee is higher than expected",
                                           "handled": "partially"}]}}

    return na.SuggestDeps(run_sync=run_sync, load_lead=load_lead,
                          brand_payments=lambda _b: payments,
                          brand_windows=lambda _b: {"converters": 900, "p50_hours": 98.0,
                                                    "p75_hours": 366.0},
                          load_brand_brain=brand_brain, load_call_analysis=analysis,
                          store=store, llm_client=llm, llm_model="gemini-test",
                          now=lambda: NOW)


def test_service_returns_the_three_part_card():
    c = ctx([tp("form_submit", h) for h in (339, 290, 3)])
    r = run(na.suggest_next_action(LEAD, make_deps(c, FakeClient([neutral_wording()]))))
    card = r["ai_suggested_next_action"]
    assert set(card) == {"urgency", "recommendation", "reason"}
    assert card["urgency"]["label"] == "HIGH URGENCY"
    assert card["recommendation"]["action_type"] == "call_probe_repeat"
    assert card["reason"]["evidence"]
    assert r["availability"]["available"] is True and r["fallback"] is False
    assert r["tiers"]["source"] == "inferred"


def test_service_reuses_cached_wording_until_facts_change():
    store = na.NextActionStore(FakeCollection(), DeletableCollection())
    c = ctx([tp("form_submit", 3)])
    llm = FakeClient([neutral_wording(), neutral_wording()])
    first = run(na.suggest_next_action(LEAD, make_deps(c, llm, store)))
    second = run(na.suggest_next_action(LEAD, make_deps(c, llm, store)))
    assert (first["cached"], second["cached"]) == (False, True)
    assert llm.models.calls == 1
    run(na.suggest_next_action(LEAD, make_deps(c, llm, store), refresh=True))
    assert llm.models.calls == 2


def test_service_uses_tier_override():
    store = na.NextActionStore(FakeCollection(), DeletableCollection())
    run(store.set_tier_override(BRAND, na.validate_override(
        [{"kind": "core", "min_amount": 100, "max_amount": 1000, "product_label": "Board Seat Course"}])))
    c = ctx([tp("payment", 24, 299)])
    r = run(na.suggest_next_action(LEAD, make_deps(c, FakeClient([neutral_wording()]), store)))
    assert r["tiers"]["source"] == "override"
    assert r["ai_suggested_next_action"]["recommendation"]["action_type"] == "onboard"


def test_service_passes_call_insights_to_the_writer():
    c = ctx([tp("form_submit", 72)], [call(24, "SQL", analysis_id="a1")])
    llm = FakeClient([neutral_wording()])
    r = run(na.suggest_next_action(LEAD, make_deps(c, llm)))
    assert r["wording"]["call_analysis_used"] is True


def test_service_unavailable_paths():
    bad = run(na.suggest_next_action("not-a-uuid", make_deps(None, FakeClient([]))))
    assert bad["availability"]["reason"] == "lead_not_found" and bad["ai_suggested_next_action"] is None
    missing = run(na.suggest_next_action(LEAD, make_deps(None, FakeClient([]))))
    assert missing["availability"]["reason"] == "lead_not_found"
    down = run(na.suggest_next_action(LEAD, make_deps(None, FakeClient([]), raise_db=True)))
    assert down["availability"]["reason"] == "database_unavailable"


# =========================================================================== #
# API
# =========================================================================== #
@pytest.fixture
def api(monkeypatch):
    store = na.NextActionStore(FakeCollection(), DeletableCollection())
    c = ctx([tp("form_submit", 3)])
    monkeypatch.setattr(app_module, "ai_next_action_store", store)
    monkeypatch.setattr(app_module, "_na_deps",
                        lambda: make_deps(c, FakeClient([neutral_wording()] * 5), store))
    monkeypatch.setattr(na_data, "brand_payments", lambda _b: LAWTORNEY_PAYMENTS)
    with TestClient(app_module.app) as client:
        yield client


def test_api_requires_key(api):
    assert api.get(f"/api/ai-suggested-next-action/{LEAD}").status_code == 401


def test_api_card(api):
    body = api.get(f"/api/ai-suggested-next-action/{LEAD}", headers=HEADERS).json()
    assert body["ai_suggested_next_action"]["urgency"]["level"] == "high"
    assert body["availability"]["available"] is True


def test_api_tier_override_round_trip(api):
    url = f"/api/ai-suggested-next-action/tiers/{BRAND}"
    before = api.get(url, headers=HEADERS).json()
    assert before["active_source"] == "inferred" and before["override"] is None

    bad = api.put(url, headers=HEADERS, json={"tiers": [
        {"kind": "entry", "min_amount": 0, "max_amount": 500},
        {"kind": "core", "min_amount": 400, "max_amount": 900}]})
    assert bad.status_code == 422

    ok = api.put(url, headers=HEADERS, json={"tiers": [
        {"kind": "entry", "min_amount": 0, "max_amount": 999, "product_label": "Webinar ticket"},
        {"kind": "core", "min_amount": 1000, "max_amount": 200000}]})
    assert ok.status_code == 200 and ok.json()["active_source"] == "override"
    assert api.get(url, headers=HEADERS).json()["active_source"] == "override"

    assert api.delete(url, headers=HEADERS).json()["deleted"] is True
    assert api.get(url, headers=HEADERS).json()["active_source"] == "inferred"


def test_api_tier_rejects_non_uuid(api):
    r = api.get("/api/ai-suggested-next-action/tiers/lawtorney", headers=HEADERS)
    assert r.status_code == 422


def test_api_tier_write_needs_storage(api, monkeypatch):
    monkeypatch.setattr(app_module, "ai_next_action_store", None)
    r = api.put(f"/api/ai-suggested-next-action/tiers/{BRAND}", headers=HEADERS,
                json={"tiers": [{"kind": "core", "min_amount": 1, "max_amount": 2}]})
    assert r.status_code == 503


def test_health_reports_feature(api):
    assert api.get("/health").json()["ai_suggested_next_action"]["available"] is True
