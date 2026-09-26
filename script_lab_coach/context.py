"""CreativeCoachContextBuilder - one canonical context object per turn.

WHAT IT GUARANTEES
  * One source of truth. Everything the coach knows about a turn arrives in a
    single CoachContext: the test, the Brand Brain and its tier, the facts, and
    whatever retrieval the intent asked for. Nothing downstream reaches for a
    database of its own.
  * Nothing is loaded that the question does not need. "Why is my hook weak?" is
    answerable from the score card alone. Loading ad insights, the creative
    corpus and version history for it would make the commonest question the
    slowest and dearest one. intents.py declares the tiers; this module honours
    them and no more.
  * Claims are impossible to make without data behind them. `capabilities` says
    what is answerable for THIS test, and the evidence list is built from what
    was actually read. A performance claim with no verified ad join cannot be
    assembled, let alone rendered.

CACHING
    Two caches, both small and both time-bounded:
      * the test row and its facts, keyed by test_id - a conversation asks many
        questions about one unchanging test, so turns after the first do no
        PostgreSQL work at all unless their intent needs a retrieval tier;
      * the Brand Brain, keyed by brand_brain_id, shared across tests.
    A stale entry is harmless here: a stored review never changes, and a Brand
    Brain edited mid-conversation is picked up within the TTL.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from . import data as _data
from . import facts as _facts
from . import intents as _intents
from .brand_brain import EMPTY as EMPTY_BRAIN
from .brand_brain import BrandBrain

logger = logging.getLogger("script_lab_coach")

TEST_TTL_SECONDS = 300
BRAIN_TTL_SECONDS = 300
PERFORMANCE_WINDOW_DAYS = 30
MIN_DAYS_FOR_CONFIDENT_PERFORMANCE = 7


class TestNotFound(Exception):
    """The test_id does not exist. A 404 at the API boundary."""


class BrandMismatch(Exception):
    """The caller's brand does not own this test. A 403 - and the gate that
    stops one brand's coaching from quoting another brand's data."""


class _TTLCache:
    """Deliberately tiny: a dict with timestamps and a size cap.

    An LRU library would be more elegant and would also be a dependency, a
    configuration surface and a thing to reason about at shutdown. This holds a
    few hundred entries for five minutes."""

    def __init__(self, ttl: float, max_items: int = 512):
        self.ttl = ttl
        self.max_items = max_items
        self._items: dict = {}

    def get(self, key):
        hit = self._items.get(key)
        if not hit:
            return None
        expires, value = hit
        if expires < time.monotonic():
            self._items.pop(key, None)
            return None
        return value

    def put(self, key, value) -> None:
        if len(self._items) >= self.max_items:
            # Drop the soonest to expire. At this size a full sort is cheaper
            # than maintaining an ordering structure.
            oldest = min(self._items, key=lambda k: self._items[k][0])
            self._items.pop(oldest, None)
        self._items[key] = (time.monotonic() + self.ttl, value)

    def clear(self) -> None:
        self._items.clear()


class CoachContext:
    """Everything one coach turn is allowed to know."""

    def __init__(self, *, test: dict, brand: Optional[dict], brain: BrandBrain,
                 intent, facts: list, performance: Optional[dict] = None,
                 creatives: Optional[dict] = None, competitors: Optional[list] = None,
                 versions: Optional[list] = None, degraded: Optional[list] = None,
                 other_brands: tuple = ()):
        self.test = test
        self.brand = brand or {}
        self.brain = brain
        # Every other brand in the system, for the leak check. Carried on the
        # context rather than configured per deployment: the list changes
        # whenever someone adds a brand, and a stale one silently stops
        # catching the leak it exists for.
        self.other_brands = tuple(other_brands)
        self.intent = intent
        self.facts = facts
        self.performance = performance
        self.creatives = creatives or {}
        self.competitors = competitors or []
        self.versions = versions or []
        # Tiers the intent asked for but that could not be read. Named, so the
        # coach can say "I could not reach your ad data" instead of implying the
        # ad has no delivery.
        self.degraded = degraded or []

    # ------------------------------------------------------------ capabilities
    @property
    def has_performance(self) -> bool:
        return bool(self.performance and self.performance.get("days_with_data"))

    @property
    def performance_is_thin(self) -> bool:
        """Enough data to mention, not enough to conclude from.

        Either too few days, or none of them recent - an ad that stopped running
        in June says nothing reliable about a script being tested in September."""
        if not self.performance:
            return False
        return bool(self.performance["days_with_data"] < MIN_DAYS_FOR_CONFIDENT_PERFORMANCE
                    or self.performance.get("stale"))

    @property
    def has_history(self) -> bool:
        return any(not v.get("ai_fallback") and v.get("score") is not None
                   for v in self.versions)

    @property
    def review_failed(self) -> bool:
        """The stored review never completed; its scores are placeholders."""
        return bool(self.test.get("ai_fallback"))

    @property
    def brand_brain_conflicts(self) -> list:
        """Other companies' names found in the material this turn was built on.

        Not a theoretical check, and not confined to the Brand Brain. DI's Brand
        Brain describes Lawttorney's customers, AND 20 of DI's 53 stored reviews
        discuss Lawttorney, because the reviewer was given that same document.
        A coach reading either one faithfully writes DI ad copy selling
        Lawttorney's product - which is what happened, and what the gate then
        rejected, costing roughly one turn in eight.

        Every source the model can see is checked, because warning it about one
        while feeding it another is how the first version of this failed to
        change anything."""
        from . import grounding

        test = self.test
        material = [self.brain.prompt_block(), str(test.get("ai_verdict") or ""),
                    str((test.get("emotional_angle") or {}).get("critique") or "")]
        material += [str(s.get("comment") or "") for s in test.get("section_breakdown") or []]
        for improvement in test.get("improvements") or []:
            material += [str(v) for v in improvement.values()]
        return grounding.foreign_brands("\n".join(material),
                                        own=self.brand.get("name"),
                                        others=self.other_brands)

    def capabilities(self) -> dict:
        return {
            "performance": self.has_performance,
            "compare": self.has_history,
            "brand_fit": self.brain.can_judge_brand_fit,
        }

    # ------------------------------------------------------------------ output
    def allowed_numbers(self) -> set:
        return _facts.allowed_numbers(self.facts)

    def facts_used(self) -> list:
        return [f.as_dict() for f in self.facts]

    def evidence(self) -> list:
        """Provenance for every claim drawn from outside the score card.

        Built from what was READ, not from what the model says it used - so a
        claim can never carry evidence that does not exist."""
        out = []
        if self.has_performance:
            perf = self.performance
            for metric in ("ctr_pct", "cpm", "cpc", "impressions", "clicks", "leads"):
                if perf.get(metric) is not None:
                    out.append({"type": "ad_performance", "ad_id": perf["ad_id"],
                                "metric": metric, "value": perf[metric],
                                "window": perf.get("window") or "30d",
                                "source": "meta_insights"})
        for item in self.creatives.get("winners", []):
            out.append({"type": "brand_creative", "ad_id": item["ad_id"],
                        "metric": "creative_score", "value": item["score"],
                        "window": "latest", "source": "mi_ad_creative_analyses"})
        for item in self.competitors:
            out.append({"type": "competitor_ad", "ad_id": item["ad_archive_id"],
                        "metric": "est_run_days", "value": item["est_run_days"],
                        "window": "all", "source": "mi_competitor_ads"})
        for version in self.versions:
            if version.get("score") is not None and not version.get("ai_fallback"):
                out.append({"type": "past_version", "ad_id": version["test_id"],
                            "metric": "overall_score", "value": version["score"],
                            "window": version.get("created_at") or "", "source":
                            "sl_script_lab_tests"})
        if self.brain.used:
            out.append({"type": "brand_brain", "ad_id": None, "metric": "tier",
                        "value": self.brain.tier, "window": "current",
                        "source": "brand_brains"})
        return out

    def summary(self) -> dict:
        """What monitoring records about this turn's context."""
        return {"test_id": self.test.get("test_id"), "brand_id": self.test.get("brand_id"),
                "intent": self.intent.name, "brand_brain_tier": self.brain.tier,
                "tiers": {"performance": self.has_performance,
                          "corpus": bool(self.creatives or self.competitors),
                          "history": bool(self.versions)},
                "degraded": list(self.degraded), "facts": len(self.facts),
                "review_failed": self.review_failed}


class CreativeCoachContextBuilder:
    """Builds a CoachContext. Holds the caches; owns no request state.

    `run_sync` is how synchronous psycopg work reaches a thread - app.py passes
    fastapi.concurrency.run_in_threadpool, tests pass a trivial shim. Keeping it
    injected is what lets this be tested without a running event loop policy or
    a FastAPI import."""

    def __init__(self, *, run_sync: Callable, load_brand_brain: Callable,
                 performance_window_days: int = PERFORMANCE_WINDOW_DAYS):
        self._run = run_sync
        self._load_brain = load_brand_brain
        self._window = performance_window_days
        self._tests = _TTLCache(TEST_TTL_SECONDS)
        self._brains = _TTLCache(BRAIN_TTL_SECONDS)

    # ------------------------------------------------------------------ pieces
    async def _test(self, test_id: str) -> dict:
        cached = self._tests.get(test_id)
        if cached:
            return cached
        test = await self._run(_data.load_test, test_id)
        if not test:
            raise TestNotFound(test_id)
        self._tests.put(test_id, test)
        return test

    async def _brand_and_brain(self, brand_id: Optional[str]):
        """The brand row and its Brand Brain.

        A brand_id absent from `brands` is normal in this data - 133 stored
        tests point at brands that do not exist. That is tier C, not an error."""
        if not brand_id:
            return None, EMPTY_BRAIN, ()
        cached = self._brains.get(brand_id)
        if cached:
            return cached
        try:
            brand = await self._run(_data.load_brand, brand_id)
        except _data.DataUnavailable:
            logger.warning("could not read brand %s", brand_id)
            return None, EMPTY_BRAIN, ()
        if not brand:
            return None, EMPTY_BRAIN, ()
        try:
            others = tuple(await self._run(_data.load_other_brand_names, brand_id))
        except _data.DataUnavailable:
            # An empty list disables the leak check, so it is worth a warning:
            # this is the gate that stopped the coach advertising another
            # company to this brand's user.
            logger.warning("could not read other brand names for %s", brand_id)
            others = ()
        try:
            brain = await self._load_brain(brand.get("brand_brain_id"), brand.get("name"))
        except Exception:  # noqa: BLE001 - craft mode beats an error in the panel
            logger.warning("brand brain lookup failed for brand %s", brand_id)
            return brand, EMPTY_BRAIN, others
        self._brains.put(brand_id, (brand, brain, others))
        return brand, brain, others

    async def _performance(self, test: dict, degraded: list) -> Optional[dict]:
        """Delivery for the published ad, or None.

        The join is VERIFIED rather than assumed: live data contains ad ids like
        'mi_named_0' and '202'. No matching rows means no performance tier, which
        is a different statement from "this ad got no clicks"."""
        ad_id = test.get("source_ad_id")
        if not ad_id or not test.get("brand_id"):
            return None
        try:
            return await self._run(_data.load_ad_performance, test["brand_id"], ad_id,
                                   self._window)
        except _data.DataUnavailable:
            degraded.append(_intents.PERFORMANCE)
            logger.warning("ad performance unavailable for %s", ad_id)
            return None

    async def _corpus(self, test: dict, degraded: list):
        brand_id = test.get("brand_id")
        if not brand_id:
            return {}, []
        try:
            # Same platform as the script under review, when it names one: a
            # Meta script is coached against what worked on Meta.
            platform = (test.get("source_platform") or "").strip().lower() or None
            creatives, competitors = await asyncio.gather(
                self._run(_data.load_brand_creatives, brand_id, 3, 2, platform),
                self._run(_data.load_competitor_hooks, brand_id),
            )
            return creatives, competitors
        except _data.DataUnavailable:
            degraded.append(_intents.CORPUS)
            logger.warning("creative corpus unavailable for brand %s", brand_id)
            return {}, []

    async def _versions(self, test: dict, degraded: list) -> list:
        if not test.get("brand_id"):
            return []
        try:
            return await self._run(_data.load_versions, test["brand_id"],
                                   test.get("ad_number"), test["test_id"])
        except _data.DataUnavailable:
            degraded.append(_intents.HISTORY)
            logger.warning("version history unavailable for test %s", test["test_id"])
            return []

    # ------------------------------------------------------------------- build
    async def build(self, *, test_id: str, intent_name: Optional[str] = None,
                    brand_id: Optional[str] = None,
                    extra_tiers: tuple = ()) -> CoachContext:
        """The canonical context for one turn.

        `extra_tiers` loads tiers the intent did not ask for. The opening panel
        needs it: `capabilities` has to say whether this ad has delivery and
        whether an earlier version exists, and neither is knowable without
        reading them.

        Raises TestNotFound (404) and BrandMismatch (403). Every other failure
        degrades: a tier that cannot be read is named in `degraded` and the coach
        speaks without it."""
        intent = _intents.get(intent_name)
        wanted = set(intent.needs) | set(extra_tiers)
        test = await self._test(test_id)

        # The identity gate. Cross-brand leakage is the one failure mode that
        # produces confident, plausible, completely wrong coaching, so it is
        # checked before a single byte of brand data is loaded.
        if brand_id and test.get("brand_id") and str(brand_id) != str(test["brand_id"]):
            raise BrandMismatch(test_id)

        degraded: list = []
        brand, brain, other_brands = await self._brand_and_brain(test.get("brand_id"))

        # Only the tiers this intent declared, and all of them at once.
        jobs = {}
        if _intents.PERFORMANCE in wanted:
            jobs["performance"] = self._performance(test, degraded)
        if _intents.CORPUS in wanted:
            jobs["corpus"] = self._corpus(test, degraded)
        if _intents.HISTORY in wanted:
            jobs["versions"] = self._versions(test, degraded)
        results: dict[str, Any] = {}
        if jobs:
            done = await asyncio.gather(*jobs.values(), return_exceptions=True)
            for key, value in zip(jobs, done):
                if isinstance(value, Exception):
                    logger.warning("coach tier %s failed: %s", key, value)
                    degraded.append(key)
                    value = None
                results[key] = value

        performance = results.get("performance")
        creatives, competitors = results.get("corpus") or ({}, [])
        versions = results.get("versions") or []

        computed = (_facts.from_test(test)
                    + _facts.from_performance(performance)
                    + _facts.from_versions(versions)
                    + _facts.from_creatives(creatives))

        return CoachContext(test=test, brand=brand, brain=brain, intent=intent,
                            facts=computed, performance=performance,
                            creatives=creatives, competitors=competitors,
                            versions=versions, degraded=degraded,
                            other_brands=other_brands)

    def invalidate(self, test_id: Optional[str] = None) -> None:
        """Drop cached state - used after a retest, and by the tests."""
        if test_id is None:
            self._tests.clear()
            self._brains.clear()
        else:
            self._tests._items.pop(test_id, None)
