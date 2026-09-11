"""Product tiers: which rung of a brand's price ladder a payment sits on.

"Converted" is too coarse to act on. A lead who paid Rs 299 for a webinar ticket
and one who paid Rs 30,000 for the programme are both "converted", and the right
next step is completely different. So every payment is placed in a tier:

    entry   - the cheap first purchase (ticket, trial, masterclass)
    core    - the main offer
    premium - a clearly higher tier above the main offer

Tiers are INFERRED from the brand's own payment history by default, and a brand
can replace the inference with an explicit OVERRIDE (stored in MongoDB).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Optional

KINDS = ("entry", "core", "premium")
KIND_RANK = {kind: i for i, kind in enumerate(KINDS)}


def _count(band: list) -> int:
    return sum(n for _, n, _ in band)


def _kinds(n: int) -> list[str]:
    if n == 1:
        return ["core"]
    if n == 2:
        return ["entry", "core"]
    return ["entry"] + ["core"] * (n - 2) + ["premium"]


def infer_tiers(payments: Iterable[tuple], cfg: dict) -> list[dict]:
    """Split a brand's paid amounts into price bands.

    `payments` is (amount, payment_count, product_code) per distinct amount. Amounts
    are sorted and a new band starts wherever the next amount is more than
    `band_split_ratio` times the previous one - Rs 499 -> Rs 6,999 splits, Rs 15,000
    -> Rs 30,000 does not. Bands too thin to trust (a single Rs 4.8L payment) are
    merged into their busier neighbour, so one outlier never becomes a "tier".
    """
    rows = sorted((float(a), int(n), code or None)
                  for a, n, code in payments if a and float(a) > 0 and n)
    if not rows:
        return []

    bands: list[list] = [[rows[0]]]
    for prev, cur in zip(rows, rows[1:]):
        if cur[0] > prev[0] * cfg["band_split_ratio"]:
            bands.append([cur])
        else:
            bands[-1].append(cur)

    total = sum(n for _, n, _ in rows)
    floor = max(cfg["min_band_payments"], cfg["min_band_share"] * total)
    merged = True
    while merged and len(bands) > 1:
        merged = False
        for i, band in enumerate(bands):
            if _count(band) >= floor:
                continue
            left = bands[i - 1] if i > 0 else None
            right = bands[i + 1] if i + 1 < len(bands) else None
            target = i - 1 if right is None or (
                left is not None and _count(left) >= _count(right)) else i + 1
            bands[target] = sorted(bands[target] + band)
            del bands[i]
            merged = True
            break

    tiers = []
    for i, (band, kind) in enumerate(zip(bands, _kinds(len(bands)))):
        codes = Counter()
        for _, n, code in band:
            if code:
                codes[code] += n
        tiers.append({
            "name": f"Tier {i + 1}",
            "kind": kind,
            "min_amount": band[0][0],
            "max_amount": band[-1][0],
            "payment_count": _count(band),
            # The price most leads actually paid in this band. A band can span
            # several products (Rs 6,999 to Rs 1.5L), so its minimum is not "the
            # price" of the product it is named after; the busiest price is.
            "typical_amount": max(band, key=lambda r: r[1])[0],
            "product_label": None,
            "product_codes": [c for c, _ in codes.most_common(3)],
            "source": "inferred",
        })
    return tiers


def classify_amount(amount: Optional[float], tiers: list[dict]) -> Optional[dict]:
    """The tier an amount belongs to. An amount outside every range (a new price
    point) goes to the nearest tier on a log scale, because a Rs 349 ticket is
    closer to Rs 299 than to Rs 30,000 in every sense that matters here."""
    if not amount or amount <= 0 or not tiers:
        return None
    for tier in tiers:
        if tier["min_amount"] <= amount <= tier["max_amount"]:
            return tier

    def distance(tier):
        lo, hi = max(tier["min_amount"], 0.01), max(tier["max_amount"], 0.01)
        if amount < lo:
            return math.log(lo / amount)
        return math.log(amount / hi)

    return min(tiers, key=distance)


def tier_product(tier: Optional[dict]) -> Optional[str]:
    """What to call a tier's product in a sentence: the brand's own label when it
    set one, otherwise the most common product code seen in its payments."""
    if not tier:
        return None
    if tier.get("product_label"):
        return tier["product_label"]
    codes = tier.get("product_codes") or []
    return codes[0] if codes else None


def validate_override(raw: list) -> list[dict]:
    """Normalise a brand's tier override. Raises ValueError with a message the API
    can return as-is."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("tiers must be a non-empty list")
    if len(raw) > 10:
        raise ValueError("at most 10 tiers")
    tiers = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"tiers[{i}] must be an object")
        kind = item.get("kind")
        if kind not in KINDS:
            raise ValueError(f"tiers[{i}].kind must be one of {', '.join(KINDS)}")
        try:
            lo, hi = float(item.get("min_amount")), float(item.get("max_amount"))
        except (TypeError, ValueError):
            raise ValueError(f"tiers[{i}] needs numeric min_amount and max_amount")
        if lo < 0 or hi < lo:
            raise ValueError(f"tiers[{i}] needs 0 <= min_amount <= max_amount")
        label = item.get("product_label")
        if label is not None and (not isinstance(label, str) or len(label) > 120):
            raise ValueError(f"tiers[{i}].product_label must be a string of at most 120 characters")
        name = item.get("name") or f"Tier {i + 1}"
        tiers.append({"name": str(name)[:60], "kind": kind, "min_amount": lo,
                      "max_amount": hi, "payment_count": None, "typical_amount": None,
                      "product_label": (label or "").strip() or None,
                      "product_codes": [], "source": "override"})
    tiers.sort(key=lambda t: t["min_amount"])
    for a, b in zip(tiers, tiers[1:]):
        if b["min_amount"] <= a["max_amount"]:
            raise ValueError(f"tier ranges overlap: {a['name']} and {b['name']}")
    return tiers
