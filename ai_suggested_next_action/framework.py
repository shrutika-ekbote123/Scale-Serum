"""Loads next_action_framework.json - the rulebook's numbers and templates."""
from __future__ import annotations

import json
import os
from functools import lru_cache

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_PATH = os.path.join(PKG_DIR, "next_action_framework.json")

URGENCY_LEVELS = ("high", "medium", "low")
TEMPLATE_FIELDS = ("channel", "title", "text", "reason_headline", "reason_text")


@lru_cache(maxsize=1)
def load_framework() -> dict:
    """Read and sanity-check the rulebook. Raises at import time in app.py, so a
    malformed file disables this feature loudly instead of failing per request."""
    with open(FRAMEWORK_PATH, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for level in URGENCY_LEVELS:
        entry = cfg["urgency"][level]
        if not entry.get("label") or int(entry["act_within_hours"]) <= 0:
            raise ValueError(f"urgency.{level} needs a label and positive act_within_hours")
    for action, template in cfg["actions"].items():
        missing = [f for f in TEMPLATE_FIELDS if not template.get(f)]
        if missing:
            raise ValueError(f"action {action!r} is missing {missing}")
    for key in ("windows", "thresholds", "tiers"):
        if not isinstance(cfg.get(key), dict):
            raise ValueError(f"{key} block is missing")
    return cfg
