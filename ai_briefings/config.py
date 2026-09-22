"""Loads briefing_config.json once. Thresholds live there, not in code, so a
brand-wide tuning ("ROAS below 2x is fine for us") is a config change."""
from __future__ import annotations

import json
import os
import threading

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "briefing_config.json")
_LOCK = threading.Lock()
_CFG: dict | None = None

SECTIONS = ("sales", "ads", "whatsapp", "leads")      # the four that read data
ALL_SECTIONS = ("all",) + SECTIONS                     # plus the one composed from them


def load_config(force: bool = False) -> dict:
    global _CFG
    with _LOCK:
        if _CFG is None or force:
            with open(_PATH, encoding="utf-8") as fh:
                cfg = json.load(fh)
            missing = [s for s in ALL_SECTIONS if s not in cfg["sections"]]
            if missing:
                raise ValueError(f"briefing_config.json has no label for {missing}")
            _CFG = cfg
        return _CFG
