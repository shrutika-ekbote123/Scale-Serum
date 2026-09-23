"""AI Briefings - the daily morning briefing on the Briefings page.

Five tabs, each its own stored briefing per brand and local day:

    all        composed from the four below
    sales      Sales Team, plus the Consolidated Score card and per-rep table
    ads        Ads & Marketing
    whatsapp   WhatsApp
    leads      Leads

DIVISION OF LABOUR (a correctness rule, not a style preference)
  * sections/*.py compute every number and choose every Watch item,
    deterministically, from scrumdb (read-only) and the call analyses.
  * writer.py only words them (Gemini). Any number it writes must appear in the
    facts it was given; otherwise the section's template wording is stored.

Briefings are generated ahead of time - by the briefing-worker process each
morning, or by POST /api/ai-briefings/{brand_id}/generate - and stored in
MongoDB. Reading a briefing never touches scrumdb or Gemini.
"""
from .access import Viewer, viewer_from  # noqa: F401
from .config import ALL_SECTIONS, SECTIONS, load_config  # noqa: F401
from .service import BriefingDeps, generate_day, new_run_id  # noqa: F401
from .store import BriefingStore, briefing_id  # noqa: F401
from .usage import summarize as summarize_usage  # noqa: F401
from .writer import PROMPT_VERSION  # noqa: F401

__all__ = ["Viewer", "viewer_from", "ALL_SECTIONS", "SECTIONS", "load_config",
           "BriefingDeps", "generate_day", "new_run_id", "BriefingStore", "briefing_id",
           "summarize_usage",
           "PROMPT_VERSION"]
