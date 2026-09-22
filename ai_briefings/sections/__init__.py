"""One builder per Briefings tab. See common.py for the contract."""
from . import ads, leads, overall, sales, whatsapp  # noqa: F401
from .common import DayContext  # noqa: F401

BUILDERS = {"sales": sales.build, "ads": ads.build, "whatsapp": whatsapp.build,
            "leads": leads.build}
