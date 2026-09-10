"""Moved to `transcription/deepgram_client.py`.

Vision Lab needs the same client for the audio track of a video ad, and
`vision_lab` importing from `sales_call_analyzer` would be the wrong dependency
edge - a change made for sales calls could then break creative analysis.

THIS IS AN ALIAS, NOT A RE-EXPORT.
    Replacing this module in sys.modules makes `sales_call_analyzer.deepgram_client`
    and `transcription.deepgram_client` the SAME object, so module-level state -
    the shared httpx client, the patched settings a test reaches for - is one
    thing rather than two copies that silently diverge. A plain `from ... import *`
    copies names once at import and leaves monkeypatching one module invisible to
    the other.

New code should import from `transcription` directly.
"""
import sys

from transcription import deepgram_client as _module

sys.modules[__name__] = _module
