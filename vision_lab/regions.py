"""
Step 10 - what is actually on screen. OCR, faces, brand mark, CTA.

WHY THIS EXISTS
    A saliency map says where the eye goes. On its own that is decoration. The
    marketing conclusions in the report all come from INTERSECTING it with the
    things detected here: gaze on the brand mark is Brand Memory, gaze on the
    headline is Focus, words on screen against the time they are held is
    Cognitive Demand.

EVERY DETECTOR DEGRADES RATHER THAN FAILS
    OCR unavailable, no brand mark supplied, no face model installed - each is
    reported as a fact and the affected scores say which basis they used. None
    of them stops an analysis. That is the same rule the rest of the service
    follows: a stated degradation, never a crash and never a silent zero.

OCR IS THE EXPENSIVE PART, NOT SALIENCY
    Tesseract on a 640-wide frame costs roughly 0.3-1 s. Across 48 frames that
    dwarfs everything else in the pipeline, so it is gated on shot changes -
    within a single continuous shot the on-screen text does not change, and
    re-reading it 20 times buys nothing. VL_OCR_ON_SHOT_CHANGE_ONLY controls it.
"""
from __future__ import annotations

import functools
import logging
import os
import re
import shutil
from typing import Optional

import cv2
import numpy as np

from . import framework as fw

logger = logging.getLogger("vision_lab.regions")

OCR_ENGINE = os.environ.get("VL_OCR_ENGINE", "tesseract").lower()
MIN_OCR_CONFIDENCE = int(os.environ.get("VL_OCR_MIN_CONFIDENCE", 45))

# PAGE SEGMENTATION MODE - the single most important OCR setting here.
#
# Tesseract defaults to PSM 3, "fully automatic page segmentation", which
# assumes a document: columns, paragraphs, a reading order. An ad frame is
# nothing like that - it is a few words floating over a photograph.
#
# Measured on a real ScaleSerum creative: PSM 3 returned ZERO tokens from a
# frame containing a wordmark and a subtitle. PSM 11 ("sparse text - find as
# much text as possible, no particular order") read the brand name at 92%
# confidence off the same pixels. The difference is not marginal, it is
# everything.
OCR_PSM = os.environ.get("VL_OCR_PSM", "11")

# Tesseract language packs, "+"-separated. English alone cannot read Devanagari,
# and ScaleSerum's creatives are frequently Hinglish - Hindi script mixed with
# English words. With only `eng` installed, the Hindi half of a subtitle comes
# back as noise ("a", "aie", "die") and the word count silently undercounts.
#
#   Windows: the Tesseract installer offers extra languages
#   Linux:   apt install -y tesseract-ocr-hin
#
# Then set VL_OCR_LANGUAGES=eng+hin. Languages that are not installed are
# dropped at startup with a warning rather than failing every frame.
OCR_LANGUAGES = os.environ.get("VL_OCR_LANGUAGES", "eng")

# Where the language packs live. Tesseract's own tessdata directory usually sits
# under Program Files or /usr/share, which needs administrator rights to add to.
# Pointing at a directory we own means a language pack is a file copy rather
# than a privileged install - which matters on a server we deploy to over SSH.
#
# When set, tesseract looks ONLY here, so every language in use must be present
# (eng included).
TESSDATA_DIR = os.environ.get("VL_TESSDATA_DIR", "")
MODEL_DIR = os.environ.get("VL_MODEL_DIR", "")

# Windows installs tesseract outside PATH by default, which makes "OCR does not
# work on the server but works on my machine" a very easy afternoon to lose.
_WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_ocr_ready: Optional[bool] = None
_face_detector = None


# --------------------------------------------------------------------------- OCR
def ocr_available() -> bool:
    global _ocr_ready
    if _ocr_ready is not None:
        return _ocr_ready
    if OCR_ENGINE == "none":
        _ocr_ready = False
        return False
    try:
        import pytesseract
    except ImportError:
        _ocr_ready = False
        return False

    binary = os.environ.get("VL_TESSERACT_PATH") or shutil.which("tesseract")
    if not binary and os.path.isfile(_WINDOWS_TESSERACT):
        binary = _WINDOWS_TESSERACT
    if binary:
        pytesseract.pytesseract.tesseract_cmd = binary
    try:
        pytesseract.get_tesseract_version()
        _ocr_ready = True
    except Exception as err:  # noqa: BLE001
        logger.warning("OCR unavailable: %s", err)
        _ocr_ready = False
    return _ocr_ready


def _config() -> str:
    """Tesseract flags.

    The language-pack directory is passed through TESSDATA_PREFIX rather than
    --tessdata-dir: pytesseract splits the config string on whitespace, so a
    quoted path arrives at tesseract with the quotes still attached and every
    lookup fails with a confusing "could not load any languages". The
    environment variable has no such problem and handles spaces in the path.
    """
    if TESSDATA_DIR and os.path.isdir(TESSDATA_DIR):
        os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR
    return f"--psm {OCR_PSM}"


@functools.lru_cache(maxsize=1)
def available_languages() -> str:
    """The configured languages, minus any that are not actually installed.

    Asking tesseract for a language it does not have fails EVERY frame with the
    same error. Dropping the missing one and saying so degrades a Hinglish ad to
    English-only rather than to nothing.
    """
    wanted = [part for part in OCR_LANGUAGES.split("+") if part]
    if not ocr_available():
        return "eng"
    import pytesseract
    try:
        installed = set(pytesseract.get_languages(config=_config()))
    except Exception:  # noqa: BLE001
        return OCR_LANGUAGES

    usable = [lang for lang in wanted if lang in installed]
    missing = [lang for lang in wanted if lang not in installed]
    if missing:
        logger.warning("tesseract language pack(s) not installed: %s - "
                       "text in those scripts will not be read. "
                       "Install them, or drop them from VL_OCR_LANGUAGES.",
                       ", ".join(missing))
    return "+".join(usable) or "eng"


def read_text(pixels: np.ndarray) -> dict:
    """Text boxes, the words in them, and the total word count.

    Word count is what Cognitive Demand divides by reading speed, so a frame
    where OCR failed must report None rather than 0 - "we could not read it" and
    "there was nothing to read" are different findings.
    """
    if not ocr_available():
        return {"text_boxes": [], "word_count": None, "text": "",
                "text_area_share": None, "ocr_available": False}

    import pytesseract

    height, width = pixels.shape[:2]
    # Ad typography is usually light-on-dark or dark-on-light with high contrast;
    # greyscale plus Otsu gives tesseract a cleaner target than raw colour.
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    try:
        data = pytesseract.image_to_data(
            gray, lang=available_languages(), config=_config(),
            output_type=pytesseract.Output.DICT)
    except Exception as err:  # noqa: BLE001
        logger.warning("OCR failed on a frame: %s", err)
        return {"text_boxes": [], "word_count": None, "text": "",
                "text_area_share": None, "ocr_available": False}

    boxes, words, covered = [], [], 0
    for index, raw_word in enumerate(data.get("text") or []):
        word = (raw_word or "").strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, TypeError, ValueError):
            confidence = -1.0
        if confidence < MIN_OCR_CONFIDENCE:
            continue
        x, y = data["left"][index], data["top"][index]
        w, h = data["width"][index], data["height"][index]
        words.append(word)
        covered += w * h
        boxes.append({
            "box": [round(x / width, 4), round(y / height, 4),
                    round((x + w) / width, 4), round((y + h) / height, 4)],
            "text": word,
            "confidence": round(confidence, 1),
            "height_share": round(h / height, 4),
        })

    return {
        "text_boxes": boxes,
        "word_count": len(words),
        "text": " ".join(words),
        "text_area_share": round(covered / (width * height), 4),
        "ocr_available": True,
    }


# --------------------------------------------------------------------------- faces
def face_detector():
    """YuNet, if its model is installed. Faces are the strongest single human
    attractor, so their absence from the measurement record is worth reporting
    rather than glossing over."""
    global _face_detector
    if _face_detector is not None:
        return _face_detector or None
    path = os.path.join(MODEL_DIR, "face_detection_yunet.onnx") if MODEL_DIR else ""
    if not path or not os.path.isfile(path):
        _face_detector = False
        return None
    try:
        _face_detector = cv2.FaceDetectorYN.create(path, "", (320, 320))
    except Exception as err:  # noqa: BLE001
        logger.warning("face detector unavailable: %s", err)
        _face_detector = False
        return None
    return _face_detector


def find_faces(pixels: np.ndarray) -> dict:
    detector = face_detector()
    if detector is None:
        return {"faces": [], "face_detection": "unavailable"}

    height, width = pixels.shape[:2]
    detector.setInputSize((width, height))
    try:
        _, detections = detector.detect(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    except Exception as err:  # noqa: BLE001
        logger.warning("face detection failed: %s", err)
        return {"faces": [], "face_detection": "unavailable"}

    faces = []
    for row in (detections if detections is not None else []):
        x, y, w, h = row[:4]
        faces.append({"box": [round(float(x) / width, 4), round(float(y) / height, 4),
                              round(float(x + w) / width, 4),
                              round(float(y + h) / height, 4)],
                      "confidence": round(float(row[-1]), 3)})
    return {"faces": faces, "face_detection": "available"}


# --------------------------------------------------------------------------- brand
def find_brand(pixels: np.ndarray, text: str, *,
               wordmark: Optional[np.ndarray] = None,
               brand_names: Optional[list[str]] = None) -> Optional[dict]:
    """Two independent routes, either of which counts as an appearance.

    A wordmark template match finds the logo; matching the brand name in OCR text
    finds it spelled out. Without either we report no_brand_assets and Brand
    Memory says which basis it used.
    """
    for name in (brand_names or []):
        if name and name.lower() in (text or "").lower():
            return {"box": None, "basis": "brand_name_in_text", "name": name,
                    "confidence": 1.0}

    if wordmark is None:
        return None

    height, width = pixels.shape[:2]
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(wordmark, cv2.COLOR_RGB2GRAY) \
        if wordmark.ndim == 3 else wordmark

    best = None
    # A logo appears at whatever size the edit calls for, so one scale is not
    # enough; these six cover a corner bug through to a full-screen end card.
    for scale in (0.15, 0.25, 0.35, 0.5, 0.7, 0.9):
        tw = max(8, int(width * scale))
        th = max(8, int(template_gray.shape[0] * (tw / template_gray.shape[1])))
        if th >= height or tw >= width:
            continue
        resized = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if best is None or score > best["confidence"]:
            best = {"confidence": round(float(score), 4),
                    "box": [round(location[0] / width, 4), round(location[1] / height, 4),
                            round((location[0] + tw) / width, 4),
                            round((location[1] + th) / height, 4)],
                    "basis": "wordmark_template_match"}

    threshold = float(os.environ.get("VL_BRAND_MATCH_THRESHOLD", 0.6))
    return best if best and best["confidence"] >= threshold else None


# --------------------------------------------------------------------------- CTA
def find_cta(text_boxes: list[dict]) -> Optional[dict]:
    """The strongest call to action on screen, and whether it is a weak one.

    The weak list is the same one Script Lab already flags, kept in config so the
    two features cannot drift apart on what counts as a weak CTA.
    """
    strong = [w.lower() for w in (fw.lexicon("cta_strong") or [])]
    weak = [w.lower() for w in (fw.lexicon("cta_weak") or [])]

    joined = " ".join(box["text"].lower() for box in text_boxes)
    for phrase in weak:
        if phrase in joined:
            return {"text": phrase, "strength": "weak",
                    "box": _box_for(text_boxes, phrase)}
    for word in strong:
        if re.search(rf"\b{re.escape(word)}\b", joined):
            return {"text": word, "strength": "strong",
                    "box": _box_for(text_boxes, word)}
    return None


def _box_for(text_boxes: list[dict], phrase: str) -> Optional[list[float]]:
    first = phrase.split()[0]
    for box in text_boxes:
        if first in box["text"].lower():
            return box["box"]
    return None


# A price needs at least two digits to be believed. OCR on video frames throws
# off single-character fragments constantly - "$" is a routine misread of "S" -
# and on a real webinar ad that showed no price at all this read "$8" on five
# frames, which was enough to make Anchoring applicable and invite the model to
# judge a price nobody had shown. Real prices are 499, 1,299, 25,000; a
# one-digit price is far more likely to be noise than a finding.
MIN_PRICE_DIGITS = 2


def find_prices(text: str) -> list[str]:
    """Anchoring is not applicable unless a price is actually shown."""
    pattern = fw.lexicon("price_pattern")
    if not pattern or not text:
        return []
    try:
        found = re.findall(pattern, text)
    except re.error:
        return []
    return [price for price in found
            if len(re.sub(r"\D", "", price)) >= MIN_PRICE_DIGITS]


# --------------------------------------------------------------------------- entry
def detect(frame: dict, *, wordmark: Optional[np.ndarray] = None,
           brand_names: Optional[list[str]] = None,
           run_ocr: bool = True) -> dict:
    """The VisionDeps callable. Everything on screen, for one frame."""
    pixels = frame.get("pixels")
    if pixels is None:
        return {"text_boxes": [], "word_count": None, "faces": [],
                "brand": None, "cta": None, "ocr_available": False}

    text_data = read_text(pixels) if run_ocr else {
        "text_boxes": [], "word_count": None, "text": "",
        "text_area_share": None, "ocr_available": ocr_available(),
        "ocr_skipped": True}
    faces = find_faces(pixels)
    brand = find_brand(pixels, text_data.get("text", ""),
                       wordmark=wordmark, brand_names=brand_names)

    return {
        **text_data,
        **faces,
        "brand": brand,
        "cta": find_cta(text_data.get("text_boxes") or []),
        "prices": find_prices(text_data.get("text", "")),
    }


# --------------------------------------------------------------------------- peak labels
# What a numbered marker on the Attention Report is sitting on. Coordinates tell
# the frontend WHERE to draw the circle; these tell the marketer WHAT the eye
# went to, which is the part they can act on.
ELEMENT_BRAND = "brand_mark"
ELEMENT_CTA = "call_to_action"
ELEMENT_FACE = "face"
ELEMENT_TEXT = "on_screen_text"

# How much of a detected region must fall inside the peak before we are willing
# to name the peak after it. A peak is usually larger than a word box, so this
# asks "is this element inside that hotspot", not "are they the same shape".
MIN_PEAK_OVERLAP = 0.5


def overlap_fraction(inner: list, outer: list) -> float:
    """How much of `inner` lies inside `outer`. Both fractional [x0,y0,x1,y1]."""
    if not inner or not outer or len(inner) != 4 or len(outer) != 4:
        return 0.0
    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ((x1 - x0) * (y1 - y0) / area) if area > 0 else 0.0


def label_peaks(peaks: list[dict], regions: Optional[dict]) -> list[dict]:
    """Name each attention peak after the thing sitting under it.

    THE ORDER IS BY SPECIFICITY, NOT BY OVERLAP. A call to action and a brand
    wordmark are both made of text, so whichever overlaps most would usually be
    "on_screen_text" - the least useful of the three answers. Brand and CTA are
    therefore checked first and win outright.

    A peak we cannot name gets `element: None` rather than a guess. That is the
    common case for product footage, b-roll and people the face detector missed,
    and it is a more honest answer than labelling the frame's whole background
    as "text" because one caption clipped the corner of the hotspot.
    """
    regions = regions or {}
    brand = regions.get("brand") or {}
    cta = regions.get("cta") or {}
    faces = regions.get("faces") or []
    text_boxes = regions.get("text_boxes") or []

    for peak in peaks or []:
        box = peak.get("box")
        if not box:
            continue

        element, detail = None, None

        if brand.get("box") and overlap_fraction(brand["box"], box) >= MIN_PEAK_OVERLAP:
            element, detail = ELEMENT_BRAND, brand.get("name")
        elif cta.get("box") and overlap_fraction(cta["box"], box) >= MIN_PEAK_OVERLAP:
            element, detail = ELEMENT_CTA, cta.get("text")
        elif any(overlap_fraction(f.get("box"), box) >= MIN_PEAK_OVERLAP
                 for f in faces if f.get("box")):
            element = ELEMENT_FACE
        else:
            # The words actually inside the hotspot, in reading order - so the
            # report can say "the eye went to BOARD READINESS", not "to text".
            inside = [b for b in text_boxes
                      if overlap_fraction(b.get("box"), box) >= MIN_PEAK_OVERLAP]
            if inside:
                inside.sort(key=lambda b: (round(b["box"][1], 2), b["box"][0]))
                element = ELEMENT_TEXT
                detail = " ".join(b.get("text", "") for b in inside).strip()[:80]

        peak["element"] = element
        peak["element_text"] = detail or None

    return peaks
