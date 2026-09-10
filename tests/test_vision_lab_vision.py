"""Milestone B - the vision layer, on synthetic images.

Runs on a bare CI runner: OpenCV and numpy only. No ffmpeg binary, no model
weights, no OCR binary, no network. The frames are generated here, so the
assertions are about arithmetic we control rather than about any particular ad.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

cv2 = pytest.importorskip("cv2")

from vision_lab import measure as ms  # noqa: E402
from vision_lab import saliency as sal  # noqa: E402


def frame_with_blob(width=320, height=180, centre=(160, 90), radius=20):
    """A dark frame with one bright disc - something an eye would land on."""
    pixels = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.circle(pixels, centre, radius, (240, 240, 240), -1)
    return {"t": 0.0, "index": 0, "pixels": pixels}


def flat_frame(width=320, height=180, value=8):
    return {"t": 0.0, "index": 0,
            "pixels": np.full((height, width, 3), value, dtype=np.uint8)}


# =========================================================================== #
# The invariant everything downstream depends on
# =========================================================================== #
def test_every_saliency_map_sums_to_one():
    """A map is a spatial probability distribution. That is what makes region
    mass comparable between frames, and a normalisation bug here would corrupt
    every score downstream while looking perfectly fine as a picture."""
    for frame in (frame_with_blob(), flat_frame(),
                  frame_with_blob(centre=(40, 40), radius=8)):
        result = sal.predict(frame)
        assert result["map"] is not None
        assert result["map_sum"] == pytest.approx(1.0, abs=1e-5)
        assert float(result["map"].sum()) == pytest.approx(1.0, abs=1e-5)


def test_a_flat_frame_produces_a_uniform_map_not_zeros(monkeypatch):
    """A solid colour card has nothing to find. Uniform is the honest answer;
    all-zeros would make every region mass zero and read as 'nothing was
    salient' rather than 'nothing was distinguishable'.

    This is a property of the CLASSICAL path specifically, so the backend is
    forced rather than left to the environment. Without that, importing app.py
    anywhere earlier in the suite calls load_dotenv(), which puts VL_MODEL_DIR
    into os.environ and quietly runs this assertion against a trained model that
    has no reason to return anything uniform.
    """
    monkeypatch.setattr(sal, "_load_onnx", lambda: None)
    result = sal.predict(flat_frame())
    flat_map = result["map"]
    assert flat_map.std() == pytest.approx(0.0, abs=1e-9)
    assert flat_map.sum() == pytest.approx(1.0, abs=1e-5)


def test_a_frame_with_one_subject_concentrates_more_than_noise(monkeypatch):
    monkeypatch.setattr(sal, "_load_onnx", lambda: None)
    focused = sal.predict(frame_with_blob())["concentration"]
    rng = np.random.default_rng(0)
    noisy = sal.predict({"t": 0.0, "index": 0,
                         "pixels": rng.integers(0, 255, (180, 320, 3),
                                                dtype=np.uint8)})["concentration"]
    assert focused > noisy


def test_region_mass_is_a_share_of_the_whole_map():
    result = sal.predict(frame_with_blob())
    whole = sal.region_mass(result["map"], [0.0, 0.0, 1.0, 1.0])
    corner = sal.region_mass(result["map"], [0.0, 0.0, 0.1, 0.1])
    assert whole == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= corner < whole
    # A box outside the frame is empty, not an exception.
    assert sal.region_mass(result["map"], [1.2, 1.2, 1.5, 1.5]) == 0.0


def test_peaks_are_fractional_so_the_frontend_can_scale_them():
    peaks = sal.predict(frame_with_blob())["peaks"]
    assert peaks
    for peak in peaks:
        x0, y0, x1, y1 = peak["box"]
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0
        assert 0.0 <= peak["share"] <= 1.0
    assert [p["rank"] for p in peaks] == list(range(1, len(peaks) + 1))


def test_the_backend_in_use_is_always_reported():
    """Nobody should have to guess whether a number came from a trained model or
    the classical baseline."""
    info = sal.describe()
    assert info["saliency_method"] in ("onnx", "spectral_residual")
    assert isinstance(info["saliency_trained"], bool)
    if not info["saliency_trained"]:
        assert "not a model trained on eye-tracking" in info["saliency_note"]


# =========================================================================== #
# Measurement
# =========================================================================== #
def test_an_empty_frame_is_flagged_so_it_cannot_become_the_hero():
    """A near-black end card drives the map degenerate and concentration reads
    1.0 - the highest possible score, from a frame with nothing in it."""
    empty = ms.frame_contrast(flat_frame()["pixels"])
    real = ms.frame_contrast(frame_with_blob()["pixels"])

    assert empty < 0.04
    assert real > empty
    assert ms.is_degenerate({"frame_contrast": empty}) is True
    assert ms.is_degenerate({"frame_contrast": real}) is False
    # Unknown contrast is not evidence of an empty frame.
    assert ms.is_degenerate({"frame_contrast": None}) is False


def test_motion_energy_is_none_on_the_first_frame():
    """There is nothing to compare against. Reporting 0 would claim a still
    opening we did not observe."""
    first = frame_with_blob()["pixels"]
    moved = frame_with_blob(centre=(200, 90))["pixels"]

    assert ms.motion_energy(first, None) is None
    assert ms.motion_energy(moved, first) > 0
    assert ms.motion_energy(first, first) == 0.0


def test_gaze_stability_is_one_for_an_identical_map():
    a = sal.predict(frame_with_blob())["map"]
    b = sal.predict(frame_with_blob(centre=(280, 150)))["map"]

    assert ms.gaze_stability(a, a) == pytest.approx(1.0, abs=1e-5)
    assert ms.gaze_stability(a, b) < 1.0
    assert ms.gaze_stability(a, None) is None


def test_shot_helpers_track_which_cut_we_are_in():
    shots = [0.0, 4.5, 9.0]
    assert ms.shot_index(1.0, shots) == 1
    assert ms.shot_index(5.0, shots) == 2
    assert ms.seconds_since_shot(5.0, shots) == 0.5
    assert ms.seconds_since_shot(1.0, []) is None


def test_measure_frames_records_every_frame_without_ffmpeg():
    media = {"frames": [frame_with_blob(), frame_with_blob(centre=(100, 60))],
             "shots": [0.0], "duration_seconds": 1.0, "sample_fps": 2.0}
    for index, frame in enumerate(media["frames"]):
        frame["index"] = index
        frame["t"] = index * 0.5

    records = ms.measure_frames(
        media,
        detect_regions=lambda frame, **kw: {
            "text_boxes": [], "word_count": 0, "text": "", "faces": [],
            "brand": None, "cta": None, "prices": [], "ocr_available": True},
    )

    assert len(records) == 2
    assert records[0]["motion_energy"] is None       # first frame
    assert records[1]["motion_energy"] is not None
    for record in records:
        assert record["saliency"]["map_sum"] == pytest.approx(1.0, abs=1e-5)
        assert record["frame_contrast"] is not None
        assert "mass" in record


def test_summarise_reports_reading_load_against_the_configured_speed():
    """63 words held for 2 seconds cannot be read at 4 words/second. That is the
    measurement the dead-zone finding is built on."""
    media = {"duration_seconds": 4.0, "sample_fps": 2.0, "frames": []}
    records = [
        {"index": i, "t": i * 0.5, "shot": 0,
         "saliency": {"concentration": 0.2},
         "regions": {"word_count": 63, "ocr_available": True, "brand": None,
                     "cta": None, "prices": [], "faces": 0}}
        for i in range(4)
    ]

    summary = ms.summarise(records, media)

    assert summary["max_words_on_screen"] == 63
    assert summary["reading_speed_words_per_second"] == 4.0
    overloaded = summary["overloaded_shots"]
    assert overloaded and overloaded[0]["words"] == 63
    assert overloaded[0]["seconds_needed"] > overloaded[0]["seconds_available"]


def test_summarise_reports_when_the_brand_never_appears():
    media = {"duration_seconds": 10.0, "sample_fps": 2.0, "frames": []}
    records = [{"index": 0, "t": 0.0, "shot": 0,
                "saliency": {"concentration": 0.2},
                "regions": {"word_count": 2, "ocr_available": True, "brand": None,
                            "cta": None, "prices": [], "faces": 0}}]

    brand = ms.summarise(records, media)["brand"]

    assert brand["detected"] is False
    assert brand["first_appearance_seconds"] is None
    assert brand["exposure_seconds"] == 0.0


# =========================================================================== #
# Model selection - Step 8 regression
# =========================================================================== #
def test_the_configured_model_file_is_the_one_loaded(tmp_path, monkeypatch):
    """VL_MODEL_DIR holds more than one .onnx - the face detector lives there
    too. Picking "first alphabetically" loaded face_detection_yunet.onnx and ran
    it as the saliency model, while describe() still reported UNISAL because the
    name comes from config. Only the digest check made it visible."""
    import importlib
    from vision_lab import framework as fw

    (tmp_path / "aaa_face_detector.onnx").write_bytes(b"not a saliency model")
    (tmp_path / "unisal.onnx").write_bytes(b"the real one")

    monkeypatch.setenv("VL_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(fw, "saliency_model",
                        lambda cfg=None: {"file": "unisal.onnx", "name": "UNISAL"})
    module = importlib.reload(sal)
    monkeypatch.setattr(module, "MODEL_DIR", str(tmp_path))

    captured = {}

    class FakeSession:
        def __init__(self, path, providers=None):
            captured["path"] = path
            raise RuntimeError("stop here - the filename is what matters")

    monkeypatch.setitem(sys.modules, "onnxruntime",
                        type("m", (), {"InferenceSession": FakeSession}))
    try:
        module._load_onnx()
    except RuntimeError:
        pass

    assert captured.get("path", "").endswith("unisal.onnx"), (
        "loaded the wrong file - it must follow saliency_model.file, not sort order")
    importlib.reload(sal)


def test_a_digest_mismatch_refuses_rather_than_warning(tmp_path, monkeypatch):
    """A file that is not the one the config describes would produce scores
    attributed to a model that never ran them. Falling back to the classical
    baseline is a stated degradation; running the wrong model silently is not."""
    import importlib
    from vision_lab import framework as fw

    (tmp_path / "unisal.onnx").write_bytes(b"contents that do not match the sha")
    monkeypatch.setattr(fw, "saliency_model", lambda cfg=None: {
        "file": "unisal.onnx", "name": "UNISAL", "sha256": "0" * 64})

    module = importlib.reload(sal)
    monkeypatch.setattr(module, "MODEL_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "onnxruntime",
                        type("m", (), {"InferenceSession": lambda *a, **k: object()}))

    assert module._load_onnx() is None
    importlib.reload(sal)


# =========================================================================== #
# Naming the attention peaks
# =========================================================================== #
from vision_lab import regions as rg  # noqa: E402


def test_a_peak_is_named_after_what_sits_under_it():
    """Coordinates say WHERE to draw the numbered marker on the Attention
    Report. The label says WHAT the eye went to, which is the part a marketer
    can act on - "41% of attention went to the headline", not "to a rectangle".
    """
    peaks = [{"rank": 1, "box": [0.20, 0.40, 0.70, 0.70], "share": 0.41},
             {"rank": 2, "box": [0.02, 0.05, 0.20, 0.25], "share": 0.17}]
    regions = {
        "brand": {"box": [0.04, 0.07, 0.16, 0.20], "name": "Director's Institute"},
        "cta": None,
        "faces": [],
        "text_boxes": [
            {"box": [0.30, 0.45, 0.45, 0.52], "text": "BOARD"},
            {"box": [0.47, 0.45, 0.65, 0.52], "text": "READINESS"},
            {"box": [0.80, 0.90, 0.95, 0.95], "text": "T&Cs"},   # outside both peaks
        ],
    }

    labelled = rg.label_peaks(peaks, regions)

    assert labelled[0]["element"] == rg.ELEMENT_TEXT
    # The words INSIDE the hotspot, in reading order - and only those.
    assert labelled[0]["element_text"] == "BOARD READINESS"
    assert labelled[1]["element"] == rg.ELEMENT_BRAND
    assert labelled[1]["element_text"] == "Director's Institute"


def test_the_brand_and_the_cta_beat_plain_text_on_the_same_pixels():
    """A wordmark and a call to action are both MADE of text, so whichever
    overlapped most would usually win and the answer would be "on_screen_text" -
    the least useful of the three. Specificity decides, not overlap."""
    peak = {"rank": 1, "box": [0.30, 0.80, 0.70, 0.92], "share": 0.30}
    regions = {
        "brand": None,
        "cta": {"box": [0.35, 0.82, 0.65, 0.90], "text": "Apply now"},
        "faces": [],
        # The OCR words that MAKE UP the CTA also sit inside the peak.
        "text_boxes": [{"box": [0.35, 0.82, 0.48, 0.90], "text": "Apply"},
                       {"box": [0.50, 0.82, 0.65, 0.90], "text": "now"}],
    }

    assert rg.label_peaks([peak], regions)[0]["element"] == rg.ELEMENT_CTA


def test_a_peak_over_nothing_we_detect_is_left_unnamed():
    """Product footage, b-roll and a face the detector missed are the common
    case. `None` is a more honest answer than labelling the hotspot after a
    caption that merely clipped its corner."""
    peak = {"rank": 1, "box": [0.10, 0.10, 0.50, 0.50], "share": 0.35}
    regions = {"brand": None, "cta": None, "faces": [],
               # Only 1/4 of this word box falls inside the peak.
               "text_boxes": [{"box": [0.45, 0.45, 0.65, 0.65], "text": "logo"}]}

    labelled = rg.label_peaks([peak], regions)
    assert labelled[0]["element"] is None
    assert labelled[0]["element_text"] is None


def test_a_brand_found_only_by_name_has_no_box_and_labels_nothing():
    """find_brand returns box=None when it matched the brand NAME in OCR text
    rather than the wordmark image. There is no geometry to label a peak with,
    and inventing one would put a marker on the wrong part of the frame."""
    peak = {"rank": 1, "box": [0.10, 0.10, 0.50, 0.50], "share": 0.35}
    regions = {"brand": {"box": None, "basis": "brand_name_in_text", "name": "DI"},
               "cta": None, "faces": [], "text_boxes": []}

    assert rg.label_peaks([peak], regions)[0]["element"] is None


def test_labelling_survives_a_frame_where_every_detector_came_back_empty():
    peaks = [{"rank": 1, "box": [0.1, 0.1, 0.5, 0.5], "share": 0.4}]
    for regions in ({}, None, {"brand": None, "cta": None, "faces": [],
                               "text_boxes": []}):
        assert rg.label_peaks(peaks, regions)[0]["element"] is None


def test_overlap_is_measured_against_the_element_not_the_hotspot():
    """A word box wholly inside a much larger peak is 100% overlapped, even
    though it covers a tiny share of the peak. Measuring it the other way round
    would leave every small element on a big hotspot unnamed."""
    word = [0.40, 0.40, 0.45, 0.44]
    peak = [0.10, 0.10, 0.90, 0.90]
    assert rg.overlap_fraction(word, peak) == 1.0
    assert rg.overlap_fraction(peak, word) < 0.01
    # Disjoint boxes overlap not at all, rather than raising.
    assert rg.overlap_fraction([0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6]) == 0.0


# =========================================================================== #
# The overlay itself
# =========================================================================== #
from vision_lab import heatmap as hm  # noqa: E402


def _frame_and_map(size=64):
    """A mid-grey frame with all the attention in one corner."""
    import numpy as np
    pixels = np.full((size, size, 3), 128, dtype=np.uint8)
    saliency = np.zeros((size, size), dtype=np.float32)
    saliency[4:16, 4:16] = 1.0
    return pixels, saliency / saliency.sum()


def test_the_overlay_leaves_unattended_pixels_alone():
    """REGRESSION, caught by looking at the rendered PNG rather than the code.

    The blend used one flat alpha for the whole frame, which paints the COLD end
    of the colour map over the ad as well as the hot end. JET's cold end is dark
    blue, so a real creative came back as flat lavender - the presenter, the
    office and the certificate on the wall were all invisible underneath their
    own attention report.

    An ignored region must keep the ad's real pixels.
    """
    import numpy as np
    pixels, saliency = _frame_and_map()

    rendered = hm.render(pixels, saliency)
    canvas = cv2.imdecode(np.frombuffer(rendered, np.uint8), cv2.IMREAD_COLOR)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    cold = canvas[50, 50].astype(int)          # far from the hotspot
    hot = canvas[10, 10].astype(int)           # inside it

    assert abs(cold - np.array([128, 128, 128])).max() <= 6, \
        f"an unattended pixel was repainted: {cold}"
    assert abs(hot - np.array([128, 128, 128])).max() > 40, \
        f"the attended pixel was not coloured: {hot}"


def test_colourise_returns_the_intensity_the_blend_needs():
    pixels, saliency = _frame_and_map()
    heat, intensity = hm.colourise(saliency, pixels.shape[:2])

    assert heat.shape == pixels.shape
    assert intensity.shape == pixels.shape[:2]
    assert 0.0 <= float(intensity.min()) and float(intensity.max()) <= 1.0
    # The maximum of the map is the maximum of the intensity, by construction.
    assert float(intensity.max()) == 1.0


def test_a_flat_map_renders_without_dividing_by_zero():
    """A degenerate frame - an all-black end card - can produce a map with no
    peak at all. It must render the ad untouched, not raise."""
    import numpy as np
    pixels = np.full((32, 32, 3), 200, dtype=np.uint8)
    rendered = hm.render(pixels, np.zeros((32, 32), dtype=np.float32))

    canvas = cv2.imdecode(np.frombuffer(rendered, np.uint8), cv2.IMREAD_COLOR)
    assert canvas is not None
    assert abs(int(canvas[16, 16][0]) - 200) <= 2


def test_a_transition_frame_is_not_chosen_as_the_hero():
    """REGRESSION, caught by looking at the rendered report image.

    A white flash between two shots has HIGH contrast - black letterbox bars
    over a blank body - so it passed is_degenerate, scored the best saliency
    concentration in the ad because there was nothing else to look at, and was
    picked as the hero. The Attention Report came back as a near-blank page with
    one marker on it.

    Measured on the real creative: the flash frame 0.029 detail, its neighbours
    0.063 and 0.078.
    """
    flash = {"frame_contrast": 0.4575, "frame_detail": 0.0294}
    scene = {"frame_contrast": 0.3556, "frame_detail": 0.0777}
    black = {"frame_contrast": 0.0000, "frame_detail": 0.0000}

    assert ms.is_representative(scene) is True
    assert ms.is_representative(flash) is False
    assert ms.is_representative(black) is False

    # is_degenerate is deliberately NOT tightened - the timeline uses it, and
    # moving it would move scores. It still sees the flash frame as measurable.
    assert ms.is_degenerate(flash) is False
    assert ms.is_degenerate(black) is True


def test_a_frame_measured_before_detail_existed_is_still_usable():
    """Records stored by an earlier version carry no `frame_detail`. They must
    stay eligible rather than silently excluding every frame of an old analysis
    from /rescore's heatmap selection."""
    assert ms.is_representative({"frame_contrast": 0.3}) is True
    assert ms.is_representative({"frame_contrast": 0.3, "frame_detail": None}) is True


def test_detail_separates_a_blank_frame_from_a_busy_one():
    """At the working resolution, not a toy one.

    Written first on a 120x120 frame, where two letterbox edges are a large
    share of the pixels and the "blank" frame measured 0.13 - three times the
    threshold. The measure is a MEAN over the frame, so a test image has to be
    the size of a real one or it models the wrong thing entirely.
    """
    import numpy as np
    blank = np.full((960, 640, 3), 250, dtype=np.uint8)   # the real working size
    blank[:180] = 0                                       # letterbox bars
    blank[840:] = 0
    busy = np.random.default_rng(7).integers(0, 255, (960, 640, 3), dtype=np.uint8)

    # The blank frame has high CONTRAST from its bars...
    assert ms.frame_contrast(blank) > 0.3
    # ...and almost no DETAIL, which is what tells them apart.
    assert ms.frame_detail(blank) < ms.MIN_HERO_DETAIL
    assert ms.frame_detail(busy) > ms.MIN_HERO_DETAIL
