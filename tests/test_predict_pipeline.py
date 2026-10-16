"""
Tests for the inference pipeline (src/pipeline/predict_pipeline.py).

Needs the artifacts produced by `dvc repro`: models/memory_bank.pt and
models/threshold.json. Expected scores come from the notebook heatmaps
(first image of each group); single-image scoring can differ slightly from
batch scoring, so we allow a small tolerance.

Run:  python -m pytest tests/test_predict_pipeline.py -v
"""

import io
import json
import numpy as np
import pytest
from PIL import Image

from src.pipeline.predict_pipeline import DefectPredictor
from src.config import PROJECT_ROOT, MODEL_DIR
from src.constants import MEMORY_BANK_FILE, THRESHOLD_FILE
from src.exception import InvalidInputError

TEST_DIR = PROJECT_ROOT / "data" / "raw" / "metal_nut" / "test"

# (defect type, expected score, expected to be flagged?)
# NOTE: expected scores are from the reference notebook. If your machine
# produces slightly different values, adjust with the tolerance below or
# replace with the actual scores from a single predictor.predict() run.

CASES = [
    ("good",    1.01, False),
    ("bent",    1.69, True),
    ("color",   1.44, True),
    ("flip",    1.84, True),
    ("scratch", 1.28, True),
]
SCORE_TOLERANCE = 0.05   # generous enough for float drift and resize differences

needs_artifacts = pytest.mark.skipif(
    not (
        (MODEL_DIR / MEMORY_BANK_FILE).exists()
        and (MODEL_DIR / THRESHOLD_FILE).exists()
        and TEST_DIR.exists()
    ),
    reason="Run `dvc repro` first: memory bank, threshold, and dataset are required.",
)
pytestmark = needs_artifacts

@pytest.fixture(scope="module")
def predictor():
    """Load the model once for the whole file (loading takes a few seconds)."""
    return DefectPredictor()


# ---------------------------------------------------------------- correct predictions
@pytest.mark.parametrize("defect_type, expected_score, expected_defective", CASES)
def test_known_images(predictor, defect_type, expected_score, expected_defective):
    """Score and decision of the five reference images match the notebook results."""
    result = predictor.predict(TEST_DIR / defect_type / "000.png")
    assert result["score"] == pytest.approx(expected_score, abs=SCORE_TOLERANCE)
    assert result["is_defective"] is expected_defective
    assert result["label"] == ("DEFECTIVE" if expected_defective else "NORMAL")

def test_threshold_matches_file(predictor):
    """The predictor uses exactly the threshold saved by the calibration stage."""
    saved = json.load(open(MODEL_DIR / THRESHOLD_FILE))["threshold"]
    assert predictor.threshold == pytest.approx(saved)
    assert predictor.threshold == pytest.approx(1.1666, abs=1e-3)


def test_prediction_is_deterministic(predictor):
    """The same image gives the same score twice."""
    path = TEST_DIR / "bent" / "000.png"
    assert predictor.predict(path)["score"] == pytest.approx(
        predictor.predict(path)["score"], abs=1e-5
    )

def test_file_object_equals_path(predictor):
    """The API passes uploaded files as file objects; result must equal path result."""
    path = TEST_DIR / "color" / "000.png"
    from_path = predictor.predict(path)["score"]
    from_buffer = predictor.predict(io.BytesIO(path.read_bytes()))["score"]
    assert from_buffer == pytest.approx(from_path, abs=1e-5)

# ---------------------------------------------------------------- output format
def test_output_types_and_shapes(predictor):
    result = predictor.predict(TEST_DIR / "scratch" / "000.png")

    assert set(result) >= {
        "score", "threshold", "ratio", "is_defective", "label",
        "anomaly_map", "mask", "overlay", "latency_ms",
    }
    assert isinstance(result["score"], float)
    assert result["anomaly_map"].shape == (256, 256)
    assert result["mask"].shape == (256, 256) and result["mask"].dtype == bool
    assert isinstance(result["overlay"], Image.Image) and result["overlay"].size == (256, 256)
    assert result["ratio"] == pytest.approx(result["score"] / result["threshold"])
    assert result["latency_ms"] > 0
    assert np.isfinite(result["anomaly_map"]).all()

def test_anomaly_map_invariants(predictor):
    """Anomaly distances are non-negative, and the image score is the map's maximum."""
    result = predictor.predict(TEST_DIR / "bent" / "000.png")
    amap = result["anomaly_map"]
    assert amap.min() >= 0, "Anomaly distances must be non-negative"
    assert amap.max() == pytest.approx(result["score"], abs=1e-5)


def test_mask_follows_threshold(predictor):
    """Normal images have (almost) empty masks; defective ones have visible masks."""
    normal = predictor.predict(TEST_DIR / "good" / "000.png")
    defect = predictor.predict(TEST_DIR / "bent" / "000.png")
    assert normal["mask"].sum() == 0
    assert defect["mask"].sum() > 100


def test_works_on_other_image_sizes(predictor, tmp_path):
    """Images of any size are resized; output grid is always 256x256."""
    original_path = TEST_DIR / "good" / "000.png"
    original_score = predictor.predict(original_path)["score"]

    src = Image.open(original_path)
    small = tmp_path / "small.png"
    src.resize((300, 300)).save(small)
    result = predictor.predict(small)

    assert result["anomaly_map"].shape == (256, 256)
    # Different resize path → allow larger relative tolerance
    assert result["score"] == pytest.approx(original_score, rel=0.15)


# ---------------------------------------------------------------- invalid input
def test_text_file_is_rejected(predictor, tmp_path):
    bad = tmp_path / "notes.png"                  # wrong content under an image name
    bad.write_text("this is not an image")
    with pytest.raises(InvalidInputError) as err:
        predictor.predict(bad)
    assert str(tmp_path) not in str(err.value)    # message must not leak server paths


def test_empty_file_is_rejected(predictor, tmp_path):
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    with pytest.raises(InvalidInputError):
        predictor.predict(empty)

def test_truncated_image_is_rejected(predictor, tmp_path):
    data = (TEST_DIR / "good" / "000.png").read_bytes()
    broken = tmp_path / "broken.png"
    broken.write_bytes(data[: len(data) // 3])    # cut the file
    with pytest.raises(InvalidInputError):
        predictor.predict(broken)


def test_tiny_image_is_rejected(predictor, tmp_path):
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (16, 16)).save(tiny)
    with pytest.raises(InvalidInputError, match="too small"):
        predictor.predict(tiny)


def test_invalid_input_error_is_user_safe():
    """InvalidInputError is a plain ValueError with a path-free message."""
    err = InvalidInputError("The uploaded file is not a readable image.")
    assert isinstance(err, ValueError)
    assert str(err) == "The uploaded file is not a readable image."