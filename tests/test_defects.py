"""
Unit Tests for Defect Detection Pipeline
----------------------------------------
Tests morphological anomaly segmentation, contour shape filtering,
bounding box calculations, and overlay visual synthesis.
"""

import pytest
import cv2
import numpy as np

from backend.app.core.preprocessor import ImagePreprocessor
from backend.app.core.defect_detector import DefectDetector, DefectDetectionResult


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


@pytest.fixture
def detector():
    return DefectDetector()


def create_image_with_synthetic_defect(defect_type: str) -> bytes:
    """Creates a base uniform metal surface image with a controlled anomaly."""
    h, w = 300, 300
    # Base metallic gray surface with slight texture
    base = np.full((h, w, 3), 180, dtype=np.uint8)

    if defect_type == "clean":
        pass  # Pristine
    elif defect_type == "scratch":
        # Draw a thin dark elongated scratch (high aspect ratio)
        cv2.line(base, (40, 150), (260, 160), (30, 30, 30), thickness=2)
    elif defect_type == "blemish":
        # Draw a dark circular spot (high circularity)
        cv2.circle(base, (150, 150), 12, (20, 20, 20), -1)
    elif defect_type == "crack":
        # Draw a jagged crack-like series of connected line segments
        pts = np.array([[50, 50], [90, 100], [110, 140], [160, 180], [200, 240]], np.int32)
        cv2.polylines(base, [pts], isClosed=False, color=(10, 10, 10), thickness=3)

    _, enc = cv2.imencode(".png", base)
    return enc.tobytes()


def test_clean_image_produces_zero_defects(preprocessor, detector):
    """Pristine surface should produce zero defect candidates."""
    raw = create_image_with_synthetic_defect("clean")
    prep = preprocessor.validate_and_decode(raw)
    result = detector.detect(prep)

    assert isinstance(result, DefectDetectionResult)
    assert result.total_defects == 0
    assert len(result.defects) == 0
    assert result.defect_density == 0.0


def test_scratch_detection(preprocessor, detector):
    """Elongated scratch must be detected and classified with high aspect ratio."""
    raw = create_image_with_synthetic_defect("scratch")
    prep = preprocessor.validate_and_decode(raw)
    result = detector.detect(prep)

    assert result.total_defects >= 1
    types = [d.defect_type for d in result.defects]
    assert "SCRATCH" in types

    # Check bounding box validity
    scratch = next(d for d in result.defects if d.defect_type == "SCRATCH")
    assert scratch.bounding_box.width > 100 or scratch.bounding_box.height > 100
    assert scratch.aspect_ratio >= 3.0


def test_blemish_detection(preprocessor, detector):
    """Circular dark spot should be classified as blemish with high circularity."""
    raw = create_image_with_synthetic_defect("blemish")
    prep = preprocessor.validate_and_decode(raw)
    result = detector.detect(prep)

    assert result.total_defects >= 1
    types = [d.defect_type for d in result.defects]
    assert "BLEMISH" in types

    blemish = next(d for d in result.defects if d.defect_type == "BLEMISH")
    assert blemish.circularity > 0.5


def test_annotated_and_heatmap_dimensions(preprocessor, detector):
    """Overlays must match original image dimensions."""
    raw = create_image_with_synthetic_defect("scratch")
    prep = preprocessor.validate_and_decode(raw)
    result = detector.detect(prep)

    assert result.annotated_image.shape == (300, 300, 3)
    assert result.heatmap_bgr.shape == (300, 300, 3)
    assert result.defect_mask.shape == (300, 300)
