"""
Unit Tests for Preprocessor, Feature Extractor, and Quality Analyzer
---------------------------------------------------------------------
Validates image decoding, mathematical correctness of quality metrics,
and deterministic issue classification against known synthetic patterns.
"""

import pytest
import cv2
import numpy as np

from backend.app.core.preprocessor import ImagePreprocessor, PreprocessedImage
from backend.app.core.feature_extractor import FeatureExtractor, FEATURE_NAMES
from backend.app.core.quality_analyzer import QualityAnalyzer


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


@pytest.fixture
def extractor():
    return FeatureExtractor()


@pytest.fixture
def analyzer():
    return QualityAnalyzer()


def create_synthetic_image_bytes(pattern_type: str = "checkerboard") -> bytes:
    """Helper to generate specific synthetic image byte streams for testing."""
    h, w = 200, 200
    if pattern_type == "checkerboard":
        # Sharp high-contrast checkerboard pattern
        img = np.zeros((h, w, 3), dtype=np.uint8)
        block = 20
        for y in range(0, h, block):
            for x in range(0, w, block):
                if (x // block + y // block) % 2 == 0:
                    img[y:y+block, x:x+block] = 255
    elif pattern_type == "black":
        img = np.zeros((h, w, 3), dtype=np.uint8)
    elif pattern_type == "white":
        img = np.full((h, w, 3), 255, dtype=np.uint8)
    elif pattern_type == "gray":
        img = np.full((h, w, 3), 128, dtype=np.uint8)
    elif pattern_type == "blurred":
        # Create checkerboard and apply heavy Gaussian blur
        base = np.zeros((h, w, 3), dtype=np.uint8)
        block = 20
        for y in range(0, h, block):
            for x in range(0, w, block):
                if (x // block + y // block) % 2 == 0:
                    base[y:y+block, x:x+block] = 255
        img = cv2.GaussianBlur(base, (45, 45), 15.0)
    elif pattern_type == "noisy":
        # Uniform base with Gaussian noise
        base = np.full((h, w, 3), 128, dtype=np.uint8)
        noise = np.random.normal(0, 25, (h, w, 3)).astype(np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    else:
        img = np.full((h, w, 3), 128, dtype=np.uint8)

    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


# --------------------------------------------------------------------
# 1. Preprocessor Tests
# --------------------------------------------------------------------

def test_preprocessor_valid_image(preprocessor):
    raw_bytes = create_synthetic_image_bytes("checkerboard")
    prep = preprocessor.validate_and_decode(raw_bytes)
    
    assert isinstance(prep, PreprocessedImage)
    assert prep.height == 200
    assert prep.width == 200
    assert prep.channels == 3
    assert prep.gray.shape == (200, 200)
    assert prep.hsv.shape == (200, 200, 3)
    assert prep.lab.shape == (200, 200, 3)
    assert not prep.was_resized


def test_preprocessor_empty_bytes(preprocessor):
    with pytest.raises(ValueError, match="Empty image buffer"):
        preprocessor.validate_and_decode(b"")


def test_preprocessor_corrupt_bytes(preprocessor):
    with pytest.raises(ValueError, match="Image decoding failed"):
        preprocessor.validate_and_decode(b"NOT_A_REAL_IMAGE_BYTES_12345")


# --------------------------------------------------------------------
# 2. Feature Extractor Tests
# --------------------------------------------------------------------

def test_feature_vector_structure(preprocessor, extractor):
    raw_bytes = create_synthetic_image_bytes("checkerboard")
    prep = preprocessor.validate_and_decode(raw_bytes)
    feat_dict, feat_vector = extractor.extract_features(prep)

    assert len(feat_vector) == len(FEATURE_NAMES)
    assert len(feat_dict) == 7
    for name in FEATURE_NAMES:
        assert name in feat_dict
        assert isinstance(feat_dict[name], (int, float))
        assert not np.isnan(feat_dict[name])


def test_sharpness_blur_distinction(preprocessor, extractor):
    """Sharp checkerboard must produce significantly higher Laplacian variance than blurred checkerboard."""
    sharp_prep = preprocessor.validate_and_decode(create_synthetic_image_bytes("checkerboard"))
    blur_prep = preprocessor.validate_and_decode(create_synthetic_image_bytes("blurred"))

    sharp_dict, _ = extractor.extract_features(sharp_prep)
    blur_dict, _ = extractor.extract_features(blur_prep)

    assert sharp_dict["sharpness"] > blur_dict["sharpness"] * 20
    assert blur_dict["sharpness"] < 50.0  # Heavy blur should drop well below 50


def test_exposure_detection(preprocessor, extractor):
    dark_prep = preprocessor.validate_and_decode(create_synthetic_image_bytes("black"))
    bright_prep = preprocessor.validate_and_decode(create_synthetic_image_bytes("white"))

    dark_dict, _ = extractor.extract_features(dark_prep)
    bright_dict, _ = extractor.extract_features(bright_prep)

    assert dark_dict["brightness"] == 0.0
    assert bright_dict["brightness"] == 255.0


def test_noise_estimation(preprocessor, extractor):
    clean_prep = preprocessor.validate_and_decode(create_synthetic_image_bytes("gray"))
    noisy_prep = preprocessor.validate_and_decode(create_synthetic_image_bytes("noisy"))

    clean_dict, _ = extractor.extract_features(clean_prep)
    noisy_dict, _ = extractor.extract_features(noisy_prep)

    assert clean_dict["noise"] == 0.0
    assert noisy_dict["noise"] > 8.5  # Exceeds the high noise threshold of 8.5


# --------------------------------------------------------------------
# 3. Quality Analyzer Tests
# --------------------------------------------------------------------

def test_quality_analyzer_blur_issue(preprocessor, extractor, analyzer):
    blur_prep = preprocessor.validate_and_decode(create_synthetic_image_bytes("blurred"))
    blur_dict, _ = extractor.extract_features(blur_prep)
    result = analyzer.analyze(blur_dict)

    issue_types = [issue.type for issue in result.issues]
    assert "blur" in issue_types
    assert result.quality_score < 70.0


def test_quality_analyzer_underexposed(preprocessor, extractor, analyzer):
    dark_prep = preprocessor.validate_and_decode(create_synthetic_image_bytes("black"))
    dark_dict, _ = extractor.extract_features(dark_prep)
    result = analyzer.analyze(dark_dict)

    issue_types = [issue.type for issue in result.issues]
    assert "underexposure" in issue_types
    assert result.quality_label == "DEFECTIVE"
