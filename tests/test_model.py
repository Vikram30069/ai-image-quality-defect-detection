"""
Unit Tests for Machine Learning Model Loader and Inference
----------------------------------------------------------
Validates model serialization integrity, feature vector alignment,
prediction probabilities, and graceful fallback mechanisms.
"""

import pytest
import numpy as np
from pathlib import Path

from backend.app.ml.model_loader import QualityModelLoader, MLPredictionResult
from backend.app.core.feature_extractor import FEATURE_NAMES


@pytest.fixture
def model_loader():
    return QualityModelLoader.get_instance()


def test_model_loads_successfully(model_loader):
    """Verifies that the Random Forest model and metadata are loaded from disk."""
    assert model_loader.model is not None
    assert set(model_loader.classes) == {"ACCEPTABLE", "DEGRADED", "DEFECTIVE"}
    assert model_loader.metadata.get("algorithm") == "RandomForestClassifier"


def test_prediction_output_structure(model_loader):
    """Verifies that predict returns an MLPredictionResult with valid probabilities."""
    # Synthetic feature dictionary
    sample_features = {
        "sharpness": 450.0,
        "brightness": 125.0,
        "contrast": 55.0,
        "noise": 1.2,
        "entropy": 7.2,
        "saturation": 0.65,
        "edge_density": 0.08
    }

    result = model_loader.predict(sample_features)

    assert isinstance(result, MLPredictionResult)
    assert result.predicted_label in ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.class_probabilities) == 3

    # Probabilities should sum to approximately 1.0
    prob_sum = sum(result.class_probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-3
    assert not result.is_fallback


def test_prediction_with_ordered_list(model_loader):
    """Verifies that predict accepts an ordered feature vector list."""
    # Ordered vector: [sharpness, brightness, contrast, noise, entropy, saturation, edge_density]
    vector = [12.0, 30.0, 15.0, 22.0, 4.0, 0.2, 0.001]  # Severe defect pattern
    result = model_loader.predict(vector)

    assert isinstance(result, MLPredictionResult)
    assert result.predicted_label == "DEFECTIVE"
    assert result.confidence > 0.5


def test_invalid_feature_length_raises_error(model_loader):
    """Verifies that passing a feature vector with wrong dimensions raises ValueError."""
    short_vector = [100.0, 50.0, 20.0]  # Only 3 features instead of 7
    with pytest.raises(ValueError, match="Feature vector length mismatch"):
        model_loader.predict(short_vector)


def test_heuristic_fallback_when_model_missing(tmp_path):
    """Verifies graceful fallback behavior when model file does not exist."""
    fake_path = tmp_path / "non_existent_model.pkl"
    fallback_loader = QualityModelLoader(model_path=fake_path, meta_path=fake_path)

    assert fallback_loader.model is None
    sample_features = {"sharpness": 15.0, "brightness": 20.0, "noise": 15.0}
    res = fallback_loader.predict(sample_features)

    assert res.is_fallback is True
    assert res.predicted_label == "DEFECTIVE"
    assert res.model_version == "fallback-heuristic"
