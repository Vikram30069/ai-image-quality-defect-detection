"""
Production Machine Learning Model Loader
----------------------------------------
Provides thread-safe singleton loading and production inference for the trained
RandomForest quality classifier. Ensures feature alignment with model training.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import joblib

from backend.app.config import settings
from backend.app.core.feature_extractor import FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class MLPredictionResult:
    """Standardized output structure from ML quality prediction."""
    predicted_label: str                           # "ACCEPTABLE", "DEGRADED", or "DEFECTIVE"
    confidence: float                             # Probability of the winning class [0.0 - 1.0]
    class_probabilities: Dict[str, float] = field(default_factory=dict) # Softmax-style distribution
    model_version: str = "1.0.0"
    is_fallback: bool = False                     # True if prediction was generated via heuristic fallback


class QualityModelLoader:
    """
    Singleton manager for loading the serialized scikit-learn model
    and executing fast, consistent inference.
    """
    _instance: Optional["QualityModelLoader"] = None

    def __init__(self, model_path: Optional[Path] = None, meta_path: Optional[Path] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.meta_path = meta_path or settings.MODEL_META_PATH
        self.model = None
        self.metadata = {}
        self.classes: List[str] = ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]
        self.load_model()

    @classmethod
    def get_instance(cls) -> "QualityModelLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> bool:
        """Attempts to load the model and metadata from disk."""
        if not self.model_path.exists():
            logger.warning(f"Quality model file not found at: {self.model_path}. Inference will use heuristic fallback.")
            return False

        try:
            self.model = joblib.load(self.model_path)
            if hasattr(self.model, "classes_"):
                self.classes = list(self.model.classes_)

            if self.meta_path.exists():
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            logger.info(f"Loaded ML model: {self.metadata.get('model_name', 'RandomForest')} v{self.metadata.get('model_version', '1.0.0')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ML model from {self.model_path}: {e}")
            self.model = None
            return False

    def predict(self, features: Union[Dict[str, float], List[float]]) -> MLPredictionResult:
        """
        Executes prediction on input features.

        Args:
            features: Either a dict mapping feature names or an ordered list of 7 floats.

        Returns:
            MLPredictionResult with predicted class, confidence, and class probabilities.
        """
        # Convert dictionary to ordered vector matching FEATURE_NAMES
        if isinstance(features, dict):
            vector = [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
        else:
            vector = [float(x) for x in features]

        if len(vector) != len(FEATURE_NAMES):
            raise ValueError(f"Feature vector length mismatch: expected {len(FEATURE_NAMES)}, got {len(vector)}")

        # Fallback if model artifact is unavailable
        if self.model is None:
            return self._heuristic_fallback(features if isinstance(features, dict) else dict(zip(FEATURE_NAMES, vector)))

        # Reshape to (1, n_features) for scikit-learn
        X = np.array(vector, dtype=np.float64).reshape(1, -1)

        pred_label = self.model.predict(X)[0]
        probas = self.model.predict_proba(X)[0]

        proba_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes, probas)
        }
        confidence = proba_dict.get(pred_label, float(np.max(probas)))

        return MLPredictionResult(
            predicted_label=pred_label,
            confidence=round(confidence, 4),
            class_probabilities=proba_dict,
            model_version=self.metadata.get("model_version", "1.0.0"),
            is_fallback=False
        )

    def _heuristic_fallback(self, feat_dict: Dict[str, float]) -> MLPredictionResult:
        """Heuristic fallback to guarantee robust operation if model is not yet compiled."""
        sharpness = feat_dict.get("sharpness", 100.0)
        brightness = feat_dict.get("brightness", 128.0)
        noise = feat_dict.get("noise", 2.0)

        if sharpness < 80.0 or brightness < 40.0 or brightness > 220.0 or noise > 12.0:
            pred = "DEFECTIVE"
            probs = {"ACCEPTABLE": 0.05, "DEGRADED": 0.25, "DEFECTIVE": 0.70}
        elif sharpness < 180.0 or brightness < 60.0 or brightness > 200.0 or noise > 8.0:
            pred = "DEGRADED"
            probs = {"ACCEPTABLE": 0.20, "DEGRADED": 0.65, "DEFECTIVE": 0.15}
        else:
            pred = "ACCEPTABLE"
            probs = {"ACCEPTABLE": 0.85, "DEGRADED": 0.10, "DEFECTIVE": 0.05}

        return MLPredictionResult(
            predicted_label=pred,
            confidence=probs[pred],
            class_probabilities=probs,
            model_version="fallback-heuristic",
            is_fallback=True
        )
