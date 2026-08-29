"""
Model Training Module
---------------------
Trains a Scikit-Learn RandomForestClassifier on extracted numerical image features.
Performs stratified k-fold cross-validation, saves the serialized model using joblib,
and writes comprehensive metadata for production tracking.
"""

import sys
import json
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.core.feature_extractor import FEATURE_NAMES


def train_model():
    dataset_csv = WORKSPACE_ROOT / "ml" / "dataset" / "train_features.csv"
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Training features not found at {dataset_csv}. Run generate_dataset.py and extract_features.py first.")

    df = pd.read_csv(dataset_csv)
    print(f"Loaded {len(df)} training samples across classes:\n{df['label'].value_counts()}")

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    # Model Hyperparameters
    hyperparams = {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "random_state": 42
    }

    rf = RandomForestClassifier(**hyperparams)

    # 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Fit final model on all training data
    rf.fit(X, y)

    # Prepare export paths
    target_model_path = WORKSPACE_ROOT / "backend" / "app" / "ml" / "quality_model.pkl"
    target_meta_path = WORKSPACE_ROOT / "backend" / "app" / "ml" / "model_metadata.json"

    target_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model artifact
    joblib.dump(rf, target_model_path)
    print(f"Trained model saved to: {target_model_path}")

    # Save training metadata
    metadata = {
        "model_name": "RandomForestQualityClassifier",
        "model_version": "1.0.0",
        "algorithm": "RandomForestClassifier",
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "class_labels": sorted(list(rf.classes_)),
        "num_training_samples": len(df),
        "class_distribution": df["label"].value_counts().to_dict(),
        "cross_val_accuracy_mean": round(float(np.mean(cv_scores)), 4),
        "cross_val_accuracy_std": round(float(np.std(cv_scores)), 4),
        "hyperparameters": hyperparams
    }

    with open(target_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model metadata saved to: {target_meta_path}")


if __name__ == "__main__":
    train_model()
