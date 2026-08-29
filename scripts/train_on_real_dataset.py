"""
Train Model on Real Industrial Dataset
---------------------------------------
Extracts 7 numerical image quality features from the real dataset in dataset/train
and dataset/test, trains a RandomForestClassifier, evaluates on unseen test data,
and saves the serialized model to backend/app/ml/quality_model.pkl.
"""

import os
import glob
import json
import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.core.preprocessor import ImagePreprocessor
from backend.app.core.feature_extractor import FeatureExtractor, FEATURE_NAMES

def extract_features_from_dataset(folder_path: str, max_samples: int = 500) -> pd.DataFrame:
    prep = ImagePreprocessor()
    ext = FeatureExtractor()
    
    gt_files = sorted(glob.glob(os.path.join(folder_path, "*_GT.png")))
    rows = []
    
    for gt in gt_files[:max_samples]:
        img_p = gt.replace("_GT.png", ".png")
        if not os.path.exists(img_p):
            continue
            
        mask = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
        has_defect = (mask is not None and np.sum(mask > 0) > 0)
        label = "DEFECTIVE" if has_defect else "ACCEPTABLE"
        
        try:
            p = prep.load_from_path(img_p)
            f_dict, _ = ext.extract_features(p)
            row = {"filename": os.path.basename(img_p), "label": label}
            row.update(f_dict)
            rows.append(row)
            
            # Add degraded variants (lighting / blur / noise) to ensure balanced 3-class training
            if not has_defect and len(rows) % 4 == 0:
                dark_bgr = (p.bgr * 0.35).astype(np.uint8)
                p_dark = prep.load_from_array(dark_bgr) if hasattr(prep, 'load_from_array') else None
                if p_dark is None:
                    tmp_p = WORKSPACE_ROOT / "uploads" / f"tmp_deg_{os.path.basename(img_p)}"
                    cv2.imwrite(str(tmp_p), dark_bgr)
                    p_dark = prep.load_from_path(str(tmp_p))
                    if tmp_p.exists(): tmp_p.unlink()

                f_dark, _ = ext.extract_features(p_dark)
                r_dark = {"filename": "dark_" + os.path.basename(img_p), "label": "DEGRADED"}
                r_dark.update(f_dark)
                rows.append(r_dark)
        except Exception as e:
            print(f"Skipping {img_p}: {e}")
            
    return pd.DataFrame(rows)

def main():
    print("Extracting features from dataset/train...")
    train_df = extract_features_from_dataset("dataset/train", max_samples=600)
    
    print("Extracting features from dataset/test...")
    test_df = extract_features_from_dataset("dataset/test", max_samples=200)
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Train class distribution:\n{train_df['label'].value_counts()}")
    
    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df["label"].values
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print("\n--- Real Dataset Unseen Test Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    # Save model artifact
    target_path = WORKSPACE_ROOT / "backend" / "app" / "ml" / "quality_model.pkl"
    joblib.dump(rf, str(target_path))
    print(f"Model saved to: {target_path}")
    
    # Save metadata
    meta_path = WORKSPACE_ROOT / "backend" / "app" / "ml" / "model_metadata.json"
    meta = {
        "model_name": "RandomForestQualityClassifier",
        "dataset_source": "Kolektor Surface Defect Dataset (Real Industrial Workpieces)",
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "feature_names": FEATURE_NAMES,
        "classes": sorted(list(rf.classes_))
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Metadata saved successfully.")

if __name__ == "__main__":
    main()
