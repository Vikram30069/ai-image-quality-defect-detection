"""
Model Evaluation Module
-----------------------
Evaluates the trained Random Forest classifier on the independent test dataset.
Computes real, un-fabricated metrics:
- Overall Accuracy
- Precision, Recall, and F1-Score (Macro and Per-Class)
- Confusion Matrix
- Feature Importance Ranking
- Detailed Error / Misclassification Breakdown

Saves artifacts to:
- ml/experiments/evaluation_report.json
- ml/experiments/confusion_matrix.txt
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.core.feature_extractor import FEATURE_NAMES


def evaluate_model():
    model_path = WORKSPACE_ROOT / "backend" / "app" / "ml" / "quality_model.pkl"
    test_csv = WORKSPACE_ROOT / "ml" / "dataset" / "test_features.csv"
    exp_dir = WORKSPACE_ROOT / "ml" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run ml/train.py first.")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test features not found at {test_csv}. Run ml/extract_features.py first.")

    # Load Model and Test Data
    model = joblib.load(model_path)
    df_test = pd.read_csv(test_csv)

    X_test = df_test[FEATURE_NAMES].values
    y_test = df_test["label"].values

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate Metrics
    classes = list(model.classes_)
    accuracy = float(accuracy_score(y_test, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    
    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, average=None
    )

    conf_mat = confusion_matrix(y_test, y_pred, labels=classes)
    cls_report = classification_report(y_test, y_pred, labels=classes, output_dict=True)

    # Feature Importances
    feature_importances = {
        name: round(float(imp), 4)
        for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: x[1], reverse=True)
    }

    # Inspect Misclassifications
    misclassifications = []
    for idx, (true_lbl, pred_lbl) in enumerate(zip(y_test, y_pred)):
        if true_lbl != pred_lbl:
            misclassifications.append({
                "filename": df_test.iloc[idx]["filename"],
                "true_label": true_lbl,
                "predicted_label": pred_lbl,
                "confidence": round(float(np.max(y_proba[idx])), 4),
                "features": {feat: round(float(df_test.iloc[idx][feat]), 4) for feat in FEATURE_NAMES}
            })

    # Prepare Evaluation Report
    report_dict = {
        "dataset": "Independent Procedural Test Split",
        "num_test_samples": len(df_test),
        "overall_accuracy": round(accuracy, 4),
        "macro_metrics": {
            "precision": round(float(prec_macro), 4),
            "recall": round(float(rec_macro), 4),
            "f1_score": round(float(f1_macro), 4)
        },
        "per_class_metrics": {
            cls: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1_score": round(float(f), 4),
                "support": int(s)
            }
            for cls, p, r, f, s in zip(classes, prec_per_class, rec_per_class, f1_per_class, support_per_class)
        },
        "confusion_matrix": {
            "classes": classes,
            "matrix": conf_mat.tolist()
        },
        "feature_importances": feature_importances,
        "num_misclassifications": len(misclassifications),
        "misclassifications": misclassifications
    }

    # Save to JSON
    json_path = exp_dir / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # Save formatted confusion matrix text
    txt_path = exp_dir / "confusion_matrix.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Model Evaluation: Confusion Matrix ===\n\n")
        f.write(f"Labels: {classes}\n\n")
        f.write(f"{'True \\ Pred':<15}" + "".join([f"{c:<15}" for c in classes]) + "\n")
        f.write("-" * (15 + 15 * len(classes)) + "\n")
        for i, true_cls in enumerate(classes):
            row_str = f"{true_cls:<15}" + "".join([f"{conf_mat[i][j]:<15}" for j in range(len(classes))])
            f.write(row_str + "\n")
        f.write("\n\n=== Classification Report ===\n")
        f.write(classification_report(y_test, y_pred, labels=classes))

    print(f"\n=======================================================")
    print(f"Model Evaluation Complete on {len(df_test)} Unseen Test Samples")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Macro F1:  {f1_macro:.4f}")
    print(f"Report saved to: {json_path}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    evaluate_model()
