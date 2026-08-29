"""
Batch Feature Extraction Script
-------------------------------
Extracts quality features from the train and test dataset splits using the exact
same ImagePreprocessor and FeatureExtractor pipeline used by the production API.

Saves:
- ml/dataset/train_features.csv
- ml/dataset/test_features.csv
"""

import sys
from pathlib import Path
import csv

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.core.preprocessor import ImagePreprocessor
from backend.app.core.feature_extractor import FeatureExtractor, FEATURE_NAMES


def extract_split_features(image_dir: Path, output_csv: Path):
    """
    Processes all PNG/JPG images in image_dir, extracts features,
    and writes them to output_csv with class labels.
    """
    preprocessor = ImagePreprocessor()
    extractor = FeatureExtractor()

    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    if not image_files:
        print(f"Warning: No images found in {image_dir}")
        return

    fieldnames = ["filename"] + FEATURE_NAMES + ["label"]
    rows = []

    for img_path in image_files:
        # Determine ground truth label from filename convention
        fname = img_path.name.lower()
        if "acceptable" in fname:
            label = "ACCEPTABLE"
        elif "degraded" in fname:
            label = "DEGRADED"
        elif "defective" in fname:
            label = "DEFECTIVE"
        else:
            print(f"Skipping unclassified file: {fname}")
            continue

        try:
            prep = preprocessor.load_from_path(str(img_path))
            feat_dict, _ = extractor.extract_features(prep)

            row = {"filename": img_path.name, "label": label}
            row.update(feat_dict)
            rows.append(row)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} feature records to {output_csv}")


if __name__ == "__main__":
    dataset_dir = WORKSPACE_ROOT / "ml" / "dataset"
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    print("Extracting features from Training Set...")
    extract_split_features(train_dir, dataset_dir / "train_features.csv")

    print("Extracting features from Test Set...")
    extract_split_features(test_dir, dataset_dir / "test_features.csv")
    print("Feature extraction completed successfully.")
