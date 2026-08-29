"""
Database Initialization & Seeding Script
----------------------------------------
Initializes the SQLite schema and seeds demonstration inspections from sample_images/.
"""

import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.config import settings
from backend.app.database.connection import init_database, SessionLocal
from backend.app.database.models import Inspection
from backend.app.core.preprocessor import ImagePreprocessor
from backend.app.core.feature_extractor import FeatureExtractor
from backend.app.core.quality_analyzer import QualityAnalyzer
from backend.app.core.defect_detector import DefectDetector
from backend.app.core.report_generator import ReportGenerator
from backend.app.ml.model_loader import QualityModelLoader
from backend.app.database.repository import InspectionRepository
import cv2


def init_and_seed_database():
    print("Initializing database tables...")
    init_database()

    db = SessionLocal()
    try:
        count = db.query(Inspection).count()
        if count > 0:
            print(f"Database already contains {count} inspection records. Skipping seeding.")
            return

        print("Seeding baseline demonstration inspections...")
        sample_dir = settings.SAMPLE_IMAGES_DIR
        sample_files = sorted(list(sample_dir.glob("*.png")))

        if not sample_files:
            from scripts.generate_sample_data import generate_all_samples
            generate_all_samples()
            sample_files = sorted(list(sample_dir.glob("*.png")))

        preprocessor = ImagePreprocessor()
        extractor = FeatureExtractor()
        analyzer = QualityAnalyzer()
        detector = DefectDetector()
        model_loader = QualityModelLoader.get_instance()

        for sample_path in sample_files:
            start_time = time.time()
            prep = preprocessor.load_from_path(str(sample_path))
            feat_dict, _ = extractor.extract_features(prep)
            q_res = analyzer.analyze(feat_dict)
            ml_res = model_loader.predict(feat_dict)
            d_res = detector.detect(prep)

            report = ReportGenerator.generate_report(feat_dict, q_res, d_res, ml_res, start_time)

            # Save uploaded artifacts
            stem = sample_path.stem
            orig_rel = f"/uploads/{stem}_orig.png"
            annot_rel = f"/uploads/{stem}_annot.png"
            heat_rel = f"/uploads/{stem}_heat.png"

            cv2.imwrite(str(settings.UPLOAD_DIR / f"{stem}_orig.png"), prep.bgr)
            if d_res.annotated_image is not None:
                cv2.imwrite(str(settings.UPLOAD_DIR / f"{stem}_annot.png"), d_res.annotated_image)
            if d_res.heatmap_bgr is not None:
                cv2.imwrite(str(settings.UPLOAD_DIR / f"{stem}_heat.png"), d_res.heatmap_bgr)

            InspectionRepository.save_inspection(
                db=db,
                filename=sample_path.name,
                original_path=orig_rel,
                annotated_path=annot_rel,
                heatmap_path=heat_rel,
                report=report
            )

        print(f"Successfully seeded {len(sample_files)} baseline inspections into SQLite.")
    finally:
        db.close()


if __name__ == "__main__":
    init_and_seed_database()
