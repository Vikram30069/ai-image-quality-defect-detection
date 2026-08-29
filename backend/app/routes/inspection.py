"""
Inspection API Routes
---------------------
Handles image file uploads, orchestrates the Computer Vision & ML pipeline,
persists inspection records to SQLite, and returns diagnostic results.
"""

import time
import uuid
from pathlib import Path
from typing import Optional
import cv2
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database.connection import get_db
from backend.app.database.repository import InspectionRepository
from backend.app.core.preprocessor import ImagePreprocessor
from backend.app.core.feature_extractor import FeatureExtractor
from backend.app.core.quality_analyzer import QualityAnalyzer
from backend.app.core.defect_detector import DefectDetector
from backend.app.core.report_generator import ReportGenerator
from backend.app.ml.model_loader import QualityModelLoader
from backend.app.schemas.inspection_schema import InspectionResponseSchema

router = APIRouter(tags=["Inspection"])

# Initialize analytical singletons
preprocessor = ImagePreprocessor(max_dimension=settings.MAX_IMAGE_DIMENSION)
feature_extractor = FeatureExtractor()
quality_analyzer = QualityAnalyzer()
defect_detector = DefectDetector()
model_loader = QualityModelLoader.get_instance()


@router.post("/inspect", response_model=InspectionResponseSchema, summary="Inspect image for quality and visual defects")
async def inspect_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, BMP, TIFF)"),
    db: Session = Depends(get_db)
):
    """
    Main inspection endpoint:
    1. Validates file format and size.
    2. Decodes image into multi-color spaces.
    3. Extracts 7 canonical quality features.
    4. Evaluates rule-based quality metrics & flags specific issues.
    5. Informs Machine Learning RandomForest quality classifier.
    6. Performs morphological defect segmentation & localization.
    7. Synthesizes an explainable report & persists to SQLite.
    """
    start_time = time.time()

    # 1. Validate File Extension
    filename = file.filename or "unknown.png"
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {list(settings.ALLOWED_EXTENSIONS)}"
        )

    # 2. Read File Bytes & Validate Size
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file payload: {str(e)}"
        )

    size_mb = len(image_bytes) / (1024.0 * 1024.0)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {settings.MAX_FILE_SIZE_MB} MB."
        )

    # 3. Decode & Preprocess Image
    try:
        prep = preprocessor.validate_and_decode(image_bytes)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT if hasattr(status, "HTTP_422_UNPROCESSABLE_CONTENT") else 422,
            detail=f"Image decoding failed: {str(val_err)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during image preprocessing: {str(e)}"
        )

    # 4. Extract Quality Features
    feat_dict, feat_vector = feature_extractor.extract_features(prep)

    # 5. Rule-Based Quality Analysis
    quality_result = quality_analyzer.analyze(feat_dict)

    # 6. ML Model Classification
    ml_result = model_loader.predict(feat_dict)

    # 7. Classical CV Defect Detection
    defect_result = defect_detector.detect(prep)

    # 8. Report Synthesis
    report = ReportGenerator.generate_report(
        features=feat_dict,
        quality_result=quality_result,
        defect_result=defect_result,
        ml_result=ml_result,
        start_time=start_time
    )

    # 9. Save Images to Upload Storage
    unique_id = uuid.uuid4().hex[:12]
    safe_base_name = f"{unique_id}_{Path(filename).stem}"
    
    orig_filename = f"{safe_base_name}_orig.png"
    annot_filename = f"{safe_base_name}_annotated.png"
    heat_filename = f"{safe_base_name}_heatmap.png"

    orig_path = settings.UPLOAD_DIR / orig_filename
    annot_path = settings.UPLOAD_DIR / annot_filename
    heat_path = settings.UPLOAD_DIR / heat_filename

    # Write files
    cv2.imwrite(str(orig_path), prep.bgr)
    if defect_result.annotated_image is not None:
        cv2.imwrite(str(annot_path), defect_result.annotated_image)
    if defect_result.heatmap_bgr is not None:
        cv2.imwrite(str(heat_path), defect_result.heatmap_bgr)

    # 10. Persist to Database
    db_inspection = InspectionRepository.save_inspection(
        db=db,
        filename=filename,
        original_path=f"/uploads/{orig_filename}",
        annotated_path=f"/uploads/{annot_filename}",
        heatmap_path=f"/uploads/{heat_filename}",
        report=report
    )

    return InspectionResponseSchema(
        inspection_id=db_inspection.id,
        filename=db_inspection.filename,
        quality_score=db_inspection.quality_score,
        quality_label=db_inspection.quality_label,
        confidence=db_inspection.confidence,
        primary_issue=report.primary_issue,
        explanation=report.explanation,
        issues=report.issues,
        statistics=report.statistics,
        sub_scores=report.sub_scores,
        defect_summary=report.defect_summary,
        ml_result=report.ml_result,
        processing_time_ms=report.processing_time_ms,
        image_url=db_inspection.original_image_path,
        annotated_url=db_inspection.annotated_image_path,
        heatmap_url=db_inspection.heatmap_image_path,
        created_at=db_inspection.created_at.isoformat()
    )


@router.get("/samples/{sample_name}", summary="Get sample dataset image by name")
async def get_sample_image(sample_name: str):
    """Returns a representative dataset sample image."""
    from fastapi.responses import FileResponse, Response
    
    safe_name = Path(sample_name).name.lower().replace(".png", "")
    alias_map = {
        "clean": "sample_good.png",
        "good": "sample_good.png",
        "scratch": "sample_scratched.png",
        "scratched": "sample_scratched.png",
        "crack": "sample_crack_like.png",
        "blemish": "sample_blemish.png",
        "contamination": "sample_contamination.png"
    }
    actual_file = alias_map.get(safe_name, f"sample_{safe_name}.png")
    
    candidates = [
        settings.SAMPLE_IMAGES_DIR / actual_file,
        Path.cwd() / "sample_images" / actual_file,
        Path(f"/app/sample_images/{actual_file}"),
        settings.BASE_DIR / "sample_images" / actual_file
    ]
    for c in candidates:
        if c.exists():
            return FileResponse(str(c), media_type="image/png")
            
    # Fallback to generating sample image on-the-fly if missing on disk
    from scripts.generate_sample_data import generate_base_image, apply_scratch, apply_blemish, apply_crack
    base = generate_base_image(seed=101)
    if "scratch" in actual_file:
        img = apply_scratch(base)
    elif "crack" in actual_file:
        img = apply_crack(base)
    elif "blemish" in actual_file:
        img = apply_blemish(base)
    else:
        img = base
        
    _, buf = cv2.imencode(".png", img)
    return Response(content=buf.tobytes(), media_type="image/png")

