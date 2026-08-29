"""
Export API Routes
-----------------
Provides data export endpoints for compliance auditing (CSV and JSON formats).
"""

import io
import csv
import json
from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import Inspection

router = APIRouter(prefix="/export", tags=["Export"])


@router.get("/csv", summary="Export all inspection records as CSV")
def export_csv(db: Session = Depends(get_db)):
    """Exports all inspections and their 7 quality metrics to a downloadable CSV."""
    records = db.query(Inspection).order_by(Inspection.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Inspection ID",
        "Timestamp",
        "Filename",
        "Quality Score",
        "Quality Label",
        "Confidence",
        "Processing Time (ms)",
        "Defect Count",
        "Sharpness",
        "Brightness",
        "Contrast",
        "Noise",
        "Entropy",
        "Saturation",
        "Edge Density",
        "Explanation"
    ])

    for r in records:
        m = r.metrics
        writer.writerow([
            r.id,
            r.created_at.isoformat() if r.created_at else "",
            r.filename,
            r.quality_score,
            r.quality_label,
            r.confidence,
            r.processing_time,
            len(r.defects),
            m.sharpness if m else "",
            m.brightness if m else "",
            m.contrast if m else "",
            m.noise if m else "",
            m.entropy if m else "",
            m.saturation if m else "",
            m.edge_density if m else "",
            r.explanation or ""
        ])

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=inspection_records.csv"}
    )


@router.get("/json", summary="Export all inspection records as JSON")
def export_json(db: Session = Depends(get_db)):
    """Exports all inspections with nested defects and metrics as JSON."""
    records = db.query(Inspection).order_by(Inspection.created_at.desc()).all()

    data = []
    for r in records:
        m = r.metrics
        defects = [
            {
                "type": d.defect_type,
                "severity": d.severity,
                "confidence": d.confidence,
                "area": d.area,
                "bbox": {"x": d.x, "y": d.y, "w": d.width, "h": d.height}
            }
            for d in r.defects
        ]
        data.append({
            "id": r.id,
            "filename": r.filename,
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "quality_score": r.quality_score,
            "quality_label": r.quality_label,
            "confidence": r.confidence,
            "processing_time_ms": r.processing_time,
            "explanation": r.explanation,
            "metrics": {
                "sharpness": m.sharpness if m else None,
                "brightness": m.brightness if m else None,
                "contrast": m.contrast if m else None,
                "noise": m.noise if m else None,
                "entropy": m.entropy if m else None,
                "saturation": m.saturation if m else None,
                "edge_density": m.edge_density if m else None,
            },
            "defects": defects
        })

    json_str = json.dumps(data, indent=2)
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=inspection_records.json"}
    )
