"""
History API Routes
------------------
Endpoints for browsing historical inspection logs, retrieving detailed records,
and deleting inspections.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.repository import InspectionRepository
from backend.app.schemas.inspection_schema import (
    HistoryResponseSchema,
    InspectionHistoryItemSchema,
    InspectionResponseSchema,
    IssueSchema,
    BoundingBoxSchema
)

router = APIRouter(prefix="/history", tags=["History"])


@router.get("", response_model=HistoryResponseSchema, summary="Get paginated inspection history")
def get_history(
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    label: Optional[str] = Query(None, description="Filter by label: ACCEPTABLE, DEGRADED, DEFECTIVE"),
    db: Session = Depends(get_db)
):
    records, total = InspectionRepository.get_history(db=db, limit=limit, offset=offset, label_filter=label)

    items = [
        InspectionHistoryItemSchema(
            id=r.id,
            filename=r.filename,
            quality_score=r.quality_score,
            quality_label=r.quality_label,
            confidence=r.confidence,
            processing_time=r.processing_time,
            created_at=r.created_at.isoformat(),
            defect_count=len(r.defects),
            image_url=r.original_image_path,
            annotated_url=r.annotated_image_path
        )
        for r in records
    ]

    return HistoryResponseSchema(total=total, items=items)


@router.get("/{inspection_id}", response_model=InspectionResponseSchema, summary="Get full details of a specific inspection")
def get_inspection_detail(
    inspection_id: int,
    db: Session = Depends(get_db)
):
    r = InspectionRepository.get_inspection_by_id(db=db, inspection_id=inspection_id)
    if not r:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection record #{inspection_id} not found."
        )

    # Reconstruct statistics and defect summary
    m = r.metrics
    stats = {
        "sharpness": m.sharpness if m else 0.0,
        "brightness": m.brightness if m else 0.0,
        "contrast": m.contrast if m else 0.0,
        "noise": m.noise if m else 0.0,
        "entropy": m.entropy if m else 0.0,
        "saturation": m.saturation if m else 0.0,
        "edge_density": m.edge_density if m else 0.0,
        "defect_count": len(r.defects)
    }

    defect_list = [
        {
            "type": d.defect_type,
            "severity": d.severity,
            "confidence": d.confidence,
            "area": d.area,
            "bounding_box": {
                "x": d.x,
                "y": d.y,
                "width": d.width,
                "height": d.height
            }
        }
        for d in r.defects
    ]

    issues = [
        IssueSchema(
            category="surface_defect",
            type=d.defect_type.lower(),
            severity=d.severity,
            confidence=d.confidence,
            description=f"{d.defect_type} anomaly ({d.severity} severity, area {d.area}px²)",
            bounding_box=BoundingBoxSchema(x=d.x, y=d.y, width=d.width, height=d.height)
        )
        for d in r.defects
    ]

    return InspectionResponseSchema(
        inspection_id=r.id,
        filename=r.filename,
        quality_score=r.quality_score,
        quality_label=r.quality_label,
        confidence=r.confidence,
        explanation=r.explanation or "",
        issues=issues,
        statistics=stats,
        sub_scores={},
        defect_summary={"total_defects": len(r.defects), "defect_list": defect_list},
        ml_result={"predicted_label": r.quality_label, "confidence": r.confidence},
        processing_time_ms=r.processing_time,
        image_url=r.original_image_path,
        annotated_url=r.annotated_image_path,
        heatmap_url=r.heatmap_image_path,
        created_at=r.created_at.isoformat()
    )


@router.delete("/{inspection_id}", summary="Delete an inspection record")
def delete_inspection(
    inspection_id: int,
    db: Session = Depends(get_db)
):
    success = InspectionRepository.delete_inspection(db=db, inspection_id=inspection_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection record #{inspection_id} not found."
        )
    return {"status": "success", "message": f"Inspection #{inspection_id} deleted successfully."}
