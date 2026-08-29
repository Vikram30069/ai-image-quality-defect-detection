"""
Database Repository Module
--------------------------
Encapsulates all SQL transactions and queries for saving inspection records,
retrieving paginated history, deleting records, and calculating aggregated analytics.
"""

from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.app.database.models import Inspection, QualityMetric, Defect, SystemLog
from backend.app.core.report_generator import InspectionReport


class InspectionRepository:
    """
    Data access layer for Inspections, Defects, Metrics, and System Logs.
    """

    @staticmethod
    def save_inspection(
        db: Session,
        filename: str,
        original_path: str,
        annotated_path: Optional[str],
        heatmap_path: Optional[str],
        report: InspectionReport
    ) -> Inspection:
        """
        Atomically saves the inspection header, its quality metrics, and all defect bounding boxes.
        """
        # 1. Create main inspection record
        inspection = Inspection(
            filename=filename,
            original_image_path=original_path,
            annotated_image_path=annotated_path,
            heatmap_image_path=heatmap_path,
            quality_score=report.quality_score,
            quality_label=report.quality_label,
            confidence=report.confidence,
            processing_time=report.processing_time_ms,
            explanation=report.explanation
        )
        db.add(inspection)
        db.flush()  # Generates inspection.id

        # 2. Save 7-feature quality metrics
        stats = report.statistics
        metric = QualityMetric(
            inspection_id=inspection.id,
            sharpness=stats.get("sharpness", 0.0),
            brightness=stats.get("brightness", 0.0),
            contrast=stats.get("contrast", 0.0),
            noise=stats.get("noise", 0.0),
            entropy=stats.get("entropy", 0.0),
            saturation=stats.get("saturation", 0.0),
            edge_density=stats.get("edge_density", 0.0)
        )
        db.add(metric)

        # 3. Save defect bounding boxes
        for d in report.defect_summary.get("defect_list", []):
            bbox = d.get("bounding_box", {})
            defect = Defect(
                inspection_id=inspection.id,
                defect_type=d.get("type", "UNKNOWN"),
                severity=d.get("severity", "low"),
                confidence=d.get("confidence", 0.5),
                x=bbox.get("x", 0),
                y=bbox.get("y", 0),
                width=bbox.get("width", 0),
                height=bbox.get("height", 0),
                area=d.get("area", 0.0)
            )
            db.add(defect)

        db.commit()
        db.refresh(inspection)
        return inspection

    @staticmethod
    def get_inspection_by_id(db: Session, inspection_id: int) -> Optional[Inspection]:
        """Fetches a single inspection by its primary key ID."""
        return db.query(Inspection).filter(Inspection.id == inspection_id).first()

    @staticmethod
    def get_history(
        db: Session,
        limit: int = 50,
        offset: int = 0,
        label_filter: Optional[str] = None
    ) -> Tuple[List[Inspection], int]:
        """
        Retrieves paginated inspection records sorted by creation timestamp descending.
        """
        query = db.query(Inspection)
        if label_filter and label_filter.upper() in ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]:
            query = query.filter(Inspection.quality_label == label_filter.upper())

        total_count = query.count()
        records = query.order_by(desc(Inspection.created_at)).offset(offset).limit(limit).all()
        return records, total_count

    @staticmethod
    def delete_inspection(db: Session, inspection_id: int) -> bool:
        """Deletes an inspection and its associated metrics and defects (via CASCADE)."""
        inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
        if not inspection:
            return False
        db.delete(inspection)
        db.commit()
        return True

    @staticmethod
    def get_analytics(db: Session) -> Dict[str, Any]:
        """
        Computes aggregate statistics:
        - Total inspections
        - Pass rate (% ACCEPTABLE)
        - Average quality score
        - Class distribution (ACCEPTABLE, DEGRADED, DEFECTIVE)
        - Defect distribution by type (SCRATCH, CRACK_LIKE, BLEMISH, CONTAMINATION_LIKE)
        - Average processing time
        """
        total_inspections = db.query(Inspection).count()
        if total_inspections == 0:
            return {
                "total_inspections": 0,
                "pass_rate_pct": 0.0,
                "avg_quality_score": 0.0,
                "avg_processing_time_ms": 0.0,
                "class_distribution": {"ACCEPTABLE": 0, "DEGRADED": 0, "DEFECTIVE": 0},
                "defect_distribution": {"SCRATCH": 0, "CRACK_LIKE": 0, "BLEMISH": 0, "CONTAMINATION_LIKE": 0},
                "recent_scores": []
            }

        # Class counts
        class_counts = db.query(
            Inspection.quality_label, func.count(Inspection.id)
        ).group_by(Inspection.quality_label).all()
        class_dist = {"ACCEPTABLE": 0, "DEGRADED": 0, "DEFECTIVE": 0}
        for lbl, count in class_counts:
            if lbl in class_dist:
                class_dist[lbl] = count

        acceptable_count = class_dist.get("ACCEPTABLE", 0)
        pass_rate = round((acceptable_count / float(total_inspections)) * 100.0, 1)

        # Average quality score & processing time
        avg_score = db.query(func.avg(Inspection.quality_score)).scalar() or 0.0
        avg_time = db.query(func.avg(Inspection.processing_time)).scalar() or 0.0

        # Defect counts by type
        defect_counts = db.query(
            Defect.defect_type, func.count(Defect.id)
        ).group_by(Defect.defect_type).all()
        defect_dist = {"SCRATCH": 0, "CRACK_LIKE": 0, "BLEMISH": 0, "CONTAMINATION_LIKE": 0}
        for dtype, count in defect_counts:
            defect_dist[dtype] = count

        # Recent 15 inspection scores for trend plotting
        recent_records = db.query(
            Inspection.id, Inspection.quality_score, Inspection.quality_label, Inspection.created_at
        ).order_by(desc(Inspection.created_at)).limit(15).all()

        recent_scores = [
            {
                "id": r.id,
                "score": r.quality_score,
                "label": r.quality_label,
                "timestamp": r.created_at.strftime("%H:%M:%S")
            }
            for r in reversed(recent_records)
        ]

        return {
            "total_inspections": total_inspections,
            "pass_rate_pct": pass_rate,
            "avg_quality_score": round(float(avg_score), 1),
            "avg_processing_time_ms": round(float(avg_time), 1),
            "class_distribution": class_dist,
            "defect_distribution": defect_dist,
            "recent_scores": recent_scores
        }

    @staticmethod
    def log_event(db: Session, level: str, message: str):
        """Appends an operational event into system_logs."""
        try:
            log_entry = SystemLog(level=level.upper(), message=message)
            db.add(log_entry)
            db.commit()
        except Exception:
            db.rollback()
