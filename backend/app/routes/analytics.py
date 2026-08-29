"""
Analytics & Health API Routes
-----------------------------
Endpoints for dashboard key performance indicators and system health checks.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.app.config import settings
from backend.app.database.connection import get_db
from backend.app.database.repository import InspectionRepository
from backend.app.ml.model_loader import QualityModelLoader
from backend.app.schemas.analytics_schema import AnalyticsResponseSchema, HealthResponseSchema

router = APIRouter(tags=["Analytics & System"])


@router.get("/analytics", response_model=AnalyticsResponseSchema, summary="Get aggregated inspection metrics")
def get_analytics(db: Session = Depends(get_db)):
    """Computes pass rate, average quality score, class and defect distributions."""
    analytics_data = InspectionRepository.get_analytics(db=db)
    return AnalyticsResponseSchema(**analytics_data)


@router.get("/health", response_model=HealthResponseSchema, summary="System health check")
def health_check(db: Session = Depends(get_db)):
    """Verifies database connectivity and ML model availability."""
    model_loader = QualityModelLoader.get_instance()
    is_model_loaded = model_loader.model is not None

    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        # Fallback query if text import isn't explicit
        try:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False

    return HealthResponseSchema(
        status="healthy" if (db_ok and is_model_loaded) else "degraded",
        version=settings.VERSION,
        model_loaded=is_model_loaded,
        database_connected=db_ok
    )
