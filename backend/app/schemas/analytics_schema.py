"""
Analytics Pydantic Schemas
--------------------------
API contract models for dashboard analytics and KPI summaries.
"""

from typing import Dict, List, Any
from pydantic import BaseModel


class ScoreTrendPointSchema(BaseModel):
    id: int
    score: float
    label: str
    timestamp: str


class AnalyticsResponseSchema(BaseModel):
    total_inspections: int
    pass_rate_pct: float
    avg_quality_score: float
    avg_processing_time_ms: float
    class_distribution: Dict[str, int]
    defect_distribution: Dict[str, int]
    recent_scores: List[ScoreTrendPointSchema]


class HealthResponseSchema(BaseModel):
    status: str
    version: str
    model_loaded: bool
    database_connected: bool
