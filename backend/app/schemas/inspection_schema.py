"""
Inspection Pydantic Schemas
---------------------------
Data transfer objects and API contract specifications for image inspections.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DefectItemSchema(BaseModel):
    type: str
    severity: str
    confidence: float
    area: float
    aspect_ratio: Optional[float] = None
    circularity: Optional[float] = None
    bounding_box: BoundingBoxSchema


class IssueSchema(BaseModel):
    category: str
    type: str
    severity: str
    confidence: float
    description: str
    metric_value: Optional[float] = None
    bounding_box: Optional[BoundingBoxSchema] = None


class InspectionResponseSchema(BaseModel):
    inspection_id: int
    filename: str
    quality_score: float
    quality_label: str
    confidence: float
    primary_issue: Optional[str] = None
    explanation: str
    issues: List[IssueSchema]
    statistics: Dict[str, float]
    sub_scores: Dict[str, float]
    defect_summary: Dict[str, Any]
    ml_result: Dict[str, Any]
    processing_time_ms: float
    image_url: str
    annotated_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    created_at: str


class InspectionHistoryItemSchema(BaseModel):
    id: int
    filename: str
    quality_score: float
    quality_label: str
    confidence: float
    processing_time: float
    created_at: str
    defect_count: int
    image_url: str
    annotated_url: Optional[str] = None


class HistoryResponseSchema(BaseModel):
    total: int
    items: List[InspectionHistoryItemSchema]
