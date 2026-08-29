"""
SQLAlchemy Database Models
--------------------------
Defines schemas for:
- inspections: High-level inspection runs, scores, labels, file references.
- quality_metrics: Detailed extracted 7-feature quality values.
- defects: Bounding box coordinates, defect classifications, and severity.
- system_logs: Audit logs and diagnostic events.
"""

import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from backend.app.database.connection import Base


class Inspection(Base):
    """Represents a single analyzed image record."""
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    original_image_path = Column(String(512), nullable=False)
    annotated_image_path = Column(String(512), nullable=True)
    heatmap_image_path = Column(String(512), nullable=True)
    
    quality_score = Column(Float, nullable=False)
    quality_label = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    processing_time = Column(Float, nullable=False)  # in milliseconds
    explanation = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True)

    # Relationships (Cascades delete to children records)
    metrics = relationship("QualityMetric", back_populates="inspection", uselist=False, cascade="all, delete-orphan")
    defects = relationship("Defect", back_populates="inspection", cascade="all, delete-orphan")


class QualityMetric(Base):
    """Stores the 7 canonical quality features for an inspection."""
    __tablename__ = "quality_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    sharpness = Column(Float, nullable=False)
    brightness = Column(Float, nullable=False)
    contrast = Column(Float, nullable=False)
    noise = Column(Float, nullable=False)
    entropy = Column(Float, nullable=False)
    saturation = Column(Float, nullable=False)
    edge_density = Column(Float, nullable=False)

    inspection = relationship("Inspection", back_populates="metrics")


class Defect(Base):
    """Stores individual localized defect bounding boxes and properties."""
    __tablename__ = "defects"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, index=True)
    
    defect_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    area = Column(Float, nullable=False)

    inspection = relationship("Inspection", back_populates="defects")


class SystemLog(Base):
    """Stores operational events, warnings, and error messages."""
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    level = Column(String(20), nullable=False, default="INFO")
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True)
