"""
Report & Decision Engine Module
-------------------------------
Synthesizes predictions from the Machine Learning quality classifier,
heuristic Computer Vision quality metrics, and localized surface defect detections
into an integrated, explainable industrial inspection report.

Key Enhancements:
1. Authoritative ML Confidence: Ensures identical, consistent confidence values
   across the Quality Card, Decision Explanation, API responses, and database logs.
2. Bounded Validated Defect Penalties: Penalties are strictly calculated from validated
   defect entities (zero penalty for rejected texture noise).
3. Hybrid Arbitration: Blends global ML quality state with localized physical defect findings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time

from backend.app.core.feature_extractor import FEATURE_NAMES
from backend.app.core.quality_analyzer import QualityAnalysisResult, QualityIssue
from backend.app.core.defect_detector import DefectDetectionResult, DetectedDefect
from backend.app.ml.model_loader import MLPredictionResult


@dataclass
class InspectionReport:
    """Final unified decision report for a single inspected image."""
    quality_score: float                           # Normalized final score [0.0 - 100.0]
    quality_label: str                             # "ACCEPTABLE", "DEGRADED", or "DEFECTIVE"
    confidence: float                             # Authoritative ML prediction confidence [0.0 - 1.0]
    primary_issue: Optional[str]                   # Key culprit (e.g. "Severe Blur", "Scratch Detected")
    explanation: str                              # Comprehensive human-readable explanation
    issues: List[Dict[str, Any]]                  # Consolidated list of quality and defect issues
    ml_result: Dict[str, Any]                     # Machine learning classification output
    statistics: Dict[str, float]                  # Numerical image statistics
    sub_scores: Dict[str, float]                  # Component quality scores
    defect_summary: Dict[str, Any]                # Defect count, density, and bounding boxes
    processing_time_ms: float                     # Total execution time in milliseconds


class ReportGenerator:
    """
    Synthesizes AI/ML inferences and classical CV measurements into a single decision.
    """

    @staticmethod
    def generate_report(
        features: Dict[str, float],
        quality_result: QualityAnalysisResult,
        defect_result: DefectDetectionResult,
        ml_result: MLPredictionResult,
        start_time: float
    ) -> InspectionReport:
        """
        Fuses all analytical inputs into a final inspection report.
        """
        processing_time_ms = round((time.time() - start_time) * 1000.0, 1)

        # -------------------------------------------------------------
        # 1. Base Score & Bounded Validated Defect Penalties
        # -------------------------------------------------------------
        base_score = quality_result.quality_score

        # Calculate penalty solely on validated defects (rejected texture noise has 0 penalty)
        defect_penalty = 0.0
        for d in defect_result.defects:
            if d.severity == "high":
                defect_penalty += 12.0
            elif d.severity == "medium":
                defect_penalty += 6.0
            else:
                defect_penalty += 3.0

        # Bounded defect density penalty (max 15 pts)
        defect_penalty += min(15.0, defect_result.defect_density * 8.0)
        # Cap total defect penalty at 40 pts so score does not unfairly collapse to 0 for a single minor anomaly
        defect_penalty = min(40.0, defect_penalty)

        final_score = max(0.0, min(100.0, base_score - defect_penalty))
        final_score = round(final_score, 1)

        # -------------------------------------------------------------
        # 2. Consolidated Issues List
        # -------------------------------------------------------------
        consolidated_issues = []

        # Add image quality issues
        for q_issue in quality_result.issues:
            consolidated_issues.append({
                "category": "image_quality",
                "type": q_issue.type,
                "severity": q_issue.severity,
                "confidence": q_issue.confidence,
                "description": q_issue.description,
                "metric_value": q_issue.metric_value
            })

        # Add validated physical defect issues
        for d in defect_result.defects:
            consolidated_issues.append({
                "category": "surface_defect",
                "type": d.defect_type.lower(),
                "severity": d.severity,
                "confidence": d.confidence,
                "description": f"{d.defect_type} anomaly localized at ({d.bounding_box.x}, {d.bounding_box.y}) with area {d.area}px² (Contrast: {d.local_contrast}).",
                "bounding_box": {
                    "x": d.bounding_box.x,
                    "y": d.bounding_box.y,
                    "width": d.bounding_box.width,
                    "height": d.bounding_box.height
                }
            })

        # -------------------------------------------------------------
        # 3. Hybrid Label Decision Arbitration
        # -------------------------------------------------------------
        has_critical_quality = any(i.severity == "high" for i in quality_result.issues)
        has_high_defect = any(d.severity == "high" for d in defect_result.defects)
        total_defects = defect_result.total_defects

        if (
            has_high_defect
            or (total_defects >= 2)
            or has_critical_quality
            or (ml_result.predicted_label == "DEFECTIVE" and ml_result.confidence >= 0.70)
            or final_score < 60.0
        ):
            final_label = "DEFECTIVE"
        elif (
            total_defects == 1
            or len(quality_result.issues) > 0
            or ml_result.predicted_label == "DEGRADED"
            or final_score < 88.0
        ):
            final_label = "DEGRADED"
        else:
            final_label = "ACCEPTABLE"

        # -------------------------------------------------------------
        # 4. Authoritative Confidence Selection
        # -------------------------------------------------------------
        # One authoritative confidence source from the ML prediction model
        authoritative_confidence = ml_result.confidence

        # -------------------------------------------------------------
        # 5. Natural Language Explanation Generation
        # -------------------------------------------------------------
        explanation_parts = []
        primary_issue = None

        if final_label == "ACCEPTABLE":
            primary_issue = "None"
            explanation_parts.append(
                f"Image passed all inspection criteria with optimal sharpness, balanced exposure, minimal noise, and zero localized surface defects. Machine learning model classified state as ACCEPTABLE (Confidence: {authoritative_confidence * 100:.1f}%)."
            )
        else:
            # Highlight top defect or quality concern
            if total_defects > 0:
                top_defect = max(defect_result.defects, key=lambda x: x.area)
                primary_issue = f"{top_defect.defect_type.replace('_', ' ').title()} Defect"
                explanation_parts.append(
                    f"Detected {total_defects} surface anomal{'y' if total_defects == 1 else 'ies'}, notably a {top_defect.defect_type} ({top_defect.severity} severity, {top_defect.area}px²)."
                )

            if quality_result.issues:
                top_q = quality_result.issues[0]
                if not primary_issue:
                    primary_issue = f"{top_q.type.replace('_', ' ').title()}"
                explanation_parts.append(
                    f"Quality analysis identified {top_q.description}"
                )

            explanation_parts.append(
                f"Machine learning model classified image state as {ml_result.predicted_label} (Confidence: {authoritative_confidence * 100:.1f}%)."
            )

        explanation = " ".join(explanation_parts)

        # -------------------------------------------------------------
        # 6. Assemble Final Structured Payload
        # -------------------------------------------------------------
        statistics = {name: round(float(features.get(name, 0.0)), 4) for name in FEATURE_NAMES}
        statistics["defect_count"] = total_defects
        statistics["defect_density_pct"] = defect_result.defect_density

        defect_summary = {
            "total_defects": total_defects,
            "defect_density_pct": defect_result.defect_density,
            "defect_list": [
                {
                    "type": d.defect_type,
                    "severity": d.severity,
                    "confidence": d.confidence,
                    "area": d.area,
                    "aspect_ratio": d.aspect_ratio,
                    "circularity": d.circularity,
                    "bounding_box": {
                        "x": d.bounding_box.x,
                        "y": d.bounding_box.y,
                        "width": d.bounding_box.width,
                        "height": d.bounding_box.height
                    }
                }
                for d in defect_result.defects
            ]
        }

        ml_summary = {
            "predicted_label": ml_result.predicted_label,
            "confidence": ml_result.confidence,
            "class_probabilities": ml_result.class_probabilities,
            "model_version": ml_result.model_version,
            "is_fallback": ml_result.is_fallback
        }

        return InspectionReport(
            quality_score=final_score,
            quality_label=final_label,
            confidence=authoritative_confidence,
            primary_issue=primary_issue,
            explanation=explanation,
            issues=consolidated_issues,
            ml_result=ml_summary,
            statistics=statistics,
            sub_scores=quality_result.sub_scores,
            defect_summary=defect_summary,
            processing_time_ms=processing_time_ms
        )
