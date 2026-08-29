"""
Quality Analyzer Module
-----------------------
Analyzes extracted numerical features against calibrated physical quality thresholds.
Detects specific issues (blur, underexposure, overexposure, noise, low contrast),
estimates issue severity and confidence, and calculates a normalized 0-100 quality score.

Design Principles:
- Interpretable, deterministic heuristic baseline.
- Clear separation between feature extraction, issue identification, and score synthesis.
- Provides clear explanations for human inspectors and viva review.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from backend.app.config import settings


@dataclass
class QualityIssue:
    """Represents a specific detected visual quality deficiency."""
    type: str            # e.g., "blur", "underexposure", "overexposure", "noise", "low_contrast"
    severity: str        # "low", "medium", "high"
    confidence: float    # 0.0 to 1.0 confidence score
    description: str     # Human-readable explanation of why this issue was flagged
    metric_value: float  # The exact measured numerical value that triggered the issue
    threshold: float     # The reference threshold used for comparison


@dataclass
class QualityAnalysisResult:
    """Comprehensive result of the rule-based image quality evaluation."""
    quality_score: float                  # Normalized score between 0.0 and 100.0
    quality_label: str                    # "ACCEPTABLE", "DEGRADED", or "DEFECTIVE"
    issues: List[QualityIssue] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    sub_scores: Dict[str, float] = field(default_factory=dict)  # Component scores (sharpness, exposure, noise, etc.)


class QualityAnalyzer:
    """
    Evaluates image quality metrics to produce diagnostic issues and a composite quality score.
    """

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or settings.quality

    def analyze(self, features: Dict[str, float]) -> QualityAnalysisResult:
        """
        Performs full quality analysis on extracted features.

        Args:
            features: Dictionary containing sharpness, brightness, contrast,
                      noise, entropy, saturation, edge_density.

        Returns:
            QualityAnalysisResult with scores, issues, and textual explanations.
        """
        issues: List[QualityIssue] = []
        explanations: List[str] = []

        sharpness = features.get("sharpness", 0.0)
        brightness = features.get("brightness", 128.0)
        contrast = features.get("contrast", 50.0)
        noise = features.get("noise", 0.0)
        entropy = features.get("entropy", 7.0)

        # -------------------------------------------------------------
        # 1. Blur / Sharpness Evaluation
        # -------------------------------------------------------------
        # Sharpness sub-score (0-100)
        # Using sigmoid-like saturation: 500+ Laplacian variance is considered pristine sharp
        sharpness_sub_score = min(100.0, max(0.0, (sharpness / 300.0) * 100.0))
        
        if sharpness < self.thresholds.blur_critical:
            conf = min(1.0, (self.thresholds.blur_critical - sharpness) / self.thresholds.blur_critical + 0.3)
            issues.append(QualityIssue(
                type="blur",
                severity="high",
                confidence=round(conf, 2),
                description=f"Significant blur detected (Laplacian variance {sharpness:.1f} < {self.thresholds.blur_critical:.1f}). Image lacks fine edge details.",
                metric_value=sharpness,
                threshold=self.thresholds.blur_critical
            ))
            explanations.append("Severe loss of sharpness/focus detected across the image surface.")
        elif sharpness < self.thresholds.blur_warning:
            conf = min(1.0, (self.thresholds.blur_warning - sharpness) / (self.thresholds.blur_warning - self.thresholds.blur_critical) * 0.5 + 0.4)
            issues.append(QualityIssue(
                type="blur",
                severity="medium",
                confidence=round(conf, 2),
                description=f"Moderate blur detected (Laplacian variance {sharpness:.1f} < {self.thresholds.blur_warning:.1f}). Edge definition is soft.",
                metric_value=sharpness,
                threshold=self.thresholds.blur_warning
            ))
            explanations.append("Moderate softness observed; fine textures may be partially obscured.")

        # -------------------------------------------------------------
        # 2. Exposure (Brightness) Evaluation
        # -------------------------------------------------------------
        # Exposure score: Ideal mean luminance is around 110-145
        exposure_dev = abs(brightness - 128.0)
        exposure_sub_score = max(0.0, 100.0 - (exposure_dev / 128.0) * 100.0)

        if brightness < self.thresholds.underexposed_threshold:
            severity = "high" if brightness < 35.0 else "medium"
            conf = min(1.0, (self.thresholds.underexposed_threshold - brightness) / self.thresholds.underexposed_threshold + 0.3)
            issues.append(QualityIssue(
                type="underexposure",
                severity=severity,
                confidence=round(conf, 2),
                description=f"Image is underexposed with low mean luminance ({brightness:.1f} < {self.thresholds.underexposed_threshold:.1f}). Shadow clipping likely.",
                metric_value=brightness,
                threshold=self.thresholds.underexposed_threshold
            ))
            explanations.append("Insufficient lighting/underexposure obscuring dark regions.")
        elif brightness > self.thresholds.overexposed_threshold:
            severity = "high" if brightness > 230.0 else "medium"
            conf = min(1.0, (brightness - self.thresholds.overexposed_threshold) / (255.0 - self.thresholds.overexposed_threshold) + 0.3)
            issues.append(QualityIssue(
                type="overexposure",
                severity=severity,
                confidence=round(conf, 2),
                description=f"Image is overexposed with high mean luminance ({brightness:.1f} > {self.thresholds.overexposed_threshold:.1f}). Highlight clipping present.",
                metric_value=brightness,
                threshold=self.thresholds.overexposed_threshold
            ))
            explanations.append("Excessive brightness/glare causing washed out highlight details.")

        # -------------------------------------------------------------
        # 3. Contrast Evaluation
        # -------------------------------------------------------------
        # Standard deviation of 45-65 represents healthy contrast
        contrast_sub_score = min(100.0, max(0.0, (contrast / 50.0) * 100.0))

        if contrast < self.thresholds.low_contrast_threshold:
            conf = min(1.0, (self.thresholds.low_contrast_threshold - contrast) / self.thresholds.low_contrast_threshold + 0.4)
            issues.append(QualityIssue(
                type="low_contrast",
                severity="medium" if contrast > 18.0 else "high",
                confidence=round(conf, 2),
                description=f"Low contrast detected (Intensity standard deviation {contrast:.1f} < {self.thresholds.low_contrast_threshold:.1f}).",
                metric_value=contrast,
                threshold=self.thresholds.low_contrast_threshold
            ))
            explanations.append("Low dynamic range/flat lighting reducing feature distinctness.")

        # -------------------------------------------------------------
        # 4. Noise Evaluation
        # -------------------------------------------------------------
        # MAD noise: clean images typically have noise < 4.0; noise > 8.5 indicates high grain
        noise_sub_score = max(0.0, 100.0 - (noise / 15.0) * 100.0)

        if noise > self.thresholds.high_noise_threshold:
            conf = min(1.0, (noise - self.thresholds.high_noise_threshold) / 10.0 + 0.4)
            issues.append(QualityIssue(
                type="noise",
                severity="high" if noise > 14.0 else "medium",
                confidence=round(conf, 2),
                description=f"Elevated high-frequency noise/grain detected (MAD residual estimate {noise:.2f} > {self.thresholds.high_noise_threshold:.2f}).",
                metric_value=noise,
                threshold=self.thresholds.high_noise_threshold
            ))
            explanations.append("High sensor noise or grain artifact interfering with visual clarity.")

        # -------------------------------------------------------------
        # 5. Composite Quality Score Calculation (0 - 100)
        # -------------------------------------------------------------
        # Weighted formula matching Section 16 of specifications:
        # Sharpness: 35%, Exposure: 25%, Noise: 15%, Contrast: 15%, Entropy/Other: 10%
        entropy_sub_score = min(100.0, max(0.0, (entropy / 7.5) * 100.0))

        sub_scores = {
            "sharpness": round(sharpness_sub_score, 1),
            "exposure": round(exposure_sub_score, 1),
            "contrast": round(contrast_sub_score, 1),
            "noise": round(noise_sub_score, 1),
            "entropy": round(entropy_sub_score, 1)
        }

        composite_score = (
            (sharpness_sub_score * settings.WEIGHT_SHARPNESS) +
            (exposure_sub_score * settings.WEIGHT_EXPOSURE) +
            (noise_sub_score * settings.WEIGHT_NOISE) +
            (contrast_sub_score * settings.WEIGHT_CONTRAST) +
            (entropy_sub_score * settings.WEIGHT_OTHER)
        )

        composite_score = max(0.0, min(100.0, composite_score))
        composite_score = round(composite_score, 1)

        # Map to standard quality label thresholds:
        # 90-100: ACCEPTABLE
        # 60-89:  DEGRADED
        # 0-59:   DEFECTIVE
        if composite_score >= 88.0 and not any(issue.severity == "high" for issue in issues):
            quality_label = "ACCEPTABLE"
        elif composite_score >= 55.0 and not (sum(1 for i in issues if i.severity == "high") >= 2):
            quality_label = "DEGRADED"
        else:
            quality_label = "DEFECTIVE"

        if not explanations:
            explanations.append("Image exhibits good sharpness, balanced exposure, and low noise.")

        return QualityAnalysisResult(
            quality_score=composite_score,
            quality_label=quality_label,
            issues=issues,
            explanations=explanations,
            sub_scores=sub_scores
        )
