"""
Defect Detection Module
-----------------------
Implements classical Computer Vision anomaly segmentation, multi-criteria shape validation,
fragment merging, and morphological defect classification.

Key Objectives:
1. Rejects normal surface texture and ambient lighting variations using dynamic local contrast thresholding.
2. Merges fragmented nearby defect components into single coherent physical defects.
3. Enforces strict geometric and contrast criteria for SCRATCH, CRACK_LIKE, BLEMISH, and CONTAMINATION_LIKE anomalies.
4. Generates clear visual bounding boxes and false-color spatial density heatmaps.

Notice on Industrial Application:
These detections identify candidate visual anomalies based on geometric and contrast properties;
they represent interpretable anomaly candidates rather than guaranteed physical defect types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from backend.app.config import settings
from backend.app.core.preprocessor import PreprocessedImage


@dataclass
class BoundingBox:
    """Represents the spatial bounding coordinates of a defect region."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class DetectedDefect:
    """Individual validated defect entity."""
    defect_type: str             # "SCRATCH", "CRACK_LIKE", "BLEMISH", "CONTAMINATION_LIKE"
    severity: str                # "low", "medium", "high"
    confidence: float            # 0.0 - 1.0 confidence based on contrast & geometry
    bounding_box: BoundingBox
    area: float                  # Contour area in pixels
    aspect_ratio: float          # Major / Minor axis ratio
    circularity: float           # 4 * π * Area / Perimeter²
    solidity: float = 1.0        # Area / ConvexHull Area
    local_contrast: float = 0.0  # Intensity delta from local background


@dataclass
class DefectDetectionResult:
    """Aggregated defect inspection results and visualization overlays."""
    total_defects: int
    defects: List[DetectedDefect] = field(default_factory=list)
    defect_mask: Optional[np.ndarray] = None       # Binary mask (H, W) uint8
    heatmap_bgr: Optional[np.ndarray] = None       # False-color JET heatmap overlay (H, W, 3)
    annotated_image: Optional[np.ndarray] = None   # Original image with bounding boxes & tags (H, W, 3)
    defect_density: float = 0.0                    # Percentage of surface area impacted [0.0 - 100.0]


class DefectDetector:
    """
    Robust classical Computer Vision pipeline for localizing and validating surface defects.
    """

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or settings.defects

    def detect(self, prep: PreprocessedImage) -> DefectDetectionResult:
        """
        Executes multi-scale anomaly extraction, adaptive contrast thresholding,
        fragment merging, and multi-criteria validation.

        Args:
            prep: PreprocessedImage instance (containing gray, bgr, lab).

        Returns:
            DefectDetectionResult with validated defects, masks, and annotated overlays.
        """
        gray = prep.gray
        h, w = prep.height, prep.width
        total_pixels = float(h * w)

        # -------------------------------------------------------------
        # Step 1: Multi-Scale Saliency & Anomaly Extraction
        # -------------------------------------------------------------
        saliency_map = self._compute_saliency_map(gray)

        # -------------------------------------------------------------
        # Step 2: Dynamic Saliency Thresholding (Texture Rejection)
        # -------------------------------------------------------------
        binary_mask = self._segment_candidates(saliency_map)

        # -------------------------------------------------------------
        # Step 3: Raw Contour Extraction & Fragment Merging
        # -------------------------------------------------------------
        candidate_boxes = self._extract_and_merge_candidates(binary_mask, h, w)

        # -------------------------------------------------------------
        # Step 4: Multi-Criteria Defect Validation & Classification
        # -------------------------------------------------------------
        validated_defects: List[DetectedDefect] = []
        final_defect_mask = np.zeros((h, w), dtype=np.uint8)
        total_defect_pixel_area = 0.0

        for box, mask_patch in candidate_boxes:
            validated = self._validate_and_classify_defect(box, mask_patch, gray)
            if validated is not None and validated.confidence >= self.thresholds.min_display_confidence:
                validated_defects.append(validated)
                # Overlay onto final mask
                bx, by, bw, bh = validated.bounding_box.x, validated.bounding_box.y, validated.bounding_box.width, validated.bounding_box.height
                final_defect_mask[by:by+bh, bx:bx+bw] = cv2.bitwise_or(final_defect_mask[by:by+bh, bx:bx+bw], mask_patch)
                total_defect_pixel_area += validated.area

        # -------------------------------------------------------------
        # Step 5: Visual Annotation & False-Color Heatmap
        # -------------------------------------------------------------
        annotated = prep.bgr.copy()
        for d in validated_defects:
            bx, by, bw, bh = d.bounding_box.x, d.bounding_box.y, d.bounding_box.width, d.bounding_box.height

            # Color coding by defect type
            if d.defect_type == "SCRATCH":
                color = (0, 165, 255)   # Orange in BGR
            elif d.defect_type == "CRACK_LIKE":
                color = (0, 0, 255)     # Red in BGR
            elif d.defect_type == "BLEMISH":
                color = (255, 0, 255)   # Magenta in BGR
            else:
                color = (0, 255, 255)   # Yellow in BGR

            # Draw bounding box
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)
            label_text = f"{d.defect_type} ({int(d.confidence * 100)}%)"

            # Draw label banner
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (bx, max(0, by - th - 6)), (bx + tw + 6, max(th + 6, by)), color, -1)
            cv2.putText(annotated, label_text, (bx + 3, max(th + 2, by - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Generate smooth spatial density heatmap
        density_map = cv2.GaussianBlur(final_defect_mask.astype(np.float32), (31, 31), 10.0)
        norm_density = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(norm_density, cv2.COLORMAP_JET)

        # Mask out zero-density background
        heatmap_bgr[norm_density < 10] = prep.bgr[norm_density < 10]

        defect_density_pct = round((total_defect_pixel_area / total_pixels) * 100.0, 3)

        return DefectDetectionResult(
            total_defects=len(validated_defects),
            defects=validated_defects,
            defect_mask=final_defect_mask,
            heatmap_bgr=heatmap_bgr,
            annotated_image=annotated,
            defect_density=defect_density_pct
        )

    def _compute_saliency_map(self, gray: np.ndarray) -> np.ndarray:
        """
        Extracts multi-scale high-contrast morphological features.
        Combines Black-Hat (dark anomalies), Top-Hat (bright anomalies),
        and large-kernel background subtraction.
        """
        k_size = self.thresholds.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        # Top-Hat isolates bright anomalies on dark backgrounds
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        # Black-Hat isolates dark scratches/cracks on bright backgrounds
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        morph_diff = cv2.add(top_hat, black_hat)

        # Background estimation via large Gaussian filter captures larger solid blobs
        bg_estimate = cv2.GaussianBlur(gray, (35, 35), 0)
        bg_diff = cv2.absdiff(gray, bg_estimate)

        # Combined saliency
        saliency = cv2.max(morph_diff, bg_diff)
        return cv2.GaussianBlur(saliency, (3, 3), 0)

    def _segment_candidates(self, saliency: np.ndarray) -> np.ndarray:
        """
        Dynamically thresholds the saliency map using conservative contrast gating.
        Prevents normal surface textures with low contrast variations from generating contours.
        """
        mean_sal = float(np.mean(saliency))
        std_sal = float(np.std(saliency))

        # Threshold requires both absolute contrast AND statistical deviation from background
        dynamic_thresh = max(
            float(self.thresholds.min_saliency_threshold),
            mean_sal + 2.8 * std_sal
        )

        _, binary_raw = cv2.threshold(saliency, int(dynamic_thresh), 255, cv2.THRESH_BINARY)

        # Connect split defect segments and eliminate 1-2px micro-speckles
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_closed = cv2.morphologyEx(binary_raw, cv2.MORPH_CLOSE, close_kernel)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_clean = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, open_kernel)

        return binary_clean

    def _extract_and_merge_candidates(self, binary_mask: np.ndarray, h: int, w: int) -> List[Tuple[BoundingBox, np.ndarray]]:
        """
        Extracts candidate contours and merges nearby/fragmented bounding boxes
        within merge_distance_px into single consolidated candidates.
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        initial_boxes = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.thresholds.min_defect_area or area > self.thresholds.max_defect_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < self.thresholds.min_bounding_box_dim or bh < self.thresholds.min_bounding_box_dim:
                continue

            initial_boxes.append((bx, by, bw, bh))

        if not initial_boxes:
            return []

        # Merge overlapping or nearby bounding boxes
        merge_dist = self.thresholds.merge_distance_px
        merged = []
        used = [False] * len(initial_boxes)

        for i in range(len(initial_boxes)):
            if used[i]:
                continue
            x1, y1, w1, h1 = initial_boxes[i]
            cur_x, cur_y = x1, y1
            cur_r, cur_b = x1 + w1, y1 + h1
            used[i] = True

            # Iteratively absorb any other box within merge_dist
            changed = True
            while changed:
                changed = False
                for j in range(len(initial_boxes)):
                    if not used[j]:
                        jx, jy, jw, jh = initial_boxes[j]
                        jr, jb = jx + jw, jy + jh

                        # Check proximity
                        if not (cur_r + merge_dist < jx or jr + merge_dist < cur_x or
                                cur_b + merge_dist < jy or jb + merge_dist < cur_y):
                            cur_x = min(cur_x, jx)
                            cur_y = min(cur_y, jy)
                            cur_r = max(cur_r, jr)
                            cur_b = max(cur_b, jb)
                            used[j] = True
                            changed = True

            merged_box = BoundingBox(x=int(cur_x), y=int(cur_y), width=int(cur_r - cur_x), height=int(cur_b - cur_y))
            # Extract corresponding sub-mask patch
            patch = binary_mask[merged_box.y:merged_box.y+merged_box.height, merged_box.x:merged_box.x+merged_box.width].copy()
            merged.append((merged_box, patch))

        return merged

    def _validate_and_classify_defect(
        self,
        box: BoundingBox,
        mask_patch: np.ndarray,
        gray: np.ndarray
    ) -> Optional[DetectedDefect]:
        """
        Applies rigorous multi-criteria validation (true local contrast, aspect ratio,
        circularity, solidity) to classify defect candidate.
        Rejects candidates that fail to meet strict evidence thresholds.
        """
        # Calculate pixel area within mask patch
        defect_pixel_count = int(np.count_nonzero(mask_patch))
        if defect_pixel_count < self.thresholds.min_defect_area:
            return None

        # Extract image patch
        img_patch = gray[box.y:box.y+box.height, box.x:box.x+box.width]
        if img_patch.size == 0:
            return None

        # Calculate True Local Contrast Delta:
        # Measure intensity inside defect region vs surrounding dilated border
        defect_pixels = img_patch[mask_patch > 0]
        if len(defect_pixels) == 0:
            return None

        defect_mean_intensity = float(np.mean(defect_pixels))

        # Dilate mask to sample surrounding background
        dilated_mask = cv2.dilate(mask_patch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        bg_mask = cv2.subtract(dilated_mask, mask_patch)
        bg_pixels = img_patch[bg_mask > 0]

        if len(bg_pixels) > 0:
            bg_mean_intensity = float(np.mean(bg_pixels))
            local_contrast = abs(defect_mean_intensity - bg_mean_intensity)
        else:
            # Fallback to local image patch standard deviation
            local_contrast = float(np.std(img_patch))

        # Reject candidate if local contrast is too weak (normal texture fluctuations)
        if local_contrast < self.thresholds.min_defect_contrast:
            return None

        # Geometric Descriptors
        major = max(box.width, box.height)
        minor = max(1, min(box.width, box.height))
        aspect_ratio = round(float(major) / float(minor), 2)

        # Contour extraction inside patch for precise geometric metrics
        patch_contours, _ = cv2.findContours(mask_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not patch_contours:
            return None

        main_cnt = max(patch_contours, key=cv2.contourArea)
        area = float(cv2.contourArea(main_cnt))
        if area <= 0:
            area = float(defect_pixel_count)

        perimeter = float(cv2.arcLength(main_cnt, True))
        circularity = round((4.0 * np.pi * area) / (perimeter * perimeter), 3) if perimeter > 0 else 0.0

        hull = cv2.convexHull(main_cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = round(area / hull_area, 3) if hull_area > 0 else 1.0

        # -------------------------------------------------------------
        # Classification Engine with Strict Evidence Requirements
        # -------------------------------------------------------------
        # 1. SCRATCH: Elongated line with sufficient length and aspect ratio
        if (
            aspect_ratio >= self.thresholds.scratch_aspect_ratio_min
            and major >= self.thresholds.scratch_min_length
            and local_contrast >= self.thresholds.scratch_min_contrast
        ):
            defect_type = "SCRATCH"
            conf = min(0.98, 0.55 + 0.25 * (aspect_ratio / 8.0) + 0.20 * (local_contrast / 60.0))

        # 2. CRACK_LIKE: Irregular branching geometry, concavities, structural length
        elif (
            circularity <= self.thresholds.crack_max_circularity
            and area >= self.thresholds.crack_min_area
            and solidity <= self.thresholds.crack_max_solidity
            and major >= self.thresholds.crack_min_length
        ):
            defect_type = "CRACK_LIKE"
            conf = min(0.96, 0.55 + 0.25 * (1.0 - circularity) + 0.20 * (1.0 - solidity))

        # 3. BLEMISH: Compact circular/oval discolored spot
        elif (
            circularity >= self.thresholds.blemish_min_circularity
            and area >= self.thresholds.blemish_min_area
            and aspect_ratio <= self.thresholds.blemish_max_aspect_ratio
            and local_contrast >= self.thresholds.blemish_min_contrast
        ):
            defect_type = "BLEMISH"
            conf = min(0.95, 0.55 + 0.25 * circularity + 0.20 * (local_contrast / 60.0))

        # 4. CONTAMINATION_LIKE: Larger localized foreign patch
        elif (
            area >= self.thresholds.contamination_min_area
            and local_contrast >= self.thresholds.contamination_min_contrast
        ):
            defect_type = "CONTAMINATION_LIKE"
            conf = min(0.90, 0.50 + 0.25 * (area / 300.0) + 0.25 * (local_contrast / 60.0))

        else:
            # Rejects unstructured texture noise that lacks decisive geometric evidence
            return None

        # -------------------------------------------------------------
        # Severity Calibration
        # -------------------------------------------------------------
        if area > 450 or (defect_type == "CRACK_LIKE" and area > 180) or local_contrast > 70:
            severity = "high"
        elif area > 120 or local_contrast > 38:
            severity = "medium"
        else:
            severity = "low"

        return DetectedDefect(
            defect_type=defect_type,
            severity=severity,
            confidence=round(conf, 2),
            bounding_box=box,
            area=round(area, 1),
            aspect_ratio=aspect_ratio,
            circularity=circularity,
            solidity=solidity,
            local_contrast=round(local_contrast, 1)
        )
