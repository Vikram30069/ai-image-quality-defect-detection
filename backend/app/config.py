"""
Application Configuration Module
--------------------------------
Centralized configuration management for the AI-Powered Image Quality &
Defect Detection System. Avoids magic numbers and keeps thresholds
configurable across development, testing, and production.
"""

from pathlib import Path
from pydantic import BaseModel


class QualityThresholds(BaseModel):
    """Calibrated baseline thresholds for quality issue detection."""
    # Sharpness / Blur thresholds (Laplacian variance)
    blur_critical: float = 80.0
    blur_warning: float = 180.0

    # Exposure / Brightness thresholds (0-255 scale)
    underexposed_threshold: float = 60.0
    overexposed_threshold: float = 200.0

    # Contrast threshold (standard deviation of pixel intensities)
    low_contrast_threshold: float = 30.0

    # Noise threshold (MAD estimator)
    high_noise_threshold: float = 8.5

    # Entropy thresholds (bits)
    low_entropy_threshold: float = 3.5

    # Edge density threshold (fraction of edge pixels)
    low_edge_density_threshold: float = 0.005


class DefectThresholds(BaseModel):
    """Configurable thresholds for classical CV defect segmentation and false positive rejection."""
    # General Contour & Saliency Thresholds
    min_defect_area: int = 45               # Rejects micro-noise speckles (< 45 px)
    max_defect_area: int = 50000            # Upper bound for single localized defect
    min_bounding_box_dim: int = 5           # Minimum width or height (rejects 1-2px single-pixel noise lines)
    min_defect_contrast: float = 22.0       # Minimum local intensity deviation from background (0-255 scale)
    min_saliency_threshold: float = 32.0    # Absolute contrast threshold for defect saliency map
    morph_kernel_size: int = 25             # Structuring element size for background isolation
    
    # Scratch-Specific Criteria
    scratch_aspect_ratio_min: float = 3.2   # Elongated geometry (length / width >= 3.2)
    scratch_min_length: int = 22            # Major axis dimension must be at least 22px
    scratch_min_contrast: float = 22.0      # Scratches must have distinct contrast
    
    # Crack-Specific Criteria
    crack_min_area: int = 70                # Cracks must have structural area >= 70px
    crack_max_circularity: float = 0.38     # Irregular branching perimeter (low circularity)
    crack_max_solidity: float = 0.72        # Structural concavities (solidity < 0.72)
    crack_min_length: int = 25              # Minimum crack trace length
    
    # Blemish-Specific Criteria
    blemish_min_circularity: float = 0.62   # Approximately circular/oval spot
    blemish_min_area: int = 45              # Localized spot minimum area
    blemish_min_contrast: float = 25.0      # Blemishes must exhibit clear discoloration
    blemish_max_aspect_ratio: float = 2.2   # Blemishes are compact, not elongated
    
    # Contamination-Specific Criteria
    contamination_min_area: int = 80        # Foreign contamination patch minimum area
    contamination_min_contrast: float = 24.0 # Color/intensity deviation
    
    # Spatial Clustering & Confidence
    merge_distance_px: int = 16             # Merge nearby fragmented defect bounding boxes (within 16px)
    min_display_confidence: float = 0.45    # Only display defect candidates above 45% confidence


class Settings(BaseModel):
    """Global application settings."""
    PROJECT_NAME: str = "AI-Powered Image Quality & Defect Detection"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api"

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    SAMPLE_IMAGES_DIR: Path = BASE_DIR / "sample_images"
    ML_DIR: Path = BASE_DIR / "ml"
    MODEL_PATH: Path = BASE_DIR / "backend" / "app" / "ml" / "quality_model.pkl"
    MODEL_META_PATH: Path = BASE_DIR / "backend" / "app" / "ml" / "model_metadata.json"
    DATABASE_PATH: Path = BASE_DIR / "backend" / "inspection_system.db"

    # Upload validation constraints
    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
    MAX_IMAGE_DIMENSION: int = 3840  # Max 4K dimension; larger images will be safely resized

    # Quality Weights for Rule-Based Aggregation (Sum = 1.0)
    WEIGHT_SHARPNESS: float = 0.35
    WEIGHT_EXPOSURE: float = 0.25
    WEIGHT_NOISE: float = 0.15
    WEIGHT_CONTRAST: float = 0.15
    WEIGHT_OTHER: float = 0.10

    # Quality and Defect Thresholds
    quality: QualityThresholds = QualityThresholds()
    defects: DefectThresholds = DefectThresholds()


# Global Singleton Instance
settings = Settings()

# Ensure runtime directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
