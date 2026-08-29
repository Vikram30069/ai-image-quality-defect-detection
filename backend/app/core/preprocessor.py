"""
Image Preprocessing Module
--------------------------
Handles image validation, decoding from raw bytes or files, dimension inspection,
color-space conversions (Grayscale, HSV, LAB), and safe resizing.

Design Rationale:
- Preserves raw pixel dynamics to ensure quality features (noise, blur) are authentic.
- Provides standard color representations used downstream by feature extraction
  and defect segmentation routines.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class PreprocessedImage:
    """
    Data container holding all decoded representations and metadata
    for a single processed image.
    """
    bgr: np.ndarray          # Original standard OpenCV color image (H, W, 3)
    gray: np.ndarray         # Grayscale representation for luminance & edge analysis
    hsv: np.ndarray          # Hue, Saturation, Value for color/saturation metrics
    lab: np.ndarray          # CIELAB representation for perceptual color anomaly analysis
    height: int              # Image height in pixels
    width: int               # Image width in pixels
    channels: int            # Number of color channels (usually 3)
    aspect_ratio: float      # Width / Height
    original_size_bytes: int # Size of the raw binary image in bytes
    was_resized: bool        # True if image was downscaled for computational safety


class ImagePreprocessor:
    """
    Validates and transforms raw image inputs into structured, multi-space
    representations for downstream CV and ML pipelines.
    """

    def __init__(self, max_dimension: int = 3840):
        """
        Args:
            max_dimension: Maximum allowed width or height before performing
                           aspect-ratio-preserving safe downscaling.
        """
        self.max_dimension = max_dimension

    def validate_and_decode(self, image_bytes: bytes) -> PreprocessedImage:
        """
        Validates binary image payload, safely decodes it via OpenCV,
        and generates required color space representations.

        Args:
            image_bytes: Raw binary bytes from an HTTP file upload or file read.

        Returns:
            PreprocessedImage dataclass instance.

        Raises:
            ValueError: If the buffer is empty, corrupted, or cannot be decoded.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Empty image buffer provided.")

        raw_size = len(image_bytes)

        # Convert bytes to a 1D NumPy unsigned 8-bit integer array
        np_arr = np.frombuffer(image_bytes, np.uint8)

        # Decode image to 3-channel BGR format
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if bgr is None:
            raise ValueError(
                "Image decoding failed: File format unsupported or corrupted image stream."
            )

        h, w, c = bgr.shape
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")

        # Safe downscaling if image exceeds maximum allowed dimension
        was_resized = False
        if max(h, w) > self.max_dimension:
            scale = self.max_dimension / float(max(h, w))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w, c = bgr.shape
            was_resized = True

        # Generate multi-color space representations
        # 1. Grayscale: Used for Laplacian blur, contrast standard deviation, and Canny edges
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2. HSV (Hue, Saturation, Value): Isolates chromatic saturation from intensity
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # 3. LAB: Perceptually uniform color space (L=Luminance, A=Green-Red, B=Blue-Yellow)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        aspect_ratio = round(w / float(h), 4)

        return PreprocessedImage(
            bgr=bgr,
            gray=gray,
            hsv=hsv,
            lab=lab,
            height=h,
            width=w,
            channels=c,
            aspect_ratio=aspect_ratio,
            original_size_bytes=raw_size,
            was_resized=was_resized
        )

    def load_from_path(self, file_path: str) -> PreprocessedImage:
        """
        Convenience method to load and preprocess an image directly from local disk.

        Args:
            file_path: Local filesystem path to the image file.

        Returns:
            PreprocessedImage dataclass instance.
        """
        with open(file_path, "rb") as f:
            return self.validate_and_decode(f.read())
