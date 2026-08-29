"""
Synthetic Dataset Generator Module
----------------------------------
Generates controlled, reproducible synthetic image quality datasets for training
and evaluating the Machine Learning Quality Classifier.

Degradation Pipeline:
- Clean: Procedural textures, patterns, and geometric surfaces.
- Blurred: Multi-scale Gaussian filtering (σ ∈ [2.0, 15.0]).
- Underexposed: Luminance scaling (factor ∈ [0.15, 0.45]).
- Overexposed: Luminance scaling + highlight clipping (factor ∈ [1.6, 2.5]).
- Noisy: Zero-mean Gaussian and Speckle noise addition (σ ∈ [15, 60]).
- Low Contrast: Dynamic range compression around mean (factor ∈ [0.15, 0.4]).
- Compressed / Degraded: Heavy JPEG DCT quantization (quality ∈ [5, 25]).

Data Leakage Prevention:
- Uses disjoint procedural seed sets for Train (seeds 100-299) and Test (seeds 500-599).
"""

import os
import random
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np


def generate_base_image(seed: int, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Generates a rich, high-contrast procedural synthetic surface image
    containing geometric patterns, text-like lines, and gradient textures.
    """
    rng = np.random.RandomState(seed)
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Base gradient
    c1 = rng.randint(40, 200, size=3)
    c2 = rng.randint(40, 200, size=3)
    for y in range(h):
        alpha = y / float(h)
        color = (1.0 - alpha) * c1 + alpha * c2
        img[y, :] = color.astype(np.uint8)

    # 2. Geometric grid / circuit-like tracks
    num_lines = rng.randint(8, 20)
    for _ in range(num_lines):
        pt1 = (rng.randint(0, w), rng.randint(0, h))
        pt2 = (rng.randint(0, w), rng.randint(0, h))
        thickness = rng.randint(1, 4)
        color = [int(x) for x in rng.randint(20, 240, size=3)]
        cv2.line(img, pt1, pt2, color, thickness)

    # 3. Rectangles / components
    num_rects = rng.randint(4, 10)
    for _ in range(num_rects):
        x1 = rng.randint(10, w - 60)
        y1 = rng.randint(10, h - 60)
        rw = rng.randint(20, 50)
        rh = rng.randint(20, 50)
        color = [int(x) for x in rng.randint(30, 230, size=3)]
        cv2.rectangle(img, (x1, y1), (x1 + rw, y1 + rh), color, -1)
        # Add high-contrast border
        cv2.rectangle(img, (x1, y1), (x1 + rw, y1 + rh), (255, 255, 255), 1)

    # 4. Circles / drill holes
    num_circles = rng.randint(3, 8)
    for _ in range(num_circles):
        center = (rng.randint(20, w - 20), rng.randint(20, h - 20))
        radius = rng.randint(5, 20)
        color = [int(x) for x in rng.randint(10, 250, size=3)]
        cv2.circle(img, center, radius, color, -1)

    return img


# -----------------------------------------------------------------------------
# Controlled Degradation Operators
# -----------------------------------------------------------------------------

def apply_blur(img: np.ndarray, severity: str = "medium") -> np.ndarray:
    """Applies Gaussian spatial blur."""
    ksize = 21 if severity == "heavy" else 11
    sigma = 8.0 if severity == "heavy" else 3.5
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_underexposure(img: np.ndarray, factor: float = 0.3) -> np.ndarray:
    """Simulates inadequate lighting / underexposure."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_overexposure(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Simulates harsh glare / highlight clipping."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_gaussian_noise(img: np.ndarray, sigma: float = 35.0) -> np.ndarray:
    """Adds zero-mean Gaussian sensor noise."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_low_contrast(img: np.ndarray, factor: float = 0.25) -> np.ndarray:
    """Compresses pixel dynamic range around the mean intensity."""
    mean_val = np.mean(img)
    low_c = mean_val + (img.astype(np.float32) - mean_val) * factor
    return np.clip(low_c, 0, 255).astype(np.uint8)


def apply_jpeg_compression(img: np.ndarray, quality: int = 10) -> np.ndarray:
    """Applies severe Discrete Cosine Transform (DCT) block quantization."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# -----------------------------------------------------------------------------
# Dataset Generation Routine
# -----------------------------------------------------------------------------

def generate_dataset(output_dir: Path, num_base_samples: int = 40, start_seed: int = 100):
    """
    Generates a full dataset split with balanced classes:
    - ACCEPTABLE: Clean pristine images.
    - DEGRADED: Mild blur, mild noise, moderate contrast shift, mild compression.
    - DEFECTIVE: Heavy blur, severe noise, extreme under/over-exposure, severe compression.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for i in range(num_base_samples):
        seed = start_seed + i
        base = generate_base_image(seed=seed)

        # 1. Class: ACCEPTABLE (Pristine)
        cv2.imwrite(str(output_dir / f"sample_{seed:04d}_acceptable_clean.png"), base)
        count += 1

        # 2. Class: DEGRADED (Mild degradations)
        mild_blur = apply_blur(base, severity="mild")
        cv2.imwrite(str(output_dir / f"sample_{seed:04d}_degraded_mild_blur.png"), mild_blur)

        mild_noise = apply_gaussian_noise(base, sigma=18.0)
        cv2.imwrite(str(output_dir / f"sample_{seed:04d}_degraded_mild_noise.png"), mild_noise)

        low_contrast = apply_low_contrast(base, factor=0.45)
        cv2.imwrite(str(output_dir / f"sample_{seed:04d}_degraded_low_contrast.png"), low_contrast)

        mild_underexp = apply_underexposure(base, factor=0.55)
        cv2.imwrite(str(output_dir / f"sample_{seed:04d}_degraded_mild_underexp.png"), mild_underexp)
        count += 4

        # 3. Class: DEFECTIVE (Severe failures)
        heavy_blur = apply_blur(base, severity="heavy")
        cv2.imwrite(str(output_dir / f"sample_{seed:04d}_defective_heavy_blur.png"), heavy_blur)

        heavy_noise = apply_gaussian_noise(base, sigma=50.0)
        cv2.imwrite(str(output_dir / f"sample_{seed:04d}_defective_heavy_noise.png"), heavy_noise)

        severe_underexp = apply_underexposure(base, factor=0.18)
        cv2.imwrite(str(output_dir / f"sample_{seed:04d}_defective_underexp.png"), severe_underexp)

        severe_overexp = apply_overexposure(base, factor=2.4)
        cv2.imwrite(str(output_dir / f"sample_{seed:04d}_defective_overexp.png"), severe_overexp)

        severe_jpeg = apply_jpeg_compression(base, quality=5)
        cv2.imwrite(str(output_dir / f"sample_{seed:04d}_defective_jpeg_artifact.png"), severe_jpeg)
        count += 5

    print(f"Generated {count} images in {output_dir}")


if __name__ == "__main__":
    base_ml_dir = Path(__file__).resolve().parent / "dataset"
    train_dir = base_ml_dir / "train"
    test_dir = base_ml_dir / "test"

    print("Generating Training Dataset (Seeds 100-149)...")
    generate_dataset(train_dir, num_base_samples=50, start_seed=100)

    print("Generating Independent Test Dataset (Seeds 500-519)...")
    generate_dataset(test_dir, num_base_samples=20, start_seed=500)
    print("Dataset generation completed successfully.")
