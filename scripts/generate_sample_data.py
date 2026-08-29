"""
Sample Data Generator Script
----------------------------
Generates 10 high-quality synthetic test images under sample_images/ representing:
1. clean_metal: Pristine brushed metal surface with natural surface texture (0 defects).
2. scratch: Clean metal surface with an obvious elongated scratch.
3. crack: Clean metal surface with a branching structural crack.
4. blemish: Clean metal surface with a localized dark circular blemish.
5. contamination: Clean metal surface with a foreign oil/grease contamination patch.
6. blurred: Clean image with heavy out-of-focus blur.
7. noisy: Clean image with heavy Gaussian sensor grain.
8. underexposed: Clean image with dark, inadequate lighting.
9. overexposed: Clean image with bright highlight clipping / glare.
10. low_contrast: Clean image with washed-out dynamic range.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = WORKSPACE_ROOT / "sample_images"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def create_brushed_metal_surface(h: int = 400, w: int = 400, seed: int = 42) -> np.ndarray:
    """
    Generates a realistic brushed industrial metal plate texture
    with natural horizontal micro-grain and subtle illumination gradient.
    """
    rng = np.random.RandomState(seed)
    
    # Base metallic gray with slight gradient
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    gradient = 180 + 15 * np.sin(x_coords / 60.0) + 10 * np.cos(y_coords / 80.0)
    
    # Brushed horizontal micro-texture lines (simulates normal machine tool marks)
    grain = rng.normal(0, 4.5, (h, w))
    # Streak the grain horizontally to simulate brushed texture
    brushed_kernel = np.ones((1, 9), dtype=np.float32) / 9.0
    brushed_texture = cv2.filter2D(grain, -1, brushed_kernel)
    
    metal_gray = np.clip(gradient + brushed_texture * 3.0, 0, 255).astype(np.uint8)
    # Convert to 3-channel BGR
    metal_bgr = cv2.cvtColor(metal_gray, cv2.COLOR_GRAY2BGR)
    return metal_bgr


def generate_all_samples():
    print("Generating comprehensive 10-case validation dataset in sample_images/...")

    # 1. Clean Pristine Metal Surface (Should have 0 defects)
    clean = create_brushed_metal_surface(seed=42)
    cv2.imwrite(str(SAMPLE_DIR / "sample_clean_metal.png"), clean)

    # 2. Obvious Scratch (Single thin elongated line with high contrast)
    scratch_img = create_brushed_metal_surface(seed=43)
    cv2.line(scratch_img, (60, 200), (340, 215), (40, 40, 40), thickness=3)
    cv2.line(scratch_img, (60, 201), (340, 216), (230, 230, 230), thickness=1)  # highlight edge
    cv2.imwrite(str(SAMPLE_DIR / "sample_scratch.png"), scratch_img)

    # 3. Obvious Branching Crack (Irregular crack trajectory)
    crack_img = create_brushed_metal_surface(seed=44)
    pts = np.array([[80, 80], [130, 150], [160, 180], [210, 230], [240, 310]], np.int32)
    cv2.polylines(crack_img, [pts], isClosed=False, color=(20, 20, 20), thickness=3)
    # Branch
    branch_pts = np.array([[160, 180], [210, 170], [260, 190]], np.int32)
    cv2.polylines(crack_img, [branch_pts], isClosed=False, color=(20, 20, 20), thickness=2)
    cv2.imwrite(str(SAMPLE_DIR / "sample_crack.png"), crack_img)

    # 4. Obvious Circular Blemish (Localized dark spot)
    blemish_img = create_brushed_metal_surface(seed=45)
    cv2.circle(blemish_img, (200, 200), 16, (30, 30, 30), -1)
    cv2.circle(blemish_img, (200, 200), 18, (80, 80, 80), 2)  # soft border
    cv2.imwrite(str(SAMPLE_DIR / "sample_blemish.png"), blemish_img)

    # 5. Obvious Foreign Contamination (Larger organic patch)
    contam_img = create_brushed_metal_surface(seed=46)
    cv2.ellipse(contam_img, (180, 180), (35, 20), 30, 0, 360, (50, 45, 40), -1)
    cv2.imwrite(str(SAMPLE_DIR / "sample_contamination.png"), contam_img)

    # 6. Quality-Only: Blurred Image (Should identify blur, 0 physical defects)
    blurred = cv2.GaussianBlur(create_brushed_metal_surface(seed=47), (35, 35), 10.0)
    cv2.imwrite(str(SAMPLE_DIR / "sample_blurred.png"), blurred)

    # 7. Quality-Only: Noisy Image (Should identify noise, 0 physical defects)
    noisy_base = create_brushed_metal_surface(seed=48).astype(np.float32)
    noise = np.random.normal(0, 35, noisy_base.shape).astype(np.float32)
    noisy = np.clip(noisy_base + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(SAMPLE_DIR / "sample_noisy.png"), noisy)

    # 8. Quality-Only: Underexposed Image (Dark lighting, 0 physical defects)
    underexp = np.clip(create_brushed_metal_surface(seed=49).astype(np.float32) * 0.25, 0, 255).astype(np.uint8)
    cv2.imwrite(str(SAMPLE_DIR / "sample_underexposed.png"), underexp)

    # 9. Quality-Only: Overexposed Image (Highlight clipping, 0 physical defects)
    overexp = np.clip(create_brushed_metal_surface(seed=50).astype(np.float32) * 1.8 + 40, 0, 255).astype(np.uint8)
    cv2.imwrite(str(SAMPLE_DIR / "sample_overexposed.png"), overexp)

    # 10. Quality-Only: Low Contrast Image (Compressed dynamic range, 0 physical defects)
    base_c = create_brushed_metal_surface(seed=51).astype(np.float32)
    mean_val = np.mean(base_c)
    low_c = np.clip(mean_val + (base_c - mean_val) * 0.25, 0, 255).astype(np.uint8)
    cv2.imwrite(str(SAMPLE_DIR / "sample_low_contrast.png"), low_c)

    print(f"Generated 10 validation images in {SAMPLE_DIR}")


if __name__ == "__main__":
    generate_all_samples()
