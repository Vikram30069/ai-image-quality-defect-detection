"""
Feature Extractor Module
------------------------
Extracts seven interpretable numerical image quality features from preprocessed
image representations.

Feature Vector Definition (Strictly Ordered for Model Consistency):
1. sharpness     - Variance of the 2D Laplacian operator (Laplacian Variance).
2. brightness    - Mean pixel intensity of the grayscale image [0, 255].
3. contrast      - Standard deviation of grayscale pixel intensities [0, 127.5].
4. noise         - Robust noise estimation via Median Absolute Deviation (MAD) of residual.
5. entropy       - Shannon entropy measuring grayscale intensity information distribution.
6. saturation    - Normalized mean color saturation from HSV S-channel [0.0, 1.0].
7. edge_density  - Fraction of pixels classified as edges using Canny operator [0.0, 1.0].
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np
from backend.app.core.preprocessor import PreprocessedImage


# Canonical list of feature names in strict order
FEATURE_NAMES: List[str] = [
    "sharpness",
    "brightness",
    "contrast",
    "noise",
    "entropy",
    "saturation",
    "edge_density"
]


class FeatureExtractor:
    """
    Extracts statistical and computer-vision quality features from an image.
    Ensures identical feature engineering across ML training and API inference.
    """

    @staticmethod
    def calculate_sharpness(gray: np.ndarray) -> float:
        """
        Calculates image sharpness using the Variance of Laplacian method.

        Formula:
            L(x,y) = ∇²I(x,y) = (∂²I/∂x²) + (∂²I/∂y²)
            Sharpness = Var(L) = (1/N) * Σ (L(x,y) - μ_L)²

        Explanation:
            The Laplacian operator computes the second spatial derivative, which produces
            sharp high-amplitude spikes at edges. In a sharp image, there is high edge variation
            resulting in large variance. In a blurry image, edges are smoothed out, leading to
            a low Laplacian variance.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        variance = float(laplacian.var())
        return max(0.0, round(variance, 4))

    @staticmethod
    def calculate_brightness(gray: np.ndarray) -> float:
        """
        Calculates mean luminance across the grayscale image.

        Formula:
            Brightness = (1/N) * Σ I(x,y)   for all pixels (x,y) in image I.
            Scale: [0.0 (pitch black) to 255.0 (pure white)].
        """
        mean_val = float(np.mean(gray))
        return round(mean_val, 4)

    @staticmethod
    def calculate_contrast(gray: np.ndarray) -> float:
        """
        Calculates image contrast as the standard deviation of pixel intensities.

        Formula:
            Contrast = sqrt( (1/N) * Σ (I(x,y) - μ)² )
            Scale: Low values indicate washed-out/flat tones; high values indicate dynamic range.
        """
        std_val = float(np.std(gray))
        return round(std_val, 4)

    @staticmethod
    def calculate_noise(gray: np.ndarray) -> float:
        """
        Estimates image noise using Median Absolute Deviation (MAD) on high-pass residuals.

        Formula:
            Residual R = | I - MedianFilter(I, kernel=3) |
            Noise σ = median(|R - median(R)|) / 0.6745

        Explanation:
            Subtracting a small median-filtered version of the image removes low-frequency
            structural content, leaving behind high-frequency noise and fine textures.
            The MAD is a statistically robust scale estimator that ignores outliers (e.g. true edges)
            and isolates Gaussian/sensor noise standard deviation.
        """
        median_filtered = cv2.medianBlur(gray, 3)
        residual = np.abs(gray.astype(np.float64) - median_filtered.astype(np.float64))
        
        # Calculate Median Absolute Deviation (MAD)
        median_res = np.median(residual)
        mad = np.median(np.abs(residual - median_res))
        
        # Normalization factor for Gaussian distribution: 1 / norm.ppf(0.75) ≈ 1 / 0.67448975 ≈ 1.4826
        noise_sigma = mad * 1.4826
        return max(0.0, round(float(noise_sigma), 4))

    @staticmethod
    def calculate_entropy(gray: np.ndarray) -> float:
        """
        Calculates the Shannon Entropy of the grayscale histogram.

        Formula:
            H = - Σ (p_i * log2(p_i))   for all non-zero histogram probabilities p_i.

        Explanation:
            Measures the uncertainty and information density in the image.
            A flat/blank image has low entropy (~0 bits), whereas a rich, well-exposed image
            with diverse textures typically has higher entropy (~6 to 8 bits).
        """
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        total_pixels = gray.size
        prob = hist / float(total_pixels)
        
        # Filter out zero probabilities to avoid log2(0)
        nonzero_prob = prob[prob > 0]
        entropy = -float(np.sum(nonzero_prob * np.log2(nonzero_prob)))
        return max(0.0, round(entropy, 4))

    @staticmethod
    def calculate_saturation(hsv: np.ndarray) -> float:
        """
        Calculates the normalized average color saturation from the HSV S-channel.

        Scale:
            [0.0 (completely grayscale/monochrome) to 1.0 (fully vivid/saturated)].
        """
        s_channel = hsv[:, :, 1]
        mean_saturation = float(np.mean(s_channel)) / 255.0
        return round(mean_saturation, 4)

    @staticmethod
    def calculate_edge_density(gray: np.ndarray) -> float:
        """
        Calculates the normalized edge density using the Canny edge detector.

        Formula:
            Edge Density = (Count of Canny edge pixels) / (Total pixel count)
            Scale: [0.0, 1.0].
        """
        # Calibrate thresholds relative to image contrast
        low_thresh = max(10, int(0.66 * np.mean(gray)))
        high_thresh = min(250, int(1.33 * np.mean(gray)))
        if low_thresh >= high_thresh:
            low_thresh, high_thresh = 50, 150

        edges = cv2.Canny(gray, low_thresh, high_thresh)
        edge_count = int(np.count_nonzero(edges))
        edge_density = float(edge_count) / float(gray.size)
        return round(edge_density, 4)

    def extract_features(self, prep: PreprocessedImage) -> Tuple[Dict[str, float], List[float]]:
        """
        Extracts all seven canonical features from a preprocessed image.

        Args:
            prep: PreprocessedImage instance containing gray, hsv, and dimensions.

        Returns:
            Tuple containing:
            1. feature_dict: Dictionary mapping feature names to their numeric values.
            2. feature_vector: Ordered Python list of floats matching FEATURE_NAMES.
        """
        sharpness = self.calculate_sharpness(prep.gray)
        brightness = self.calculate_brightness(prep.gray)
        contrast = self.calculate_contrast(prep.gray)
        noise = self.calculate_noise(prep.gray)
        entropy = self.calculate_entropy(prep.gray)
        saturation = self.calculate_saturation(prep.hsv)
        edge_density = self.calculate_edge_density(prep.gray)

        feature_dict = {
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast,
            "noise": noise,
            "entropy": entropy,
            "saturation": saturation,
            "edge_density": edge_density
        }

        # Ensure strict ordering matching FEATURE_NAMES
        feature_vector = [feature_dict[name] for name in FEATURE_NAMES]

        return feature_dict, feature_vector
