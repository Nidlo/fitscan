"""Loads images from disk. Implements IImageLoader."""

from pathlib import Path

import cv2
import numpy as np

from fitscan.domain.exceptions import ImageQualityError


class ImageLoader:
    """Loads images from disk. Implements IImageLoader."""

    MIN_DIMENSION = 480
    MAX_FILE_SIZE_MB = 10

    def load(self, path: Path) -> np.ndarray:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.stat().st_size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ImageQualityError(f"Image exceeds {self.MAX_FILE_SIZE_MB}MB limit.")
        image = cv2.imread(str(path))
        if image is None:
            raise ImageQualityError(f"Could not decode image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def validate_quality(self, image: np.ndarray) -> tuple[bool, str]:
        h, w = image.shape[:2]
        if h < self.MIN_DIMENSION or w < self.MIN_DIMENSION:
            return (
                False,
                f"Image too small: {w}x{h}. Minimum {self.MIN_DIMENSION}px.",
            )
        blur_score = cv2.Laplacian(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F
        ).var()
        if blur_score < 80:
            return (
                False,
                f"Image too blurry (score: {blur_score:.1f}). Retake photo.",
            )
        return True, "OK"
