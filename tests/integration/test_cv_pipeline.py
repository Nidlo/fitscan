"""
Integration tests using synthetic test images.
These images are generated programmatically — no external dataset required.
"""
import cv2
import numpy as np
import pytest
from pathlib import Path

from fitscan.cv_pipeline.preprocessor import ImagePreprocessor
from fitscan.infrastructure.image_loader import ImageLoader


def create_test_image(height_px=800, width_px=400) -> np.ndarray:
    """Create a simple synthetic 'person' image for testing pipeline logic."""
    img = np.ones((height_px, width_px, 3), dtype=np.uint8) * 240
    cx = width_px // 2
    cv2.ellipse(img, (cx, 80), (40, 50), 0, 0, 360, (180, 140, 100), -1)
    cv2.rectangle(img, (cx - 50, 130), (cx + 50, 380), (80, 100, 160), -1)
    cv2.rectangle(img, (cx - 55, 380), (cx + 5, 700), (80, 100, 160), -1)
    cv2.rectangle(img, (cx + 5, 380), (cx + 55, 700), (80, 100, 160), -1)
    cv2.rectangle(
        img, (cx - 100, 130), (cx - 50, 330), (80, 100, 160), -1
    )
    cv2.rectangle(
        img, (cx + 50, 130), (cx + 100, 330), (80, 100, 160), -1
    )
    return img


@pytest.fixture
def test_image_path(tmp_path) -> Path:
    img = create_test_image()
    path = tmp_path / "test_person.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


class TestCVPipeline:
    def test_image_loader_loads_file(self, test_image_path):
        loader = ImageLoader()
        img = loader.load(test_image_path)
        assert img.shape[0] == 800
        assert img.shape[1] == 400
        assert img.shape[2] == 3

    def test_preprocessor_returns_processed_image(self, test_image_path):
        loader = ImageLoader()
        img = loader.load(test_image_path)
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(img)
        assert result.silhouette_mask.shape[:2] == img.shape[:2]
        assert result.width > 0 and result.height > 0

    def test_silhouette_extraction(self, test_image_path):
        loader = ImageLoader()
        img = loader.load(test_image_path)
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(img)
        # Silhouette should have some non-zero pixels
        assert np.sum(result.silhouette_mask > 0) > 0
