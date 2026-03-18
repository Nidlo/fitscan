"""Tests for image loader."""
import numpy as np
import pytest

from fitscan.domain.exceptions import ImageQualityError
from fitscan.infrastructure.image_loader import ImageLoader


class TestImageLoader:
    loader = ImageLoader()

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load("/nonexistent/path.jpg")

    def test_validate_quality_rejects_small_image(self):
        small = np.zeros((100, 100, 3), dtype=np.uint8)
        ok, reason = self.loader.validate_quality(small)
        assert not ok
        assert "too small" in reason.lower()

    def test_validate_quality_accepts_large_sharp_image(self):
        img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        ok, _ = self.loader.validate_quality(img)
        # Random noise has high variance → passes blur check
        assert ok
