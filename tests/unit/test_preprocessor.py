"""Tests for image preprocessor."""
import numpy as np

from fitscan.cv_pipeline.preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    preprocessor = ImagePreprocessor()

    def test_extract_silhouette_from_rgba(self):
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[20:80, 20:80, :3] = 200
        rgba[20:80, 20:80, 3] = 255
        mask = self.preprocessor.extract_silhouette(rgba)
        assert mask.shape == (100, 100)
        assert np.sum(mask > 0) > 0

    def test_extract_silhouette_from_rgb(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[20:80, 20:80] = 200
        mask = self.preprocessor.extract_silhouette(rgb)
        assert mask.shape == (100, 100)

    def test_process_returns_all_fields(self):
        img = np.ones((500, 500, 3), dtype=np.uint8) * 240
        img[100:400, 150:350] = 50
        result = self.preprocessor.process(img)
        assert result.width == 500
        assert result.height == 500
        assert result.original.shape == (500, 500, 3)
        assert result.silhouette_mask.shape == (500, 500)

    def test_threshold_remove_fallback(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 240
        img[20:80, 20:80] = 50
        result = self.preprocessor._threshold_remove(img)
        assert result.shape[2] == 4  # RGBA output
