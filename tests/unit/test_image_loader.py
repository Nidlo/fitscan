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

    def test_load_rejects_oversized_file(self, tmp_path):
        """FS-NIDLO-TEST-09 — ImageLoader.MAX_FILE_SIZE_MB cap. The
        route already streams + caps at 10MB before this code runs,
        but the loader cap is the second line of defence (called
        from CLI / direct code paths that bypass the route). Pinning
        so a refactor can't quietly drop the safety net.
        """
        oversize = tmp_path / "huge.jpg"
        # MAX_FILE_SIZE_MB = 10; write 11MB of zeros.
        oversize.write_bytes(b"\x00" * (11 * 1024 * 1024))
        with pytest.raises(ImageQualityError, match="exceeds"):
            self.loader.load(oversize)

    def test_load_accepts_file_at_size_cap(self, tmp_path):
        """A real JPEG header so cv2.imread doesn't reject before the
        size check has a chance to pass. We construct it via opencv
        rather than fake bytes so the file is genuinely loadable."""
        import cv2

        path = tmp_path / "small.jpg"
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(path), img)
        # File is well under 10MB; should load without raising.
        loaded = self.loader.load(path)
        assert loaded.shape[:2] == (200, 200)
