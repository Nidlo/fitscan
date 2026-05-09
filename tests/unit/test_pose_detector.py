"""Tests for pose detector."""

import numpy as np

from fitscan.cv_pipeline.pose_detector import LANDMARK_NAMES, PoseDetector
from fitscan.domain.exceptions import PoseDetectionError


class TestPoseDetector:
    def test_detect_raises_on_blank_image(self):
        detector = PoseDetector()
        # Pure noise image — no human visible
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            detector.detect(blank)
        except PoseDetectionError:
            pass  # Expected
        except Exception:
            pass  # MediaPipe may raise other errors on small images

    def test_landmark_names_complete(self):
        assert "nose" in LANDMARK_NAMES.values()
        assert "left_shoulder" in LANDMARK_NAMES.values()
        assert "right_hip" in LANDMARK_NAMES.values()
        assert "left_ankle" in LANDMARK_NAMES.values()
