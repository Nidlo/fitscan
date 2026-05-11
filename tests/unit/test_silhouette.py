"""Tests for silhouette measurement functions."""

import numpy as np

from fitscan.cv_pipeline.silhouette import (
    SilhouetteMeasurer,
    ellipse_circumference,
    px_to_cm,
)
from fitscan.domain.models import BodyPose, Keypoint, ScaleFactor


class TestPxToCm:
    def test_basic_conversion(self):
        scale = ScaleFactor(pixels_per_cm=5.0, source="test", confidence=1.0)
        assert px_to_cm(50.0, scale) == 10.0

    def test_large_scale(self):
        scale = ScaleFactor(pixels_per_cm=10.0, source="test", confidence=1.0)
        assert px_to_cm(100.0, scale) == 10.0


class TestEllipseCircumference:
    def test_circle(self):
        circ = ellipse_circumference(5.0, 5.0)
        assert abs(circ - 2 * np.pi * 5.0) < 0.1

    def test_ellipse_larger_than_circle(self):
        circle = ellipse_circumference(5.0, 5.0)
        ellipse = ellipse_circumference(6.0, 4.0)
        assert ellipse > circle * 0.95  # Ellipse should be reasonably close


class TestSilhouetteMeasurer:
    def test_measure_returns_body_measurements(self):
        measurer = SilhouetteMeasurer()
        mask = np.zeros((800, 400), dtype=np.uint8)
        # Draw a body-like shape
        mask[100:700, 150:250] = 255

        pose = BodyPose(
            keypoints={
                "left_shoulder": Keypoint(0.35, 0.15, 0, 1.0),
                "right_shoulder": Keypoint(0.65, 0.15, 0, 1.0),
                "left_hip": Keypoint(0.40, 0.55, 0, 1.0),
                "right_hip": Keypoint(0.60, 0.55, 0, 1.0),
            },
            image_width=400,
            image_height=800,
        )
        scale = ScaleFactor(pixels_per_cm=5.0, source="test", confidence=1.0)
        result = measurer.measure(mask, pose, scale)
        assert result.source == "silhouette"
        assert result.gender_profile == "ladies"
