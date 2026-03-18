"""Tests for geometric measurements from pose keypoints."""
from fitscan.cv_pipeline.geometric import GeometricMeasurer
from fitscan.domain.models import BodyPose, Keypoint, ScaleFactor


class TestGeometricMeasurer:
    def test_measure_returns_shoulder_width(self):
        measurer = GeometricMeasurer()
        pose = BodyPose(
            keypoints={
                "left_shoulder": Keypoint(0.3, 0.2, 0, 1.0),
                "right_shoulder": Keypoint(0.7, 0.2, 0, 1.0),
                "left_hip": Keypoint(0.35, 0.6, 0, 1.0),
                "right_hip": Keypoint(0.65, 0.6, 0, 1.0),
                "left_wrist": Keypoint(0.15, 0.5, 0, 1.0),
                "right_wrist": Keypoint(0.85, 0.5, 0, 1.0),
            },
            image_width=500,
            image_height=1000,
        )
        scale = ScaleFactor(
            pixels_per_cm=5.0, source="test", confidence=1.0
        )
        result = measurer.measure(pose, scale)
        assert result.shoulder_cm is not None
        assert result.shoulder_cm.value_cm > 0
        assert result.source == "geometric"

    def test_measure_handles_missing_keypoints(self):
        measurer = GeometricMeasurer()
        pose = BodyPose(
            keypoints={
                "left_shoulder": Keypoint(0.3, 0.2, 0, 1.0),
            },
            image_width=500,
            image_height=1000,
        )
        scale = ScaleFactor(
            pixels_per_cm=5.0, source="test", confidence=1.0
        )
        result = measurer.measure(pose, scale)
        assert result.shoulder_cm is None
