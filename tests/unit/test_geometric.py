"""
Numerical regression tests for GeometricMeasurer.

Each test pins exact expected values computed analytically from the input
keypoints and scale. Tolerances are ±0.2 cm to catch formula drift while
tolerating Python float arithmetic.
"""
import math

import pytest

from fitscan.cv_pipeline.geometric import GeometricMeasurer
from fitscan.domain.models import BodyPose, Keypoint, ScaleFactor


def make_standard_pose(img_w: int = 500, img_h: int = 1000) -> BodyPose:
    """
    Canonical pose fixture:
      left_shoulder  (0.3, 0.2) → pixel (150, 200)
      right_shoulder (0.7, 0.2) → pixel (350, 200)  shoulder_width = 200 px
      left_hip       (0.35, 0.6) → pixel (175, 600)
      right_hip      (0.65, 0.6) → pixel (325, 600)
      left_wrist     (0.15, 0.5) → pixel (75, 500)

    With scale 5.0 px/cm:
      shoulder       = 200 / 5.0 = 40.0 cm
      shoulder_to_hip (vertical) = |600 - 200| = 400 px → 80.0 cm
      shoulder_to_waist = 400 * 0.6 = 240 px → 48.0 cm
      sleeve (euclidean) = sqrt((150-75)^2 + (200-500)^2)
                         = sqrt(75^2 + 300^2) = sqrt(96 825) ≈ 311.2 px → 62.2 cm
    """
    return BodyPose(
        keypoints={
            "left_shoulder": Keypoint(0.3, 0.2, 0, 1.0),
            "right_shoulder": Keypoint(0.7, 0.2, 0, 1.0),
            "left_hip": Keypoint(0.35, 0.6, 0, 1.0),
            "right_hip": Keypoint(0.65, 0.6, 0, 1.0),
            "left_wrist": Keypoint(0.15, 0.5, 0, 1.0),
            "right_wrist": Keypoint(0.85, 0.5, 0, 1.0),
        },
        image_width=img_w,
        image_height=img_h,
    )


SCALE_5 = ScaleFactor(pixels_per_cm=5.0, source="test", confidence=1.0)


class TestGeometricMeasurerRegression:
    measurer = GeometricMeasurer()

    def test_shoulder_width_exact(self):
        """
        Shoulder width = euclidean((150,200)→(350,200)) = 200 px / 5.0 = 40.0 cm.
        """
        result = self.measurer.measure(make_standard_pose(), SCALE_5)
        assert result.shoulder_cm is not None
        assert abs(result.shoulder_cm.value_cm - 40.0) < 0.2, (
            f"Expected 40.0 cm, got {result.shoulder_cm.value_cm}"
        )
        assert result.shoulder_cm.source == "geometric"
        assert result.source == "geometric"

    def test_shoulder_to_hip_exact(self):
        """Vertical distance: |y_hip - y_shoulder| = 400 px / 5.0 = 80.0 cm."""
        result = self.measurer.measure(make_standard_pose(), SCALE_5)
        assert result.shoulder_to_hip_cm is not None
        assert abs(result.shoulder_to_hip_cm.value_cm - 80.0) < 0.2, (
            f"Expected 80.0 cm, got {result.shoulder_to_hip_cm.value_cm}"
        )

    def test_shoulder_to_waist_is_60_percent_of_hip(self):
        """Waist proxy = 60 % of shoulder_to_hip → 48.0 cm."""
        result = self.measurer.measure(make_standard_pose(), SCALE_5)
        assert result.shoulder_to_waist_cm is not None
        assert abs(result.shoulder_to_waist_cm.value_cm - 48.0) < 0.2, (
            f"Expected 48.0 cm, got {result.shoulder_to_waist_cm.value_cm}"
        )

    def test_sleeve_length_exact(self):
        """
        Euclidean (150,200)→(75,500) = sqrt(75² + 300²) = sqrt(96825) ≈ 311.2 px
        / 5.0 = 62.2 cm.
        """
        expected_px = math.sqrt(75**2 + 300**2)
        expected_cm = expected_px / 5.0
        result = self.measurer.measure(make_standard_pose(), SCALE_5)
        assert result.sleeve_length_cm is not None
        assert abs(result.sleeve_length_cm.value_cm - expected_cm) < 0.2, (
            f"Expected {expected_cm:.1f} cm, got {result.sleeve_length_cm.value_cm}"
        )

    def test_scale_proportionality(self):
        """Doubling px/cm halves all centimetre measurements."""
        result_5 = self.measurer.measure(make_standard_pose(), SCALE_5)
        scale_10 = ScaleFactor(pixels_per_cm=10.0, source="test", confidence=1.0)
        result_10 = self.measurer.measure(make_standard_pose(), scale_10)

        assert result_5.shoulder_cm is not None
        assert result_10.shoulder_cm is not None
        ratio = result_5.shoulder_cm.value_cm / result_10.shoulder_cm.value_cm
        assert abs(ratio - 2.0) < 0.05, (
            f"Expected 2× ratio, got {ratio:.4f}"
        )

    def test_confidence_values_are_correct(self):
        result = self.measurer.measure(make_standard_pose(), SCALE_5)
        assert result.shoulder_cm is not None
        assert result.shoulder_cm.confidence == pytest.approx(0.75)
        assert result.shoulder_to_waist_cm is not None
        assert result.shoulder_to_waist_cm.confidence == pytest.approx(0.72)

    def test_missing_shoulder_keypoints_returns_none(self):
        pose = BodyPose(
            keypoints={"left_shoulder": Keypoint(0.3, 0.2, 0, 1.0)},
            image_width=500,
            image_height=1000,
        )
        result = self.measurer.measure(pose, SCALE_5)
        assert result.shoulder_cm is None
        assert result.shoulder_to_hip_cm is None
        assert result.sleeve_length_cm is None

    def test_no_wrist_gives_none_sleeve(self):
        pose = BodyPose(
            keypoints={
                "left_shoulder": Keypoint(0.3, 0.2, 0, 1.0),
                "right_shoulder": Keypoint(0.7, 0.2, 0, 1.0),
                "left_hip": Keypoint(0.35, 0.6, 0, 1.0),
                "right_hip": Keypoint(0.65, 0.6, 0, 1.0),
            },
            image_width=500,
            image_height=1000,
        )
        result = self.measurer.measure(pose, SCALE_5)
        assert result.shoulder_cm is not None
        assert result.sleeve_length_cm is None

    def test_gender_profile_propagated(self):
        result = self.measurer.measure(
            make_standard_pose(), SCALE_5, gender_profile="mens"
        )
        assert result.gender_profile == "mens"
