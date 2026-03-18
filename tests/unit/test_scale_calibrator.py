"""Tests for ScaleCalibrator. No external dependencies."""
import pytest

from fitscan.cv_pipeline.scale_calibrator import ScaleCalibrator
from fitscan.domain.exceptions import ScaleCalibrationError
from fitscan.domain.models import BodyPose, Keypoint


def make_pose(
    head_y=0.05, ankle_y=0.95, img_h=1000, img_w=500
) -> BodyPose:
    kps = {
        "nose": Keypoint(x=0.5, y=head_y, z=0, visibility=1.0),
        "left_ankle": Keypoint(x=0.4, y=ankle_y, z=0, visibility=1.0),
        "right_ankle": Keypoint(x=0.6, y=ankle_y, z=0, visibility=1.0),
        "left_foot_index": Keypoint(
            x=0.4, y=ankle_y + 0.01, z=0, visibility=0.8
        ),
        "right_foot_index": Keypoint(
            x=0.6, y=ankle_y + 0.01, z=0, visibility=0.8
        ),
    }
    return BodyPose(keypoints=kps, image_width=img_w, image_height=img_h)


class TestScaleCalibrator:
    calibrator = ScaleCalibrator()

    def test_calibrate_from_height_returns_correct_px_per_cm(self):
        pose = make_pose(head_y=0.05, ankle_y=0.95, img_h=1000)
        scale = self.calibrator.calibrate_from_height(pose, 175.0)
        assert abs(scale.pixels_per_cm - 5.50) < 0.20
        assert scale.source == "user_height"
        assert scale.confidence == 0.90

    def test_raises_on_implausible_height(self):
        pose = make_pose()
        with pytest.raises(ScaleCalibrationError):
            self.calibrator.calibrate_from_height(pose, 300.0)
        with pytest.raises(ScaleCalibrationError):
            self.calibrator.calibrate_from_height(pose, 30.0)

    def test_calibrate_from_height_consistency(self):
        pose = make_pose()
        s1 = self.calibrator.calibrate_from_height(pose, 170.0)
        s2 = self.calibrator.calibrate_from_height(pose, 170.0)
        assert s1.pixels_per_cm == s2.pixels_per_cm

    def test_taller_person_lower_px_per_cm(self):
        pose = make_pose()
        s1 = self.calibrator.calibrate_from_height(pose, 150.0)
        s2 = self.calibrator.calibrate_from_height(pose, 190.0)
        assert s1.pixels_per_cm > s2.pixels_per_cm
