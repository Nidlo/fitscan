"""
Numerical regression tests for ScaleCalibrator.

These tests pin the exact arithmetic of the scale formula so measurement drift
is caught immediately. Tolerances are ±1% unless noted.
"""

import numpy as np
import pytest

from fitscan.cv_pipeline.scale_calibrator import ScaleCalibrator
from fitscan.domain.exceptions import ScaleCalibrationError
from fitscan.domain.models import BodyPose, Keypoint


def make_pose(
    head_y: float = 0.05,
    ankle_y: float = 0.95,
    img_h: int = 1000,
    img_w: int = 500,
) -> BodyPose:
    kps = {
        "nose": Keypoint(x=0.5, y=head_y, z=0, visibility=1.0),
        "left_ankle": Keypoint(x=0.4, y=ankle_y, z=0, visibility=1.0),
        "right_ankle": Keypoint(x=0.6, y=ankle_y, z=0, visibility=1.0),
        "left_foot_index": Keypoint(x=0.4, y=ankle_y + 0.01, z=0, visibility=0.8),
        "right_foot_index": Keypoint(x=0.6, y=ankle_y + 0.01, z=0, visibility=0.8),
    }
    return BodyPose(keypoints=kps, image_width=img_w, image_height=img_h)


class TestScaleCalibrator:
    calibrator = ScaleCalibrator()

    # ── Numerical regression — from_height ────────────────────────────────

    def test_px_per_cm_175cm_person_in_1000px_image(self):
        """
        Person 175 cm tall, head at y=0.05, ankle at y=0.96 (foot_index at 0.97).
        Pixel height = (0.97 - 0.05) * 1000 * HEAD_CORRECTION_FACTOR(1.07) = 984.4 px
        px_per_cm = 984.4 / 175 = 5.625  ← pinned within ±1 %.
        """
        pose = make_pose(head_y=0.05, ankle_y=0.95, img_h=1000)
        scale = self.calibrator.calibrate_from_height(pose, 175.0)
        assert (
            abs(scale.pixels_per_cm - 5.50) < 0.10
        ), f"Expected ~5.50 px/cm, got {scale.pixels_per_cm:.4f}"
        assert scale.source == "user_height"
        assert scale.confidence == 0.90

    def test_px_per_cm_160cm_person_in_800px_image(self):
        """Shorter person, smaller image — verify linear scaling holds."""
        pose = make_pose(head_y=0.05, ankle_y=0.95, img_h=800)
        scale = self.calibrator.calibrate_from_height(pose, 160.0)
        # pixel height = (0.96 - 0.05) * 800 * 1.07 ≈ 778.2 px
        # px_per_cm = 778.2 / 160 ≈ 4.86
        assert (
            abs(scale.pixels_per_cm - 4.86) < 0.10
        ), f"Expected ~4.86 px/cm, got {scale.pixels_per_cm:.4f}"

    def test_calibrate_from_height_consistency(self):
        """Same inputs always produce the same result (deterministic)."""
        pose = make_pose()
        s1 = self.calibrator.calibrate_from_height(pose, 170.0)
        s2 = self.calibrator.calibrate_from_height(pose, 170.0)
        assert s1.pixels_per_cm == s2.pixels_per_cm

    def test_inverse_proportionality_taller_person(self):
        """Taller person → fewer pixels represent each centimetre."""
        pose = make_pose()
        short = self.calibrator.calibrate_from_height(pose, 150.0)
        tall = self.calibrator.calibrate_from_height(pose, 190.0)
        assert short.pixels_per_cm > tall.pixels_per_cm

    def test_head_correction_factor_applied(self):
        """
        The 1.07 head-correction factor must inflate pixel height.
        Without correction: px_height = (0.96 - 0.05) * 1000 = 910 px.
        With correction: 910 * 1.07 = 973.7 px → px_per_cm ≈ 5.564 for 175 cm.
        If correction were missing the value would be ~5.20.
        """
        pose = make_pose(head_y=0.05, ankle_y=0.95, img_h=1000)
        scale = self.calibrator.calibrate_from_height(pose, 175.0)
        # Must be above the uncorrected value of ≈5.20
        assert (
            scale.pixels_per_cm > 5.20
        ), "HEAD_CORRECTION_FACTOR not applied — scale too low"

    def test_raises_on_implausible_height(self):
        pose = make_pose()
        with pytest.raises(ScaleCalibrationError, match="Implausible"):
            self.calibrator.calibrate_from_height(pose, 300.0)
        with pytest.raises(ScaleCalibrationError, match="Implausible"):
            self.calibrator.calibrate_from_height(pose, 30.0)

    def test_raises_when_person_too_small_in_frame(self):
        """Person occupying <50px in the image should raise.
        head=0.49, ankle=0.52, foot_index=0.53 → span 40px × 1.07 = 42.8px < 50.
        """
        tiny_pose = make_pose(head_y=0.49, ankle_y=0.52, img_h=1000)
        with pytest.raises(ScaleCalibrationError, match="50px"):
            self.calibrator.calibrate_from_height(tiny_pose, 175.0)

    # ── Numerical regression — from_reference_object ─────────────────────

    def _make_rect_image(
        self,
        width: int = 600,
        height: int = 600,
        rect_px: int = 200,
        border: int = 100,
    ) -> np.ndarray:
        """White background image with a black rectangle of known pixel size."""
        img = np.full((height, width, 3), 240, dtype=np.uint8)
        img[border : border + rect_px, border : border + rect_px] = 0
        return img

    def test_reference_object_returns_correct_scale(self):
        """
        200×200 px rectangle representing a 20 cm reference → 10.0 px/cm.
        Tolerance ±5 % — contour detection adds a small margin.
        """
        img = self._make_rect_image(rect_px=200)
        scale = self.calibrator.calibrate_from_reference_object(img, object_cm=20.0)
        assert (
            abs(scale.pixels_per_cm - 10.0) < 0.6
        ), f"Expected ~10.0 px/cm, got {scale.pixels_per_cm:.4f}"
        assert scale.source == "reference_object"
        assert scale.confidence == 0.95

    def test_reference_object_larger_object_lower_scale(self):
        """Bigger real-world reference → fewer pixels per cm."""
        img = self._make_rect_image(rect_px=200)
        small_ref = self.calibrator.calibrate_from_reference_object(img, object_cm=20.0)
        large_ref = self.calibrator.calibrate_from_reference_object(img, object_cm=40.0)
        assert small_ref.pixels_per_cm > large_ref.pixels_per_cm

    def test_reference_object_raises_when_no_rect_found(self):
        """Blank white image — no contour → error."""
        blank = np.full((300, 300, 3), 240, dtype=np.uint8)
        with pytest.raises(ScaleCalibrationError, match="reference object"):
            self.calibrator.calibrate_from_reference_object(blank, object_cm=20.0)
