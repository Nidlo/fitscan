"""
Computes pixel-per-centimeter scale factor.
This is the most critical component — all measurements depend on it.
Implements IScaleCalibrator.
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from fitscan.domain.exceptions import ScaleCalibrationError
from fitscan.domain.models import BodyPose, ScaleFactor


class ScaleCalibrator:
    """Establishes scale from user-provided height or reference object."""

    HEAD_TOP_LANDMARK = "nose"
    FOOT_LANDMARKS = (
        "left_ankle",
        "right_ankle",
        "left_foot_index",
        "right_foot_index",
    )
    HEAD_CORRECTION_FACTOR = 1.07

    def calibrate_from_height(self, pose: BodyPose, height_cm: float) -> ScaleFactor:
        """Use user's stated height to compute pixels_per_cm."""
        if height_cm < 50 or height_cm > 250:
            raise ScaleCalibrationError(
                f"Implausible height: {height_cm}cm. Must be 50-250cm."
            )
        head_y = pose.keypoints[self.HEAD_TOP_LANDMARK].y * pose.image_height
        foot_y = max(
            pose.keypoints[name].y * pose.image_height
            for name in self.FOOT_LANDMARKS
            if name in pose.keypoints
        )
        pixel_height = (foot_y - head_y) * self.HEAD_CORRECTION_FACTOR
        if pixel_height < 50:
            raise ScaleCalibrationError(
                "Person occupies less than 50px height — "
                "image too small or pose incomplete."
            )
        px_per_cm = pixel_height / height_cm
        logger.info(
            f"Scale: {px_per_cm:.2f} px/cm (pixel height: {pixel_height:.0f}px)"
        )
        return ScaleFactor(
            pixels_per_cm=px_per_cm,
            source="user_height",
            confidence=0.90,
            target_height_cm=height_cm,
        )

    def calibrate_from_reference_object(
        self, image: np.ndarray, object_cm: float
    ) -> ScaleFactor:
        """Detect a rectangular reference object and compute scale."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best_rect = None
        best_area = 0
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                area = cv2.contourArea(cnt)
                if area > best_area:
                    best_area = area
                    best_rect = approx
        if best_rect is None or best_area < 1000:
            raise ScaleCalibrationError(
                "Could not detect reference object. "
                "Ensure A4/reference is clearly visible."
            )
        rect = cv2.minAreaRect(best_rect)
        _, (w, h), _ = rect
        pixel_long_side = max(w, h)
        px_per_cm = pixel_long_side / object_cm
        return ScaleFactor(
            pixels_per_cm=px_per_cm,
            source="reference_object",
            confidence=0.95,
        )
