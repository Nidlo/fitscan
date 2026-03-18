"""Geometric measurements computed directly from pose keypoints."""

from __future__ import annotations

import numpy as np

from fitscan.cv_pipeline.silhouette import px_to_cm
from fitscan.domain.models import (
    BodyMeasurements,
    BodyPose,
    ScaleFactor,
    SingleMeasurement,
)


class GeometricMeasurer:
    """Computes vertical/distance measurements from 2D keypoints."""

    def measure(
        self,
        pose: BodyPose,
        scale: ScaleFactor,
        gender_profile: str = "ladies",
    ) -> BodyMeasurements:
        def dist(lm1: str, lm2: str) -> float | None:
            if lm1 not in pose.keypoints or lm2 not in pose.keypoints:
                return None
            p1 = pose.pixel_coords(lm1)
            p2 = pose.pixel_coords(lm2)
            return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

        def vert_dist(lm1: str, lm2: str) -> float | None:
            if lm1 not in pose.keypoints or lm2 not in pose.keypoints:
                return None
            y1 = pose.keypoints[lm1].y * pose.image_height
            y2 = pose.keypoints[lm2].y * pose.image_height
            return abs(y2 - y1)

        def m(px: float | None, conf: float = 0.70) -> SingleMeasurement | None:
            if px is None:
                return None
            return SingleMeasurement(
                value_cm=round(px_to_cm(px, scale), 1),
                confidence=conf,
                source="geometric",
            )

        shoulder_px = dist("left_shoulder", "right_shoulder")
        s_to_hip_px = vert_dist("left_shoulder", "left_hip")
        s_to_waist_px = s_to_hip_px * 0.6 if s_to_hip_px is not None else None
        sleeve_px = dist("left_shoulder", "left_wrist")

        return BodyMeasurements(
            shoulder_cm=m(shoulder_px, 0.75),
            shoulder_to_waist_cm=m(s_to_waist_px, 0.72),
            shoulder_to_hip_cm=m(s_to_hip_px, 0.74),
            sleeve_length_cm=m(sleeve_px, 0.70),
            gender_profile=gender_profile,
            source="geometric",
        )
