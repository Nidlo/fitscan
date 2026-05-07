"""
Extracts body measurements from the 2D silhouette contour.
Uses front-view silhouette for widths, side-view for depths.
Circumferences are estimated as ellipse perimeters where width=front, depth=side.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fitscan.domain.models import (
    BodyMeasurements,
    BodyPose,
    ScaleFactor,
    SingleMeasurement,
)


def px_to_cm(pixels: float, scale: ScaleFactor) -> float:
    """DRY: single conversion function used everywhere."""
    return pixels / scale.pixels_per_cm


def ellipse_circumference(a: float, b: float) -> float:
    """Ramanujan's approximation for ellipse perimeter. DRY."""
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return float(np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h))))


class SilhouetteMeasurer:
    """
    Extracts measurements from the body silhouette mask and pose keypoints.
    Works from a single front-facing image only.
    """

    def measure(
        self,
        front_mask: np.ndarray,
        pose: BodyPose,
        scale: ScaleFactor,
        side_mask: np.ndarray | None = None,
        gender_profile: str = "ladies",
    ) -> BodyMeasurements:
        widths = self._extract_horizontal_widths(front_mask, pose, scale)
        depths = self._estimate_depths(widths, side_mask, scale)
        return self._widths_to_measurements(widths, depths, scale, gender_profile)

    def _extract_horizontal_widths(
        self, mask: np.ndarray, pose: BodyPose, scale: ScaleFactor
    ) -> dict[str, float]:
        """Get horizontal body width in pixels at each anatomical level."""
        widths: dict[str, float] = {}
        levels = {
            "shoulder": self._get_y_between(pose, "left_shoulder", "right_shoulder"),
            "across_chest": self._get_y_below(pose, "left_shoulder", 0.06),
            "bust": self._get_y_below(pose, "left_shoulder", 0.12),
            "under_breast": self._get_y_below(pose, "left_shoulder", 0.17),
            "waist": self._get_y_between(pose, "left_shoulder", "left_hip", ratio=0.6),
            "hip": self._get_y_between(pose, "left_hip", "right_hip"),
            "thigh": self._get_y_below(pose, "left_hip", 0.10),
        }
        for name, y_px in levels.items():
            if y_px is None:
                continue
            y = int(np.clip(y_px, 0, mask.shape[0] - 1))
            width = self._width_at_row(mask, y)
            if width is not None:
                widths[name] = width
        return widths

    def _width_at_row(
        self, mask: np.ndarray, y: int, half_window: int = 5
    ) -> float | None:
        """Sample silhouette width at row `y` and return the median over
        `2*half_window+1` rows centred on it. Single-row scans overshoot
        by 5-20px on hair/cloth-fold noise; the median collapses those
        spikes without dragging the value off real anatomy because the
        surrounding rows have the right width. Falls back to single-row
        when the window doesn't fit (top/bottom of frame, very short
        masks). (FS-NIDLO-CV-02)
        """
        h = mask.shape[0]
        y0 = max(0, y - half_window)
        y1 = min(h, y + half_window + 1)
        sample_widths: list[float] = []
        for row_y in range(y0, y1):
            row = mask[row_y, :]
            nonzero = np.where(row > 127)[0]
            if len(nonzero) >= 2:
                sample_widths.append(float(nonzero[-1] - nonzero[0]))
        if not sample_widths:
            return None
        return float(np.median(sample_widths))

    def _get_y_between(
        self, pose: BodyPose, lm1: str, lm2: str, ratio: float = 0.5
    ) -> float | None:
        if lm1 not in pose.keypoints or lm2 not in pose.keypoints:
            return None
        y1 = pose.keypoints[lm1].y * pose.image_height
        y2 = pose.keypoints[lm2].y * pose.image_height
        return y1 + (y2 - y1) * ratio

    def _get_y_below(
        self, pose: BodyPose, landmark: str, fraction: float
    ) -> float | None:
        if landmark not in pose.keypoints:
            return None
        y = pose.keypoints[landmark].y * pose.image_height
        return y + fraction * pose.image_height

    def _estimate_depths(
        self,
        widths: dict[str, float],
        side_mask: np.ndarray | None,
        scale: ScaleFactor,
    ) -> dict[str, float]:
        """Estimate body depth at each level using population ratios."""
        depth_width_ratios = {
            "shoulder": 0.42,
            "chest": 0.60,
            "bust": 0.60,
            "waist": 0.75,
            "hip": 0.65,
            "thigh": 0.80,
        }
        return {k: widths.get(k, 0) * depth_width_ratios.get(k, 0.65) for k in widths}

    def _widths_to_measurements(
        self,
        widths: dict[str, float],
        depths: dict[str, float],
        scale: ScaleFactor,
        gender_profile: str = "ladies",
    ) -> BodyMeasurements:
        def circ(name: str, conf_boost: float = 0.0) -> Optional[SingleMeasurement]:
            w = widths.get(name)
            d = depths.get(name)
            if not w or not d:
                return None
            a = px_to_cm(w / 2, scale)
            b = px_to_cm(d / 2, scale)
            circ_cm = ellipse_circumference(a, b)
            return SingleMeasurement(
                value_cm=round(circ_cm, 1),
                confidence=round(0.68 + conf_boost, 2),
                source="silhouette",
            )

        shoulder_px = widths.get("shoulder")
        across_chest_px = widths.get("across_chest")

        return BodyMeasurements(
            bust_cm=circ("bust") if gender_profile == "ladies" else None,
            chest_cm=circ("bust") if gender_profile != "ladies" else None,
            waist_cm=circ("waist"),
            hip_cm=circ("hip", conf_boost=0.05),
            under_breast_waist_cm=circ("under_breast"),
            thigh_cm=circ("thigh"),
            shoulder_cm=(
                SingleMeasurement(
                    round(px_to_cm(shoulder_px, scale), 1),
                    0.80,
                    "silhouette",
                )
                if shoulder_px
                else None
            ),
            across_chest_cm=(
                SingleMeasurement(
                    round(px_to_cm(across_chest_px, scale), 1),
                    0.75,
                    "silhouette",
                )
                if across_chest_px
                else None
            ),
            gender_profile=gender_profile,
            source="silhouette",
        )
