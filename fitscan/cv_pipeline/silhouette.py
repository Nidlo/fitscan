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
        side_pose: BodyPose | None = None,
        gender_profile: str = "ladies",
    ) -> BodyMeasurements:
        widths = self._extract_horizontal_widths(front_mask, pose, scale)
        depths = self._estimate_depths(widths, side_mask, side_pose, pose, scale)
        return self._widths_to_measurements(
            widths,
            depths,
            scale,
            gender_profile,
            depth_source="side_silhouette" if side_mask is not None else "ratio_guess",
        )

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

    # Adult-population depth-to-width ratios. Used as a fallback only when
    # there's no side photo. CAESAR Phase II derived; these are the median
    # silhouette aspect ratios for adults at each anatomical level. The
    # error vs an actual side measurement is typically 20-40% on the
    # depth dimension — which propagates to 10-20% on the circumference,
    # which propagates to 40-60% on the perceived measurement when the
    # ratio is well off for a specific user. That's the single biggest
    # accuracy hole in single-view body measurement.
    _DEPTH_WIDTH_RATIOS: dict[str, float] = {
        "shoulder": 0.42,
        "chest": 0.60,
        "bust": 0.60,
        "across_chest": 0.60,
        "across_back": 0.50,
        "under_breast": 0.65,
        "waist": 0.75,
        "hip": 0.65,
        "thigh": 0.80,
    }

    def _estimate_depths(
        self,
        widths: dict[str, float],
        side_mask: np.ndarray | None,
        side_pose: BodyPose | None,
        front_pose: BodyPose,
        scale: ScaleFactor,
    ) -> dict[str, float]:
        """Body depth in pixels at each anatomical level.

        Sprint 34 Phase B — when both `side_mask` and `side_pose` are
        supplied, read DEPTH (the horizontal width of the side silhouette)
        directly at each anatomical y. This is the single biggest
        accuracy unlock for circumference measurements: front-only must
        guess depth from width via a population-mean ratio, which gives
        ~10-20% error on circumference for any user whose actual body
        proportions deviate from the mean (i.e., most users).

        Vertical-position lookup is normalised to each photo's own body
        bbox so cropping / framing differences between the two photos
        don't matter — we don't assume the camera was at the same
        distance.

        Falls back to the ratio-based estimate from `_DEPTH_WIDTH_RATIOS`
        when no side mask is supplied OR when the side silhouette at a
        given level fails the sanity check.
        """
        # Front-only fallback path — same behaviour as pre-Phase B.
        if side_mask is None or side_pose is None:
            return {
                k: widths.get(k, 0) * self._DEPTH_WIDTH_RATIOS.get(k, 0.65)
                for k in widths
            }

        # Both photos: read actual depth at each anatomical level from the
        # side silhouette. We measure depth in side-image pixels then
        # rescale to front-image-pixel-equivalents so the
        # ellipse-circumference math (which expects both axes in front-px)
        # operates consistently.
        front_pix_height = self._body_pixel_height(front_pose)
        side_pix_height = self._body_pixel_height(side_pose)
        if front_pix_height is None or side_pix_height is None or side_pix_height <= 0:
            return {
                k: widths.get(k, 0) * self._DEPTH_WIDTH_RATIOS.get(k, 0.65)
                for k in widths
            }
        side_to_front_px = front_pix_height / side_pix_height

        # Vertical landmark levels on the SIDE photo, parallel to the
        # front-photo levels. Use the same anatomical y-positions
        # (proportional to body height) so e.g. waist on the side image
        # lines up with waist on the front.
        side_levels = self._anatomical_y_positions(side_pose, side_mask.shape[0])

        depths: dict[str, float] = {}
        for name in widths:
            depth_px_side = None
            y_side = side_levels.get(name)
            if y_side is not None:
                y_side_clamped = int(np.clip(y_side, 0, side_mask.shape[0] - 1))
                depth_px_side = self._width_at_row(side_mask, y_side_clamped)

            # Sanity bounds: a depth that's narrower than 25% of the front
            # width, OR wider than 200%, is almost certainly a bad
            # silhouette extraction (rembg failure, occlusion, arm-in-
            # frame). Fall back to the ratio in those cases — better a
            # population-mean estimate than a junk one.
            front_w = widths.get(name, 0)
            if (
                depth_px_side is not None
                and front_w > 0
                and 0.25 <= (depth_px_side * side_to_front_px) / front_w <= 2.0
            ):
                depths[name] = depth_px_side * side_to_front_px
            else:
                depths[name] = front_w * self._DEPTH_WIDTH_RATIOS.get(name, 0.65)

        return depths

    def _body_pixel_height(self, pose: BodyPose) -> float | None:
        """Distance from highest landmark (typically nose / ear) to lowest
        (typically foot or ankle). Used to put two photos at different
        camera distances on the same proportional scale."""
        head_candidates = ("nose", "left_ear", "right_ear")
        foot_candidates = (
            "left_foot_index",
            "right_foot_index",
            "left_ankle",
            "right_ankle",
        )
        head_y = min(
            (
                pose.keypoints[name].y * pose.image_height
                for name in head_candidates
                if name in pose.keypoints
            ),
            default=None,
        )
        foot_y = max(
            (
                pose.keypoints[name].y * pose.image_height
                for name in foot_candidates
                if name in pose.keypoints
            ),
            default=None,
        )
        if head_y is None or foot_y is None or foot_y <= head_y:
            return None
        return foot_y - head_y

    def _anatomical_y_positions(
        self, pose: BodyPose, image_height: int
    ) -> dict[str, float]:
        """Anatomical-landmark y positions in absolute pixel coords for
        the SIDE photo — mirrors `_extract_horizontal_widths`'s levels
        dict so depth lookups land at the same anatomical heights as
        the width lookups."""
        return {
            "shoulder": self._get_y_between(pose, "left_shoulder", "right_shoulder")
            or 0,
            "across_chest": self._get_y_below(pose, "left_shoulder", 0.06) or 0,
            "across_back": self._get_y_below(pose, "left_shoulder", 0.06) or 0,
            "bust": self._get_y_below(pose, "left_shoulder", 0.12) or 0,
            "under_breast": self._get_y_below(pose, "left_shoulder", 0.17) or 0,
            "waist": self._get_y_between(pose, "left_shoulder", "left_hip", ratio=0.6)
            or 0,
            "hip": self._get_y_between(pose, "left_hip", "right_hip") or 0,
            "thigh": self._get_y_below(pose, "left_hip", 0.10) or 0,
        }

    def _widths_to_measurements(
        self,
        widths: dict[str, float],
        depths: dict[str, float],
        scale: ScaleFactor,
        gender_profile: str = "ladies",
        depth_source: str = "ratio_guess",
    ) -> BodyMeasurements:
        # Sprint 34 Phase B — when depth came from a real side silhouette
        # rather than the population-mean ratio, bump baseline confidence
        # (0.68 → 0.80). The fuser uses this to weight silhouette
        # results vs SMPL results; with credible depth, silhouette
        # circumferences are often more accurate than SMPL's because
        # SMPL's betas can latch onto a wrong body type while silhouette
        # measures pixels directly.
        base_conf = 0.80 if depth_source == "side_silhouette" else 0.68
        source_tag = (
            "silhouette_multiview"
            if depth_source == "side_silhouette"
            else "silhouette"
        )

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
                confidence=round(base_conf + conf_boost, 2),
                source=source_tag,
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
