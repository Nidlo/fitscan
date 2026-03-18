"""
Extracts body measurements from a 3D SMPL mesh.
All circumferences computed by slicing the mesh at anatomical heights.
Implements IMeasurementStrategy.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from loguru import logger

from fitscan.domain.models import BodyMeasurements, ScaleFactor, SingleMeasurement


class MeshAnalyzer:
    """Extracts body measurements from a trimesh.Trimesh SMPL mesh."""

    CIRCUMFERENCE_HEIGHTS = {
        "neck": 0.920,
        "bust": 0.780,
        "chest": 0.780,
        "under_breast": 0.740,
        "waist": 0.635,
        "hip": 0.525,
        "thigh": 0.445,
    }

    ARM_HEIGHT_RATIOS = {
        "around_arm": 0.810,
        "around_arm_3_4": 0.740,
        "wrist": 0.650,
    }

    LANDMARK_HEIGHTS = {
        "shoulder_seam": 0.850,
        "nipple": 0.760,
        "underbust": 0.730,
        "waist": 0.635,
        "hip_fullest": 0.525,
        "nape": 0.900,
        "crotch": 0.480,
        "ankle": 0.050,
    }

    def measure(
        self,
        mesh: Any,
        scale: ScaleFactor,
        gender_profile: str = "ladies",
    ) -> BodyMeasurements:
        """Extract all tailor-book measurements from SMPL mesh."""
        y_min, y_max = mesh.bounds[0][1], mesh.bounds[1][1]
        y_range = y_max - y_min
        m_to_cm = 100.0

        smpl_height_cm = y_range * m_to_cm
        height_correction = 1.0

        def y_at(landmark: str) -> float:
            return y_min + self.LANDMARK_HEIGHTS[landmark] * y_range

        def circ(level: str, conf: float = 0.88) -> Optional[SingleMeasurement]:
            rel_h = self.CIRCUMFERENCE_HEIGHTS.get(level)
            if rel_h is None:
                return None
            y_cut = y_min + rel_h * y_range
            perimeter = self._circumference_at_y(mesh, y_cut)
            if perimeter <= 0:
                return None
            return SingleMeasurement(
                value_cm=round(perimeter * m_to_cm * height_correction, 1),
                confidence=conf,
                source="smpl",
            )

        def vert_dist(
            lm_top: str, lm_bot: str, conf: float = 0.85
        ) -> Optional[SingleMeasurement]:
            d = abs(y_at(lm_top) - y_at(lm_bot)) * m_to_cm * height_correction
            return SingleMeasurement(round(d, 1), conf, "smpl")

        def horiz_width(
            rel_h: float, conf: float = 0.82
        ) -> Optional[SingleMeasurement]:
            y_cut = y_min + rel_h * y_range
            verts_near = mesh.vertices[np.abs(mesh.vertices[:, 1] - y_cut) < 0.015]
            if len(verts_near) < 2:
                return None
            w_m = float(verts_near[:, 0].max() - verts_near[:, 0].min())
            return SingleMeasurement(
                round(w_m * m_to_cm * height_correction, 1), conf, "smpl"
            )

        # ── Circumferences ──
        bust = circ("bust")
        chest = circ("chest")
        waist = circ("waist")
        hip = circ("hip")
        ubw = circ("under_breast", conf=0.80)
        neck = circ("neck", conf=0.87)
        thigh = circ("thigh", conf=0.82)

        arm_circ = self._arm_circumference(
            mesh, self.ARM_HEIGHT_RATIOS["around_arm"], m_to_cm
        )
        arm_3_4_circ = self._arm_circumference(
            mesh, self.ARM_HEIGHT_RATIOS["around_arm_3_4"], m_to_cm
        )
        wrist_circ = self._arm_circumference(
            mesh, self.ARM_HEIGHT_RATIOS["wrist"], m_to_cm
        )

        # ── Widths ──
        shoulder_w = horiz_width(0.850, conf=0.86)
        across_back_w = horiz_width(0.820, conf=0.80)
        across_chest_w = horiz_width(0.790, conf=0.80)

        ntn = self._nipple_to_nipple(mesh, m_to_cm)

        # ── Vertical lengths ──
        s_to_nipple = vert_dist("shoulder_seam", "nipple", conf=0.82)
        s_to_underbust = vert_dist("shoulder_seam", "underbust", conf=0.83)
        s_to_waist = vert_dist("shoulder_seam", "waist", conf=0.88)
        s_to_hip = vert_dist("shoulder_seam", "hip_fullest", conf=0.87)
        nape_to_waist = self._geodesic_nape_to_waist(mesh, m_to_cm)
        hip_depth = vert_dist("waist", "hip_fullest", conf=0.88)
        inseam = vert_dist("crotch", "ankle", conf=0.83)

        # ── Garment/arm lengths ──
        sleeve = self._sleeve_length(mesh, m_to_cm)
        height_m = SingleMeasurement(
            round(smpl_height_cm * height_correction, 1), 0.92, "smpl"
        )

        blouse_len = s_to_hip
        kaba_len = (
            SingleMeasurement(
                round((s_to_hip.value_cm if s_to_hip else 0) + 30, 1),
                0.75,
                "smpl",
            )
            if s_to_hip
            else None
        )
        skirt_len = vert_dist("waist", "ankle", conf=0.80)
        dress_len = SingleMeasurement(round(height_m.value_cm * 0.95, 1), 0.72, "smpl")

        return BodyMeasurements(
            bust_cm=bust if gender_profile == "ladies" else None,
            chest_cm=chest if gender_profile != "ladies" else None,
            waist_cm=waist,
            hip_cm=hip,
            under_breast_waist_cm=ubw,
            around_arm_cm=arm_circ,
            around_arm_3_4_cm=arm_3_4_circ,
            wrist_cm=wrist_circ,
            neck_cm=neck,
            thigh_cm=thigh,
            shoulder_cm=shoulder_w,
            across_back_cm=across_back_w,
            across_chest_cm=across_chest_w,
            shoulder_to_nipple_cm=(s_to_nipple if gender_profile == "ladies" else None),
            nipple_to_nipple_cm=(ntn if gender_profile == "ladies" else None),
            shoulder_to_underbust_cm=(
                s_to_underbust if gender_profile == "ladies" else None
            ),
            shoulder_to_waist_cm=s_to_waist,
            shoulder_to_hip_cm=s_to_hip,
            nape_to_waist_cm=nape_to_waist,
            height_cm=height_m,
            inseam_cm=inseam,
            hip_depth_cm=hip_depth,
            sleeve_length_cm=sleeve,
            blouse_length_cm=(blouse_len if gender_profile == "ladies" else None),
            kaba_length_cm=(kaba_len if gender_profile == "ladies" else None),
            skirt_length_cm=(skirt_len if gender_profile == "ladies" else None),
            dress_length_cm=(dress_len if gender_profile == "ladies" else None),
            slit_length_cm=(
                SingleMeasurement(40.0, 0.40, "default")
                if gender_profile == "ladies"
                else None
            ),
            cape_cm=(
                SingleMeasurement(60.0, 0.40, "default")
                if gender_profile == "ladies"
                else None
            ),
            gender_profile=gender_profile,
            source="smpl",
        )

    def _circumference_at_y(self, mesh: Any, y: float) -> float:
        """Slice mesh at height y, return perimeter. DRY."""
        try:
            section = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
            if section is None:
                return 0.0
            path2d, _ = section.to_planar()
            return float(sum(e.length for e in path2d.entities))
        except Exception as e:
            logger.debug(f"Mesh section failed at y={y:.3f}: {e}")
            return 0.0

    def _arm_circumference(
        self, mesh: Any, height_ratio: float, m_to_cm: float
    ) -> Optional[SingleMeasurement]:
        """Circumference of arm at a given height ratio."""
        y_min, y_max = mesh.bounds[0][1], mesh.bounds[1][1]
        y_cut = y_min + height_ratio * (y_max - y_min)
        arm_verts = mesh.vertices[
            (np.abs(mesh.vertices[:, 1] - y_cut) < 0.015)
            & (np.abs(mesh.vertices[:, 0]) > 0.08)
        ]
        if len(arm_verts) < 6:
            return None
        pts = arm_verts[:, [0, 2]]
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(pts)
            perim = float(hull.area)
            return SingleMeasurement(round(perim * m_to_cm, 1), 0.80, "smpl")
        except Exception:
            return None

    def _nipple_to_nipple(
        self, mesh: Any, m_to_cm: float
    ) -> Optional[SingleMeasurement]:
        y_min, y_max = mesh.bounds[0][1], mesh.bounds[1][1]
        y_nipple = y_min + self.LANDMARK_HEIGHTS["nipple"] * (y_max - y_min)
        front_verts = mesh.vertices[
            (np.abs(mesh.vertices[:, 1] - y_nipple) < 0.015) & (mesh.vertices[:, 2] > 0)
        ]
        if len(front_verts) < 4:
            return None
        dist = abs(float(front_verts[:, 0].max()) - float(front_verts[:, 0].min()))
        return SingleMeasurement(round(dist * m_to_cm, 1), 0.75, "smpl")

    def _geodesic_nape_to_waist(
        self, mesh: Any, m_to_cm: float
    ) -> Optional[SingleMeasurement]:
        y_min, y_max = mesh.bounds[0][1], mesh.bounds[1][1]
        y_nape = y_min + self.LANDMARK_HEIGHTS["nape"] * (y_max - y_min)
        y_waist = y_min + self.LANDMARK_HEIGHTS["waist"] * (y_max - y_min)
        # Fallback to straight-line distance
        d = abs(y_nape - y_waist) * m_to_cm
        return SingleMeasurement(round(d, 1), 0.65, "smpl_approx")

    def _sleeve_length(self, mesh: Any, m_to_cm: float) -> Optional[SingleMeasurement]:
        y_min, y_max = mesh.bounds[0][1], mesh.bounds[1][1]
        y_shoulder = y_min + self.LANDMARK_HEIGHTS["shoulder_seam"] * (y_max - y_min)
        y_wrist = y_min + self.ARM_HEIGHT_RATIOS["wrist"] * (y_max - y_min)
        d = abs(y_shoulder - y_wrist) * m_to_cm * 1.12
        return SingleMeasurement(round(d, 1), 0.65, "smpl_approx")
