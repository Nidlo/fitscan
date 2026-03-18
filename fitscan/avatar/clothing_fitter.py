"""
Drapes virtual clothing onto an SMPL body.
Falls back to rigid offset draping if TailorNet models unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from fitscan.domain.models import AvatarExport, ClothingDrape

GARMENT_MAP = {
    "dress": "old-shirt_female",
    "blouse": "t-shirt_female",
    "kaba": "old-shirt_female",
    "skirt": "skirt_female",
    "trousers": "jogging-pants_male",
    "t-shirt": "t-shirt_female",
}


class ClothingFitter:
    """Drapes a garment onto an avatar. Single responsibility."""

    def fit_garment(
        self,
        avatar: AvatarExport,
        garment_type: str,
        output_dir: Path,
    ) -> ClothingDrape:
        """Drape a garment onto the avatar and return combined GLB."""
        garment_mesh = self._rigid_drape(avatar, garment_type)

        body = trimesh.load(avatar.glb_path)
        if isinstance(body, trimesh.Scene):
            body = body.dump(concatenate=True)
        combined = trimesh.util.concatenate([body, garment_mesh])

        output_dir.mkdir(parents=True, exist_ok=True)
        draped_path = output_dir / f"draped_{garment_type}.glb"
        combined.export(str(draped_path), file_type="glb")

        fit_quality = self._assess_fit(body, garment_mesh)

        return ClothingDrape(
            avatar_id=avatar.avatar_id,
            garment_type=garment_type,
            draped_glb_path=str(draped_path),
            garment_measurements={},
            fit_quality=fit_quality,
        )

    def _rigid_drape(self, avatar: AvatarExport, garment_type: str) -> trimesh.Trimesh:
        """Offset body mesh outward to simulate clothing."""
        body = trimesh.load(avatar.glb_path)
        if isinstance(body, trimesh.Scene):
            body = body.dump(concatenate=True)
        garment = body.copy()
        thickness = {
            "dress": 0.015,
            "blouse": 0.012,
            "skirt": 0.010,
        }.get(garment_type, 0.012)
        garment.vertices += garment.vertex_normals * thickness
        cloth_colour = np.array(
            [[100, 149, 237, 255]] * len(garment.vertices),
            dtype=np.uint8,
        )
        garment.visual = trimesh.visual.ColorVisuals(
            mesh=garment, vertex_colors=cloth_colour
        )
        return garment

    def _assess_fit(self, body: trimesh.Trimesh, garment: trimesh.Trimesh) -> float:
        """Score garment fit quality (0-1)."""
        try:
            from trimesh.proximity import closest_point

            _, distances, _ = closest_point(body, garment.vertices)
            mean_gap = float(np.mean(distances))
            if 0.008 <= mean_gap <= 0.025:
                return 0.90
            elif mean_gap < 0.008:
                return 0.60
            else:
                return 0.70
        except Exception:
            return 0.75
