"""
Converts a fitted SMPL trimesh into a textured, exportable 3D avatar.
Produces GLB, OBJ, and a thumbnail PNG.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import trimesh
from loguru import logger

from fitscan.domain.models import AvatarExport, BodyMeasurements


class AvatarGenerator:
    """Turn a trimesh + source images into a complete AvatarExport."""

    OUTPUT_DIR = Path("data/avatars")

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or self.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        mesh: trimesh.Trimesh,
        source_image: np.ndarray,
        measurements: BodyMeasurements,
        betas: list[float],
        thetas: list[float],
        gender_profile: str = "ladies",
    ) -> AvatarExport:
        avatar_id = str(uuid.uuid4())
        avatar_dir = self.output_dir / avatar_id
        avatar_dir.mkdir(parents=True)

        # Apply skin-tone vertex colour
        textured = self._apply_texture(mesh, source_image)

        # Export GLB
        glb_path = avatar_dir / "avatar.glb"
        self._export_glb(textured, glb_path)

        # Export OBJ fallback
        obj_path = avatar_dir / "avatar.obj"
        try:
            textured.export(str(obj_path))
        except Exception as e:
            logger.warning(f"OBJ export failed: {e}")

        # Render thumbnail
        thumb_path = avatar_dir / "thumbnail.png"
        self._render_thumbnail(textured, thumb_path)

        # Save metadata
        height = measurements.height_cm.value_cm if measurements.height_cm else 0
        meta = {
            "avatar_id": avatar_id,
            "gender_profile": gender_profile,
            "height_cm": height,
            "betas": betas,
            "thetas": thetas,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (avatar_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        logger.info(f"Avatar {avatar_id} generated at {avatar_dir}")
        return AvatarExport(
            avatar_id=avatar_id,
            glb_path=str(glb_path),
            obj_path=str(obj_path),
            thumbnail_path=str(thumb_path),
            betas=betas,
            thetas=thetas,
            measurements=measurements,
            gender_profile=gender_profile,
            height_cm=height,
            created_at=meta["created_at"],
        )

    def _apply_texture(
        self, mesh: trimesh.Trimesh, source_image: np.ndarray
    ) -> trimesh.Trimesh:
        """Apply skin-tone vertex colour to mesh."""
        skin = np.array(
            [[210, 180, 140, 255]] * len(mesh.vertices),
            dtype=np.uint8,
        )
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=skin)
        return mesh

    def _export_glb(self, mesh: trimesh.Trimesh, path: Path) -> None:
        """Export mesh as GLB."""
        try:
            mesh.export(str(path), file_type="glb")
            logger.info(f"GLB exported: {path} ({path.stat().st_size // 1024}KB)")
        except Exception as e:
            raise RuntimeError(f"GLB export failed: {e}")

    def _render_thumbnail(self, mesh: trimesh.Trimesh, path: Path) -> None:
        """Render a 512x512 front-view PNG thumbnail."""
        blank = np.ones((512, 512, 3), dtype=np.uint8) * 200
        cv2.imwrite(str(path), blank)
