"""
Fits an SMPL 3D body model to a 2D image using HMR2.0.
Returns the fitted mesh with shape and pose parameters.
Implements IBodyFitter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from fitscan.domain.exceptions import SMPLFittingError
from fitscan.domain.models import BodyPose


class SMPLBodyFitter:
    """
    Wraps HMR2.0 to fit SMPL parameters from a single image.
    Falls back to SMPLify optimization if HMR2 is unavailable.
    """

    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        self.device = self._resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self._model: Any = None

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from hmr2.models import HMR2

            self._model = HMR2.load_from_checkpoint(
                str(self.checkpoint_path), strict=False
            )
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info(f"HMR2.0 loaded on {self.device}")
        except ImportError:
            logger.warning("HMR2 not installed. Falling back to SMPLify.")
            self._model = "smplify_fallback"
        except Exception as e:
            logger.warning(f"HMR2 load failed: {e}. Using SMPLify fallback.")
            self._model = "smplify_fallback"

    def fit(self, image: np.ndarray, pose: BodyPose) -> Any:
        """Returns a trimesh.Trimesh object representing the fitted body."""
        self._load_model()
        if self._model == "smplify_fallback":
            return self._smplify_fallback(image, pose)
        return self._hmr2_fit(image)

    def _hmr2_fit(self, image: np.ndarray) -> Any:
        import trimesh

        img_tensor = self._preprocess_image(image)
        with torch.no_grad():
            out = self._model({"img": img_tensor.to(self.device)})
        vertices = out["pred_vertices"][0].cpu().numpy()
        faces = self._model.smpl.faces
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def _smplify_fallback(self, image: np.ndarray, pose: BodyPose) -> Any:
        """Pure-Python SMPLify: optimize SMPL betas to match 2D keypoints."""
        try:
            import smplx
            import trimesh

            model = smplx.create(
                "data/models/smpl",
                model_type="smpl",
                gender="neutral",
                num_betas=10,
            ).to(self.device)

            betas = torch.zeros(1, 10, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([betas], lr=0.05)
            target_joints = self._pose_to_target_joints(pose)

            for _ in range(100):
                optimizer.zero_grad()
                output = model(betas=betas, return_verts=True)
                pred_joints = output.joints[0, : target_joints.shape[0], :2]
                loss = torch.nn.functional.mse_loss(pred_joints, target_joints)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                output = model(betas=betas, return_verts=True)
            verts = output.vertices[0].cpu().numpy()
            faces = model.faces
            return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        except ImportError:
            raise SMPLFittingError(
                "Neither HMR2 nor smplx installed. Cannot fit SMPL model."
            )

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        import torchvision.transforms.functional as tvf
        from PIL import Image

        pil = Image.fromarray(image).resize((256, 256))
        tensor = tvf.to_tensor(pil).unsqueeze(0)
        return tvf.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _pose_to_target_joints(self, pose: BodyPose) -> torch.Tensor:
        joint_map = [
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        ]
        coords = []
        for name in joint_map:
            kp = pose.keypoints.get(name)
            if kp:
                coords.append([kp.x, kp.y])
            else:
                coords.append([0.5, 0.5])
        return torch.tensor(coords, dtype=torch.float32, device=self.device)
