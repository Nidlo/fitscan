"""
Inverse SMPL: given target measurements, find SMPL betas that produce them.
Uses gradient descent on PyTorch.
"""

from __future__ import annotations

from loguru import logger

from fitscan.domain.models import BodyMeasurements, ScaleFactor
from fitscan.smpl.mesh_analyzer import MeshAnalyzer


class InverseSMPLOptimizer:
    """Adjusts SMPL betas to match target measurements."""

    def __init__(
        self,
        gender: str = "ladies",
        learning_rate: float = 0.02,
        max_iterations: int = 200,
        convergence_threshold: float = 0.05,
    ) -> None:
        self.gender = gender
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.conv_threshold = convergence_threshold
        self._analyzer = MeshAnalyzer()
        self._dummy_scale = ScaleFactor(
            pixels_per_cm=1.0, source="mesh_direct", confidence=1.0
        )

    def optimise(
        self,
        initial_betas: list[float],
        target_overrides: dict[str, float],
        gender_profile: str = "ladies",
    ) -> tuple[list[float], BodyMeasurements]:
        """Returns (updated_betas, recalculated_measurements)."""
        logger.info(
            f"Inverse SMPL: targets={target_overrides}, " f"max_iter={self.max_iter}"
        )
        # Return initial betas unchanged as a stub
        measurements = BodyMeasurements(
            gender_profile=gender_profile, source="inverse_smpl"
        )
        return initial_betas, measurements
