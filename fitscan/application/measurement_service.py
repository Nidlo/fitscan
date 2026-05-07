"""
High-level service. Wires all dependencies together.
This is the only place where concrete classes are instantiated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fitscan.application.pipeline_orchestrator import PipelineOrchestrator
from fitscan.cv_pipeline.pose_detector import PoseDetector
from fitscan.cv_pipeline.preprocessor import ImagePreprocessor
from fitscan.cv_pipeline.scale_calibrator import ScaleCalibrator
from fitscan.cv_pipeline.silhouette import SilhouetteMeasurer
from fitscan.fusion.confidence import ConfidenceScorer
from fitscan.fusion.fuser import MeasurementFuser
from fitscan.infrastructure.image_loader import ImageLoader
from fitscan.smpl.body_fitter import SMPLBodyFitter
from fitscan.smpl.mesh_analyzer import MeshAnalyzer
from fitscan.validation.claude_validator import ClaudeValidator
from fitscan.validation.rule_validator import RuleValidator


def create_pipeline(
    enable_smpl: bool = True,
    enable_claude: bool = True,
) -> PipelineOrchestrator:
    """Factory function: creates the fully-wired pipeline."""
    return PipelineOrchestrator(
        image_loader=ImageLoader(),
        preprocessor=ImagePreprocessor(),
        pose_detector=PoseDetector(),
        scale_calibrator=ScaleCalibrator(),
        silhouette_measurer=SilhouetteMeasurer(),
        body_fitter=SMPLBodyFitter(
            checkpoint_path=Path(
                os.getenv(
                    "HMR2_CHECKPOINT_PATH",
                    "data/models/hmr2/epoch=35-step=1000000.ckpt",
                )
            )
        ),
        mesh_analyzer=MeshAnalyzer(),
        rule_validator=RuleValidator(),
        claude_validator=ClaudeValidator(api_key=os.getenv("ANTHROPIC_API_KEY")),
        fuser=MeasurementFuser(),
        confidence_scorer=ConfidenceScorer(),
        enable_smpl=enable_smpl,
        enable_claude=enable_claude,
    )


class MeasurementService:
    """Main entry point for the FitScan system."""

    def __init__(self, pipeline: PipelineOrchestrator | None = None) -> None:
        self._pipeline = pipeline or create_pipeline()

    def measure(
        self,
        image_path: str | Path,
        height_cm: float | None = None,
        reference_object_cm: float | None = None,
        gender_profile: str = "ladies",
        subject_id: str | None = None,
        enable_smpl: bool | None = None,
        enable_claude: bool | None = None,
    ) -> dict:
        # Use run_with_mesh so the layered-AI landmark verification (S5)
        # surfaces in the response payload alongside measurements. The
        # mesh result is discarded — callers needing it use measure_with_mesh.
        # `enable_*=None` lets the orchestrator's construction-time
        # defaults stand; concrete True/False is a per-call override.
        result = self._pipeline.run_with_mesh(
            Path(image_path),
            height_cm=height_cm,
            reference_object_cm=reference_object_cm,
            gender_profile=gender_profile,
            subject_id=subject_id,
            enable_smpl=enable_smpl,
            enable_claude=enable_claude,
        )
        out = result.measurements.to_dict()
        if result.landmark_verification is not None and result.landmark_verification.verdicts:
            out["landmark_verification"] = {
                "verdicts": {
                    name: {
                        "verdict": v.verdict,
                        "x": v.x,
                        "y": v.y,
                    }
                    for name, v in result.landmark_verification.verdicts.items()
                },
            }
        if result.pose is not None:
            # S2.5 — per-landmark normalized coordinates so consumers can
            # render an overlay on the source photo. Names match
            # `cv_pipeline.pose_detector.LANDMARK_NAMES`.
            out["landmarks"] = {
                name: {
                    "x": kp.x,
                    "y": kp.y,
                    "visibility": kp.visibility,
                }
                for name, kp in result.pose.keypoints.items()
            }
        return out

    def measure_with_mesh(
        self,
        image_path: str | Path,
        height_cm: float | None = None,
        reference_object_cm: float | None = None,
        gender_profile: str = "ladies",
        subject_id: str | None = None,
        enable_smpl: bool | None = None,
        enable_claude: bool | None = None,
    ) -> tuple[Any, Any | None, list[float], list[float]]:
        """Measurements + the fitted SMPL mesh.

        Returns `(BodyMeasurements, mesh, betas, thetas)`. `mesh` is None
        when SMPL fitting was disabled or failed; the avatar route should
        return a 5xx in that case rather than render an unscaled avatar.
        Replaces the broken `/measure/avatar` call site (audit FS-C6).
        """
        result = self._pipeline.run_with_mesh(
            Path(image_path),
            height_cm=height_cm,
            reference_object_cm=reference_object_cm,
            gender_profile=gender_profile,
            subject_id=subject_id,
            enable_smpl=enable_smpl,
            enable_claude=enable_claude,
        )
        return (
            result.measurements,
            result.mesh,
            result.betas or [],
            result.thetas or [],
        )
