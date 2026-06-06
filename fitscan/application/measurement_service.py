"""
High-level service. Wires all dependencies together.
This is the only place where concrete classes are instantiated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fitscan.application.pipeline_orchestrator import PipelineOrchestrator
from fitscan.config import get_settings
from fitscan.cv_pipeline.pose_detector import PoseDetector
from fitscan.cv_pipeline.preprocessor import ImagePreprocessor
from fitscan.cv_pipeline.scale_calibrator import ScaleCalibrator
from fitscan.cv_pipeline.silhouette import SilhouetteMeasurer
from fitscan.fusion.confidence import ConfidenceScorer
from fitscan.fusion.fuser import MeasurementFuser
from fitscan.infrastructure.image_loader import ImageLoader
from fitscan.llm import create_llm_provider
from fitscan.smpl.mesh_analyzer import MeshAnalyzer
from fitscan.validation.claude_validator import ClaudeValidator
from fitscan.validation.rule_validator import RuleValidator

# SMPLBodyFitter is imported lazily inside create_pipeline() - its
# module-level `import torch` would otherwise force every importer of
# this module (unit tests, ops scripts, anything that wires up the
# pipeline) to install the [ml] optional extra. Lazy keeps the base
# install lean while still wiring the same fitter into production.


def _resolve_anthropic_key() -> str:
    """Read ANTHROPIC_API_KEY with .env taking precedence over os.environ.

    Other tools that share the user's shell (Claude Code, for one) also
    export `ANTHROPIC_API_KEY` - typically a *session* token (`sk-ant-sid...`)
    that the Anthropic Public API rejects with 401. Fitscan's `.env` is the
    deploy-controlled surface for *its* Anthropic API key (`sk-ant-api03-...`),
    and the worker can't tell a session token from an API key by inspection,
    so the rule is: when `.env` defines the key, it wins. Falls back to
    os.environ only when the `.env` doesn't carry one, which preserves the
    production path where the orchestrator injects secrets as env vars and
    there's no `.env` on disk.
    """
    from dotenv import dotenv_values

    repo_root = Path(__file__).resolve().parent.parent.parent
    env_file = repo_root / ".env"
    if env_file.exists():
        values = dotenv_values(env_file)
        env_key = values.get("ANTHROPIC_API_KEY")
        if env_key:
            return env_key

    return os.getenv("ANTHROPIC_API_KEY", "")


def _build_llm_provider() -> Any:
    """Build the configured LLM provider from settings + the resolved key.

    Provider selection (`FITSCAN_LLM_PROVIDER`) and Claude Code auth live in
    `Settings`; the Anthropic key keeps its `.env`-precedence resolution so
    a stray Claude Code session token in the shell can't shadow the
    deploy-controlled key on the direct-API path.
    """
    settings = get_settings()
    return create_llm_provider(
        provider=settings.fitscan_llm_provider,
        api_key=_resolve_anthropic_key(),
        claude_code_auth=settings.fitscan_claude_code_auth,
        claude_code_oauth_token=settings.fitscan_claude_code_oauth_token,
        claude_code_model=settings.fitscan_claude_code_model,
        claude_code_mcp_servers=settings.fitscan_claude_code_mcp_servers,
    )


def create_pipeline(
    enable_smpl: bool = False,
    enable_claude: bool = True,
) -> PipelineOrchestrator:
    """Factory function: creates the fully-wired pipeline."""
    # Lazy: keeps the [ml]-stack torch import out of base-install paths.
    from fitscan.smpl.body_fitter import SMPLBodyFitter

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
        claude_validator=ClaudeValidator(
            llm=_build_llm_provider(),
            ensemble_size=get_settings().fitscan_measure_ensemble_size,
        ),
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
        side_image_path: str | Path | None = None,
        height_cm: float | None = None,
        reference_object_cm: float | None = None,
        estimate_height: bool = False,
        gender_profile: str = "ladies",
        subject_id: str | None = None,
        enable_smpl: bool | None = None,
        enable_claude: bool | None = None,
    ) -> dict:
        # Use run_with_mesh so the layered-AI landmark verification (S5)
        # surfaces in the response payload alongside measurements. The
        # mesh result is discarded - callers needing it use measure_with_mesh.
        # `enable_*=None` lets the orchestrator's construction-time
        # defaults stand; concrete True/False is a per-call override.
        # Sprint 34 Phase B - `side_image_path`, when present, lets the
        # silhouette measurer read real body depth instead of the
        # width × 0.6 guess.
        result = self._pipeline.run_with_mesh(
            Path(image_path),
            side_image_path=(
                Path(side_image_path) if side_image_path is not None else None
            ),
            height_cm=height_cm,
            reference_object_cm=reference_object_cm,
            estimate_height=estimate_height,
            gender_profile=gender_profile,
            subject_id=subject_id,
            enable_smpl=enable_smpl,
            enable_claude=enable_claude,
        )
        out = result.measurements.to_dict()
        if (
            result.landmark_verification is not None
            and result.landmark_verification.verdicts
        ):
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
            # S2.5 - per-landmark normalized coordinates so consumers can
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
        # FS-NIDLO-VALID-03 - Always emit `degraded_modes` (empty list
        # when nothing degraded) so consumers can rely on the field
        # existing without doing per-version key checks.
        out["degraded_modes"] = list(result.degraded_modes)
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
