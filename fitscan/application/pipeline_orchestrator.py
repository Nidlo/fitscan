"""
Orchestrates the complete measurement pipeline.
Follows the dependency injection pattern — all dependencies injected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from fitscan.domain.exceptions import ImageQualityError
from fitscan.domain.models import BodyMeasurements, ValidationResult


@dataclass(frozen=True)
class PipelineResult:
    """Full output of the measurement pipeline.

    `measurements` is always populated. `mesh`, `betas`, `thetas` are
    populated only when SMPL fitting was enabled and succeeded; otherwise
    they're None and downstream avatar generation should be skipped.
    """

    measurements: BodyMeasurements
    mesh: Any | None = None
    betas: list[float] | None = None
    thetas: list[float] | None = None


class PipelineOrchestrator:
    """
    Runs the full body measurement pipeline:
    load -> preprocess -> pose -> scale -> cv_measure -> smpl_fit ->
    smpl_measure -> validate -> fuse -> score -> return
    """

    def __init__(
        self,
        image_loader: Any,
        preprocessor: Any,
        pose_detector: Any,
        scale_calibrator: Any,
        silhouette_measurer: Any,
        body_fitter: Any,
        mesh_analyzer: Any,
        rule_validator: Any,
        claude_validator: Any,
        fuser: Any,
        confidence_scorer: Any,
        enable_claude: bool = True,
        enable_smpl: bool = True,
    ) -> None:
        self._loader = image_loader
        self._preprocessor = preprocessor
        self._pose_detector = pose_detector
        self._scale_calibrator = scale_calibrator
        self._silhouette = silhouette_measurer
        self._fitter = body_fitter
        self._mesh = mesh_analyzer
        self._rule_validator = rule_validator
        self._claude_validator = claude_validator
        self._fuser = fuser
        self._scorer = confidence_scorer
        self._enable_claude = enable_claude
        self._enable_smpl = enable_smpl

    def run(
        self,
        image_path: Path,
        height_cm: float | None = None,
        reference_object_cm: float | None = None,
        gender_profile: str = "ladies",
        subject_id: str | None = None,
    ) -> BodyMeasurements:
        return self.run_with_mesh(
            image_path=image_path,
            height_cm=height_cm,
            reference_object_cm=reference_object_cm,
            gender_profile=gender_profile,
            subject_id=subject_id,
        ).measurements

    def run_with_mesh(
        self,
        image_path: Path,
        height_cm: float | None = None,
        reference_object_cm: float | None = None,
        gender_profile: str = "ladies",
        subject_id: str | None = None,
    ) -> PipelineResult:
        """Like `run()` but also surfaces the SMPL mesh + parameters when
        available. Used by `/measure/avatar` to render the GLB without
        re-running the pipeline.
        """
        logger.info(f"Processing {image_path}")
        image = self._loader.load(image_path)
        ok, reason = self._loader.validate_quality(image)
        if not ok:
            raise ImageQualityError(reason)

        processed = self._preprocessor.process(image)
        pose = self._pose_detector.detect(processed.original)

        if height_cm:
            scale = self._scale_calibrator.calibrate_from_height(pose, height_cm)
        elif reference_object_cm:
            scale = self._scale_calibrator.calibrate_from_reference_object(
                image, reference_object_cm
            )
        else:
            raise ValueError("Must provide height_cm or reference_object_cm for scale.")

        sources: list[BodyMeasurements] = []
        silhouette_result = self._silhouette.measure(
            processed.silhouette_mask,
            pose,
            scale,
            gender_profile=gender_profile,
        )
        sources.append(silhouette_result)

        mesh: Any | None = None
        if self._enable_smpl:
            try:
                raw_mesh = self._fitter.fit(processed.original, pose)
                mesh = self._rescale_mesh_to_height(raw_mesh, scale)
                smpl_result = self._mesh.measure(mesh, scale, gender_profile)
                sources.append(smpl_result)
            except Exception as e:
                logger.warning(f"SMPL fitting failed, using silhouette only: {e}")
                mesh = None

        fused = self._fuser.fuse(sources)

        rule_result = self._rule_validator.validate(image, fused)

        if self._enable_claude:
            try:
                # Body photo is shipped to Anthropic only when subject_id is
                # supplied (explicit consent). Without it the validator no-ops
                # and returns a default-pass result.
                claude_result = self._claude_validator.validate(
                    image, fused, subject_id=subject_id
                )
                validation = self._merge_validations(rule_result, claude_result)
            except Exception as e:
                logger.warning(f"Claude validation skipped: {e}")
                validation = rule_result
        else:
            validation = rule_result

        final = self._scorer.score(fused, validation, sources)
        logger.info(f"Pipeline complete. Height: {final.height_cm}")
        return PipelineResult(measurements=final, mesh=mesh)

    @staticmethod
    def _rescale_mesh_to_height(mesh: Any, scale: Any) -> Any:
        """Rescale an SMPL mesh from canonical template space to the user's
        actual height, when one was supplied at calibration time. Without
        this every height/circumference output clusters around the SMPL
        template defaults (audit FS-H13).
        """
        if scale.target_height_cm is None:
            return mesh

        import trimesh

        from fitscan.smpl.smpl_utils import scale_mesh_to_height

        scaled = scale_mesh_to_height(np.asarray(mesh.vertices), scale.target_height_cm)
        return trimesh.Trimesh(vertices=scaled, faces=mesh.faces, process=False)

    def _merge_validations(
        self, rule: ValidationResult, claude: ValidationResult
    ) -> ValidationResult:
        flagged = list(set(rule.flagged_fields + claude.flagged_fields))
        issues = rule.quality_issues + claude.quality_issues
        score = (rule.photo_quality_score + claude.photo_quality_score) / 2
        recommend = (
            "retake"
            if "retake" in [rule.recommendation, claude.recommendation]
            else "proceed"
        )
        return ValidationResult(
            is_valid=len(flagged) <= 1,
            flagged_fields=flagged,
            photo_quality_score=score,
            quality_issues=issues,
            recommendation=recommend,
            raw_response=(f"rule: {rule.raw_response} | claude: {claude.raw_response}"),
        )
