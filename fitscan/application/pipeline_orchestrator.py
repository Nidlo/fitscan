"""
Orchestrates the complete measurement pipeline.
Follows the dependency injection pattern — all dependencies injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from fitscan.cv_pipeline.pose_detector import PoseDetector
from fitscan.domain.exceptions import ImageQualityError
from fitscan.domain.models import (
    BodyMeasurements,
    BodyPose,
    LandmarkVerification,
    ValidationResult,
)


@dataclass(frozen=True)
class PipelineResult:
    """Full output of the measurement pipeline.

    `measurements` is always populated. `mesh`, `betas`, `thetas` are
    populated only when SMPL fitting was enabled and succeeded; otherwise
    they're None and downstream avatar generation should be skipped.

    `landmark_verification` is populated only when occluded landmarks
    were detected AND the caller opted in via `subject_id` (S5).

    `pose` is the raw MediaPipe pose detection — surfaced so the API
    response can carry per-landmark coordinates for the S2.5 photo
    overlay.
    """

    measurements: BodyMeasurements
    mesh: Any | None = None
    betas: list[float] | None = None
    thetas: list[float] | None = None
    landmark_verification: LandmarkVerification | None = None
    pose: BodyPose | None = None
    # FS-NIDLO-VALID-03 — Surface `ValidationResult.degraded_modes`
    # at the pipeline boundary so the API response can carry it.
    # Empty list when nothing degraded.
    degraded_modes: list[str] = field(default_factory=list)


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
        enable_smpl: bool | None = None,
        enable_claude: bool | None = None,
    ) -> BodyMeasurements:
        return self.run_with_mesh(
            image_path=image_path,
            height_cm=height_cm,
            reference_object_cm=reference_object_cm,
            gender_profile=gender_profile,
            subject_id=subject_id,
            enable_smpl=enable_smpl,
            enable_claude=enable_claude,
        ).measurements

    def run_with_mesh(
        self,
        image_path: Path,
        height_cm: float | None = None,
        reference_object_cm: float | None = None,
        gender_profile: str = "ladies",
        subject_id: str | None = None,
        enable_smpl: bool | None = None,
        enable_claude: bool | None = None,
        side_image_path: Path | None = None,
        estimate_height: bool = False,
    ) -> PipelineResult:
        """Like `run()` but also surfaces the SMPL mesh + parameters when
        available. Used by `/measure/avatar` to render the GLB without
        re-running the pipeline.

        `enable_smpl` and `enable_claude` are per-call overrides. None
        falls back to the orchestrator-level defaults set at construction
        time. The pipeline is `lru_cache(maxsize=1)`-singleton per worker
        so we can't toggle the flags by rebuilding it; per-call overrides
        let `/measure/measurements` honour the form fields without
        invalidating the cache. (FS-NIDLO-MEAS-05)

        `side_image_path` (Sprint 34 Phase B) — optional side-profile
        photo. When present, the silhouette measurer reads body depth from
        its silhouette mask instead of guessing depth = width × 0.6. This
        is the highest-impact accuracy unlock: a single front photo can't
        tell a deep torso from a flat one, but front + side triangulates
        the third dimension geometrically.
        """
        # Resolve per-call overrides; None = use orchestrator default.
        smpl_on = self._enable_smpl if enable_smpl is None else enable_smpl
        claude_on = self._enable_claude if enable_claude is None else enable_claude

        logger.info(f"Processing {image_path}")
        image = self._loader.load(image_path)
        ok, reason = self._loader.validate_quality(image)
        if not ok:
            raise ImageQualityError(reason)

        processed = self._preprocessor.process(image)
        pose = self._pose_detector.detect(processed.original)

        # Sprint 34 Phase B — process the side photo when supplied. We
        # only need its silhouette mask + a usable bounding box; pose
        # detection runs to find head + foot for vertical normalisation.
        # `side_processed_image` is also surfaced for Phase C+ (Claude
        # measure) so the same preprocessed bytes get reused.
        side_mask = None
        side_pose = None
        side_processed_image: np.ndarray | None = None
        if side_image_path is not None:
            try:
                side_image = self._loader.load(side_image_path)
                side_ok, _ = self._loader.validate_quality(side_image)
                if side_ok:
                    side_processed = self._preprocessor.process(side_image)
                    side_mask = side_processed.silhouette_mask
                    side_pose = self._pose_detector.detect(side_processed.original)
                    side_processed_image = side_processed.original
            except Exception as e:
                # Don't fail the whole scan on a bad side image — degrade
                # to front-only with a logged warning. The silhouette
                # measurer handles `None` side_mask by falling back to
                # the depth-from-width ratio (status quo pre-Phase-B).
                logger.warning(
                    f"Side image processing failed, falling back to front-only: {e}"
                )
                side_mask = None
                side_pose = None
                side_processed_image = None

        # S5 — feed any low-visibility landmarks to Claude for vision-based
        # verification BEFORE measurements compute. Consent-gated by
        # subject_id (audit FS-C3); skipped silently when Claude is off
        # or no landmarks are occluded.
        landmark_verification: LandmarkVerification | None = None
        if claude_on:
            occluded = PoseDetector.find_occluded_landmarks(pose)
            if occluded:
                try:
                    landmark_verification = self._claude_validator.verify_landmarks(
                        processed.original,
                        occluded=occluded,
                        subject_id=subject_id,
                    )
                except Exception as e:
                    logger.warning(f"Landmark verification skipped: {e}")
                    landmark_verification = None

        # Sprint 35 — Claude estimates the height from the front photo when
        # the caller didn't provide height_cm OR a reference object AND
        # explicitly opted into estimation. Cues: door frames, furniture,
        # body proportions, footwear. Returns None if Claude isn't confident
        # enough (< 0.5) — caller then has to error out, since we've run
        # out of scale anchors. The estimated value gets logged so an
        # operator can see what Claude guessed.
        if (
            height_cm is None
            and reference_object_cm is None
            and estimate_height
            and claude_on
            and subject_id
        ):
            try:
                estimated = self._claude_validator.estimate_height(
                    image=processed.original,
                    subject_id=subject_id,
                )
                if estimated is not None:
                    height_cm = estimated
                    logger.info(f"Using Claude-estimated height: {estimated} cm")
            except Exception as e:
                logger.warning(f"Claude height-estimate failed: {e}")

        if height_cm:
            scale = self._scale_calibrator.calibrate_from_height(pose, height_cm)
        elif reference_object_cm:
            scale = self._scale_calibrator.calibrate_from_reference_object(
                image, reference_object_cm
            )
        else:
            raise ValueError("Must provide height_cm or reference_object_cm for scale.")

        # Sprint 36 Stage 1 — Claude is the primary measurer. The CV
        # measurers (silhouette ellipse-perimeter, SMPL beta-fit) were
        # pulling accuracy DOWN on diverse adult bodies (SMPL betas
        # trained on a thin western distribution routinely inflated
        # circumferences 30-50% on real users). They now stay around
        # only as the no-Claude fallback: when consent gate fails, no
        # API key, or Claude errors. SMPL still fits when the avatar
        # route asks for a mesh (smpl_on=True), but its MEASUREMENTS
        # no longer enter fusion — mesh is for rendering, not for
        # estimating waistlines.
        sources: list[BodyMeasurements] = []
        claude_body: BodyMeasurements | None = None
        if claude_on and height_cm is not None and subject_id:
            try:
                claude_body = self._claude_validator.measure(
                    front_image=processed.original,
                    side_image=side_processed_image,
                    height_cm=float(height_cm),
                    gender_profile=gender_profile,
                    subject_id=subject_id,
                )
                if claude_body is not None:
                    sources.append(claude_body)
                    field_count = sum(
                        1
                        for f in claude_body.__dataclass_fields__
                        if getattr(claude_body, f, None) is not None
                        and f not in ("gender_profile", "source")
                    )
                    logger.info(f"Claude measured {field_count} fields")
            except Exception as e:
                logger.warning(f"Claude measure failed: {e}")
                claude_body = None

        if claude_body is None:
            # Fallback: Claude unreachable (no consent, no key, API
            # error). Use silhouette so the caller never receives an
            # empty body. Operators see the WARNING above; the response
            # carries the same `source='silhouette'` tag so consumers
            # know which path produced it.
            logger.info(
                "Claude unavailable for measurement — falling back to silhouette"
            )
            silhouette_result = self._silhouette.measure(
                processed.silhouette_mask,
                pose,
                scale,
                gender_profile=gender_profile,
                side_mask=side_mask,
                side_pose=side_pose,
            )
            sources.append(silhouette_result)

        # SMPL fit produces ONLY the mesh now — its measurements don't
        # join fusion (that's the whole point of Stage 1). The avatar
        # route consumes `mesh` directly; the measurement route ignores
        # it. Default `enable_smpl=False` everywhere except the avatar
        # endpoint.
        mesh: Any | None = None
        if smpl_on:
            try:
                raw_mesh = self._fitter.fit(processed.original, pose)
                mesh = self._rescale_mesh_to_height(raw_mesh, scale)
            except Exception as e:
                logger.warning(f"SMPL fitting failed (mesh unavailable): {e}")
                mesh = None

        fused = self._fuser.fuse(sources)

        rule_result = self._rule_validator.validate(image, fused)

        if claude_on:
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

        # FS-NIDLO-SMPL-09 (Sprint 34 Phase A3) — anthropometric sanity
        # check against adult-population proportion bounds. Each violation
        # surfaces as `anthropometric:<field>` in degraded_modes so the
        # frontend can flag the specific measurement for manual review.
        from fitscan.validation.anthropometric import check as check_anthropometric

        anthropometric_flags = check_anthropometric(final)

        # Sprint 34 Phase C — ask Claude for visual corrections on the
        # flagged fields. Only fires when Claude is enabled, an explicit
        # consent token (subject_id) was supplied, and there's at least
        # one flag to correct. Cost-bounded: one Claude call per scan max,
        # only when the upstream pipeline already produced suspect output.
        applied_corrections: list[str] = []
        if claude_on and anthropometric_flags and height_cm is not None:
            flagged_field_names = [
                tag.split(":", 1)[1]
                for tag in anthropometric_flags
                if tag.startswith("anthropometric:")
            ]
            try:
                corrections = self._claude_validator.correct(
                    image=processed.original,
                    measurements=final,
                    flagged_fields=flagged_field_names,
                    height_cm=float(height_cm),
                    subject_id=subject_id,
                )
                # BodyMeasurements is frozen=True, so setattr blows up with
                # "cannot assign to field <name>". Build the replacement
                # kwargs first, then rebuild the dataclass via
                # `dataclasses.replace`. The "claude_corrected" source tag
                # lets downstream consumers know which fields got rewritten;
                # confidence 0.70 reflects Claude's sanity-but-not-tape
                # accuracy.
                replacements: dict[str, Any] = {}
                for field_name, corrected_value in corrections.items():
                    current = getattr(final, field_name, None)
                    if current is None:
                        continue
                    replacements[field_name] = type(current)(
                        value_cm=round(corrected_value, 1),
                        confidence=0.70,
                        source="claude_corrected",
                    )
                    applied_corrections.append(field_name)
                if replacements:
                    final = replace(final, **replacements)
            except Exception as e:
                logger.warning(f"Claude correction skipped: {e}")

        # Re-run the anthropometric check post-correction. Fields Claude
        # corrected may now pass; only the still-failing ones stay flagged.
        anthropometric_flags = check_anthropometric(final)

        # Sprint 36 booth-coverage — proportional backfill for the
        # bust-relative fields Claude refuses on clothed photos. Without
        # this, the booth's nipple_to_nipple / shoulder_to_nipple /
        # shoulder_to_underbust come back null every time on a typical
        # T-shirt scan, which forces the designer to ask the customer
        # to take an awkward second photo. The backfill uses
        # population-average ratios off bust + shoulder_to_waist; values
        # carry confidence 0.40 and source="anthropometric_estimate" so
        # consumers know these are derived, not measured.
        final = self._backfill_proportional_fields(final)

        logger.info(
            f"Pipeline complete. Height: {final.height_cm} "
            f"corrections={applied_corrections} flags={anthropometric_flags}"
        )

        degraded_modes = list(validation.degraded_modes) + anthropometric_flags
        if applied_corrections:
            degraded_modes.append("claude_corrections:" + ",".join(applied_corrections))

        return PipelineResult(
            measurements=final,
            mesh=mesh,
            landmark_verification=landmark_verification,
            pose=pose,
            degraded_modes=degraded_modes,
        )

    @staticmethod
    def _backfill_proportional_fields(
        measurements: BodyMeasurements,
    ) -> BodyMeasurements:
        """Fill in null bust-relative fields from population-average ratios.

        Nipples + bust-apex landmarks aren't visible through clothing,
        so Claude correctly returns null for nipple_to_nipple,
        shoulder_to_nipple, and shoulder_to_underbust on clothed photos.
        This helper backfills them off:

          - nipple_to_nipple ≈ 0.21 × bust (≈ 19-22 cm on a typical
            adult bust, matching the ISO-derived 0.20-0.24 band)
          - shoulder_to_nipple ≈ 0.50 × shoulder_to_waist
          - shoulder_to_underbust ≈ 0.70 × shoulder_to_waist

        Values get `source="anthropometric_estimate"` and `confidence=0.40`
        so downstream consumers (and the FE's low-confidence badge) can
        distinguish derived from directly-measured. Skips when the
        anchor fields are themselves null — no point hallucinating off
        nothing.
        """
        from fitscan.domain.models import SingleMeasurement

        replacements: dict[str, Any] = {}
        bust = measurements.bust_cm
        s_to_waist = measurements.shoulder_to_waist_cm

        if measurements.nipple_to_nipple_cm is None and bust is not None:
            replacements["nipple_to_nipple_cm"] = SingleMeasurement(
                value_cm=round(0.21 * bust.value_cm, 1),
                confidence=0.40,
                source="anthropometric_estimate",
            )
        if measurements.shoulder_to_nipple_cm is None and s_to_waist is not None:
            replacements["shoulder_to_nipple_cm"] = SingleMeasurement(
                value_cm=round(0.50 * s_to_waist.value_cm, 1),
                confidence=0.40,
                source="anthropometric_estimate",
            )
        if measurements.shoulder_to_underbust_cm is None and s_to_waist is not None:
            replacements["shoulder_to_underbust_cm"] = SingleMeasurement(
                value_cm=round(0.70 * s_to_waist.value_cm, 1),
                confidence=0.40,
                source="anthropometric_estimate",
            )

        if not replacements:
            return measurements
        return replace(measurements, **replacements)

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
        # FS-NIDLO-VALID-03 — when rule + Claude disagree on overall
        # is_valid, surface a structured signal so the frontend can
        # render a "manual review recommended" badge. The merged
        # `is_valid` still reflects the combined flagged-fields count
        # (the historical contract); `degraded_modes` is additive
        # information operators can route on without changing the
        # accept/reject decision.
        degraded: list[str] = []
        if rule.is_valid != claude.is_valid:
            degraded.append("validators_disagree")
        return ValidationResult(
            is_valid=len(flagged) <= 1,
            flagged_fields=flagged,
            photo_quality_score=score,
            quality_issues=issues,
            recommendation=recommend,
            raw_response=(f"rule: {rule.raw_response} | claude: {claude.raw_response}"),
            degraded_modes=degraded,
        )
