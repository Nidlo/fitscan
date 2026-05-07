"""Tests for pipeline orchestrator with mock dependencies."""
import numpy as np
import pytest

from fitscan.application.pipeline_orchestrator import PipelineOrchestrator
from fitscan.domain.exceptions import ImageQualityError
from fitscan.domain.models import (
    BodyMeasurements,
    BodyPose,
    Keypoint,
    ProcessedImage,
    ScaleFactor,
    SingleMeasurement,
    ValidationResult,
)


def _make_pose():
    kps = {
        "nose": Keypoint(0.5, 0.05, 0, 1.0),
        "left_shoulder": Keypoint(0.35, 0.2, 0, 1.0),
        "right_shoulder": Keypoint(0.65, 0.2, 0, 1.0),
        "left_hip": Keypoint(0.4, 0.55, 0, 1.0),
        "right_hip": Keypoint(0.6, 0.55, 0, 1.0),
        "left_ankle": Keypoint(0.4, 0.95, 0, 1.0),
        "right_ankle": Keypoint(0.6, 0.95, 0, 1.0),
        "left_foot_index": Keypoint(0.4, 0.96, 0, 0.8),
        "right_foot_index": Keypoint(0.6, 0.96, 0, 0.8),
    }
    return BodyPose(keypoints=kps, image_width=400, image_height=800)


class MockImageLoader:
    def load(self, path):
        return np.zeros((800, 400, 3), dtype=np.uint8)

    def validate_quality(self, image):
        return True, "OK"


class MockFailLoader:
    def load(self, path):
        return np.zeros((800, 400, 3), dtype=np.uint8)

    def validate_quality(self, image):
        return False, "Too blurry"


class MockPreprocessor:
    def process(self, image):
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return ProcessedImage(
            original=image,
            no_background=image,
            silhouette_mask=mask,
            width=w,
            height=h,
        )


class MockPoseDetector:
    def detect(self, image):
        return _make_pose()


class MockScaleCalibrator:
    def calibrate_from_height(self, pose, height_cm):
        return ScaleFactor(pixels_per_cm=5.0, source="user_height", confidence=0.9)

    def calibrate_from_reference_object(self, image, object_cm):
        return ScaleFactor(pixels_per_cm=5.0, source="reference_object", confidence=0.95)


class MockSilhouetteMeasurer:
    def measure(self, mask, pose, scale, **kwargs):
        return BodyMeasurements(
            waist_cm=SingleMeasurement(75.0, 0.7, "silhouette"),
            source="silhouette",
        )


class MockRuleValidator:
    def validate(self, image, measurements):
        return ValidationResult(
            is_valid=True,
            flagged_fields=[],
            photo_quality_score=8.0,
            quality_issues=[],
            recommendation="proceed",
            raw_response="rule_ok",
        )


class MockClaudeValidator:
    def validate(self, image, measurements):
        return ValidationResult(
            is_valid=True,
            flagged_fields=[],
            photo_quality_score=9.0,
            quality_issues=[],
            recommendation="proceed",
            raw_response="claude_ok",
        )


class MockFuser:
    def fuse(self, sources):
        return sources[0] if sources else BodyMeasurements()


class MockScorer:
    def score(self, measurements, validation, sources):
        return measurements


class MockBodyFitter:
    def fit(self, image, pose):
        raise RuntimeError("SMPL not available")


class MockMeshAnalyzer:
    def measure(self, mesh, scale, gender_profile="ladies"):
        return BodyMeasurements(source="smpl")


class TestPipelineOrchestrator:
    def _make_pipeline(self, **kwargs):
        defaults = dict(
            image_loader=MockImageLoader(),
            preprocessor=MockPreprocessor(),
            pose_detector=MockPoseDetector(),
            scale_calibrator=MockScaleCalibrator(),
            silhouette_measurer=MockSilhouetteMeasurer(),
            body_fitter=MockBodyFitter(),
            mesh_analyzer=MockMeshAnalyzer(),
            rule_validator=MockRuleValidator(),
            claude_validator=MockClaudeValidator(),
            fuser=MockFuser(),
            confidence_scorer=MockScorer(),
            enable_claude=False,
            enable_smpl=False,
        )
        defaults.update(kwargs)
        return PipelineOrchestrator(**defaults)

    def test_run_basic(self, tmp_path):
        pipeline = self._make_pipeline()
        result = pipeline.run(tmp_path / "test.jpg", height_cm=170.0)
        assert result.source == "silhouette"

    def test_run_with_claude(self, tmp_path):
        pipeline = self._make_pipeline(enable_claude=True)
        result = pipeline.run(tmp_path / "test.jpg", height_cm=170.0)
        assert result is not None

    def test_run_raises_on_bad_quality(self, tmp_path):
        pipeline = self._make_pipeline(image_loader=MockFailLoader())
        with pytest.raises(ImageQualityError):
            pipeline.run(tmp_path / "test.jpg", height_cm=170.0)

    def test_run_requires_scale_input(self, tmp_path):
        pipeline = self._make_pipeline()
        with pytest.raises(ValueError, match="Must provide"):
            pipeline.run(tmp_path / "test.jpg")

    def test_smpl_failure_graceful(self, tmp_path):
        pipeline = self._make_pipeline(enable_smpl=True)
        result = pipeline.run(tmp_path / "test.jpg", height_cm=170.0)
        assert result is not None

    def test_validators_disagree_surfaces_in_degraded_modes(self, tmp_path):
        """FS-NIDLO-VALID-03 — when rule + Claude disagree on is_valid,
        the merged ValidationResult sets `degraded_modes=['validators_disagree']`
        and the PipelineResult forwards it to the response payload.
        Frontend renders a "manual review recommended" badge off this.
        """

        class DisagreeingClaude:
            def validate(self, image, measurements, subject_id=None):
                return ValidationResult(
                    is_valid=False,  # Claude says NO; rule mock says YES
                    flagged_fields=[],
                    photo_quality_score=4.0,
                    quality_issues=["bad_lighting"],
                    recommendation="retake",
                    raw_response="claude_disagree",
                )

            def verify_landmarks(self, image, occluded, subject_id=None):
                from fitscan.domain.models import LandmarkVerification

                return LandmarkVerification(verdicts={}, raw_response="")

        pipeline = self._make_pipeline(
            enable_claude=True,
            claude_validator=DisagreeingClaude(),
        )
        result = pipeline.run_with_mesh(tmp_path / "test.jpg", height_cm=170.0)

        assert "validators_disagree" in result.degraded_modes

    def test_validators_agree_emits_empty_degraded_modes(self, tmp_path):
        """The flip side: when both validators agree on is_valid, no
        degradation flag fires. Pinning the empty default so an
        accidentally-always-True helper can't slip through review."""
        pipeline = self._make_pipeline(enable_claude=True)
        result = pipeline.run_with_mesh(tmp_path / "test.jpg", height_cm=170.0)

        assert result.degraded_modes == []
