"""S5 — Layered-AI landmark verification.

Tests the three new contracts:
  - PoseDetector.find_occluded_landmarks (pure function over a BodyPose)
  - ClaudeValidator.verify_landmarks (consent gate + Anthropic round-trip)
  - PipelineOrchestrator wires them only when SMPL/Claude is enabled,
    landmarks are occluded, AND subject_id is provided.

All Anthropic interactions are monkeypatched — no real API calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from fitscan.cv_pipeline.pose_detector import PoseDetector
from fitscan.domain.models import (
    BodyMeasurements,
    BodyPose,
    Keypoint,
    LandmarkVerdict,
    LandmarkVerification,
    ProcessedImage,
    ScaleFactor,
    SingleMeasurement,
    ValidationResult,
)
from fitscan.validation.claude_validator import ClaudeValidator


def _pose_with_visibilities(viz: dict[str, float]) -> BodyPose:
    return BodyPose(
        keypoints={
            name: Keypoint(x=0.5, y=0.5, z=0.0, visibility=v)
            for name, v in viz.items()
        },
        image_width=400,
        image_height=800,
    )


# ────────────────────── PoseDetector.find_occluded_landmarks ──────────────────────


class TestFindOccludedLandmarks:
    def test_returns_empty_when_all_visible(self):
        pose = _pose_with_visibilities(
            {"left_shoulder": 0.9, "right_shoulder": 0.95, "left_hip": 0.8}
        )
        assert PoseDetector.find_occluded_landmarks(pose) == []

    def test_flags_landmarks_below_threshold(self):
        pose = _pose_with_visibilities(
            {
                "left_shoulder": 0.9,   # visible
                "right_shoulder": 0.1,  # occluded
                "left_hip": 0.05,       # occluded
                "right_hip": 0.95,      # visible
            }
        )
        result = PoseDetector.find_occluded_landmarks(pose)
        assert result == ["left_hip", "right_shoulder"]  # sorted

    def test_threshold_is_strict_less_than(self):
        # Exactly at threshold counts as VISIBLE — only `< 0.3` is occluded.
        pose = _pose_with_visibilities({"left_shoulder": 0.3})
        assert PoseDetector.find_occluded_landmarks(pose) == []

    def test_custom_threshold(self):
        pose = _pose_with_visibilities(
            {"left_shoulder": 0.4, "right_shoulder": 0.6}
        )
        # Stricter threshold → both occluded
        assert PoseDetector.find_occluded_landmarks(pose, threshold=0.7) == [
            "left_shoulder",
            "right_shoulder",
        ]


# ────────────────────── ClaudeValidator.verify_landmarks ──────────────────────


class TestClaudeVerifyLandmarks:
    def _img(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_skips_when_no_occluded_landmarks(self):
        # No occluded → no API call needed → empty verification
        validator = ClaudeValidator(api_key="YOUR_FAKE_ANT_KEY_for_consent_gate_tests")
        result = validator.verify_landmarks(
            self._img(), occluded=[], subject_id="user-1"
        )
        assert isinstance(result, LandmarkVerification)
        assert result.verdicts == {}
        assert result.raw_response == ""

    def test_skips_when_subject_id_missing_even_with_real_key(self):
        # The S5 path inherits the S6 consent gate: no subject_id → no
        # photo lands at Anthropic, regardless of occluded list size.
        validator = ClaudeValidator(api_key="YOUR_FAKE_ANT_KEY_for_consent_gate_tests")
        result = validator.verify_landmarks(
            self._img(), occluded=["left_hip"], subject_id=None
        )
        assert result.verdicts == {}
        assert result.raw_response == ""

    def test_skips_with_test_key_placeholder(self):
        validator = ClaudeValidator(api_key="test_key_placeholder")
        result = validator.verify_landmarks(
            self._img(), occluded=["left_hip"], subject_id="user-1"
        )
        assert result.verdicts == {}

    def test_calls_anthropic_with_metadata_and_parses_verdicts(self, monkeypatch):
        captured: dict = {}

        class _FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)

                class _Content:
                    text = (
                        '{"verifications": {'
                        '"left_hip": {"verdict": "visible_at_xy", "x": 0.42, "y": 0.55},'
                        '"right_shoulder": {"verdict": "pose_estimate", "x": 0.6, "y": 0.22},'
                        '"left_wrist": {"verdict": "occluded_unrecoverable", "x": null, "y": null}'
                        '}}'
                    )

                class _Resp:
                    content = [_Content()]

                return _Resp()

        class _FakeAnthropic:
            def __init__(self, api_key=None):
                self.messages = _FakeMessages()

        import anthropic
        monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

        validator = ClaudeValidator(api_key="YOUR_FAKE_ANT_KEY_for_consent_gate_tests")
        result = validator.verify_landmarks(
            self._img(),
            occluded=["left_hip", "right_shoulder", "left_wrist"],
            subject_id="user-abc-123",
        )

        # Consent token forwarded for retention control (S6)
        assert captured["metadata"] == {"user_id": "user-abc-123"}

        # Three verdicts parsed cleanly
        assert set(result.verdicts.keys()) == {
            "left_hip",
            "right_shoulder",
            "left_wrist",
        }
        assert result.verdicts["left_hip"].verdict == "visible_at_xy"
        assert result.verdicts["left_hip"].x == pytest.approx(0.42)
        assert result.verdicts["left_wrist"].verdict == "occluded_unrecoverable"
        assert result.verdicts["left_wrist"].x is None
        assert result.verdicts["left_wrist"].y is None

    def test_drops_unknown_verdicts_silently(self, monkeypatch):
        """If Claude returns a verdict string we don't recognise, ignore
        that entry rather than crashing — defence in depth."""

        class _FakeMessages:
            def create(self, **kwargs):
                class _Content:
                    text = (
                        '{"verifications": {'
                        '"left_hip": {"verdict": "made_up_verdict", "x": 0.4, "y": 0.5},'
                        '"right_hip": {"verdict": "visible_at_xy", "x": 0.6, "y": 0.5}'
                        '}}'
                    )

                class _Resp:
                    content = [_Content()]

                return _Resp()

        class _FakeAnthropic:
            def __init__(self, api_key=None):
                self.messages = _FakeMessages()

        import anthropic
        monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

        validator = ClaudeValidator(api_key="YOUR_FAKE_ANT_KEY_for_consent_gate_tests")
        result = validator.verify_landmarks(
            self._img(),
            occluded=["left_hip", "right_hip"],
            subject_id="user-1",
        )

        # Only the recognised verdict survives
        assert "right_hip" in result.verdicts
        assert "left_hip" not in result.verdicts

    def test_recovers_from_anthropic_error(self, monkeypatch):
        class _FakeMessages:
            def create(self, **kwargs):
                raise RuntimeError("Anthropic 502")

        class _FakeAnthropic:
            def __init__(self, api_key=None):
                self.messages = _FakeMessages()

        import anthropic
        monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

        validator = ClaudeValidator(api_key="YOUR_FAKE_ANT_KEY_for_consent_gate_tests")
        result = validator.verify_landmarks(
            self._img(), occluded=["left_hip"], subject_id="user-1"
        )
        # Soft-fail: empty verdicts, never raises through to the pipeline
        assert result.verdicts == {}


# ────────────────────── PipelineOrchestrator wiring ──────────────────────


class TestPipelineOrchestratorVerification:
    """Light wiring tests: confirm the orchestrator calls verify_landmarks
    only when the right preconditions hold (Claude enabled + occluded
    landmarks present), and that the result threads through to the
    PipelineResult."""

    def _build_pipeline(self, *, claude_validator, enable_claude, pose):
        """Mock everything except claude + pose so we can drive
        run_with_mesh deterministically."""
        from fitscan.application.pipeline_orchestrator import PipelineOrchestrator

        class _Loader:
            def load(self, _):
                return np.zeros((400, 400, 3), dtype=np.uint8)

            def validate_quality(self, _):
                return True, "OK"

        class _Pre:
            def process(self, image):
                return ProcessedImage(
                    original=image,
                    no_background=image,
                    silhouette_mask=np.ones((400, 400), dtype=np.uint8) * 255,
                    width=400,
                    height=400,
                )

        class _Pose:
            def detect(self, _):
                return pose

        class _Scale:
            def calibrate_from_height(self, _, h):
                return ScaleFactor(
                    pixels_per_cm=5.0,
                    source="user_height",
                    confidence=0.9,
                    target_height_cm=h,
                )

            def calibrate_from_reference_object(self, *_):
                return ScaleFactor(
                    pixels_per_cm=5.0, source="reference", confidence=0.9
                )

        class _Sil:
            def measure(self, *_, **__):
                return BodyMeasurements(
                    waist_cm=SingleMeasurement(75.0, 0.7, "silhouette"),
                    source="silhouette",
                )

        class _Rules:
            def validate(self, *_):
                return ValidationResult(
                    is_valid=True,
                    flagged_fields=[],
                    photo_quality_score=8.0,
                    quality_issues=[],
                    recommendation="proceed",
                    raw_response="",
                )

        class _Fuse:
            def fuse(self, sources):
                return sources[0] if sources else BodyMeasurements()

        class _Score:
            def score(self, m, *_):
                return m

        class _Fitter:
            def fit(self, *_):
                raise RuntimeError("no SMPL")

        class _Mesh:
            def measure(self, *_, **__):
                return BodyMeasurements(source="smpl")

        return PipelineOrchestrator(
            image_loader=_Loader(),
            preprocessor=_Pre(),
            pose_detector=_Pose(),
            scale_calibrator=_Scale(),
            silhouette_measurer=_Sil(),
            body_fitter=_Fitter(),
            mesh_analyzer=_Mesh(),
            rule_validator=_Rules(),
            claude_validator=claude_validator,
            fuser=_Fuse(),
            confidence_scorer=_Score(),
            enable_smpl=False,
            enable_claude=enable_claude,
        )

    def test_skips_verification_when_claude_disabled(self, tmp_path):
        called = []

        class _StubClaude:
            def verify_landmarks(self, *args, **kwargs):
                called.append(kwargs)
                return LandmarkVerification(verdicts={}, raw_response="")

            def validate(self, *args, **kwargs):
                return ValidationResult(
                    is_valid=True,
                    flagged_fields=[],
                    photo_quality_score=8.0,
                    quality_issues=[],
                    recommendation="proceed",
                    raw_response="",
                )

        # Pose with one occluded landmark — but Claude is OFF
        pose = _pose_with_visibilities(
            {"left_shoulder": 0.9, "right_hip": 0.1}
        )
        pipeline = self._build_pipeline(
            claude_validator=_StubClaude(),
            enable_claude=False,
            pose=pose,
        )

        result = pipeline.run_with_mesh(
            tmp_path / "x.jpg",
            height_cm=170.0,
            subject_id="user-1",
        )
        assert called == []
        assert result.landmark_verification is None

    def test_skips_verification_when_no_occluded_landmarks(self, tmp_path):
        called = []

        class _StubClaude:
            def verify_landmarks(self, *args, **kwargs):
                called.append(kwargs)
                return LandmarkVerification(verdicts={}, raw_response="")

            def validate(self, *args, **kwargs):
                return ValidationResult(
                    is_valid=True,
                    flagged_fields=[],
                    photo_quality_score=8.0,
                    quality_issues=[],
                    recommendation="proceed",
                    raw_response="",
                )

        # All landmarks above threshold
        pose = _pose_with_visibilities(
            {"left_shoulder": 0.9, "right_hip": 0.85}
        )
        pipeline = self._build_pipeline(
            claude_validator=_StubClaude(),
            enable_claude=True,
            pose=pose,
        )

        result = pipeline.run_with_mesh(
            tmp_path / "x.jpg",
            height_cm=170.0,
            subject_id="user-1",
        )
        assert called == []  # no occluded → no Claude call
        assert result.landmark_verification is None

    def test_calls_verification_and_threads_result_through(self, tmp_path):
        """Happy path — Claude enabled + occluded landmarks present →
        verify_landmarks called with the right arguments and the result
        ends up on PipelineResult.landmark_verification."""

        class _StubClaude:
            def verify_landmarks(self, image, occluded, subject_id):
                return LandmarkVerification(
                    verdicts={
                        "right_hip": LandmarkVerdict(
                            landmark="right_hip",
                            verdict="pose_estimate",
                            x=0.6,
                            y=0.5,
                        )
                    },
                    raw_response="ok",
                )

            def validate(self, *args, **kwargs):
                return ValidationResult(
                    is_valid=True,
                    flagged_fields=[],
                    photo_quality_score=8.0,
                    quality_issues=[],
                    recommendation="proceed",
                    raw_response="",
                )

        pose = _pose_with_visibilities(
            {"left_shoulder": 0.9, "right_hip": 0.1}
        )
        pipeline = self._build_pipeline(
            claude_validator=_StubClaude(),
            enable_claude=True,
            pose=pose,
        )

        result = pipeline.run_with_mesh(
            tmp_path / "x.jpg",
            height_cm=170.0,
            subject_id="user-1",
        )
        assert result.landmark_verification is not None
        assert "right_hip" in result.landmark_verification.verdicts
        assert (
            result.landmark_verification.verdicts["right_hip"].verdict
            == "pose_estimate"
        )


# ────────────────────── API schema surface ──────────────────────


class TestApiSchemaLandmarkVerification:
    def test_landmark_verification_field_defaults_to_none(self):
        from fitscan.api.schemas import MeasurementsResponse

        resp = MeasurementsResponse(status="success")
        assert resp.landmark_verification is None

    def test_landmark_verification_serialises_in_dump(self):
        from fitscan.api.schemas import MeasurementsResponse

        resp = MeasurementsResponse(
            status="success",
            landmark_verification={
                "verdicts": {
                    "left_hip": {
                        "verdict": "visible_at_xy",
                        "x": 0.42,
                        "y": 0.55,
                    }
                }
            },
        )
        dumped = resp.model_dump()
        assert dumped["landmark_verification"]["verdicts"]["left_hip"][
            "verdict"
        ] == "visible_at_xy"
