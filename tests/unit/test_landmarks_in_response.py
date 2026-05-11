"""S2.5 — `MeasurementService.measure()` surfaces the MediaPipe pose's
per-landmark coordinates so frontend consumers can render an overlay.

Tests are hermetic — `pipeline.run_with_mesh` is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitscan.application.measurement_service import MeasurementService
from fitscan.application.pipeline_orchestrator import PipelineResult
from fitscan.domain.models import (
    BodyMeasurements,
    BodyPose,
    Keypoint,
    SingleMeasurement,
)


def _measurements() -> BodyMeasurements:
    return BodyMeasurements(
        shoulder_cm=SingleMeasurement(40.0, 0.8, "geometric"),
        source="geometric",
    )


def _pose_with(visibilities: dict[str, float]) -> BodyPose:
    return BodyPose(
        keypoints={
            name: Keypoint(x=0.5, y=0.5, z=0.0, visibility=v)
            for name, v in visibilities.items()
        },
        image_width=400,
        image_height=800,
    )


class TestLandmarksInResponse:
    def test_response_includes_landmarks_when_pose_is_present(self):
        mock_pipeline = MagicMock()
        mock_pipeline.run_with_mesh.return_value = PipelineResult(
            measurements=_measurements(),
            pose=_pose_with(
                {
                    "left_shoulder": 0.92,
                    "right_shoulder": 0.91,
                    "left_hip": 0.4,
                }
            ),
        )

        svc = MeasurementService(pipeline=mock_pipeline)
        result = svc.measure("/tmp/x.jpg", height_cm=170.0)

        assert "landmarks" in result
        # Each landmark serialises as { x, y, visibility }
        assert result["landmarks"]["left_shoulder"]["x"] == 0.5
        assert result["landmarks"]["left_shoulder"]["y"] == 0.5
        assert result["landmarks"]["left_shoulder"]["visibility"] == 0.92
        # All three landmarks present
        assert set(result["landmarks"].keys()) == {
            "left_shoulder",
            "right_shoulder",
            "left_hip",
        }

    def test_response_omits_landmarks_when_pose_is_none(self):
        # E.g. pipeline failed pose detection — landmarks key shouldn't
        # appear at all (so existing consumers don't see a None and choke).
        mock_pipeline = MagicMock()
        mock_pipeline.run_with_mesh.return_value = PipelineResult(
            measurements=_measurements(),
            pose=None,
        )

        svc = MeasurementService(pipeline=mock_pipeline)
        result = svc.measure("/tmp/x.jpg", height_cm=170.0)

        assert "landmarks" not in result

    def test_landmarks_coexist_with_landmark_verification(self):
        from fitscan.domain.models import LandmarkVerdict, LandmarkVerification

        mock_pipeline = MagicMock()
        mock_pipeline.run_with_mesh.return_value = PipelineResult(
            measurements=_measurements(),
            pose=_pose_with({"left_shoulder": 0.9}),
            landmark_verification=LandmarkVerification(
                verdicts={
                    "left_hip": LandmarkVerdict(
                        landmark="left_hip",
                        verdict="pose_estimate",
                        x=0.45,
                        y=0.55,
                    )
                },
                raw_response="ok",
            ),
        )

        svc = MeasurementService(pipeline=mock_pipeline)
        result = svc.measure("/tmp/x.jpg", height_cm=170.0)

        # Both fields surface independently
        assert "landmarks" in result
        assert "landmark_verification" in result
        assert result["landmarks"]["left_shoulder"]["visibility"] == 0.9
        assert (
            result["landmark_verification"]["verdicts"]["left_hip"]["verdict"]
            == "pose_estimate"
        )
