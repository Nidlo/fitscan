"""
Unit tests for MeasurementService.

All I/O is mocked — no real images or model weights required.
Verifies that the service correctly delegates to the pipeline and
surfaces results in the expected dict shape.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fitscan.application.measurement_service import MeasurementService
from fitscan.application.pipeline_orchestrator import PipelineResult
from fitscan.domain.models import BodyMeasurements, SingleMeasurement


def make_measurements(**kwargs) -> BodyMeasurements:
    defaults = dict(
        shoulder_cm=SingleMeasurement(value_cm=40.0, confidence=0.75, source="geometric"),
        shoulder_to_hip_cm=SingleMeasurement(value_cm=80.0, confidence=0.74, source="geometric"),
        source="geometric",
        gender_profile="ladies",
    )
    defaults.update(kwargs)
    return BodyMeasurements(**defaults)


class TestMeasurementService:
    def test_measure_delegates_to_pipeline_and_returns_dict(self):
        mock_pipeline = MagicMock()
        measurements = make_measurements()
        mock_pipeline.run_with_mesh.return_value = PipelineResult(measurements=measurements)

        svc = MeasurementService(pipeline=mock_pipeline)
        result = svc.measure("/tmp/front.jpg", height_cm=170.0)

        mock_pipeline.run_with_mesh.assert_called_once_with(
            Path("/tmp/front.jpg"),
            height_cm=170.0,
            reference_object_cm=None,
            gender_profile="ladies",
            subject_id=None,
            # enable_*=None — per-call override falls back to the
            # orchestrator-level defaults set at construction time.
            # (FS-NIDLO-MEAS-05)
            enable_smpl=None,
            enable_claude=None,
        )
        assert isinstance(result, dict)
        assert result["source"] == "geometric"
        assert result["gender_profile"] == "ladies"

    def test_measure_passes_all_kwargs_to_pipeline(self):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = PipelineResult(measurements=make_measurements())

        svc = MeasurementService(pipeline=mock_pipeline)
        svc.measure(
            "/tmp/img.jpg",
            height_cm=160.0,
            reference_object_cm=21.0,
            gender_profile="mens",
            subject_id="user-abc",
        )

        mock_pipeline.run_with_mesh.assert_called_once_with(
            Path("/tmp/img.jpg"),
            height_cm=160.0,
            reference_object_cm=21.0,
            gender_profile="mens",
            subject_id="user-abc",
            enable_smpl=None,
            enable_claude=None,
        )

    def test_measure_forwards_enable_smpl_and_enable_claude_overrides(self):
        # FS-NIDLO-MEAS-05 — `/measure/measurements` exposes enable_smpl
        # and enable_claude form fields. They were silently dropped
        # before; this test pins the wiring so a future refactor can't
        # quietly stop honouring them.
        mock_pipeline = MagicMock()
        mock_pipeline.run_with_mesh.return_value = PipelineResult(
            measurements=make_measurements()
        )

        svc = MeasurementService(pipeline=mock_pipeline)
        svc.measure("/tmp/img.jpg", enable_smpl=False, enable_claude=False)

        kwargs = mock_pipeline.run_with_mesh.call_args.kwargs
        assert kwargs["enable_smpl"] is False
        assert kwargs["enable_claude"] is False

    def test_measure_result_includes_measurement_values(self):
        mock_pipeline = MagicMock()
        measurements = make_measurements(
            shoulder_cm=SingleMeasurement(value_cm=38.5, confidence=0.80, source="geometric"),
        )
        mock_pipeline.run_with_mesh.return_value = PipelineResult(measurements=measurements)

        svc = MeasurementService(pipeline=mock_pipeline)
        result = svc.measure("/tmp/img.jpg")

        assert "shoulder_cm" in result
        assert result["shoulder_cm"]["value_cm"] == 38.5
        assert result["shoulder_cm"]["confidence"] == 0.80

    def test_measure_with_mesh_returns_tuple(self):
        mock_pipeline = MagicMock()
        measurements = make_measurements()
        mock_pipeline.run_with_mesh.return_value = PipelineResult(
            measurements=measurements,
            mesh=None,
            betas=[0.1, 0.2],
            thetas=[0.0, 0.1],
        )

        svc = MeasurementService(pipeline=mock_pipeline)
        result_measurements, mesh, betas, thetas = svc.measure_with_mesh("/tmp/img.jpg")

        assert result_measurements is measurements
        assert mesh is None
        assert betas == [0.1, 0.2]
        assert thetas == [0.0, 0.1]

    def test_create_pipeline_uses_disable_smpl_flag(self):
        """create_pipeline(enable_smpl=False) should not instantiate SMPLBodyFitter."""
        with patch("fitscan.application.measurement_service.SMPLBodyFitter") as mock_smpl:
            from fitscan.application.measurement_service import create_pipeline
            with patch("fitscan.application.measurement_service.PipelineOrchestrator"):
                create_pipeline(enable_smpl=False, enable_claude=False)
            mock_smpl.assert_called_once()
