"""Tests for API schemas."""
from fitscan.api.schemas import MeasurementsResponse, SingleMeasurementOut


class TestMeasurementsResponse:
    def test_create_with_measurements(self):
        resp = MeasurementsResponse(
            status="success",
            gender_profile="ladies",
            bust_cm=SingleMeasurementOut(
                value_cm=90.0, confidence=0.85, source="fused"
            ),
        )
        assert resp.status == "success"
        assert resp.bust_cm is not None
        assert resp.bust_cm.value_cm == 90.0

    def test_create_minimal(self):
        resp = MeasurementsResponse(status="error", error="test error")
        assert resp.bust_cm is None
        assert resp.error == "test error"
