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

    def test_unit_defaults_to_cm(self):
        """Every response carries an explicit unit field so the Nidlo backend
        and other consumers never infer it from the `_cm` field suffixes.
        """
        resp = MeasurementsResponse(status="success")
        assert resp.unit == "cm"

    def test_unit_serialises_in_dump(self):
        resp = MeasurementsResponse(status="success")
        dumped = resp.model_dump()
        assert dumped["unit"] == "cm"

    def test_landmarks_field_defaults_to_none(self):
        resp = MeasurementsResponse(status="success")
        assert resp.landmarks is None

    def test_landmarks_serialises_when_set(self):
        # S2.5 — per-landmark normalized coords for the photo overlay
        resp = MeasurementsResponse(
            status="success",
            landmarks={
                "left_shoulder": {"x": 0.35, "y": 0.21, "visibility": 0.92},
                "right_shoulder": {"x": 0.65, "y": 0.21, "visibility": 0.91},
            },
        )
        dumped = resp.model_dump()
        assert dumped["landmarks"]["left_shoulder"]["x"] == 0.35
        assert dumped["landmarks"]["left_shoulder"]["visibility"] == 0.92
