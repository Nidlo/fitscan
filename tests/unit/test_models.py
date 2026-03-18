"""Tests for domain models."""
from fitscan.domain.models import BodyMeasurements, SingleMeasurement


class TestBodyMeasurements:
    def test_to_dict_includes_measurements(self):
        m = BodyMeasurements(
            chest_cm=SingleMeasurement(95.0, 0.8, "smpl"),
            waist_cm=SingleMeasurement(75.0, 0.85, "smpl"),
            gender_profile="men",
            source="smpl",
        )
        d = m.to_dict()
        assert d["gender_profile"] == "men"
        assert d["chest_cm"]["value_cm"] == 95.0
        assert d["waist_cm"]["confidence"] == 0.85

    def test_to_dict_skips_none_fields(self):
        m = BodyMeasurements(source="test")
        d = m.to_dict()
        assert "bust_cm" not in d
        assert d["source"] == "test"

    def test_applicable_fields_ladies(self):
        m = BodyMeasurements(gender_profile="ladies")
        fields = m.applicable_fields()
        assert "bust_cm" in fields
        assert "chest_cm" not in fields

    def test_applicable_fields_men(self):
        m = BodyMeasurements(gender_profile="men")
        fields = m.applicable_fields()
        assert "chest_cm" in fields
        assert "bust_cm" not in fields
        assert "kaba_length_cm" not in fields

    def test_applicable_fields_children(self):
        m = BodyMeasurements(gender_profile="children")
        fields = m.applicable_fields()
        assert "height_cm" in fields
        assert "bust_cm" not in fields
        assert "chest_cm" not in fields


class TestSingleMeasurement:
    def test_frozen(self):
        m = SingleMeasurement(95.0, 0.8, "smpl")
        assert m.value_cm == 95.0
        assert m.confidence == 0.8
        assert m.source == "smpl"
