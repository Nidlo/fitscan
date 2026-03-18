"""Tests for rule-based validation."""
import numpy as np

from fitscan.domain.models import BodyMeasurements, SingleMeasurement
from fitscan.validation.rule_validator import RuleValidator


class TestRuleValidator:
    validator = RuleValidator()
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_valid_measurements_pass(self):
        m = BodyMeasurements(
            bust_cm=SingleMeasurement(90.0, 0.9, "smpl"),
            waist_cm=SingleMeasurement(70.0, 0.9, "smpl"),
            hip_cm=SingleMeasurement(95.0, 0.9, "smpl"),
            height_cm=SingleMeasurement(165.0, 0.9, "smpl"),
            neck_cm=SingleMeasurement(35.0, 0.9, "smpl"),
            shoulder_cm=SingleMeasurement(42.0, 0.9, "smpl"),
            inseam_cm=SingleMeasurement(78.0, 0.9, "smpl"),
            around_arm_cm=SingleMeasurement(30.0, 0.9, "smpl"),
            wrist_cm=SingleMeasurement(16.0, 0.9, "smpl"),
            under_breast_waist_cm=SingleMeasurement(80.0, 0.9, "smpl"),
            shoulder_to_nipple_cm=SingleMeasurement(25.0, 0.9, "smpl"),
            nipple_to_nipple_cm=SingleMeasurement(20.0, 0.9, "smpl"),
            shoulder_to_waist_cm=SingleMeasurement(40.0, 0.9, "smpl"),
            shoulder_to_hip_cm=SingleMeasurement(55.0, 0.9, "smpl"),
            source="smpl",
        )
        result = self.validator.validate(self.dummy_image, m)
        assert result.is_valid

    def test_implausible_height_flagged(self):
        m = BodyMeasurements(
            height_cm=SingleMeasurement(300.0, 0.9, "smpl"),
            source="smpl",
        )
        result = self.validator.validate(self.dummy_image, m)
        assert not result.is_valid
        assert "height_cm" in result.flagged_fields

    def test_implausible_waist_flagged(self):
        m = BodyMeasurements(
            waist_cm=SingleMeasurement(200.0, 0.9, "smpl"),
            source="smpl",
        )
        result = self.validator.validate(self.dummy_image, m)
        assert "waist_cm" in result.flagged_fields

    def test_relational_rule_hip_waist(self):
        m = BodyMeasurements(
            hip_cm=SingleMeasurement(50.0, 0.9, "smpl"),
            waist_cm=SingleMeasurement(100.0, 0.9, "smpl"),
            source="smpl",
        )
        result = self.validator.validate(self.dummy_image, m)
        assert "hip_cm" in result.flagged_fields
