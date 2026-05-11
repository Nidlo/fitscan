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


class TestRuleValidatorBoundaries:
    """FS-NIDLO-TEST-06 — boundary tests for absolute MEASUREMENT_BOUNDS_CM
    and the RELATIONAL_RULES ratios. Pins the inclusive/exclusive nature
    of every check so a future refactor can't quietly tighten or loosen
    bounds without flipping these red.
    """

    validator = RuleValidator()
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    def _is_flagged(self, field: str, value: float) -> bool:
        m = BodyMeasurements(
            **{field: SingleMeasurement(value, 0.9, "smpl")},
            source="smpl",
        )
        result = self.validator.validate(self.dummy_image, m)
        return field in result.flagged_fields

    def test_height_at_lower_bound_passes(self):
        # MEASUREMENT_BOUNDS_CM["height_cm"] = (80, 230) — inclusive.
        assert not self._is_flagged("height_cm", 80.0)

    def test_height_just_below_lower_bound_flagged(self):
        assert self._is_flagged("height_cm", 79.9)

    def test_height_at_upper_bound_passes(self):
        assert not self._is_flagged("height_cm", 230.0)

    def test_height_just_above_upper_bound_flagged(self):
        assert self._is_flagged("height_cm", 230.1)

    def test_waist_at_lower_bound_passes(self):
        # MEASUREMENT_BOUNDS_CM["waist_cm"] = (55, 160) — inclusive.
        assert not self._is_flagged("waist_cm", 55.0)

    def test_waist_just_below_lower_bound_flagged(self):
        assert self._is_flagged("waist_cm", 54.9)

    def test_relational_hip_waist_at_lower_ratio_passes(self):
        # hip/waist >= 0.85 — exactly 0.85 should pass.
        m = BodyMeasurements(
            hip_cm=SingleMeasurement(85.0, 0.9, "smpl"),
            waist_cm=SingleMeasurement(100.0, 0.9, "smpl"),
            source="smpl",
        )
        result = self.validator.validate(self.dummy_image, m)
        assert "hip_cm" not in result.flagged_fields

    def test_relational_hip_waist_just_below_ratio_flagged(self):
        # 84/100 = 0.84 < 0.85 → flagged.
        m = BodyMeasurements(
            hip_cm=SingleMeasurement(84.0, 0.9, "smpl"),
            waist_cm=SingleMeasurement(100.0, 0.9, "smpl"),
            source="smpl",
        )
        result = self.validator.validate(self.dummy_image, m)
        assert "hip_cm" in result.flagged_fields

    def test_no_measurements_skips_relational_rules(self):
        # When neither side of a relational rule has a value, the rule
        # is skipped (not raised). Matches the `if ma and mb` guard.
        m = BodyMeasurements(source="smpl")
        result = self.validator.validate(self.dummy_image, m)
        assert result.is_valid
        assert result.flagged_fields == []

    def test_zero_denominator_does_not_zero_divide(self):
        # If a relational rule's denominator is 0, the rule is skipped.
        # Pinning so a future "simplification" doesn't turn this into a
        # ZeroDivisionError in production.
        m = BodyMeasurements(
            shoulder_cm=SingleMeasurement(40.0, 0.9, "smpl"),
            neck_cm=SingleMeasurement(0.0, 0.9, "smpl"),
            source="smpl",
        )
        result = self.validator.validate(self.dummy_image, m)
        # neck_cm at 0 fails the absolute bound (15-50) and gets flagged
        # for that reason; the relational rule itself didn't crash.
        assert "neck_cm" in result.flagged_fields
