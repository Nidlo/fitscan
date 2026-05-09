"""Tests for confidence scoring."""

from fitscan.domain.models import (
    BodyMeasurements,
    SingleMeasurement,
    ValidationResult,
)
from fitscan.fusion.confidence import ConfidenceScorer


class TestConfidenceScorer:
    scorer = ConfidenceScorer()

    def test_high_quality_preserves_confidence(self):
        m = BodyMeasurements(
            chest_cm=SingleMeasurement(95.0, 0.90, "fused"),
            source="fused",
        )
        validation = ValidationResult(
            is_valid=True,
            flagged_fields=[],
            photo_quality_score=9.0,
            quality_issues=[],
            recommendation="proceed",
            raw_response="",
        )
        result = self.scorer.score(m, validation, [m])
        assert result.chest_cm is not None
        assert result.chest_cm.confidence > 0.7

    def test_flagged_field_reduces_confidence(self):
        m = BodyMeasurements(
            chest_cm=SingleMeasurement(95.0, 0.90, "fused"),
            source="fused",
        )
        validation = ValidationResult(
            is_valid=False,
            flagged_fields=["chest_cm"],
            photo_quality_score=8.0,
            quality_issues=[],
            recommendation="proceed",
            raw_response="",
        )
        result = self.scorer.score(m, validation, [m])
        assert result.chest_cm is not None
        assert result.chest_cm.confidence < 0.90

    def test_low_quality_reduces_confidence(self):
        m = BodyMeasurements(
            chest_cm=SingleMeasurement(95.0, 0.90, "fused"),
            source="fused",
        )
        validation = ValidationResult(
            is_valid=True,
            flagged_fields=[],
            photo_quality_score=3.0,
            quality_issues=["bad_lighting"],
            recommendation="retake",
            raw_response="",
        )
        result = self.scorer.score(m, validation, [m])
        assert result.chest_cm is not None
        assert result.chest_cm.confidence < 0.50
