"""Tests for Claude validator."""
import numpy as np

from fitscan.domain.models import BodyMeasurements, SingleMeasurement
from fitscan.validation.claude_validator import ClaudeValidator


class TestClaudeValidator:
    def test_returns_default_with_test_key(self):
        validator = ClaudeValidator(api_key="test_key_placeholder")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = BodyMeasurements(source="test")
        result = validator.validate(image, m)
        assert result.is_valid
        assert result.recommendation == "proceed"
        assert result.photo_quality_score == 7.0

    def test_returns_default_with_no_key(self):
        validator = ClaudeValidator(api_key=None)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = BodyMeasurements(source="test")
        result = validator.validate(image, m)
        assert result.is_valid

    def test_returns_default_with_empty_key(self):
        validator = ClaudeValidator(api_key="")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = BodyMeasurements(source="test")
        result = validator.validate(image, m)
        assert result.is_valid
