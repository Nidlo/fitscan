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


class TestClaudeValidatorConsentGate:
    """Audit FS-C3: body photos are sensitive PII. The validator must
    refuse to ship a photo to Anthropic without an explicit per-user
    `subject_id` consent token, AND must forward `metadata.user_id` so
    Anthropic-side retention controls can target the request."""

    def test_skips_when_subject_id_is_none_even_with_real_key(self):
        """A real-looking key alone is NOT consent. Without subject_id we
        must not call Anthropic at all — the validator returns the default
        pass result instead."""
        validator = ClaudeValidator(api_key="YOUR_FAKE_ANT_KEY_for_consent_gate_tests")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = BodyMeasurements(source="test")
        # subject_id omitted intentionally
        result = validator.validate(image, m)
        # Default-pass result means no API call was made
        assert result.is_valid
        assert result.recommendation == "proceed"
        assert result.photo_quality_score == 7.0
        assert result.raw_response == ""

    def test_skips_when_subject_id_is_empty_string(self):
        validator = ClaudeValidator(api_key="YOUR_FAKE_ANT_KEY_for_consent_gate_tests")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = BodyMeasurements(source="test")
        result = validator.validate(image, m, subject_id="")
        assert result.raw_response == ""

    def test_calls_anthropic_with_metadata_user_id_when_consent_given(
        self, monkeypatch
    ):
        """When subject_id IS supplied, the validator must (a) call
        Anthropic and (b) forward subject_id as metadata.user_id so the
        retention controls work."""
        captured: dict = {}

        class _FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)

                class _FakeContent:
                    def __init__(self):
                        self.text = (
                            '{"is_valid": true, "flagged_measurements": [],'
                            ' "photo_quality_score": 9.0, "quality_issues": [],'
                            ' "recommendation": "proceed", "notes": "ok"}'
                        )

                class _FakeResponse:
                    content = [_FakeContent()]

                return _FakeResponse()

        class _FakeAnthropic:
            def __init__(self, api_key=None):
                self.messages = _FakeMessages()

        # The validator imports `Anthropic` lazily; patch the module-level name
        import anthropic

        monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

        validator = ClaudeValidator(api_key="YOUR_FAKE_ANT_KEY_for_consent_gate_tests")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        m = BodyMeasurements(source="test")
        result = validator.validate(image, m, subject_id="user-abc-123")

        # Anthropic was called …
        assert "metadata" in captured
        # … and the consent token was forwarded for retention control
        assert captured["metadata"] == {"user_id": "user-abc-123"}
        # Real response parsed correctly
        assert result.raw_response != ""
        assert result.photo_quality_score == 9.0
