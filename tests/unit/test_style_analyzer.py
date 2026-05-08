"""Tests for the Anthropic style classifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fitscan.application.style_analyzer import StyleAnalyzer
from fitscan.domain.style_models import StyleProfileRequest


def _request(**overrides) -> StyleProfileRequest:
    """Build a minimal StyleProfileRequest with sensible defaults."""
    base = {
        "subject_id": "user-abc",
        "measurements_mm": {"vertical": {"full_height": 1700.0}},
        "fashion_interests": ["wedding", "kaba_and_slit"],
        "available_specializations": ["kaba_and_slit", "wedding_dress", "agbada"],
    }
    base.update(overrides)
    return StyleProfileRequest(**base)


class TestStyleAnalyzerFallback:
    def test_no_api_key_returns_fallback(self) -> None:
        analyzer = StyleAnalyzer(api_key=None)
        result = analyzer.analyze(_request())
        assert result.from_fallback is True
        assert result.body_shape == "rectangle"

    def test_test_key_placeholder_returns_fallback(self) -> None:
        analyzer = StyleAnalyzer(api_key="test_key_placeholder")
        result = analyzer.analyze(_request())
        assert result.from_fallback is True

    def test_no_subject_id_returns_fallback(self) -> None:
        analyzer = StyleAnalyzer(api_key="real-key")
        result = analyzer.analyze(_request(subject_id=None))
        assert result.from_fallback is True

    def test_fallback_picks_specializations_from_interests(self) -> None:
        """The fallback path still tries to be useful: if the user's interests
        overlap with the platform's specializations, recommend those."""
        analyzer = StyleAnalyzer(api_key=None)
        result = analyzer.analyze(_request())
        # "kaba_and_slit" appears in both interests + available_specializations
        assert "kaba_and_slit" in result.recommended_specializations


class TestStyleAnalyzerHappyPath:
    @patch("fitscan.application.style_analyzer._anthropic_client")
    def test_parses_claude_json_response(self, mock_client_factory) -> None:
        # Stub Claude's response shape
        fake_response = MagicMock()
        fake_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "body_shape": "hourglass",
                        "body_shape_confidence": 0.82,
                        "flattering_silhouettes": [
                            "wrap dress",
                            "fitted bodice",
                            "high-waist skirt",
                        ],
                        "color_palette": [
                            "#1a1612",
                            "#d68a4f",
                            "#fbf7f0",
                            "#5a3a2c",
                        ],
                        "fabric_recommendations": ["silk", "ankara", "linen"],
                        "recommended_specializations": ["kaba_and_slit"],
                        "summary": (
                            "Your hourglass shape suits structured silhouettes "
                            "that follow your natural waist."
                        ),
                    }
                )
            )
        ]
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response
        mock_client_factory.return_value = fake_client

        analyzer = StyleAnalyzer(api_key="real-key")
        result = analyzer.analyze(_request())

        assert result.from_fallback is False
        assert result.body_shape == "hourglass"
        assert result.body_shape_confidence == 0.82
        assert "kaba_and_slit" in result.recommended_specializations

        # Verify the consent token was forwarded as Anthropic metadata
        call_args = fake_client.messages.create.call_args
        assert call_args.kwargs["metadata"]["user_id"] == "user-abc"

    @patch("fitscan.application.style_analyzer._anthropic_client")
    def test_invalid_json_falls_back(self, mock_client_factory) -> None:
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="not valid json")]
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_response
        mock_client_factory.return_value = fake_client

        analyzer = StyleAnalyzer(api_key="real-key")
        result = analyzer.analyze(_request())

        assert result.from_fallback is True

    @patch("fitscan.application.style_analyzer._anthropic_client")
    def test_anthropic_exception_falls_back(self, mock_client_factory) -> None:
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = RuntimeError("api down")
        mock_client_factory.return_value = fake_client

        analyzer = StyleAnalyzer(api_key="real-key")
        result = analyzer.analyze(_request())

        assert result.from_fallback is True
