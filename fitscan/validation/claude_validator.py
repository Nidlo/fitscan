"""
Uses Claude Vision API to validate measurements and photo quality.
Implements IValidationProvider.
"""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
from loguru import logger

from fitscan.domain.models import BodyMeasurements, ValidationResult

VALIDATION_PROMPT = """
You are a body measurement validation assistant for a fashion tech system.
A computer vision pipeline has extracted these body measurements from the photo:

{measurements}

Analyse the photo and respond with ONLY valid JSON (no prose, no markdown):
{{
  "is_valid": true,
  "flagged_measurements": [],
  "photo_quality_score": 8.5,
  "quality_issues": [],
  "recommendation": "proceed",
  "notes": "Brief assessment"
}}

Rules:
- flagged_measurements: list any measurement names that seem implausible
- photo_quality_score: 0-10
- quality_issues: list from ["loose_clothing", "partial_body", "bad_lighting",
  "motion_blur", "low_resolution", "obscured_limbs", "bad_angle"]
- recommendation: "proceed" if quality >= 6, else "retake"
"""


class ClaudeValidator:
    """Validates measurements using Claude Vision API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def validate(
        self, image: np.ndarray, measurements: BodyMeasurements
    ) -> ValidationResult:
        if not self._api_key or self._api_key == "test_key_placeholder":
            return self._default_result()

        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self._api_key)
            img_b64 = self._encode_image(image)
            measurements_str = json.dumps(measurements.to_dict(), indent=2)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": VALIDATION_PROMPT.format(
                                    measurements=measurements_str
                                ),
                            },
                        ],
                    }
                ],
            )
            raw = response.content[0].text
            parsed = json.loads(raw)
            return ValidationResult(
                is_valid=parsed.get("is_valid", True),
                flagged_fields=parsed.get("flagged_measurements", []),
                photo_quality_score=float(parsed.get("photo_quality_score", 7.0)),
                quality_issues=parsed.get("quality_issues", []),
                recommendation=parsed.get("recommendation", "proceed"),
                raw_response=raw,
            )
        except json.JSONDecodeError:
            logger.warning("Claude returned non-JSON. Defaulting to proceed.")
            return self._default_result()
        except Exception as e:
            logger.warning(f"Claude validation failed: {e}")
            return self._default_result()

    def _encode_image(self, image: np.ndarray) -> str:
        rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode()

    def _default_result(self) -> ValidationResult:
        return ValidationResult(
            is_valid=True,
            flagged_fields=[],
            photo_quality_score=7.0,
            quality_issues=[],
            recommendation="proceed",
            raw_response="",
        )
