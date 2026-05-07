"""
Uses Claude Vision API to validate measurements and photo quality.
Implements IValidationProvider.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from loguru import logger


@lru_cache(maxsize=4)
def _anthropic_client(api_key: str) -> Any:
    """Memoize one Anthropic client per API key (FS-NIDLO-OPS-05).

    Constructing `Anthropic(...)` per call (the previous shape) tore
    down + rebuilt the underlying httpx connection pool every time,
    measurably increasing tail latency under load. A module-level
    `lru_cache` keyed on the API key keeps a stable client per key
    (we only ever have one in practice; the cache size is room for
    test fixtures that mint per-test keys).

    NOTE — prompt caching: marking the static
    `VALIDATION_PROMPT` / `LANDMARK_VERIFICATION_PROMPT` blocks with
    `cache_control={"type": "ephemeral"}` was considered. It does not
    help here: every call carries a unique user photo as the first
    content block, which invalidates the cache boundary regardless
    of what we mark on the text block. Revisit if we ever pre-bind
    the rules into a `system=` parameter (Anthropic caches system
    blocks independently).
    """
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)

from fitscan.domain.models import (
    BodyMeasurements,
    LandmarkVerdict,
    LandmarkVerification,
    ValidationResult,
)

LANDMARK_VERIFICATION_PROMPT = """
You are a body-landmark verification assistant for a fashion-tech CV pipeline.
The pipeline detected these landmarks at low visibility — clothing folds,
shadows, hair, or pose may be hiding them. Look at the photo and place each
one as best you can.

Landmarks: {landmarks}

Respond with ONLY valid JSON:
{{
  "verifications": {{
    "<landmark_name>": {{
      "verdict": "<visible_at_xy | pose_estimate | occluded_unrecoverable>",
      "x": <float 0-1 or null>,
      "y": <float 0-1 or null>
    }}
  }}
}}

Verdict rules:
- visible_at_xy: you can SEE the landmark; (x, y) are normalised image coords
- pose_estimate: you cannot see it directly but can infer from anatomy +
  surrounding visible joints. (x, y) is your best guess.
- occluded_unrecoverable: even with anatomy you can't place it. x, y null.
"""

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
    """Validates measurements using Claude Vision API.

    The body photo is sensitive PII. Shipping it to Anthropic only happens
    when the caller passes an explicit consent token (`subject_id`). Without
    one, validation is skipped and a default-pass result is returned. The
    `subject_id` is also forwarded as `metadata.user_id` so Anthropic
    retention controls can target it.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def validate(
        self,
        image: np.ndarray,
        measurements: BodyMeasurements,
        subject_id: str | None = None,
    ) -> ValidationResult:
        if not self._api_key or self._api_key == "test_key_placeholder":
            return self._default_result()
        if not subject_id:
            logger.info(
                "claude validation skipped: no subject_id consent token provided"
            )
            return self._default_result()

        try:
            client = _anthropic_client(self._api_key)
            img_b64 = self._encode_image(image)
            measurements_str = json.dumps(measurements.to_dict(), indent=2)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                metadata={"user_id": subject_id},
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

    def verify_landmarks(
        self,
        image: np.ndarray,
        occluded: list[str],
        subject_id: str | None = None,
    ) -> LandmarkVerification:
        """S5 — Claude vision verifies CV landmarks the MediaPipe pose
        detector flagged as low-visibility. Returns per-landmark
        verdicts with optional Claude-suggested coordinates.

        Same consent gate as `validate()`: without an `api_key` AND a
        `subject_id` the photo is never shipped to Anthropic and an
        empty verification is returned (no verdicts, no API call).
        """
        if not occluded:
            return LandmarkVerification(verdicts={}, raw_response="")
        if not self._api_key or self._api_key == "test_key_placeholder":
            return LandmarkVerification(verdicts={}, raw_response="")
        if not subject_id:
            logger.info(
                "claude landmark verification skipped: no subject_id consent token"
            )
            return LandmarkVerification(verdicts={}, raw_response="")

        try:
            client = _anthropic_client(self._api_key)
            img_b64 = self._encode_image(image)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                metadata={"user_id": subject_id},
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
                                "text": LANDMARK_VERIFICATION_PROMPT.format(
                                    landmarks=", ".join(occluded),
                                ),
                            },
                        ],
                    }
                ],
            )
            raw = response.content[0].text
            parsed = json.loads(raw)
            verdicts: dict[str, LandmarkVerdict] = {}
            for name, payload in (parsed.get("verifications") or {}).items():
                if not isinstance(payload, dict):
                    continue
                verdict = str(payload.get("verdict", "occluded_unrecoverable"))
                if verdict not in {
                    "visible_at_xy",
                    "pose_estimate",
                    "occluded_unrecoverable",
                }:
                    continue
                x = payload.get("x")
                y = payload.get("y")
                verdicts[name] = LandmarkVerdict(
                    landmark=name,
                    verdict=verdict,
                    x=float(x) if isinstance(x, (int, float)) else None,
                    y=float(y) if isinstance(y, (int, float)) else None,
                )
            return LandmarkVerification(verdicts=verdicts, raw_response=raw)
        except json.JSONDecodeError:
            logger.warning(
                "Claude landmark verification returned non-JSON; skipping."
            )
            return LandmarkVerification(verdicts={}, raw_response="")
        except Exception as e:  # noqa: BLE001 — defence in depth
            logger.warning(f"Claude landmark verification failed: {e}")
            return LandmarkVerification(verdicts={}, raw_response="")

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
