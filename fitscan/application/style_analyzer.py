"""StyleAnalyzer — Anthropic Claude wrapper for style profile classification.

Mirrors the pattern of `fitscan.validation.claude_validator`: same consent
gate (no `subject_id` → no API call → fallback), same metadata forwarding
for Anthropic retention controls, same defensive JSON parsing.

Public contract:
    profile = StyleAnalyzer(api_key).analyze(request)
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from loguru import logger

from fitscan.domain.style_models import StyleProfileRequest, StyleProfileResponse


@lru_cache(maxsize=4)
def _anthropic_client(api_key: str) -> Any:
    """One Anthropic client per key, mirroring claude_validator. Constructing
    a new client per call tears down the httpx connection pool every time."""
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


STYLE_PROMPT = """\
You are a fashion stylist for a Ghana / West-Africa marketplace called Nidlo.
You help clients understand their body shape and discover flattering cuts,
fabrics, and silhouettes.

A client has shared:
- Body measurements (millimetres, sectioned by body region)
- Their stated fashion interests (drawn from their onboarding)

Your job: produce a single STRUCTURED JSON response describing their style
profile. Do not include prose outside the JSON. Do not wrap in markdown.

Body measurements (mm): {measurements}

Fashion interests: {interests}

Available specializations on the platform (recommend by EXACT slug):
{specializations}

Respond with ONLY this JSON shape:
{{
  "body_shape": "<rectangle | hourglass | pear | inverted_triangle | oval>",
  "body_shape_confidence": 0.85,
  "flattering_silhouettes": ["a-line dress", "wrap top", "high-waist trousers"],
  "color_palette": ["#1a1612", "#d68a4f", "#fbf7f0", "#5a3a2c", "#b8860b"],
  "fabric_recommendations": ["ankara", "linen", "silk"],
  "recommended_specializations": ["kaba_and_slit"],
  "summary": "Short one-paragraph summary of the style profile, written for the client."
}}

Rules:
- body_shape: pick exactly one of the five values
- body_shape_confidence: float 0-1; lower it if measurements are sparse
- color_palette: 4-8 hex strings; bias toward earthy / warm tones
  for African textiles
- fabric_recommendations: 3-5 lowercase fabric names common to West African fashion
- flattering_silhouettes: 3-5 short descriptors
- recommended_specializations: 1-3 slugs from the list above
  (or empty array if none fit)
- summary: 2-4 sentences, second-person ("Your shape...")
"""


class StyleAnalyzer:
    """Calls Claude (Sonnet 4.6) to produce a StyleProfileResponse.

    The body photo is NOT sent for v1 — measurements + interests carry
    enough signal for an initial style profile. Photo-aware refinement is
    a Phase 2 follow-up that would also require explicit consent under
    the existing AnthropicConsentDialog.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def analyze(self, request: StyleProfileRequest) -> StyleProfileResponse:
        # Consent + config gate. Without a subject_id we don't ship to
        # Anthropic. Without an API key we can't call at all. Both fall
        # back to a deterministic stub so the caller never crashes.
        if not self._api_key or self._api_key == "test_key_placeholder":
            logger.info("style: anthropic api key absent — returning fallback")
            return self._fallback(request)
        if not request.subject_id:
            logger.info("style: no subject_id consent token — returning fallback")
            return self._fallback(request)

        try:
            client = _anthropic_client(self._api_key)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                metadata={"user_id": request.subject_id},
                messages=[
                    {
                        "role": "user",
                        "content": STYLE_PROMPT.format(
                            measurements=json.dumps(request.measurements_mm),
                            interests=json.dumps(request.fashion_interests),
                            specializations=json.dumps(
                                request.available_specializations
                            ),
                        ),
                    }
                ],
            )
            raw = response.content[0].text
            parsed = json.loads(raw)
            return StyleProfileResponse(
                body_shape=parsed["body_shape"],
                body_shape_confidence=float(parsed.get("body_shape_confidence", 0.7)),
                flattering_silhouettes=parsed["flattering_silhouettes"],
                color_palette=parsed["color_palette"],
                fabric_recommendations=parsed["fabric_recommendations"],
                recommended_specializations=parsed.get(
                    "recommended_specializations", []
                ),
                summary=parsed["summary"],
                from_fallback=False,
            )
        except json.JSONDecodeError:
            logger.warning("style: claude returned non-JSON — fallback")
            return self._fallback(request)
        except Exception as e:  # noqa: BLE001 — defensive: never propagate
            logger.warning(f"style: claude call failed — fallback: {e}")
            return self._fallback(request)

    def _fallback(self, request: StyleProfileRequest) -> StyleProfileResponse:
        """Deterministic stub for no-consent / no-key / Claude-failure paths.

        Returns a "rectangle" default with brand colors + interest-aware
        specialization picks. Frontend should treat `from_fallback=True`
        as "ask the user for consent / try again later".
        """
        # Pick up to 3 specializations the user is interested in.
        interests = set(request.fashion_interests)
        recommended = [
            slug for slug in request.available_specializations if slug in interests
        ][:3]

        return StyleProfileResponse(
            body_shape="rectangle",
            body_shape_confidence=0.5,
            flattering_silhouettes=[
                "tailored a-line",
                "wrap dress",
                "structured blazer",
            ],
            color_palette=["#1a1612", "#d68a4f", "#fbf7f0", "#5a3a2c"],
            fabric_recommendations=["ankara", "cotton", "linen"],
            recommended_specializations=recommended,
            summary=(
                "We couldn't generate a personalised style profile yet. "
                "Once you grant photo consent and save a measurement, "
                "we'll tailor recommendations to your specific shape."
            ),
            from_fallback=True,
        )
