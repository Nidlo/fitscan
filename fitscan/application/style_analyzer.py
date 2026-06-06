"""StyleAnalyzer, Anthropic Claude wrapper for style profile classification.

Mirrors the pattern of `fitscan.validation.claude_validator`: same consent
gate (no `subject_id` → no API call → fallback), same metadata forwarding
for Anthropic retention controls, same defensive JSON parsing.

Public contract:
    profile = StyleAnalyzer(api_key).analyze(request)
"""

from __future__ import annotations

import json

from loguru import logger

from fitscan.domain.style_models import StyleProfileRequest, StyleProfileResponse
from fitscan.llm.anthropic_api_provider import AnthropicApiProvider
from fitscan.llm.provider import LLMProvider

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

The client prefers measurements expressed in **{display_unit}**. When the
summary refers to any measurement value, write it in {display_unit} (converted
from the mm payload, rounded to the nearest whole {display_unit_unit}).
Do NOT mix units inside the summary. NEVER cite mm.

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
- summary: 2-4 sentences, second-person ("Your shape...");
  aim for 400-1000 characters total, hard ceiling 1400.
  Never wrap in quotes.
  Use {display_unit} for any numeric measurement you mention.
"""

# Defensive ceiling matched to StyleProfileResponse.summary's max_length.
# Claude usually honours the prompt's 1000-char target; this is the safety
# net for the rare overrun so we never throw away a real classification on
# a 50-char overshoot.
_SUMMARY_HARD_CAP = 1500


class StyleAnalyzer:
    """Calls Claude (Sonnet 4.6) to produce a StyleProfileResponse.

    The body photo is NOT sent for v1, measurements + interests carry
    enough signal for an initial style profile. Photo-aware refinement is
    a Phase 2 follow-up that would also require explicit consent under
    the existing AnthropicConsentDialog.
    """

    def __init__(
        self,
        api_key: str | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        """:param api_key: Anthropic API key; builds the default direct-API
            provider when `llm` is not supplied (back-compat).
        :param llm: explicit provider. When given it drives the call and its
            `is_enabled` replaces the old api-key gate.
        """
        self._llm: LLMProvider = llm or AnthropicApiProvider(api_key)

    def analyze(self, request: StyleProfileRequest) -> StyleProfileResponse:
        # Consent + config gate. Without a subject_id we don't ship to
        # Anthropic. Without an enabled provider we can't call at all. Both
        # fall back to a deterministic stub so the caller never crashes.
        if not self._llm.is_enabled:
            logger.info("style: llm provider not enabled, returning fallback")
            return self._fallback(request)
        if not request.subject_id:
            logger.info("style: no subject_id consent token, returning fallback")
            return self._fallback(request)

        # Spell out the unit name so the prompt reads naturally
        # ("rounded to the nearest whole centimetre") rather than
        # leaking the API-level slug into Claude's prose.
        unit_name = "inch" if request.display_unit == "inches" else "centimetre"

        try:
            raw = self._llm.complete(
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
                            display_unit=request.display_unit,
                            display_unit_unit=unit_name,
                        ),
                    }
                ],
            )
            parsed = json.loads(raw)
            summary = str(parsed["summary"])
            if len(summary) > _SUMMARY_HARD_CAP:
                # Claude very occasionally over-runs the prompt's char
                # target. Truncate at the last sentence boundary that
                # still fits so the user sees a coherent answer rather
                # than a deterministic fallback.
                truncated = summary[:_SUMMARY_HARD_CAP]
                last_period = truncated.rfind(". ")
                summary = truncated[: last_period + 1] if last_period > 0 else truncated
                logger.info(
                    f"style: summary trimmed from {len(parsed['summary'])} "
                    f"to {len(summary)} chars"
                )
            return StyleProfileResponse(
                body_shape=parsed["body_shape"],
                body_shape_confidence=float(parsed.get("body_shape_confidence", 0.7)),
                flattering_silhouettes=parsed["flattering_silhouettes"],
                color_palette=parsed["color_palette"],
                fabric_recommendations=parsed["fabric_recommendations"],
                recommended_specializations=parsed.get(
                    "recommended_specializations", []
                ),
                summary=summary,
                from_fallback=False,
            )
        except json.JSONDecodeError:
            logger.warning("style: claude returned non-JSON, fallback")
            return self._fallback(request)
        except Exception as e:  # noqa: BLE001, defensive: never propagate
            logger.warning(f"style: claude call failed, fallback: {e}")
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
