"""Pydantic schemas for the Anthropic-powered style classifier (v1).

The classifier consumes a user's body measurements + their stated fashion
interests and returns a brand-aligned style profile: body shape, flattering
silhouettes, color palette, fabric recommendations, and 1-3 specialization
names matched against the platform's existing specializations table.

This is the BOUNDARY contract — what FastAPI accepts/returns. The Claude
prompt JSON shape is deliberately a subset of these fields.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BodyShape = Literal[
    "rectangle",
    "hourglass",
    "pear",
    "inverted_triangle",
    "oval",
]

DisplayUnit = Literal["cm", "inches"]


class StyleProfileRequest(BaseModel):
    """Inbound request from the Laravel backend."""

    # Consent gating — same shape as the measurement endpoints. Without a
    # subject_id we don't ship to Anthropic and return a stub response.
    subject_id: str | None = Field(
        default=None,
        description="Per-user consent token; required to call Claude.",
    )

    # Body data — already extracted by the measurement pipeline. Kept loose
    # (dict) so the backend can pass through whatever payload it has cached.
    measurements_mm: dict[str, dict[str, float | None]] = Field(
        default_factory=dict,
        description="Section-keyed measurements in mm (matches the existing"
        " measurement payload shape).",
    )

    # Self-reported preferences from client onboarding (e.g. ["wedding",
    # "kaba_and_slit", "modern_traditional"]).
    fashion_interests: list[str] = Field(
        default_factory=list,
        description="User's selected fashion interests.",
    )

    # The canonical specialization slug list, sent so Claude can recommend
    # specializations that exist in the platform rather than free-form ones.
    available_specializations: list[str] = Field(
        default_factory=list,
        description="Slugs of specializations the platform supports.",
    )

    # Whether the summary's prose should refer to measurements in cm or
    # inches. The body data itself is always sent in mm — this only
    # controls how Claude writes the natural-language paragraph. Defaults
    # to cm to preserve the prior behaviour.
    display_unit: DisplayUnit = Field(
        default="cm",
        description="Unit Claude should use when citing measurements in the summary.",
    )


class StyleProfileResponse(BaseModel):
    """Returned to the backend; backend persists this as users.style_profile."""

    body_shape: BodyShape
    body_shape_confidence: float = Field(ge=0, le=1)

    flattering_silhouettes: list[str] = Field(
        description="3-5 short silhouette descriptors, e.g. 'a-line dress'.",
        min_length=1,
        max_length=8,
    )
    color_palette: list[str] = Field(
        description="Hex strings (e.g. '#1a1612'). 4-8 colors.",
        min_length=1,
        max_length=10,
    )
    fabric_recommendations: list[str] = Field(
        description="3-5 fabric names, e.g. 'linen', 'ankara', 'silk'.",
        min_length=1,
        max_length=8,
    )
    recommended_specializations: list[str] = Field(
        description="1-3 slugs matched against available_specializations.",
        min_length=0,
        max_length=5,
    )
    summary: str = Field(
        description="One-paragraph natural language style summary.",
        min_length=20,
        # 1500 ≈ 4–5 sentences of typical English prose. Claude's
        # second-person summaries with a measurements-grounded body-shape
        # description often run 700–1200 chars; the old 600 cap rejected
        # real classifications and forced the deterministic fallback,
        # losing the actual answer the user paid for.
        max_length=1500,
    )

    # Indicates the response came from the rule-based / stub path
    # (no Anthropic call). Frontend can show a "consent for AI" prompt if true.
    from_fallback: bool = False
