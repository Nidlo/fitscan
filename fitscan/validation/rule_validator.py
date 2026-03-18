"""
Anatomical rule-based sanity checks. No ML, no API calls.
Catches physically impossible measurements before they reach the user.
"""

from __future__ import annotations

import numpy as np

from fitscan.domain.models import BodyMeasurements, ValidationResult

MEASUREMENT_BOUNDS_CM = {
    "bust_cm": (65, 165),
    "chest_cm": (65, 165),
    "waist_cm": (55, 160),
    "hip_cm": (65, 175),
    "under_breast_waist_cm": (58, 145),
    "around_arm_cm": (22, 65),
    "around_arm_3_4_cm": (20, 58),
    "wrist_cm": (13, 30),
    "neck_cm": (28, 55),
    "thigh_cm": (35, 90),
    "shoulder_cm": (30, 60),
    "across_back_cm": (25, 55),
    "across_chest_cm": (24, 52),
    "shoulder_to_nipple_cm": (18, 40),
    "nipple_to_nipple_cm": (12, 28),
    "shoulder_to_underbust_cm": (22, 42),
    "shoulder_to_waist_cm": (30, 52),
    "shoulder_to_hip_cm": (48, 80),
    "nape_to_waist_cm": (32, 55),
    "hip_depth_cm": (14, 32),
    "height_cm": (80, 230),
    "inseam_cm": (40, 110),
    "sleeve_length_cm": (40, 90),
    "blouse_length_cm": (40, 90),
    "kaba_length_cm": (70, 140),
    "skirt_length_cm": (35, 120),
    "dress_length_cm": (80, 200),
    "slit_length_cm": (20, 80),
    "cape_cm": (30, 100),
}

RELATIONAL_RULES = [
    ("hip_cm", "waist_cm", 0.85, 1.80, "hip >= waist"),
    ("bust_cm", "neck_cm", 2.0, 5.5, "bust 2-5.5x neck"),
    ("chest_cm", "neck_cm", 2.0, 5.5, "chest 2-5.5x neck"),
    ("height_cm", "inseam_cm", 1.5, 2.5, "height 1.5-2.5x inseam"),
    ("shoulder_cm", "neck_cm", 0.8, 2.0, "shoulders wider than neck"),
    (
        "shoulder_to_hip_cm",
        "shoulder_to_waist_cm",
        1.2,
        2.0,
        "hip lower than waist",
    ),
    ("around_arm_cm", "wrist_cm", 1.3, 3.5, "upper arm > wrist"),
    (
        "bust_cm",
        "under_breast_waist_cm",
        1.02,
        1.50,
        "bust > underbust",
    ),
    (
        "nipple_to_nipple_cm",
        "shoulder_cm",
        0.25,
        0.70,
        "nipple span < shoulder width",
    ),
]


class RuleValidator:
    """Applies anatomical bounds and relational rules to measurements."""

    def validate(
        self, image: np.ndarray, measurements: BodyMeasurements
    ) -> ValidationResult:
        flagged: list[str] = []
        issues: list[str] = []

        # Check absolute bounds
        for field, (lo, hi) in MEASUREMENT_BOUNDS_CM.items():
            m = getattr(measurements, field, None)
            if m is None:
                continue
            if not (lo <= m.value_cm <= hi):
                flagged.append(field)
                issues.append(f"{field}={m.value_cm}cm outside range [{lo}, {hi}]")

        # Check relational rules
        for fa, fb, min_r, max_r, desc in RELATIONAL_RULES:
            ma = getattr(measurements, fa, None)
            mb = getattr(measurements, fb, None)
            if ma and mb and mb.value_cm > 0:
                ratio = ma.value_cm / mb.value_cm
                if not (min_r <= ratio <= max_r):
                    flagged.append(fa)
                    issues.append(
                        f"Relational rule violated: {desc} " f"(ratio={ratio:.2f})"
                    )

        quality_score = max(0.0, 10.0 - len(flagged) * 2.0)
        return ValidationResult(
            is_valid=len(flagged) == 0,
            flagged_fields=flagged,
            photo_quality_score=quality_score,
            quality_issues=issues,
            recommendation=("proceed" if len(flagged) <= 1 else "retake"),
            raw_response="rule_based",
        )
