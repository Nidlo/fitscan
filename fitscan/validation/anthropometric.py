"""Anthropometric sanity bounds for measurement output.

Adult human bodies sit inside a fairly narrow proportion band. CAESAR and
ANSUR II (US Army Anthropometric Survey, public) provide the ground truth;
the bounds below cover ~99% of adult populations. Anything outside is
either a measurement error or a non-human body shape — either way, the
frontend should flag it for manual review rather than display it as gospel.

The pipeline appends `"anthropometric:<field>"` entries to
`ValidationResult.degraded_modes` for each violation. The frontend reads
these and renders a "low confidence — please verify" badge on the matching
field. We deliberately don't auto-correct: the user is the source of
truth, and silently clamping an AI estimate to a population mean would
hide the underlying model failure.

Sprint 34 Phase A3.
"""

from __future__ import annotations

from typing import Optional

from fitscan.domain.models import BodyMeasurements, SingleMeasurement

# Ratios are field-cm / height-cm. Bounds widened slightly past CAESAR's
# 99th-percentile envelope so genuine athletes / large-frame adults don't
# false-positive. Tighten in a follow-up sprint once we have a calibration
# dataset of confirmed-by-tape measurements.
RATIO_BOUNDS: dict[str, tuple[float, float]] = {
    # Circumferences
    "waist_cm": (0.35, 0.65),
    "hip_cm": (0.40, 0.68),
    "bust_cm": (0.40, 0.70),
    "chest_cm": (0.40, 0.70),
    "under_breast_waist_cm": (0.35, 0.65),
    "neck_cm": (0.16, 0.28),
    "thigh_cm": (0.25, 0.42),
    "around_arm_cm": (0.13, 0.30),
    "around_arm_3_4_cm": (0.12, 0.28),
    "wrist_cm": (0.07, 0.13),
    "knee_cm": (0.18, 0.30),
    "ankle_cm": (0.10, 0.18),
    # Widths
    "shoulder_cm": (0.20, 0.32),
    "across_back_cm": (0.18, 0.30),
    "across_chest_cm": (0.18, 0.30),
    # Vertical lengths (relative to height)
    "shoulder_to_waist_cm": (0.18, 0.30),
    "shoulder_to_hip_cm": (0.30, 0.45),
    "nape_to_waist_cm": (0.20, 0.32),
    "inseam_cm": (0.40, 0.52),
    "outseam_cm": (0.50, 0.65),
    "waist_to_knee_cm": (0.20, 0.35),
    "waist_to_floor_cm": (0.45, 0.62),
    "sleeve_length_cm": (0.28, 0.42),
}

# Pairwise ordering invariants that always hold for adult bodies regardless
# of shape. Violations indicate the mesh fit got the body shape wrong;
# Phase B's multi-view fitting should largely eliminate these.
PAIRWISE_INVARIANTS: list[tuple[str, str]] = [
    # (smaller, larger) — `smaller` must be < `larger`
    ("wrist_cm", "around_arm_cm"),  # forearm narrows toward wrist
    ("around_arm_cm", "waist_cm"),  # bicep < waist (universal for adults)
    ("ankle_cm", "knee_cm"),  # ankle narrows toward foot
    ("knee_cm", "thigh_cm"),  # knee < thigh
]


def _value(m: Optional[SingleMeasurement]) -> Optional[float]:
    return m.value_cm if m is not None else None


def check(measurements: BodyMeasurements) -> list[str]:
    """Return `degraded_modes` entries naming each field that violates an
    anthropometric bound. Empty when everything looks plausible.

    Entries are formatted as `"anthropometric:<field>"` — the frontend
    splits on `:` to discover which field to flag.
    """
    violations: list[str] = []

    height = _value(measurements.height_cm)
    if height is None or height < 50 or height > 250:
        # No usable height anchor — ratio bounds can't fire. Pairwise
        # invariants still apply.
        height = None

    if height is not None:
        for field, (low, high) in RATIO_BOUNDS.items():
            val = _value(getattr(measurements, field, None))
            if val is None:
                continue
            ratio = val / height
            if ratio < low or ratio > high:
                violations.append(f"anthropometric:{field}")

    for smaller_field, larger_field in PAIRWISE_INVARIANTS:
        smaller = _value(getattr(measurements, smaller_field, None))
        larger = _value(getattr(measurements, larger_field, None))
        if smaller is None or larger is None:
            continue
        if smaller >= larger:
            # Flag both — the user / FE needs to know either could be wrong.
            tag_smaller = f"anthropometric:{smaller_field}"
            tag_larger = f"anthropometric:{larger_field}"
            if tag_smaller not in violations:
                violations.append(tag_smaller)
            if tag_larger not in violations:
                violations.append(tag_larger)

    return violations
