"""
Assigns a final confidence score to each measurement after fusion.
Considers: number of sources agreeing, validation flags, photo quality.
"""

from __future__ import annotations

from dataclasses import fields

from fitscan.domain.models import (
    BodyMeasurements,
    SingleMeasurement,
    ValidationResult,
)


class ConfidenceScorer:
    """Adjusts confidence based on validation results and source agreement."""

    def score(
        self,
        measurements: BodyMeasurements,
        validation: ValidationResult,
        sources: list[BodyMeasurements],
    ) -> BodyMeasurements:
        quality_multiplier = validation.photo_quality_score / 10.0
        flagged = set(validation.flagged_fields)

        updated: dict = {}
        for f in fields(BodyMeasurements):
            if f.name in ("source", "gender_profile"):
                continue
            m = getattr(measurements, f.name)
            if m is None or not isinstance(m, SingleMeasurement):
                updated[f.name] = m
                continue

            conf = m.confidence * quality_multiplier

            # Agreement bonus
            source_values = [
                getattr(s, f.name).value_cm
                for s in sources
                if getattr(s, f.name) is not None
            ]
            if len(source_values) >= 2:
                spread = max(source_values) - min(source_values)
                if spread < 2.0:
                    conf = min(1.0, conf * 1.15)
                elif spread > 8.0:
                    conf *= 0.75

            # Penalize flagged measurements
            if f.name in flagged:
                conf *= 0.50

            updated[f.name] = SingleMeasurement(
                value_cm=m.value_cm,
                confidence=round(conf, 3),
                source=m.source,
            )

        return BodyMeasurements(
            **updated,
            gender_profile=measurements.gender_profile,
            source="fused_scored",
        )
