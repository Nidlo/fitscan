"""
Fuses measurements from multiple sources (SMPL, silhouette, geometric)
into a single best estimate using weighted averaging.
Implements IFuser.
"""

from __future__ import annotations

from dataclasses import fields

from fitscan.domain.models import BodyMeasurements, SingleMeasurement

SOURCE_WEIGHTS = {
    "smpl": 0.65,
    "silhouette": 0.25,
    "geometric": 0.10,
}


class MeasurementFuser:
    """Combines multiple BodyMeasurements objects into one fused result."""

    def fuse(self, sources: list[BodyMeasurements]) -> BodyMeasurements:
        if not sources:
            raise ValueError("No measurement sources to fuse.")
        if len(sources) == 1:
            return sources[0]

        fused_fields: dict = {}
        for f in fields(BodyMeasurements):
            if f.name in ("source", "gender_profile"):
                continue
            candidates = [
                getattr(s, f.name) for s in sources if getattr(s, f.name) is not None
            ]
            if not candidates:
                fused_fields[f.name] = None
                continue
            if isinstance(candidates[0], SingleMeasurement):
                fused_fields[f.name] = self._weighted_average(candidates)
            else:
                fused_fields[f.name] = candidates[0]

        return BodyMeasurements(
            **fused_fields,
            gender_profile=sources[0].gender_profile,
            source="fused",
        )

    def _weighted_average(
        self, candidates: list[SingleMeasurement]
    ) -> SingleMeasurement:
        total_weight = 0.0
        weighted_value = 0.0
        for m in candidates:
            w = SOURCE_WEIGHTS.get(m.source, 0.10) * m.confidence
            weighted_value += m.value_cm * w
            total_weight += w
        if total_weight == 0:
            return candidates[0]
        avg_value = weighted_value / total_weight
        avg_confidence = min(
            1.0, sum(m.confidence for m in candidates) / len(candidates)
        )
        return SingleMeasurement(
            value_cm=round(avg_value, 1),
            confidence=round(avg_confidence, 3),
            source="fused",
        )
