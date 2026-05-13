"""
Fuses measurements from multiple sources (SMPL, silhouette, geometric)
into a single best estimate using weighted averaging.
Implements IFuser.
"""

from __future__ import annotations

from dataclasses import fields

from fitscan.domain.models import BodyMeasurements, SingleMeasurement

# Sprint 35 — Claude vision (with both photos + the tape-measure technique
# prompt + height anchor) lands closer to ground truth on most fields
# than SMPL alone. SMPL produces a 3D mesh but its shape betas
# under-estimate diverse body types it didn't see in training, so it
# regularly returns anatomically-impossible circumferences for non-mean
# users. Trust Claude's high-confidence votes; silhouette + SMPL still
# contribute when Claude is uncertain.
#
# These priors are PER-SOURCE caps. The real per-field weight is
# priority × confidence, so a Claude reading with confidence 0.9 still
# beats an SMPL reading with confidence 0.4 (0.9 × 1.00 vs 0.4 × 0.65).
SOURCE_PRIORITY = {
    "claude": 1.00,
    "silhouette_multiview": 0.80,
    "claude_corrected": 0.95,
    "smpl": 0.55,
    "silhouette": 0.40,
    "geometric": 0.30,
}

# When a single source is dramatically more confident than all the others,
# don't dilute it by averaging. This catches the "Claude says waist=95cm
# with 0.85 confidence; SMPL says waist=146cm with 0.3 confidence" case —
# the previous weighted average produced ~115cm; the dominance rule
# returns 95cm.
#
# 1.4 is the smallest ratio that wins for a realistic Claude-vs-others
# spread: Claude conf 0.85 × priority 1.0 = 0.85; silhouette_multiview
# conf 0.68 × priority 0.80 = 0.544; ratio = 1.56 ≥ 1.4 → Claude wins.
# Lower than this and routine 3-source ties pull values toward
# silhouette mean; higher and Claude rarely dominates even when it
# obviously should.
DOMINANCE_RATIO = 1.4


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
                fused_fields[f.name] = self._fuse_candidates(candidates)
            else:
                fused_fields[f.name] = candidates[0]

        return BodyMeasurements(
            **fused_fields,
            gender_profile=sources[0].gender_profile,
            source="fused",
        )

    def _fuse_candidates(
        self, candidates: list[SingleMeasurement]
    ) -> SingleMeasurement:
        """Per-field fusion.

        Each candidate has a `weight = SOURCE_PRIORITY[source] × confidence`.
        If the highest-weight candidate beats the next one by
        ≥ `DOMINANCE_RATIO`×, we use it directly — silently averaging a
        confident measurement against several uncertain ones drags
        accurate values toward inaccurate population means. Otherwise we
        weighted-average, same as before. Sprint 35.
        """
        weighted: list[tuple[float, SingleMeasurement]] = [
            (SOURCE_PRIORITY.get(m.source, 0.20) * m.confidence, m) for m in candidates
        ]
        weighted.sort(key=lambda pair: -pair[0])

        # Dominance shortcut — the strongest candidate is meaningfully
        # stronger than runner-up, return it as-is.
        if len(weighted) >= 2 and weighted[0][0] >= DOMINANCE_RATIO * weighted[1][0]:
            top_weight, top = weighted[0]
            return SingleMeasurement(
                value_cm=top.value_cm,
                confidence=round(min(1.0, top.confidence), 3),
                source=top.source,
            )
        if len(weighted) == 1:
            return weighted[0][1]

        total_weight = sum(w for w, _ in weighted)
        if total_weight == 0:
            return candidates[0]
        avg_value = sum(w * m.value_cm for w, m in weighted) / total_weight
        # Confidence is the WEIGHT-WEIGHTED mean confidence — using the
        # plain average pulled it toward low-confidence outliers.
        avg_confidence = sum(w * m.confidence for w, m in weighted) / total_weight
        return SingleMeasurement(
            value_cm=round(avg_value, 1),
            confidence=round(min(1.0, avg_confidence), 3),
            source="fused",
        )
