"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MeasureRequest(BaseModel):
    height_cm: Optional[float] = Field(
        None, ge=50, le=250, description="User height in cm"
    )
    reference_object_cm: Optional[float] = Field(None, ge=5, le=200)
    gender_profile: str = Field("ladies", pattern="^(ladies|men|children)$")
    enable_smpl: bool = True
    enable_claude_validation: bool = True


class SingleMeasurementOut(BaseModel):
    value_cm: float
    confidence: float
    source: str


class MeasurementsResponse(BaseModel):
    status: str
    gender_profile: str = "ladies"
    avatar_id: Optional[str] = None
    # Explicit unit declaration on every response so callers (the Nidlo
    # Laravel backend, the SDK, third parties) never have to infer the unit
    # from the `_cm` field suffixes. Internal pipeline values are always cm;
    # downstream presentation layers convert to inches / mm at their own
    # boundary. Adding a new unit here is a contract change — coordinate
    # with the backend's UnitConverter and the frontend's preferences store.
    unit: str = "cm"
    # Circumferences
    bust_cm: Optional[SingleMeasurementOut] = None
    chest_cm: Optional[SingleMeasurementOut] = None
    waist_cm: Optional[SingleMeasurementOut] = None
    hip_cm: Optional[SingleMeasurementOut] = None
    under_breast_waist_cm: Optional[SingleMeasurementOut] = None
    around_arm_cm: Optional[SingleMeasurementOut] = None
    around_arm_3_4_cm: Optional[SingleMeasurementOut] = None
    wrist_cm: Optional[SingleMeasurementOut] = None
    neck_cm: Optional[SingleMeasurementOut] = None
    thigh_cm: Optional[SingleMeasurementOut] = None
    # Widths
    shoulder_cm: Optional[SingleMeasurementOut] = None
    across_back_cm: Optional[SingleMeasurementOut] = None
    across_chest_cm: Optional[SingleMeasurementOut] = None
    # Vertical lengths
    shoulder_to_nipple_cm: Optional[SingleMeasurementOut] = None
    nipple_to_nipple_cm: Optional[SingleMeasurementOut] = None
    shoulder_to_underbust_cm: Optional[SingleMeasurementOut] = None
    shoulder_to_waist_cm: Optional[SingleMeasurementOut] = None
    shoulder_to_hip_cm: Optional[SingleMeasurementOut] = None
    nape_to_waist_cm: Optional[SingleMeasurementOut] = None
    # Body + garment lengths
    height_cm: Optional[SingleMeasurementOut] = None
    inseam_cm: Optional[SingleMeasurementOut] = None
    hip_depth_cm: Optional[SingleMeasurementOut] = None
    sleeve_length_cm: Optional[SingleMeasurementOut] = None
    blouse_length_cm: Optional[SingleMeasurementOut] = None
    kaba_length_cm: Optional[SingleMeasurementOut] = None
    skirt_length_cm: Optional[SingleMeasurementOut] = None
    slit_length_cm: Optional[SingleMeasurementOut] = None
    dress_length_cm: Optional[SingleMeasurementOut] = None
    cape_cm: Optional[SingleMeasurementOut] = None
    # Status
    source: Optional[str] = None
    error: Optional[str] = None
    # S5 — Claude vision verification of CV landmarks the MediaPipe pose
    # detector flagged as low-visibility. Populated only when the pipeline
    # detected occluded landmarks AND a `subject_id` consent token was
    # provided. Shape:
    #   {"verdicts": {<landmark_name>:
    #       {"verdict": str, "x": float|null, "y": float|null}}}
    # Verdict values: visible_at_xy | pose_estimate | occluded_unrecoverable.
    landmark_verification: Optional[dict] = None
    # S2.5 — Per-landmark normalized coordinates (0-1) from MediaPipe pose
    # detection, surfaced so consumers can render an overlay on the
    # source photo. Shape: {<name>: {"x": float, "y": float, "visibility": float}}.
    # Names match `LANDMARK_NAMES` in cv_pipeline.pose_detector.
    landmarks: Optional[dict] = None
    # FS-NIDLO-VALID-03 — Structured signals about pipeline state the
    # frontend can route on without re-deriving from confidence numbers.
    # Empty list when nothing degraded. Currently emitted:
    #   "validators_disagree" — rule + Claude disagreed on `is_valid`.
    # The merged is_valid + recommendation still reflect the historical
    # contract; this is additive metadata for "manual review needed"
    # badges and ops dashboards.
    degraded_modes: list[str] = Field(default_factory=list)


class MeasurementOverrideRequest(BaseModel):
    overrides: dict[str, float] = Field(
        ...,
        description="Dict of {field_name: corrected_value_cm}",
    )


class OverrideResponse(BaseModel):
    avatar_id: str
    recalculated_measurements: dict
    updated_betas: list[float]
    convergence_note: str
