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
