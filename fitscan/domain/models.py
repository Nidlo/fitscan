"""Immutable domain value objects. Pure data, no logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Keypoint:
    x: float  # normalized 0-1
    y: float  # normalized 0-1
    z: float  # depth (if available)
    visibility: float  # 0.0-1.0


@dataclass(frozen=True)
class BodyPose:
    """33 MediaPipe landmarks indexed by name."""

    keypoints: dict[str, Keypoint]
    image_width: int
    image_height: int

    def pixel_coords(self, name: str) -> tuple[int, int]:
        kp = self.keypoints[name]
        return int(kp.x * self.image_width), int(kp.y * self.image_height)


@dataclass(frozen=True)
class ScaleFactor:
    pixels_per_cm: float
    source: str  # "user_height" | "reference_object" | "estimated"
    confidence: float
    # Set when scale was calibrated from the user's stated height. The SMPL
    # mesh comes out of the body-fitter at template scale (~1.7m), so
    # MeshAnalyzer rescales it to this value before extracting measurements.
    # When None (e.g. reference-object calibration), measurements fall back
    # to silhouette-based scaling.
    target_height_cm: float | None = None


@dataclass(frozen=True)
class SingleMeasurement:
    value_cm: float
    confidence: float
    source: str  # "smpl" | "silhouette" | "geometric" | "fused"


@dataclass(frozen=True)
class BodyMeasurements:
    """
    Full tailor-grade measurement set.
    Derived from professional Ghanaian tailor measurement book
    (Ladies Measurement — Dress/Skirt/Blouse/Kaba & Slit standard).
    """

    # ── CIRCUMFERENCES ──────────────────────────────────────────────
    bust_cm: Optional[SingleMeasurement] = None
    chest_cm: Optional[SingleMeasurement] = None
    waist_cm: Optional[SingleMeasurement] = None
    hip_cm: Optional[SingleMeasurement] = None
    under_breast_waist_cm: Optional[SingleMeasurement] = None
    around_arm_cm: Optional[SingleMeasurement] = None
    around_arm_3_4_cm: Optional[SingleMeasurement] = None
    wrist_cm: Optional[SingleMeasurement] = None
    neck_cm: Optional[SingleMeasurement] = None
    thigh_cm: Optional[SingleMeasurement] = None

    # ── WIDTHS ───────────────────────────────────────────────────────
    shoulder_cm: Optional[SingleMeasurement] = None
    across_back_cm: Optional[SingleMeasurement] = None
    across_chest_cm: Optional[SingleMeasurement] = None

    # ── VERTICAL LENGTHS ─────────────────────────────────────────────
    shoulder_to_nipple_cm: Optional[SingleMeasurement] = None
    nipple_to_nipple_cm: Optional[SingleMeasurement] = None
    shoulder_to_underbust_cm: Optional[SingleMeasurement] = None
    shoulder_to_waist_cm: Optional[SingleMeasurement] = None
    shoulder_to_hip_cm: Optional[SingleMeasurement] = None
    nape_to_waist_cm: Optional[SingleMeasurement] = None

    # ── GARMENT LENGTHS ──────────────────────────────────────────────
    height_cm: Optional[SingleMeasurement] = None
    inseam_cm: Optional[SingleMeasurement] = None
    hip_depth_cm: Optional[SingleMeasurement] = None
    sleeve_length_cm: Optional[SingleMeasurement] = None
    blouse_length_cm: Optional[SingleMeasurement] = None
    kaba_length_cm: Optional[SingleMeasurement] = None
    skirt_length_cm: Optional[SingleMeasurement] = None
    slit_length_cm: Optional[SingleMeasurement] = None
    dress_length_cm: Optional[SingleMeasurement] = None
    cape_cm: Optional[SingleMeasurement] = None

    # ── META ──────────────────────────────────────────────────────────
    gender_profile: str = "ladies"
    source: str = "unknown"

    def to_dict(self) -> dict:
        result: dict = {
            "gender_profile": self.gender_profile,
            "source": self.source,
        }
        for field_name in self.__dataclass_fields__:
            if field_name in ("gender_profile", "source"):
                continue
            val = getattr(self, field_name)
            if isinstance(val, SingleMeasurement):
                result[field_name] = {
                    "value_cm": val.value_cm,
                    "confidence": val.confidence,
                    "source": val.source,
                }
        return result

    def applicable_fields(self) -> list[str]:
        """Returns field names relevant to this gender profile."""
        common = [
            "height_cm",
            "waist_cm",
            "hip_cm",
            "shoulder_cm",
            "across_back_cm",
            "across_chest_cm",
            "shoulder_to_waist_cm",
            "shoulder_to_hip_cm",
            "nape_to_waist_cm",
            "sleeve_length_cm",
            "around_arm_cm",
            "around_arm_3_4_cm",
            "wrist_cm",
            "neck_cm",
            "thigh_cm",
            "inseam_cm",
            "hip_depth_cm",
            "under_breast_waist_cm",
        ]
        ladies_only = [
            "bust_cm",
            "shoulder_to_nipple_cm",
            "nipple_to_nipple_cm",
            "shoulder_to_underbust_cm",
            "blouse_length_cm",
            "kaba_length_cm",
            "skirt_length_cm",
            "slit_length_cm",
            "dress_length_cm",
            "cape_cm",
        ]
        men_only = ["chest_cm"]
        if self.gender_profile == "ladies":
            return common + ladies_only
        if self.gender_profile == "men":
            return common + men_only
        return common  # children


@dataclass(frozen=True)
class ProcessedImage:
    original: np.ndarray
    no_background: np.ndarray
    silhouette_mask: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class MeshMeasurements:
    """Raw measurements from 3D SMPL mesh before scaling."""

    raw_values: dict[str, float]
    scale_factor: ScaleFactor


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    flagged_fields: list[str]
    photo_quality_score: float  # 0-10
    quality_issues: list[str]
    recommendation: str  # "proceed" | "retake"
    raw_response: str


@dataclass(frozen=True)
class LandmarkVerdict:
    """Per-landmark verdict from Claude vision (S5).

    `verdict`:
      - "visible_at_xy" — Claude identified the landmark; (x, y) populated
      - "pose_estimate" — Claude inferred from anatomy; (x, y) is best guess
      - "occluded_unrecoverable" — even Claude can't place it; (x, y) None
    """

    landmark: str
    verdict: str
    x: Optional[float] = None
    y: Optional[float] = None


@dataclass(frozen=True)
class LandmarkVerification:
    """Result of `ClaudeValidator.verify_landmarks` — the layered-AI
    verification pass that runs between MediaPipe pose detection and
    measurement computation. Only populated when the pipeline detected
    occluded landmarks AND the caller supplied a consent token (audit
    FS-C3 / S5)."""

    verdicts: dict[str, LandmarkVerdict]
    raw_response: str


@dataclass(frozen=True)
class AvatarExport:
    """Exportable 3D avatar produced from SMPL mesh fitting."""

    avatar_id: str
    glb_path: str
    obj_path: str
    thumbnail_path: str
    betas: list[float]
    thetas: list[float]
    measurements: BodyMeasurements
    gender_profile: str
    height_cm: float
    created_at: str


@dataclass(frozen=True)
class ClothingDrape:
    """Result of draping a garment over an avatar."""

    avatar_id: str
    garment_type: str
    draped_glb_path: str
    garment_measurements: dict
    fit_quality: float
