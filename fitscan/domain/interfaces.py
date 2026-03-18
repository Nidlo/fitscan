"""All domain interfaces. No concrete implementations allowed here."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .models import (
    BodyMeasurements,
    BodyPose,
    ScaleFactor,
    ValidationResult,
)


@runtime_checkable
class IImageLoader(Protocol):
    def load(self, path: Path) -> np.ndarray: ...
    def validate_quality(self, image: np.ndarray) -> tuple[bool, str]: ...


@runtime_checkable
class IImagePreprocessor(Protocol):
    def remove_background(self, image: np.ndarray) -> np.ndarray: ...
    def extract_silhouette(self, image: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class IPoseDetector(Protocol):
    def detect(self, image: np.ndarray) -> BodyPose: ...


@runtime_checkable
class IScaleCalibrator(Protocol):
    def calibrate_from_height(
        self, pose: BodyPose, height_cm: float
    ) -> ScaleFactor: ...
    def calibrate_from_reference_object(
        self, image: np.ndarray, object_cm: float
    ) -> ScaleFactor: ...


@runtime_checkable
class IBodyFitter(Protocol):
    def fit(self, image: np.ndarray, pose: BodyPose) -> object: ...


@runtime_checkable
class IMeasurementStrategy(Protocol):
    def measure(self, *args: object, **kwargs: object) -> BodyMeasurements: ...


@runtime_checkable
class IValidationProvider(Protocol):
    def validate(
        self, image: np.ndarray, measurements: BodyMeasurements
    ) -> ValidationResult: ...


@runtime_checkable
class IFuser(Protocol):
    def fuse(self, sources: list[BodyMeasurements]) -> BodyMeasurements: ...
