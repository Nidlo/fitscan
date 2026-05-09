"""Tests for domain exceptions."""

from fitscan.domain.exceptions import (
    FitScanError,
    ImageQualityError,
    MeasurementError,
    ModelNotFoundError,
    PoseDetectionError,
    ScaleCalibrationError,
    SMPLFittingError,
    ValidationError,
)


class TestExceptions:
    def test_all_inherit_from_fitscan_error(self):
        errors = [
            ImageQualityError,
            PoseDetectionError,
            ScaleCalibrationError,
            SMPLFittingError,
            MeasurementError,
            ValidationError,
            ModelNotFoundError,
        ]
        for err_class in errors:
            assert issubclass(err_class, FitScanError)

    def test_error_messages(self):
        e = ImageQualityError("too blurry")
        assert str(e) == "too blurry"
