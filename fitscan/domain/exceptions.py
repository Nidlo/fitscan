"""Domain-specific exceptions. All inherit from FitScanError."""


class FitScanError(Exception):
    """Base exception for all FitScan errors."""


class ImageQualityError(FitScanError):
    """Image does not meet minimum quality requirements."""


class PoseDetectionError(FitScanError):
    """Could not detect a valid human pose in the image."""


class ScaleCalibrationError(FitScanError):
    """Could not establish a valid scale factor."""


class SMPLFittingError(FitScanError):
    """SMPL model fitting failed or did not converge."""


class MeasurementError(FitScanError):
    """Could not extract a reliable measurement."""


class ValidationError(FitScanError):
    """Validation provider returned an error."""


class ModelNotFoundError(FitScanError):
    """Required ML model weight file not found."""
