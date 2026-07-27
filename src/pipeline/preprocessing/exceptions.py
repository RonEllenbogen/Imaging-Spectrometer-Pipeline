"""
Exceptions for the preprocessing package. Callers outside this package
should only ever need to catch PreprocessingError to handle "something in
preprocessing went wrong" broadly, or one of the specific subclasses below
when they need to act differently depending on which failure occurred.
"""

# Imports

# Constants

#Classes

class PreprocessingError(Exception):
    """Base class for all preprocessing-related errors."""


class SettingsMismatchError(PreprocessingError):
    """Raised when a science frame's acquisition settings don't match the
    settings a loaded calibration artifact was tagged with. This is the
    check FrameData's exposure_us/gain_db exist to enable, compared
    against a CalibrationRecord from sensor_calibration/metadata.py."""

    def __init__(self, parameter: str, frame_value, calibration_value):
        super().__init__(
            f"frame {parameter}={frame_value} does not match "
            f"calibration {parameter}={calibration_value}"
        )
        self.parameter = parameter
        self.frame_value = frame_value
        self.calibration_value = calibration_value


class SaturationError(PreprocessingError):
    """Raised when a frame's peak signal exceeds config.py's saturation
    threshold."""

    def __init__(self, peak_value: float, threshold: float, n_saturated_pixels: int):
        super().__init__(
            f"peak value {peak_value} exceeds saturation threshold {threshold} "
            f"({n_saturated_pixels} pixel(s) affected)"
        )
        self.peak_value = peak_value
        self.threshold = threshold
        self.n_saturated_pixels = n_saturated_pixels


class InvalidFlatFieldError(PreprocessingError):
    """Raised when a candidate flat field fails validation -- saturated source frames, or a
    shape/dtype mismatch. reason should say which."""

    def __init__(self, reason: str):
        super().__init__(f"invalid flat field: {reason}")
        self.reason = reason


class NoSignalError(PreprocessingError):
    """Raised when a raw frame is entirely zero -- a strong signal that
    something is broken (camera not triggered, cable fault, absurdly
    short exposure) rather than a subtle low-light condition, since even
    read noise alone should produce some nonzero counts."""

    def __init__(self, frame_id: int):
        super().__init__(f"frame {frame_id} contains no signal at all (max pixel value is 0)")
        self.frame_id = frame_id

# Functions


__all__ = [
    "PreprocessingError",
    "SettingsMismatchError",
    "SaturationError",
    "InvalidFlatFieldError",
    "NoSignalError"
]