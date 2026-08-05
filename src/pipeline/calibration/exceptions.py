"""
Exceptions for the calibration package. Callers outside this package
should only ever need to catch CalibrationError to handle "something in
calibration went wrong" broadly, or one of the specific subclasses below
when they need to act differently depending on which failure occurred.

SettingsMismatchError and InvalidFlatFieldError used to live in
preprocessing/exceptions.py as PreprocessingError subclasses -- they moved
here because the functions that raise them (check_settings_match,
build_flat_field) moved to calibration/sensor/. preprocessing/ still
raises SettingsMismatchError (apply_baseline calls check_settings_match),
and re-exports it from pipeline.preprocessing for caller convenience, but
it is no longer a PreprocessingError -- callers that need to catch both
preprocessing- and calibration-raised failures broadly now catch
(PreprocessingError, CalibrationError), not PreprocessingError alone.
"""

# Imports

# Constants

# Classes

class CalibrationError(Exception):
    """Base class for all calibration-related errors."""


class SettingsMismatchError(CalibrationError):
    """Raised when a science frame's acquisition settings don't match the
    settings a loaded calibration artifact was tagged with. This is the
    check FrameData's exposure_us/gain_db exist to enable, compared
    against a CalibrationRecord from calibration/sensor/metadata.py."""

    def __init__(self, parameter: str, frame_value, calibration_value):
        super().__init__(
            f"frame {parameter}={frame_value} does not match "
            f"calibration {parameter}={calibration_value}"
        )
        self.parameter = parameter
        self.frame_value = frame_value
        self.calibration_value = calibration_value


class InvalidFlatFieldError(CalibrationError):
    """Raised when a candidate flat field fails validation -- saturated source frames, or a
    shape/dtype mismatch. reason should say which."""

    def __init__(self, reason: str):
        super().__init__(f"invalid flat field: {reason}")
        self.reason = reason

# Functions


__all__ = ["CalibrationError", "SettingsMismatchError", "InvalidFlatFieldError"]
