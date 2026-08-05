"""
Exceptions for the preprocessing package. Callers outside this package
should only ever need to catch PreprocessingError to handle "something in
preprocessing went wrong" broadly, or one of the specific subclasses below
when they need to act differently depending on which failure occurred.

SettingsMismatchError and InvalidFlatFieldError used to live here too, but
moved to calibration/exceptions.py -- the functions that raise them
(check_settings_match, build_flat_field) moved to calibration/sensor/.
apply_baseline() (steps/baseline.py) still raises SettingsMismatchError by
calling check_settings_match(), and pipeline.preprocessing still
re-exports it for caller convenience, but it is no longer a
PreprocessingError -- a caller that wants to catch everything either
package can raise now needs
except (PreprocessingError, CalibrationError), not PreprocessingError alone.
"""

# Imports

# Constants

#Classes

class PreprocessingError(Exception):
    """Base class for all preprocessing-related errors."""


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
    "SaturationError",
    "NoSignalError"
]
