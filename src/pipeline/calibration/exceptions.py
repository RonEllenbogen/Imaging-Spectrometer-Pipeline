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
    against a CalibrationRecord from calibration/shared/metadata.py."""

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


class InvalidConversionGainError(CalibrationError):
    """Raised when a candidate conversion-gain (photon transfer curve)
    measurement fails validation -- a saturated frame in the exposure
    sweep, or a fitted variance-vs-mean slope that isn't positive (a
    non-positive slope is physically invalid; sensor noise variance can't
    decrease with increasing signal, so this means something in the sweep
    -- illumination drift, non-linearity, insufficient dynamic range -- was
    wrong, not just an unlucky fit). reason should say which."""

    def __init__(self, reason: str):
        super().__init__(f"invalid conversion gain measurement: {reason}")
        self.reason = reason


class InsufficientDataError(CalibrationError):
    """Raised when shared/fitting.py's polynomial fit is asked to solve for
    more coefficients than it has data points to usefully estimate them
    from -- a hard mathematical requirement (degree + 2 points minimum,
    not degree + 1), not a configurable threshold. degree + 1 points is
    enough to solve for the coefficients themselves (an exact
    interpolation), but leaves zero residual degrees of freedom -- no
    excess data to estimate a reduced chi-squared or a coefficient
    uncertainty FROM, so scipy.odr reports both as (near-)zero rather
    than a real number. degree + 2 is the smallest point count with at
    least one residual degree of freedom, so a fit's reported uncertainty
    is always statistically meaningful, never degenerate. Mirrors
    analysis/exceptions.py's InsufficientDataError; kept as a separate
    CalibrationError subclass rather than reused directly, since
    calibration/ must not depend on analysis/ (see shared/fitting.py's
    module docstring)."""

    def __init__(self, degree: int, n_points: int):
        super().__init__(
            f"cannot fit degree-{degree} polynomial with only {n_points} "
            f"point(s); need at least {degree + 2} for a meaningful "
            f"uncertainty estimate (degree + 1 alone is an exact "
            f"interpolation with no residual degrees of freedom)"
        )
        self.degree = degree
        self.n_points = n_points


class LineMatchingError(CalibrationError):
    """Raised when spectral/line_matching.py's match_lines() cannot find a
    confident correspondence between detected spectral peaks and known
    reference lines -- too few peaks detected in the lamp image, or no
    candidate identification scored well enough against the predicted
    geometry pattern. reason should say which."""

    def __init__(self, reason: str):
        super().__init__(f"spectral line matching failed: {reason}")
        self.reason = reason

# Functions


__all__ = [
    "CalibrationError", "SettingsMismatchError", "InvalidFlatFieldError",
    "InvalidConversionGainError", "InsufficientDataError", "LineMatchingError",
]
