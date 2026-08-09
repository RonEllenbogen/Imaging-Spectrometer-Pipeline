"""
Shared calibration metadata. CalibrationRecord tags a calibration
artifact with the settings it was captured under, so a science frame's
actual settings can be checked against it before the artifact is applied.
Lives in shared/ (not sensor/) since more than one calibration subpackage
now consumes it: calibration/sensor/'s baseline/flat-field/bad-pixel-map
artifacts, and calibration/spectral/'s wavelength calibration (built from
lamp frames, which carry exposure/gain/timing exactly like any other
frame-based artifact) -- calibration/spatial/ does NOT use this, since
its scale factor isn't built from a captured frame at any particular
setting (see calibration/spatial/io.py's own ScaleFactorRecord instead).
Building the artifacts themselves is each artifact type's own
responsibility -- this module only provides the shared record shape and
the comparison logic every one of them needs.
"""

# Imports

import time
from dataclasses import dataclass

from pipeline.acquisition import FrameData
from ..exceptions import SettingsMismatchError

# Constants

EXPOSURE_MATCH_TOLERANCE_REL = 0.01   # 1% relative
GAIN_MATCH_TOLERANCE_ABS = 0.05        # dB, absolute

# Classes

@dataclass(frozen=True, slots=True)
class CalibrationRecord:

    '''
    Tags a calibration artifact with the settings, time, and frame count
    it was captured under.

    Deliberately does NOT include pixel_format -- FrameData's own
    construction already guarantees its dtype matches CANONICAL_DTYPE
    (itself derived from pixel_format), so a frame that exists at all has
    already been validated against the current pixel format; re-checking
    it here would be redundant. Deliberately does NOT include temperature
    -- no code in this project currently reads it from the camera; add it
    if a real source for it is ever built.

    Parameters
    ----------
    exposure_us
        Exposure time the source frame(s) were captured under.
    gain_db
        Gain the source frame(s) were captured under.
    timestamp
        time.time() when this record was created -- NOT time.monotonic(),
        since a calibration built in one process run may be loaded and
        compared against in a later one, where a monotonic clock's
        arbitrary per-process epoch would make the comparison meaningless.
    source_frame_count
        Number of frames averaged into this artifact. Informational only
        -- no validation logic in this package currently reads it.
    '''

    exposure_us: float
    gain_db: float
    timestamp: float
    source_frame_count: int

    def __post_init__(self) -> None:
        if self.exposure_us <= 0:
            raise ValueError(f"exposure_us must be positive, got {self.exposure_us}")
        if not (self.gain_db == self.gain_db):  # NaN check without importing math/numpy here
            raise ValueError("gain_db must not be NaN")
        if self.source_frame_count < 1:
            raise ValueError(f"source_frame_count must be at least 1, got {self.source_frame_count}")

    @property
    def age_seconds(self) -> float:

        '''
        Seconds elapsed since this calibration was captured. Pure and
        side-effect-free, same as FrameData.age_seconds -- logging based
        on this value is each artifact's own load function's
        responsibility, not this class's, and should happen once at load
        time, never inside a per-frame processing step.

        Returns
        -------
        float
        '''

        return time.time() - self.timestamp

# Functions

def check_settings_match(frame: FrameData, record: CalibrationRecord) -> None:

    '''
    Checks a frame's actual exposure/gain against a loaded calibration
    record's tagged settings. Raises on mismatch -- this stays a hard
    failure, unlike calibration age, since it's a concrete, checkable
    fact about the frame in front of you, not a judgment call based on
    an invented threshold.

    Parameters
    ----------
    frame
        The frame about to be processed.
    record
        The CalibrationRecord the artifact being applied was tagged with.

    Returns
    -------
    None

    Raises
    ------
    SettingsMismatchError
        If exposure_us or gain_db differ from the record by more than
        their respective tolerances.
    '''

    exposure_diff_rel = abs(frame.exposure_us - record.exposure_us) / record.exposure_us
    if exposure_diff_rel > EXPOSURE_MATCH_TOLERANCE_REL:
        raise SettingsMismatchError("exposure_us", frame.exposure_us, record.exposure_us)

    gain_diff_abs = abs(frame.gain_db - record.gain_db)
    if gain_diff_abs > GAIN_MATCH_TOLERANCE_ABS:
        raise SettingsMismatchError("gain_db", frame.gain_db, record.gain_db)


__all__ = ["CalibrationRecord", "check_settings_match", "EXPOSURE_MATCH_TOLERANCE_REL", "GAIN_MATCH_TOLERANCE_ABS"]
