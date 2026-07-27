"""
Builds and applies the flat field: a normalized
map of each pixel's relative gain, dividing out PRNU -- a fixed,
multiplicative per-pixel gain pattern that doesn't average away and
isn't corrected by anything else in this pipeline.

Built infrequently (project start, after realignment) from frames of
uniform illumination captured directly on the sensor -- not through the
spectrometer's dispersing optics.
"""

# Imports

import time

import numpy as np

from pipeline.acquisition import FrameData
from ..processed_frame import ProcessedFrame
from ..exceptions import InvalidFlatFieldError
from ..steps.saturation import check_saturation
from .baseline import build_baseline
from .metadata import CalibrationRecord

# Constants

# Floor applied before division, guaranteeing apply_flat_field() can
# never produce inf/nan -- independent of whether bad_pixel_map.py has
# run yet. Defense in depth, not a substitute for proper bad-pixel handling.
MIN_FLAT_FIELD_VALUE = 0.01

# Classes

# Functions


def build_flat_field(
    illuminated_frames: list[FrameData], dark_frames: list[FrameData]
) -> tuple[np.ndarray, CalibrationRecord]:

    '''
    Builds a normalized flat field from uniformly-illuminated frames,
    dark-subtracted using dark_frames captured at the same settings.

    PRNU is multiplicative; DSNU is additive. Normalizing illuminated
    frames without subtracting their own dark level first would produce
    a flat field contaminated by DSNU -- the wrong correction to apply
    via division.

    Parameters
    ----------
    illuminated_frames
        Frames of uniform illumination, all at identical exposure_us/gain_db.
    dark_frames
        Background frames at the SAME settings as illuminated_frames --
        not the science-session baseline, which is very likely different.

    Returns
    -------
    tuple[np.ndarray, CalibrationRecord]

    Raises
    ------
    InvalidFlatFieldError
        If any illuminated frame is saturated -- rejects the whole build
        rather than excluding frames, since a saturated calibration
        source is a setup problem (illumination too bright, exposure too
        long) that should be fixed, not routed around.
    ValueError
        Propagated from build_baseline() if frame settings are inconsistent.
    '''

    for frame in illuminated_frames:
        result = check_saturation(frame)
        if result.is_saturated:
            raise InvalidFlatFieldError(
                f"illuminated frame {frame.frame_id} is saturated "
                f"({result.n_saturated_pixels} px at {result.peak_value}); "
                f"reduce illumination or exposure and recapture"
            )

    illuminated_average, illum_record = build_baseline(illuminated_frames)
    dark_average, _ = build_baseline(dark_frames)

    dark_subtracted = np.clip(illuminated_average - dark_average, 0, None)

    mean_value = dark_subtracted.mean()
    if mean_value <= 0:
        raise InvalidFlatFieldError(
            "dark-subtracted illuminated average has non-positive mean -- "
            "check illumination source and dark frame settings"
        )

    flat_field = dark_subtracted / mean_value

    record = CalibrationRecord(
        exposure_us=illum_record.exposure_us,
        gain_db=illum_record.gain_db,
        timestamp=time.time(),
        source_frame_count=len(illuminated_frames),
    )
    return flat_field, record


def apply_flat_field(frame: ProcessedFrame, flat_field: np.ndarray, record: CalibrationRecord) -> ProcessedFrame:

    '''
    Divides frame by flat_field, correcting PRNU. Deliberately does NOT
    check frame's settings against record's;
    PRNU is treated as exposure/gain-independent within the linear regime.

    Parameters
    ----------
    frame
        The (already baseline-subtracted) frame to correct.
    flat_field
        The normalized flat field, from build_flat_field().
    record
        Retained for logging/provenance only -- not used to gate this call.

    Returns
    -------
    ProcessedFrame

    Raises
    ------
    ValueError
        If flat_field's shape doesn't match frame.image's.
    '''

    if flat_field.shape != frame.image.shape:
        raise ValueError(
            f"flat_field shape {flat_field.shape} does not match frame shape {frame.image.shape}"
        )

    safe_flat_field = np.clip(flat_field, MIN_FLAT_FIELD_VALUE, None)
    corrected = frame.image / safe_flat_field

    return ProcessedFrame(
        image=corrected, frame_id=frame.frame_id, timestamp=frame.timestamp,
        exposure_us=frame.exposure_us, gain_db=frame.gain_db,
    )


__all__ = ["build_flat_field", "apply_flat_field", "MIN_FLAT_FIELD_VALUE"]