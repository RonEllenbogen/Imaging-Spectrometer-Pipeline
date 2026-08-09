"""
Applies the flat field to one (already baseline-subtracted) science
frame: divides out PRNU -- a fixed, multiplicative per-pixel gain pattern
that doesn't average away and isn't corrected by anything else in this
pipeline. Building the flat field itself is calibration/sensor/flat_field.py's
job, not this module's.
"""

# Imports

import numpy as np

from pipeline.calibration.shared import CalibrationRecord

from ..processed_frame import ProcessedFrame

# Constants

# Floor applied before division, guaranteeing apply_flat_field() can
# never produce inf/nan -- independent of whether bad_pixel_map.py has
# run yet. Defense in depth, not a substitute for proper bad-pixel handling.
MIN_FLAT_FIELD_VALUE = 0.01

# Classes

# Functions


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


__all__ = ["apply_flat_field", "MIN_FLAT_FIELD_VALUE"]
