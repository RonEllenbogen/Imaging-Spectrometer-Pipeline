"""
Derives and applies a bad-pixel map: pixels with a
fixed, structural defect (dead -- no response to light, or hot --
anomalously elevated regardless of illumination), as distinct from
steps/saturation.py's job, which catches dynamic over-exposure.

Derived directly from the flat field's own normalized values, rather
than a separate capture pass -- a pixel far from the population's
typical response in uniform illumination is a strong signal of a
structural defect.

Masks (zeroes) defective pixels rather than interpolating -- mathematically
exact for a weighted centroid (a zeroed pixel contributes nothing to
either the numerator or denominator, identical to it not existing), and
avoids the bias risk interpolation carries near a sharp gradient like the
beam's edge.
"""

# Imports

import time

import numpy as np

from ..processed_frame import ProcessedFrame
from .metadata import CalibrationRecord

# Constants

# Unverified starting point -- tune once real flat-field data exists.
# 5 MAD is a fairly conservative bar, chosen to avoid flagging ordinary
# statistical variation as a defect.
MAD_THRESHOLD = 5.0

# Classes

# Functions


def build_bad_pixel_map(flat_field: np.ndarray, flat_field_record: CalibrationRecord) -> tuple[np.ndarray, CalibrationRecord]:

    '''
    Flags pixels whose normalized flat-field value is more than
    MAD_THRESHOLD median-absolute-deviations from the population median
    -- catching both dead (near-zero) and hot (anomalously high) pixels
    in one pass.

    Parameters
    ----------
    flat_field
        The normalized flat field, from build_flat_field().
    flat_field_record
        The CalibrationRecord the flat field was tagged with. Its
        exposure_us/gain_db are carried through into the returned record
        for provenance -- not because a bad-pixel map is itself tied to
        specific settings (a permanent hardware defect isn't), but to
        avoid introducing a second metadata type for one file.

    Returns
    -------
    tuple[np.ndarray, CalibrationRecord]
        A boolean mask (True = defective), same shape as flat_field, and
        a record noting when this map was derived and from how large a
        flat field.
    '''

    median = np.median(flat_field)
    mad = np.median(np.abs(flat_field - median))

    # Guard against a degenerate all-identical flat field, where mad=0
    # would make every nonzero deviation look infinitely significant.
    if mad == 0:
        mask = np.zeros_like(flat_field, dtype=bool)
    else:
        deviation_in_mads = np.abs(flat_field - median) / mad
        mask = deviation_in_mads > MAD_THRESHOLD

    record = CalibrationRecord(
        exposure_us=flat_field_record.exposure_us,
        gain_db=flat_field_record.gain_db,
        timestamp=time.time(),
        source_frame_count=flat_field_record.source_frame_count,
    )
    return mask, record


def apply_bad_pixel_map(frame: ProcessedFrame, mask: np.ndarray) -> ProcessedFrame:

    '''
    Zeroes every pixel flagged in mask.

    Parameters
    ----------
    frame
        The frame to correct -- expected to already be baseline-subtracted
        and flat-field-divided, per the §6 processing order.
    mask
        Boolean array from build_bad_pixel_map(), True = defective.

    Returns
    -------
    ProcessedFrame

    Raises
    ------
    ValueError
        If mask's shape doesn't match frame.image's.
    '''

    if mask.shape != frame.image.shape:
        raise ValueError(f"mask shape {mask.shape} does not match frame shape {frame.image.shape}")

    masked_image = frame.image.copy()
    masked_image[mask] = 0

    return ProcessedFrame(
        image=masked_image, frame_id=frame.frame_id, timestamp=frame.timestamp,
        exposure_us=frame.exposure_us, gain_db=frame.gain_db,
    )


__all__ = ["build_bad_pixel_map", "apply_bad_pixel_map", "MAD_THRESHOLD"]