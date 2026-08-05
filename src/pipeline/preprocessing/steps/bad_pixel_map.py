"""
Applies a bad-pixel map to one (already baseline-subtracted,
flat-field-divided) science frame, zeroing every flagged pixel. Building
the map itself is calibration/sensor/bad_pixel_map.py's job, not this
module's.

Masks (zeroes) defective pixels rather than interpolating -- mathematically
exact for a weighted centroid (a zeroed pixel contributes nothing to
either the numerator or denominator, identical to it not existing), and
avoids the bias risk interpolation carries near a sharp gradient like the
beam's edge.
"""

# Imports

import numpy as np

from ..processed_frame import ProcessedFrame

# Constants

# Classes

# Functions


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


__all__ = ["apply_bad_pixel_map"]
