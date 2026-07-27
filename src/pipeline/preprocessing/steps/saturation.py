"""
Raw-frame saturation check. Runs on the RAW frame -- before flat-field
division changes the numeric domain and decouples pixel values from the
original ADC ceiling -- checking for genuine ADC clipping, which no
downstream correction can recover.

Returns a result the caller acts on rather than raising: whether to
discard a saturated frame, log it and continue, or escalate to a hard
failure is a decision that depends on context this module doesn't have
-- e.g. a live single-shot capture vs. one frame of a larger batch where
losing one shot out of N is fine.
"""

# Imports

from dataclasses import dataclass

import numpy as np

from pipeline.acquisition import FrameData, CANONICAL_MAX_VALUE

# Constants

# Classes


@dataclass(frozen=True, slots=True)
class SaturationCheckResult:

    '''
    Result of a saturation check against one raw frame.

    Parameters
    ----------
    frame_id
        Identifies which frame this result belongs to -- useful when
        checking many frames in a batch and needing to know which ones
        triggered a flag.
    is_saturated
        True if at least one non-excluded pixel is at threshold.
    peak_value
        The highest value among non-excluded pixels.
    n_saturated_pixels
        Count of non-excluded pixels at or above threshold.
    threshold
        The value checked against -- CANONICAL_MAX_VALUE at the time
        this check ran.
    '''

    frame_id: int
    is_saturated: bool
    peak_value: int
    n_saturated_pixels: int
    threshold: int


# Functions


def check_saturation(frame: FrameData, bad_pixel_mask: "np.ndarray | None" = None) -> SaturationCheckResult:

    '''
    Checks a raw frame for pixels at CANONICAL_MAX_VALUE, excluding any
    pixels flagged in bad_pixel_mask (known static sensor defects, not
    genuine dynamic over-exposure).

    Parameters
    ----------
    frame
        The RAW frame to check -- must be checked before flat-field
        division or linearity correction, since those change pixel
        values in ways that no longer correspond to the original ADC
        reading.
    bad_pixel_mask
        Boolean array, same shape as frame.image, True where a pixel is
        a known static defect (from bad_pixel_map.py) and should be
        excluded. None (the default) checks every pixel -- reasonable
        before a bad-pixel map has been built yet.

    Returns
    -------
    SaturationCheckResult
        Does not raise on its own -- see module docstring.

    Raises
    ------
    ValueError
        If bad_pixel_mask is provided but its shape doesn't match
        frame.image.
    '''

    if bad_pixel_mask is not None and bad_pixel_mask.shape != frame.image.shape:
        raise ValueError(
            f"bad_pixel_mask shape {bad_pixel_mask.shape} does not match "
            f"frame.image shape {frame.image.shape}"
        )

    checked_pixels = frame.image[~bad_pixel_mask] if bad_pixel_mask is not None else frame.image

    if checked_pixels.size == 0:
        # every pixel excluded -- nothing left to check
        return SaturationCheckResult(
            frame_id=frame.frame_id, is_saturated=False,
            peak_value=0, n_saturated_pixels=0, threshold=CANONICAL_MAX_VALUE,
        )

    peak_value = int(checked_pixels.max())
    n_saturated_pixels = int(np.count_nonzero(checked_pixels >= CANONICAL_MAX_VALUE))

    return SaturationCheckResult(
        frame_id=frame.frame_id,
        is_saturated=n_saturated_pixels > 0,
        peak_value=peak_value,
        n_saturated_pixels=n_saturated_pixels,
        threshold=CANONICAL_MAX_VALUE,
    )


__all__ = ["SaturationCheckResult", "check_saturation"]