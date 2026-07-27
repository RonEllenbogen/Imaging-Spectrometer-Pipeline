"""
Applies the spatial-axis region of interest by zeroing pixels outside it,
rather than cropping the array -- this keeps CANONICAL_SHAPE intact and
is mathematically exact for a weighted centroid: a zeroed row contributes
nothing to either the numerator or denominator, identical to a row that
was never there.

Whether ROI restriction is actually necessary for this setup is still an
open, empirical question -- pending a check of a real background frame's
histogram for genuine zero-clipping bias. apply_roi() is optional in
preprocessing_pipeline.py (roi_bounds=None skips it entirely) until
that's resolved.
"""

# Imports

from ..processed_frame import ProcessedFrame

# Constants

# Classes

# Functions


def apply_roi(frame: ProcessedFrame, row_min: int, row_max: int) -> ProcessedFrame:

    '''
    Zeroes every pixel outside [row_min, row_max) along the spatial axis.

    Parameters
    ----------
    frame
        The frame to mask -- expected to already be baseline-subtracted,
        flat-field-divided, and bad-pixel-masked, since ROI is the last
        step in the processing order.
    row_min, row_max
        The spatial-axis window to keep, in absolute detector row
        coordinates. row_max is exclusive.

    Returns
    -------
    ProcessedFrame
        Same shape, frame_id, timestamp, exposure_us, and gain_db as the
        input -- only pixel values outside the ROI change. Row indices
        stay absolute; no offset bookkeeping is needed downstream.

    Raises
    ------
    ValueError
        If row_min/row_max fall outside the frame's row range, or
        row_min >= row_max.
    '''

    n_rows = frame.image.shape[0]
    if not (0 <= row_min < row_max <= n_rows):
        raise ValueError(f"invalid ROI [{row_min}, {row_max}) for frame with {n_rows} rows")

    masked_image = frame.image.copy()
    masked_image[:row_min, :] = 0
    masked_image[row_max:, :] = 0

    return ProcessedFrame(
        image=masked_image, timestamp=frame.timestamp, frame_id=frame.frame_id,
        exposure_us=frame.exposure_us, gain_db=frame.gain_db,
    )


__all__ = ["apply_roi"]