"""
Lets a user manually override which spectral columns are treated as
signal-bearing, in place of preprocessing.steps.signal_threshold's
automatic per-column SNR gate -- e.g. after visually confirming, from the
live feed, that the beam only spans a narrower (or wider) column range
than the SNR heuristic picked. Per docs/project_state.md's "GUI live view:
manual ROI entry -- spectral axis" item.

Deliberately shaped like apply_signal_threshold(), not apply_roi():
this only ever replaces ProcessedFrame.valid_columns, never zeros
frame.image -- an override of the *decision* signal_threshold.py makes,
not an additional physical mask like the spatial ROI's apply_roi(). A
column inside the manual window is treated as valid regardless of its
actual SNR; a column outside it is excluded regardless of how strong its
signal is.
"""

# Imports

import numpy as np

from pipeline.acquisition import SPECTRAL_AXIS

from ..processed_frame import ProcessedFrame

# Constants

# Classes

# Functions


def apply_spectral_roi(frame: ProcessedFrame, column_min: int, column_max: int) -> ProcessedFrame:

    '''
    Overrides valid_columns with a manual [column_min, column_max) window,
    replacing whatever apply_signal_threshold() (or a prior call to this
    function) computed -- see module docstring for why this replaces
    rather than zeros.

    Parameters
    ----------
    frame
        The frame to override. Order relative to apply_signal_threshold()
        doesn't matter for correctness (this simply overwrites
        valid_columns), but preprocessing_pipeline.py runs it after, so a
        caller can always see this as "the last word" on which columns
        are valid.
    column_min, column_max
        The spectral-axis window to keep, in absolute detector column
        coordinates. column_max is exclusive -- same half-open convention
        as steps/roi.py's apply_roi().

    Returns
    -------
    ProcessedFrame
        Same image, frame_id, timestamp, exposure_us, and gain_db as the
        input -- only valid_columns changes.

    Raises
    ------
    ValueError
        If column_min/column_max fall outside the frame's column range,
        or column_min >= column_max.
    '''

    n_columns = frame.image.shape[SPECTRAL_AXIS]
    if not (0 <= column_min < column_max <= n_columns):
        raise ValueError(
            f"invalid spectral ROI [{column_min}, {column_max}) for frame with {n_columns} columns"
        )

    valid_columns = np.zeros(n_columns, dtype=bool)
    valid_columns[column_min:column_max] = True

    return ProcessedFrame(
        image=frame.image, frame_id=frame.frame_id, timestamp=frame.timestamp,
        exposure_us=frame.exposure_us, gain_db=frame.gain_db, valid_columns=valid_columns,
    )


__all__ = ["apply_spectral_roi"]
