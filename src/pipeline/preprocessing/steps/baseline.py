"""
Applies the per-session background baseline to one science frame:
subtract, then clip at zero (a pixel count can't be physically negative)
-- the first real correction step, and the direct fix for the
background-clipping bias demonstrated earlier in this project. Building
the baseline itself is calibration/sensor/baseline.py's job, not this
module's.
"""

# Imports

import numpy as np

from pipeline.acquisition import FrameData
from pipeline.calibration.shared import CalibrationRecord, check_settings_match

from ..processed_frame import ProcessedFrame

# Constants

# Classes

# Functions


def apply_baseline(frame: FrameData, baseline: np.ndarray, record: CalibrationRecord) -> ProcessedFrame:

    '''
    Subtracts a baseline from a science frame, clipping negative results
    at zero (a pixel count can't be physically negative).

    Parameters
    ----------
    frame
        The raw science frame to correct.
    baseline
        The averaged background, from build_baseline().
    record
        The CalibrationRecord the baseline was tagged with -- checked
        against frame's actual settings before subtracting.

    Returns
    -------
    ProcessedFrame
        frame_id, timestamp, exposure_us, and gain_db are preserved from
        the input frame.

    Raises
    ------
    SettingsMismatchError
        If frame's exposure_us/gain_db don't match record's, within
        tolerance.
    ValueError
        If baseline's shape doesn't match frame.image's.
    '''

    check_settings_match(frame, record)

    if baseline.shape != frame.image.shape:
        raise ValueError(
            f"baseline shape {baseline.shape} does not match frame shape {frame.image.shape}"
        )

    subtracted = np.clip(frame.image.astype(np.float64) - baseline, 0, None)

    return ProcessedFrame(
        image=subtracted,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        exposure_us=frame.exposure_us,
        gain_db=frame.gain_db,
    )


__all__ = ["apply_baseline"]
