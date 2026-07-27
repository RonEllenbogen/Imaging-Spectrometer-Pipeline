"""
Builds and applies the per-session background baseline: average several beam-blocked frames into one low-noise estimate,
then subtract it from each science frame -- the first real correction
step, and the direct fix for the background-clipping bias demonstrated
earlier in this project.
"""

# Imports

import time

import numpy as np

from pipeline.acquisition import FrameData
from ..processed_frame import ProcessedFrame
from .metadata import CalibrationRecord, check_settings_match

# Constants

# Classes

# Functions


def build_baseline(frames: list[FrameData]) -> tuple[np.ndarray, CalibrationRecord]:

    '''
    Averages several background frames (laser blocked) into one baseline.

    Parameters
    ----------
    frames
        Background FrameData objects, all captured under identical
        exposure_us/gain_db -- averaging frames from different settings
        would produce a number with no physical meaning.

    Returns
    -------
    tuple[np.ndarray, CalibrationRecord]
        The averaged baseline (float64, same shape as the input frames)
        and a record tagging it with the settings and frame count it was
        built from.

    Raises
    ------
    ValueError
        If frames is empty, or if any frame's exposure_us/gain_db differs
        from the first frame's.
    '''

    if len(frames) == 0:
        raise ValueError("build_baseline() requires at least one frame")

    reference_exposure = frames[0].exposure_us
    reference_gain = frames[0].gain_db

    for f in frames[1:]:
        if f.exposure_us != reference_exposure or f.gain_db != reference_gain:
            raise ValueError(
                "all frames must share identical exposure_us and gain_db to be "
                "averaged into one baseline -- got a mismatch against the first frame"
            )
        # Exact equality is correct here, unlike check_settings_match()'s
        # tolerance-based comparison -- these frames all come from the same
        # batch capture on the same CameraStream instance, so they should
        # match exactly. Tolerance exists to absorb drift/round-tripping
        # across separate sessions, not within a single capture.

    stacked = np.stack([f.image.astype(np.float64) for f in frames], axis=0)
    averaged = stacked.mean(axis=0)

    record = CalibrationRecord(
        exposure_us=reference_exposure,
        gain_db=reference_gain,
        timestamp=time.time(),
        source_frame_count=len(frames),
    )
    return averaged, record


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


__all__ = ["build_baseline", "apply_baseline"]