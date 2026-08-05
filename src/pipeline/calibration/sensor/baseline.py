"""
Builds the per-session background baseline: average several beam-blocked
frames into one low-noise estimate -- the direct fix for the
background-clipping bias demonstrated earlier in this project. Applying
it to a science frame (subtract, clip at zero) is preprocessing/'s job
(preprocessing/steps/baseline.py), not this module's -- this module only
builds the artifact.
"""

# Imports

import logging
import time
from pathlib import Path

import numpy as np

from pipeline.acquisition import FrameData
from ..shared.io import save_artifact, load_artifact
from .metadata import CalibrationRecord

# Constants

logger = logging.getLogger(__name__)

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


def save_baseline(path: str | Path, baseline: np.ndarray, record: CalibrationRecord) -> None:

    '''
    Saves a baseline artifact to path, so it can be reused in a later
    session without recapturing background frames. Overwrites whatever
    was already at path -- a baseline is current instrument state, not a
    history to keep.

    Parameters
    ----------
    path
        Destination file. A ".npz" suffix is added if not already
        present.
    baseline
        The array returned by build_baseline().
    record
        The CalibrationRecord returned alongside it.

    Returns
    -------
    None
    '''

    save_artifact(path, baseline, record)


def load_baseline(path: str | Path) -> tuple[np.ndarray, CalibrationRecord]:

    '''
    Loads a baseline previously saved via save_baseline().

    Parameters
    ----------
    path
        The file to load. A ".npz" suffix is added if not already
        present.

    Returns
    -------
    tuple[np.ndarray, CalibrationRecord]

    Raises
    ------
    FileNotFoundError
        If path doesn't exist.
    '''

    baseline, record = load_artifact(path, CalibrationRecord)
    logger.info("loaded baseline from %s (age %.1fs)", path, record.age_seconds)
    return baseline, record


__all__ = ["build_baseline", "save_baseline", "load_baseline"]
