"""
Builds the flat field: a normalized map of each pixel's relative gain,
dividing out PRNU -- a fixed, multiplicative per-pixel gain pattern that
doesn't average away and isn't corrected by anything else in this
pipeline. Applying it to a science frame (divide, floor at
MIN_FLAT_FIELD_VALUE) is preprocessing/'s job
(preprocessing/steps/flat_field.py), not this module's -- this module
only builds the artifact.

Built infrequently (project start, after realignment) from frames of
uniform illumination captured directly on the sensor -- not through the
spectrometer's dispersing optics.
"""

# Imports

import logging
import time
from pathlib import Path

import numpy as np

from pipeline.acquisition import FrameData
from ..exceptions import InvalidFlatFieldError
from ..shared.io import save_artifact, load_artifact
from .saturation import check_saturation
from .baseline import build_baseline
from .metadata import CalibrationRecord

# Constants

logger = logging.getLogger(__name__)

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


def save_flat_field(path: str | Path, flat_field: np.ndarray, record: CalibrationRecord) -> None:

    '''
    Saves a flat-field artifact to path, so it can be reused in a later
    session without recapturing illuminated/dark frames. Overwrites
    whatever was already at path -- a flat field is current instrument
    state, not a history to keep.

    Parameters
    ----------
    path
        Destination file. A ".npz" suffix is added if not already
        present.
    flat_field
        The array returned by build_flat_field().
    record
        The CalibrationRecord returned alongside it.

    Returns
    -------
    None
    '''

    save_artifact(path, flat_field, record)


def load_flat_field(path: str | Path) -> tuple[np.ndarray, CalibrationRecord]:

    '''
    Loads a flat field previously saved via save_flat_field().

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

    flat_field, record = load_artifact(path, CalibrationRecord)
    logger.info("loaded flat field from %s (age %.1fs)", path, record.age_seconds)
    return flat_field, record


__all__ = ["build_flat_field", "save_flat_field", "load_flat_field"]
