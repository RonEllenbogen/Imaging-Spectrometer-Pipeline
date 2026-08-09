"""
Persists the active spatial scale factor (default or manually calibrated)
so a GUI-entered override survives across sessions (docs/project_state.md).
"""

# Imports

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..shared.io import save_artifact, load_artifact
from .calibrate import ScaleFactorPositionCalibration, DEFAULT_SCALE_FACTOR

# Constants

logger = logging.getLogger(__name__)

_VALID_SOURCES = ("default", "manual")

# Classes

@dataclass(frozen=True, slots=True)
class ScaleFactorRecord:

    '''
    Tags a persisted scale factor with where it came from and when.
    Deliberately carries no exposure_us/gain_db, unlike
    calibration/shared/metadata.py's CalibrationRecord -- the spatial
    scale factor is a fixed optical-design ratio (or a manual measurement
    of it), not built from a captured frame at any particular setting.

    Parameters
    ----------
    source
        "default" (DEFAULT_SCALE_FACTOR, f1/f2) or "manual" (a
        better-measured value entered by the GUI user).
    timestamp
        time.time() when this record was created.
    '''

    source: str
    timestamp: float

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(f"source must be one of {_VALID_SOURCES}, got {self.source!r}")

    @property
    def age_seconds(self) -> float:

        '''Seconds elapsed since this record was created.'''

        return time.time() - self.timestamp


# Functions

def save_scale_factor(path: str | Path, calibration: ScaleFactorPositionCalibration, source: str) -> None:

    '''
    Saves the active scale factor to path, so a manual override survives
    across sessions. Overwrites whatever was already there -- current
    instrument state, not a history to keep.

    Parameters
    ----------
    path
        Destination file. A ".npz" suffix is added if not already
        present.
    calibration
        The ScaleFactorPositionCalibration to persist.
    source
        "default" or "manual" -- see ScaleFactorRecord.

    Returns
    -------
    None
    '''

    record = ScaleFactorRecord(source=source, timestamp=time.time())
    save_artifact(path, {"scale_factor": np.array(calibration.scale_factor)}, record)


def load_scale_factor(path: str | Path) -> tuple[ScaleFactorPositionCalibration, ScaleFactorRecord]:

    '''
    Loads a scale factor previously saved via save_scale_factor(). Unlike
    calibration/sensor/'s load_*() functions, a missing file is not an
    error here -- a fresh instrument with no saved override yet is the
    expected common case (the scale factor always has a physically valid
    default, unlike a baseline or flat field, which have none), so this
    falls back to DEFAULT_SCALE_FACTOR tagged "default" instead of
    raising.

    Parameters
    ----------
    path
        The file to load. A ".npz" suffix is added if not already
        present.

    Returns
    -------
    tuple[ScaleFactorPositionCalibration, ScaleFactorRecord]
    '''

    try:
        arrays, record = load_artifact(path, ScaleFactorRecord)
    except FileNotFoundError:
        record = ScaleFactorRecord(source="default", timestamp=time.time())
        logger.info("no saved scale factor at %s -- using default (%.3f)", path, DEFAULT_SCALE_FACTOR)
        return ScaleFactorPositionCalibration(), record

    calibration = ScaleFactorPositionCalibration(scale_factor=float(arrays["scale_factor"]))
    logger.info(
        "loaded scale factor %.3f from %s (source=%s, age %.1fs)",
        calibration.scale_factor, path, record.source, record.age_seconds,
    )
    return calibration, record


__all__ = ["ScaleFactorRecord", "save_scale_factor", "load_scale_factor"]
