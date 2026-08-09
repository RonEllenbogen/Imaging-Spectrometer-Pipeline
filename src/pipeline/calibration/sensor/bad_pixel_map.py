"""
Derives a bad-pixel map: pixels with a fixed, structural defect (dead --
no response to light, or hot -- anomalously elevated regardless of
illumination), as distinct from sensor/saturation.py's job, which catches
dynamic over-exposure. Applying it to a science frame (zeroing flagged
pixels) is preprocessing/'s job (preprocessing/steps/bad_pixel_map.py),
not this module's -- this module only builds the artifact.

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

import logging
import time
from pathlib import Path

import numpy as np

from ..shared.io import save_artifact, load_artifact
from ..shared.metadata import CalibrationRecord

# Constants

logger = logging.getLogger(__name__)

# Unverified starting point -- tune once real flat-field data exists.
# 5 standard deviations is a fairly conservative bar, chosen to avoid
# flagging ordinary statistical variation as a defect.
SIGMA_THRESHOLD = 5.0

# Classes

# Functions


def build_bad_pixel_map(flat_field: np.ndarray, flat_field_record: CalibrationRecord) -> tuple[np.ndarray, CalibrationRecord]:

    '''
    Flags pixels whose normalized flat-field value is more than
    SIGMA_THRESHOLD standard deviations from the population mean --
    catching both dead (near-zero) and hot (anomalously high) pixels in
    one pass.

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

    mean = flat_field.mean()
    std = flat_field.std()

    if std == 0:
        mask = np.zeros_like(flat_field, dtype=bool)
    else:
        deviation_in_sigma = np.abs(flat_field - mean) / std
        mask = deviation_in_sigma > SIGMA_THRESHOLD

    record = CalibrationRecord(
        exposure_us=flat_field_record.exposure_us,
        gain_db=flat_field_record.gain_db,
        timestamp=time.time(),
        source_frame_count=flat_field_record.source_frame_count,
    )
    return mask, record


def save_bad_pixel_map(path: str | Path, mask: np.ndarray, record: CalibrationRecord) -> None:

    '''
    Saves a bad-pixel-map artifact to path, so it can be reused in a
    later session without rebuilding it from a flat field. Overwrites
    whatever was already at path -- a bad-pixel map is current instrument
    state, not a history to keep.

    Parameters
    ----------
    path
        Destination file. A ".npz" suffix is added if not already
        present.
    mask
        The boolean array returned by build_bad_pixel_map().
    record
        The CalibrationRecord returned alongside it.

    Returns
    -------
    None
    '''

    save_artifact(path, {"mask": mask}, record)


def load_bad_pixel_map(path: str | Path) -> tuple[np.ndarray, CalibrationRecord]:

    '''
    Loads a bad-pixel map previously saved via save_bad_pixel_map().

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

    arrays, record = load_artifact(path, CalibrationRecord)
    logger.info("loaded bad-pixel map from %s (age %.1fs)", path, record.age_seconds)
    return arrays["mask"], record


__all__ = ["build_bad_pixel_map", "SIGMA_THRESHOLD", "save_bad_pixel_map", "load_bad_pixel_map"]
