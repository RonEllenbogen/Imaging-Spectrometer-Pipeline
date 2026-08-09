"""
Builds the per-session background baseline: average several beam-blocked
frames into one low-noise estimate -- the direct fix for the
background-clipping bias demonstrated earlier in this project. Applying
it to a science frame (subtract, clip at zero) is preprocessing/'s job
(preprocessing/steps/baseline.py), not this module's -- this module only
builds the artifact.

Also measures background_sigma ("b" in the Thompson-Larson-Webb centroid
uncertainty formula, see analysis/noise_model.py) from the same stacked
frames the baseline mean is computed from -- effectively free, since the
per-pixel spread across frames is exactly what's needed and the frames
are already in memory. Reduced to a single scalar (the median per-pixel
sample standard deviation) because analysis/noise_model.py's
SensorNoiseModel.background_sigma is a single float, assumed uniform
across the sensor -- median rather than mean specifically because no
bad-pixel mask exists yet at the point build_baseline() runs (it's built
later, from the flat field), so a robust statistic is used to avoid a
handful of hot/dead pixels skewing the one number relied on everywhere.
"""

# Imports

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.acquisition import FrameData
from ..shared.io import save_artifact, load_artifact
from ..shared.metadata import CalibrationRecord

# Constants

logger = logging.getLogger(__name__)

# Classes

@dataclass(frozen=True, slots=True)
class BaselineResult:

    '''
    build_baseline()'s output: the averaged baseline plus the background
    noise level measured alongside it. Kept separate from
    CalibrationRecord, which stays scoped to pure settings/provenance
    metadata shared across artifact types, not physical measurements
    specific to one of them.

    Parameters
    ----------
    baseline
        The averaged background, float64, same shape as the input frames.
    background_sigma
        Median per-pixel sample standard deviation (ddof=1) across the
        source frames, in ADU -- "b" in the Thompson-Larson-Webb formula.
    '''

    baseline: np.ndarray
    background_sigma: float

    def __post_init__(self) -> None:
        if self.background_sigma < 0:
            raise ValueError(f"background_sigma must be non-negative, got {self.background_sigma}")
        self.baseline.flags.writeable = False


# Functions


def build_baseline(frames: list[FrameData]) -> tuple[BaselineResult, CalibrationRecord]:

    '''
    Averages several background frames (laser blocked) into one baseline,
    and measures the background noise level across the same frames.

    Parameters
    ----------
    frames
        Background FrameData objects, all captured under identical
        exposure_us/gain_db -- averaging frames from different settings
        would produce a number with no physical meaning. At least 2
        required -- see Raises.

    Returns
    -------
    tuple[BaselineResult, CalibrationRecord]
        The averaged baseline plus measured background_sigma, and a
        record tagging it with the settings and frame count it was built
        from.

    Raises
    ------
    ValueError
        If fewer than 2 frames are given, or if any frame's
        exposure_us/gain_db differs from the first frame's. 2 frames are
        required (not 1) because background_sigma is a sample standard
        deviation (ddof=1) across frames -- undefined at n=1, and a
        silently-returned 0.0 there would be indistinguishable from a
        real "no noise measured" result rather than "not enough frames to
        measure it at all".
    '''

    if len(frames) < 2:
        raise ValueError(
            f"build_baseline() requires at least 2 frames (to measure background_sigma "
            f"across them), got {len(frames)}"
        )

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
    background_sigma = float(np.median(stacked.std(axis=0, ddof=1)))

    record = CalibrationRecord(
        exposure_us=reference_exposure,
        gain_db=reference_gain,
        timestamp=time.time(),
        source_frame_count=len(frames),
    )
    return BaselineResult(baseline=averaged, background_sigma=background_sigma), record


def save_baseline(path: str | Path, result: BaselineResult, record: CalibrationRecord) -> None:

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
    result
        The BaselineResult returned by build_baseline().
    record
        The CalibrationRecord returned alongside it.

    Returns
    -------
    None
    '''

    save_artifact(
        path,
        {"baseline": result.baseline, "background_sigma": np.array(result.background_sigma)},
        record,
    )


def load_baseline(path: str | Path) -> tuple[BaselineResult, CalibrationRecord]:

    '''
    Loads a baseline previously saved via save_baseline().

    Parameters
    ----------
    path
        The file to load. A ".npz" suffix is added if not already
        present.

    Returns
    -------
    tuple[BaselineResult, CalibrationRecord]

    Raises
    ------
    FileNotFoundError
        If path doesn't exist.
    '''

    arrays, record = load_artifact(path, CalibrationRecord)
    logger.info("loaded baseline from %s (age %.1fs)", path, record.age_seconds)
    result = BaselineResult(
        baseline=arrays["baseline"], background_sigma=float(arrays["background_sigma"]),
    )
    return result, record


__all__ = ["BaselineResult", "build_baseline", "save_baseline", "load_baseline"]
