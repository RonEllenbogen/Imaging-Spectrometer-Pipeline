"""
The "press start" entry point gluing acquisition + preprocessing +
line_matching + calibrate for spectral calibration. Single-button, unlike
flat-field's two-phase capture -- a lamp calibration frame needs only one
physical setup.

Requires the caller to already have a built CalibrationSet (baseline,
flat field, bad-pixel map from calibration/sensor/): the lamp frame(s)
need the full existing preprocessing pipeline (dark/baseline subtraction,
flat-field division, bad-pixel masking) before peak detection, same as
any other science frame. This mirrors calibration/sensor/workflow.py's
existing pattern of the caller sequencing already-built pieces, rather
than this module loading anything from a hardcoded path.

Geometric tilt (calibration/spectral/geometric_tilt.py) is the one
artifact this function builds itself, from the very same raw lamp frames,
rather than requiring the caller to supply it -- unlike baseline/flat-
field/bad-pixel-map, it needs nothing but a lamp exposure to build (no
separate physical setup), so there's no reason to make the caller run a
second capture session for it. It's built from the raw frames (before
run_preprocessing() -- build_geometric_tilt_linear() does its own per-line
background handling and doesn't need baseline/flat-field applied first),
then folded into sensor_calibration for every run_preprocessing() call
below, so the averaged image line_matching.py's match_lines() sees is
already tilt-corrected: without that, match_lines()'s peak detection
would be working on row-tilt-smeared (broadened, less resolved) lines,
per scripts/measure_spectrometer_tilt.py's own exploratory findings.

Each captured frame is preprocessed individually via run_preprocessing()
(not averaged first as raw frames the way build_baseline() averages
background frames) -- run_preprocessing() only accepts one raw FrameData
at a time, and averaging raw lamp frames before dark/flat-field
correction would reintroduce the same DSNU-contamination problem
build_flat_field() avoids by dark-subtracting before normalizing. The N
resulting ProcessedFrame images are averaged afterward, purely to improve
line-detection SNR before match_lines() runs.

line_matching.py's match_lines() -- the step between "preprocessed,
averaged lamp image" and "matched (pixel, wavelength_nm) pairs" -- is not
yet implemented (blocked on lamp/reference-line selection, see its own
module docstring). run_spectral_calibration() below is written now so the
overall shape is settled; it will raise NotImplementedError (propagated
from match_lines()) until that module is filled in.
"""

# Imports

# Postpones annotation evaluation (PEP 563) so the CalibrationSet type hint
# below doesn't need pipeline.preprocessing imported at module load time --
# see run_spectral_calibration()'s local import for why: preprocessing/
# now imports calibration/spectral/ too (geometric_tilt.py's
# GeometricTiltResult, used as a CalibrationSet field), so importing
# pipeline.preprocessing here at module scope would be circular.
from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pipeline.acquisition import CameraStream

from ..shared.metadata import CalibrationRecord
from .calibrate import calibrate_spectral, WavelengthCalibrationResult
from .geometric_tilt import (
    build_geometric_tilt_linear,
    save_geometric_tilt,
    GeometricTiltResult,
    PLACEHOLDER_GAIN_E_PER_ADU,
)
from .io import save_spectral_calibration
from .line_matching import match_lines

if TYPE_CHECKING:
    from pipeline.preprocessing import CalibrationSet

# Constants

logger = logging.getLogger(__name__)

# Classes

# Functions

def run_spectral_calibration(
    camera_stream: CameraStream,
    n_frames: int,
    sensor_calibration: CalibrationSet,
    path: str | Path,
    geometric_tilt_path: str | Path,
    degree: int = 1,
    gain_e_per_adu: float = PLACEHOLDER_GAIN_E_PER_ADU,
) -> tuple[WavelengthCalibrationResult, GeometricTiltResult]:

    '''
    Acquires n_frames lamp frames from an already-running camera_stream,
    builds a geometric tilt calibration from them, preprocesses and
    averages them (tilt-corrected), matches spectral lines, fits
    pixel->wavelength_nm, and saves both results -- all in one call, since
    spectral calibration is single-phase (see module docstring for why
    geometric tilt is folded in here rather than requiring a separate
    caller-driven step).

    Parameters
    ----------
    camera_stream
        An already-running CameraStream (see
        calibration/sensor/workflow.py's run_baseline_calibration() for
        why this function does not call start()/stop() itself).
    n_frames
        Number of lamp frames to capture and average before line-matching.
    sensor_calibration
        The already-built baseline/flat-field/bad-pixel-map artifacts to
        preprocess each captured lamp frame with -- its geometric_tilt
        field is ignored/overwritten with the one built here, so it need
        not (and normally won't yet) have one set.
    path
        Where to save the resulting spectral calibration artifact.
    geometric_tilt_path
        Where to save the geometric tilt calibration built from the same
        captured frames.
    degree
        Polynomial degree for the pixel->wavelength_nm fit -- see
        calibrate.calibrate_spectral().
    gain_e_per_adu
        Real measured conversion gain (calibration/sensor/conversion_
        gain.py), passed through to build_geometric_tilt_linear() alongside
        sensor_calibration.background_sigma (always threaded through
        unconditionally -- it's a required CalibrationSet field, so
        there's no reason not to) for the Thompson-Larson-Webb centroid
        uncertainty its shared row_shift curve is now weighted by (see
        build_geometric_tilt_linear()'s own docstring). Defaults to
        PLACEHOLDER_GAIN_E_PER_ADU for a caller with no real conversion-
        gain calibration to pass -- real callers (cli/calibration.py's
        spectral-capture, gui/calibration_dialogs.py's
        SpectralCalibrationDialog) always load one and pass it.

    Returns
    -------
    tuple[WavelengthCalibrationResult, GeometricTiltResult]

    Raises
    ------
    NotImplementedError
        Propagated from match_lines() -- see module docstring.
    RuntimeError, CameraError
        Propagated from CameraStream.collect_n_frames() if camera_stream
        isn't running, or dies while collecting.
    ValueError
        If the collected frames don't all share identical exposure_us/
        gain_db -- a lamp calibration batch is captured back-to-back in
        one session, so any drift between frames indicates a setup
        problem, not something to average over.
    LineMatchingError
        Propagated from build_geometric_tilt_linear() (too few usable lines
        to build the tilt calibration) or match_lines() (too few peaks
        matched against the reference wavelength list).
    '''

    from pipeline.preprocessing import run_preprocessing   # see module docstring's import note

    frames = camera_stream.collect_n_frames(n_frames)
    reference_exposure_us = frames[0].exposure_us
    reference_gain_db = frames[0].gain_db
    for frame in frames:
        if frame.exposure_us != reference_exposure_us or frame.gain_db != reference_gain_db:
            raise ValueError(
                f"frame {frame.frame_id} has exposure_us={frame.exposure_us}, "
                f"gain_db={frame.gain_db}, but the first collected frame had "
                f"exposure_us={reference_exposure_us}, gain_db={reference_gain_db} -- "
                f"all lamp frames in one batch must share identical settings"
            )

    tilt_result = build_geometric_tilt_linear(
        frames, gain_e_per_adu=gain_e_per_adu, background_sigma=sensor_calibration.background_sigma,
    )
    save_geometric_tilt(geometric_tilt_path, tilt_result)
    calibration_with_tilt = dataclasses.replace(sensor_calibration, geometric_tilt=tilt_result)

    processed_images = [
        run_preprocessing(frame, calibration_with_tilt)[0].image for frame in frames
    ]
    averaged_image = np.mean(processed_images, axis=0)

    pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm = match_lines(averaged_image)

    reference_frame = frames[0]
    record = CalibrationRecord(
        exposure_us=reference_frame.exposure_us,
        gain_db=reference_frame.gain_db,
        timestamp=time.time(),
        source_frame_count=len(frames),
    )
    result = calibrate_spectral(
        pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm, record, degree=degree,
    )
    save_spectral_calibration(path, result)
    logger.info(
        "spectral calibration complete: %d frames -> %s (geometric tilt -> %s)",
        n_frames, path, geometric_tilt_path,
    )
    return result, tilt_result


__all__ = ["run_spectral_calibration"]
