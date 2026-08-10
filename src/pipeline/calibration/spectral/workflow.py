"""
The "press start" entry point gluing acquisition + preprocessing +
line_matching + calibrate for spectral calibration. Single-button, unlike
flat-field's two-phase capture -- a lamp calibration frame needs only one
physical setup (docs/project_handover.md §5's own classification).

Requires the caller to already have a built CalibrationSet (baseline,
flat field, bad-pixel map from calibration/sensor/): the lamp frame(s)
need the full existing preprocessing pipeline (dark/baseline subtraction,
flat-field division, bad-pixel masking) before peak detection, same as
any other science frame. This mirrors calibration/sensor/workflow.py's
existing pattern of the caller sequencing already-built pieces, rather
than this module loading anything from a hardcoded path.

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

import logging
import time
from pathlib import Path

import numpy as np

from pipeline.acquisition import CameraStream
from pipeline.preprocessing import run_preprocessing, CalibrationSet

from ..shared.metadata import CalibrationRecord
from .calibrate import calibrate_spectral, WavelengthCalibrationResult
from .io import save_spectral_calibration
from .line_matching import match_lines

# Constants

logger = logging.getLogger(__name__)

# Classes

# Functions

def run_spectral_calibration(
    camera_stream: CameraStream,
    n_frames: int,
    sensor_calibration: CalibrationSet,
    path: str | Path,
    degree: int = 1,
) -> WavelengthCalibrationResult:

    '''
    Acquires n_frames lamp frames from an already-running camera_stream,
    preprocesses and averages them, matches spectral lines, fits
    pixel->wavelength_nm, and saves the result to path -- all in one call,
    since spectral calibration is single-phase.

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
        preprocess each captured lamp frame with.
    path
        Where to save the resulting spectral calibration artifact.
    degree
        Polynomial degree for the pixel->wavelength_nm fit -- see
        calibrate.calibrate_spectral().

    Returns
    -------
    WavelengthCalibrationResult

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
    '''

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
    processed_images = [
        run_preprocessing(frame, sensor_calibration)[0].image for frame in frames
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
    logger.info("spectral calibration complete: %d frames -> %s", n_frames, path)
    return result


__all__ = ["run_spectral_calibration"]
