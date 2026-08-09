"""
The "press start" entry points that glue acquisition/'s CameraStream to
this package's own build_*()/save_*() -- acquire frames from an already-
running stream, build the artifact, save it. Neither CameraStream nor
build_*()/save_*() need to know about each other for this to work; this
module is pure orchestration.

Baseline calibration is single-phase (one physical setup: laser blocked)
so it gets one function that does acquire -> build -> save in one call.

Flat-field calibration needs the physical setup changed partway through
(dark, then uniformly illuminated, or vice versa) -- forcing that into
one function would mean either blocking synchronously mid-call waiting
for a human to change the setup (freezing whatever thread calls it,
including a future GUI event loop), or accepting a callback parameter to
signal "ready for phase two". Neither is decided yet (gui/ is not
started), so this module exposes the two acquisition phases separately
instead -- capture_dark_frames() and capture_illuminated_frames() -- and
leaves combining them (build_flat_field() + save_flat_field()) to
finish_flat_field_calibration(), called once the caller has both.

Conversion-gain calibration sweeps exposure time across several levels,
which -- unlike flat-field's physical setup change -- is entirely
software-controlled, so it CAN be done in one blocking call despite also
being multi-step: run_conversion_gain_calibration() repeatedly
stops/reconfigures/restarts camera_stream itself as it steps through the
sweep (see its own docstring for why: CameraStream has no way to change
exposure_us while running). This interrupts live view on that stream for
the sweep's duration, restored to its original exposure_us once it's done.

build_bad_pixel_map() has no workflow function here -- it's derived
purely from an already-built flat field, with no CameraStream involved,
so there's no acquisition step for this module to add value around.
Callers chain build_bad_pixel_map() + save_bad_pixel_map() directly.
"""

# Imports

import logging
from pathlib import Path

import numpy as np

from pipeline.acquisition import CameraStream, FrameData

from ..shared.metadata import CalibrationRecord
from .baseline import build_baseline, save_baseline
from .conversion_gain import (
    build_conversion_gain, save_conversion_gain, ConversionGainRecord,
    MIN_ILLUMINATION_LEVELS, MIN_FRAMES_PER_LEVEL,
)
from .flat_field import build_flat_field, save_flat_field

# Constants

logger = logging.getLogger(__name__)

# Classes

# Functions


def run_baseline_calibration(camera_stream: CameraStream, n_frames: int, path: str | Path) -> CalibrationRecord:

    '''
    Acquires n_frames background frames (laser blocked) from an already-
    running camera_stream, builds a baseline, and saves it to path -- all
    in one call, since baseline calibration is single-phase.

    Parameters
    ----------
    camera_stream
        An already-running CameraStream. This function does not call
        start()/stop() itself -- it's meant to share whatever stream is
        already running for live view, via collect_n_frames() (see its
        own docstring for why: GigE allows only one open connection per
        camera).
    n_frames
        Number of background frames to average into the baseline.
    path
        Where to save the resulting baseline artifact.

    Returns
    -------
    CalibrationRecord
        The record the newly built (and saved) baseline was tagged with.

    Raises
    ------
    ValueError
        Propagated from build_baseline() (e.g. n_frames < 2).
    RuntimeError, CameraError
        Propagated from CameraStream.collect_n_frames() if camera_stream
        isn't running, or dies while collecting.
    '''

    frames = camera_stream.collect_n_frames(n_frames)
    result, record = build_baseline(frames)
    save_baseline(path, result, record)
    logger.info("baseline calibration complete: %d frames -> %s", n_frames, path)
    return record


def capture_dark_frames(camera_stream: CameraStream, n_frames: int) -> list[FrameData]:

    '''
    Acquires n_frames with flat-field calibration's illumination source
    off/blocked -- the first of its two capture phases (see module
    docstring for why flat-field calibration is split across functions
    rather than a single run_*_calibration() call).

    Parameters
    ----------
    camera_stream
        An already-running CameraStream.
    n_frames
        Number of dark frames to capture.

    Returns
    -------
    list[FrameData]
    '''

    return camera_stream.collect_n_frames(n_frames)


def capture_illuminated_frames(camera_stream: CameraStream, n_frames: int) -> list[FrameData]:

    '''
    Acquires n_frames with uniform illumination on the sensor (not
    through the spectrometer's dispersing optics) -- flat-field
    calibration's second capture phase, taken after the physical setup
    has been changed from capture_dark_frames()'s (see module docstring).

    Parameters
    ----------
    camera_stream
        An already-running CameraStream.
    n_frames
        Number of illuminated frames to capture.

    Returns
    -------
    list[FrameData]
    '''

    return camera_stream.collect_n_frames(n_frames)


def finish_flat_field_calibration(
    illuminated_frames: list[FrameData], dark_frames: list[FrameData], path: str | Path
) -> CalibrationRecord:

    '''
    Builds and saves a flat field from the two phases captured via
    capture_illuminated_frames()/capture_dark_frames(), once the caller
    has both. Touches no CameraStream at all -- just build_flat_field() +
    save_flat_field(), named/grouped here so the full sequence
    (capture_dark_frames -> capture_illuminated_frames ->
    finish_flat_field_calibration) reads as one workflow.

    Parameters
    ----------
    illuminated_frames, dark_frames
        As returned by capture_illuminated_frames()/capture_dark_frames().
    path
        Where to save the resulting flat-field artifact.

    Returns
    -------
    CalibrationRecord

    Raises
    ------
    InvalidFlatFieldError
        Propagated from build_flat_field() if illuminated_frames contains
        a saturated frame.
    '''

    flat_field, record = build_flat_field(illuminated_frames, dark_frames)
    save_flat_field(path, flat_field, record)
    logger.info(
        "flat-field calibration complete: %d illuminated + %d dark frames -> %s",
        len(illuminated_frames), len(dark_frames), path,
    )
    return record


def run_conversion_gain_calibration(
    camera_stream: CameraStream,
    exposure_min_us: float,
    exposure_max_us: float,
    n_levels: int,
    n_frames_per_level: int,
    path: str | Path,
) -> ConversionGainRecord:

    '''
    Sweeps camera_stream's exposure time across n_levels evenly-spaced
    values between exposure_min_us and exposure_max_us, capturing
    n_frames_per_level frames at each, then builds and saves a
    conversion-gain measurement from the results -- all in one call.

    Unlike every other workflow function in this module, this one DOES
    stop()/start() camera_stream itself, repeatedly -- CameraStream has no
    way to change exposure_us while running (see conversion_gain.py's
    module docstring), so a stop -> mutate exposure_us -> start cycle per
    level is the only way to sweep it. This interrupts live view (if any)
    sharing this stream for the whole duration of the sweep -- unlike
    baseline/flat-field calibration, which never touch the stream's
    settings and can run alongside live view untouched.
    camera_stream's original exposure_us is restored (and the stream
    restarted) once the sweep finishes, successfully or not.

    Parameters
    ----------
    camera_stream
        An already-running CameraStream.
    exposure_min_us, exposure_max_us
        Bounds of the exposure sweep, in microseconds. Choosing a range
        that spans from just above the noise floor to just below
        saturation, at this setup's fixed illumination brightness, is the
        caller's responsibility -- see conversion_gain.py's module
        docstring for why this isn't determined automatically.
    n_levels
        Number of evenly-spaced exposure levels to sample. At least
        MIN_ILLUMINATION_LEVELS.
    n_frames_per_level
        Number of frames to capture at each level. At least
        MIN_FRAMES_PER_LEVEL.
    path
        Where to save the resulting conversion-gain artifact.

    Returns
    -------
    ConversionGainRecord

    Raises
    ------
    ValueError
        If n_levels/n_frames_per_level are below their minimums, or
        exposure_max_us <= exposure_min_us.
    InvalidConversionGainError
        Propagated from build_conversion_gain() if a level saturates or
        the fit comes out physically invalid.
    RuntimeError
        If camera_stream isn't running when this is called.
    '''

    if n_levels < MIN_ILLUMINATION_LEVELS:
        raise ValueError(f"n_levels must be at least {MIN_ILLUMINATION_LEVELS}, got {n_levels}")
    if n_frames_per_level < MIN_FRAMES_PER_LEVEL:
        raise ValueError(
            f"n_frames_per_level must be at least {MIN_FRAMES_PER_LEVEL}, got {n_frames_per_level}"
        )
    if exposure_max_us <= exposure_min_us:
        raise ValueError(
            f"exposure_max_us ({exposure_max_us}) must be greater than "
            f"exposure_min_us ({exposure_min_us})"
        )
    if not camera_stream.is_running:
        raise RuntimeError("run_conversion_gain_calibration() requires an already-running CameraStream")

    original_exposure_us = camera_stream.exposure_us
    exposure_levels = np.linspace(exposure_min_us, exposure_max_us, n_levels)

    frames_by_exposure: dict[float, list[FrameData]] = {}
    try:
        for exposure_us in exposure_levels:
            exposure_us = float(exposure_us)
            camera_stream.stop()
            camera_stream.exposure_us = exposure_us
            camera_stream.start()
            frames_by_exposure[exposure_us] = camera_stream.collect_n_frames(n_frames_per_level)
    finally:
        camera_stream.stop()
        camera_stream.exposure_us = original_exposure_us
        camera_stream.start()

    result, record = build_conversion_gain(frames_by_exposure)
    save_conversion_gain(path, result, record)
    logger.info(
        "conversion gain calibration complete: %d levels x %d frames -> %s (gain=%.4f e-/ADU)",
        n_levels, n_frames_per_level, path, result.gain_e_per_adu,
    )
    return record


__all__ = [
    "run_baseline_calibration",
    "capture_dark_frames", "capture_illuminated_frames", "finish_flat_field_calibration",
    "run_conversion_gain_calibration",
]
