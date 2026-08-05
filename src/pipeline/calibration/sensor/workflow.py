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
finish_flat_field_calibration(), called once the caller has both. This
mirrors how spatial/'s SpatialCalibrationSession is designed as a
multi-step, caller-paced interaction rather than one blocking call, for
the same reason (see docs/project_handover.md §5/§6).

build_bad_pixel_map() has no workflow function here -- it's derived
purely from an already-built flat field, with no CameraStream involved,
so there's no acquisition step for this module to add value around.
Callers chain build_bad_pixel_map() + save_bad_pixel_map() directly.
"""

# Imports

import logging
from pathlib import Path

from pipeline.acquisition import CameraStream, FrameData

from .baseline import build_baseline, save_baseline
from .flat_field import build_flat_field, save_flat_field
from .metadata import CalibrationRecord

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
        Propagated from build_baseline() (e.g. n_frames < 1).
    RuntimeError, CameraError
        Propagated from CameraStream.collect_n_frames() if camera_stream
        isn't running, or dies while collecting.
    '''

    frames = camera_stream.collect_n_frames(n_frames)
    baseline, record = build_baseline(frames)
    save_baseline(path, baseline, record)
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


__all__ = [
    "run_baseline_calibration",
    "capture_dark_frames", "capture_illuminated_frames", "finish_flat_field_calibration",
]
