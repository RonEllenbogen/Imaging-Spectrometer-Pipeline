"""
Content-sanity check for raw frames. Structural validity (shape, dtype)
is already guaranteed by FrameData's own construction -- the only place
FrameData is ever created is CameraStream._run(), so a frame that exists
at all has already passed that check before preprocessing ever sees it.
This module checks something FrameData's constructor has no way to know:
whether the pixel values represent anything meaningful at all.
"""

# Imoprts

from pipeline.acquisition import FrameData
from ..exceptions import NoSignalError

# Constants

# Classes

#Functions


def check_frame_sanity(frame: FrameData) -> None:

    '''
    Checks that a raw frame contains at least some signal. Raises if
    every pixel is exactly zero.

    Saturation is deliberately NOT checked here -- that's
    steps/saturation.py's job, later in the pipeline, checking the fully
    corrected frame rather than the raw one.

    Parameters
    ----------
    frame
        The raw frame to check, before any correction steps are applied.

    Returns
    -------
    None

    Raises
    ------
    NoSignalError
        If frame.image.max() == 0.
    '''

    if frame.image.max() == 0:
        raise NoSignalError(frame.frame_id)


__all__ = ["check_frame_sanity"]