"""
The float-image counterpart to acquisition's FrameData, for data that has
passed through baseline subtraction or any later correction step. Those
corrections require float precision (and the ability to go negative
before clipping) in a way FrameData's strict integer CANONICAL_DTYPE
can't represent -- see the baseline.py design discussion. Shape stays
validated against CANONICAL_SHAPE throughout preprocessing; only the
integer-dtype requirement is relaxed.
"""

# Imports

import time
from dataclasses import dataclass

import numpy as np

from pipeline.acquisition import CANONICAL_SHAPE

# Constants

# Classes

@dataclass(frozen=True, slots=True, eq=False)
class ProcessedFrame:

    '''
    A frame that has been through at least one preprocessing correction
    step. Same shape and preserved acquisition metadata as the FrameData
    it originated from, but holding a float image rather than the raw
    integer dtype -- see the module docstring for why.

    Parameters
    ----------
    image
        Float array, shape CANONICAL_SHAPE.
    frame_id
        Preserved from the originating FrameData.
    timestamp
        Preserved from the originating FrameData (time.monotonic()).
    exposure_us
        Preserved from the originating FrameData.
    gain_db
        Preserved from the originating FrameData.
    '''

    image: np.ndarray
    frame_id: int
    timestamp: float
    exposure_us: float
    gain_db: float

    def __post_init__(self) -> None:
        if self.image.shape != CANONICAL_SHAPE:
            raise ValueError(
                f"ProcessedFrame.image must have shape {CANONICAL_SHAPE}, got {self.image.shape}"
            )
        if not np.issubdtype(self.image.dtype, np.floating):
            raise ValueError(
                f"ProcessedFrame.image must be a floating-point dtype, got {self.image.dtype}"
            )
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative, got {self.frame_id}")
        if self.exposure_us <= 0:
            raise ValueError(f"exposure_us must be positive, got {self.exposure_us}")
        if not np.isfinite(self.gain_db):
            raise ValueError(f"gain_db must be finite, got {self.gain_db}")
        # Same immutability guarantee as FrameData -- frozen=True alone
        # doesn't stop in-place mutation of the array's contents.
        self.image.flags.writeable = False

    @property
    def age_seconds(self) -> float:
        '''Seconds elapsed since the originating frame was captured.'''
        return time.monotonic() - self.timestamp

# Functions

__all__ = ["ProcessedFrame"]