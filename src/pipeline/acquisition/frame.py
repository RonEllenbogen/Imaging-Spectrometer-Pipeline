'''
This file encapsulates the FrameData class, containing a single frame's raw data + metadata
'''

# Imports
from dataclasses import dataclass
import numpy as np
import time

from ..utils.helpers import load_config
from .pixel_formats import dtype_for_pixel_format

# Constants
config = load_config("configs/default.yaml")
CANONICAL_DTYPE = dtype_for_pixel_format(config["camera"]["pixel_format"]).type
CANONICAL_NDIM = 2


# Classes

@dataclass(frozen=True, slots=True, eq=False)
class FrameData:

    '''
    A single captured frame plus the metadata needed to reason about it downstream.
    '''

    image: np.ndarray # Actual frame data
    timestamp: float # Timestamp when frame is recieved
    frame_id: int # Lets caller know whether two frames are copies of eachother or just similar

    def __post_init__(self) -> None:

        '''
        Post-initialization checks to ensure that the FrameData object is valid.
        '''

        if self.image.ndim != CANONICAL_NDIM:
            raise ValueError(
                f"FrameData.image must be {CANONICAL_NDIM}D, got shape {self.image.shape}"
            )
        if self.image.dtype != CANONICAL_DTYPE:
            raise ValueError(
                f"FrameData.image must be dtype {CANONICAL_DTYPE}, got {self.image.dtype}"
            )
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative, got {self.frame_id}")
        # frozen=True only stops reassigning `self.image` itself -- it does NOT
        # stop someone mutating the array's contents in place. Lock it explicitly.
        self.image.flags.writeable = False

    @property
    def age_seconds(self) -> float:

        """
        Seconds elapsed since this frame was captured.
        """

        return time.monotonic() - self.timestamp
    
__all__ = ["FrameData", "CANONICAL_DTYPE", "CANONICAL_NDIM"]

# Functions

#if __name__ == "__main__":