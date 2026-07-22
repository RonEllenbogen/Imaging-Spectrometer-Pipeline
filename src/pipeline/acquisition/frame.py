'''
This file encapsulates the FrameData class, containing raw frame data + metadata
'''

# Imports
from dataclasses import dataclass
import numpy as np
import time

from pipeline.utils.helpers import load_config

# Constants
config = load_config("configs/default.yaml")
dtype_config = config["canonical_dtype"]["serial_number"]
CANONICAL_DTYPE = np.dtype(dtype_config).type
CANONICAL_NDIM = 2


# Classes

@dataclass(frozen=True, slots=True, eq=False)
class FrameData:

    '''
    Contains raw frame data + metadata
    '''

    image: np.ndarray # Actual frame data
    timestamp: float # Timestamp when frame is recieved
    frame_id: int # Lets caller know whether two frames are copies of eachother or just similar

# Functions

#if __name__ == "__main__":
    