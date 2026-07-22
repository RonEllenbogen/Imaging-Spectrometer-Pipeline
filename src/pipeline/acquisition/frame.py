'''
This file encapsulates the FrameData class, containing raw frame data + metadata
'''

# Imports
from dataclasses import dataclass
import numpy as np

# Constants

# Classes

@dataclass(frozen=True)
class FrameData:

    '''
    Contains raw frame data + metadata
    '''

    image: np.ndarray # Actual frame data
    timestamp: float # Timestamp when frame is recieved
    frame_id: int # Lets caller know whether two frames are copies of eachother or just similar

# Functions

#if __name__ == "__main__":
    