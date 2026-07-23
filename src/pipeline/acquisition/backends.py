'''
This file defines the swappable backends that either connect (for real data), or don't connect to the camera (for synthetic data)
'''
# Imports
from typing import Protocol
import numpy as np

from .exceptions import CameraConfigurationError, CameraTimeoutError
from .pixel_formats import PIXEL_FORMAT_DTYPES

# Constants

# Classes

class CameraBackend(Protocol):

    '''
    Structural interface every camera backend must match.
    '''

    def connect(self) -> None:

        '''
        Open the device and prepare it to grab frames.
        '''

        ...

    def configure(self, exposure_us: float, gain_db: float, pixel_format: str) -> None:

        '''
        Apply acquisition settings before grabbing begins.
        '''
        
        ...

    def grab_one(self, timeout_ms: int) -> np.ndarray:
        
        '''
        Block until a frame is available, then return it.
        '''
        
        ...

    def close(self) -> None:

        '''
        Release the device and any resources it holds.
        '''
        
        ...

class PylonBackend:

    '''
    Real backend, wrapping pypylon calls. Only place in the codebase that talks to the actual SDK
    '''

    def __init__(self): ...
    def connect(self) -> None: ...
    def configure(self, exposure_us, gain_db) -> None: ...
    def grab_one(self, timeout_ms) -> np.ndarray: ...
    def close(self) -> None: ...

class SyntheticBackend:

    '''
    Generates fake frames with a known injected chirp and noise, no hardware involved.
    '''

    def __innit__(
        self,
        shape: tuple[int, int] = (1200, 1920),
        centroid0_px: float = 600.0,
        slope_px_per_col: float = 0.0,
        beam_sigma_px: float = 15.0,
        peak_counts: float = 3000.0,
        noise_std: float = 5.0,
        timeout_probability: float = 0.0,
        seed: int | None = None,
    ): # Default parameters

    def connect(self) -> None: ...
    def configure(self, exposure_us, gain_db) -> None: ...
    def grab_one(self, timeout_ms) -> np.ndarray: ...
    def close(self) -> None: ...

# Functions

#if __name__ == "__main__":
    