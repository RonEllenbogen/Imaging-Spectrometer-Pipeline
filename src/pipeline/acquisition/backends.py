'''
This file defines the swappable backends that either connect (for real data), or don't connect to the camera (for synthetic data)
'''
# Imports
from typing import Protocol
import numpy as np

# Constants

# Classes

class CameraBackend(Protocol):

    '''
    Defines a structure both real and fake backends must satisfy
    '''

    def connect(self) -> None: ...
    def configure(self, exposure_us: float, gain_db: float) -> None: ...
    def grab_one(self, timeout_ms: int) -> np.ndarray: ...
    def close(self) -> None: ...

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

    def __innit__(self, shape=(1200,1920), noise_std=5.0, zeta_true=0.0): ... # Default parameters that should always be overrided with actual configurations
    def connect(self) -> None: ...
    def configure(self, exposure_us, gain_db) -> None: ...
    def grab_one(self, timeout_ms) -> np.ndarray: ...
    def close(self) -> None: ...

# Functions

#if __name__ == "__main__":
    