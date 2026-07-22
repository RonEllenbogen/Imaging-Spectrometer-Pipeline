'''
This file is in charge of orchestratring the camera-grabbing loop in the background, and ensuring that a given script can access the newest picture safely.
'''
# Imports
from .backends import CameraBackend
from .frame import FrameData

# Constants

# Classes

class CameraStream:

    '''
    Keeps a continous grab loop running independently of any caller, wraps a CameraBackend in a background thread.
    '''

    def __init__(self, exposure_us: float, gain_db: float, backend: CameraBackend | None = None, timeout_ms: int = 5000): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_latest_frame(self) -> FrameData | None: ...

    @property
    def is_running(self) -> bool: ...
    @property
    def fps(self) -> float: ...

    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...
    def _run(self) -> None: ...

# Functions

#if __name__ == "__main__":
    