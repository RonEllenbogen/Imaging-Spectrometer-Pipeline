'''
This file defines the swappable backends that either connect (for real data), or don't connect to the camera (for synthetic data)
'''
# Imports
from typing import Protocol
import numpy as np

from .exceptions import CameraConfigurationError, CameraTimeoutError
from .pixel_formats import PIXEL_FORMAT_INFO, max_value_for_pixel_format, dtype_for_pixel_format

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

        Parameters
        ----------
        exposure_us
            The exposure time in microseconds to set on the camera.
        gain_db
            The gain in decibels to set on the camera.
        pixel_format
            The pixel format to set on the camera. Must be one of the keys in PIXEL_FORMAT_DTYPES.
        '''
        
        ...

    def grab_one(self, timeout_ms: int) -> np.ndarray:
        
        '''
        Block until a frame is available, then return it.

        Parameters
        ----------
        timeout_ms
            The maximum time to wait for a frame in milliseconds. If no frame is available within this time, a CameraTimeoutError is raised.
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

    def __init__(
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
        self.shape = shape
        self.centroid0_px = centroid0_px
        self.slope_px_per_col = slope_px_per_col
        self.beam_sigma_px = beam_sigma_px
        self.peak_counts = peak_counts
        self.noise_std = noise_std
        self.timeout_probability = timeout_probability
        self._rng = np.random.default_rng(seed)

        self._connected = False
        self._dtype: np.dtype | None = None

    def connect(self) -> None:

        '''
        Pretend to open the device
        '''

        self._connected = True

    def configure(self, exposure_us: float, gain_db: float, pixel_format: str) -> None:

        '''
        Pretend to apply acquisition settings before grabbing begins.

        Parameters
        ----------
        exposure_us
            The exposure time in microseconds to set on the camera.
        gain_db
            The gain in decibels to set on the camera.
        pixel_format
            The pixel format to set on the camera. Must be one of the keys in PIXEL_FORMAT_DTYPES.
        '''

        if pixel_format not in PIXEL_FORMAT_INFO:
            raise CameraConfigurationError(
                "pixel_format", pixel_format,
                f"must be one of {list(PIXEL_FORMAT_INFO)}",
            )
        self._dtype = dtype_for_pixel_format(pixel_format)
        self._max_value = max_value_for_pixel_format(pixel_format)
        # Exposure_us / gain_db accepted for interface compatibility, not currently used to scale the synthetic signal.

    def grab_one(self, timeout_ms: int) -> np.ndarray:

        '''
        Builds synthetic frame with a known injected chirp and noise, no hardware involved.

        Parameters
        ----------
        timeout_ms
            The maximum time to wait for a frame in milliseconds. If no frame is available within this time, a CameraTimeoutError is raised.
        
        Returns
        -------
        frame.astype(self._dtype)
            The synthetic frame as a numpy array with the configured pixel format.
        '''

        # Guard against being called before connect() and configure()
        if not self._connected or self._dtype is None:
            raise RuntimeError("grab_one() called before connect()/configure()")

        # Simulate a timeout with the specified probability
        if self._rng.random() < self.timeout_probability:
            raise CameraTimeoutError(timeout_ms)

        # Build the synthetic frame
        rows, cols = self.shape
        row_axis = np.arange(rows).reshape(-1, 1)   # Spatial axis, as a column vector
        col_axis = np.arange(cols).reshape(1, -1)   # Spectral axis, as a row vector
        x0 = self.centroid0_px + self.slope_px_per_col * col_axis

        # Generate a 2D Gaussian signal with the specified parameters, then add Gaussian noise and clip to valid range
        signal = self.peak_counts * np.exp(
            -((row_axis - x0) ** 2) / (2 * self.beam_sigma_px ** 2)
        )
        noise = self._rng.normal(0.0, self.noise_std, size=self.shape)

        # Build frame as signal + noise
        frame = np.clip(signal + noise, 0, self._max_value)
        return frame.astype(self._dtype)

    def close(self) -> None:
        
        '''
        Pretend to release the device and any resources it holds.
        '''

        self._connected = False
        self._dtype = None

# Functions

#if __name__ == "__main__":
    