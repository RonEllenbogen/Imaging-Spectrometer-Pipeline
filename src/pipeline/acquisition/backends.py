'''
This file defines the swappable backends that either connect (for real data), or don't connect to the camera (for synthetic data)
'''
# Imports
from typing import Protocol
import numpy as np
from pypylon import pylon, genicam

from .exceptions import CameraError, CameraConnectionError, CameraConfigurationError, CameraTimeoutError
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
    Real hardware backend, wrapping pypylon/pylon SDK calls for the Basler
    ace 2 a2A1920-51gmBAS. Satisfies CameraBackend structurally -- see the
    earlier discussion on why Protocol was chosen over inheritance.

    auto_exposure and auto_timeout are deliberately NOT part of the shared
    CameraBackend contract -- SyntheticBackend has no equivalent concept,
    so these live here as PylonBackend-specific constructor arguments only.
    '''

    def __init__(self, serial_number: str, auto_exposure: bool = False, auto_timeout: int = 5000):

        '''
        Stores configuration and initializes internal state. Does NOT
        connect to hardware -- that happens in connect(), matching the
        same "construction never touches hardware" principle as
        CameraStream and SyntheticBackend.

        Parameters
        ----------
        serial_number
            The serial number of the specific camera to connect to.
            Required, since multiple Basler devices may be present on
            the network.
        auto_exposure
            If True, configure() runs a one-time auto-exposure convergence
            (ExposureAuto="Once") instead of setting exposure_us directly.
        auto_timeout
            Timeout, in milliseconds, for each RetrieveResult() call made
            while waiting for auto-exposure to converge. Only relevant if
            auto_exposure is True.
        '''

        self.serial_number = serial_number
        self.auto_exposure = auto_exposure
        self.auto_timeout = auto_timeout
        self._camera: pylon.InstantCamera | None = None
        self._configured = False

    def connect(self) -> None:

        '''
        Finds the device matching serial_number among all enumerated
        devices, and opens it. Does NOT configure or start grabbing --
        that is configure()'s job.

        Returns
        -------
        None

        Raises
        ------
        CameraConnectionError
            If no device with a matching serial number is found, or if
            opening the device fails.
        '''

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()

        matching_device = next(
            (d for d in devices if d.GetSerialNumber() == self.serial_number), None
        )
        if matching_device is None:
            raise CameraConnectionError(
                f"no device found with serial number {self.serial_number!r}"
            )

        try:
            self._camera = pylon.InstantCamera(tl_factory.CreateDevice(matching_device))
            self._camera.Open()
        except genicam.GenericException as e:
            raise CameraConnectionError(
                f"failed to open device with serial number {self.serial_number!r}: {e}"
            ) from e

    def configure(self, exposure_us: float, gain_db: float, pixel_format: str) -> None:

        '''
        Applies pixel format and gain unconditionally, then either runs a
        one-time auto-exposure convergence or sets exposure_us directly,
        depending on self.auto_exposure. Ends with grabbing already
        started (GrabStrategy_LatestImageOnly) -- grab_one() assumes this
        is already true by the time it's called.

        Parameters
        ----------
        exposure_us
            Exposure time in microseconds. Ignored if self.auto_exposure
            is True -- the converged value is used instead.
        gain_db
            Sensor gain in decibels.
        pixel_format
            PixelFormat string, e.g. "Mono8". Validated against
            PIXEL_FORMAT_INFO before being sent to the camera.

        Returns
        -------
        None

        Raises
        ------
        CameraConfigurationError
            If pixel_format is not recognized, if the camera rejects
            pixel_format/gain_db/exposure_us, or if auto-exposure fails
            to converge within auto_timeout.
        RuntimeError
        If called before connect().
        '''

        if self._camera is None:
            raise RuntimeError("configure() called before connect()")
        camera = self._camera   # narrowed local -- Pylance now treats this as never None

        if pixel_format not in PIXEL_FORMAT_INFO:
            raise CameraConfigurationError(
                "pixel_format", pixel_format, f"must be one of {list(PIXEL_FORMAT_INFO)}"
            )

        try:
            self._camera.PixelFormat.SetValue(pixel_format)
            self._camera.Gain.SetValue(gain_db)
        except genicam.GenericException as e:
            raise CameraConfigurationError(
                "pixel_format/gain_db", (pixel_format, gain_db), str(e)
            ) from e

        if self.auto_exposure:
            self._converge_auto_exposure(self._camera)
        else:
            try:
                self._camera.ExposureAuto.SetValue("Off")
                self._camera.ExposureTime.SetValue(exposure_us)
            except genicam.GenericException as e:
                raise CameraConfigurationError("exposure_us", exposure_us, str(e)) from e
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def _converge_auto_exposure(self, camera: pylon.InstantCamera) -> None:

        '''
        Runs ExposureAuto="Once" to convergence, matching the working
        bring-up script's approach. Starts grabbing to feed the
        auto-exposure algorithm and deliberately leaves grabbing running
        afterward -- unlike a one-shot script, this needs to feed
        straight into a continuous grab_one() loop, not stop and restart.

        Private to PylonBackend -- not part of the shared CameraBackend
        contract, since SyntheticBackend has no equivalent concept.

        Returns
        -------
        None

        Raises
        ------
        CameraConfigurationError
            If convergence does not complete within auto_timeout, or if
            the camera rejects the ExposureAuto setting itself.
        '''

        try:
            camera.ExposureAuto.SetValue("Once")
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            while camera.ExposureAuto.GetValue() != "Off":
                warm_up_grab = camera.RetrieveResult(
                    self.auto_timeout, pylon.TimeoutHandling_ThrowException
                )
                warm_up_grab.Release()
        except genicam.TimeoutException as e:
            raise CameraConfigurationError(
                "auto_exposure", True, f"did not converge within {self.auto_timeout}ms"
            ) from e
        except genicam.GenericException as e:
            raise CameraConfigurationError("auto_exposure", True, str(e)) from e

    def grab_one(self, timeout_ms: int) -> np.ndarray:

        '''
        Retrieves the next frame from the already-running grab engine
        (started by configure()) and returns it as a numpy array.

        Parameters
        ----------
        timeout_ms
            Maximum time to wait for a frame, in milliseconds.

        Returns
        -------
        np.ndarray
            A copy of the grabbed frame -- never a view into a pylon
            buffer, which is reclaimed the moment Release() is called
            below.

        Raises
        ------
        CameraTimeoutError
            If no frame arrives within timeout_ms.
        CameraError
            If the grab completes without timing out but still fails
            (e.g. dropped packets, sensor fault).
        RuntimeError
            If called before connect() and configure(), or after close().
        '''

        if self._camera is None or not self._configured:
            raise RuntimeError("grab_one() called before connect()/configure()")
        camera = self._camera

        try:
            grab_result = self._camera.RetrieveResult(
                timeout_ms, pylon.TimeoutHandling_ThrowException
            )
        except genicam.TimeoutException as e:
            raise CameraTimeoutError(timeout_ms) from e

        if not grab_result.GrabSucceeded():
            error_description = grab_result.ErrorDescription
            grab_result.Release()
            raise CameraError(f"grab failed: {error_description}")

        frame = grab_result.Array.copy()
        grab_result.Release()
        return frame

    def close(self) -> None:

        '''
        Stops grabbing (if active) and closes the device (if open). Safe
        to call in any state, including if connect() was never called or
        failed partway through -- matching the guarantee CameraStream's
        cleanup logic relies on. Also resets internal state, so a grab_one() call after close()
        correctly raises RuntimeError via the same guard used for "never
        connected", rather than attempting to use a closed device.

        Returns
        -------
        None
        '''

        if self._camera is not None:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
            if self._camera.IsOpen():
                self._camera.Close()

        self._camera = None
        self._configured = False

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
    