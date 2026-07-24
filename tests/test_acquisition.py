'''
This file is a test suite for CameraBackend implementations (SyntheticBackend, PylonBackend) and the shared contract both must satisfy.
'''
# Imports
import os
import numpy as np
import pytest
import time

from pipeline.acquisition import (
    SyntheticBackend, PylonBackend, FrameData, CameraStream,
    CameraConfigurationError, CameraTimeoutError, CameraConnectionError,
    CANONICAL_SHAPE, CANONICAL_DTYPE,
)
from pipeline.utils.helpers import load_config

# Constants and Functions (ordering is important for pytest parametrize)

# --- Environment / hardware gating ---
HARDWARE_ENV_VAR = "SPECTROMETER_HARDWARE_TESTS"

# --- `backend` / `raw_backend` fixture configuration ---
# Deliberately independent of configs/default.yaml -- see reasoning below.
FIXTURE_SEED = 42
FIXTURE_EXPOSURE_US = 2000
FIXTURE_GAIN_DB = 0.0
FIXTURE_PIXEL_FORMAT = "Mono8"

# --- Grab timeouts ---
# Deliberately much shorter than configs/default.yaml's camera.timeout (100000ms).
# Production's long timeout tolerates occasional network hiccups during a live
# experiment; this suite's goal is the opposite -- fail fast if something is
# genuinely broken, rather than let a hung grab stall the whole run.
TEST_GRAB_TIMEOUT_MS = 5000

# --- test_configure_rejects_unknown_pixel_format ---
INVALID_PIXEL_FORMAT = "Bogus99"

# --- test_injected_slope_is_recoverable / test_zero_slope_recovers_flat_centroid ---
SLOPE_TEST_SEED = 3
SLOPE_TEST_NOISE_STD = 2.0
INJECTED_SLOPE_PX_PER_COL = 0.02
SLOPE_RECOVERY_TOLERANCE_PX = 0.005   # unverified guess -- tighten once run for real

# --- test_same_seed_produces_identical_frames ---
DETERMINISM_TEST_SEED = 7

# --- test_timeout_probability_one_always_raises ---
TIMEOUT_TEST_SEED = 1
TIMEOUT_TEST_TIMEOUT_MS = 250

# --- test_frame_values_respect_bit_depth_ceiling ---
# True per-format ceilings, independent of pixel_formats.py's own mapping --
# these are physical facts (bit depth), used here as the test's ground truth
# rather than imported, so the test isn't just checking the module against itself.
BIT_DEPTH_TEST_SEED = 1
PIXEL_FORMAT_TRUE_MAX = {
    "Mono8": 255,
    "Mono10": 1023,
    "Mono12": 4095,
    "Mono16": 65535,
}

# --- test_successive_grabs_are_not_identical ---
SUCCESSIVE_GRABS_SEED = 11

# --- TestPylonBackendOnly ---
# Loaded once at collection time -- harmless if HARDWARE_AVAILABLE is False,
# since it's just reading a value nothing then uses.
SERIAL_NUMBER = load_config("configs/default.yaml")["camera"]["serial_number"]

INVALID_SERIAL_NUMBER = "00000000"  # placeholder, should not match any real device
EXPECTED_MODEL_NAME_SUBSTRING = "a2A1920"

AUTO_EXPOSURE_TEST_TIMEOUT_MS = 10000  # generous -- convergence speed depends on lighting

# Both unverified guesses -- tighten once you've seen real GetValue() readback
# against these settings; the camera may quantize to its nearest achievable step.
GAIN_TOLERANCE_DB = 0.5
EXPOSURE_TOLERANCE_REL = 0.05  # 5% relative tolerance

SUSTAINED_GRAB_DURATION_S = 3.0
MIN_ACCEPTABLE_FPS = 5.0  # deliberately loose -- a sanity floor, not a performance target

def _make_synthetic():

    '''
    Initialises a synthetic backend.

    Returns
    -------
    SyntheticBackend
        A synthetic backend.
    '''

    return SyntheticBackend(seed=FIXTURE_SEED)


def _make_pylon():

    '''
    Initialises a Pylon backend.

    Returns
    -------
    PylonBackend
        A Pylon backend.
    '''

    return PylonBackend(serial_number=SERIAL_NUMBER)

HARDWARE_AVAILABLE = os.environ.get(HARDWARE_ENV_VAR) == "1"
# If I want to run tests on PylonBackend with camera connected and powered on, must run export SPECTROMETER_HARDWARE_TESTS=1
# in terminal before running pytest. Otherwise, tests will be skipped.

BACKEND_FACTORIES = [
    pytest.param(_make_synthetic, id="synthetic"),
    pytest.param(
        _make_pylon, id="pylon",
        marks=pytest.mark.skipif(not HARDWARE_AVAILABLE, reason="camera not connected"),
    ),
]

@pytest.fixture(params=BACKEND_FACTORIES)
def raw_backend(request):
    """Unconnected, unconfigured. For testing lifecycle ordering itself."""
    return request.param()

@pytest.fixture(params=BACKEND_FACTORIES)
def backend(request):

    """
    Already connected and configured -- for tests that need a working backend.
    """

    b = request.param()
    b.connect()
    b.configure(
        exposure_us=FIXTURE_EXPOSURE_US,
        gain_db=FIXTURE_GAIN_DB,
        pixel_format=FIXTURE_PIXEL_FORMAT,
    )
    yield b
    b.close()

def _weighted_centroid_per_column(frame: np.ndarray) -> np.ndarray:

    '''
    Minimal standalone intensity-weighted centroid calculation, for test
    purposes only. NOT a substitute for analysis/computations.py, which
    will also compute per-column uncertainty and handle low-signal columns.

    Parameters
    ----------
    frame
        A 2D array with the spatial axis as rows and the spectral axis as
        columns, matching CANONICAL_SHAPE.

    Returns
    -------
    np.ndarray
        1D array of length frame.shape[1], the intensity-weighted centroid
        (in row/spatial-pixel units) for each column.
    '''

    # Row indices as a column vector, broadcasting against every column
    rows = np.arange(frame.shape[0]).reshape(-1, 1)
    weights = frame.astype(float)

    # Sum of intensity per column, used as the normalizing denominator
    col_sums = weights.sum(axis=0)

    return (rows * weights).sum(axis=0) / col_sums

# Classes

class TestBackendContract:

    """
    Behavior every CameraBackend must satisfy. Runs against both
    SyntheticBackend and PylonBackend via the `backend` fixture.
    """

    def test_grab_one_shape_matches_canonical(self, backend):

        '''
        Checks that a frame grabbed from any conforming backend has the canonical shape.

        Parameters
        ----------
        backend
            A connected and configured CameraBackend instance (SyntheticBackend or
            PylonBackend), supplied by the `backend` fixture.
        '''

        # Grab a single frame from whichever backend is under test on this run
        frame = backend.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

        # Confirm the frame matches the canonical (spatial, spectral) shape
        assert frame.shape == CANONICAL_SHAPE

    def test_grab_one_dtype_matches_canonical(self, backend):

        '''
        Checks that a frame grabbed from any conforming backend has the canonical dtype.

        Parameters
        ----------
        backend
            A connected and configured CameraBackend instance (SyntheticBackend or
            PylonBackend), supplied by the `backend` fixture.
        '''

        # Grab a single frame from whichever backend is under test on this run
        frame = backend.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

        # Confirm the frame's pixel type matches the dtype the configured pixel_format implies
        assert frame.dtype == CANONICAL_DTYPE

    def test_configure_rejects_unknown_pixel_format(self, backend):

        '''
        Checks that configure() rejects a pixel_format string it doesn't recognize.

        Parameters
        ----------
        backend
            A connected and configured CameraBackend instance, supplied by the
            `backend` fixture. Already configured once during fixture setup;
            this test calls configure() again with a deliberately invalid value.
        '''

        # Attempt to configure with a pixel_format absent from PIXEL_FORMAT_DTYPES
        with pytest.raises(CameraConfigurationError):
            backend.configure(exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB, pixel_format=INVALID_PIXEL_FORMAT)

    def test_frame_constructs_valid_framedata(self, backend):

        '''
        Checks that a frame grabbed from any conforming backend can construct a valid FrameData.

        Parameters
        ----------
        backend
            A connected and configured CameraBackend instance, supplied by the
            `backend` fixture.
        '''

        # Grab a single frame from whichever backend is under test on this run
        frame = backend.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

        # FrameData's own __post_init__ validates shape and dtype; must not raise
        # for a frame produced by a genuinely conforming backend
        fd = FrameData(image=frame, timestamp=0.0, frame_id=0)

        # Double check the resulting FrameData carries the canonical dtype
        assert fd.image.dtype == CANONICAL_DTYPE

class TestBackendLifecycle:

    """
    Ordering and repeated-call guarantees every backend must honor.
    Also runs against both backends, since these are contract guarantees
    too, just ones that need an unconnected or freshly-closed instance.
    """

    def test_grab_one_before_connect_raises(self, raw_backend):

        '''
        Checks that grab_one() raises if called before connect() and configure().

        Parameters
        ----------
        raw_backend
            An unconnected, unconfigured CameraBackend instance (SyntheticBackend
            or PylonBackend), supplied by the `raw_backend` fixture.
        '''

        # connect() and configure() have not been called -- grab_one() must refuse
        with pytest.raises(RuntimeError):
            raw_backend.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

    def test_grab_one_after_connect_but_before_configure_raises(self, raw_backend):

        '''
        Checks that grab_one() raises if called after connect() but before configure().

        Parameters
        ----------
        raw_backend
            An unconnected, unconfigured CameraBackend instance, supplied by the
            `raw_backend` fixture. This test connects it directly, deliberately
            skipping configure().
        '''

        # Connect, but deliberately skip configure() -- grab_one() must still refuse
        raw_backend.connect()

        with pytest.raises(RuntimeError):
            raw_backend.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

    def test_close_before_connect_is_safe(self, raw_backend):

        '''
        Checks that close() does not raise when called on a backend that was
        never connected.

        Parameters
        ----------
        raw_backend
            An unconnected, unconfigured CameraBackend instance, supplied by the
            `raw_backend` fixture.
        '''

        # close() must be safe to call regardless of prior lifecycle state --
        # this is the guarantee CameraStream's cleanup logic relies on
        raw_backend.close()

    def test_close_is_idempotent(self, backend):

        '''
        Checks that calling close() a second time does not raise.

        Parameters
        ----------
        backend
            A connected and configured CameraBackend instance, supplied by the
            `backend` fixture.
        '''

        # First close, as part of the guarantee under test
        backend.close()

        # Second close on an already-closed backend must also not raise
        backend.close()

    def test_grab_one_after_close_raises(self, backend):

        '''
        Checks that grab_one() raises after the backend has been closed.

        Parameters
        ----------
        backend
            A connected and configured CameraBackend instance, supplied by the
            `backend` fixture.
        '''

        # Close the backend, then attempt to grab -- must refuse, same as an
        # unconnected/unconfigured backend
        backend.close()

        with pytest.raises(RuntimeError):
            backend.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

class TestSyntheticBackendOnly:

    """
    Behavior with no real-hardware equivalent, injecting a known
    ground truth, forcing determinism, simulating failures on demand.
    Each test builds its own SyntheticBackend rather than using the
    shared fixture, since each needs different constructor arguments.
    """

    def test_injected_slope_is_recoverable(self):

        '''
        Checks that a linear fit to the per-column centroid recovers the
        slope that was injected into the synthetic frame.
        '''

        # Build a backend with a known, non-zero injected centroid slope
        b = SyntheticBackend(
            seed=SLOPE_TEST_SEED,
            slope_px_per_col=INJECTED_SLOPE_PX_PER_COL,
            noise_std=SLOPE_TEST_NOISE_STD,
        )
        b.connect()
        b.configure(
            exposure_us=FIXTURE_EXPOSURE_US,
            gain_db=FIXTURE_GAIN_DB,
            pixel_format=FIXTURE_PIXEL_FORMAT,
        )
        frame = b.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

        # Recompute centroids independently of SyntheticBackend's own generation
        # logic, then fit a line the same way analysis/ eventually will
        cols = np.arange(frame.shape[1])
        centroids = _weighted_centroid_per_column(frame)
        fitted_slope, _ = np.polyfit(cols, centroids, 1)

        assert fitted_slope == pytest.approx(
            INJECTED_SLOPE_PX_PER_COL, abs=SLOPE_RECOVERY_TOLERANCE_PX
        )

    def test_zero_slope_recovers_flat_centroid(self):

        '''
        Checks that a linear fit to the per-column centroid recovers a
        slope of zero when no chirp was injected, the null case.
        '''

        # Same setup as the injected-slope test, but with slope_px_per_col
        # left at its default of 0.0
        b = SyntheticBackend(seed=SLOPE_TEST_SEED, noise_std=SLOPE_TEST_NOISE_STD)
        b.connect()
        b.configure(
            exposure_us=FIXTURE_EXPOSURE_US,
            gain_db=FIXTURE_GAIN_DB,
            pixel_format=FIXTURE_PIXEL_FORMAT,
        )
        frame = b.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

        cols = np.arange(frame.shape[1])
        centroids = _weighted_centroid_per_column(frame)
        fitted_slope, _ = np.polyfit(cols, centroids, 1)

        assert fitted_slope == pytest.approx(0.0, abs=SLOPE_RECOVERY_TOLERANCE_PX)

    def test_same_seed_produces_identical_frames(self):

        '''
        Checks that two separately constructed backends with the same seed
        produce bit-for-bit identical frames.
        '''

        # Two independent instances, same seed, should produce the same
        # "random" noise realization on their first grab
        b1 = SyntheticBackend(seed=DETERMINISM_TEST_SEED)
        b2 = SyntheticBackend(seed=DETERMINISM_TEST_SEED)

        for b in (b1, b2):
            b.connect()
            b.configure(
                exposure_us=FIXTURE_EXPOSURE_US,
                gain_db=FIXTURE_GAIN_DB,
                pixel_format=FIXTURE_PIXEL_FORMAT,
            )

        frame1 = b1.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)
        frame2 = b2.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

        assert np.array_equal(frame1, frame2)

    def test_successive_grabs_are_not_identical(self):

        '''
        Checks that two successive grabs from the same backend instance
        differ, since the RNG advances its internal state between calls.
        '''

        # A single instance, two consecutive grabs -- the shared RNG should
        # produce a different noise realization each time
        b = SyntheticBackend(seed=SUCCESSIVE_GRABS_SEED)
        b.connect()
        b.configure(
            exposure_us=FIXTURE_EXPOSURE_US,
            gain_db=FIXTURE_GAIN_DB,
            pixel_format=FIXTURE_PIXEL_FORMAT,
        )

        frame1 = b.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)
        frame2 = b.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

        assert not np.array_equal(frame1, frame2)

    def test_timeout_probability_one_always_raises(self):

        '''
        Checks that grab_one() raises CameraTimeoutError, carrying the
        requested timeout value, when timeout_probability is forced to 1.0.
        '''

        # timeout_probability=1.0 forces every grab_one() call to simulate a timeout
        b = SyntheticBackend(seed=TIMEOUT_TEST_SEED, timeout_probability=1.0)
        b.connect()
        b.configure(
            exposure_us=FIXTURE_EXPOSURE_US,
            gain_db=FIXTURE_GAIN_DB,
            pixel_format=FIXTURE_PIXEL_FORMAT,
        )

        with pytest.raises(CameraTimeoutError) as exc_info:
            b.grab_one(timeout_ms=TIMEOUT_TEST_TIMEOUT_MS)

        assert exc_info.value.timeout_ms == TIMEOUT_TEST_TIMEOUT_MS

    @pytest.mark.parametrize(
        "pixel_format,expected_dtype,true_max",
        [
            ("Mono8", np.uint8, PIXEL_FORMAT_TRUE_MAX["Mono8"]),
            ("Mono10", np.uint16, PIXEL_FORMAT_TRUE_MAX["Mono10"]),
            ("Mono12", np.uint16, PIXEL_FORMAT_TRUE_MAX["Mono12"]),
            ("Mono16", np.uint16, PIXEL_FORMAT_TRUE_MAX["Mono16"]),
        ],
    )
    def test_frame_values_respect_bit_depth_ceiling(self, pixel_format, expected_dtype, true_max):

        '''
        Checks that a saturating synthetic frame never exceeds the true
        bit-depth ceiling of the configured pixel format -- not just the
        numpy container's ceiling.

        Parameters
        ----------
        pixel_format
            The PixelFormat string under test for this parametrized run.
        expected_dtype
            The numpy dtype the frame should be stored as for this format.
        true_max
            The true maximum pixel value the format allows (e.g. 4095 for
            Mono12), independent of the numpy container's own maximum.
        '''

        # Force saturation with a peak well above every format's true ceiling,
        # and disable noise so the only source of values above true_max would
        # be a clipping bug, not a random fluctuation
        b = SyntheticBackend(
            seed=BIT_DEPTH_TEST_SEED,
            peak_counts=true_max * 2,
            noise_std=0.0,
        )
        b.connect()
        b.configure(
            exposure_us=FIXTURE_EXPOSURE_US,
            gain_db=FIXTURE_GAIN_DB,
            pixel_format=pixel_format,
        )
        frame = b.grab_one(timeout_ms=TEST_GRAB_TIMEOUT_MS)

        assert frame.dtype == expected_dtype
        assert frame.max() <= true_max

class TestPylonBackendOnly:
    """
    Real-hardware-specific checks: device discovery, whether SDK calls
    actually take effect on the physical camera, and the
    close()-then-reconnect guarantee that matters specifically because a
    GigE device only accepts one connection at a time.
    """

    pytestmark = pytest.mark.skipif(not HARDWARE_AVAILABLE, reason="camera not connected")

    def test_connect_with_wrong_serial_number_raises(self):

        '''
        Checks that connect() raises CameraConnectionError, with the
        offending serial number in the message, when no device matches.
        '''

        backend = PylonBackend(serial_number=INVALID_SERIAL_NUMBER)
        with pytest.raises(CameraConnectionError) as exc_info:
            backend.connect()
        assert INVALID_SERIAL_NUMBER in str(exc_info.value)

    def test_connect_with_correct_serial_number_succeeds(self):

        '''
        Checks that connect() succeeds against the real configured
        camera, and that the connected device is genuinely the expected
        model -- not just any device that happened to match.
        '''

        backend = PylonBackend(serial_number=SERIAL_NUMBER)
        backend.connect()
        assert backend._camera is not None
        camera = backend._camera
        try:
            model_name = camera.GetDeviceInfo().GetModelName()
            assert EXPECTED_MODEL_NAME_SUBSTRING in model_name
        finally:
            backend.close()

    def test_pixel_format_actually_applied_on_device(self):

        '''
        Checks that configure() genuinely sets PixelFormat on the real
        camera, not just that PylonBackend believes it did.
        '''

        backend = PylonBackend(serial_number=SERIAL_NUMBER)
        backend.connect()
        try:
            backend.configure(
                exposure_us=FIXTURE_EXPOSURE_US,
                gain_db=FIXTURE_GAIN_DB,
                pixel_format=FIXTURE_PIXEL_FORMAT,
            )
            assert backend._camera is not None
            assert backend._camera.PixelFormat.GetValue() == FIXTURE_PIXEL_FORMAT
        finally:
            backend.close()

    def test_gain_actually_applied_on_device(self):

        '''
        Checks that configure() genuinely sets Gain on the real camera.
        '''

        backend = PylonBackend(serial_number=SERIAL_NUMBER)
        backend.connect()
        try:
            backend.configure(
                exposure_us=FIXTURE_EXPOSURE_US,
                gain_db=FIXTURE_GAIN_DB,
                pixel_format=FIXTURE_PIXEL_FORMAT,
            )
            assert backend._camera is not None
            actual_gain = backend._camera.Gain.GetValue()
            assert actual_gain == pytest.approx(FIXTURE_GAIN_DB, abs=GAIN_TOLERANCE_DB)
        finally:
            backend.close()

    def test_manual_exposure_actually_applied_on_device(self):

        '''
        Checks that configure() with auto_exposure=False genuinely sets
        ExposureTime on the real camera, allowing for the camera
        quantizing to its nearest achievable value.
        '''

        backend = PylonBackend(serial_number=SERIAL_NUMBER, auto_exposure=False)
        backend.connect()
        try:
            backend.configure(
                exposure_us=FIXTURE_EXPOSURE_US,
                gain_db=FIXTURE_GAIN_DB,
                pixel_format=FIXTURE_PIXEL_FORMAT,
            )
            assert backend._camera is not None
            actual_exposure = backend._camera.ExposureTime.GetValue()
            assert actual_exposure == pytest.approx(FIXTURE_EXPOSURE_US, rel=EXPOSURE_TOLERANCE_REL)
        finally:
            backend.close()

    def test_auto_exposure_converges(self):

        '''
        Checks that configure() with auto_exposure=True runs convergence
        to completion -- ExposureAuto settles back to "Off" and
        ExposureTime ends up at some genuine positive value, rather than
        configure() returning early or hanging.
        '''

        backend = PylonBackend(
            serial_number=SERIAL_NUMBER, auto_exposure=True, auto_timeout=AUTO_EXPOSURE_TEST_TIMEOUT_MS
        )
        backend.connect()
        try:
            backend.configure(
                exposure_us=FIXTURE_EXPOSURE_US,  # ignored when auto_exposure=True
                gain_db=FIXTURE_GAIN_DB,
                pixel_format=FIXTURE_PIXEL_FORMAT,
            )
            assert backend._camera is not None
            assert backend._camera.ExposureAuto.GetValue() == "Off"
            assert backend._camera.ExposureTime.GetValue() > 0
        finally:
            backend.close()

    def test_close_releases_connection_for_reuse(self):

        '''
        Regression test for the "camera left open, refuses to reconnect
        until power-cycled" failure mode flagged during hardware bring-up.
        Opens, closes, then opens again with a completely separate
        PylonBackend instance -- the second connect() only succeeds if
        the first close() genuinely released the device back to the
        network, not just to this one Python object.
        '''

        first = PylonBackend(serial_number=SERIAL_NUMBER)
        first.connect()
        first.close()

        second = PylonBackend(serial_number=SERIAL_NUMBER)
        try:
            second.connect()  # must not raise
        finally:
            second.close()

    def test_sustained_grab_achieves_reasonable_frame_rate(self):

        '''
        Runs a real CameraStream (not just a single grab_one() call) for
        a few seconds and checks the achieved fps clears a loose sanity
        floor -- a genuine hardware acceptance check that a single frame
        grab can't reveal, since it says nothing about sustained
        throughput over the network.
        '''

        stream = CameraStream(
            exposure_us=FIXTURE_EXPOSURE_US,
            gain_db=FIXTURE_GAIN_DB,
            pixel_format=FIXTURE_PIXEL_FORMAT,
            timeout_ms=TEST_GRAB_TIMEOUT_MS,
            backend=PylonBackend(serial_number=SERIAL_NUMBER),
        )
        stream.start()
        try:
            time.sleep(SUSTAINED_GRAB_DURATION_S)
            measured_fps = stream.fps
            print(f"measured fps: {measured_fps:.2f}")
            assert measured_fps > MIN_ACCEPTABLE_FPS
        finally:
            stream.stop()


#if __name__ == "__main__":
    