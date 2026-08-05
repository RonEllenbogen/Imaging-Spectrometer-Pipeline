"""
Test suite for CameraStream. Every test here runs against SyntheticBackend
-- no hardware required, no HARDWARE_AVAILABLE gating needed, since
CameraStream itself never touches pypylon directly.
"""

import time

import pytest

from pipeline.acquisition import (
    CameraStream,
    DEFAULT_MAX_CONSECUTIVE_TIMEOUTS,
    SyntheticBackend,
    FrameData,
    CameraTimeoutError,
)
from pipeline.acquisition.backends import CameraBackend

# --- fixed configuration for every CameraStream constructed in this file ---
STREAM_SEED = 55
STREAM_EXPOSURE_US = 2000
STREAM_GAIN_DB = 0.0
STREAM_PIXEL_FORMAT = "Mono8"
STREAM_TIMEOUT_MS = 5000   # generous -- SyntheticBackend grabs are near-instant anyway

# --- forced-failure scenarios: short timeout, low threshold, so the
# background thread dies quickly rather than the test waiting around ---
FAST_FAIL_TIMEOUT_MS = 50
FAST_FAIL_MAX_CONSECUTIVE_TIMEOUTS = 3

# --- polling, since the background thread's state changes asynchronously ---
POLL_INTERVAL_S = 0.01
POLL_TIMEOUT_S = 2.0


def _wait_until(condition, timeout_s: float = POLL_TIMEOUT_S, interval_s: float = POLL_INTERVAL_S) -> bool:

    '''
    Polls a zero-argument callable until it returns truthy or timeout_s
    elapses. Used throughout this file to wait for asynchronous state
    changes on CameraStream's background thread -- a new frame arriving,
    the thread dying -- without relying on a fixed sleep duration that
    could be too short (flaky) or wastefully long.

    Parameters
    ----------
    condition
        Zero-argument callable returning a truthy/falsy value.
    timeout_s
        Maximum time to poll before giving up.
    interval_s
        Time to sleep between poll attempts.

    Returns
    -------
    bool
        True if condition() became truthy within timeout_s, False if the
        deadline was reached first.
    '''

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval_s)
    return False


class _CloseTrackingBackend:

    '''
    Wraps a real CameraBackend, recording whether close() was called --
    used only to verify CameraStream.stop()'s cleanup behavior. Satisfies
    CameraBackend structurally (per the earlier Protocol discussion) by
    matching its shape, without inheriting from anything.
    '''

    def __init__(self, wrapped: CameraBackend):
        self._wrapped = wrapped
        self.close_called = False

    def connect(self) -> None:
        self._wrapped.connect()

    def configure(self, exposure_us: float, gain_db: float, pixel_format: str) -> None:
        self._wrapped.configure(exposure_us, gain_db, pixel_format)

    def grab_one(self, timeout_ms: int):
        return self._wrapped.grab_one(timeout_ms)

    def close(self) -> None:
        self.close_called = True
        self._wrapped.close()


def _make_stream(backend=None, timeout_ms=STREAM_TIMEOUT_MS, max_consecutive_timeouts=None):

    '''
    Constructs a CameraStream with this file's standard configuration,
    defaulting to a fresh SyntheticBackend if none is supplied.

    Parameters
    ----------
    backend
        A CameraBackend to use; defaults to SyntheticBackend(seed=STREAM_SEED).
    timeout_ms
        Overrides STREAM_TIMEOUT_MS if provided.
    max_consecutive_timeouts
        Overrides CameraStream's own default if provided.

    Returns
    -------
    CameraStream
    '''

    resolved_backend = backend if backend is not None else SyntheticBackend(seed=STREAM_SEED)
    resolved_max_timeouts = (
        max_consecutive_timeouts
        if max_consecutive_timeouts is not None
        else DEFAULT_MAX_CONSECUTIVE_TIMEOUTS
    )

    return CameraStream(
        exposure_us=STREAM_EXPOSURE_US,
        gain_db=STREAM_GAIN_DB,
        pixel_format=STREAM_PIXEL_FORMAT,
        timeout_ms=timeout_ms,
        backend=resolved_backend,
        max_consecutive_timeouts=resolved_max_timeouts,
    )

def _wait_for_frame(stream: CameraStream, timeout_s: float = POLL_TIMEOUT_S, interval_s: float = POLL_INTERVAL_S) -> FrameData:

    '''
    Polls stream.get_latest_frame() until it returns a FrameData, then
    returns that exact frame -- avoiding a second, separate call that
    neither Pylance nor the code itself can guarantee still returns the
    same (or any) frame.

    Parameters
    ----------
    stream
        The CameraStream to poll.
    timeout_s
        Maximum time to poll before giving up.
    interval_s
        Time to sleep between poll attempts.

    Returns
    -------
    FrameData

    Raises
    ------
    AssertionError
        If no frame arrives within timeout_s.
    '''

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = stream.get_latest_frame()
        if frame is not None:
            return frame
        time.sleep(interval_s)
    raise AssertionError(f"no frame arrived within {timeout_s}s")

class TestCameraStreamLifecycle:
    """start()/stop()/is_running/context-manager behavior."""

    def test_is_running_false_before_start(self):
        stream = _make_stream()
        assert stream.is_running is False

    def test_is_running_true_after_start(self):
        stream = _make_stream()
        stream.start()
        try:
            assert stream.is_running is True
        finally:
            stream.stop()

    def test_is_running_false_after_stop(self):
        stream = _make_stream()
        stream.start()
        stream.stop()
        assert stream.is_running is False

    def test_stop_before_start_is_safe(self):
        stream = _make_stream()
        stream.stop()   # must not raise

    def test_stop_is_idempotent(self):
        stream = _make_stream()
        stream.start()
        stream.stop()
        stream.stop()   # must not raise a second time

    def test_context_manager_starts_and_stops(self):
        stream = _make_stream()
        with stream as entered:
            assert entered is stream
            assert stream.is_running is True
        assert stream.is_running is False


class TestCameraStreamFrameDelivery:
    """get_latest_frame() and frame_id progression."""

    def test_get_latest_frame_before_start_returns_none(self):
        stream = _make_stream()
        assert stream.get_latest_frame() is None

    def test_get_latest_frame_returns_frame_data_after_start(self):
        stream = _make_stream()
        stream.start()
        try:
            frame = _wait_for_frame(stream)
            assert isinstance(frame, FrameData)
        finally:
            stream.stop()

    def test_frame_id_increases_over_successive_frames(self):
        stream = _make_stream()
        stream.start()
        try:
            first_frame = _wait_for_frame(stream)
            first_id = first_frame.frame_id

            def frame_id_increased() -> bool:
                frame = stream.get_latest_frame()
                return frame is not None and frame.frame_id > first_id

            increased = _wait_until(frame_id_increased)
            assert increased, "frame_id never advanced past its first observed value"
        finally:
            stream.stop()


class TestCameraStreamFps:
    """fps property, before and during a running stream."""

    def test_fps_zero_before_any_frames(self):
        stream = _make_stream()
        assert stream.fps == 0.0

    def test_fps_positive_after_running_briefly(self):
        stream = _make_stream()
        stream.start()
        try:
            became_positive = _wait_until(lambda: stream.fps > 0.0)
            assert became_positive, "fps never rose above zero"
        finally:
            stream.stop()


class TestCameraStreamErrorHandling:
    """last_error, and the timeout-threshold/natural-death behavior it exists for."""

    def test_last_error_none_during_normal_operation(self):
        stream = _make_stream()
        stream.start()
        try:
            _wait_until(lambda: stream.get_latest_frame() is not None)
            assert stream.last_error is None
        finally:
            stream.stop()

    def test_last_error_set_after_exceeding_max_consecutive_timeouts(self):
        backend = SyntheticBackend(seed=STREAM_SEED, timeout_probability=1.0)
        stream = _make_stream(
            backend=backend,
            timeout_ms=FAST_FAIL_TIMEOUT_MS,
            max_consecutive_timeouts=FAST_FAIL_MAX_CONSECUTIVE_TIMEOUTS,
        )
        stream.start()

        thread_died = _wait_until(lambda: not stream.is_running)
        assert thread_died, "expected the background thread to terminate after repeated timeouts"
        assert isinstance(stream.last_error, CameraTimeoutError)

    def test_stop_after_natural_thread_death_still_closes_backend(self):
        '''
        Regression test for the join()/is_running bug: stop() must close
        the backend even when the thread already died on its own from a
        fatal error, not only when it's still alive at the moment stop()
        is called.
        '''
        real_backend = SyntheticBackend(seed=STREAM_SEED, timeout_probability=1.0)
        tracking_backend = _CloseTrackingBackend(real_backend)
        stream = _make_stream(
            backend=tracking_backend,
            timeout_ms=FAST_FAIL_TIMEOUT_MS,
            max_consecutive_timeouts=FAST_FAIL_MAX_CONSECUTIVE_TIMEOUTS,
        )
        stream.start()

        _wait_until(lambda: not stream.is_running)
        # the thread dying on its own must NOT have closed the backend --
        # only stop() is responsible for that
        assert tracking_backend.close_called is False

        stream.stop()
        assert tracking_backend.close_called is True


class TestCollectNFrames:
    """collect_n_frames() -- polling get_latest_frame() for distinct frames."""

    def test_returns_n_frames(self):
        stream = _make_stream()
        stream.start()
        try:
            frames = stream.collect_n_frames(5)
            assert len(frames) == 5
            assert all(isinstance(f, FrameData) for f in frames)
        finally:
            stream.stop()

    def test_returns_distinct_increasing_frame_ids(self):
        stream = _make_stream()
        stream.start()
        try:
            frames = stream.collect_n_frames(5)
            frame_ids = [f.frame_id for f in frames]
            assert frame_ids == sorted(set(frame_ids)), "frame_ids must be distinct and increasing"
        finally:
            stream.stop()

    def test_includes_already_latest_frame_as_first(self):
        '''
        A frame already sitting in get_latest_frame() when
        collect_n_frames() is called is still a valid sample at this
        stream's (fixed, never-changing) exposure/gain -- no need to
        wait for a fresh grab before starting to count.
        '''
        stream = _make_stream()
        stream.start()
        try:
            already_latest = _wait_for_frame(stream)
            frames = stream.collect_n_frames(1)
            assert frames[0].frame_id == already_latest.frame_id
        finally:
            stream.stop()

    @pytest.mark.parametrize("bad_n", [0, -1])
    def test_rejects_non_positive_n(self, bad_n):
        stream = _make_stream()
        stream.start()
        try:
            with pytest.raises(ValueError):
                stream.collect_n_frames(bad_n)
        finally:
            stream.stop()

    def test_raises_if_not_running(self):
        stream = _make_stream()
        with pytest.raises(RuntimeError):
            stream.collect_n_frames(3)

    def test_raises_last_error_if_stream_dies_while_waiting(self):
        backend = SyntheticBackend(seed=STREAM_SEED, timeout_probability=1.0)
        stream = _make_stream(
            backend=backend,
            timeout_ms=FAST_FAIL_TIMEOUT_MS,
            max_consecutive_timeouts=FAST_FAIL_MAX_CONSECUTIVE_TIMEOUTS,
        )
        stream.start()
        try:
            with pytest.raises(CameraTimeoutError):
                stream.collect_n_frames(3)
        finally:
            stream.stop()