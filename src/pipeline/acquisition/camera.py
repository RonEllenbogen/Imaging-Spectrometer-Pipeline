'''
This file is in charge of orchestratring the camera-grabbing loop in the background, and ensuring that a given script can access the newest picture safely.
'''
# Imports
import logging
import threading
import time
from collections import deque

from .backends import CameraBackend, PylonBackend
from .exceptions import CameraError, CameraConfigurationError, CameraTimeoutError
from .frame import FrameData

# Constants

logger = logging.getLogger(__name__)

# How many recent frame timestamps to keep for the rolling fps estimate.
# 30 frames is roughly a half-second window at the camera's target frame rate.
FPS_WINDOW_SIZE = 30

# How many consecutive grab_one() timeouts to tolerate before treating the
# stream as genuinely broken rather than experiencing a transient hiccup.
DEFAULT_MAX_CONSECUTIVE_TIMEOUTS = 5

# How long collect_n_frames() sleeps between polls of get_latest_frame().
# Small relative to any realistic frame interval (this camera tops out
# around 51fps, ~20ms/frame), so it adds negligible latency, but not so
# small it busy-spins the polling thread.
COLLECT_POLL_INTERVAL_S = 0.005

# Classes

class CameraStream:

    '''
    Keeps a continuous grab loop running independently of any caller, wraps
    a CameraBackend in a background thread.
    '''

    def __init__(
        self,
        exposure_us: float,
        gain_db: float,
        pixel_format: str,
        timeout_ms: int,
        backend: CameraBackend | None = None,
        serial_number: str | None = None,
        max_consecutive_timeouts: int = DEFAULT_MAX_CONSECUTIVE_TIMEOUTS,
        auto_exposure: bool = False,
        auto_timeout_ms: int = 5000,
    ):

        '''
        Parameters
        ----------
        exposure_us
            Exposure time in microseconds, passed to backend.configure().
            Ignored (but still required) when auto_exposure is True -- the
            real converged value is read back from configure()'s return
            and takes over as self.exposure_us instead (see start()).
        gain_db
            Sensor gain in decibels, passed to backend.configure().
        pixel_format
            PixelFormat string (e.g. "Mono8"), passed to backend.configure().
        backend
            A CameraBackend instance. Defaults to a real PylonBackend() if
            None -- pass a SyntheticBackend here for testing.
        timeout_ms
            Timeout passed to backend.grab_one() on every grab attempt.
        max_consecutive_timeouts
            Number of consecutive CameraTimeoutErrors to tolerate before
            _run() treats the stream as fatally broken and exits.
        auto_exposure
            If True and backend is None (real PylonBackend construction),
            configure() runs a one-time ExposureAuto convergence instead of
            setting exposure_us directly -- see PylonBackend. Has no effect
            if an explicit backend (e.g. SyntheticBackend) is supplied;
            pass auto_exposure directly to that backend's own constructor
            instead, if it supports the concept.
        auto_timeout_ms
            Timeout, per grab, while waiting for auto-exposure to converge.
            Only relevant when auto_exposure is True and backend is None.
        '''

        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.pixel_format = pixel_format
        self.timeout_ms = timeout_ms
        self.max_consecutive_timeouts = max_consecutive_timeouts
        self.serial_number = serial_number

        if backend is None:
            if serial_number is None:
                raise ValueError("serial_number is required when backend is not provided")
            backend = PylonBackend(
                serial_number=serial_number,
                auto_exposure=auto_exposure,
                auto_timeout=auto_timeout_ms,
            )
        self._backend = backend

        # Threading machinery -- no thread exists until start() creates one
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Frame hand-off: a lock-guarded variable, always holding just the
        # newest frame. get_latest_frame() is a plain read -- repeated calls
        # correctly return the same frame until _run() writes a new one.
        self._latest_frame: FrameData | None = None
        self._frame_lock = threading.Lock()

        self._frame_counter = 0
        self._frame_timestamps: deque[float] = deque(maxlen=FPS_WINDOW_SIZE)

        # Set by _run() if the thread terminates from an unrecoverable
        # CameraError; None while running normally or before the first start()
        self._last_error: Exception | None = None

    def start(self) -> None:

        '''
        Connects the backend, configures it, and starts the background
        grab thread. Blocks until connected and configured -- does NOT
        wait for a first frame to arrive. A no-op if already running.

        self.exposure_us is updated to whatever configure() reports was
        actually applied -- identical to the value passed in for a fixed
        exposure, but the real converged value when auto_exposure is on
        (see CameraBackend.configure()'s docstring) -- so every FrameData
        grabbed by _run() afterward carries the true applied exposure, not
        a stale nominal one.
        '''

        if self.is_running:
            logger.info("start() called but stream is already running; no-op")
            return

        # connect() failures propagate as-is -- nothing to clean up yet
        self._backend.connect()

        try:
            self.exposure_us = self._backend.configure(
                self.exposure_us, self.gain_db, self.pixel_format
            )
        except CameraConfigurationError:
            # Connected but misconfigured -- close before re-raising so the
            # device isn't left open, blocking the next start() attempt
            self._backend.close()
            raise

        # Clear state left over from a previous start()/stop() cycle
        self._stop_event.clear()
        self._last_error = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:

        '''
        Signals the background thread to exit (if still running), waits for
        it to actually finish, then closes the backend. Synchronous --
        callers can trust the device is free the moment this returns. A
        no-op only if start() was never called; safe to call even if the
        thread already terminated on its own from a fatal error.
        '''

        if self._thread is None:
            logger.info("stop() called but stream was never started; no-op")
            return

        self._stop_event.set()
        self._thread.join()   # returns immediately if the thread already exited
        self._backend.close()

    def get_latest_frame(self) -> FrameData | None:

        '''
        Non-blocking, thread-safe read of the most recent frame. Returns
        the same FrameData on repeated calls until _run() writes a new one.

        Returns
        -------
        FrameData | None
            The latest available frame, or None if start() hasn't been
            called yet or no frame has arrived so far.
        '''

        with self._frame_lock:
            return self._latest_frame

    def collect_n_frames(self, n: int) -> list[FrameData]:

        '''
        Collects n distinct frames from this already-running stream, by
        polling get_latest_frame() and keeping only frames whose frame_id
        hasn't been seen yet. Does NOT perform any grabs of its own --
        deliberately reuses whatever background thread is already
        running, since GigE's one-connection-per-camera limit means live
        view and batch/calibration capture can't hold separate
        connections open at the same time; this is how they share one.

        A plain repeated call to get_latest_frame() would risk counting
        the same FrameData twice, since it returns the same object on
        every call until _run() writes a new one -- comparing frame_id
        is what makes the polling loop only count genuinely new frames.

        Parameters
        ----------
        n
            Number of distinct frames to collect. Must be positive.

        Returns
        -------
        list[FrameData]
            Exactly n frames, in the order the camera produced them
            (increasing frame_id).

        Raises
        ------
        ValueError
            If n is not positive.
        RuntimeError
            If the stream isn't running when this is called, or stops
            running -- without last_error being set, e.g. a concurrent
            stop() -- while this call is still waiting.
        CameraError
            Re-raised from last_error if the background thread terminates
            from a fatal camera error while this call is still waiting.
        '''

        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        if not self.is_running:
            if self._last_error is not None:
                raise self._last_error
            raise RuntimeError("collect_n_frames() requires an already-running CameraStream")

        collected: list[FrameData] = []
        last_seen_frame_id: int | None = None

        while len(collected) < n:
            frame = self.get_latest_frame()

            if frame is not None and frame.frame_id != last_seen_frame_id:
                collected.append(frame)
                last_seen_frame_id = frame.frame_id
                continue

            if not self.is_running:
                if self._last_error is not None:
                    raise self._last_error
                raise RuntimeError(
                    "stream stopped running before collect_n_frames() collected all "
                    f"{n} requested frames ({len(collected)} collected)"
                )

            time.sleep(COLLECT_POLL_INTERVAL_S)

        return collected

    @property
    def is_running(self) -> bool:

        '''
        Whether the background thread is genuinely alive right now.
        Derived directly from the thread's actual state rather than a
        separately tracked flag, so it can never drift out of sync with
        reality.

        Returns
        -------
        bool
        '''

        return self._thread is not None and self._thread.is_alive()

    @property
    def fps(self) -> float:

        '''
        Rolling frame-rate estimate over the last FPS_WINDOW_SIZE frames.

        Returns
        -------
        float
            0.0 if fewer than two timestamps are available yet (can't
            compute a rate from a single point), otherwise frames per
            second over the current window.
        '''

        # list() copies out of the deque -- safe against the writer thread
        # appending concurrently, since the copy happens under the GIL
        timestamps = list(self._frame_timestamps)

        if len(timestamps) < 2:
            return 0.0

        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(timestamps) - 1) / elapsed

    @property
    def last_error(self) -> Exception | None:

        '''
        The exception that caused the background thread to terminate, if
        it terminated from a CameraError. None while running normally, or
        if stop() was called deliberately rather than the thread failing.

        Returns
        -------
        Exception | None
        '''

        return self._last_error

    def __enter__(self):

        '''
        Calls start() and returns self, enabling `with CameraStream(...) as cam:`.

        Returns
        -------
        CameraStream
        '''

        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):

        '''
        Calls stop() unconditionally, regardless of whether the `with`
        block raised. Returns False so any such exception continues to
        propagate rather than being silently swallowed.

        Returns
        -------
        bool
        '''

        self.stop()
        return False

    def _run(self) -> None:

        '''
        The background thread's target. Loops grabbing frames until
        stop() signals exit or an unrecoverable error occurs. Never
        called directly -- only ever invoked by the threading.Thread
        created in start().

        A CameraTimeoutError is treated as tolerable up to
        max_consecutive_timeouts, then fatal. Any other CameraError is
        fatal immediately. Any non-CameraError exception is a genuine bug,
        not a camera problem -- it is deliberately NOT caught here, so it
        surfaces as a full traceback rather than being silently absorbed
        into last_error.
        '''

        consecutive_timeouts = 0

        while not self._stop_event.is_set():
            try:
                raw_frame = self._backend.grab_one(timeout_ms=self.timeout_ms)
            except CameraTimeoutError as e:
                consecutive_timeouts += 1
                logger.warning(
                    "grab_one() timed out (%d/%d consecutive)",
                    consecutive_timeouts, self.max_consecutive_timeouts,
                )
                if consecutive_timeouts >= self.max_consecutive_timeouts:
                    self._last_error = e
                    logger.error("exceeded max consecutive timeouts; stopping stream")
                    return
                continue
            except CameraError as e:
                # Any other domain error (e.g. connection dropped mid-stream)
                # is fatal immediately -- no retry
                self._last_error = e
                logger.error("fatal camera error, stopping stream: %s", e)
                return

            consecutive_timeouts = 0   # reset only after a successful grab

            frame_data = FrameData(
                image=raw_frame, timestamp=time.monotonic(), frame_id=self._frame_counter, exposure_us=self.exposure_us, gain_db=self.gain_db,
            )
            self._frame_counter += 1

            with self._frame_lock:
                self._latest_frame = frame_data

            self._frame_timestamps.append(frame_data.timestamp)

# Functions

#if __name__ == "__main__":
    