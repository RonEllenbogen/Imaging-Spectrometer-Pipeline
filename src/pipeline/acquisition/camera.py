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
        max_consecutive_timeouts: int = DEFAULT_MAX_CONSECUTIVE_TIMEOUTS,
    ):

        '''
        Parameters
        ----------
        exposure_us
            Exposure time in microseconds, passed to backend.configure().
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
        '''

        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.pixel_format = pixel_format
        self.timeout_ms = timeout_ms
        self.max_consecutive_timeouts = max_consecutive_timeouts

        self._backend = backend if backend is not None else PylonBackend()

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
        '''

        if self.is_running:
            logger.info("start() called but stream is already running; no-op")
            return

        # connect() failures propagate as-is -- nothing to clean up yet
        self._backend.connect()

        try:
            self._backend.configure(self.exposure_us, self.gain_db, self.pixel_format)
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
                image=raw_frame, timestamp=time.monotonic(), frame_id=self._frame_counter
            )
            self._frame_counter += 1

            with self._frame_lock:
                self._latest_frame = frame_data

            self._frame_timestamps.append(frame_data.timestamp)

# Functions

#if __name__ == "__main__":
    