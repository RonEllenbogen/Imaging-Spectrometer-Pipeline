'''
Top-level application shell: MainWindow wires CalibrationScreen ->
LiveViewWidget -> ExtendedMeasurementScreen navigation together in a
single QStackedWidget, and is what makes this package independently
launchable (python src/pipeline/gui/app.py) the same way every other
gui/ screen already is via scripts/demo_live_view.py.

Nothing about this wiring is a placeholder -- CalibrationScreen's load/
create paths, LiveViewWidget's real-time polling loop, and
ExtendedMeasurementScreen's acquire-and-combine flow are all real (see
each module's own docstring). MainWindow's own job is narrower: page
navigation, and owning the one shared CameraStream both downstream
screens are built around (matching the project's
one-camera-connection-at-a-time constraint).

CalibrationScreen is built immediately, since it needs no inputs (it's
what produces them). LiveViewWidget and ExtendedMeasurementScreen are
both built lazily, once CalibrationBundle actually exists to construct
them from: LiveViewWidget when calibration_ready first fires,
ExtendedMeasurementScreen the first time LiveViewWidget's
extended_measurement_requested fires.

MainWindow starts the shared CameraStream itself, in _on_calibration_ready,
right after building it and before constructing LiveViewWidget -- neither
downstream screen ever does this itself (LiveViewWidget's polling loop
only ever reads via get_latest_frame(); ExtendedMeasurementScreen's own
_maybe_reconfigure_camera_stream() stops/reconfigures/restarts it around a
settings change, but assumes it was already running to begin with). It's
stopped again in closeEvent(), so a running background acquisition thread
never outlives the window -- and also in _on_back_to_calibration_requested,
since the project's one-camera-connection-at-a-time constraint means it
must be free again before CalibrationScreen's own dialogs can reconnect.
That handler tears LiveViewWidget/ExtendedMeasurementScreen down entirely
(not just hides them) rather than leaving them parked mid-stack, since
_on_calibration_ready can now genuinely fire more than once in one
MainWindow's lifetime -- every previous round-trip's stale instances must
be gone first, not just hidden behind a fresh pair.
'''

# Imports

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from pipeline.acquisition import CameraError
from pipeline.cli.calibration import build_camera_stream
from pipeline.gui.calibration_dialogs import show_camera_error_dialog
from pipeline.gui.calibration_screen import CalibrationBundle, CalibrationScreen
from pipeline.gui.extended_measurement import ExtendedMeasurementScreen
from pipeline.gui.live_view import LiveViewWidget

# Constants

DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 900

# Classes


class MainWindow(QMainWindow):

    '''
    Top-level window: a QStackedWidget holding CalibrationScreen and (once
    built) LiveViewWidget and ExtendedMeasurementScreen, with navigation
    wired between them. See module docstring for the full hand-off/
    lazy-construction story.
    '''

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._bundle: CalibrationBundle | None = None
        self._camera_stream = None
        self._live_view: LiveViewWidget | None = None
        self._extended_measurement = None

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._calibration_screen = CalibrationScreen()
        self._calibration_screen.calibration_ready.connect(self._on_calibration_ready)
        self._stack.addWidget(self._calibration_screen)

    def _on_calibration_ready(self, bundle: CalibrationBundle) -> None:

        '''
        Handler for CalibrationScreen.calibration_ready: builds the one
        shared CameraStream both downstream screens will be constructed
        around, then builds and switches to LiveViewWidget.
        ExtendedMeasurementScreen is not built here -- it's built lazily,
        the first time LiveViewWidget's extended_measurement_requested
        fires (see _on_extended_measurement_requested).

        By the time this handler runs, bundle.calibration_set and
        bundle.noise_model are guaranteed non-None (see
        CalibrationScreen's class docstring's hand-off contract). Can fire
        more than once per MainWindow -- once after a
        back_to_calibration_requested round-trip, self._live_view/
        self._extended_measurement/self._camera_stream are already torn
        down and None (see _on_back_to_calibration_requested), so this
        just runs the same fresh-build path it does the first time.

        If the camera itself fails to start (CameraError -- no device
        found, already open elsewhere, etc.), shows the same
        show_camera_error_dialog() every other real camera-touching call
        in this codebase routes to, and leaves self._bundle/
        self._camera_stream reset to None rather than half-set -- the
        user stays on CalibrationScreen (never switched away from), which
        needs no state reset of its own to retry: clicking "Load Existing
        Calibrations" (or completing CreatePage) again just re-fires
        calibration_ready and re-enters this method fresh.
        '''

        camera_stream = build_camera_stream(
            gain_db=bundle.calibration_set.baseline_record.gain_db,
            exposure_us=bundle.calibration_set.baseline_record.exposure_us,
        )
        try:
            camera_stream.start()
        except CameraError as error:
            show_camera_error_dialog(self, str(error))
            return

        self._bundle = bundle
        self._camera_stream = camera_stream

        self._live_view = LiveViewWidget(
            calibration_set=bundle.calibration_set,
            noise_model=bundle.noise_model,
            position_calibration=bundle.position_calibration,
            wavelength_axis=bundle.wavelength_axis,
            camera_stream=self._camera_stream,
            conversion_gain_record=bundle.conversion_gain_record,
        )
        self._live_view.extended_measurement_requested.connect(
            self._on_extended_measurement_requested
        )
        self._live_view.back_to_calibration_requested.connect(
            self._on_back_to_calibration_requested
        )
        self._stack.addWidget(self._live_view)
        self._stack.setCurrentWidget(self._live_view)

    def _on_extended_measurement_requested(self) -> None:

        '''
        Handler for LiveViewWidget.extended_measurement_requested: builds
        ExtendedMeasurementScreen the first time this fires, reusing the
        same CameraStream and CalibrationBundle _on_calibration_ready
        stored; reuses the already-built instance on every subsequent
        call instead of rebuilding it. Either way, switches the stack to
        it.
        '''

        if self._extended_measurement is None:
            self._extended_measurement = ExtendedMeasurementScreen(
                calibration_set=self._bundle.calibration_set,
                noise_model=self._bundle.noise_model,
                position_calibration=self._bundle.position_calibration,
                wavelength_axis=self._bundle.wavelength_axis,
                camera_stream=self._camera_stream,
                conversion_gain_record=self._bundle.conversion_gain_record,
            )
            self._extended_measurement.back_requested.connect(
                lambda: self._stack.setCurrentWidget(self._live_view)
            )
            self._stack.addWidget(self._extended_measurement)

        self._stack.setCurrentWidget(self._extended_measurement)

    def _on_back_to_calibration_requested(self) -> None:

        '''
        Handler for LiveViewWidget.back_to_calibration_requested: stops
        the shared CameraStream -- freeing it for CalibrationScreen's own
        dialogs, per the project's one-camera-connection-at-a-time
        constraint -- then tears down LiveViewWidget and (if it was ever
        built) ExtendedMeasurementScreen entirely, removing them from
        self._stack and dropping every reference, before switching back
        to CalibrationScreen. Full teardown rather than just hiding them:
        both were built around this specific (now-stopped) CameraStream
        and CalibrationBundle, so leaving them alive would either show
        stale state if reused, or silently leak a second live pair
        alongside whatever _on_calibration_ready builds next if a fresh
        calibration is completed from here.
        '''

        if self._camera_stream is not None and self._camera_stream.is_running:
            self._camera_stream.stop()
        self._camera_stream = None
        self._bundle = None

        if self._extended_measurement is not None:
            self._stack.removeWidget(self._extended_measurement)
            self._extended_measurement.deleteLater()
            self._extended_measurement = None

        if self._live_view is not None:
            self._stack.removeWidget(self._live_view)
            self._live_view.deleteLater()
            self._live_view = None

        self._stack.setCurrentWidget(self._calibration_screen)

    def closeEvent(self, event) -> None:

        '''
        Stops the shared CameraStream's background acquisition thread
        (if one was ever started -- CalibrationScreen may never have
        reached calibration_ready) before the window closes, so it never
        outlives the GUI.
        '''

        if self._camera_stream is not None and self._camera_stream.is_running:
            self._camera_stream.stop()
        super().closeEvent(event)


# Functions


if __name__ == "__main__":
    app = QApplication.instance() or QApplication([])

    window = MainWindow()
    window.setWindowTitle("Imaging Spectrometer Pipeline")
    window.resize(DEFAULT_WIDTH, DEFAULT_HEIGHT)
    window.show()

    app.exec()
