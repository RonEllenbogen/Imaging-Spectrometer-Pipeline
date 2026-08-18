"""
Opens the real gui/app.py MainWindow, pre-seeded with a placeholder
CalibrationBundle instead of one loaded/created through CalibrationScreen
-- so MainWindow's real navigation (CalibrationScreen -> LiveViewWidget ->
ExtendedMeasurementScreen -> back) can be reviewed and driven end-to-end
without an existing calibration_artifacts/ directory or a connected
camera. Complements demo_live_view.py, which instead opens each screen
as its own disconnected window for a quick visual check -- this script
exercises the real signal-based wiring between them, per gui/app.py's
own module docstring.

MainWindow is built normally (so its CalibrationScreen instance still exists at index 0 of the
stack, and is what --screenshot captures first -- both its own Welcome page, then navigated to
CreatePage), then CalibrationScreen.calibration_ready is emitted directly with a synthetic
CalibrationBundle, the same
hand-off MainWindow would receive from a real WelcomePage "Load Existing Calibrations" click (see
calibration_screen.py's class docstring) -- MainWindow's _on_calibration_ready() handler runs for real
from there, including starting the camera stream it builds. `pipeline.gui.app.build_camera_stream` is
monkeypatched to a SyntheticBackend-driven CameraStream before that handler runs, so this script needs
no physical camera attached; without that override, MainWindow would build (and now, since it also
starts it, actually try to connect) a real PylonBackend stream and hang/error with no hardware present.

Usage (opens one real window, requires a display):
    python scripts/demo_app.py

Usage (no display required -- saves a screenshot of each navigated-to screen instead, running a real
20-shot measurement via SyntheticBackend so the extended-measurement screenshot shows actual plotted
centroids/fit rather than its pre-run empty state):
    QT_QPA_PLATFORM=offscreen python scripts/demo_app.py --screenshot
    QT_QPA_PLATFORM=offscreen python scripts/demo_app.py --screenshot --output-dir out/
"""

# Imports

import argparse
import time
from pathlib import Path

import numpy as np

from pipeline.acquisition import CANONICAL_SHAPE, CameraStream, SyntheticBackend
from pipeline.analysis import SensorNoiseModel
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.preprocessing import CalibrationSet

# Constants

DEFAULT_OUTPUT_DIR = Path("assets/images")
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 900


# Functions

def _build_synthetic_camera_stream(gain_db, *, exposure_us=None, auto_exposure=False):

    '''
    Stand-in for pipeline.cli.calibration.build_camera_stream() -- the
    real one constructs a PylonBackend, which MainWindow now starts (see
    module docstring), so without this override main() would try to
    connect to real hardware. Mirrors demo_live_view.py's
    build_placeholder_camera_stream() and every gui/ test file's own copy
    of this same substitution.
    '''

    return CameraStream(
        exposure_us=exposure_us if exposure_us is not None else 2000.0,
        gain_db=gain_db, pixel_format="Mono8", timeout_ms=5000,
        backend=SyntheticBackend(seed=0),
    )


def build_placeholder_calibration_bundle():

    '''
    A CalibrationBundle wrapping a clean CalibrationSet (uniform baseline,
    flat flat-field, empty bad-pixel mask) -- mirrors demo_live_view.py's
    build_placeholder_calibration_set(), not exercising real
    calibration-build logic, just giving MainWindow's real
    _on_calibration_ready() handler something structurally valid to
    build LiveViewWidget/ExtendedMeasurementScreen from. No
    conversion_gain_record or wavelength_axis, matching demo_live_view.py's
    same "no conversion-gain artifact loaded / no wavelength calibration
    yet" placeholder state.
    '''

    from pipeline.gui.calibration_screen import CalibrationBundle

    baseline = np.full(CANONICAL_SHAPE, 10.0, dtype=np.float64)
    flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
    bad_pixel_mask = np.zeros(CANONICAL_SHAPE, dtype=bool)
    record = CalibrationRecord(
        exposure_us=2000.0, gain_db=0.0, timestamp=time.time(), source_frame_count=50,
    )
    calibration_set = CalibrationSet(
        baseline=baseline, baseline_record=record,
        flat_field=flat_field, flat_field_record=record,
        bad_pixel_mask=bad_pixel_mask, background_sigma=1.0,
    )
    return CalibrationBundle(
        calibration_set=calibration_set,
        noise_model=SensorNoiseModel(gain_e_per_adu=2.2, background_sigma=1.0),
        position_calibration=ScaleFactorPositionCalibration(),
    )


def _save_window_screenshot(window, screen_widget, output_path: Path) -> None:

    '''Switches window's top-level stack to screen_widget, then grabs
    screen_widget itself rather than window -- grabbing the enclosing
    QMainWindow directly does not reliably composite an offscreen,
    never-shown QStackedWidget page's own styled background (reproduced
    against calibration_screen.py's CreatePage specifically: its palette
    resolves correctly, but window.grab() still showed the QMainWindow's
    unstyled default background behind it), while grabbing the page
    widget directly renders correctly every time.'''

    window._stack.setCurrentWidget(screen_widget)
    screen_widget.grab().save(str(output_path))
    print(f"Saved screenshot to {output_path.resolve()}")


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screenshot", action="store_true",
        help="Save a screenshot of the welcome, calibration, live-view, and "
        "extended-measurement screens instead of opening a real window (for use with "
        "QT_QPA_PLATFORM=offscreen).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--degree", type=int, default=1, choices=(1, 2, 3),
        help="Fit degree to select on both screens before showing/"
        "screenshotting them (default: 1).",
    )
    args = parser.parse_args()

    # Imported here, not at module level, so this script can be imported
    # (e.g. by a test) without requiring a QApplication to exist yet --
    # same rationale as demo_live_view.py's own main().
    from PySide6.QtWidgets import QApplication
    import pipeline.gui.app as app_module
    from pipeline.gui.app import MainWindow
    from pipeline.gui.live_view import DEGREE_CHOICES

    app = QApplication.instance() or QApplication([])

    app_module.build_camera_stream = _build_synthetic_camera_stream

    window = MainWindow()
    window.setWindowTitle("Imaging Spectrometer Pipeline (placeholder calibrations)")
    window.resize(args.width, args.height)

    if args.screenshot:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Welcome page -- CalibrationScreen's own first page (index 0
        # of its internal stack), already MainWindow's current widget at
        # construction time, before any navigation at all.
        app.processEvents()
        _save_window_screenshot(
            window, window._calibration_screen, args.output_dir / "welcome_screen_sample.png"
        )

        # 2. Calibration screen, navigated to CreatePage (the per-type
        # "create new calibration" cards) -- more informative for a
        # screenshot than the bare Welcome page, and it's still
        # MainWindow's current widget at this point (index 0), so no
        # calibration_ready hand-off has happened yet.
        window._calibration_screen.welcome_page.create_requested.emit()
        app.processEvents()
        _save_window_screenshot(
            window, window._calibration_screen, args.output_dir / "calibration_screen_sample.png"
        )

        # 3. Hand off to MainWindow exactly as a real "Continue to Main
        # Window"/"Load Existing Calibrations" click would, then let
        # live_view's real QTimer polling loop actually populate a frame
        # before capturing it -- it only fires while the Qt event loop is
        # running, and app.exec() never runs in this branch, so without
        # pumping events here the screenshot would still only show
        # live_view's construction-time placeholder paint.
        window._calibration_screen.calibration_ready.emit(build_placeholder_calibration_bundle())
        window._live_view._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(args.degree))
        for _ in range(10):
            app.processEvents()
            time.sleep(0.05)
        _save_window_screenshot(window, window._live_view, args.output_dir / "live_view_sample.png")

        # 4. Extended measurement: actually run a (synthetic) measurement
        # via the real _on_run_clicked() handler before capturing it, so
        # the screenshot shows real plotted centroids/fit rather than its
        # pre-run empty state. Synchronous -- see its own docstring --
        # but a few processEvents() calls afterward let pyqtgraph finish
        # redrawing before the grab.
        window._live_view.extended_measurement_requested.emit()
        window._extended_measurement._degree_selector.setCurrentIndex(
            DEGREE_CHOICES.index(args.degree)
        )
        window._extended_measurement._on_run_clicked()
        for _ in range(5):
            app.processEvents()
        _save_window_screenshot(
            window, window._extended_measurement, args.output_dir / "extended_measurement_sample.png"
        )
        return

    window._calibration_screen.calibration_ready.emit(build_placeholder_calibration_bundle())
    window._live_view._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(args.degree))
    window._live_view.extended_measurement_requested.emit()
    window._extended_measurement._degree_selector.setCurrentIndex(
        DEGREE_CHOICES.index(args.degree)
    )
    window._stack.setCurrentWidget(window._live_view)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
