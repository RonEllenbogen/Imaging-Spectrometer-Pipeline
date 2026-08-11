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

CalibrationScreen itself is skipped, not shown: MainWindow is built
normally (so its CalibrationScreen instance still exists at index 0 of
the stack), then CalibrationScreen.calibration_ready is emitted directly
with a synthetic CalibrationBundle, the same hand-off MainWindow would
receive from a real WelcomePage "Load Existing Calibrations" click (see
calibration_screen.py's class docstring) -- MainWindow's
_on_calibration_ready() handler runs for real from there, including its
real (hardware-safe at construction time) build_camera_stream() call.

Usage (opens one real window, requires a display):
    python scripts/demo_app.py

Usage (no display required -- saves a screenshot of each navigated-to
screen instead):
    QT_QPA_PLATFORM=offscreen python scripts/demo_app.py --screenshot
    QT_QPA_PLATFORM=offscreen python scripts/demo_app.py --screenshot --output-dir out/
"""

# Imports

import argparse
import time
from pathlib import Path

import numpy as np

from pipeline.acquisition import CANONICAL_SHAPE
from pipeline.analysis import SensorNoiseModel
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.preprocessing import CalibrationSet

# Constants

DEFAULT_OUTPUT_DIR = Path("assets/images")
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 900


# Functions

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


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screenshot", action="store_true",
        help="Save a screenshot of the live-view and extended-measurement "
        "screens instead of opening a real window (for use with "
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
    from pipeline.gui.app import MainWindow
    from pipeline.gui.live_view import DEGREE_CHOICES

    app = QApplication.instance() or QApplication([])

    window = MainWindow()
    window.setWindowTitle("Imaging Spectrometer Pipeline (placeholder calibrations)")
    window.resize(args.width, args.height)

    window._calibration_screen.calibration_ready.emit(build_placeholder_calibration_bundle())
    window._live_view._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(args.degree))

    window._live_view.extended_measurement_requested.emit()
    window._extended_measurement._degree_selector.setCurrentIndex(
        DEGREE_CHOICES.index(args.degree)
    )

    if args.screenshot:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, screen in (
            ("live_view", window._live_view),
            ("extended_measurement", window._extended_measurement),
        ):
            window._stack.setCurrentWidget(screen)
            output_path = args.output_dir / f"{name}_skeleton_sample.png"
            window.grab().save(str(output_path))
            print(f"Saved screenshot to {output_path.resolve()}")
        return

    window._stack.setCurrentWidget(window._live_view)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
