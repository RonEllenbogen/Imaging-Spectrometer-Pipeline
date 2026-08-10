"""
Opens the calibration screen and the live-view screen together, each as
its own real window, in their current in-development states -- for
reviewing both screens interactively in one run rather than launching
each by hand. LiveViewWidget is built with placeholder/synthetic
calibration objects (assembling them for real is the calibration
screen's job, not this script's -- see live_view.py's module docstring);
CalibrationScreen needs no such inputs, since it's what produces them.

Usage (opens two real windows, requires a display):
    python scripts/demo_live_view.py

Usage (no display required -- saves a screenshot of each screen instead):
    QT_QPA_PLATFORM=offscreen python scripts/demo_live_view.py --screenshot
    QT_QPA_PLATFORM=offscreen python scripts/demo_live_view.py --screenshot --output-dir out/
"""

# Imports

import argparse
import time
from pathlib import Path

import numpy as np

from pipeline.acquisition import CameraStream, SyntheticBackend
from pipeline.analysis import SensorNoiseModel
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.preprocessing import CalibrationSet
from pipeline.acquisition import CANONICAL_SHAPE

# Constants

DEFAULT_OUTPUT_DIR = Path("assets/images")
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 900


# Functions

def build_placeholder_calibration_set() -> CalibrationSet:

    '''
    A clean CalibrationSet (uniform baseline, flat flat-field, empty
    bad-pixel mask) -- mirrors tests/test_preprocessing.py's
    _make_clean_calibration_set(), not exercising real calibration-build
    logic, just giving LiveViewWidget something structurally valid to
    hold onto.
    '''

    baseline = np.full(CANONICAL_SHAPE, 10.0, dtype=np.float64)
    flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
    bad_pixel_mask = np.zeros(CANONICAL_SHAPE, dtype=bool)
    record = CalibrationRecord(
        exposure_us=2000.0, gain_db=0.0, timestamp=time.time(), source_frame_count=50,
    )
    return CalibrationSet(
        baseline=baseline, baseline_record=record,
        flat_field=flat_field, flat_field_record=record,
        bad_pixel_mask=bad_pixel_mask, background_sigma=1.0,
    )


def build_placeholder_camera_stream() -> CameraStream:

    '''A CameraStream over SyntheticBackend -- not started; held, not polled, by the widget in this phase.'''

    return CameraStream(
        exposure_us=2000.0, gain_db=0.0, pixel_format="Mono8", timeout_ms=5000,
        backend=SyntheticBackend(seed=0),
    )


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screenshot", action="store_true",
        help="Save a screenshot of each screen instead of opening real windows "
        "(for use with QT_QPA_PLATFORM=offscreen).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--degree", type=int, default=1, choices=(1, 2, 3),
        help="Fit degree to select on the live-view screen before showing/"
        "screenshotting it (default: 1).",
    )
    args = parser.parse_args()

    # Imported here, not at module level, so this script can be imported
    # (e.g. by a test) without requiring a QApplication to exist yet.
    from PySide6.QtWidgets import QApplication
    from pipeline.gui.calibration_screen import CalibrationScreen
    from pipeline.gui.live_view import DEGREE_CHOICES, LiveViewWidget

    app = QApplication.instance() or QApplication([])

    calibration_screen = CalibrationScreen()
    calibration_screen.setWindowTitle("Calibration Screen")
    calibration_screen.resize(args.width, args.height)

    live_view = LiveViewWidget(
        calibration_set=build_placeholder_calibration_set(),
        noise_model=SensorNoiseModel(gain_e_per_adu=2.2, background_sigma=1.0),
        position_calibration=ScaleFactorPositionCalibration(),
        wavelength_axis=None,   # the expected v1 state -- see module docstring
        camera_stream=build_placeholder_camera_stream(),
    )
    live_view.setWindowTitle("Live View")
    live_view.resize(args.width, args.height)
    live_view._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(args.degree))

    if args.screenshot:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, widget in (("calibration_screen", calibration_screen), ("live_view", live_view)):
            output_path = args.output_dir / f"{name}_skeleton_sample.png"
            widget.grab().save(str(output_path))
            print(f"Saved screenshot to {output_path.resolve()}")
        return

    calibration_screen.show()
    live_view.show()
    app.exec()


if __name__ == "__main__":
    main()
