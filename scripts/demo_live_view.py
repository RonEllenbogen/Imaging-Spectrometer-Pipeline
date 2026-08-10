"""
Builds a LiveViewWidget with placeholder/synthetic calibration objects and
saves a screenshot -- for visual sanity-checking the GUI skeleton without
a display attached, and without needing the (not-yet-built) calibration
screen to assemble real inputs.

Assembling CalibrationSet/SensorNoiseModel/ScaleFactorPositionCalibration/
WavelengthAxis/CameraStream for real is the calibration screen's job
(see src/pipeline/gui/live_view.py's module docstring) -- this script
stands in for that with synthetic values, the same way the rest of this
codebase's test suite does.

Usage (no display required):
    QT_QPA_PLATFORM=offscreen python scripts/demo_live_view.py
    QT_QPA_PLATFORM=offscreen python scripts/demo_live_view.py --output out.png
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

DEFAULT_OUTPUT_PATH = Path("assets/images/live_view_skeleton_sample.png")
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--degree", type=int, default=1, choices=(1, 2, 3),
        help="Fit degree to select before the screenshot is taken (default: 1).",
    )
    args = parser.parse_args()

    # Imported here, not at module level, so this script can be imported
    # (e.g. by a test) without requiring a QApplication to exist yet.
    from PySide6.QtWidgets import QApplication
    from pipeline.gui.live_view import DEGREE_CHOICES, LiveViewWidget

    app = QApplication.instance() or QApplication([])

    widget = LiveViewWidget(
        calibration_set=build_placeholder_calibration_set(),
        noise_model=SensorNoiseModel(gain_e_per_adu=2.2, background_sigma=1.0),
        position_calibration=ScaleFactorPositionCalibration(),
        wavelength_axis=None,   # the expected v1 state -- see module docstring
        camera_stream=build_placeholder_camera_stream(),
    )
    widget.resize(args.width, args.height)
    widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(args.degree))

    pixmap = widget.grab()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(args.output))
    print(f"Saved screenshot to {args.output.resolve()}")


if __name__ == "__main__":
    main()
