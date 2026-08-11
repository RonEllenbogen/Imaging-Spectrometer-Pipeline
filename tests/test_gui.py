'''
Test suite for app.py's MainWindow -- the cross-cutting screen-navigation
wiring that ties calibration_screen.py, live_view.py, and
extended_measurement.py together, so it doesn't belong in any one of
those screens' own dedicated test file (see
tests/test_calibration_screen.py's module docstring for why this repo now
has one test file per gui/ screen instead of one big tests/test_gui.py).

PySide6/pyqtgraph/pytest-qt are local-only dependencies (see CLAUDE.md --
pyproject.toml/requirements.txt are deliberately left empty), so this
whole module is skipped, not failed, wherever they aren't installed --
the same "gate, don't require" pattern tests/test_acquisition.py uses for
hardware-only tests.
'''

# Imports

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 is a local-only GUI dependency")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is a local-only GUI dependency")
pytest.importorskip("pytestqt", reason="pytest-qt is a local-only GUI dependency")

# Must be set before any QApplication is constructed -- pytest-qt creates
# one lazily the first time a test requests the qtbot fixture.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pipeline.acquisition import CANONICAL_SHAPE  # noqa: E402
from pipeline.calibration.shared import CalibrationRecord  # noqa: E402
from pipeline.calibration.spatial import ScaleFactorPositionCalibration  # noqa: E402
import pipeline.gui.calibration_screen as calibration_screen_module  # noqa: E402
from pipeline.gui.app import MainWindow  # noqa: E402
from pipeline.gui.extended_measurement import ExtendedMeasurementScreen  # noqa: E402

# Constants

FIXTURE_EXPOSURE_US = 2000.0
FIXTURE_GAIN_DB = 0.0

# Classes

# Functions

def _patch_successful_calibration_load(monkeypatch) -> tuple[SimpleNamespace, SimpleNamespace, ScaleFactorPositionCalibration]:
    '''
    Patches calibration_screen's load_baseline()/load_flat_field()/
    load_bad_pixel_map()/load_conversion_gain()/load_scale_factor() calls
    to all succeed with synthetic data. Returns (baseline_result,
    conversion_gain_result, position_calibration) so callers can assert
    the returned CalibrationBundle was actually built from these values.
    Mirrors tests/test_calibration_screen.py's own copy of this helper --
    kept as a separate local copy rather than a shared import, matching
    this codebase's existing per-file test-helper convention (e.g.
    tests/test_calibration.py and tests/test_preprocessing.py each define
    their own local _frame()/_uniform() rather than sharing one module).
    '''
    record = CalibrationRecord(
        exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
        timestamp=time.time(), source_frame_count=50,
    )
    baseline_result = SimpleNamespace(
        baseline=np.full(CANONICAL_SHAPE, 10.0, dtype=np.float64), background_sigma=1.0
    )
    flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
    bad_pixel_mask = np.zeros(CANONICAL_SHAPE, dtype=bool)
    conversion_gain_result = SimpleNamespace(gain_e_per_adu=2.2)
    position_calibration = ScaleFactorPositionCalibration()

    monkeypatch.setattr(
        calibration_screen_module, "load_baseline", lambda path: (baseline_result, record)
    )
    monkeypatch.setattr(
        calibration_screen_module, "load_flat_field", lambda path: (flat_field, record)
    )
    monkeypatch.setattr(
        calibration_screen_module, "load_bad_pixel_map", lambda path: (bad_pixel_mask, record)
    )
    monkeypatch.setattr(
        calibration_screen_module,
        "load_conversion_gain",
        lambda path: (conversion_gain_result, record),
    )
    monkeypatch.setattr(
        calibration_screen_module,
        "load_scale_factor",
        lambda path: (position_calibration, object()),
    )
    return baseline_result, conversion_gain_result, position_calibration


# ---------------------------------------------------------------------------
# app.py -- MainWindow navigation (pytest-qt, offscreen)
# ---------------------------------------------------------------------------

class TestMainWindowNavigation:

    def test_calibration_to_live_view_to_extended_measurement_and_back(
        self, qtbot, monkeypatch
    ):
        _patch_successful_calibration_load(monkeypatch)
        window = MainWindow()
        qtbot.addWidget(window)

        window._calibration_screen.welcome_page.load_requested.emit()

        assert window._stack.currentWidget() is window._live_view
        assert window._live_view is not None

        live_view = window._live_view

        window._live_view.extended_measurement_requested.emit()

        assert isinstance(window._extended_measurement, ExtendedMeasurementScreen)
        assert window._stack.currentWidget() is window._extended_measurement

        extended_measurement = window._extended_measurement

        window._extended_measurement.back_requested.emit()

        assert window._stack.currentWidget() is window._live_view
        assert window._live_view is live_view

        window._live_view.extended_measurement_requested.emit()

        assert window._extended_measurement is extended_measurement
        assert window._stack.currentWidget() is window._extended_measurement
