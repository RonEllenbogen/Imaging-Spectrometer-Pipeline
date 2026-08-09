"""
Test suite for the gui package. Currently covers live_view.py's Phase-1
visual skeleton (see its module docstring): plain, non-Qt presentational
helper functions get ordinary unit tests; LiveViewWidget itself gets
pytest-qt smoke tests run offscreen (QT_QPA_PLATFORM=offscreen) --
window launches, expected child widgets exist, the degree selector
changes state. Nothing here exercises real camera/preprocessing/analysis
logic, matching the skeleton itself.

PySide6/pyqtgraph/pytest-qt are not (yet) declared in
requirements.txt/pyproject.toml (both deliberately empty placeholders --
see CLAUDE.md), so this whole module is skipped, not failed, wherever
they aren't installed -- the same "gate, don't require" pattern
tests/test_acquisition.py uses for hardware-only tests.
"""

# Imports

import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pg = pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

# Must be set before any QApplication is constructed -- pytest-qt creates
# one lazily the first time a test requests the qtbot fixture.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pipeline.acquisition import CameraStream, SyntheticBackend, CANONICAL_SHAPE
from pipeline.analysis import SensorNoiseModel
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.preprocessing import CalibrationSet
from pipeline.gui.live_view import (
    DEFAULT_DEGREE,
    DEGREE_CHOICES,
    LiveViewWidget,
    heatmap_x_extent,
    wavelength_axis_label,
)

# Constants

FIXTURE_EXPOSURE_US = 2000.0
FIXTURE_GAIN_DB = 0.0


# Classes

class _FakeWavelengthAxis:

    '''Minimal WavelengthAxis stand-in -- a trivial linear pixel->nm map.'''

    def wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:
        return 500.0 + 0.01 * np.asarray(pixel, dtype=float)

    def sigma_wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(pixel, dtype=float), 0.05)


# Functions

def _calibration_set() -> CalibrationSet:
    '''Mirrors tests/test_preprocessing.py's _make_clean_calibration_set().'''
    baseline = np.full(CANONICAL_SHAPE, 10.0, dtype=np.float64)
    flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
    bad_pixel_mask = np.zeros(CANONICAL_SHAPE, dtype=bool)
    record = CalibrationRecord(
        exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
        timestamp=time.time(), source_frame_count=50,
    )
    return CalibrationSet(
        baseline=baseline, baseline_record=record,
        flat_field=flat_field, flat_field_record=record,
        bad_pixel_mask=bad_pixel_mask, background_sigma=1.0,
    )


def _camera_stream() -> CameraStream:
    '''An unstarted CameraStream over SyntheticBackend -- never polled by these tests.'''
    return CameraStream(
        exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
        pixel_format="Mono8", timeout_ms=5000, backend=SyntheticBackend(seed=0),
    )


def _make_widget(qtbot, wavelength_axis=None) -> LiveViewWidget:
    widget = LiveViewWidget(
        calibration_set=_calibration_set(),
        noise_model=SensorNoiseModel(gain_e_per_adu=2.2, background_sigma=1.0),
        position_calibration=ScaleFactorPositionCalibration(),
        wavelength_axis=wavelength_axis,
        camera_stream=_camera_stream(),
    )
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# live_view.py -- plain (non-Qt) helper functions
# ---------------------------------------------------------------------------

class TestWavelengthAxisLabel:

    def test_none_gives_pixel_fallback_label(self):
        label = wavelength_axis_label(None)
        assert "pixel column" in label.lower()
        assert "not yet available" in label.lower()

    def test_real_axis_gives_wavelength_label(self):
        assert wavelength_axis_label(_FakeWavelengthAxis()) == "Wavelength (nm)"


class TestHeatmapXExtent:

    def test_none_returns_raw_pixel_column_extent(self):
        assert heatmap_x_extent(None, 10, 1909) == (10.0, 1909.0)

    def test_real_axis_returns_wavelength_endpoints(self):
        axis = _FakeWavelengthAxis()
        x0, x1 = heatmap_x_extent(axis, 0, 1919)
        assert x0 == pytest.approx(500.0)
        assert x1 == pytest.approx(500.0 + 0.01 * 1919)


# ---------------------------------------------------------------------------
# LiveViewWidget -- pytest-qt smoke tests (offscreen)
# ---------------------------------------------------------------------------

class TestLiveViewWidgetSmoke:

    def test_widget_constructs_and_shows(self, qtbot):
        widget = _make_widget(qtbot)
        widget.resize(800, 600)
        widget.show()
        qtbot.waitExposed(widget)
        assert widget.isVisible()

    def test_core_child_widgets_exist(self, qtbot):
        widget = _make_widget(qtbot)
        assert widget._main_plot is not None
        assert widget._strip_plot is not None
        assert widget._image_item is not None
        assert widget._scatter is not None
        assert widget._fit_curve is not None
        assert widget._degree_selector.count() == len(DEGREE_CHOICES)
        assert widget._extended_measurement_button is not None
        assert widget._chi_squared_label.text() != "--"

    def test_degree_selector_defaults_to_linear(self, qtbot):
        widget = _make_widget(qtbot)
        assert widget._degree_selector.currentData() == DEFAULT_DEGREE
        assert widget._current_degree == DEFAULT_DEGREE
        assert widget._zeta_note_label.text() == ""

    def test_degree_selector_changes_state(self, qtbot):
        widget = _make_widget(qtbot)
        quadratic_index = DEGREE_CHOICES.index(2)

        widget._degree_selector.setCurrentIndex(quadratic_index)

        assert widget._current_degree == 2
        assert widget._degree_selector.currentData() == 2
        # Degree > 1 must show the "no internal uncertainty" note (see
        # module docstring / SIDE INFO PANEL spec) -- never a fabricated sigma.
        assert widget._zeta_note_label.text() != ""
        assert "uncertainty not available" in widget._zeta_note_label.text().lower()

    def test_degree_selector_does_not_touch_camera_or_analysis(self, qtbot):
        # Phase-1 guarantee: changing degree must not start/poll the
        # camera stream -- it only swaps pre-baked placeholder text.
        widget = _make_widget(qtbot)
        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(3))
        assert not widget._camera_stream.is_running

    def test_wavelength_axis_none_uses_pixel_fallback_label(self, qtbot):
        widget = _make_widget(qtbot, wavelength_axis=None)
        axis_label = widget._main_plot.getAxis("bottom").labelText
        assert "not yet available" in axis_label.lower()

    def test_wavelength_axis_present_uses_wavelength_label(self, qtbot):
        widget = _make_widget(qtbot, wavelength_axis=_FakeWavelengthAxis())
        axis_label = widget._main_plot.getAxis("bottom").labelText
        assert axis_label == "Wavelength (nm)"

    def test_extended_measurement_button_present_but_unwired(self, qtbot):
        # Non-goal per spec: the button exists as a landing spot for a
        # future feature, but must not be connected to anything yet.
        widget = _make_widget(qtbot)
        button = widget._extended_measurement_button
        assert button.text() == "Extended Measurement..."
        assert button.receivers("2clicked()") == 0
