'''
Test suite for extended_measurement.py's Phase-1 visual skeleton, as
widget-layer smoke tests. Split out of tests/test_gui.py -- see
tests/test_calibration_screen.py's module docstring for why.

Most of this widget has no real camera/analysis call wired in yet (see
extended_measurement.py's own module docstring), so most of these tests
only check structure/state transitions -- not behavior, e.g. that
clicking "Run Measurement" updates the n_shots label, not that it
acquires any real frames. A follow-up pass adds the remaining real-logic
tests once the real acquisition/combine_shots()/sigma_zeta() calls are
wired.

PySide6/pyqtgraph/pytest-qt are local-only dependencies (see CLAUDE.md --
pyproject.toml/requirements.txt are deliberately left empty), so this
whole module is skipped, not failed, wherever they aren't installed --
the same "gate, don't require" pattern tests/test_acquisition.py uses for
hardware-only tests.
'''

# Imports

import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 is a local-only GUI dependency")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is a local-only GUI dependency")
pytest.importorskip("pytestqt", reason="pytest-qt is a local-only GUI dependency")

# Must be set before any QApplication is constructed -- pytest-qt creates
# one lazily the first time a test requests the qtbot fixture.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QDoubleSpinBox, QGroupBox, QSpinBox  # noqa: E402

from pipeline.acquisition import CameraStream, SyntheticBackend, CANONICAL_SHAPE  # noqa: E402
from pipeline.analysis import SensorNoiseModel  # noqa: E402
from pipeline.calibration.shared import CalibrationRecord, GAIN_MATCH_TOLERANCE_ABS  # noqa: E402
from pipeline.calibration.spatial import ScaleFactorPositionCalibration  # noqa: E402
from pipeline.preprocessing import CalibrationSet  # noqa: E402
from pipeline.gui.extended_measurement import (  # noqa: E402
    DEFAULT_N_SHOTS,
    ExtendedMeasurementScreen,
)
from pipeline.gui.live_view import DEFAULT_DEGREE, DEGREE_CHOICES  # noqa: E402

# Constants

FIXTURE_EXPOSURE_US = 2000.0
FIXTURE_GAIN_DB = 0.0

# Classes

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


def _make_extended_measurement_widget(qtbot, wavelength_axis=None) -> ExtendedMeasurementScreen:
    widget = ExtendedMeasurementScreen(
        calibration_set=_calibration_set(),
        noise_model=SensorNoiseModel(gain_e_per_adu=2.2, background_sigma=1.0),
        position_calibration=ScaleFactorPositionCalibration(),
        wavelength_axis=wavelength_axis,
        camera_stream=_camera_stream(),
    )
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# extended_measurement.py -- ExtendedMeasurementScreen pytest-qt smoke tests
# (offscreen)
# ---------------------------------------------------------------------------

class TestExtendedMeasurementScreenSmoke:

    def test_widget_constructs_and_shows(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        widget.resize(800, 600)
        widget.show()
        qtbot.waitExposed(widget)
        assert widget.isVisible()

    def test_expected_group_boxes_present(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        titles = {box.title() for box in widget.findChildren(QGroupBox)}
        assert {
            "Run Configuration",
            "Acquisition Settings",
            "Spatial ROI",
            "Fit Degree",
            "Combined Result",
        } <= titles

    def test_core_child_widgets_exist(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        assert isinstance(widget._n_shots_spin, QSpinBox)
        assert widget._n_shots_spin.value() == DEFAULT_N_SHOTS
        assert widget._run_button is not None
        assert widget._main_plot is not None
        assert widget._residual_plot is not None
        assert widget._scatter is not None
        assert widget._error_bars is not None
        assert widget._fit_curve is not None
        assert widget._residual_scatter is not None
        assert isinstance(widget._exposure_spin, QDoubleSpinBox)
        assert isinstance(widget._gain_spin, QDoubleSpinBox)
        assert widget._roi_control is not None
        assert widget._degree_selector.count() == len(DEGREE_CHOICES)
        assert widget._n_shots_label.text() != ""
        assert widget._spatial_dispersion_label.text() != ""
        assert widget._reduced_chi_squared_label.text() != ""
        assert widget._back_button is not None

    def test_run_button_updates_n_shots_label(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        widget._n_shots_spin.setValue(42)

        widget._run_button.click()

        assert widget._n_shots_label.text() == "42"

    def test_degree_one_default_has_no_note(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        assert widget._degree_selector.currentData() == DEFAULT_DEGREE
        assert widget._degree_note_label.text() == ""

    def test_degree_greater_than_one_populates_note(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))
        assert widget._degree_note_label.text() != ""

    def test_evaluate_at_row_hidden_for_degree_one_shown_for_degree_gt_one(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        assert widget._combined_result_form.isRowVisible(widget._evaluated_at_spin) is False
        assert widget._combined_result_form.isRowVisible(widget._evaluated_at_label) is False

        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))

        assert widget._combined_result_form.isRowVisible(widget._evaluated_at_spin) is True
        assert widget._combined_result_form.isRowVisible(widget._evaluated_at_label) is True
        assert widget._evaluated_at_label.text() != ""

    def test_editing_evaluate_at_updates_readout_and_survives_degree_roundtrip(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))

        widget._evaluated_at_spin.setValue(500.0)
        assert "500" in widget._evaluated_at_label.text()

        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(3))
        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))

        assert widget._evaluated_at_spin.value() == pytest.approx(500.0)

    def test_back_button_emits_signal(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        with qtbot.waitSignal(widget.back_requested, timeout=1000):
            widget._back_button.click()


# ---------------------------------------------------------------------------
# extended_measurement.py -- Acquisition Settings panel (pytest-qt, offscreen)
# ---------------------------------------------------------------------------

class TestExtendedMeasurementAcquisitionSettingsPanel:

    def test_fields_prefilled_from_baseline_record(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        assert widget._exposure_spin.value() == pytest.approx(FIXTURE_EXPOSURE_US)
        assert widget._gain_spin.value() == pytest.approx(FIXTURE_GAIN_DB)

    def test_drifted_state_nas_combined_result_and_hides_overlay_and_warns_once(
        self, qtbot, monkeypatch
    ):
        warning_calls = []
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning",
            lambda *args, **kwargs: warning_calls.append(args),
        )
        widget = _make_extended_measurement_widget(qtbot)
        drifted_gain_db = FIXTURE_GAIN_DB + GAIN_MATCH_TOLERANCE_ABS * 2

        widget._gain_spin.setValue(drifted_gain_db)

        assert widget._settings_drifted is True
        assert widget._n_shots_label.text() == "N/A"
        assert widget._spatial_dispersion_label.text() == "N/A"
        assert widget._reduced_chi_squared_label.text() == "N/A"
        assert widget._degree_note_label.text() == ""
        assert widget._scatter.isVisible() is False
        assert widget._error_bars.isVisible() is False
        assert widget._fit_curve.isVisible() is False
        assert widget._residual_scatter.isVisible() is False
        assert len(warning_calls) == 1

    def test_exiting_drifted_state_restores_combined_result_and_overlay(
        self, qtbot, monkeypatch
    ):
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning",
            lambda *args, **kwargs: None,
        )
        widget = _make_extended_measurement_widget(qtbot)
        drifted_gain_db = FIXTURE_GAIN_DB + GAIN_MATCH_TOLERANCE_ABS * 2

        widget._gain_spin.setValue(drifted_gain_db)
        assert widget._settings_drifted is True

        widget._gain_spin.setValue(FIXTURE_GAIN_DB)

        assert widget._settings_drifted is False
        assert widget._n_shots_label.text() != "N/A"
        assert widget._spatial_dispersion_label.text() != "N/A"
        assert widget._reduced_chi_squared_label.text() != "N/A"
        assert widget._scatter.isVisible() is True
        assert widget._error_bars.isVisible() is True
        assert widget._fit_curve.isVisible() is True
        assert widget._residual_scatter.isVisible() is True
