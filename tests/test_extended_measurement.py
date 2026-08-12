'''
Test suite for extended_measurement.py, as widget-layer smoke tests plus
real acquire/preprocess/analyze/combine wiring tests. Split out of
tests/test_gui.py -- see tests/test_calibration_screen.py's module
docstring for why.

TestExtendedMeasurementScreenSmoke/TestExtendedMeasurementAcquisitionSettingsPanel
check structure/state transitions against a plain, hand-built
CalibrationSet (see _calibration_set()) -- most don't need real camera
data at all, and the one that does (clicking "Run Measurement") only
checks that acquiring the requested count actually happened, not
particular numeric results.
TestExtendedMeasurementRealMeasurement exercises the real pipeline
end-to-end against build_realistic_calibration_bundle()'s real (non-
placeholder) calibration artifacts (see gui_fixture_helpers.py), including
a known-injected-chirp correctness check.

PySide6/pyqtgraph/pytest-qt are local-only dependencies (see CLAUDE.md --
pyproject.toml/requirements.txt are deliberately left empty), so this
whole module is skipped, not failed, wherever they aren't installed --
the same "gate, don't require" pattern tests/test_acquisition.py uses for
hardware-only tests.
'''

# Imports

import dataclasses
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

from gui_fixture_helpers import (  # noqa: E402
    build_realistic_calibration_bundle,
    FIXTURE_EXPOSURE_US as REALISTIC_FIXTURE_EXPOSURE_US,
    FIXTURE_GAIN_DB as REALISTIC_FIXTURE_GAIN_DB,
)

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


def _make_extended_measurement_widget(
    qtbot, wavelength_axis=None, camera_stream=None
) -> ExtendedMeasurementScreen:
    widget = ExtendedMeasurementScreen(
        calibration_set=_calibration_set(),
        noise_model=SensorNoiseModel(gain_e_per_adu=2.2, background_sigma=1.0),
        position_calibration=ScaleFactorPositionCalibration(),
        wavelength_axis=wavelength_axis,
        camera_stream=camera_stream if camera_stream is not None else _camera_stream(),
    )
    qtbot.addWidget(widget)
    return widget


def _make_widget_from_bundle(qtbot, bundle, camera_stream) -> ExtendedMeasurementScreen:

    '''
    Constructs ExtendedMeasurementScreen from a real CalibrationBundle
    (see gui_fixture_helpers.py). That fixture's dark frames are perfectly
    uniform (no injected noise -- see its own module docstring), so
    baseline_result.background_sigma, and therefore
    calibration_set.background_sigma, comes out exactly 0.0 -- degenerate
    input for run_preprocessing()'s signal-threshold masking step, which
    requires a strictly positive value (preprocessing/steps/
    signal_threshold.py). Patched here to a small representative value via
    dataclasses.replace() rather than editing that read-only fixture file.
    noise_model.background_sigma (the separate quantity the Thompson-
    Larson-Webb centroid-uncertainty formula uses) is left exactly as the
    fixture provides it -- zero is a valid, if approximated, input there.
    '''

    calibration_set = dataclasses.replace(bundle.calibration_set, background_sigma=1.0)
    widget = ExtendedMeasurementScreen(
        calibration_set=calibration_set,
        noise_model=bundle.noise_model,
        position_calibration=bundle.position_calibration,
        wavelength_axis=bundle.wavelength_axis,
        camera_stream=camera_stream,
        conversion_gain_record=bundle.conversion_gain_record,
    )
    qtbot.addWidget(widget)
    return widget


def _realistic_running_camera_stream(slope_px_per_col: float = 0.0, seed: int = 5) -> CameraStream:
    '''
    A started CameraStream, paired with build_realistic_calibration_bundle()'s
    own fixture settings (matching exposure/gain avoids a spurious
    SettingsMismatchError from run_preprocessing()'s baseline check).
    peak_counts is kept well under Mono8's 255 ceiling -- the same
    reasoning as that bundle's own dark=10/illuminated=150 levels -- with
    an optional known injected chirp for wiring-correctness checks. Caller
    must stop() it (no fixture-managed teardown, since a few tests below
    need a custom slope_px_per_col per call).
    '''
    stream = CameraStream(
        exposure_us=REALISTIC_FIXTURE_EXPOSURE_US, gain_db=REALISTIC_FIXTURE_GAIN_DB,
        pixel_format="Mono8", timeout_ms=5000,
        backend=SyntheticBackend(
            seed=seed, slope_px_per_col=slope_px_per_col, peak_counts=180.0, noise_std=3.0,
        ),
    )
    stream.start()
    return stream


@pytest.fixture
def running_camera_stream():
    '''
    A started CameraStream over a plain SyntheticBackend (no injected
    chirp), matching _calibration_set()'s fixture settings -- for tests
    that need collect_n_frames() to succeed against a genuinely running
    stream (unlike _camera_stream(), which is deliberately never started
    -- see its own docstring). Stopped in teardown so no background grab
    thread leaks across tests.
    '''
    stream = CameraStream(
        exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
        pixel_format="Mono8", timeout_ms=5000,
        backend=SyntheticBackend(seed=3, peak_counts=180.0, noise_std=3.0),
    )
    stream.start()
    yield stream
    stream.stop()


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

    def test_run_button_updates_n_shots_label(self, qtbot, monkeypatch, running_camera_stream):
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        widget = _make_extended_measurement_widget(qtbot, camera_stream=running_camera_stream)
        widget._n_shots_spin.setValue(4)

        widget._run_button.click()

        assert widget._n_shots_label.text() == "4"

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


# ---------------------------------------------------------------------------
# extended_measurement.py -- real acquire/preprocess/analyze/combine wiring
# (pytest-qt, offscreen)
# ---------------------------------------------------------------------------

class TestExtendedMeasurementRealMeasurement:

    '''
    Exercises the real acquire -> preprocess -> analyze -> combine ->
    render path "Run Measurement" now runs, against
    build_realistic_calibration_bundle()'s real (non-placeholder)
    calibration artifacts and a real, started CameraStream. Every test
    here mocks QMessageBox.warning defensively (see this module's
    docstring's CRITICAL note) even where no drift is expected, since a
    real analysis run could plausibly cross a threshold unexpectedly.
    '''

    def test_run_measurement_populates_real_combined_result(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)

            widget._run_button.click()

            assert widget._shot_results is not None
            assert len(widget._shot_results) == 3
            assert widget._n_shots_label.text() == "3"
            assert "±" in widget._spatial_dispersion_label.text()
            assert widget._reduced_chi_squared_label.text() != "--"
            assert len(widget._scatter.data) > 0
            fit_x, fit_y = widget._fit_curve.getData()
            assert fit_x is not None and len(fit_x) > 0
            assert len(widget._residual_scatter.data) > 0
            # Acquisition Settings left untouched -> no camera reconfigure.
            assert stream.exposure_us == pytest.approx(REALISTIC_FIXTURE_EXPOSURE_US)
        finally:
            stream.stop()

    def test_run_measurement_reports_known_injected_slope(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        slope_px_per_col = 0.01
        # build_realistic_calibration_bundle()'s wavelength_axis is linear
        # with dwavelength_nm/dcolumn = 0.05 (coefficients=[500.0, 0.05]),
        # so a frame with injected dx0/dcolumn = slope_px_per_col implies a
        # known zeta = slope_px_per_col / 0.05 -- a real end-to-end check
        # that the wiring reports the right physical quantity, not just
        # "some number".
        expected_zeta = slope_px_per_col / 0.05
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream(slope_px_per_col=slope_px_per_col, seed=11)
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)

            widget._run_button.click()

            combined = widget._compute_combined_result(1)
            assert combined.zeta_combined == pytest.approx(expected_zeta, rel=0.3)
        finally:
            stream.stop()

    def test_degree_switch_after_run_does_not_reacquire(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            widget._run_button.click()
            shot_results = widget._shot_results
            n_shots_text = widget._n_shots_label.text()

            widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))

            assert widget._shot_results is shot_results
            assert widget._n_shots_label.text() == n_shots_text
            assert widget._spatial_dispersion_label.text() != "--"
            assert widget._degree_note_label.text() != ""
        finally:
            stream.stop()

    def test_evaluated_at_change_recombines_for_degree_gt_one(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            widget._run_button.click()
            widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))

            widget._evaluated_at_spin.setValue(200.0)

            combined = widget._compute_combined_result(2)
            assert "nm" in widget._evaluated_at_label.text()
            assert combined.sigma_zeta_combined > 0
        finally:
            stream.stop()

    def test_run_reconfigures_camera_when_exposure_changed(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            # Small enough to stay within EXPOSURE_MATCH_TOLERANCE_REL
            # (1%) so this doesn't also trip the baseline-drift path.
            new_exposure_us = REALISTIC_FIXTURE_EXPOSURE_US * 1.002
            widget._exposure_spin.setValue(new_exposure_us)

            widget._run_button.click()

            assert stream.exposure_us == pytest.approx(new_exposure_us)
            assert stream.is_running is True
            assert widget._shot_results is not None
        finally:
            stream.stop()
