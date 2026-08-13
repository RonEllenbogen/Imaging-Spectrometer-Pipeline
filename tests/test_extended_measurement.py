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
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 is a local-only GUI dependency")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is a local-only GUI dependency")
pytest.importorskip("pytestqt", reason="pytest-qt is a local-only GUI dependency")

# Must be set before any QApplication is constructed -- pytest-qt creates
# one lazily the first time a test requests the qtbot fixture.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QDoubleSpinBox, QGroupBox, QSpinBox  # noqa: E402

from pipeline.acquisition import (  # noqa: E402
    CameraConnectionError, CameraStream, SyntheticBackend, CANONICAL_SHAPE,
)
from pipeline.analysis import InsufficientDataError, SensorNoiseModel  # noqa: E402
from pipeline.calibration.shared import CalibrationRecord, GAIN_MATCH_TOLERANCE_ABS  # noqa: E402
from pipeline.calibration.spatial import ScaleFactorPositionCalibration  # noqa: E402
from pipeline.cli.calibration import DEFAULT_ARTIFACT_DIR  # noqa: E402
from pipeline.preprocessing import CalibrationSet  # noqa: E402
from pipeline.gui.extended_measurement import (  # noqa: E402
    DEFAULT_N_SHOTS,
    FIT_CURVE_N_POINTS,
    ExtendedMeasurementScreen,
    compute_combined_result_for_degree,
    compute_fit_line_and_residuals,
)
from pipeline.gui.formatting import format_value_with_uncertainty, MICRONS_PER_MM  # noqa: E402
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
            "Spectral ROI",
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
        assert widget._spectral_roi_control is not None
        assert widget._degree_selector.count() == len(DEGREE_CHOICES)
        assert widget._n_shots_label.text() != ""
        assert widget._coefficients_label.text() != ""
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

    def test_column_bounds_are_applied_to_real_measurement(self, qtbot, monkeypatch):
        # Regression test for the "Run Measurement" wiring of
        # column_bounds=self._spectral_roi_control.column_bounds(): narrow
        # the spectral ROI before running, then confirm every shot's
        # centroid columns actually fall inside the window -- i.e.
        # apply_spectral_roi()'s valid_columns override reached
        # extract_centroids(), not just the unmasked automatic SNR gate.
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            widget._spectral_roi_control._min_spin.setValue(300)
            widget._spectral_roi_control._max_spin.setValue(1000)

            widget._run_button.click()

            assert widget._shot_results is not None
            assert widget._measurement_columns.size > 0
            assert np.all(
                (widget._measurement_columns >= 300) & (widget._measurement_columns < 1000)
            )
        finally:
            stream.stop()

    def test_run_measurement_insufficient_data_aborts_without_partial_update(
        self, qtbot, monkeypatch
    ):
        # Regression test: analyze_shot() (via TotalLeastSquaresFit) can
        # raise InsufficientDataError for a genuinely marginal shot -- a
        # real session hit exactly this (an uncaught crash) via
        # LiveViewWidget's polling loop; this screen had the identical
        # unguarded analyze_shot() call with no protection at all, a
        # button click away from the same crash. Confirms the fix: the
        # error is caught, reported via QMessageBox, and the run aborts
        # cleanly -- no partial/inconsistent shot_results.
        warning_calls = []
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning",
            lambda *a, **k: warning_calls.append(a),
        )

        def _raise_insufficient_data(*args, **kwargs):
            raise InsufficientDataError(degree=3, n_points=4)

        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.analyze_shot", _raise_insufficient_data
        )

        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            assert widget._shot_results is None

            widget._run_button.click()

            assert widget._shot_results is None
            assert widget._n_shots_label.text() == "--"
            assert len(warning_calls) == 1
            title, message = warning_calls[0][1], warning_calls[0][2]
            assert title == "Measurement Failed"
            assert "could not be analyzed" in message
        finally:
            stream.stop()

    def test_run_measurement_camera_error_during_acquisition_aborts_cleanly(
        self, qtbot, monkeypatch
    ):
        # Regression test: collect_n_frames() had zero exception handling
        # -- a real camera dropping mid-acquisition (a plausible lab event:
        # a cable wiggle, a GigE hiccup) would raise CameraError/RuntimeError
        # straight out of this button's click handler. Confirms the fix:
        # routed through show_camera_error_dialog(), the same convention
        # every other real camera-touching call in this codebase uses, run
        # aborted with no partial shot_results update.
        camera_error_calls = []
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.show_camera_error_dialog",
            lambda parent, message: camera_error_calls.append(message),
        )

        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)

            def _raise(*a, **k):
                raise CameraConnectionError("device disconnected mid-acquisition")

            monkeypatch.setattr(stream, "collect_n_frames", _raise)

            widget._run_button.click()

            assert widget._shot_results is None
            assert widget._n_shots_label.text() == "--"
            assert len(camera_error_calls) == 1
            assert "disconnected" in camera_error_calls[0]
        finally:
            stream.stop()

    def test_run_measurement_camera_error_on_reconfigure_aborts_cleanly(
        self, qtbot, monkeypatch
    ):
        # Regression test: _maybe_reconfigure_camera_stream()'s .start()
        # call (after stopping the stream to apply new exposure/gain) was
        # likewise unguarded. Confirms the fix catches a failed restart
        # the same way.
        #
        # Setting the exposure spin below to something that actually
        # differs (needed to make _maybe_reconfigure_camera_stream() take
        # its real branch at all) can itself cross the settings-drift
        # threshold against calibration_set.baseline_record and pop a
        # real QMessageBox.warning() -- a real (unmocked) modal
        # QMessageBox.exec() blocks forever offscreen, so this is mocked
        # defensively here the same way every other exposure/gain-editing
        # test in this file already does.
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        camera_error_calls = []
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.show_camera_error_dialog",
            lambda parent, message: camera_error_calls.append(message),
        )

        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            # Force a real mismatch so _maybe_reconfigure_camera_stream()
            # actually attempts a stop/reconfigure/restart cycle --
            # _maybe_reconfigure_camera_stream()'s own match check uses
            # rel_tol=1e-9, so any nonzero difference is enough. Kept well
            # under EXPOSURE_MATCH_TOLERANCE_REL (1%, i.e. ~20us at this
            # fixture's 2000us) so this doesn't *also* cross the
            # settings-drift threshold -- that's a separate concern from
            # what this test is checking, and would otherwise change
            # _n_shots_label's text to "N/A" (drifted) instead of "--"
            # (never-run) for a reason unrelated to the camera error here.
            widget._exposure_spin.setValue(stream.exposure_us + 1.0)

            def _raise(*a, **k):
                raise CameraConnectionError("device did not respond to restart")

            monkeypatch.setattr(stream, "start", _raise)

            widget._run_button.click()

            assert widget._shot_results is None
            assert widget._n_shots_label.text() == "--"
            assert len(camera_error_calls) == 1
            assert "restart" in camera_error_calls[0]
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

            # The displayed "Spatial Dispersion" label must be zeta_combined
            # converted to physical units (mm/nm) via the widget's own
            # ScaleFactorPositionCalibration -- not the raw px/nm value
            # _compute_combined_result() returns internally. Recomputes the
            # expected conversion independently (PIXEL_PITCH_UM * scale_factor,
            # microns -> mm) rather than calling widget._zeta_to_mm() itself,
            # so this actually exercises the wiring rather than restating it.
            expected_zeta_mm, expected_sigma_mm = ScaleFactorPositionCalibration().convert(
                np.array([combined.zeta_combined]), np.array([combined.sigma_zeta_combined])
            )
            expected_zeta_mm = float(expected_zeta_mm[0]) / MICRONS_PER_MM
            expected_sigma_mm = float(expected_sigma_mm[0]) / MICRONS_PER_MM
            assert widget._spatial_dispersion_label.text() == format_value_with_uncertainty(
                expected_zeta_mm, expected_sigma_mm
            )
            # Sanity check this isn't just approving whatever the raw px/nm
            # text would already have been -- the conversion factor here
            # (PIXEL_PITCH_UM * 1.5 / 1000 ~ 0.005) is nowhere near 1.
            assert widget._spatial_dispersion_label.text() != format_value_with_uncertainty(
                combined.zeta_combined, combined.sigma_zeta_combined
            )
        finally:
            stream.stop()

    def test_coefficients_label_reports_c0_and_c1(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream(slope_px_per_col=0.01, seed=11)
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)

            widget._run_button.click()

            text = widget._coefficients_label.text()
            assert "c<sub>0</sub>" in text
            assert "c<sub>1</sub>" in text

            # c1 must be exactly the same number already shown as "Spatial
            # Dispersion" -- they're the same combined zeta, just surfaced
            # in two places (see _build_combined_result_group()'s docstring).
            assert widget._spatial_dispersion_label.text() in text

            # c0 (the intercept) is independently recomputable from the
            # exact same weighted-mean formula _recompute_fit_and_residuals()
            # uses -- this exercises the actual wiring rather than trusting
            # whatever self._measurement_intercept_px already holds.
            combined = widget._compute_combined_result(1)
            weights = 1.0 / widget._measurement_sigma_x0_px ** 2
            sum_weights = np.sum(weights)
            expected_intercept_px = float(
                np.sum(
                    weights * (
                        widget._measurement_x0_px
                        - combined.zeta_combined * widget._measurement_wavelength_nm
                    )
                ) / sum_weights
            )
            expected_intercept_sigma_px = float(np.sqrt(1.0 / sum_weights))
            assert widget._measurement_intercept_px == pytest.approx(expected_intercept_px)
            assert widget._measurement_intercept_sigma_px == pytest.approx(expected_intercept_sigma_px)

            expected_c0_mm, expected_c0_sigma_mm = ScaleFactorPositionCalibration().convert(
                np.array([expected_intercept_px]), np.array([expected_intercept_sigma_px])
            )
            expected_c0_text = format_value_with_uncertainty(
                float(expected_c0_mm[0]) / MICRONS_PER_MM, float(expected_c0_sigma_mm[0]) / MICRONS_PER_MM
            )
            assert f"c<sub>0</sub> = {expected_c0_text}" in text
        finally:
            stream.stop()

    def test_coefficients_change_with_degree(self, qtbot, monkeypatch):
        # The user-facing requirement: switching the fit degree must
        # recompute the displayed coefficients, not just leave them stale
        # from whichever degree was selected at "Run Measurement" time.
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            widget._run_button.click()
            degree_1_text = widget._coefficients_label.text()

            widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))
            degree_2_text = widget._coefficients_label.text()

            assert degree_2_text != "--"
            assert degree_2_text != degree_1_text
        finally:
            stream.stop()

    def test_coefficients_label_resets_when_no_measurement(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        assert widget._coefficients_label.text() == "--"

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


# ---------------------------------------------------------------------------
# extended_measurement.py -- "Save Record" (pytest-qt, offscreen)
# ---------------------------------------------------------------------------

class TestExtendedMeasurementSaveRecord:

    def test_save_button_disabled_until_measurement_exists(self, qtbot):
        widget = _make_extended_measurement_widget(qtbot)
        assert widget._save_record_button.isEnabled() is False

    def test_save_button_enabled_after_successful_run(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)

            widget._run_button.click()

            assert widget._save_record_button.isEnabled() is True
        finally:
            stream.stop()

    def test_save_button_stays_enabled_after_a_later_failed_run(self, qtbot, monkeypatch):
        # A failed re-run leaves shot_results (and therefore the ability
        # to save the previous successful run) untouched -- see
        # _on_run_clicked()'s own docstring ("display is left exactly as
        # it was before this click").
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            widget._run_button.click()
            assert widget._save_record_button.isEnabled() is True

            def _raise(*a, **k):
                raise InsufficientDataError(degree=3, n_points=4)

            monkeypatch.setattr("pipeline.gui.extended_measurement.analyze_shot", _raise)
            widget._run_button.click()

            assert widget._save_record_button.isEnabled() is True
        finally:
            stream.stop()

    def test_save_record_uses_roi_captured_at_run_time_not_live_controls(self, qtbot, monkeypatch):
        # Regression test for the correctness fix documented in
        # _set_measurement_data()/_on_run_clicked(): editing a ROI control
        # after "Run Measurement" but before "Save Record" must not change
        # what gets reported for that already-completed measurement.
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.information", lambda *a, **k: None
        )
        save_calls = []

        def _fake_save_measurement_record(*args, **kwargs):
            save_calls.append((args, kwargs))
            return Path("/fake/record/dir")

        monkeypatch.setattr(
            "pipeline.gui.measurement_record.save_measurement_record", _fake_save_measurement_record
        )

        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            widget._spectral_roi_control._min_spin.setValue(300)
            widget._spectral_roi_control._max_spin.setValue(1000)

            widget._run_button.click()
            run_time_roi_bounds_px = widget._measurement_roi_bounds_px
            run_time_column_bounds = widget._measurement_column_bounds
            assert run_time_column_bounds == (300, 1000)

            # Edit the spectral ROI AFTER the run, before saving.
            widget._spectral_roi_control._min_spin.setValue(50)
            widget._spectral_roi_control._max_spin.setValue(1900)
            assert widget._spectral_roi_control.column_bounds() != run_time_column_bounds

            widget._save_record_button.click()

            assert len(save_calls) == 1
            args, _kwargs = save_calls[0]
            # roi_bounds_px is save_measurement_record()'s 8th positional
            # parameter, spectral_column_bounds its 9th.
            assert args[7] == run_time_roi_bounds_px
            assert args[8] == run_time_column_bounds
        finally:
            stream.stop()

    def test_save_record_calls_through_with_current_measurement(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        info_calls = []
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.information",
            lambda *a, **k: info_calls.append(a),
        )
        save_calls = []

        def _fake_save_measurement_record(*args, **kwargs):
            save_calls.append((args, kwargs))
            return Path("/fake/record/dir")

        monkeypatch.setattr(
            "pipeline.gui.measurement_record.save_measurement_record", _fake_save_measurement_record
        )

        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream()
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            widget._run_button.click()

            widget._save_record_button.click()

            assert len(save_calls) == 1
            args, kwargs = save_calls[0]
            assert args[0] is widget._shot_results
            assert args[1] is widget._measurement_stacked_image
            assert args[2] is widget._measurement_representative_frames
            assert kwargs["artifact_dir"] == DEFAULT_ARTIFACT_DIR
            assert len(info_calls) == 1
            assert "/fake/record/dir" in info_calls[0][2]
        finally:
            stream.stop()

    def test_extracted_helpers_match_widget_state(self, qtbot, monkeypatch):
        # Confirms the free-function extraction refactor
        # (compute_combined_result_for_degree()/compute_fit_line_and_residuals())
        # produces exactly the same numbers the widget itself displays --
        # calling the free functions directly and comparing against the
        # widget's own cached state, rather than trusting that the earlier
        # per-degree/per-coefficient tests above merely still pass.
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *a, **k: None
        )
        bundle = build_realistic_calibration_bundle()
        stream = _realistic_running_camera_stream(slope_px_per_col=0.01, seed=11)
        try:
            widget = _make_widget_from_bundle(qtbot, bundle, stream)
            widget._n_shots_spin.setValue(3)
            widget._run_button.click()

            combined = compute_combined_result_for_degree(
                widget._shot_results, 1, widget._evaluated_at_wavelength_nm()
            )
            widget_combined = widget._compute_combined_result(1)
            assert combined.zeta_combined == widget_combined.zeta_combined
            assert combined.sigma_zeta_combined == widget_combined.sigma_zeta_combined

            intercept_px, intercept_sigma_px, _fit_x, _fit_y_px, _residual_px = compute_fit_line_and_residuals(
                widget._measurement_x0_px, widget._measurement_sigma_x0_px, widget._measurement_wavelength_nm,
                combined.zeta_combined,
                (float(widget._measurement_x_values.min()), float(widget._measurement_x_values.max())),
                FIT_CURVE_N_POINTS,
            )
            assert intercept_px == pytest.approx(widget._measurement_intercept_px)
            assert intercept_sigma_px == pytest.approx(widget._measurement_intercept_sigma_px)
        finally:
            stream.stop()
