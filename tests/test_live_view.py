'''
Test suite for live_view.py's (and roi_control.py's) Phase-1 visual
skeleton, as widget-layer smoke tests plus ordinary unit tests of
live_view.py's plain, non-Qt presentational helper functions. Split out
of tests/test_gui.py -- see tests/test_calibration_screen.py's module
docstring for why.

Most of this widget has no real camera/analysis call wired in yet (see
live_view.py's own module docstring), so most of these tests only check
structure/state transitions -- not behavior, e.g. that degree selection
changes LiveViewWidget's *displayed* placeholder numbers, not that it
triggers a real refit. A follow-up pass adds the remaining real-logic
tests (the QTimer-driven update loop) once it's wired.

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

from PySide6.QtWidgets import QGroupBox  # noqa: E402

from pipeline.acquisition import CameraStream, SyntheticBackend, CANONICAL_SHAPE  # noqa: E402
from pipeline.analysis import SensorNoiseModel  # noqa: E402
from pipeline.calibration.sensor import ConversionGainRecord  # noqa: E402
from pipeline.calibration.shared import (  # noqa: E402
    CalibrationRecord,
    EXPOSURE_MATCH_TOLERANCE_REL,
    GAIN_MATCH_TOLERANCE_ABS,
)
from pipeline.calibration.spatial import ScaleFactorPositionCalibration  # noqa: E402
from pipeline.calibration.spatial.calibrate import PIXEL_PITCH_UM  # noqa: E402
from pipeline.preprocessing import CalibrationSet  # noqa: E402
from pipeline.gui.live_view import (  # noqa: E402
    DEFAULT_DEGREE,
    DEFAULT_UPDATE_INTERVAL_MS,
    DEGREE_CHOICES,
    EVALUATED_AT_COLUMN,
    MAX_CONSECUTIVE_SKIPS,
    MICRONS_PER_MM,
    LiveViewWidget,
    _PowerOfTenAxisItem,
    evaluated_at_text,
    exposure_has_drifted,
    fit_formula_html,
    format_power_of_ten_superscript,
    gain_has_drifted,
    heatmap_x_extent,
    wavelength_axis_label,
)
from pipeline.gui.roi_control import SpatialROIControl  # noqa: E402

# gui_fixture_helpers.py is a plain sibling module in this directory (no
# tests/__init__.py, so pytest's rootless import mode puts tests/ on
# sys.path directly) -- see its own module docstring for why it's the
# shared, read-only "realistic CalibrationBundle" builder for any gui/
# wiring test that needs one, this file included.
from gui_fixture_helpers import build_realistic_calibration_bundle  # noqa: E402

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


def _make_live_view_widget(qtbot, wavelength_axis=None) -> LiveViewWidget:
    widget = LiveViewWidget(
        calibration_set=_calibration_set(),
        noise_model=SensorNoiseModel(gain_e_per_adu=2.2, background_sigma=1.0),
        position_calibration=ScaleFactorPositionCalibration(),
        wavelength_axis=wavelength_axis,
        camera_stream=_camera_stream(),
    )
    qtbot.addWidget(widget)
    return widget


def _realistic_bundle(background_sigma: float = 1.0):
    '''
    build_realistic_calibration_bundle()'s dark/illuminated synthetic
    frames are bit-for-bit uniform (see gui_fixture_helpers.py's _frame()
    helper), so the per-pixel sample standard deviation build_baseline()
    measures across them comes out exactly 0.0. run_preprocessing()'s
    signal-threshold step requires a strictly positive background_sigma
    (there's no meaningful SNR against a zero noise floor) and raises
    ValueError otherwise -- so every real-tick test below overrides it to
    a small, physically reasonable positive value via dataclasses.replace()
    on the returned (frozen) dataclasses, rather than editing the shared,
    read-only fixture module itself. A very large background_sigma is
    also used deliberately by the insufficient-signal tests below, to
    force every column below SNR_THRESHOLD and reliably trigger
    InsufficientDataError on every tick.
    '''
    bundle = build_realistic_calibration_bundle()
    calibration_set = dataclasses.replace(bundle.calibration_set, background_sigma=background_sigma)
    noise_model = dataclasses.replace(bundle.noise_model, background_sigma=background_sigma)
    return dataclasses.replace(bundle, calibration_set=calibration_set, noise_model=noise_model)


def _make_real_live_view_widget(
    qtbot, camera_stream: CameraStream, background_sigma: float = 1.0,
    update_interval_ms: int = 20,
) -> LiveViewWidget:
    '''
    Builds a LiveViewWidget over a real, non-placeholder CalibrationBundle
    (see _realistic_bundle()) and an already-STARTED camera_stream --
    unlike _make_live_view_widget()'s deliberately-unstarted stream above,
    the real update-loop tests need get_latest_frame() to actually return
    data. update_interval_ms defaults to a small value so qtbot.waitUntil()
    below sees a tick quickly instead of waiting out
    DEFAULT_UPDATE_INTERVAL_MS's full ~5Hz interval.
    '''
    bundle = _realistic_bundle(background_sigma=background_sigma)
    widget = LiveViewWidget(
        calibration_set=bundle.calibration_set,
        noise_model=bundle.noise_model,
        position_calibration=bundle.position_calibration,
        wavelength_axis=bundle.wavelength_axis,
        camera_stream=camera_stream,
        conversion_gain_record=bundle.conversion_gain_record,
        update_interval_ms=update_interval_ms,
    )
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def started_camera_stream():
    '''
    A running CameraStream over a seeded SyntheticBackend, for the real
    update-loop tests that need get_latest_frame() to return actual data
    (unlike this file's _camera_stream() helper, explicitly never started).
    Always stopped on teardown -- even on a test failure -- so a leaked
    background grab thread can't make an unrelated, later test flaky or
    hang (see this task's own lifecycle warning about exactly that).
    '''
    stream = CameraStream(
        exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
        pixel_format="Mono8", timeout_ms=5000, backend=SyntheticBackend(seed=0),
    )
    stream.start()
    try:
        yield stream
    finally:
        stream.stop()


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


class TestEvaluatedAtText:

    def test_none_axis_gives_pixel_column_fallback(self):
        assert evaluated_at_text(EVALUATED_AT_COLUMN, None) == "Pixel column 960"

    def test_real_axis_gives_wavelength_text(self):
        axis = _FakeWavelengthAxis()
        # axis.wavelength_nm(960) == 500.0 + 0.01 * 960 == 509.6
        assert evaluated_at_text(960.0, axis) == "509.6 nm"

    def test_pixel_column_rounds_to_nearest_int(self):
        assert evaluated_at_text(959.6, None) == "Pixel column 960"


class TestFitFormulaHtml:

    def test_degree_one_pixel_fallback(self):
        assert fit_formula_html(1, None) == "x<sub>0</sub>(column) = c<sub>0</sub> + c<sub>1</sub>·column"

    def test_degree_one_wavelength(self):
        axis = _FakeWavelengthAxis()
        assert fit_formula_html(1, axis) == "x<sub>0</sub>(λ) = c<sub>0</sub> + c<sub>1</sub>λ"

    def test_degree_two_adds_squared_term(self):
        axis = _FakeWavelengthAxis()
        html = fit_formula_html(2, axis)
        assert html.endswith("c<sub>2</sub>λ<sup>2</sup>")

    def test_degree_three_adds_cubed_term(self):
        axis = _FakeWavelengthAxis()
        html = fit_formula_html(3, axis)
        assert html.endswith("c<sub>3</sub>λ<sup>3</sup>")

    def test_degree_increases_term_count(self):
        html1 = fit_formula_html(1, None)
        html3 = fit_formula_html(3, None)
        assert html1.count("c<sub>") == 2
        assert html3.count("c<sub>") == 4


class TestFormatPowerOfTenSuperscript:

    def test_negative_exponent(self):
        assert format_power_of_ten_superscript(0.001) == "×10⁻³"

    def test_positive_exponent(self):
        assert format_power_of_ten_superscript(1000.0) == "×10³"

    def test_zero_exponent(self):
        assert format_power_of_ten_superscript(1.0) == "×10⁰"

    def test_double_digit_exponent(self):
        assert format_power_of_ten_superscript(1e-12) == "×10⁻¹²"

    def test_non_power_of_ten_raises(self):
        with pytest.raises(ValueError):
            format_power_of_ten_superscript(0.002)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            format_power_of_ten_superscript(0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            format_power_of_ten_superscript(-0.001)


# ---------------------------------------------------------------------------
# live_view.py -- LiveViewWidget pytest-qt smoke tests (offscreen)
# ---------------------------------------------------------------------------

class TestLiveViewWidgetSmoke:

    def test_widget_constructs_and_shows(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        widget.resize(800, 600)
        widget.show()
        qtbot.waitExposed(widget)
        assert widget.isVisible()

    def test_core_child_widgets_exist(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        assert widget._main_plot is not None
        assert widget._strip_plot is not None
        assert widget._image_item is not None
        assert widget._scatter is not None
        assert widget._fit_curve is not None
        assert widget._degree_selector.count() == len(DEGREE_CHOICES)
        assert widget._extended_measurement_button is not None
        assert widget._chi_squared_label.text() != "--"

    def test_degree_selector_defaults_to_linear(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        assert widget._degree_selector.currentData() == DEFAULT_DEGREE

    def test_heatmap_positioned_in_data_space_not_unit_square(self, qtbot):
        # Regression test: ImageItem.setRect() scales by self.width()/
        # self.height(), which silently fall back to 1.0 if setImage()
        # hasn't been called yet -- if _generate_placeholder_data() ever
        # calls setRect() before assigning an image again, the transform's
        # scale factors become the raw setRect() width/height (~1919,
        # ~6.21) instead of those values divided by the image's real
        # 1920x1200 pixel dimensions (~0.9995, ~0.005175), stretching the
        # heatmap ~1920x/1200x too large so only its extreme top-left
        # corner is visible (a flat, uniform color, not a heatmap).
        widget = _make_live_view_widget(qtbot)
        transform = widget._image_item.transform()
        assert transform.m11() < 10.0
        assert transform.m22() < 10.0
        assert widget._current_degree == DEFAULT_DEGREE
        assert widget._zeta_note_label.text() == ""

    def test_degree_selector_changes_state(self, qtbot):
        widget = _make_live_view_widget(qtbot)
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
        widget = _make_live_view_widget(qtbot)
        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(3))
        assert not widget._camera_stream.is_running

    def test_wavelength_axis_none_uses_pixel_fallback_label(self, qtbot):
        widget = _make_live_view_widget(qtbot, wavelength_axis=None)
        axis_label = widget._main_plot.getAxis("bottom").labelText
        assert "not yet available" in axis_label.lower()

    def test_wavelength_axis_present_uses_wavelength_label(self, qtbot):
        widget = _make_live_view_widget(qtbot, wavelength_axis=_FakeWavelengthAxis())
        axis_label = widget._main_plot.getAxis("bottom").labelText
        assert axis_label == "Wavelength (nm)"

    def test_extended_measurement_button_emits_signal(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        button = widget._extended_measurement_button
        assert button.text() == "Extended Measurement..."
        with qtbot.waitSignal(widget.extended_measurement_requested, timeout=1000):
            button.click()

    def test_back_to_calibration_button_emits_signal(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        button = widget._back_to_calibration_button
        assert button.text() == "Back to Calibration"
        with qtbot.waitSignal(widget.back_to_calibration_requested, timeout=1000):
            button.click()


# ---------------------------------------------------------------------------
# live_view.py -- Phase-1 review refinements (colormap, mm conversion,
# background styling, formatting, formula box, evaluated-at note)
# ---------------------------------------------------------------------------

class TestLiveViewWidgetRefinements:

    def test_heatmap_has_a_colormap_lookup_table(self, qtbot):
        # A plain-greyscale ImageItem has no lookup table at all -- once a
        # colormap is applied, .lut is populated.
        widget = _make_live_view_widget(qtbot)
        assert widget._image_item.lut is not None

    def test_zeta_label_has_word_wrap(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        assert widget._zeta_label.wordWrap()

    def test_evaluated_at_label_has_word_wrap(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        assert widget._evaluated_at_label.wordWrap()

    def test_formula_label_has_word_wrap(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        assert widget._formula_label.wordWrap()

    def test_group_boxes_and_combo_box_have_explicit_background(self, qtbot):
        # See _group_box_stylesheet()/_combo_box_stylesheet() -- the
        # top-level stylesheet doesn't reliably cascade into these widgets,
        # so each carries its own explicit background-color rule.
        widget = _make_live_view_widget(qtbot)
        for box in widget.findChildren(QGroupBox):
            assert "background-color" in box.styleSheet()
        assert "background-color" in widget._degree_selector.styleSheet()

    def test_convert_to_mm_matches_known_hardware_geometry(self, qtbot):
        # 1200 spatial pixels * 3.45 um/pixel * 1.5 scale factor ==
        # 6210 um == 6.21 mm at the slit -- see calibration/spatial/
        # calibrate.py's module docstring and docs/project_state.md's
        # "Bug fixed during GUI review" note.
        widget = _make_live_view_widget(qtbot)
        y_mm, _ = widget._convert_to_mm(np.array([1200.0]), np.array([0.0]))
        assert y_mm[0] == pytest.approx(6.21)

    def test_coefficients_and_zeta_use_plus_minus_character(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        assert "±" in widget._coefficients_label.text()
        assert "+/-" not in widget._coefficients_label.text()
        assert "±" in widget._zeta_label.text()
        assert "+/-" not in widget._zeta_label.text()

    def test_formula_label_updates_with_degree(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        linear_text = widget._formula_label.text()

        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))

        assert widget._formula_label.text() != linear_text
        assert "c<sub>2</sub>" in widget._formula_label.text()

    def test_evaluated_at_label_only_populated_for_degree_greater_than_one(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        assert widget._evaluated_at_label.text() == ""

        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))
        assert widget._evaluated_at_label.text() != ""
        assert "960" in widget._evaluated_at_label.text()

    def test_zeta_note_no_longer_mentions_median_x_value_inline(self, qtbot):
        # The evaluated-at value moved to its own labeled row (see above)
        # rather than being buried in this note's prose.
        widget = _make_live_view_widget(qtbot)
        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))
        assert "median" not in widget._zeta_note_label.text().lower()

    def test_y_axis_label_says_relative_physical_position(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        assert widget._main_plot.getAxis("left").labelText == "Relative Physical Position (mm)"

    def test_fit_curve_is_black_and_thicker_than_default(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        pen = widget._fit_curve.opts["pen"]
        assert pen.color().name() == "#000000"
        assert pen.width() >= 3

    def test_strip_chart_left_axis_is_power_of_ten_axis(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        assert isinstance(widget._strip_plot.getAxis("left"), _PowerOfTenAxisItem)

    def test_extended_measurement_button_is_prominent(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        button = widget._extended_measurement_button
        assert button.minimumHeight() >= 56
        assert "background-color" in button.styleSheet()


# ---------------------------------------------------------------------------
# live_view.py -- Acquisition Settings / drift-detection pure helpers
# ---------------------------------------------------------------------------

class TestExposureHasDrifted:

    def test_within_tolerance_is_not_drifted(self):
        baseline_exposure_us = 2000.0
        current = baseline_exposure_us * (1 + EXPOSURE_MATCH_TOLERANCE_REL * 0.5)
        assert not exposure_has_drifted(current, baseline_exposure_us)

    def test_beyond_tolerance_is_drifted(self):
        baseline_exposure_us = 2000.0
        current = baseline_exposure_us * (1 + EXPOSURE_MATCH_TOLERANCE_REL * 2)
        assert exposure_has_drifted(current, baseline_exposure_us)


class TestGainHasDrifted:

    def test_within_tolerance_is_not_drifted(self):
        baseline_gain_db = 10.0
        current = baseline_gain_db + GAIN_MATCH_TOLERANCE_ABS * 0.5
        assert not gain_has_drifted(current, baseline_gain_db)

    def test_beyond_tolerance_is_drifted(self):
        baseline_gain_db = 10.0
        current = baseline_gain_db + GAIN_MATCH_TOLERANCE_ABS * 2
        assert gain_has_drifted(current, baseline_gain_db)


# ---------------------------------------------------------------------------
# live_view.py -- Acquisition Settings panel (pytest-qt, offscreen)
# ---------------------------------------------------------------------------

class TestAcquisitionSettingsPanel:

    def test_fields_prefilled_from_baseline_record(self, qtbot):
        widget = _make_live_view_widget(qtbot)
        assert widget._exposure_spin.value() == pytest.approx(FIXTURE_EXPOSURE_US)
        assert widget._gain_spin.value() == pytest.approx(FIXTURE_GAIN_DB)

    def test_recalibration_requested_fires_baseline_for_exposure_drift(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.live_view.QMessageBox.warning", lambda *args, **kwargs: None
        )
        widget = _make_live_view_widget(qtbot)
        drifted_exposure_us = FIXTURE_EXPOSURE_US * (1 + EXPOSURE_MATCH_TOLERANCE_REL * 2)

        with qtbot.waitSignal(widget.recalibration_requested, timeout=1000) as blocker:
            widget._exposure_spin.setValue(drifted_exposure_us)

        assert blocker.args == ["baseline"]

    def test_recalibration_requested_fires_conversion_gain_for_isolated_drift(
        self, qtbot, monkeypatch
    ):
        # conversion_gain_record.gain_db is chosen within tolerance of
        # baseline_record.gain_db (FIXTURE_GAIN_DB) so the widget does NOT
        # start already drifted at construction (see __init__'s own
        # _recompute_settings_drift() call) -- the spin box is then moved to
        # a value that stays within tolerance of baseline but drifts beyond
        # tolerance from conversion_gain_record, isolating the
        # conversion-gain-specific signal from the baseline one, per the
        # spec's "drift independently" note. ConversionGainRecord.gain_db
        # has no widget-range constraint (unlike the gain spin box itself,
        # whose minimum is 0.0), so it can sit on the opposite side of
        # FIXTURE_GAIN_DB from the spin box's new value.
        monkeypatch.setattr(
            "pipeline.gui.live_view.QMessageBox.warning", lambda *args, **kwargs: None
        )
        conversion_gain_record = ConversionGainRecord(
            gain_db=FIXTURE_GAIN_DB - GAIN_MATCH_TOLERANCE_ABS * 0.9,
            timestamp=time.time(), n_illumination_levels=5,
        )
        widget = LiveViewWidget(
            calibration_set=_calibration_set(),
            noise_model=SensorNoiseModel(gain_e_per_adu=2.2, background_sigma=1.0),
            position_calibration=ScaleFactorPositionCalibration(),
            wavelength_axis=None,
            camera_stream=_camera_stream(),
            conversion_gain_record=conversion_gain_record,
        )
        qtbot.addWidget(widget)
        assert widget._settings_drifted is False

        near_baseline_gain_db = FIXTURE_GAIN_DB + GAIN_MATCH_TOLERANCE_ABS * 0.9
        assert not gain_has_drifted(near_baseline_gain_db, FIXTURE_GAIN_DB)
        assert gain_has_drifted(near_baseline_gain_db, conversion_gain_record.gain_db)

        with qtbot.waitSignal(widget.recalibration_requested, timeout=1000) as blocker:
            widget._gain_spin.setValue(near_baseline_gain_db)

        assert blocker.args == ["conversion_gain"]

    def test_drifted_state_hides_diagnostics_and_overlay_and_warns_once(
        self, qtbot, monkeypatch
    ):
        warning_calls = []
        monkeypatch.setattr(
            "pipeline.gui.live_view.QMessageBox.warning",
            lambda *args, **kwargs: warning_calls.append(args),
        )
        widget = _make_live_view_widget(qtbot)
        drifted_gain_db = FIXTURE_GAIN_DB + GAIN_MATCH_TOLERANCE_ABS * 2

        widget._gain_spin.setValue(drifted_gain_db)

        assert widget._settings_drifted is True
        assert widget._chi_squared_label.text() == "N/A"
        assert widget._coefficients_label.text() == "N/A"
        assert widget._zeta_label.text() == "N/A"
        assert widget._zeta_note_label.text() == ""
        assert widget._evaluated_at_label.text() == ""
        assert widget._scatter.isVisible() is False
        assert widget._error_bars.isVisible() is False
        assert widget._fit_curve.isVisible() is False
        # The raw heatmap must keep displaying -- only the fit overlay hides.
        assert widget._image_item.isVisible() is True
        assert len(warning_calls) == 1

    def test_drift_popup_and_signal_fire_once_per_episode(self, qtbot, monkeypatch):
        warning_calls = []
        monkeypatch.setattr(
            "pipeline.gui.live_view.QMessageBox.warning",
            lambda *args, **kwargs: warning_calls.append(args),
        )
        widget = _make_live_view_widget(qtbot)
        received = []
        widget.recalibration_requested.connect(received.append)
        drifted_gain_db = FIXTURE_GAIN_DB + GAIN_MATCH_TOLERANCE_ABS * 2

        widget._gain_spin.setValue(drifted_gain_db)
        # Still drifted, just a different drifted value -- no new episode,
        # so no additional popup/signal.
        widget._gain_spin.setValue(drifted_gain_db + GAIN_MATCH_TOLERANCE_ABS)

        assert len(warning_calls) == 1
        assert len(received) == 1

    def test_exiting_drifted_state_restores_diagnostics_and_overlay(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.live_view.QMessageBox.warning", lambda *args, **kwargs: None
        )
        widget = _make_live_view_widget(qtbot)
        drifted_gain_db = FIXTURE_GAIN_DB + GAIN_MATCH_TOLERANCE_ABS * 2

        widget._gain_spin.setValue(drifted_gain_db)
        assert widget._settings_drifted is True

        widget._gain_spin.setValue(FIXTURE_GAIN_DB)

        assert widget._settings_drifted is False
        assert widget._chi_squared_label.text() != "N/A"
        assert widget._coefficients_label.text() != "N/A"
        assert widget._zeta_label.text() != "N/A"
        assert widget._scatter.isVisible() is True
        assert widget._error_bars.isVisible() is True
        assert widget._fit_curve.isVisible() is True


# ---------------------------------------------------------------------------
# roi_control.py -- SpatialROIControl (pytest-qt, offscreen)
# ---------------------------------------------------------------------------

class TestSpatialROIControl:

    def _make_widget(self, qtbot, scale_factor: float = 1.5, n_rows: int = 1000) -> SpatialROIControl:
        widget = SpatialROIControl(
            position_calibration=ScaleFactorPositionCalibration(scale_factor=scale_factor),
            n_rows=n_rows,
        )
        qtbot.addWidget(widget)
        # error_label.isVisible() below reflects real on-screen visibility,
        # not just the internal setVisible() flag -- requires the widget
        # to actually be shown, matching TestLiveViewWidgetSmoke's convention.
        widget.show()
        qtbot.waitExposed(widget)
        return widget

    def test_default_construction_spans_full_range(self, qtbot):
        scale_factor, n_rows = 1.5, 1000
        widget = self._make_widget(qtbot, scale_factor=scale_factor, n_rows=n_rows)

        expected_extent_mm = (PIXEL_PITCH_UM * scale_factor * n_rows) / MICRONS_PER_MM

        min_mm, max_mm = widget.roi_bounds_mm()
        assert min_mm == pytest.approx(0.0)
        assert max_mm == pytest.approx(expected_extent_mm)

    def test_valid_change_updates_bounds_and_emits_signal(self, qtbot):
        widget = self._make_widget(qtbot)
        received = []
        widget.roi_changed.connect(lambda min_mm, max_mm: received.append((min_mm, max_mm)))

        widget._min_spin.setValue(1.0)

        assert widget.roi_bounds_mm()[0] == pytest.approx(1.0)
        assert len(received) == 1
        assert received[0][0] == pytest.approx(1.0)
        assert received[0][1] == pytest.approx(widget._max_spin.value())

    def test_invalid_change_shows_error_and_reverts_without_emitting(self, qtbot):
        widget = self._make_widget(qtbot)
        last_valid = widget._last_valid_bounds
        received = []
        widget.roi_changed.connect(lambda min_mm, max_mm: received.append((min_mm, max_mm)))

        current_max = widget._max_spin.value()
        widget._min_spin.setValue(current_max)

        assert widget._error_label.isVisible() is True
        assert received == []
        assert widget.roi_bounds_mm() == pytest.approx(last_valid)

    def test_reset_button_restores_full_range_and_emits_signal(self, qtbot):
        widget = self._make_widget(qtbot)
        widget._min_spin.setValue(1.0)

        received = []
        widget.roi_changed.connect(lambda min_mm, max_mm: received.append((min_mm, max_mm)))
        widget._reset_button.click()

        expected_extent_mm = widget._extent_mm
        assert widget.roi_bounds_mm() == pytest.approx((0.0, expected_extent_mm))
        assert received == [(0.0, expected_extent_mm)]
        assert widget._error_label.isVisible() is False

    def test_roi_bounds_px_matches_hand_computed_expectation(self, qtbot):
        scale_factor, n_rows = 2.0, 500
        widget = self._make_widget(qtbot, scale_factor=scale_factor, n_rows=n_rows)
        combined_factor = PIXEL_PITCH_UM * scale_factor

        widget._min_spin.setValue(1.0)
        widget._max_spin.setValue(2.0)

        expected_min_px = round((1.0 * MICRONS_PER_MM) / combined_factor)
        expected_max_px = round((2.0 * MICRONS_PER_MM) / combined_factor)

        assert widget.roi_bounds_px() == (expected_min_px, expected_max_px)


# ---------------------------------------------------------------------------
# live_view.py -- LiveViewWidget wired to SpatialROIControl (pytest-qt, offscreen)
# ---------------------------------------------------------------------------

class TestLiveViewSpatialROI:

    def test_roi_change_narrows_plot_y_range(self, qtbot):
        widget = _make_live_view_widget(qtbot)

        min_mm, max_mm = 1.0, 5.0
        widget._roi_control._min_spin.setValue(min_mm)
        widget._roi_control._max_spin.setValue(max_mm)

        y_range = widget._main_plot.viewRange()[1]
        assert tuple(y_range) == pytest.approx((min_mm, max_mm))

    def test_roi_change_drops_out_of_window_scatter_points(self, qtbot):
        widget = _make_live_view_widget(qtbot)

        _, y_before = widget._scatter.getData()
        count_before = len(y_before)

        # Median of the full-range scatter's own y-values -- guaranteed to
        # cut roughly half the points (those below it) while leaving the
        # rest, without hand-computing the fake centroid trend's mm range.
        min_mm = float(np.median(y_before))
        widget._roi_control._min_spin.setValue(min_mm)
        max_mm = widget._roi_control.roi_bounds_mm()[1]

        _, y_after = widget._scatter.getData()

        assert len(y_after) < count_before
        assert np.all((y_after >= min_mm) & (y_after <= max_mm))


# ---------------------------------------------------------------------------
# live_view.py -- real QTimer-driven update loop (pytest-qt, offscreen)
# ---------------------------------------------------------------------------
#
# Uses a real, STARTED CameraStream (started_camera_stream fixture) and a
# real, non-placeholder CalibrationBundle (_realistic_bundle()), unlike
# every test above -- these exercise the actual run_preprocessing() ->
# analyze_shot() chain per tick, not just the widget's presentational
# state. update_interval_ms is shrunk to 20ms (see
# _make_real_live_view_widget()) so qtbot.waitUntil() below resolves
# quickly and reliably instead of waiting out the real ~5Hz default.

class TestLiveViewRealUpdateLoop:

    def test_tick_populates_scatter_from_real_analysis(self, qtbot, started_camera_stream):
        widget = _make_real_live_view_widget(qtbot, started_camera_stream)
        widget.show()
        qtbot.waitExposed(widget)

        # The placeholder feed only ever plots ~127 points (see
        # _generate_placeholder_data()'s columns = np.arange(200, 1720, 12));
        # a real tick fits every one of SyntheticBackend's ~1920 valid
        # columns, so a scatter count comfortably above that placeholder
        # count is an unambiguous signal that real data has landed, without
        # hand-computing the synthetic beam's exact centroid trend.
        qtbot.waitUntil(lambda: len(widget._scatter.getData()[0]) > 500, timeout=3000)

        x_values, y_values = widget._scatter.getData()
        assert len(x_values) == len(y_values) > 500
        assert np.all(np.isfinite(y_values))
        assert widget._consecutive_skips == 0
        assert widget._insufficient_signal is False

    def test_tick_updates_fit_diagnostics_from_real_result(self, qtbot, started_camera_stream):
        widget = _make_real_live_view_widget(qtbot, started_camera_stream)
        widget.show()
        qtbot.waitExposed(widget)

        placeholder_chi_squared = widget._chi_squared_label.text()
        qtbot.waitUntil(
            lambda: widget._chi_squared_label.text() != placeholder_chi_squared, timeout=3000
        )

        # Real coefficients/zeta must actually be finite, formatted numbers
        # -- not "N/A" (the insufficient-signal/drifted placeholder) and not
        # left showing the construction-time placeholder text.
        assert "N/A" not in widget._chi_squared_label.text()
        assert "±" in widget._zeta_label.text()
        assert "N/A" not in widget._zeta_label.text()

    def test_heatmap_updates_from_real_processed_frame(self, qtbot, started_camera_stream):
        widget = _make_real_live_view_widget(qtbot, started_camera_stream)
        widget.show()
        qtbot.waitExposed(widget)

        initial_image = widget._image_item.image.copy()
        qtbot.waitUntil(
            lambda: not np.array_equal(widget._image_item.image, initial_image), timeout=3000
        )

        assert widget._image_item.image.shape == CANONICAL_SHAPE

    def test_status_label_reflects_running_stream(self, qtbot, started_camera_stream):
        widget = _make_real_live_view_widget(qtbot, started_camera_stream)
        qtbot.waitUntil(lambda: widget._status_label.text() == "Status: OK", timeout=3000)

    def test_status_label_reflects_stopped_stream(self, qtbot):
        # Deliberately the never-started stream (unlike started_camera_stream
        # above) -- is_running is False from construction onward.
        widget = _make_live_view_widget(qtbot)
        assert widget._status_label.text() == "Status: Camera stopped"

    def test_roi_bounds_are_applied_to_real_preprocessing(self, qtbot, started_camera_stream):
        # Regression test for the live-loop's roi_bounds=self._roi_control.
        # roi_bounds_px() wiring: narrow the ROI to exclude the top half of
        # the spatial axis, then confirm a real tick's displayed heatmap
        # actually reflects apply_roi()'s row-zeroing, not just the
        # unmasked raw frame.
        widget = _make_real_live_view_widget(qtbot, started_camera_stream)
        widget.show()
        qtbot.waitExposed(widget)

        qtbot.waitUntil(lambda: widget._consecutive_skips == 0 and widget._scatter.getData()[0].size > 0, timeout=3000)

        full_extent_mm = widget._roi_control.roi_bounds_mm()[1]
        widget._roi_control._min_spin.setValue(full_extent_mm / 2.0)
        row_min_px, _ = widget._roi_control.roi_bounds_px()
        assert row_min_px > 0

        def _top_rows_zeroed() -> bool:
            image = widget._image_item.image
            return image is not None and np.all(image[:row_min_px, :] == 0)

        qtbot.waitUntil(_top_rows_zeroed, timeout=3000)

    def test_insufficient_signal_state_after_consecutive_skips(self, qtbot, started_camera_stream):
        # An enormous background_sigma pushes every column's SNR below
        # SNR_THRESHOLD, so every tick's analyze_shot() call raises
        # InsufficientDataError -- see _realistic_bundle()'s docstring.
        widget = _make_real_live_view_widget(
            qtbot, started_camera_stream, background_sigma=1.0e6, update_interval_ms=10,
        )
        widget.show()
        qtbot.waitExposed(widget)

        qtbot.waitUntil(lambda: widget._insufficient_signal is True, timeout=5000)

        assert widget._consecutive_skips >= MAX_CONSECUTIVE_SKIPS
        assert widget._chi_squared_label.text() == "N/A"
        assert widget._zeta_label.text() == "N/A"
        assert widget._scatter.isVisible() is False
        assert widget._error_bars.isVisible() is False
        assert widget._fit_curve.isVisible() is False
        # The heatmap keeps updating on a skip -- preprocessing itself
        # succeeded, only the fit lacked enough columns (see
        # _on_timer_tick()'s InsufficientDataError branch).
        assert widget._image_item.isVisible() is True

    def test_settings_drift_pauses_real_updates(self, qtbot, started_camera_stream, monkeypatch):
        monkeypatch.setattr(
            "pipeline.gui.live_view.QMessageBox.warning", lambda *args, **kwargs: None
        )
        widget = _make_real_live_view_widget(qtbot, started_camera_stream)
        widget.show()
        qtbot.waitExposed(widget)

        # Let at least one real tick land first, so there's genuine
        # (non-placeholder) state that a drifted tick must NOT overwrite.
        qtbot.waitUntil(lambda: widget._scatter.getData()[0].size > 500, timeout=3000)

        drifted_gain_db = FIXTURE_GAIN_DB + GAIN_MATCH_TOLERANCE_ABS * 2
        widget._gain_spin.setValue(drifted_gain_db)
        assert widget._settings_drifted is True

        # Give the real loop several more tick intervals' worth of real
        # camera time to (wrongly) act on, then confirm it didn't: the
        # drifted state's own "N/A"/hidden-overlay treatment must still be
        # showing, untouched by any real tick that fired in the meantime.
        qtbot.wait(150)

        assert widget._chi_squared_label.text() == "N/A"
        assert widget._zeta_label.text() == "N/A"
        assert widget._scatter.isVisible() is False
        assert widget._error_bars.isVisible() is False
        assert widget._fit_curve.isVisible() is False

    def test_roi_change_after_real_data_does_not_flash_placeholder(self, qtbot, started_camera_stream):
        # Regression test: _on_roi_changed() used to call _apply_roi_bounds()
        # unconditionally, which always repaints from self._placeholder_*
        # (stale, fake construction-time data) -- so touching the ROI
        # control after real data was already on screen replaced it with
        # placeholder data until the next tick overwrote it again. Confirms
        # the fix: once real data has landed, changing the ROI must only
        # rescale the plot's y-range, never touch the scatter/heatmap data.
        widget = _make_real_live_view_widget(qtbot, started_camera_stream)
        widget.show()
        qtbot.waitExposed(widget)

        qtbot.waitUntil(lambda: widget._scatter.getData()[0].size > 500, timeout=3000)
        real_x, real_y = widget._scatter.getData()
        real_image = widget._image_item.image.copy()

        widget._roi_control._min_spin.setValue(1.0)

        # No qtbot.wait()/processEvents() between the ROI edit and these
        # assertions -- Qt dispatches roi_changed -> _on_roi_changed()
        # synchronously inside setValue(), so if it had wrongly fallen back
        # to _apply_roi_bounds(), the scatter would already have shrunk to
        # the placeholder's ~127-point array by this point.
        after_x, after_y = widget._scatter.getData()
        assert len(after_x) == len(real_x)
        assert np.array_equal(after_x, real_x)
        assert np.array_equal(after_y, real_y)
        assert np.array_equal(widget._image_item.image, real_image)

    def test_degree_change_after_real_data_does_not_flash_placeholder_note(
        self, qtbot, started_camera_stream
    ):
        # Regression test: _on_degree_changed() used to call
        # _update_fit_panel() unconditionally, which always repaints from
        # self._placeholder_fits -- including a "Uncertainty not available
        # for degree > 1" note real data never actually has (real
        # sigma_zeta() is available at every degree, see
        # _update_fit_panel_from_result()). So switching to degree 2/3
        # after real data was already on screen replaced the genuine
        # diagnostics with that stale placeholder note until the next real
        # tick overwrote it again. Confirms the fix: once real data has
        # landed, changing the degree must only update
        # self._current_degree, never repaint placeholder text.
        widget = _make_real_live_view_widget(qtbot, started_camera_stream)
        widget.show()
        qtbot.waitExposed(widget)

        qtbot.waitUntil(lambda: widget._scatter.getData()[0].size > 500, timeout=3000)
        assert widget._zeta_note_label.text() == ""
        real_chi_squared = widget._chi_squared_label.text()

        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))

        # No qtbot.wait()/processEvents() between the degree change and
        # these assertions -- Qt dispatches currentIndexChanged ->
        # _on_degree_changed() synchronously inside setCurrentIndex(), so
        # if it had wrongly fallen back to _update_fit_panel(), the
        # placeholder note would already be showing by this point.
        assert widget._current_degree == 2
        assert widget._zeta_note_label.text() == ""
        assert widget._chi_squared_label.text() == real_chi_squared

        # A subsequent real tick must genuinely refit at the new degree --
        # not just leave the pre-switch degree-1 numbers in place.
        qtbot.waitUntil(
            lambda: widget._chi_squared_label.text() != real_chi_squared, timeout=3000
        )
        assert widget._zeta_note_label.text() == ""
        assert widget._coefficients_label.text().count("c<sub>") == 3

    def test_drift_exit_at_degree_gt_1_after_real_data_does_not_flash_placeholder_note(
        self, qtbot, started_camera_stream, monkeypatch
    ):
        # Regression test: _exit_drifted_state() had the same bug as
        # _on_degree_changed() above -- unconditionally calling
        # _update_fit_panel(), which shows the placeholder-only
        # "Uncertainty not available for degree > 1" note. Exercises it at
        # degree 2, where the bug would actually be visible (the note only
        # ever appears for degree > 1).
        monkeypatch.setattr(
            "pipeline.gui.live_view.QMessageBox.warning", lambda *args, **kwargs: None
        )
        widget = _make_real_live_view_widget(qtbot, started_camera_stream)
        widget.show()
        qtbot.waitExposed(widget)

        qtbot.waitUntil(lambda: widget._scatter.getData()[0].size > 500, timeout=3000)
        widget._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(2))
        qtbot.waitUntil(lambda: widget._coefficients_label.text().count("c<sub>") == 3, timeout=3000)

        drifted_gain_db = FIXTURE_GAIN_DB + GAIN_MATCH_TOLERANCE_ABS * 2
        widget._gain_spin.setValue(drifted_gain_db)
        assert widget._settings_drifted is True

        widget._gain_spin.setValue(FIXTURE_GAIN_DB)
        assert widget._settings_drifted is False

        # No qtbot.wait()/processEvents() between exiting the drifted state
        # and this assertion -- _exit_drifted_state() runs synchronously
        # inside the setValue() call above, so if it had wrongly fallen
        # back to _update_fit_panel(), the placeholder note would already
        # be showing by this point.
        assert widget._zeta_note_label.text() == ""
