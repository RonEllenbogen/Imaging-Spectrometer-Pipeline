"""
Test suite for the gui package. Covers calibration_screen.py/
calibration_dialogs.py's Phase-1 visual skeleton, and live_view.py's
Phase-1 visual skeleton, as widget-layer smoke tests plus (for live_view.py)
ordinary unit tests of its plain, non-Qt presentational helper functions.

Most of both screens has no real camera/calibration-package/preprocessing/
analysis call wired in yet (see each module's own docstring), so most of
these tests only check structure/layout/state transitions -- not behavior,
e.g. that CreatePage exposes exactly the four enabled type cards plus a
disabled spectral card, not that clicking "Configure..." actually
acquires anything; that degree selection changes LiveViewWidget's
*displayed* placeholder numbers, not that it triggers a real refit. The
exception is WelcomePage's "Load Existing Calibrations" flow, which is
fully wired (see calibration_screen.CalibrationScreen.
_attempt_load_existing_calibrations()) -- its tests mock calibration_
screen's load_*()/show_calibration_error_dialog() calls rather than
touching a real calibration_artifacts/ directory or camera. A follow-up
pass adds the remaining real-logic tests (automatic bad-pixel-map
chaining on CreatePage, error-dialog paths there, the QTimer-driven
update loop) once each is wired, using the same mocked-call convention.

PySide6/pyqtgraph/pytest-qt are local-only dependencies (see CLAUDE.md --
pyproject.toml/requirements.txt are deliberately left empty), so this
whole module is skipped, not failed, wherever they aren't installed --
the same "gate, don't require" pattern tests/test_acquisition.py uses for
hardware-only tests.
"""

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

from PySide6.QtWidgets import QGroupBox  # noqa: E402

from pipeline.acquisition import CameraStream, SyntheticBackend, CANONICAL_SHAPE  # noqa: E402
from pipeline.analysis import SensorNoiseModel  # noqa: E402
from pipeline.calibration.shared import CalibrationRecord  # noqa: E402
from pipeline.calibration.spatial import ScaleFactorPositionCalibration  # noqa: E402
from pipeline.preprocessing import CalibrationSet  # noqa: E402
import pipeline.gui.calibration_screen as calibration_screen_module  # noqa: E402
from pipeline.gui.calibration_dialogs import (  # noqa: E402
    BaselineDialog,
    ConversionGainDialog,
    FlatFieldDialog,
    SpatialCalibrationDialog,
)
from pipeline.gui.calibration_screen import (  # noqa: E402
    CalibrationScreen,
    SPECTRAL_UNAVAILABLE_NOTE,
)
from pipeline.gui.live_view import (  # noqa: E402
    DEFAULT_DEGREE,
    DEGREE_CHOICES,
    EVALUATED_AT_COLUMN,
    MICRONS_PER_MM,
    LiveViewWidget,
    _PowerOfTenAxisItem,
    evaluated_at_text,
    fit_formula_html,
    format_power_of_ten_superscript,
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


def _patch_successful_calibration_load(monkeypatch) -> tuple[SimpleNamespace, SimpleNamespace, ScaleFactorPositionCalibration]:
    '''
    Patches calibration_screen's load_baseline()/load_flat_field()/
    load_bad_pixel_map()/load_conversion_gain()/load_scale_factor() calls
    to all succeed with synthetic data, mirroring _calibration_set() above.
    Returns (baseline_result, conversion_gain_result, position_calibration)
    so callers can assert the returned CalibrationBundle was actually built
    from these values.
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


def _patch_missing_calibration_load(monkeypatch) -> list:
    '''
    Patches every load_*() call calibration_screen makes to raise
    FileNotFoundError, and show_calibration_error_dialog() to record its
    arguments instead of opening a real modal dialog. Returns the list
    show_calibration_error_dialog() calls get appended to, as
    (title, message) tuples.
    '''
    def _raise_missing(path):
        raise FileNotFoundError(path)

    for name in ("load_baseline", "load_flat_field", "load_bad_pixel_map", "load_conversion_gain"):
        monkeypatch.setattr(calibration_screen_module, name, _raise_missing)

    error_calls = []
    monkeypatch.setattr(
        calibration_screen_module,
        "show_calibration_error_dialog",
        lambda parent, title, message: error_calls.append((title, message)),
    )
    return error_calls


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


# ---------------------------------------------------------------------------
# calibration_screen.py / calibration_dialogs.py
# ---------------------------------------------------------------------------

def test_calibration_screen_launches(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    assert screen.welcome_page is not None
    assert screen.create_page is not None
    assert screen.get_calibration_bundle() is None


def test_welcome_page_create_requested_navigates_to_create_page(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)

    screen.welcome_page.create_requested.emit()
    assert screen._stack.currentWidget() is screen.create_page


def test_load_existing_calibrations_success_emits_calibration_ready(qtbot, monkeypatch):
    baseline_result, conversion_gain_result, position_calibration = (
        _patch_successful_calibration_load(monkeypatch)
    )
    screen = CalibrationScreen()
    qtbot.addWidget(screen)

    received = []
    screen.calibration_ready.connect(received.append)

    screen.welcome_page.load_requested.emit()

    # No intermediate review page -- the hand-off happens immediately,
    # in place, without ever leaving WelcomePage.
    assert screen._stack.currentWidget() is screen.welcome_page

    assert len(received) == 1
    bundle = received[0]
    assert bundle is screen.get_calibration_bundle()
    assert bundle.calibration_set.background_sigma == pytest.approx(
        baseline_result.background_sigma
    )
    assert bundle.noise_model.gain_e_per_adu == pytest.approx(
        conversion_gain_result.gain_e_per_adu
    )
    assert bundle.position_calibration is position_calibration


def test_load_existing_calibrations_missing_artifact_shows_error(qtbot, monkeypatch):
    error_calls = _patch_missing_calibration_load(monkeypatch)
    screen = CalibrationScreen()
    qtbot.addWidget(screen)

    received = []
    screen.calibration_ready.connect(received.append)

    screen.welcome_page.load_requested.emit()

    assert received == []
    assert screen.get_calibration_bundle() is None
    assert screen._stack.currentWidget() is screen.welcome_page
    assert len(error_calls) == 1
    title, message = error_calls[0]
    assert title == "No Existing Calibrations"
    assert message == "No existing calibrations found. Please create new calibrations."


def test_create_page_has_four_enabled_type_cards_and_no_bad_pixel_option(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    create_page = screen.create_page

    for card in (
        create_page.baseline_card,
        create_page.flat_field_card,
        create_page.conversion_gain_card,
        create_page.spatial_card,
    ):
        assert card.action_button.isEnabled()

    assert not create_page.spectral_card.action_button.isEnabled()
    assert not hasattr(create_page, "bad_pixel_card")


def test_spectral_card_shows_unavailable_note(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    assert screen.create_page.spectral_card.description_label.text() == SPECTRAL_UNAVAILABLE_NOTE


def test_baseline_dialog_has_n_frames_and_gain_fields(qtbot):
    dialog = BaselineDialog()
    qtbot.addWidget(dialog)
    assert dialog.n_frames_spin.value() > 0
    assert dialog.gain_db_spin is not None
    assert dialog.start_button is not None


def test_flat_field_dialog_two_phase_sequence(qtbot):
    dialog = FlatFieldDialog()
    qtbot.addWidget(dialog)

    assert dialog._phase == dialog.PHASE_DARK
    assert "Block the beam" in dialog.instruction_label.text()

    dialog._advance_phase()
    assert dialog._phase == dialog.PHASE_ILLUMINATED
    assert "uniform illumination" in dialog.instruction_label.text()

    dialog._advance_phase()
    assert dialog._phase == dialog.PHASE_FINISHING
    assert "automatically" in dialog.status_label.text()


def test_conversion_gain_dialog_has_all_required_fields(qtbot):
    dialog = ConversionGainDialog()
    qtbot.addWidget(dialog)
    assert dialog.exposure_min_spin is not None
    assert dialog.exposure_max_spin is not None
    assert dialog.n_levels_spin is not None
    assert dialog.n_frames_per_level_spin is not None
    assert dialog.gain_db_spin is not None


def test_spatial_dialog_defaults_to_given_scale_factor(qtbot):
    dialog = SpatialCalibrationDialog(1.5)
    qtbot.addWidget(dialog)
    assert dialog.scale_factor_spin.value() == pytest.approx(1.5)


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

    def test_extended_measurement_button_present_but_unwired(self, qtbot):
        # Non-goal per spec: the button exists as a landing spot for a
        # future feature, but must not be connected to anything yet.
        widget = _make_live_view_widget(qtbot)
        button = widget._extended_measurement_button
        assert button.text() == "Extended Measurement..."
        assert button.receivers("2clicked()") == 0


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
