"""
Test suite for the gui package. Covers calibration_screen.py/
calibration_dialogs.py's Phase-1 visual skeleton, and live_view.py's
Phase-1 visual skeleton, as widget-layer smoke tests plus (for live_view.py)
ordinary unit tests of its plain, non-Qt presentational helper functions.

Neither screen has any real camera/calibration-package/preprocessing/
analysis call wired in yet (see each module's own docstring), so these
tests only check structure/layout/state transitions -- not behavior, e.g.
that CreatePage exposes exactly the four enabled type cards plus a
disabled spectral card, not that clicking "Configure..." actually
acquires anything; that degree selection changes LiveViewWidget's
*displayed* placeholder numbers, not that it triggers a real refit. A
follow-up pass adds real-logic tests (automatic bad-pixel-map chaining,
error-dialog paths, the QTimer-driven update loop) once each is wired,
using mocked calibration/camera calls rather than a real camera.

PySide6/pyqtgraph/pytest-qt are local-only dependencies (see CLAUDE.md --
pyproject.toml/requirements.txt are deliberately left empty), so this
whole module is skipped, not failed, wherever they aren't installed --
the same "gate, don't require" pattern tests/test_acquisition.py uses for
hardware-only tests.
"""

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

from pipeline.acquisition import CameraStream, SyntheticBackend, CANONICAL_SHAPE  # noqa: E402
from pipeline.analysis import SensorNoiseModel  # noqa: E402
from pipeline.calibration.shared import CalibrationRecord  # noqa: E402
from pipeline.calibration.spatial import ScaleFactorPositionCalibration  # noqa: E402
from pipeline.preprocessing import CalibrationSet  # noqa: E402
from pipeline.gui.calibration_dialogs import (  # noqa: E402
    BaselineDialog,
    ConversionGainDialog,
    FlatFieldDialog,
    SpatialCalibrationDialog,
    SpectralCalibrationDialog,
)
from pipeline.gui.calibration_screen import (  # noqa: E402
    CalibrationScreen,
    _LOAD_ROWS,
)
from pipeline.gui.live_view import (  # noqa: E402
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
    assert screen.load_page is not None
    assert screen.create_page is not None
    assert screen.get_calibration_bundle() is None


def test_welcome_page_navigates_to_load_and_create(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)

    screen.welcome_page.load_requested.emit()
    assert screen._stack.currentWidget() is screen.load_page

    screen.welcome_page.create_requested.emit()
    assert screen._stack.currentWidget() is screen.create_page


def test_load_page_lists_five_artifact_rows(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    assert len(_LOAD_ROWS) == 5
    assert not screen.load_page.continue_button.isEnabled()


def test_create_page_has_five_enabled_type_cards_and_no_bad_pixel_option(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    create_page = screen.create_page

    for card in (
        create_page.baseline_card,
        create_page.flat_field_card,
        create_page.conversion_gain_card,
        create_page.spatial_card,
        create_page.spectral_card,
    ):
        assert card.action_button.isEnabled()

    assert not hasattr(create_page, "bad_pixel_card")


def test_spectral_card_has_real_action_button(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    card = screen.create_page.spectral_card
    assert card.action_button.isEnabled()
    assert card.action_button.text() != "Unavailable"
    assert card.description_label.text() != ""


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


def test_spectral_dialog_defaults_to_capture_mode(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    assert dialog.capture_mode_radio.isChecked()
    assert dialog._mode_stack.currentIndex() == 0
    # isHidden() (an explicit-hide flag) rather than isVisible() (which
    # also depends on the top-level dialog having been shown -- always
    # False here since the test never calls dialog.show()).
    assert not dialog.start_button.isHidden()
    assert dialog.save_button.isHidden()


def test_spectral_dialog_mode_selector_switches_sections(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)

    dialog.manual_mode_radio.setChecked(True)

    assert dialog._mode_stack.currentIndex() == 1
    assert not dialog.save_button.isHidden()
    assert dialog.start_button.isHidden()

    dialog.capture_mode_radio.setChecked(True)

    assert dialog._mode_stack.currentIndex() == 0
    assert not dialog.start_button.isHidden()
    assert dialog.save_button.isHidden()


def test_spectral_dialog_capture_mode_has_frame_gain_and_degree_fields(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    assert dialog.n_frames_spin.value() > 0
    assert dialog.gain_db_spin is not None
    assert dialog.capture_degree_selector.currentData() == DEFAULT_DEGREE


def test_spectral_dialog_manual_degree_defaults_to_two_coefficient_rows(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    dialog.manual_mode_radio.setChecked(True)
    assert dialog.manual_degree_selector.currentData() == DEFAULT_DEGREE
    assert len(dialog._coefficient_rows) == DEFAULT_DEGREE + 1


def test_spectral_dialog_manual_degree_change_rebuilds_coefficient_rows(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    dialog.manual_mode_radio.setChecked(True)

    quadratic_index = DEGREE_CHOICES.index(2)
    dialog.manual_degree_selector.setCurrentIndex(quadratic_index)
    assert len(dialog._coefficient_rows) == 3

    cubic_index = DEGREE_CHOICES.index(3)
    dialog.manual_degree_selector.setCurrentIndex(cubic_index)
    assert len(dialog._coefficient_rows) == 4

    linear_index = DEGREE_CHOICES.index(1)
    dialog.manual_degree_selector.setCurrentIndex(linear_index)
    assert len(dialog._coefficient_rows) == 2


def test_spectral_dialog_manual_getters_return_entered_values(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    dialog.manual_mode_radio.setChecked(True)

    values = [780.0, 0.045]
    sigmas = [0.5, 0.001]
    for (value_spin, sigma_spin), value, sigma in zip(dialog._coefficient_rows, values, sigmas):
        value_spin.setValue(value)
        sigma_spin.setValue(sigma)

    assert dialog.coefficients() == pytest.approx(values)
    assert dialog.coefficient_sigma() == pytest.approx(sigmas)


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
