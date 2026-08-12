'''
Test suite for calibration_screen.py/calibration_dialogs.py's Phase-1
visual skeleton. Split out of tests/test_gui.py (which grew to cover all
four gui/ screens in one file) so calibration-wiring work, live-view
wiring work, and extended-measurement wiring work can each land in their
own dedicated test file with zero risk of the same file being edited by
more than one of those efforts at once -- mirrors tests/test_calibration.py
being split out of tests/test_preprocessing.py for the same reason
(see that file's own module docstring).

Most of this screen has no real camera/calibration-package call wired in
yet (see calibration_screen.py's and calibration_dialogs.py's own module
docstrings), so most of these tests only check structure/layout/state
transitions -- not behavior, e.g. that CreatePage exposes exactly five
enabled type cards (bad-pixel-map has no manual "create" option of its
own), not that clicking "Configure..." actually acquires anything. The
exception is WelcomePage's "Load Existing Calibrations" flow, which is
fully wired (see calibration_screen.CalibrationScreen.
_attempt_load_existing_calibrations()) -- its tests mock calibration_
screen's load_*()/show_calibration_error_dialog() calls rather than
touching a real calibration_artifacts/ directory or camera. A follow-up
pass adds the remaining real-logic tests (automatic bad-pixel-map
chaining on CreatePage, error-dialog paths there) once each is wired,
using the same mocked-call convention.

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
from pipeline.calibration.sensor import ConversionGainRecord  # noqa: E402
from pipeline.calibration.shared import CalibrationRecord, GAIN_MATCH_TOLERANCE_ABS  # noqa: E402
from pipeline.calibration.spatial import ScaleFactorPositionCalibration  # noqa: E402
import pipeline.gui.calibration_screen as calibration_screen_module  # noqa: E402
from pipeline.gui.calibration_dialogs import (  # noqa: E402
    BaselineDialog,
    ConversionGainDialog,
    FlatFieldDialog,
    SpatialCalibrationDialog,
    SpectralCalibrationDialog,
    manual_spectral_formula_html,
    DEFAULT_DEGREE as SPECTRAL_DEFAULT_DEGREE,
)
from pipeline.gui.calibration_screen import (  # noqa: E402
    CalibrationScreen,
)
from pipeline.gui.live_view import DEGREE_CHOICES  # noqa: E402

# Constants

FIXTURE_EXPOSURE_US = 2000.0
FIXTURE_GAIN_DB = 0.0

# Distinguishes "caller didn't pass geometric_tilt" (-> default to a
# present artifact) from "caller explicitly passed None" (-> simulate a
# missing geometric_tilt.npz) in _patch_successful_calibration_load().
_MISSING = object()

# Classes

class _FakeWavelengthAxis:

    '''Minimal WavelengthAxis stand-in -- a trivial linear pixel->nm map.
    Local copy of tests/test_live_view.py's own copy -- see this file's
    module docstring for why these small per-file duplicates exist rather
    than a shared import.'''

    def wavelength_nm(self, pixel):
        return 500.0 + 0.01 * np.asarray(pixel, dtype=float)

    def sigma_wavelength_nm(self, pixel):
        return np.full_like(np.asarray(pixel, dtype=float), 0.05)


# Functions

def _patch_successful_calibration_load(
    monkeypatch, geometric_tilt=_MISSING,
) -> tuple[SimpleNamespace, SimpleNamespace, ScaleFactorPositionCalibration, _FakeWavelengthAxis]:
    '''
    Patches calibration_screen's load_baseline()/load_flat_field()/
    load_bad_pixel_map()/load_conversion_gain()/load_spectral_calibration()/
    load_scale_factor()/load_geometric_tilt() calls to all succeed with
    synthetic data, mirroring _calibration_set() above. Returns
    (baseline_result, conversion_gain_result, position_calibration,
    wavelength_axis) so callers can assert the returned CalibrationBundle
    was actually built from these values.

    geometric_tilt
        The value load_geometric_tilt() should return. Defaults to a
        sentinel object (not None) so tests can also cover the "artifact
        genuinely exists" path -- pass None explicitly to instead make
        load_geometric_tilt() raise FileNotFoundError (the "not built
        yet" case CalibrationBundle's docstring says is fine).
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
    wavelength_axis = _FakeWavelengthAxis()
    tilt_result = object() if geometric_tilt is _MISSING else geometric_tilt

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
        calibration_screen_module, "load_spectral_calibration", lambda path: wavelength_axis
    )
    monkeypatch.setattr(
        calibration_screen_module,
        "load_scale_factor",
        lambda path: (position_calibration, object()),
    )

    def _load_geometric_tilt(path):
        if tilt_result is None:
            raise FileNotFoundError(path)
        return tilt_result

    monkeypatch.setattr(calibration_screen_module, "load_geometric_tilt", _load_geometric_tilt)

    return baseline_result, conversion_gain_result, position_calibration, wavelength_axis


def _patch_missing_calibration_load(monkeypatch) -> list:
    '''
    Patches every hard-required load_*() call calibration_screen makes to
    raise FileNotFoundError, and show_calibration_error_dialog() to record
    its arguments instead of opening a real modal dialog. Returns the list
    show_calibration_error_dialog() calls get appended to, as
    (title, message) tuples.
    '''
    def _raise_missing(path):
        raise FileNotFoundError(path)

    for name in (
        "load_baseline", "load_flat_field", "load_bad_pixel_map",
        "load_conversion_gain", "load_spectral_calibration",
    ):
        monkeypatch.setattr(calibration_screen_module, name, _raise_missing)

    error_calls = []
    monkeypatch.setattr(
        calibration_screen_module,
        "show_calibration_error_dialog",
        lambda parent, title, message: error_calls.append((title, message)),
    )
    return error_calls


def _patch_mismatched_gain_calibration_load(monkeypatch) -> list:
    '''
    Like _patch_successful_calibration_load(), but baseline_record and
    conversion_gain_record are tagged with gain_db values that differ by
    more than GAIN_MATCH_TOLERANCE_ABS -- exercises
    check_conversion_gain_matches_baseline()'s SettingsMismatchError path
    in _attempt_load_existing_calibrations(). Also patches
    show_calibration_error_dialog() to record its arguments instead of
    opening a real modal dialog. Returns the list its calls get appended
    to, as (title, message) tuples.
    '''
    baseline_record = CalibrationRecord(
        exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
        timestamp=time.time(), source_frame_count=50,
    )
    conversion_gain_record = ConversionGainRecord(
        gain_db=FIXTURE_GAIN_DB + GAIN_MATCH_TOLERANCE_ABS * 2,
        timestamp=time.time(), n_illumination_levels=5,
    )
    baseline_result = SimpleNamespace(
        baseline=np.full(CANONICAL_SHAPE, 10.0, dtype=np.float64), background_sigma=1.0
    )
    flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
    bad_pixel_mask = np.zeros(CANONICAL_SHAPE, dtype=bool)
    conversion_gain_result = SimpleNamespace(gain_e_per_adu=2.2)

    monkeypatch.setattr(
        calibration_screen_module,
        "load_baseline",
        lambda path: (baseline_result, baseline_record),
    )
    monkeypatch.setattr(
        calibration_screen_module, "load_flat_field", lambda path: (flat_field, baseline_record)
    )
    monkeypatch.setattr(
        calibration_screen_module,
        "load_bad_pixel_map",
        lambda path: (bad_pixel_mask, baseline_record),
    )
    monkeypatch.setattr(
        calibration_screen_module,
        "load_conversion_gain",
        lambda path: (conversion_gain_result, conversion_gain_record),
    )
    # Reached before the gain-mismatch check runs (see
    # _attempt_load_existing_calibrations()'s load order) -- must succeed
    # so the flow reaches that check rather than failing earlier as
    # "No Existing Calibrations".
    monkeypatch.setattr(
        calibration_screen_module, "load_spectral_calibration", lambda path: _FakeWavelengthAxis()
    )

    error_calls = []
    monkeypatch.setattr(
        calibration_screen_module,
        "show_calibration_error_dialog",
        lambda parent, title, message: error_calls.append((title, message)),
    )
    return error_calls


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
    baseline_result, conversion_gain_result, position_calibration, wavelength_axis = (
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
    assert bundle.wavelength_axis is wavelength_axis


def test_load_existing_calibrations_missing_spectral_shows_error(qtbot, monkeypatch):
    # Spectral is hard-required, same as baseline/flat-field/bad-pixel-map/
    # conversion-gain -- unlike geometric tilt (see the soft-optional test
    # below), a missing spectral.npz must block the load.
    def _raise_missing(path):
        raise FileNotFoundError(path)

    _patch_successful_calibration_load(monkeypatch)
    monkeypatch.setattr(calibration_screen_module, "load_spectral_calibration", _raise_missing)
    error_calls = []
    monkeypatch.setattr(
        calibration_screen_module,
        "show_calibration_error_dialog",
        lambda parent, title, message: error_calls.append((title, message)),
    )

    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    received = []
    screen.calibration_ready.connect(received.append)

    screen.welcome_page.load_requested.emit()

    assert received == []
    assert screen.get_calibration_bundle() is None
    assert len(error_calls) == 1


def test_load_existing_calibrations_missing_geometric_tilt_still_succeeds(qtbot, monkeypatch):
    # Geometric tilt is soft-optional (see CalibrationBundle's own
    # docstring): a missing geometric_tilt.npz must NOT block the load --
    # it just leaves CalibrationSet.geometric_tilt as None.
    _patch_successful_calibration_load(monkeypatch, geometric_tilt=None)
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    received = []
    screen.calibration_ready.connect(received.append)

    screen.welcome_page.load_requested.emit()

    assert len(received) == 1
    assert received[0].calibration_set.geometric_tilt is None


def test_load_existing_calibrations_present_geometric_tilt_is_threaded_through(qtbot, monkeypatch):
    sentinel_tilt = object()
    _patch_successful_calibration_load(monkeypatch, geometric_tilt=sentinel_tilt)
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    received = []
    screen.calibration_ready.connect(received.append)

    screen.welcome_page.load_requested.emit()

    assert len(received) == 1
    assert received[0].calibration_set.geometric_tilt is sentinel_tilt


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


def test_load_existing_calibrations_gain_mismatch_shows_error_and_blocks_ready(
    qtbot, monkeypatch
):
    error_calls = _patch_mismatched_gain_calibration_load(monkeypatch)
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
    assert title == "Calibration Settings Mismatch"
    assert "different" in message.lower()
    assert "gain" in message.lower()


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


def test_baseline_dialog_defaults_to_auto_exposure(qtbot):
    dialog = BaselineDialog()
    qtbot.addWidget(dialog)
    assert dialog.auto_exposure() is True
    assert dialog.exposure_us() is None
    assert not dialog.exposure_us_spin.isEnabled()
    assert dialog.gain_db_spin.value() == pytest.approx(0.0)


def test_baseline_dialog_manual_exposure_enables_field_and_getters(qtbot):
    dialog = BaselineDialog()
    qtbot.addWidget(dialog)
    dialog.exposure_mode_combo.setCurrentText("Manual")

    assert dialog.exposure_us_spin.isEnabled()
    assert dialog.auto_exposure() is False

    dialog.exposure_us_spin.setValue(5000.0)
    assert dialog.exposure_us() == pytest.approx(5000.0)


def test_baseline_dialog_manual_does_not_reset_gain(qtbot):
    dialog = BaselineDialog()
    qtbot.addWidget(dialog)
    dialog.exposure_mode_combo.setCurrentText("Manual")
    dialog.gain_db_spin.setValue(12.5)
    dialog.exposure_us_spin.setValue(5000.0)

    assert dialog.gain_db_spin.value() == pytest.approx(12.5)


def test_baseline_dialog_switching_back_to_auto_resets_gain(qtbot):
    dialog = BaselineDialog()
    qtbot.addWidget(dialog)
    dialog.exposure_mode_combo.setCurrentText("Manual")
    dialog.gain_db_spin.setValue(12.5)

    dialog.exposure_mode_combo.setCurrentText("Auto")

    assert dialog.gain_db_spin.value() == pytest.approx(0.0)
    assert not dialog.exposure_us_spin.isEnabled()
    assert dialog.exposure_us() is None


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


def test_flat_field_dialog_defaults_to_auto_exposure(qtbot):
    dialog = FlatFieldDialog()
    qtbot.addWidget(dialog)
    assert dialog.auto_exposure() is True
    assert dialog.exposure_us() is None
    assert not dialog.exposure_us_spin.isEnabled()
    assert dialog.gain_db_spin.value() == pytest.approx(0.0)


def test_flat_field_dialog_manual_exposure_enables_field_and_getters(qtbot):
    dialog = FlatFieldDialog()
    qtbot.addWidget(dialog)
    dialog.exposure_mode_combo.setCurrentText("Manual")

    assert dialog.exposure_us_spin.isEnabled()
    assert dialog.auto_exposure() is False

    dialog.exposure_us_spin.setValue(3500.0)
    assert dialog.exposure_us() == pytest.approx(3500.0)


def test_flat_field_dialog_switching_back_to_auto_resets_gain(qtbot):
    dialog = FlatFieldDialog()
    qtbot.addWidget(dialog)
    dialog.exposure_mode_combo.setCurrentText("Manual")
    dialog.gain_db_spin.setValue(8.0)

    dialog.exposure_mode_combo.setCurrentText("Auto")

    assert dialog.gain_db_spin.value() == pytest.approx(0.0)
    assert not dialog.exposure_us_spin.isEnabled()
    assert dialog.exposure_us() is None


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
    assert dialog.capture_degree_selector.currentData() == SPECTRAL_DEFAULT_DEGREE


def test_spectral_dialog_manual_degree_defaults_to_cubic_coefficient_rows(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    dialog.manual_mode_radio.setChecked(True)
    assert dialog.manual_degree_selector.currentData() == SPECTRAL_DEFAULT_DEGREE
    assert len(dialog._coefficient_rows) == SPECTRAL_DEFAULT_DEGREE + 1


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


def test_spectral_dialog_manual_formula_label_matches_default_degree(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    dialog.manual_mode_radio.setChecked(True)
    default_degree = dialog.manual_degree_selector.currentData()
    assert dialog.formula_label.text() == manual_spectral_formula_html(default_degree)


def test_spectral_dialog_manual_formula_label_updates_with_degree(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    dialog.manual_mode_radio.setChecked(True)

    quadratic_index = DEGREE_CHOICES.index(2)
    dialog.manual_degree_selector.setCurrentIndex(quadratic_index)
    assert dialog.formula_label.text() == manual_spectral_formula_html(2)


class TestManualSpectralFormulaHtml:

    def test_degree_one(self):
        assert manual_spectral_formula_html(1) == "λ = c<sub>0</sub> + c<sub>1</sub>x"

    def test_degree_two_adds_squared_term(self):
        html = manual_spectral_formula_html(2)
        assert html.endswith("c<sub>2</sub>x<sup>2</sup>")

    def test_degree_three_adds_cubed_term(self):
        html = manual_spectral_formula_html(3)
        assert html.endswith("c<sub>3</sub>x<sup>3</sup>")

    def test_degree_increases_term_count(self):
        html1 = manual_spectral_formula_html(1)
        html3 = manual_spectral_formula_html(3)
        assert html1.count("c<sub>") == 2
        assert html3.count("c<sub>") == 4


def test_spectral_dialog_manual_getters_return_entered_values(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    dialog.manual_mode_radio.setChecked(True)
    dialog.manual_degree_selector.setCurrentIndex(DEGREE_CHOICES.index(1))

    values = [780.0, 0.045]
    sigmas = [0.5, 0.001]
    for (value_spin, sigma_spin), value, sigma in zip(dialog._coefficient_rows, values, sigmas):
        value_spin.setValue(value)
        sigma_spin.setValue(sigma)

    assert dialog.coefficients() == pytest.approx(values)
    assert dialog.coefficient_sigma() == pytest.approx(sigmas)
