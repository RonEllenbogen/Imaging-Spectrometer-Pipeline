'''
Test suite for calibration_screen.py's/calibration_dialogs.py's real
wiring: WelcomePage's "Load Existing Calibrations" flow, CreatePage's
per-type completion tracking gating "Continue to Main Window" through to
the same CalibrationScreen._attempt_load_existing_calibrations() call, and
each dialog's Start/Continue/Save path calling the actual
calibration/sensor/, calibration/spatial/, calibration/spectral/
functions -- with error handling routing CameraError to
show_camera_error_dialog() and calibration-specific failures to
show_calibration_error_dialog() (leaving the originating dialog open so
the user can retry). Split out of tests/test_gui.py (which grew to cover
all four gui/ screens in one file) so calibration-wiring work, live-view
wiring work, and extended-measurement wiring work can each land in their
own dedicated test file with zero risk of the same file being edited by
more than one of those efforts at once -- mirrors tests/test_calibration.py
being split out of tests/test_preprocessing.py for the same reason (see
that file's own module docstring).

Structural/layout smoke tests for the five calibration dialogs (that they
expose the expected fields, that mode selectors/phase state machines
transition correctly) live alongside the real-wiring tests below rather
than in a separate file -- they exercise the same dialog classes and
would otherwise duplicate setup.

Every test that exercises a code path which *could* open a real
QMessageBox (show_camera_error_dialog()/show_calibration_error_dialog())
mocks it -- a real QMessageBox.exec()/QDialog.exec() call in an offscreen
test blocks forever waiting for a click that will never come (diagnosed,
the hard way, via an lldb backtrace showing a hung test process parked in
QDialog::exec() -- see this repo's other gui/ test files for the same
note). Where a dialog's own accept path is exercised end-to-end, it is
driven by clicking its buttons directly (never by calling dialog.exec(),
which would itself block); CreatePage-level tests that need to go through
a dialog's real exec() call instead monkeypatch that one dialog class's
exec() to click its own button and return the resulting result code,
without ever starting a real nested Qt event loop.

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

from PySide6.QtWidgets import QDialog  # noqa: E402

from pipeline.acquisition import (  # noqa: E402
    CameraConnectionError,
    CameraStream,
    CANONICAL_SHAPE,
    SyntheticBackend,
)
from pipeline.calibration.exceptions import (  # noqa: E402
    InvalidConversionGainError,
    InvalidFlatFieldError,
    LineMatchingError,
)
from pipeline.calibration.sensor import (  # noqa: E402
    BaselineResult,
    ConversionGainRecord,
    ConversionGainResult,
    save_bad_pixel_map,
    save_baseline,
    save_conversion_gain,
    save_flat_field,
)
from pipeline.calibration.shared import (  # noqa: E402
    CalibrationRecord,
    GAIN_MATCH_TOLERANCE_ABS,
    PolynomialFitResult,
)
from pipeline.calibration.spatial import (  # noqa: E402
    DEFAULT_SCALE_FACTOR,
    ScaleFactorPositionCalibration,
    load_scale_factor,
)
from pipeline.calibration.spectral import load_spectral_calibration  # noqa: E402
from pipeline.cli.calibration import (  # noqa: E402
    DEFAULT_BAD_PIXEL_MAP_FILENAME,
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_CONVERSION_GAIN_FILENAME,
    DEFAULT_FLAT_FIELD_FILENAME,
    DEFAULT_GEOMETRIC_TILT_FILENAME,
    DEFAULT_SCALE_FACTOR_FILENAME,
    DEFAULT_SPECTRAL_FILENAME,
)
from pipeline.preprocessing import NoSignalError  # noqa: E402

import pipeline.gui.calibration_dialogs as calibration_dialogs_module  # noqa: E402
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
    CreatePage,
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


def _synthetic_camera_stream(peak_counts: float = 150.0, seed: int = 0) -> CameraStream:
    '''
    An unstarted CameraStream over SyntheticBackend -- non-saturating
    peak_counts (Mono8's ceiling is 255) so flat-field capture doesn't
    trip build_flat_field()'s saturation check, mirroring tests/
    test_calibration.py's TestFlatFieldCalibrationWorkflow.
    '''
    return CameraStream(
        exposure_us=2000.0, gain_db=FIXTURE_GAIN_DB, pixel_format="Mono8", timeout_ms=5000,
        backend=SyntheticBackend(seed=seed, peak_counts=peak_counts),
    )


def _record() -> CalibrationRecord:
    return CalibrationRecord(
        exposure_us=2000.0, gain_db=FIXTURE_GAIN_DB, timestamp=time.time(), source_frame_count=5,
    )


def _save_sensor_artifacts(artifact_dir) -> None:
    '''Writes minimal, valid baseline/flat-field/bad-pixel-map/conversion-
    gain artifacts to artifact_dir (at the same DEFAULT_*_FILENAME paths
    calibration_dialogs.py's dialogs read from/write to) -- enough for
    SpectralCalibrationDialog's capture-mode _load_sensor_calibration()
    to succeed without needing a full real flat-field/conversion-gain
    capture session.'''
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record = _record()
    save_baseline(
        artifact_dir / DEFAULT_BASELINE_FILENAME,
        BaselineResult(baseline=np.full(CANONICAL_SHAPE, 10.0), background_sigma=1.0),
        record,
    )
    save_flat_field(artifact_dir / DEFAULT_FLAT_FIELD_FILENAME, np.ones(CANONICAL_SHAPE), record)
    save_bad_pixel_map(
        artifact_dir / DEFAULT_BAD_PIXEL_MAP_FILENAME, np.zeros(CANONICAL_SHAPE, dtype=bool), record
    )
    save_conversion_gain(
        artifact_dir / DEFAULT_CONVERSION_GAIN_FILENAME,
        ConversionGainResult(fit=PolynomialFitResult(
            degree=1,
            coefficients=np.array([0.5, 0.4]),
            coefficient_sigma=np.array([0.1, 0.05]),
            reduced_chi_squared=1.0,
            residuals=np.zeros(3),
            normalized_residuals=np.zeros(3),
        )),
        ConversionGainRecord(gain_db=FIXTURE_GAIN_DB, timestamp=time.time(), n_illumination_levels=5),
    )


def _record_calls(monkeypatch, target_module, name: str) -> list:
    '''Monkeypatches target_module.name to record its call args (as a
    dict of the positional/keyword args it was called with) instead of
    opening a real modal dialog, returning the list those records get
    appended to.'''
    calls = []

    def _fake(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(target_module, name, _fake)
    return calls


# ---------------------------------------------------------------------------
# calibration_screen.py / calibration_dialogs.py -- structure and
# WelcomePage's "Load Existing Calibrations" flow
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


def test_create_page_no_existing_calibrations_leaves_cards_unmarked(qtbot, monkeypatch):
    _patch_missing_calibration_load(monkeypatch)
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    create_page = screen.create_page

    for card in (
        create_page.baseline_card,
        create_page.flat_field_card,
        create_page.conversion_gain_card,
        create_page.spectral_card,
    ):
        assert card.status_label.isHidden()
    assert not create_page.continue_button.isEnabled()


def test_create_page_preloads_existing_calibrations_from_disk(qtbot, monkeypatch):
    '''
    Regression test: a user who created baseline/flat-field/conversion-
    gain/spectral calibrations in an earlier app run, then closed and
    relaunched, must not have to redo them just to reach "Continue to Main
    Window" again -- see CreatePage._mark_existing_calibrations().
    '''
    _patch_successful_calibration_load(monkeypatch)
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    create_page = screen.create_page

    for card in (
        create_page.baseline_card,
        create_page.flat_field_card,
        create_page.conversion_gain_card,
        create_page.spectral_card,
    ):
        # isHidden() rather than isVisible() -- see
        # test_spectral_dialog_defaults_to_capture_mode()'s comment.
        assert not card.status_label.isHidden()
        assert "existing calibration" in card.status_label.text().lower()

    # every gated type was found on disk -- Continue is enabled without
    # the user ever opening a single dialog this session
    assert create_page.continue_button.isEnabled()


def test_create_page_partial_existing_calibrations_does_not_enable_continue(qtbot, monkeypatch):
    _patch_successful_calibration_load(monkeypatch)

    def _raise_missing(path):
        raise FileNotFoundError(path)

    # spectral was never captured in the previous session
    monkeypatch.setattr(calibration_screen_module, "load_spectral_calibration", _raise_missing)

    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    create_page = screen.create_page

    assert not create_page.baseline_card.status_label.isHidden()
    assert not create_page.flat_field_card.status_label.isHidden()
    assert not create_page.conversion_gain_card.status_label.isHidden()
    assert create_page.spectral_card.status_label.isHidden()
    assert not create_page.continue_button.isEnabled()


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


def test_flat_field_dialog_two_phase_sequence(qtbot, monkeypatch):
    # _advance_phase() now drives real camera capture on phases 1->2/2->3
    # (see calibration_dialogs.py) -- build_camera_stream is mocked so
    # this structural test doesn't touch real hardware.
    monkeypatch.setattr(
        calibration_dialogs_module, "build_camera_stream",
        lambda *a, **k: _synthetic_camera_stream(),
    )
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


def test_spectral_dialog_defaults_to_auto_exposure(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    assert dialog.auto_exposure() is True
    assert dialog.exposure_us() is None
    assert not dialog.exposure_us_spin.isEnabled()
    assert dialog.gain_db_spin.value() == pytest.approx(0.0)


def test_spectral_dialog_manual_exposure_enables_field_and_getters(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    dialog.exposure_mode_combo.setCurrentText("Manual")

    assert dialog.exposure_us_spin.isEnabled()
    assert dialog.auto_exposure() is False

    dialog.exposure_us_spin.setValue(1000.0)
    assert dialog.exposure_us() == pytest.approx(1000.0)


def test_spectral_dialog_switching_back_to_auto_resets_gain(qtbot):
    dialog = SpectralCalibrationDialog()
    qtbot.addWidget(dialog)
    dialog.exposure_mode_combo.setCurrentText("Manual")
    dialog.gain_db_spin.setValue(9.0)

    dialog.exposure_mode_combo.setCurrentText("Auto")

    assert dialog.gain_db_spin.value() == pytest.approx(0.0)
    assert not dialog.exposure_us_spin.isEnabled()
    assert dialog.exposure_us() is None


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


# ---------------------------------------------------------------------------
# BaselineDialog -- real wiring
# ---------------------------------------------------------------------------

class TestBaselineDialogWiring:

    def test_start_button_runs_real_baseline_calibration_and_accepts(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        monkeypatch.setattr(
            calibration_dialogs_module, "build_camera_stream",
            lambda *a, **k: _synthetic_camera_stream(),
        )
        dialog = BaselineDialog()
        qtbot.addWidget(dialog)
        dialog.n_frames_spin.setValue(3)

        dialog.start_button.click()

        assert dialog.result() == QDialog.DialogCode.Accepted
        assert (tmp_path / DEFAULT_BASELINE_FILENAME).exists()
        assert dialog.status_label.text() == "Baseline calibration complete."

    def test_camera_error_shows_camera_dialog_and_leaves_dialog_open(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)

        def _raise_camera_error(*a, **k):
            raise CameraConnectionError("no device found")

        monkeypatch.setattr(calibration_dialogs_module, "build_camera_stream", _raise_camera_error)
        camera_error_calls = _record_calls(monkeypatch, calibration_dialogs_module, "show_camera_error_dialog")
        calibration_error_calls = _record_calls(
            monkeypatch, calibration_dialogs_module, "show_calibration_error_dialog"
        )

        dialog = BaselineDialog()
        qtbot.addWidget(dialog)

        dialog.start_button.click()

        assert dialog.result() != QDialog.DialogCode.Accepted
        assert dialog.start_button.isEnabled()
        assert len(camera_error_calls) == 1
        assert calibration_error_calls == []

    def test_calibration_error_shows_calibration_dialog_and_leaves_dialog_open(
        self, qtbot, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        monkeypatch.setattr(
            calibration_dialogs_module, "build_camera_stream",
            lambda *a, **k: _synthetic_camera_stream(),
        )

        def _raise_no_signal(*a, **k):
            raise NoSignalError(frame_id=0)

        monkeypatch.setattr(calibration_dialogs_module, "run_baseline_calibration", _raise_no_signal)
        calibration_error_calls = _record_calls(
            monkeypatch, calibration_dialogs_module, "show_calibration_error_dialog"
        )
        camera_error_calls = _record_calls(monkeypatch, calibration_dialogs_module, "show_camera_error_dialog")

        dialog = BaselineDialog()
        qtbot.addWidget(dialog)

        dialog.start_button.click()

        assert dialog.result() != QDialog.DialogCode.Accepted
        assert dialog.start_button.isEnabled()
        assert len(calibration_error_calls) == 1
        assert camera_error_calls == []
        title, message = calibration_error_calls[0][0][1], calibration_error_calls[0][0][2]
        assert title == "Baseline Calibration Failed"
        assert "no signal" in message.lower()


# ---------------------------------------------------------------------------
# FlatFieldDialog -- real wiring
# ---------------------------------------------------------------------------

class TestFlatFieldDialogWiring:

    def test_full_two_phase_capture_builds_real_flat_field_and_bad_pixel_map(
        self, qtbot, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        monkeypatch.setattr(
            calibration_dialogs_module, "build_camera_stream",
            lambda *a, **k: _synthetic_camera_stream(),
        )
        dialog = FlatFieldDialog()
        qtbot.addWidget(dialog)
        dialog.n_frames_spin.setValue(3)

        dialog.continue_button.click()   # phase 1 -> 2 (real dark capture)
        assert dialog._phase == dialog.PHASE_ILLUMINATED
        assert dialog._camera_stream is not None
        assert dialog._camera_stream.is_running
        assert len(dialog._dark_frames) == 3

        dialog.continue_button.click()   # phase 2 -> 3 (real illuminated capture + finish)
        assert dialog._phase == dialog.PHASE_FINISHING
        assert (tmp_path / DEFAULT_FLAT_FIELD_FILENAME).exists()
        assert (tmp_path / DEFAULT_BAD_PIXEL_MAP_FILENAME).exists()
        assert dialog._camera_stream is None   # stopped + cleared after finishing

        dialog.continue_button.click()   # phase 3 -> accept
        assert dialog.result() == QDialog.DialogCode.Accepted

    def test_camera_error_during_dark_phase_resets_and_shows_camera_dialog(
        self, qtbot, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)

        def _raise_camera_error(*a, **k):
            raise CameraConnectionError("no device found")

        monkeypatch.setattr(calibration_dialogs_module, "build_camera_stream", _raise_camera_error)
        camera_error_calls = _record_calls(monkeypatch, calibration_dialogs_module, "show_camera_error_dialog")

        dialog = FlatFieldDialog()
        qtbot.addWidget(dialog)

        dialog.continue_button.click()

        assert dialog._phase == dialog.PHASE_DARK
        assert dialog.result() != QDialog.DialogCode.Accepted
        assert len(camera_error_calls) == 1

    def test_invalid_flat_field_during_finish_resets_to_dark_phase_and_shows_calibration_dialog(
        self, qtbot, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        monkeypatch.setattr(
            calibration_dialogs_module, "build_camera_stream",
            lambda *a, **k: _synthetic_camera_stream(),
        )

        def _raise_invalid_flat_field(*a, **k):
            raise InvalidFlatFieldError("saturated source frame")

        monkeypatch.setattr(
            calibration_dialogs_module, "finish_flat_field_calibration", _raise_invalid_flat_field
        )
        calibration_error_calls = _record_calls(
            monkeypatch, calibration_dialogs_module, "show_calibration_error_dialog"
        )

        dialog = FlatFieldDialog()
        qtbot.addWidget(dialog)
        dialog.n_frames_spin.setValue(3)

        dialog.continue_button.click()   # phase 1 -> 2, real dark capture succeeds
        assert dialog._phase == dialog.PHASE_ILLUMINATED

        dialog.continue_button.click()   # phase 2 finish raises -> reset to phase 1

        assert dialog._phase == dialog.PHASE_DARK
        assert dialog._camera_stream is None
        assert dialog._dark_frames is None
        assert len(calibration_error_calls) == 1
        assert not (tmp_path / DEFAULT_FLAT_FIELD_FILENAME).exists()


# ---------------------------------------------------------------------------
# ConversionGainDialog -- real wiring
# ---------------------------------------------------------------------------

class TestConversionGainDialogWiring:

    def _fill_required_fields(self, dialog: ConversionGainDialog) -> None:
        dialog.exposure_min_spin.setValue(1000.0)
        dialog.exposure_max_spin.setValue(5000.0)
        dialog.n_levels_spin.setValue(3)
        dialog.n_frames_per_level_spin.setValue(3)

    def test_start_sweep_calls_run_conversion_gain_calibration_and_accepts(
        self, qtbot, monkeypatch, tmp_path
    ):
        # run_conversion_gain_calibration() is mocked rather than run for
        # real: SyntheticBackend doesn't scale its injected noise by
        # exposure_us (see backends.py), so a real sweep's variance-vs-mean
        # fit has no genuine positive slope to recover -- the same reason
        # tests/test_calibration.py's TestRunConversionGainCalibration
        # stubs build_conversion_gain() for its own wiring test.
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        stream = _synthetic_camera_stream()
        monkeypatch.setattr(calibration_dialogs_module, "build_camera_stream", lambda *a, **k: stream)
        calls = _record_calls(monkeypatch, calibration_dialogs_module, "run_conversion_gain_calibration")

        dialog = ConversionGainDialog()
        qtbot.addWidget(dialog)
        self._fill_required_fields(dialog)
        dialog.gain_db_spin.setValue(2.5)

        dialog.start_button.click()

        assert dialog.result() == QDialog.DialogCode.Accepted
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] is stream
        assert args[1] == pytest.approx(1000.0)
        assert args[2] == pytest.approx(5000.0)
        assert args[3] == 3
        assert args[4] == 3
        assert args[5] == tmp_path / DEFAULT_CONVERSION_GAIN_FILENAME
        # Stream must be stopped again once the (mocked) sweep returns.
        assert not stream.is_running

    def test_camera_error_shows_camera_dialog(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)

        def _raise_camera_error(*a, **k):
            raise CameraConnectionError("no device found")

        monkeypatch.setattr(calibration_dialogs_module, "build_camera_stream", _raise_camera_error)
        camera_error_calls = _record_calls(monkeypatch, calibration_dialogs_module, "show_camera_error_dialog")

        dialog = ConversionGainDialog()
        qtbot.addWidget(dialog)
        self._fill_required_fields(dialog)

        dialog.start_button.click()

        assert dialog.result() != QDialog.DialogCode.Accepted
        assert len(camera_error_calls) == 1

    def test_invalid_conversion_gain_shows_calibration_dialog(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        monkeypatch.setattr(
            calibration_dialogs_module, "build_camera_stream",
            lambda *a, **k: _synthetic_camera_stream(),
        )

        def _raise_invalid(*a, **k):
            raise InvalidConversionGainError("non-positive slope")

        monkeypatch.setattr(calibration_dialogs_module, "run_conversion_gain_calibration", _raise_invalid)
        calibration_error_calls = _record_calls(
            monkeypatch, calibration_dialogs_module, "show_calibration_error_dialog"
        )

        dialog = ConversionGainDialog()
        qtbot.addWidget(dialog)
        self._fill_required_fields(dialog)

        dialog.start_button.click()

        assert dialog.result() != QDialog.DialogCode.Accepted
        assert len(calibration_error_calls) == 1


# ---------------------------------------------------------------------------
# SpatialCalibrationDialog -- real wiring
# ---------------------------------------------------------------------------

class TestSpatialCalibrationDialogWiring:

    def test_save_button_persists_scale_factor_and_accepts(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        dialog = SpatialCalibrationDialog(DEFAULT_SCALE_FACTOR)
        qtbot.addWidget(dialog)
        dialog.scale_factor_spin.setValue(1.75)

        dialog.save_button.click()

        assert dialog.result() == QDialog.DialogCode.Accepted
        path = tmp_path / DEFAULT_SCALE_FACTOR_FILENAME
        assert path.exists()
        loaded_calibration, loaded_record = load_scale_factor(path)
        assert loaded_calibration.scale_factor == pytest.approx(1.75)
        assert loaded_record.source == "manual"


# ---------------------------------------------------------------------------
# SpectralCalibrationDialog -- real wiring
# ---------------------------------------------------------------------------

class TestSpectralCalibrationDialogWiring:

    def test_capture_missing_sensor_calibration_shows_error_and_does_not_start(
        self, qtbot, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        calibration_error_calls = _record_calls(
            monkeypatch, calibration_dialogs_module, "show_calibration_error_dialog"
        )
        run_calls = _record_calls(monkeypatch, calibration_dialogs_module, "run_spectral_calibration")

        dialog = SpectralCalibrationDialog()
        qtbot.addWidget(dialog)

        dialog.start_button.click()

        assert dialog.result() != QDialog.DialogCode.Accepted
        assert len(calibration_error_calls) == 1
        title = calibration_error_calls[0][0][1]
        assert title == "Missing Sensor Calibration"
        assert run_calls == []

    def test_capture_mode_calls_run_spectral_calibration_with_loaded_sensor_calibration(
        self, qtbot, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        _save_sensor_artifacts(tmp_path)
        stream = _synthetic_camera_stream()
        monkeypatch.setattr(calibration_dialogs_module, "build_camera_stream", lambda *a, **k: stream)
        calls = _record_calls(monkeypatch, calibration_dialogs_module, "run_spectral_calibration")

        dialog = SpectralCalibrationDialog()
        qtbot.addWidget(dialog)
        dialog.n_frames_spin.setValue(4)

        dialog.start_button.click()

        assert dialog.result() == QDialog.DialogCode.Accepted
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] is stream
        assert args[1] == 4
        sensor_calibration = args[2]
        assert sensor_calibration.background_sigma == pytest.approx(1.0)
        assert args[3] == tmp_path / DEFAULT_SPECTRAL_FILENAME
        assert args[4] == tmp_path / DEFAULT_GEOMETRIC_TILT_FILENAME
        assert not stream.is_running

    def test_capture_mode_manual_exposure_threads_through_to_camera_stream(
        self, qtbot, monkeypatch, tmp_path
    ):
        '''
        Regression test: manual exposure_us must reach build_camera_stream()
        unchanged, not get silently discarded in favor of auto-exposure --
        see class docstring on SpectralCalibrationDialog for why this
        specifically matters (a lamp frame's actual exposure_us has to be
        able to match the loaded baseline's for check_settings_match() to
        pass downstream in run_spectral_calibration()).
        '''
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        _save_sensor_artifacts(tmp_path)
        stream = _synthetic_camera_stream()
        build_calls = []

        def _fake_build_camera_stream(*args, **kwargs):
            build_calls.append((args, kwargs))
            return stream

        monkeypatch.setattr(calibration_dialogs_module, "build_camera_stream", _fake_build_camera_stream)
        _record_calls(monkeypatch, calibration_dialogs_module, "run_spectral_calibration")

        dialog = SpectralCalibrationDialog()
        qtbot.addWidget(dialog)
        dialog.exposure_mode_combo.setCurrentText("Manual")
        dialog.exposure_us_spin.setValue(1000.0)
        dialog.gain_db_spin.setValue(5.0)

        dialog.start_button.click()

        assert dialog.result() == QDialog.DialogCode.Accepted
        assert len(build_calls) == 1
        args, kwargs = build_calls[0]
        assert args[0] == pytest.approx(5.0)
        assert kwargs["exposure_us"] == pytest.approx(1000.0)
        assert kwargs["auto_exposure"] is False

    def test_capture_mode_camera_error_shows_camera_dialog(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        _save_sensor_artifacts(tmp_path)

        def _raise_camera_error(*a, **k):
            raise CameraConnectionError("no device found")

        monkeypatch.setattr(calibration_dialogs_module, "build_camera_stream", _raise_camera_error)
        camera_error_calls = _record_calls(monkeypatch, calibration_dialogs_module, "show_camera_error_dialog")

        dialog = SpectralCalibrationDialog()
        qtbot.addWidget(dialog)

        dialog.start_button.click()

        assert dialog.result() != QDialog.DialogCode.Accepted
        assert len(camera_error_calls) == 1

    def test_capture_mode_line_matching_error_shows_calibration_dialog(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        _save_sensor_artifacts(tmp_path)
        monkeypatch.setattr(
            calibration_dialogs_module, "build_camera_stream",
            lambda *a, **k: _synthetic_camera_stream(),
        )

        def _raise_line_matching_error(*a, **k):
            raise LineMatchingError("too few peaks detected")

        monkeypatch.setattr(calibration_dialogs_module, "run_spectral_calibration", _raise_line_matching_error)
        calibration_error_calls = _record_calls(
            monkeypatch, calibration_dialogs_module, "show_calibration_error_dialog"
        )

        dialog = SpectralCalibrationDialog()
        qtbot.addWidget(dialog)

        dialog.start_button.click()

        assert dialog.result() != QDialog.DialogCode.Accepted
        assert len(calibration_error_calls) == 1

    def test_manual_mode_save_button_builds_and_saves_real_calibration(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        dialog = SpectralCalibrationDialog()
        qtbot.addWidget(dialog)
        dialog.manual_mode_radio.setChecked(True)
        dialog.manual_degree_selector.setCurrentIndex(0)   # degree 1 -> two rows
        (c0_value, c0_sigma), (c1_value, c1_sigma) = dialog._coefficient_rows
        c0_value.setValue(400.0)
        c0_sigma.setValue(1.0)
        c1_value.setValue(0.5)
        c1_sigma.setValue(0.01)

        dialog.save_button.click()

        assert dialog.result() == QDialog.DialogCode.Accepted
        path = tmp_path / DEFAULT_SPECTRAL_FILENAME
        assert path.exists()
        loaded = load_spectral_calibration(path)
        assert np.allclose(loaded.fit.coefficients, [400.0, 0.5])
        assert np.allclose(loaded.wavelength_nm(np.array([0.0])), [400.0])

    def test_manual_mode_invalid_coefficients_shows_calibration_dialog(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)

        def _raise_value_error(*a, **k):
            raise ValueError("coefficient_sigma must be strictly positive")

        monkeypatch.setattr(calibration_dialogs_module, "build_manual_spectral_calibration", _raise_value_error)
        calibration_error_calls = _record_calls(
            monkeypatch, calibration_dialogs_module, "show_calibration_error_dialog"
        )

        dialog = SpectralCalibrationDialog()
        qtbot.addWidget(dialog)
        dialog.manual_mode_radio.setChecked(True)

        dialog.save_button.click()

        assert dialog.result() != QDialog.DialogCode.Accepted
        assert len(calibration_error_calls) == 1
        assert not (tmp_path / DEFAULT_SPECTRAL_FILENAME).exists()


# ---------------------------------------------------------------------------
# CreatePage -- per-type completion tracking / Continue gating
# ---------------------------------------------------------------------------

class TestCreatePageCompletionTracking:

    def test_continue_button_starts_disabled(self, qtbot):
        create_page = CreatePage()
        qtbot.addWidget(create_page)
        assert not create_page.continue_button.isEnabled()

    def test_continue_button_enables_only_once_all_four_gated_types_done(self, qtbot):
        create_page = CreatePage()
        qtbot.addWidget(create_page)

        create_page._baseline_done = True
        create_page._update_continue_button()
        assert not create_page.continue_button.isEnabled()

        create_page._flat_field_done = True
        create_page._update_continue_button()
        assert not create_page.continue_button.isEnabled()

        create_page._conversion_gain_done = True
        create_page._update_continue_button()
        assert not create_page.continue_button.isEnabled()

        create_page._spectral_done = True
        create_page._update_continue_button()
        assert create_page.continue_button.isEnabled()

    def test_baseline_dialog_accepted_via_card_marks_done_and_updates_continue(
        self, qtbot, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        monkeypatch.setattr(
            calibration_dialogs_module, "build_camera_stream",
            lambda *a, **k: _synthetic_camera_stream(),
        )

        def _fake_exec(self):
            self.n_frames_spin.setValue(3)
            self.start_button.click()
            return self.result()

        monkeypatch.setattr(BaselineDialog, "exec", _fake_exec)

        create_page = CreatePage()
        qtbot.addWidget(create_page)
        create_page.baseline_card.action_button.click()

        assert create_page._baseline_done is True
        assert not create_page.continue_button.isEnabled()   # other three still missing

    def test_baseline_dialog_rejected_via_card_does_not_mark_done(self, qtbot, monkeypatch):
        monkeypatch.setattr(BaselineDialog, "exec", lambda self: QDialog.DialogCode.Rejected)

        create_page = CreatePage()
        qtbot.addWidget(create_page)
        create_page.baseline_card.action_button.click()

        assert create_page._baseline_done is False
        assert not create_page.continue_button.isEnabled()

    def test_spatial_dialog_acceptance_does_not_gate_continue(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(calibration_dialogs_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        monkeypatch.setattr(SpatialCalibrationDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

        create_page = CreatePage()
        qtbot.addWidget(create_page)
        create_page._baseline_done = True
        create_page._flat_field_done = True
        create_page._conversion_gain_done = True
        create_page._spectral_done = True
        create_page._update_continue_button()
        assert create_page.continue_button.isEnabled()

        create_page.spatial_card.action_button.click()

        # Still enabled -- spatial isn't part of the gate either way.
        assert create_page.continue_button.isEnabled()

    def test_continue_button_click_emits_continue_requested(self, qtbot):
        create_page = CreatePage()
        qtbot.addWidget(create_page)
        create_page._baseline_done = True
        create_page._flat_field_done = True
        create_page._conversion_gain_done = True
        create_page._spectral_done = True
        create_page._update_continue_button()

        with qtbot.waitSignal(create_page.continue_requested, timeout=1000):
            create_page.continue_button.click()


# ---------------------------------------------------------------------------
# CalibrationScreen -- CreatePage's Continue wired to
# _attempt_load_existing_calibrations()
# ---------------------------------------------------------------------------

class TestCalibrationScreenContinueWiring:

    def _patch_successful_calibration_load(self, monkeypatch):
        # Must cover every load_*() call _attempt_load_existing_
        # calibrations() makes -- including load_spectral_calibration()/
        # load_geometric_tilt(), both hard/soft-required respectively (see
        # module-level _patch_successful_calibration_load() above) -- or
        # this test hits a real, unmocked show_calibration_error_dialog()
        # call, which blocks forever on QDialog.exec() in an offscreen
        # run (see module docstring).
        record = _record()
        baseline_result = BaselineResult(baseline=np.full(CANONICAL_SHAPE, 10.0), background_sigma=1.0)
        flat_field = np.ones(CANONICAL_SHAPE)
        bad_pixel_mask = np.zeros(CANONICAL_SHAPE, dtype=bool)

        conversion_gain_result = SimpleNamespace(gain_e_per_adu=2.2)
        position_calibration = ScaleFactorPositionCalibration()
        wavelength_axis = _FakeWavelengthAxis()

        monkeypatch.setattr(calibration_screen_module, "load_baseline", lambda path: (baseline_result, record))
        monkeypatch.setattr(calibration_screen_module, "load_flat_field", lambda path: (flat_field, record))
        monkeypatch.setattr(
            calibration_screen_module, "load_bad_pixel_map", lambda path: (bad_pixel_mask, record)
        )
        monkeypatch.setattr(
            calibration_screen_module, "load_conversion_gain",
            lambda path: (conversion_gain_result, record),
        )
        monkeypatch.setattr(
            calibration_screen_module, "load_spectral_calibration", lambda path: wavelength_axis
        )
        monkeypatch.setattr(
            calibration_screen_module, "load_scale_factor",
            lambda path: (position_calibration, object()),
        )

        def _raise_missing_geometric_tilt(path):
            raise FileNotFoundError(path)

        monkeypatch.setattr(
            calibration_screen_module, "load_geometric_tilt", _raise_missing_geometric_tilt
        )

    def test_create_page_continue_triggers_load_existing_calibrations_and_emits_ready(
        self, qtbot, monkeypatch
    ):
        self._patch_successful_calibration_load(monkeypatch)
        screen = CalibrationScreen()
        qtbot.addWidget(screen)

        received = []
        screen.calibration_ready.connect(received.append)

        screen.create_page.continue_requested.emit()

        assert len(received) == 1
        assert received[0] is screen.get_calibration_bundle()

    def test_create_page_continue_missing_artifact_shows_error_and_blocks_ready(
        self, qtbot, monkeypatch
    ):
        def _raise_missing(path):
            raise FileNotFoundError(path)

        for name in ("load_baseline", "load_flat_field", "load_bad_pixel_map", "load_conversion_gain"):
            monkeypatch.setattr(calibration_screen_module, name, _raise_missing)

        error_calls = []
        monkeypatch.setattr(
            calibration_screen_module, "show_calibration_error_dialog",
            lambda parent, title, message: error_calls.append((title, message)),
        )

        screen = CalibrationScreen()
        qtbot.addWidget(screen)
        received = []
        screen.calibration_ready.connect(received.append)

        screen.create_page.continue_requested.emit()

        assert received == []
        assert screen.get_calibration_bundle() is None
