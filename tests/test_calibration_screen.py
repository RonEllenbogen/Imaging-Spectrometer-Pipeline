"""
Test suite for calibration_dialogs.py's/calibration_screen.py's real
wiring -- each dialog's Start/Continue/Save path calling the actual
calibration/sensor/, calibration/spatial/, calibration/spectral/
functions, error handling routing CameraError to
show_camera_error_dialog() and calibration-specific failures to
show_calibration_error_dialog() (leaving the originating dialog open),
and CreatePage's per-type completion tracking gating "Continue to Main
Window" through to CalibrationScreen._attempt_load_existing_calibrations().

Structural/layout smoke tests for these same dialogs already live in
tests/test_gui.py -- this file only covers the real acquire/build/save
calls layered on top, added once calibration_dialogs.py/calibration_
screen.py moved past their Phase-1 visual skeleton.

Every test that exercises a code path which *could* open a real
QMessageBox (show_camera_error_dialog()/show_calibration_error_dialog())
mocks it -- a real QMessageBox.exec()/QDialog.exec() call in an offscreen
test blocks forever waiting for a click that will never come. Where a
dialog's own accept path is exercised end-to-end, it is driven by clicking
its buttons directly (never by calling dialog.exec(), which would itself
block); CreatePage-level tests that need to go through a dialog's real
exec() call instead monkeypatch that one dialog class's exec() to click
its own button and return the resulting result code, without ever
starting a real nested Qt event loop.

PySide6/pyqtgraph/pytest-qt are local-only dependencies (see CLAUDE.md),
so this whole module is skipped, not failed, wherever they aren't
installed -- same pattern as tests/test_gui.py.
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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QDialog  # noqa: E402

from pipeline.acquisition import (  # noqa: E402
    CameraStream,
    CameraConnectionError,
    SyntheticBackend,
    CANONICAL_SHAPE,
)
from pipeline.calibration.exceptions import (  # noqa: E402
    InvalidConversionGainError,
    InvalidFlatFieldError,
    LineMatchingError,
    SettingsMismatchError,
)
from pipeline.calibration.sensor import (  # noqa: E402
    BaselineResult,
    save_baseline,
    save_bad_pixel_map,
    save_flat_field,
)
from pipeline.calibration.shared import CalibrationRecord  # noqa: E402
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
)
from pipeline.gui.calibration_screen import CalibrationScreen, CreatePage  # noqa: E402

# Constants

FIXTURE_GAIN_DB = 0.0

# Functions


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
    '''Writes minimal, valid baseline/flat-field/bad-pixel-map artifacts
    to artifact_dir (at the same DEFAULT_*_FILENAME paths calibration_
    dialogs.py's dialogs read from/write to) -- enough for
    SpectralCalibrationDialog's capture-mode _load_sensor_calibration()
    to succeed without needing a full real flat-field capture session.'''
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
# BaselineDialog
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
# FlatFieldDialog
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
# ConversionGainDialog
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
# SpatialCalibrationDialog
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
# SpectralCalibrationDialog
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
        record = _record()
        baseline_result = BaselineResult(baseline=np.full(CANONICAL_SHAPE, 10.0), background_sigma=1.0)
        flat_field = np.ones(CANONICAL_SHAPE)
        bad_pixel_mask = np.zeros(CANONICAL_SHAPE, dtype=bool)

        from types import SimpleNamespace
        conversion_gain_result = SimpleNamespace(gain_e_per_adu=2.2)
        position_calibration = ScaleFactorPositionCalibration()

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
            calibration_screen_module, "load_scale_factor",
            lambda path: (position_calibration, object()),
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
        assert len(error_calls) == 1
