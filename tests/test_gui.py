'''
Test suite for app.py's MainWindow -- the cross-cutting screen-navigation
wiring that ties calibration_screen.py, live_view.py, and
extended_measurement.py together, so it doesn't belong in any one of
those screens' own dedicated test file (see
tests/test_calibration_screen.py's module docstring for why this repo now
has one test file per gui/ screen instead of one big tests/test_gui.py).

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

from pipeline.acquisition import CameraStream, SyntheticBackend, CANONICAL_SHAPE  # noqa: E402
from pipeline.calibration.shared import CalibrationRecord  # noqa: E402
from pipeline.calibration.spatial import ScaleFactorPositionCalibration  # noqa: E402
import pipeline.gui.calibration_screen as calibration_screen_module  # noqa: E402
from pipeline.gui.app import MainWindow  # noqa: E402
from pipeline.gui.extended_measurement import ExtendedMeasurementScreen  # noqa: E402

# Constants

FIXTURE_EXPOSURE_US = 2000.0
FIXTURE_GAIN_DB = 0.0

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

def _patch_successful_calibration_load(monkeypatch) -> tuple[SimpleNamespace, SimpleNamespace, ScaleFactorPositionCalibration]:
    '''
    Patches calibration_screen's load_baseline()/load_flat_field()/
    load_bad_pixel_map()/load_conversion_gain()/load_spectral_calibration()/
    load_scale_factor()/load_geometric_tilt() calls to all succeed (or, for
    geometric tilt, to cleanly report "not built yet" -- see
    CalibrationBundle's own docstring for why that's a valid outcome, not
    a failure) with synthetic data. Returns (baseline_result,
    conversion_gain_result, position_calibration) so callers can assert
    the returned CalibrationBundle was actually built from these values.
    Mirrors tests/test_calibration_screen.py's own copy of this helper --
    kept as a separate local copy rather than a shared import, matching
    this codebase's existing per-file test-helper convention (e.g.
    tests/test_calibration.py and tests/test_preprocessing.py each define
    their own local _frame()/_uniform() rather than sharing one module).
    Every load_*() call calibration_screen.py makes must be covered here --
    missing even one turns "succeeds" into a real FileNotFoundError deep
    inside a Qt signal handler, which surfaces as a real (unmocked)
    show_calibration_error_dialog() call and hangs the whole test on
    QDialog.exec() rather than failing cleanly (this happened for real
    while wiring the geometric-tilt/spectral load calls in -- see
    TestMainWindowNavigation's own defensive show_calibration_error_dialog
    patch below).
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
        calibration_screen_module, "load_spectral_calibration", lambda path: _FakeWavelengthAxis()
    )
    monkeypatch.setattr(
        calibration_screen_module,
        "load_scale_factor",
        lambda path: (position_calibration, object()),
    )

    def _raise_missing_geometric_tilt(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(
        calibration_screen_module, "load_geometric_tilt", _raise_missing_geometric_tilt
    )
    return baseline_result, conversion_gain_result, position_calibration


# ---------------------------------------------------------------------------
# app.py -- MainWindow navigation (pytest-qt, offscreen)
# ---------------------------------------------------------------------------

def _fake_build_camera_stream(gain_db, *, exposure_us=None, auto_exposure=False):
    '''
    A SyntheticBackend-driven CameraStream standing in for
    cli.calibration.build_camera_stream() -- the real one constructs a
    PylonBackend (see its own docstring: "CameraStream configured for the
    real PylonBackend"), which without a physical camera connected can
    block for an unpredictable, sometimes very long time on device
    enumeration. app.py's MainWindow._on_calibration_ready() calls the
    real one unconditionally, so any test exercising that real code path
    -- like this one -- needs this substitute the same way every other
    camera-touching path in this test suite avoids real hardware.
    '''
    return CameraStream(
        exposure_us=exposure_us if exposure_us is not None else FIXTURE_EXPOSURE_US,
        gain_db=gain_db, pixel_format="Mono8", timeout_ms=5000,
        backend=SyntheticBackend(seed=0),
    )


class TestMainWindowNavigation:

    def test_calibration_to_live_view_to_extended_measurement_and_back(
        self, qtbot, monkeypatch
    ):
        _patch_successful_calibration_load(monkeypatch)
        monkeypatch.setattr("pipeline.gui.app.build_camera_stream", _fake_build_camera_stream)
        # Any real (unmocked) modal QDialog/QMessageBox.exec() blocks
        # forever waiting for user interaction that will never come in an
        # offscreen/automated run -- diagnosed via lldb backtrace showing
        # the process parked in QDialog::exec(). None of these are
        # expected to actually fire on the success path this test
        # exercises; patched to raise loudly (not hang) if one does.
        def _fail_if_called(*args, **kwargs):
            raise AssertionError(f"unexpected error dialog: {args!r} {kwargs!r}")

        monkeypatch.setattr(
            "pipeline.gui.live_view.QMessageBox.warning", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            "pipeline.gui.extended_measurement.QMessageBox.warning", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(calibration_screen_module, "show_calibration_error_dialog", _fail_if_called)
        window = MainWindow()
        qtbot.addWidget(window)

        window._calibration_screen.welcome_page.load_requested.emit()

        assert window._stack.currentWidget() is window._live_view
        assert window._live_view is not None

        live_view = window._live_view

        window._live_view.extended_measurement_requested.emit()

        assert isinstance(window._extended_measurement, ExtendedMeasurementScreen)
        assert window._stack.currentWidget() is window._extended_measurement

        extended_measurement = window._extended_measurement

        window._extended_measurement.back_requested.emit()

        assert window._stack.currentWidget() is window._live_view
        assert window._live_view is live_view

        window._live_view.extended_measurement_requested.emit()

        assert window._extended_measurement is extended_measurement
        assert window._stack.currentWidget() is window._extended_measurement

    def test_back_to_calibration_stops_camera_and_rebuilds_fresh_live_view(
        self, qtbot, monkeypatch
    ):
        _patch_successful_calibration_load(monkeypatch)
        monkeypatch.setattr("pipeline.gui.app.build_camera_stream", _fake_build_camera_stream)
        monkeypatch.setattr(
            "pipeline.gui.live_view.QMessageBox.warning", lambda *args, **kwargs: None
        )
        window = MainWindow()
        qtbot.addWidget(window)

        window._calibration_screen.welcome_page.load_requested.emit()
        assert window._stack.currentWidget() is window._live_view

        first_live_view = window._live_view
        first_camera_stream = window._camera_stream
        assert first_camera_stream.is_running is True

        window._live_view.extended_measurement_requested.emit()
        first_extended_measurement = window._extended_measurement
        assert window._stack.currentWidget() is first_extended_measurement

        window._live_view.back_to_calibration_requested.emit()

        # Stream stopped (freed for CalibrationScreen's own dialogs) and
        # both downstream screens fully torn down, not just hidden -- see
        # _on_back_to_calibration_requested()'s own docstring for why.
        assert first_camera_stream.is_running is False
        assert window._live_view is None
        assert window._extended_measurement is None
        assert window._camera_stream is None
        assert window._stack.currentWidget() is window._calibration_screen
        assert window._stack.indexOf(first_live_view) == -1
        assert window._stack.indexOf(first_extended_measurement) == -1

        # Completing calibration again must build a genuinely fresh
        # LiveViewWidget/CameraStream, not resurrect or reuse the torn-down
        # ones -- this is the exact round-trip _on_calibration_ready()'s
        # own docstring says must work.
        window._calibration_screen.welcome_page.load_requested.emit()

        assert window._stack.currentWidget() is window._live_view
        assert window._live_view is not None
        assert window._live_view is not first_live_view
        assert window._camera_stream is not first_camera_stream
        assert window._camera_stream.is_running is True
