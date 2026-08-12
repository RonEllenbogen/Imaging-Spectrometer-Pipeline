'''
End-to-end offscreen smoke test: drives MainWindow through Calibration ->
Live View -> Extended Measurement exactly the way a real user would --
real calibration artifacts saved to disk (not an in-memory
CalibrationBundle handed to a widget directly, the way every other gui/
test file's fixtures work), loaded via WelcomePage's real
"Load Existing Calibrations" flow, feeding a real (SyntheticBackend-
driven) CameraStream that MainWindow itself starts, through to both
downstream screens' real analysis code.

This is deliberately NOT a duplicate of tests/test_calibration_screen.py/
test_live_view.py/test_extended_measurement.py's own real-wiring tests --
those each verify one screen's internals in isolation (constructing the
widget directly from a hand-built CalibrationSet/CalibrationBundle, in
test_live_view.py's/test_extended_measurement.py's case with a
CameraStream the test itself starts). This file instead exists to catch
cross-screen integration bugs that isolated testing can't -- and it found
one: MainWindow._on_calibration_ready() built the shared CameraStream via
build_camera_stream() but never called .start() on it (build_camera_stream()
explicitly documents "Does not start or stop the stream -- callers own
stream lifecycle"), so LiveViewWidget's real polling loop would never see
a frame and ExtendedMeasurementScreen's collect_n_frames() would hang
forever -- neither Agent B's nor Agent C's own tests caught this, since
both start their own CameraStream directly rather than going through
MainWindow. Fixed in app.py alongside this test (see its module
docstring).

PySide6/pyqtgraph/pytest-qt are local-only dependencies (see CLAUDE.md),
so this whole module is skipped, not failed, wherever they aren't
installed -- same pattern as every other gui/ test file.
'''

# Imports

import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 is a local-only GUI dependency")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is a local-only GUI dependency")
pytest.importorskip("pytestqt", reason="pytest-qt is a local-only GUI dependency")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pipeline.acquisition import CameraStream, CANONICAL_SHAPE, SyntheticBackend  # noqa: E402
from pipeline.calibration.sensor import (  # noqa: E402
    build_bad_pixel_map,
    build_baseline,
    build_conversion_gain,
    build_flat_field,
    save_bad_pixel_map,
    save_baseline,
    save_conversion_gain,
    save_flat_field,
)
from pipeline.calibration.spatial import (  # noqa: E402
    DEFAULT_SCALE_FACTOR,
    ScaleFactorPositionCalibration,
    save_scale_factor,
)
from pipeline.calibration.spectral import build_manual_spectral_calibration  # noqa: E402
from pipeline.calibration.spectral.io import save_spectral_calibration  # noqa: E402
from pipeline.cli.calibration import (  # noqa: E402
    DEFAULT_BAD_PIXEL_MAP_FILENAME,
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_CONVERSION_GAIN_FILENAME,
    DEFAULT_FLAT_FIELD_FILENAME,
    DEFAULT_SCALE_FACTOR_FILENAME,
    DEFAULT_SPECTRAL_FILENAME,
)
from pipeline.acquisition import FrameData  # noqa: E402

import pipeline.gui.app as app_module  # noqa: E402
import pipeline.gui.calibration_screen as calibration_screen_module  # noqa: E402
import pipeline.gui.extended_measurement as extended_measurement_module  # noqa: E402
import pipeline.gui.live_view as live_view_module  # noqa: E402
from pipeline.gui.app import MainWindow  # noqa: E402
from pipeline.gui.extended_measurement import ExtendedMeasurementScreen  # noqa: E402

# Constants

FIXTURE_EXPOSURE_US = 2000.0
FIXTURE_GAIN_DB = 0.0

# Functions

def _frame(
    value: float, rng: np.random.Generator,
    frame_id: int = 0, exposure_us: float = FIXTURE_EXPOSURE_US,
) -> FrameData:
    '''A FrameData with small per-pixel noise -- see
    tests/gui_fixture_helpers.py's _noisy_frame() docstring for why
    bit-for-bit-uniform frames aren't usable here (background_sigma would
    come out exactly zero, which run_preprocessing() rejects).'''
    noise = rng.normal(loc=0.0, scale=0.6, size=CANONICAL_SHAPE)
    image = np.clip(np.round(value + noise), 0, 255).astype(np.uint8)
    return FrameData(
        image=image, frame_id=frame_id, timestamp=time.monotonic(),
        exposure_us=exposure_us, gain_db=FIXTURE_GAIN_DB,
    )


def _save_real_calibration_artifacts(artifact_dir) -> None:
    '''
    Builds and saves a complete set of real calibration artifacts to
    artifact_dir, at the exact DEFAULT_*_FILENAME paths WelcomePage's
    "Load Existing Calibrations" flow reads from -- baseline/flat-field/
    bad-pixel-map/conversion-gain via their real build_*() calls (mirrors
    tests/gui_fixture_helpers.py's build_realistic_calibration_bundle(),
    but SAVED to disk rather than kept in memory, since this test drives
    the real disk-load path rather than constructing a screen directly).
    Spectral is built via the manual-entry path (same reason
    gui_fixture_helpers.py uses it -- SyntheticBackend has no discrete
    line peaks for lamp-capture line matching). Geometric tilt is
    deliberately left unsaved -- it's soft-optional (see
    CalibrationBundle's own docstring), so this also exercises that path.
    '''
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=0)

    dark = [_frame(10.0, rng, frame_id=i) for i in range(3)]
    illuminated = [_frame(150.0, rng, frame_id=i) for i in range(3)]

    baseline_result, baseline_record = build_baseline(dark)
    save_baseline(artifact_dir / DEFAULT_BASELINE_FILENAME, baseline_result, baseline_record)

    flat_field, flat_field_record = build_flat_field(illuminated, dark)
    save_flat_field(artifact_dir / DEFAULT_FLAT_FIELD_FILENAME, flat_field, flat_field_record)

    bad_pixel_mask, bad_pixel_record = build_bad_pixel_map(flat_field, flat_field_record)
    save_bad_pixel_map(artifact_dir / DEFAULT_BAD_PIXEL_MAP_FILENAME, bad_pixel_mask, bad_pixel_record)

    exposure_levels = [1000.0, 2000.0, 3000.0, 4000.0]
    ds = [2, 3, 4, 5]
    frames_by_exposure = {}
    for exposure_us, d in zip(exposure_levels, ds):
        mean = 4 * d * d - 2
        frames_by_exposure[exposure_us] = [
            _frame(mean - d, rng, frame_id=0, exposure_us=exposure_us),
            _frame(mean + d, rng, frame_id=1, exposure_us=exposure_us),
        ]
    conversion_gain_result, conversion_gain_record = build_conversion_gain(frames_by_exposure)
    save_conversion_gain(
        artifact_dir / DEFAULT_CONVERSION_GAIN_FILENAME, conversion_gain_result, conversion_gain_record
    )

    wavelength_axis = build_manual_spectral_calibration(
        coefficients=np.array([500.0, 0.05]),
        coefficient_sigma=np.array([0.5, 0.001]),
        record=baseline_record,
    )
    save_spectral_calibration(artifact_dir / DEFAULT_SPECTRAL_FILENAME, wavelength_axis)

    save_scale_factor(
        artifact_dir / DEFAULT_SCALE_FACTOR_FILENAME,
        ScaleFactorPositionCalibration(scale_factor=DEFAULT_SCALE_FACTOR),
        source="default",
    )


def _fake_build_camera_stream(gain_db, *, exposure_us=None, auto_exposure=False):
    '''SyntheticBackend stand-in for cli.calibration.build_camera_stream()
    -- see tests/test_gui.py's own copy for why this is needed (the real
    one constructs a PylonBackend, which can block indefinitely without a
    physical camera attached).'''
    return CameraStream(
        exposure_us=exposure_us if exposure_us is not None else FIXTURE_EXPOSURE_US,
        gain_db=gain_db, pixel_format="Mono8", timeout_ms=5000,
        backend=SyntheticBackend(seed=1),
    )


# ---------------------------------------------------------------------------
# Full navigation with real artifacts, real analysis, real acquisition
# ---------------------------------------------------------------------------

class TestFullApplicationFlow:

    def test_calibration_to_live_data_to_measurement_result(self, qtbot, monkeypatch, tmp_path):
        _save_real_calibration_artifacts(tmp_path)
        monkeypatch.setattr(calibration_screen_module, "DEFAULT_ARTIFACT_DIR", tmp_path)
        monkeypatch.setattr(app_module, "build_camera_stream", _fake_build_camera_stream)

        # Any real (unmocked) modal QDialog/QMessageBox.exec() blocks
        # forever offscreen -- see every other gui/ test file's module
        # docstring for the lldb-diagnosed reason. None of these are
        # expected to fire on this success path; patched to fail loudly
        # (not hang) if one unexpectedly does.
        def _fail_if_called(*args, **kwargs):
            raise AssertionError(f"unexpected error dialog: {args!r} {kwargs!r}")

        monkeypatch.setattr(calibration_screen_module, "show_calibration_error_dialog", _fail_if_called)
        monkeypatch.setattr(live_view_module.QMessageBox, "warning", lambda *a, **k: None)
        monkeypatch.setattr(extended_measurement_module.QMessageBox, "warning", lambda *a, **k: None)

        window = MainWindow()
        qtbot.addWidget(window)

        # -- Calibration: real disk load ---------------------------------
        window._calibration_screen.welcome_page.load_requested.emit()

        assert window._stack.currentWidget() is window._live_view
        bundle = window._bundle
        assert bundle.wavelength_axis is not None
        assert bundle.calibration_set.geometric_tilt is None   # never saved above

        # -- Live view: the shared CameraStream must actually be running,
        # confirming this test's own regression fix (see module docstring)
        assert window._camera_stream.is_running

        live_view = window._live_view
        live_view._update_timer.setInterval(15)   # speed up polling for the test

        initial_scatter_y = live_view._scatter.getData()[1]
        assert initial_scatter_y is not None and len(initial_scatter_y) > 0   # placeholder data

        def _real_data_displayed() -> bool:
            current_y = live_view._scatter.getData()[1]
            if current_y is None or len(current_y) != len(initial_scatter_y):
                return len(current_y) > 0 if current_y is not None else False
            return not np.allclose(current_y, initial_scatter_y)

        qtbot.waitUntil(_real_data_displayed, timeout=5000)

        # -- Extended measurement: real N-frame acquisition + combination
        window._live_view.extended_measurement_requested.emit()
        assert isinstance(window._extended_measurement, ExtendedMeasurementScreen)
        assert window._stack.currentWidget() is window._extended_measurement

        extended_measurement = window._extended_measurement
        extended_measurement._n_shots_spin.setValue(3)
        assert extended_measurement._n_shots_label.text() == "--"

        extended_measurement._run_button.click()

        assert extended_measurement._n_shots_label.text() == "3"
        assert extended_measurement._spatial_dispersion_label.text() != "--"
        assert extended_measurement._reduced_chi_squared_label.text() != "--"

        window.close()
        assert not window._camera_stream.is_running
