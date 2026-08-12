'''
A single, realistic CalibrationBundle builder, shared by tests/test_live_
view.py and tests/test_extended_measurement.py (and any future gui/ test
file that needs one) -- built via real build_*() calls, not hand-typed
arrays, so wiring tests that exercise real run_preprocessing()/
analyze_shot() behavior have one consistent, genuinely-calibrated state to
test against instead of each file inventing its own. This is the
"realistic" counterpart to scripts/demo_app.py's
build_placeholder_calibration_bundle(), which deliberately stays
hand-built/fake-shaped for that script's simpler navigation-only purpose.

This module is READ-ONLY for anyone wiring live_view.py or
extended_measurement.py -- import build_realistic_calibration_bundle()
from it, do not edit it. Keeping it a separate, stable file (rather than
inlining a copy into each test file, this codebase's usual per-file-
duplication convention for SMALL helpers -- see
tests/test_live_view.py's/tests/test_extended_measurement.py's own
_calibration_set()/_camera_stream()) is deliberate here specifically
because this builder is large enough that duplicating it would be a real
maintenance burden, and because two independent wiring efforts (live view,
extended measurement) both need the exact same realistic state to be
directly comparable.

Two artifacts can't be built the "fully real" way from SyntheticBackend
data, and are handled differently, each documented at its own call site
below:
  - wavelength_axis is built via build_manual_spectral_calibration()
    (the manual-entry path, not lamp capture) -- a real function, just
    not the lamp-capture one, since SyntheticBackend's frames are a smooth
    Gaussian beam profile with no discrete line peaks for match_lines()
    to find (see tests/test_calibration.py's
    test_propagates_line_matching_error_for_non_line_like_signal).
  - geometric_tilt is constructed directly as a trivial (all-zero shift)
    but fully valid GeometricTiltResult, for the same reason
    build_geometric_tilt() can't run against SyntheticBackend data
    either -- it also needs discrete line peaks.
'''

# Imports

import time

import numpy as np

from pipeline.acquisition import CANONICAL_SHAPE
from pipeline.calibration.sensor import (
    build_bad_pixel_map,
    build_baseline,
    build_conversion_gain,
    build_flat_field,
)
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.calibration.spectral import build_manual_spectral_calibration
from pipeline.calibration.spectral.geometric_tilt import GeometricTiltResult
from pipeline.acquisition import FrameData
from pipeline.analysis import SensorNoiseModel
from pipeline.gui.calibration_screen import CalibrationBundle
from pipeline.preprocessing import CalibrationSet

# Constants

FIXTURE_EXPOSURE_US = 2000.0
FIXTURE_GAIN_DB = 0.0

# Classes

# Functions

def _frame(value: float, exposure_us: float = FIXTURE_EXPOSURE_US, frame_id: int = 0) -> FrameData:
    '''A uniform-value FrameData at the fixture's default settings.'''
    image = np.full(CANONICAL_SHAPE, value, dtype=np.uint8)
    return FrameData(
        image=image, frame_id=frame_id, timestamp=time.monotonic(),
        exposure_us=exposure_us, gain_db=FIXTURE_GAIN_DB,
    )


def _noisy_frame(
    value: float, rng: np.random.Generator,
    exposure_us: float = FIXTURE_EXPOSURE_US, frame_id: int = 0,
) -> FrameData:
    '''
    A FrameData at the given mean value with small seeded per-pixel noise --
    unlike _frame(), which is bit-for-bit uniform and so produces a sample
    standard deviation of exactly zero across a frame stack (every pixel has
    an identical value in every frame). build_baseline()'s background_sigma
    is that per-pixel sample std, and
    preprocessing/steps/signal_threshold.py requires it to be strictly
    positive -- so the dark/illuminated source frames need this instead of
    _frame() for the resulting CalibrationSet to be usable by
    run_preprocessing().
    '''
    noise = rng.normal(loc=0.0, scale=0.6, size=CANONICAL_SHAPE)
    image = np.clip(np.round(value + noise), 0, 255).astype(np.uint8)
    return FrameData(
        image=image, frame_id=frame_id, timestamp=time.monotonic(),
        exposure_us=exposure_us, gain_db=FIXTURE_GAIN_DB,
    )


def _conversion_gain_frames_by_exposure() -> dict[float, list[FrameData]]:
    '''
    A frames_by_exposure sweep with an EXACT, known variance_ADU =
    mean_ADU/2.0 + 1.0 relationship (gain=2.0 e-/ADU, read-noise
    variance=1.0 ADU^2) -- same construction as
    tests/test_calibration.py's _ptc_frames_by_exposure(), duplicated here
    rather than imported (that file's helper is private/test-calibration-
    specific, and this fixture needs no other part of that module).
    '''
    exposure_levels = [1000.0, 2000.0, 3000.0, 4000.0]
    ds = [2, 3, 4, 5]
    frames_by_exposure = {}
    for exposure_us, d in zip(exposure_levels, ds):
        mean = 4 * d * d - 2
        frames_by_exposure[exposure_us] = [
            _frame(mean - d, exposure_us=exposure_us, frame_id=0),
            _frame(mean + d, exposure_us=exposure_us, frame_id=1),
        ]
    return frames_by_exposure


def build_realistic_calibration_bundle() -> CalibrationBundle:

    '''
    Builds a complete, realistic CalibrationBundle: baseline, flat field,
    bad-pixel map, and conversion gain via their real build_*() calls
    against synthetic (but not hand-faked) frame data; wavelength_axis and
    geometric_tilt via the two documented exceptions in this module's own
    docstring. Every field is populated (never None) so callers can
    exercise the fully-wired code path without extra None-handling.

    Returns
    -------
    CalibrationBundle
    '''

    rng = np.random.default_rng(seed=0)
    illuminated = [_noisy_frame(150.0, rng, frame_id=i) for i in range(3)]
    dark = [_noisy_frame(10.0, rng, frame_id=i) for i in range(3)]

    baseline_result, baseline_record = build_baseline(dark)
    flat_field, flat_field_record = build_flat_field(illuminated, dark)
    bad_pixel_mask, _ = build_bad_pixel_map(flat_field, flat_field_record)
    conversion_gain_result, conversion_gain_record = build_conversion_gain(
        _conversion_gain_frames_by_exposure()
    )

    # Trivial (all-zero shift) but fully valid GeometricTiltResult -- see
    # module docstring for why this is constructed directly rather than
    # via build_geometric_tilt().
    n_rows = CANONICAL_SHAPE[0]
    geometric_tilt = GeometricTiltResult(
        row_shift=np.zeros(n_rows, dtype=np.float64),
        reference_row=n_rows // 2,
        residual_slope_columns=np.array([500.0, 1400.0], dtype=np.float64),
        residual_slope_values=np.array([0.0, 0.0], dtype=np.float64),
        record=CalibrationRecord(
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            timestamp=time.time(), source_frame_count=1,
        ),
    )

    calibration_set = CalibrationSet(
        baseline=baseline_result.baseline,
        baseline_record=baseline_record,
        flat_field=flat_field,
        flat_field_record=flat_field_record,
        bad_pixel_mask=bad_pixel_mask,
        background_sigma=baseline_result.background_sigma,
        geometric_tilt=geometric_tilt,
    )
    noise_model = SensorNoiseModel(
        gain_e_per_adu=conversion_gain_result.gain_e_per_adu,
        background_sigma=baseline_result.background_sigma,
    )
    wavelength_axis = build_manual_spectral_calibration(
        coefficients=np.array([500.0, 0.05]),
        coefficient_sigma=np.array([0.5, 0.001]),
        record=CalibrationRecord(
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            timestamp=time.time(), source_frame_count=1,
        ),
    )

    return CalibrationBundle(
        calibration_set=calibration_set,
        noise_model=noise_model,
        position_calibration=ScaleFactorPositionCalibration(),
        wavelength_axis=wavelength_axis,
        conversion_gain_record=conversion_gain_record,
    )


__all__ = ["build_realistic_calibration_bundle", "FIXTURE_EXPOSURE_US", "FIXTURE_GAIN_DB"]
