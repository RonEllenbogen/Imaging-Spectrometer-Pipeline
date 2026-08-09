"""
Test suite for the calibration package: calibration/sensor/, shared/, and
the parts of spatial/ and spectral/ that don't depend on an actual lamp
(spectral/line_matching.py's match_lines() is a NotImplementedError stub
until a reference lamp is chosen -- see its own module docstring). Every
test operates on synthetic FrameData/array data with known, injected
values -- no camera or real calibration data involved, following the same
"prove against known ground truth first" principle used throughout
acquisition/'s test suite.

Split out of test_preprocessing.py when sensor_calibration/ moved from
preprocessing/ to calibration/sensor/ -- this file covers build_*() (the
artifact-construction side); test_preprocessing.py keeps apply_*() (the
per-frame-correction side) plus the end-to-end pipeline test.
"""

# Imports

import time

import numpy as np
import pytest

from pipeline.acquisition import FrameData, CANONICAL_SHAPE, CANONICAL_DTYPE, CANONICAL_MAX_VALUE, CameraStream, SyntheticBackend

from pipeline.calibration.exceptions import (
    SettingsMismatchError, InvalidFlatFieldError, InvalidConversionGainError, InsufficientDataError,
)
from pipeline.calibration.shared import (
    save_artifact, load_artifact,
    CalibrationRecord, check_settings_match,
    EXPOSURE_MATCH_TOLERANCE_REL, GAIN_MATCH_TOLERANCE_ABS,
    PolynomialFitResult, PolynomialFitter, TotalLeastSquaresFit,
)
from pipeline.calibration.sensor import (
    build_baseline, save_baseline, load_baseline,
    build_flat_field, save_flat_field, load_flat_field,
    build_bad_pixel_map, save_bad_pixel_map, load_bad_pixel_map,
    ConversionGainRecord, ConversionGainResult,
    build_conversion_gain, save_conversion_gain, load_conversion_gain,
    check_saturation, SaturationCheckResult,
    run_baseline_calibration,
    capture_dark_frames, capture_illuminated_frames, finish_flat_field_calibration,
    run_conversion_gain_calibration,
)
from pipeline.calibration.spatial import (
    ScaleFactorPositionCalibration, DEFAULT_SCALE_FACTOR,
    ScaleFactorRecord, save_scale_factor, load_scale_factor,
)
from pipeline.calibration.spectral import (
    calibrate_spectral, WavelengthCalibrationResult,
    save_spectral_calibration, load_spectral_calibration,
    match_lines, run_spectral_calibration,
)
from pipeline.preprocessing import CalibrationSet

# Constants

# --- fixed configuration for every synthetic frame in this file ---
FIXTURE_EXPOSURE_US = 2000.0
FIXTURE_GAIN_DB = 0.0
FIXTURE_FRAME_ID = 0
FIXTURE_TIMESTAMP = 0.0


# Functions

def _frame(image: np.ndarray, frame_id=FIXTURE_FRAME_ID, timestamp=FIXTURE_TIMESTAMP,
           exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB) -> FrameData:
    '''Builds a valid FrameData from a raw array, casting to CANONICAL_DTYPE.'''
    return FrameData(
        image=image.astype(CANONICAL_DTYPE), frame_id=frame_id, timestamp=timestamp,
        exposure_us=exposure_us, gain_db=gain_db,
    )


def _uniform(value: float, dtype=CANONICAL_DTYPE) -> np.ndarray:
    '''A full-CANONICAL_SHAPE array filled with a single value.'''
    return np.full(CANONICAL_SHAPE, value, dtype=dtype)


def _record(exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            timestamp=None, source_frame_count=1) -> CalibrationRecord:
    '''Builds a CalibrationRecord with sensible defaults.'''
    return CalibrationRecord(
        exposure_us=exposure_us, gain_db=gain_db,
        timestamp=timestamp if timestamp is not None else time.time(),
        source_frame_count=source_frame_count,
    )


def _make_running_stream(backend: SyntheticBackend | None = None) -> CameraStream:
    '''A started CameraStream against SyntheticBackend, for workflow.py tests.'''
    stream = CameraStream(
        exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
        pixel_format="Mono8", timeout_ms=5000,
        backend=backend if backend is not None else SyntheticBackend(seed=1),
    )
    stream.start()
    return stream

# Classes

# ---------------------------------------------------------------------------
# shared/io.py
# ---------------------------------------------------------------------------

class TestSharedIO:

    def test_round_trips_array_and_record(self, tmp_path):
        array = np.arange(12, dtype=np.float64).reshape(3, 4)
        record = _record(exposure_us=1234.0, gain_db=1.5, source_frame_count=9)

        path = tmp_path / "artifact.npz"
        save_artifact(path, {"array": array}, record)
        loaded_arrays, loaded_record = load_artifact(path, CalibrationRecord)

        assert np.array_equal(loaded_arrays["array"], array)
        assert loaded_record == record

    def test_round_trips_multiple_arrays(self, tmp_path):
        coefficients = np.array([1.0, 2.0])
        coefficient_sigma = np.array([0.1, 0.2])
        record = _record()

        path = tmp_path / "multi.npz"
        save_artifact(
            path, {"coefficients": coefficients, "coefficient_sigma": coefficient_sigma}, record,
        )
        loaded_arrays, loaded_record = load_artifact(path, CalibrationRecord)

        assert np.array_equal(loaded_arrays["coefficients"], coefficients)
        assert np.array_equal(loaded_arrays["coefficient_sigma"], coefficient_sigma)
        assert loaded_record == record

    def test_rejects_array_key_colliding_with_record_prefix(self, tmp_path):
        record = _record()
        with pytest.raises(ValueError):
            save_artifact(tmp_path / "bad.npz", {"record__oops": np.ones(2)}, record)

    def test_round_trips_bool_array(self, tmp_path):
        array = np.zeros((3, 4), dtype=bool)
        array[1, 2] = True
        record = _record()

        path = tmp_path / "mask.npz"
        save_artifact(path, {"mask": array}, record)
        loaded_arrays, _ = load_artifact(path, CalibrationRecord)

        assert loaded_arrays["mask"].dtype == bool
        assert np.array_equal(loaded_arrays["mask"], array)

    def test_npz_suffix_added_if_missing(self, tmp_path):
        array = np.ones((2, 2))
        record = _record()

        path_without_suffix = tmp_path / "artifact"
        save_artifact(path_without_suffix, {"array": array}, record)

        assert (tmp_path / "artifact.npz").exists()
        loaded_arrays, _ = load_artifact(path_without_suffix, CalibrationRecord)
        assert np.array_equal(loaded_arrays["array"], array)

    def test_creates_parent_directory(self, tmp_path):
        array = np.ones((2, 2))
        record = _record()

        nested_path = tmp_path / "nested" / "dir" / "artifact.npz"
        save_artifact(nested_path, {"array": array}, record)

        assert nested_path.exists()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifact(tmp_path / "does_not_exist.npz", CalibrationRecord)


# ---------------------------------------------------------------------------
# shared/metadata.py
# ---------------------------------------------------------------------------

class TestCalibrationRecord:

    def test_rejects_non_positive_exposure(self):
        with pytest.raises(ValueError):
            CalibrationRecord(exposure_us=0.0, gain_db=0.0, timestamp=time.time(), source_frame_count=1)

    def test_rejects_nan_gain(self):
        with pytest.raises(ValueError):
            CalibrationRecord(exposure_us=2000.0, gain_db=float("nan"), timestamp=time.time(), source_frame_count=1)

    def test_rejects_invalid_frame_count(self):
        with pytest.raises(ValueError):
            CalibrationRecord(exposure_us=2000.0, gain_db=0.0, timestamp=time.time(), source_frame_count=0)

    def test_age_seconds_is_small_and_non_negative_just_after_construction(self):
        record = _record()
        assert 0 <= record.age_seconds < 1.0


class TestCheckSettingsMatch:

    def test_passes_on_exact_match(self):
        record = _record(exposure_us=2000.0, gain_db=0.0)
        frame = _frame(_uniform(10), exposure_us=2000.0, gain_db=0.0)
        check_settings_match(frame, record)   # must not raise

    def test_passes_within_tolerance(self):
        record = _record(exposure_us=2000.0, gain_db=0.0)
        # just inside 1% relative exposure tolerance and 0.05dB gain tolerance
        frame = _frame(_uniform(10), exposure_us=2019.0, gain_db=0.04)
        check_settings_match(frame, record)   # must not raise

    def test_raises_on_exposure_mismatch(self):
        record = _record(exposure_us=2000.0, gain_db=0.0)
        frame = _frame(_uniform(10), exposure_us=3000.0, gain_db=0.0)
        with pytest.raises(SettingsMismatchError):
            check_settings_match(frame, record)

    def test_raises_on_gain_mismatch(self):
        record = _record(exposure_us=2000.0, gain_db=0.0)
        frame = _frame(_uniform(10), exposure_us=2000.0, gain_db=5.0)
        with pytest.raises(SettingsMismatchError):
            check_settings_match(frame, record)


# ---------------------------------------------------------------------------
# sensor/saturation.py
# ---------------------------------------------------------------------------

class TestSaturation:

    def test_no_saturation(self):
        frame = _frame(_uniform(100))
        result = check_saturation(frame)
        assert result.is_saturated is False
        assert result.peak_value == 100
        assert result.n_saturated_pixels == 0

    def test_saturation_detected(self):
        image = _uniform(50)
        image[10, 10] = CANONICAL_MAX_VALUE
        image[10, 11] = CANONICAL_MAX_VALUE
        frame = _frame(image)
        result = check_saturation(frame)
        assert result.is_saturated is True
        assert result.peak_value == CANONICAL_MAX_VALUE
        assert result.n_saturated_pixels == 2

    def test_bad_pixel_mask_excludes_known_defect(self):
        image = _uniform(50)
        image[10, 10] = CANONICAL_MAX_VALUE   # the ONLY saturated pixel
        frame = _frame(image)

        mask = np.zeros(CANONICAL_SHAPE, dtype=bool)
        mask[10, 10] = True

        result = check_saturation(frame, bad_pixel_mask=mask)
        assert result.is_saturated is False

    def test_mismatched_mask_shape_raises(self):
        frame = _frame(_uniform(50))
        bad_mask = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError):
            check_saturation(frame, bad_pixel_mask=bad_mask)


# ---------------------------------------------------------------------------
# sensor/baseline.py
# ---------------------------------------------------------------------------

class TestBaseline:

    def test_build_baseline_averages_frames(self):
        frames = [_frame(_uniform(10)), _frame(_uniform(12)), _frame(_uniform(14))]
        result, record = build_baseline(frames)
        assert np.allclose(result.baseline, 12.0)
        assert record.source_frame_count == 3
        assert record.exposure_us == FIXTURE_EXPOSURE_US

    def test_build_baseline_measures_background_sigma(self):
        # Every pixel takes exactly (10, 12, 14) across the 3 frames --
        # sample std (ddof=1) of that fixed set is 2.0 everywhere, so the
        # per-pixel median collapses to exactly 2.0 too.
        frames = [_frame(_uniform(10)), _frame(_uniform(12)), _frame(_uniform(14))]
        result, _ = build_baseline(frames)
        assert np.isclose(result.background_sigma, 2.0)

    def test_build_baseline_rejects_empty_list(self):
        with pytest.raises(ValueError):
            build_baseline([])

    def test_build_baseline_rejects_single_frame(self):
        with pytest.raises(ValueError):
            build_baseline([_frame(_uniform(10))])

    def test_build_baseline_rejects_mismatched_settings(self):
        frames = [_frame(_uniform(10), exposure_us=2000.0), _frame(_uniform(10), exposure_us=3000.0)]
        with pytest.raises(ValueError):
            build_baseline(frames)

    def test_save_load_round_trips(self, tmp_path):
        frames = [_frame(_uniform(10)), _frame(_uniform(12))]
        result, record = build_baseline(frames)

        path = tmp_path / "baseline.npz"
        save_baseline(path, result, record)
        loaded_result, loaded_record = load_baseline(path)

        assert np.array_equal(loaded_result.baseline, result.baseline)
        assert np.isclose(loaded_result.background_sigma, result.background_sigma)
        assert loaded_record == record


# ---------------------------------------------------------------------------
# sensor/flat_field.py
# ---------------------------------------------------------------------------

class TestFlatField:

    def test_build_flat_field_normalizes_uniform_response_to_one(self):
        illuminated = [_frame(_uniform(150)) for _ in range(3)]
        dark = [_frame(_uniform(10)) for _ in range(3)]
        flat_field, record = build_flat_field(illuminated, dark)
        assert np.allclose(flat_field, 1.0)
        assert record.source_frame_count == 3

    def test_build_flat_field_rejects_saturated_source(self):
        illuminated = [_frame(_uniform(CANONICAL_MAX_VALUE))]
        dark = [_frame(_uniform(10))]
        with pytest.raises(InvalidFlatFieldError):
            build_flat_field(illuminated, dark)

    def test_save_load_round_trips(self, tmp_path):
        illuminated = [_frame(_uniform(150)) for _ in range(3)]
        dark = [_frame(_uniform(10)) for _ in range(3)]
        flat_field, record = build_flat_field(illuminated, dark)

        path = tmp_path / "flat_field.npz"
        save_flat_field(path, flat_field, record)
        loaded_flat_field, loaded_record = load_flat_field(path)

        assert np.array_equal(loaded_flat_field, flat_field)
        assert loaded_record == record


# ---------------------------------------------------------------------------
# sensor/bad_pixel_map.py
# ---------------------------------------------------------------------------

class TestBadPixelMap:

    def test_flags_injected_outlier(self):
        flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
        flat_field[300, 400] = 0.0    # dead
        flat_field[300, 401] = 5.0    # hot
        record = _record()

        mask, _ = build_bad_pixel_map(flat_field, record)

        assert mask[300, 400] == True
        assert mask[300, 401] == True
        assert mask[0, 0] == False
        assert mask.sum() == 2

    def test_uniform_flat_field_flags_nothing(self):
        # mad == 0 guard: no deviation to measure against.
        flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
        record = _record()
        mask, _ = build_bad_pixel_map(flat_field, record)
        assert not np.any(mask)

    def test_record_carries_forward_flat_field_settings(self):
        flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
        flat_field_record = _record(exposure_us=3333.0, gain_db=2.0, source_frame_count=7)
        _, bad_pixel_record = build_bad_pixel_map(flat_field, flat_field_record)
        assert bad_pixel_record.exposure_us == 3333.0
        assert bad_pixel_record.gain_db == 2.0
        assert bad_pixel_record.source_frame_count == 7

    def test_save_load_round_trips(self, tmp_path):
        flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
        flat_field[300, 400] = 0.0
        record = _record()
        mask, mask_record = build_bad_pixel_map(flat_field, record)

        path = tmp_path / "bad_pixel_map.npz"
        save_bad_pixel_map(path, mask, mask_record)
        loaded_mask, loaded_record = load_bad_pixel_map(path)

        assert loaded_mask.dtype == bool
        assert np.array_equal(loaded_mask, mask)
        assert loaded_record == mask_record


# ---------------------------------------------------------------------------
# sensor/conversion_gain.py
# ---------------------------------------------------------------------------

def _ptc_frames_by_exposure(exposure_levels: list[float], ds: list[int]) -> dict:
    '''
    Builds frames_by_exposure with an EXACT, known variance_ADU = mean_ADU/2.0 + 1.0
    relationship (gain=2.0 e-/ADU, read-noise variance=1.0 ADU^2) -- 2 integer
    frames per level at [mean-d, mean+d], chosen so the ddof=1 sample variance
    for n=2 (exactly 2*d^2) matches the target relationship exactly, and
    everything stays integer so casting to CANONICAL_DTYPE doesn't distort it.
    mean = 4*d^2 - 2 solves mean/2.0 + 1.0 == 2*d^2 for integer d.
    '''
    frames_by_exposure = {}
    for exposure_us, d in zip(exposure_levels, ds):
        mean = 4 * d * d - 2
        frames_by_exposure[exposure_us] = [
            _frame(_uniform(mean - d), exposure_us=exposure_us),
            _frame(_uniform(mean + d), exposure_us=exposure_us),
        ]
    return frames_by_exposure


class TestConversionGain:

    def test_build_conversion_gain_recovers_known_gain(self):
        frames_by_exposure = _ptc_frames_by_exposure(
            [1000.0, 2000.0, 3000.0, 4000.0], [2, 3, 4, 5],
        )

        result, record = build_conversion_gain(frames_by_exposure)

        assert np.isclose(result.gain_e_per_adu, 2.0, atol=0.01)
        assert np.isclose(result.fit.coefficients[0], 1.0, atol=0.1)
        assert record.n_illumination_levels == 4
        assert record.gain_db == FIXTURE_GAIN_DB

    def test_rejects_too_few_levels(self):
        frames_by_exposure = _ptc_frames_by_exposure([1000.0], [2])
        with pytest.raises(ValueError):
            build_conversion_gain(frames_by_exposure)

    def test_rejects_too_few_frames_per_level(self):
        frames_by_exposure = _ptc_frames_by_exposure([1000.0, 2000.0], [2, 3])
        frames_by_exposure[1000.0] = frames_by_exposure[1000.0][:1]
        with pytest.raises(ValueError):
            build_conversion_gain(frames_by_exposure)

    def test_rejects_exposure_mismatch(self):
        frames_by_exposure = _ptc_frames_by_exposure([1000.0, 2000.0], [2, 3])
        frames_by_exposure[1000.0][0] = _frame(_uniform(10), exposure_us=999.0)
        with pytest.raises(ValueError):
            build_conversion_gain(frames_by_exposure)

    def test_rejects_gain_mismatch_across_levels(self):
        frames_by_exposure = _ptc_frames_by_exposure([1000.0, 2000.0], [2, 3])
        frames_by_exposure[2000.0] = [
            _frame(_uniform(30), exposure_us=2000.0, gain_db=5.0),
            _frame(_uniform(40), exposure_us=2000.0, gain_db=5.0),
        ]
        with pytest.raises(ValueError):
            build_conversion_gain(frames_by_exposure)

    def test_rejects_saturated_frame(self):
        frames_by_exposure = _ptc_frames_by_exposure([1000.0], [2])
        frames_by_exposure[2000.0] = [
            _frame(_uniform(CANONICAL_MAX_VALUE), exposure_us=2000.0),
            _frame(_uniform(CANONICAL_MAX_VALUE), exposure_us=2000.0),
        ]
        with pytest.raises(InvalidConversionGainError):
            build_conversion_gain(frames_by_exposure)

    def test_rejects_non_positive_slope(self):
        # mean increases 20 -> 80 while variance DECREASES 200 -> 50 --
        # physically invalid (noise variance can't fall as signal rises).
        frames_by_exposure = {
            1000.0: [_frame(_uniform(10), exposure_us=1000.0), _frame(_uniform(30), exposure_us=1000.0)],
            2000.0: [_frame(_uniform(75), exposure_us=2000.0), _frame(_uniform(85), exposure_us=2000.0)],
        }
        with pytest.raises(InvalidConversionGainError):
            build_conversion_gain(frames_by_exposure)

    def test_save_load_round_trips(self, tmp_path):
        frames_by_exposure = _ptc_frames_by_exposure([1000.0, 2000.0, 3000.0], [2, 3, 4])
        result, record = build_conversion_gain(frames_by_exposure)

        path = tmp_path / "conversion_gain.npz"
        save_conversion_gain(path, result, record)
        loaded_result, loaded_record = load_conversion_gain(path)

        assert np.isclose(loaded_result.gain_e_per_adu, result.gain_e_per_adu)
        assert np.allclose(loaded_result.fit.coefficients, result.fit.coefficients)
        assert loaded_record == record


# ---------------------------------------------------------------------------
# sensor/workflow.py
# ---------------------------------------------------------------------------

class TestRunBaselineCalibration:

    def test_acquires_builds_saves_and_returns_record(self, tmp_path):
        stream = _make_running_stream()
        try:
            path = tmp_path / "baseline.npz"
            record = run_baseline_calibration(stream, n_frames=5, path=path)

            assert record.source_frame_count == 5
            assert path.exists()

            loaded_result, loaded_record = load_baseline(path)
            assert loaded_result.baseline.shape == CANONICAL_SHAPE
            assert loaded_record == record
        finally:
            stream.stop()

    def test_raises_if_stream_not_running(self, tmp_path):
        stream = _make_running_stream()
        stream.stop()
        with pytest.raises(RuntimeError):
            run_baseline_calibration(stream, n_frames=3, path=tmp_path / "baseline.npz")


class TestFlatFieldCalibrationWorkflow:

    def test_capture_phases_then_finish(self, tmp_path):
        # peak_counts well under Mono8's 255 ceiling -- the default
        # SyntheticBackend peak (3000) saturates on an 8-bit sensor and
        # would trip build_flat_field()'s saturation check regardless of
        # exposure/gain, which SyntheticBackend doesn't use to scale the
        # signal (see backends.py).
        stream = _make_running_stream(backend=SyntheticBackend(seed=1, peak_counts=150.0))
        try:
            dark = capture_dark_frames(stream, 3)
            illuminated = capture_illuminated_frames(stream, 3)

            assert len(dark) == 3
            assert len(illuminated) == 3
            assert all(isinstance(f, FrameData) for f in dark + illuminated)

            path = tmp_path / "flat_field.npz"
            record = finish_flat_field_calibration(illuminated, dark, path)

            assert record.source_frame_count == 3
            assert path.exists()

            loaded_flat_field, loaded_record = load_flat_field(path)
            assert loaded_flat_field.shape == CANONICAL_SHAPE
            assert loaded_record == record
        finally:
            stream.stop()


class TestRunConversionGainCalibration:

    def _fake_result_and_record(self, frames_by_exposure):
        fit = PolynomialFitResult(
            degree=1,
            coefficients=np.array([1.0, 0.5]),
            coefficient_sigma=np.array([0.1, 0.05]),
            reduced_chi_squared=1.0,
            residuals=np.zeros(len(frames_by_exposure)),
            normalized_residuals=np.zeros(len(frames_by_exposure)),
        )
        record = ConversionGainRecord(
            gain_db=FIXTURE_GAIN_DB, timestamp=time.time(),
            n_illumination_levels=len(frames_by_exposure),
        )
        return ConversionGainResult(fit=fit), record

    def test_sweeps_exposure_and_restores_original(self, tmp_path, monkeypatch):
        # SyntheticBackend doesn't scale its signal by exposure_us (see
        # TestFlatFieldCalibrationWorkflow's comment), so this stubs
        # build_conversion_gain() itself to test the stop/reconfigure/
        # start/collect sequencing and exposure-restoration behavior --
        # the real wiring concern unique to this function -- independent
        # of whether the backend can produce a realistic PTC relationship.
        captured_exposures = []

        def fake_build_conversion_gain(frames_by_exposure, fitter=None):
            captured_exposures.extend(sorted(frames_by_exposure.keys()))
            return self._fake_result_and_record(frames_by_exposure)

        monkeypatch.setattr(
            "pipeline.calibration.sensor.workflow.build_conversion_gain",
            fake_build_conversion_gain,
        )

        stream = _make_running_stream()
        original_exposure = stream.exposure_us
        try:
            path = tmp_path / "conversion_gain.npz"
            record = run_conversion_gain_calibration(
                stream, exposure_min_us=1000.0, exposure_max_us=4000.0,
                n_levels=4, n_frames_per_level=3, path=path,
            )

            assert record.n_illumination_levels == 4
            assert path.exists()
            assert len(captured_exposures) == 4
            assert np.isclose(captured_exposures[0], 1000.0)
            assert np.isclose(captured_exposures[-1], 4000.0)

            # original exposure restored and stream left running afterward
            assert stream.exposure_us == original_exposure
            assert stream.is_running
        finally:
            stream.stop()

    def test_raises_if_stream_not_running(self, tmp_path):
        stream = _make_running_stream()
        stream.stop()
        with pytest.raises(RuntimeError):
            run_conversion_gain_calibration(
                stream, exposure_min_us=1000.0, exposure_max_us=2000.0,
                n_levels=2, n_frames_per_level=2, path=tmp_path / "cg.npz",
            )

    def test_rejects_too_few_levels(self):
        stream = CameraStream(
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            pixel_format="Mono8", timeout_ms=5000, backend=SyntheticBackend(seed=1),
        )
        with pytest.raises(ValueError):
            run_conversion_gain_calibration(
                stream, exposure_min_us=1000.0, exposure_max_us=2000.0,
                n_levels=1, n_frames_per_level=2, path="unused",
            )

    def test_rejects_too_few_frames_per_level(self):
        stream = CameraStream(
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            pixel_format="Mono8", timeout_ms=5000, backend=SyntheticBackend(seed=1),
        )
        with pytest.raises(ValueError):
            run_conversion_gain_calibration(
                stream, exposure_min_us=1000.0, exposure_max_us=2000.0,
                n_levels=2, n_frames_per_level=1, path="unused",
            )

    def test_rejects_invalid_exposure_range(self):
        stream = CameraStream(
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            pixel_format="Mono8", timeout_ms=5000, backend=SyntheticBackend(seed=1),
        )
        with pytest.raises(ValueError):
            run_conversion_gain_calibration(
                stream, exposure_min_us=2000.0, exposure_max_us=1000.0,
                n_levels=2, n_frames_per_level=2, path="unused",
            )


# ---------------------------------------------------------------------------
# shared/fitting.py, shared/result.py
# ---------------------------------------------------------------------------

class TestTotalLeastSquaresFit:

    def test_recovers_known_linear_relationship(self):
        true_intercept, true_slope = 5.0, 2.0
        x = np.linspace(0, 10, 20)
        y = true_intercept + true_slope * x
        sigma_x = np.full_like(x, 0.01)
        sigma_y = np.full_like(y, 0.01)

        fit = TotalLeastSquaresFit().fit(x, y, sigma_x, sigma_y, degree=1)

        assert isinstance(fit, PolynomialFitResult)
        assert fit.degree == 1
        assert np.isclose(fit.coefficients[0], true_intercept, atol=0.1)
        assert np.isclose(fit.coefficients[1], true_slope, atol=0.1)
        assert np.allclose(fit.residuals, 0.0, atol=0.1)

    def test_insufficient_points_raises(self):
        x = np.array([1.0])
        y = np.array([2.0])
        sigma = np.array([0.1])
        with pytest.raises(InsufficientDataError):
            TotalLeastSquaresFit().fit(x, y, sigma, sigma, degree=1)

    def test_evaluate_and_derivative(self):
        true_intercept, true_slope = 3.0, 4.0
        x = np.linspace(0, 10, 20)
        y = true_intercept + true_slope * x
        sigma = np.full_like(x, 0.01)

        fit = TotalLeastSquaresFit().fit(x, y, sigma, sigma, degree=1)

        assert np.isclose(fit.evaluate(np.array([0.0]))[0], true_intercept, atol=0.1)
        assert np.isclose(fit.evaluate_derivative(np.array([0.0]))[0], true_slope, atol=0.1)


# ---------------------------------------------------------------------------
# spatial/calibrate.py, spatial/io.py
# ---------------------------------------------------------------------------

class TestScaleFactorPositionCalibration:

    def test_default_scale_factor(self):
        calibration = ScaleFactorPositionCalibration()
        assert calibration.scale_factor == DEFAULT_SCALE_FACTOR

    def test_convert_scales_position_and_sigma(self):
        calibration = ScaleFactorPositionCalibration(scale_factor=2.0)
        x0 = np.array([1.0, 2.0, 3.0])
        sigma_x0 = np.array([0.1, 0.2, 0.3])

        converted_x0, converted_sigma = calibration.convert(x0, sigma_x0)

        assert np.array_equal(converted_x0, x0 * 2.0)
        assert np.array_equal(converted_sigma, sigma_x0 * 2.0)

    def test_rejects_non_positive_scale_factor(self):
        with pytest.raises(ValueError):
            ScaleFactorPositionCalibration(scale_factor=0.0)


class TestScaleFactorPersistence:

    def test_load_missing_file_returns_default(self, tmp_path):
        calibration, record = load_scale_factor(tmp_path / "does_not_exist.npz")
        assert calibration.scale_factor == DEFAULT_SCALE_FACTOR
        assert record.source == "default"

    def test_save_load_round_trips_manual_override(self, tmp_path):
        path = tmp_path / "scale_factor.npz"
        calibration = ScaleFactorPositionCalibration(scale_factor=1.62)
        save_scale_factor(path, calibration, source="manual")

        loaded_calibration, loaded_record = load_scale_factor(path)

        assert np.isclose(loaded_calibration.scale_factor, 1.62)
        assert loaded_record.source == "manual"

    def test_rejects_invalid_source(self):
        with pytest.raises(ValueError):
            ScaleFactorRecord(source="bogus", timestamp=time.time())


# ---------------------------------------------------------------------------
# spectral/calibrate.py, spectral/io.py, spectral/line_matching.py
# ---------------------------------------------------------------------------

class TestCalibrateSpectral:

    def test_fits_and_implements_wavelength_axis(self):
        true_intercept, true_slope = 400.0, 0.5
        pixel = np.linspace(0, 1900, 10)
        wavelength_nm = true_intercept + true_slope * pixel
        sigma_pixel = np.full_like(pixel, 0.05)
        sigma_wavelength_nm = np.full_like(pixel, 0.01)
        record = _record()

        result = calibrate_spectral(
            pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm, record, degree=1,
        )

        assert isinstance(result, WavelengthCalibrationResult)
        recovered = result.wavelength_nm(np.array([0.0]))
        assert np.isclose(recovered[0], true_intercept, atol=1.0)

        sigma = result.sigma_wavelength_nm(np.array([0.0, 1000.0]))
        assert np.all(sigma > 0)

    def test_save_load_round_trips(self, tmp_path):
        pixel = np.linspace(0, 1900, 10)
        wavelength_nm = 400.0 + 0.5 * pixel
        sigma_pixel = np.full_like(pixel, 0.05)
        sigma_wavelength_nm = np.full_like(pixel, 0.01)
        record = _record()

        result = calibrate_spectral(
            pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm, record, degree=1,
        )

        path = tmp_path / "spectral.npz"
        save_spectral_calibration(path, result)
        loaded = load_spectral_calibration(path)

        assert loaded.fit.degree == result.fit.degree
        assert np.allclose(loaded.fit.coefficients, result.fit.coefficients)
        assert np.allclose(loaded.fit.coefficient_sigma, result.fit.coefficient_sigma)
        assert loaded.record == result.record


class TestLineMatchingStub:

    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            match_lines(np.zeros(CANONICAL_SHAPE))


def _sensor_calibration_set() -> CalibrationSet:
    '''A trivial, no-op CalibrationSet (zero baseline, unity flat field,
    no bad pixels) -- enough for run_preprocessing() to run without
    altering the synthetic frame meaningfully, for workflow-wiring tests
    that don't care about the correction math itself.'''
    record = _record()
    return CalibrationSet(
        baseline=_uniform(0, dtype=np.float64),
        baseline_record=record,
        flat_field=np.ones(CANONICAL_SHAPE, dtype=np.float64),
        flat_field_record=record,
        bad_pixel_mask=np.zeros(CANONICAL_SHAPE, dtype=bool),
    )


class TestRunSpectralCalibration:

    def test_wiring_with_stubbed_line_matching(self, tmp_path, monkeypatch):
        pixel = np.linspace(0, 1900, 10)
        wavelength_nm = 400.0 + 0.5 * pixel
        sigma_pixel = np.full_like(pixel, 0.05)
        sigma_wavelength_nm = np.full_like(pixel, 0.01)

        monkeypatch.setattr(
            "pipeline.calibration.spectral.workflow.match_lines",
            lambda image: (pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm),
        )

        stream = _make_running_stream()
        try:
            path = tmp_path / "spectral.npz"
            result = run_spectral_calibration(
                stream, n_frames=3, sensor_calibration=_sensor_calibration_set(),
                path=path, degree=1,
            )

            assert path.exists()
            assert result.record.source_frame_count == 3

            loaded = load_spectral_calibration(path)
            assert np.allclose(loaded.fit.coefficients, result.fit.coefficients)
        finally:
            stream.stop()

    def test_propagates_not_implemented_when_line_matching_missing(self, tmp_path):
        stream = _make_running_stream()
        try:
            with pytest.raises(NotImplementedError):
                run_spectral_calibration(
                    stream, n_frames=1, sensor_calibration=_sensor_calibration_set(),
                    path=tmp_path / "spectral.npz",
                )
        finally:
            stream.stop()
