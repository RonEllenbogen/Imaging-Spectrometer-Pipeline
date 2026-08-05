"""
Test suite for the calibration package (calibration/sensor/ so far --
spectral/ and spatial/ don't exist yet). Every test operates on synthetic
FrameData objects with known, injected values -- no camera or real
calibration data involved, following the same "prove against known
ground truth first" principle used throughout acquisition/'s test suite.

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

from pipeline.calibration.exceptions import SettingsMismatchError, InvalidFlatFieldError
from pipeline.calibration.shared import save_artifact, load_artifact
from pipeline.calibration.sensor import (
    build_baseline, save_baseline, load_baseline,
    build_flat_field, save_flat_field, load_flat_field,
    build_bad_pixel_map, save_bad_pixel_map, load_bad_pixel_map,
    CalibrationRecord, check_settings_match,
    EXPOSURE_MATCH_TOLERANCE_REL, GAIN_MATCH_TOLERANCE_ABS,
    check_saturation, SaturationCheckResult,
    run_baseline_calibration,
    capture_dark_frames, capture_illuminated_frames, finish_flat_field_calibration,
)

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
        save_artifact(path, array, record)
        loaded_array, loaded_record = load_artifact(path, CalibrationRecord)

        assert np.array_equal(loaded_array, array)
        assert loaded_record == record

    def test_round_trips_bool_array(self, tmp_path):
        array = np.zeros((3, 4), dtype=bool)
        array[1, 2] = True
        record = _record()

        path = tmp_path / "mask.npz"
        save_artifact(path, array, record)
        loaded_array, _ = load_artifact(path, CalibrationRecord)

        assert loaded_array.dtype == bool
        assert np.array_equal(loaded_array, array)

    def test_npz_suffix_added_if_missing(self, tmp_path):
        array = np.ones((2, 2))
        record = _record()

        path_without_suffix = tmp_path / "artifact"
        save_artifact(path_without_suffix, array, record)

        assert (tmp_path / "artifact.npz").exists()
        loaded_array, _ = load_artifact(path_without_suffix, CalibrationRecord)
        assert np.array_equal(loaded_array, array)

    def test_creates_parent_directory(self, tmp_path):
        array = np.ones((2, 2))
        record = _record()

        nested_path = tmp_path / "nested" / "dir" / "artifact.npz"
        save_artifact(nested_path, array, record)

        assert nested_path.exists()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifact(tmp_path / "does_not_exist.npz", CalibrationRecord)


# ---------------------------------------------------------------------------
# sensor/metadata.py
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
        baseline, record = build_baseline(frames)
        assert np.allclose(baseline, 12.0)
        assert record.source_frame_count == 3
        assert record.exposure_us == FIXTURE_EXPOSURE_US

    def test_build_baseline_rejects_empty_list(self):
        with pytest.raises(ValueError):
            build_baseline([])

    def test_build_baseline_rejects_mismatched_settings(self):
        frames = [_frame(_uniform(10), exposure_us=2000.0), _frame(_uniform(10), exposure_us=3000.0)]
        with pytest.raises(ValueError):
            build_baseline(frames)

    def test_save_load_round_trips(self, tmp_path):
        frames = [_frame(_uniform(10)), _frame(_uniform(12))]
        baseline, record = build_baseline(frames)

        path = tmp_path / "baseline.npz"
        save_baseline(path, baseline, record)
        loaded_baseline, loaded_record = load_baseline(path)

        assert np.array_equal(loaded_baseline, baseline)
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

            loaded_baseline, loaded_record = load_baseline(path)
            assert loaded_baseline.shape == CANONICAL_SHAPE
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
