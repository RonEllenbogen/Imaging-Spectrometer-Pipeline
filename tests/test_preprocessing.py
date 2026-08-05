"""
Test suite for the preprocessing package. Every test operates on
synthetic FrameData/ProcessedFrame objects with known, injected values --
no camera or real calibration data involved -- following the same
"prove against known ground truth first" principle used throughout
acquisition/'s test suite.

Covers apply_*() (per-frame correction) and the end-to-end pipeline;
build_*() (calibration-artifact construction) moved to
tests/test_calibration.py alongside calibration/sensor/, which this file
still uses (via build_*()) to set up realistic fixtures for the
end-to-end test.

Kept as a single file per the explicit request, despite covering many
source files -- a deviation from the one-test-file-per-module convention
used elsewhere. Worth splitting into per-module files (test_baseline.py,
test_flat_field.py, etc.) if this grows unwieldy.
"""

# Imports

import time

import numpy as np
import pytest

from pipeline.acquisition import FrameData, CANONICAL_SHAPE, CANONICAL_DTYPE, CANONICAL_MAX_VALUE

from pipeline.preprocessing import (
    run_preprocessing, CalibrationSet, ProcessedFrame,
    PreprocessingError, SettingsMismatchError, SaturationError, NoSignalError,
)
from pipeline.preprocessing.validation import check_frame_sanity
from pipeline.preprocessing.steps import apply_roi, apply_baseline, apply_flat_field, MIN_FLAT_FIELD_VALUE, apply_bad_pixel_map
from pipeline.calibration.sensor import (
    build_baseline, build_flat_field, build_bad_pixel_map, CalibrationRecord,
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


def _processed(image: np.ndarray, frame_id=FIXTURE_FRAME_ID, timestamp=FIXTURE_TIMESTAMP,
               exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB) -> ProcessedFrame:
    '''Builds a valid ProcessedFrame from a raw array, casting to float64.'''
    return ProcessedFrame(
        image=image.astype(np.float64), frame_id=frame_id, timestamp=timestamp,
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


def _make_clean_calibration_set(baseline_value=10.0, exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB) -> CalibrationSet:
    '''
    A calibration set with a uniform baseline, a flat (no correction
    needed) flat field, and an empty bad-pixel mask -- for pipeline tests
    that aren't specifically exercising calibration-artifact behavior.
    '''
    baseline = np.full(CANONICAL_SHAPE, baseline_value, dtype=np.float64)
    baseline_record = _record(exposure_us=exposure_us, gain_db=gain_db)
    flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
    flat_field_record = _record(exposure_us=exposure_us, gain_db=gain_db)
    bad_pixel_mask = np.zeros(CANONICAL_SHAPE, dtype=bool)
    return CalibrationSet(
        baseline=baseline, baseline_record=baseline_record,
        flat_field=flat_field, flat_field_record=flat_field_record,
        bad_pixel_mask=bad_pixel_mask,
    )

# Classes

# ---------------------------------------------------------------------------
# validation/frame_checks.py
# ---------------------------------------------------------------------------

class TestFrameSanity:

    def test_all_zero_frame_raises_no_signal(self):
        frame = _frame(_uniform(0))
        with pytest.raises(NoSignalError):
            check_frame_sanity(frame)

    def test_frame_with_signal_passes(self):
        image = _uniform(0)
        image[600, 960] = 100
        frame = _frame(image)
        check_frame_sanity(frame)   # must not raise


# ---------------------------------------------------------------------------
# steps/roi.py
# ---------------------------------------------------------------------------

class TestRoi:

    def test_preserves_inside_window_zeroes_outside(self):
        image = np.full(CANONICAL_SHAPE, 7.0)
        frame = _processed(image)

        result = apply_roi(frame, row_min=500, row_max=700)

        assert np.all(result.image[500:700, :] == 7.0)
        assert np.all(result.image[:500, :] == 0.0)
        assert np.all(result.image[700:, :] == 0.0)

    def test_full_frame_bounds_is_a_noop(self):
        image = np.full(CANONICAL_SHAPE, 3.0)
        frame = _processed(image)
        result = apply_roi(frame, row_min=0, row_max=CANONICAL_SHAPE[0])
        assert np.array_equal(result.image, image)

    def test_metadata_preserved(self):
        frame = _processed(np.full(CANONICAL_SHAPE, 1.0), frame_id=42, exposure_us=2500.0, gain_db=1.5)
        result = apply_roi(frame, row_min=100, row_max=200)
        assert result.frame_id == 42
        assert result.exposure_us == 2500.0
        assert result.gain_db == 1.5

    @pytest.mark.parametrize("row_min,row_max", [
        (700, 500),
        (-1, 500),
        (500, CANONICAL_SHAPE[0] + 1),
    ])
    def test_invalid_bounds_raise(self, row_min, row_max):
        frame = _processed(np.full(CANONICAL_SHAPE, 1.0))
        with pytest.raises(ValueError):
            apply_roi(frame, row_min=row_min, row_max=row_max)


# ---------------------------------------------------------------------------
# steps/baseline.py
# ---------------------------------------------------------------------------

class TestApplyBaseline:

    def test_apply_baseline_subtracts(self):
        frame = _frame(_uniform(50))
        baseline = _uniform(10, dtype=np.float64)
        record = _record()
        result = apply_baseline(frame, baseline, record)
        assert np.allclose(result.image, 40.0)

    def test_apply_baseline_clips_at_zero(self):
        frame = _frame(_uniform(5))
        baseline = _uniform(10, dtype=np.float64)   # baseline exceeds signal everywhere
        record = _record()
        result = apply_baseline(frame, baseline, record)
        assert np.all(result.image == 0.0)   # would be -5 unclipped

    def test_apply_baseline_raises_on_settings_mismatch(self):
        frame = _frame(_uniform(50), exposure_us=2000.0)
        baseline = _uniform(10, dtype=np.float64)
        record = _record(exposure_us=5000.0)
        with pytest.raises(SettingsMismatchError):
            apply_baseline(frame, baseline, record)

    def test_apply_baseline_shape_mismatch_raises(self):
        frame = _frame(_uniform(50))
        bad_baseline = np.zeros((10, 10))
        record = _record()
        with pytest.raises(ValueError):
            apply_baseline(frame, bad_baseline, record)


# ---------------------------------------------------------------------------
# steps/flat_field.py
# ---------------------------------------------------------------------------

class TestApplyFlatField:

    def test_apply_flat_field_removes_known_gain_pattern(self):
        # A true uniform scene T=100, observed through a sensor with a
        # known gain pattern -- half the frame at 1.1x, half at 0.9x.
        true_value = 100.0
        gain = np.ones(CANONICAL_SHAPE, dtype=np.float64)
        gain[:600, :] = 1.1
        gain[600:, :] = 0.9

        observed = _processed(true_value * gain)
        record = _record()

        result = apply_flat_field(observed, gain, record)
        assert np.allclose(result.image, true_value, rtol=1e-9)

    def test_apply_flat_field_safe_division_no_nan_or_inf(self):
        flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
        flat_field[0, 0] = 0.0   # a dead pixel

        frame = _processed(np.full(CANONICAL_SHAPE, 50.0))
        record = _record()

        result = apply_flat_field(frame, flat_field, record)
        assert np.all(np.isfinite(result.image))

    def test_apply_flat_field_shape_mismatch_raises(self):
        frame = _processed(np.full(CANONICAL_SHAPE, 50.0))
        bad_flat_field = np.ones((10, 10))
        record = _record()
        with pytest.raises(ValueError):
            apply_flat_field(frame, bad_flat_field, record)

    def test_apply_flat_field_does_not_check_settings(self):
        # Deliberate: PRNU is treated as exposure/gain-independent, unlike baseline.
        frame = _processed(np.full(CANONICAL_SHAPE, 50.0), exposure_us=9999.0)
        flat_field = np.ones(CANONICAL_SHAPE, dtype=np.float64)
        record = _record(exposure_us=1.0)
        apply_flat_field(frame, flat_field, record)   # must not raise


# ---------------------------------------------------------------------------
# steps/bad_pixel_map.py
# ---------------------------------------------------------------------------

class TestApplyBadPixelMap:

    def test_apply_bad_pixel_map_zeroes_flagged_pixels(self):
        image = np.full(CANONICAL_SHAPE, 42.0)
        frame = _processed(image)

        mask = np.zeros(CANONICAL_SHAPE, dtype=bool)
        mask[5, 5] = True

        result = apply_bad_pixel_map(frame, mask)
        assert result.image[5, 5] == 0.0
        assert result.image[0, 0] == 42.0

    def test_apply_bad_pixel_map_shape_mismatch_raises(self):
        frame = _processed(np.full(CANONICAL_SHAPE, 42.0))
        bad_mask = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError):
            apply_bad_pixel_map(frame, bad_mask)


# ---------------------------------------------------------------------------
# preprocessing_pipeline.py -- end-to-end wiring
# ---------------------------------------------------------------------------

class TestPreprocessingPipeline:

    def test_end_to_end_recovers_known_signal(self):
        '''
        Builds real calibration artifacts via build_baseline()/
        build_flat_field()/build_bad_pixel_map() -- not hand-constructed
        -- then confirms a raw synthetic frame (true signal plus a known
        offset) comes out clean after run_preprocessing().
        '''
        true_signal = np.zeros(CANONICAL_SHAPE, dtype=np.float64)
        true_signal[550:650, :] = 80.0   # a bright "beam" band
        offset = 20.0

        background_frames = [_frame(_uniform(offset)) for _ in range(5)]
        baseline, baseline_record = build_baseline(background_frames)

        illuminated_frames = [_frame(_uniform(150)) for _ in range(5)]
        dark_frames = [_frame(_uniform(offset)) for _ in range(5)]
        flat_field, flat_field_record = build_flat_field(illuminated_frames, dark_frames)

        bad_pixel_mask, _ = build_bad_pixel_map(flat_field, flat_field_record)

        calibration = CalibrationSet(
            baseline=baseline, baseline_record=baseline_record,
            flat_field=flat_field, flat_field_record=flat_field_record,
            bad_pixel_mask=bad_pixel_mask,
        )

        raw_image = np.clip(true_signal + offset, 0, CANONICAL_MAX_VALUE)
        raw_frame = _frame(raw_image, frame_id=7)

        processed, saturation_result = run_preprocessing(raw_frame, calibration, roi_bounds=None)

        assert np.allclose(processed.image, true_signal, atol=1.0)
        assert saturation_result.is_saturated is False
        assert processed.frame_id == 7

    def test_no_signal_raises(self):
        calibration = _make_clean_calibration_set()
        blank_frame = _frame(_uniform(0))
        with pytest.raises(NoSignalError):
            run_preprocessing(blank_frame, calibration)

    def test_saturation_is_returned_not_raised(self):
        calibration = _make_clean_calibration_set(baseline_value=0.0)
        image = _uniform(50)
        image[0, 0] = CANONICAL_MAX_VALUE
        frame = _frame(image)

        processed, saturation_result = run_preprocessing(frame, calibration)   # must not raise
        assert saturation_result.is_saturated is True

    def test_settings_mismatch_raises(self):
        calibration = _make_clean_calibration_set(exposure_us=2000.0)
        frame = _frame(_uniform(50), exposure_us=9000.0)
        with pytest.raises(SettingsMismatchError):
            run_preprocessing(frame, calibration)

    def test_roi_applied_when_bounds_given(self):
        calibration = _make_clean_calibration_set(baseline_value=0.0)
        frame = _frame(_uniform(50))

        processed, _ = run_preprocessing(frame, calibration, roi_bounds=(500, 700))

        assert np.all(processed.image[:500, :] == 0.0)
        assert np.all(processed.image[700:, :] == 0.0)
        assert np.all(processed.image[500:700, :] == 50.0)

    def test_roi_skipped_when_bounds_none(self):
        calibration = _make_clean_calibration_set(baseline_value=0.0)
        frame = _frame(_uniform(50))

        processed, _ = run_preprocessing(frame, calibration, roi_bounds=None)

        assert np.all(processed.image == 50.0)