"""
Single public entry point for preprocessing. Enforces the correction
order as a property of the code, not documentation the caller has to
remember: frame validation -> saturation check -> baseline subtraction ->
flat-field division -> bad-pixel masking -> signal-threshold masking ->
optional ROI masking.

Building calibration artifacts (baseline, flat field, bad-pixel map) is
NOT this function's job -- that happens once, infrequently, via each
artifact's own build_*() function in calibration/sensor/. This function
only applies already-built artifacts to one science frame at a time.
"""

from dataclasses import dataclass

import numpy as np

from pipeline.acquisition import FrameData
from pipeline.calibration.sensor import check_saturation, SaturationCheckResult
from pipeline.calibration.shared import CalibrationRecord

from .processed_frame import ProcessedFrame
from .validation import check_frame_sanity
from .steps import apply_roi, apply_baseline, apply_flat_field, apply_bad_pixel_map, apply_signal_threshold


@dataclass(frozen=True, slots=True)
class CalibrationSet:

    '''
    Bundles the calibration artifacts run_preprocessing() needs. Built
    once via each artifact's own build_*() function, then reused across
    every science frame in a session.

    No bad_pixel_record field -- apply_bad_pixel_map() only needs the
    mask itself, so there's no consumer for it here. baseline_record and
    flat_field_record ARE both consumed (settings-check and
    logging/provenance respectively), so both stay.

    background_sigma is required (no default): every call to
    run_preprocessing() now runs signal-threshold masking, which cannot
    compute a noise floor without it -- see steps/signal_threshold.py.
    '''

    baseline: np.ndarray
    baseline_record: CalibrationRecord
    flat_field: np.ndarray
    flat_field_record: CalibrationRecord
    bad_pixel_mask: np.ndarray
    background_sigma: float


def run_preprocessing(
    frame: FrameData,
    calibration: CalibrationSet,
    roi_bounds: tuple[int, int] | None = None,
) -> tuple[ProcessedFrame, SaturationCheckResult]:

    '''
    Runs one raw science frame through the full preprocessing pipeline.

    Parameters
    ----------
    frame
        The raw science frame to process. NOT for background/baseline
        captures -- those go through build_baseline()/build_flat_field()
        directly, since near-zero signal is expected and correct there,
        not something check_frame_sanity() should reject.
    calibration
        The pre-built baseline, flat field, bad-pixel map, and
        background_sigma to apply.
    roi_bounds
        (row_min, row_max) to mask outside of, or None to skip ROI
        entirely. Still pending an empirical check (a real background
        frame's histogram) on whether it's needed for this setup at all
        -- kept optional rather than mandatory until that's resolved.

    Returns
    -------
    tuple[ProcessedFrame, SaturationCheckResult]
        The corrected frame, and the raw-frame saturation result.
        Saturation does NOT raise here -- returned so the caller (e.g. a
        batch-capture loop) decides whether to discard, log, or escalate,
        the same "caller decides" principle steps/saturation.py was
        built around.

    Raises
    ------
    NoSignalError
        If the raw frame contains no signal at all -- unlike saturation,
        this can't be meaningfully processed further, so it's a hard
        failure here rather than a returned result.
    SettingsMismatchError
        If frame's exposure_us/gain_db don't match
        calibration.baseline_record's, beyond tolerance.
    '''

    check_frame_sanity(frame)

    saturation_result = check_saturation(frame, bad_pixel_mask=calibration.bad_pixel_mask)

    processed = apply_baseline(frame, calibration.baseline, calibration.baseline_record)
    processed = apply_flat_field(processed, calibration.flat_field, calibration.flat_field_record)
    processed = apply_bad_pixel_map(processed, calibration.bad_pixel_mask)
    processed = apply_signal_threshold(processed, calibration.background_sigma)
    valid_columns = processed.valid_columns

    if roi_bounds is not None:
        row_min, row_max = roi_bounds
        processed = apply_roi(processed, row_min, row_max)
        # apply_roi() (unchanged -- see steps/roi.py) rebuilds ProcessedFrame
        # without carrying valid_columns forward, so it's reattached here
        # rather than lost; ROI only zeroes spatial rows and doesn't change
        # which spectral columns are signal-bearing.
        processed = ProcessedFrame(
            image=processed.image, frame_id=processed.frame_id, timestamp=processed.timestamp,
            exposure_us=processed.exposure_us, gain_db=processed.gain_db,
            valid_columns=valid_columns,
        )

    return processed, saturation_result


__all__ = ["CalibrationSet", "run_preprocessing"]