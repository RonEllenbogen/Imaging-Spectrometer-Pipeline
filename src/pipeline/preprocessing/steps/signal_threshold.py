"""
Flags spectral columns whose total signal is statistically
indistinguishable from background noise, rather than zeroing pixels --
unlike bad_pixel_map/roi, there's no defective/out-of-window pixel to
zero here, only a per-column decision that downstream consumers
(analysis/centroiding.py's extract_centroids()) need in order to skip a
near-zero total_intensity denominator instead of dividing by it
unguarded (see centroiding.py's module docstring).

Runs before ROI masking in preprocessing_pipeline.py's correction order,
not after: ROI zeroes rows outside the spatial window, which would make
n_spatial_pixels here (still the full canonical spatial extent) an
overestimate of how many real, noisy pixels remain in a column once ROI
has run, silently biasing the noise floor down.
"""

# Imports

import numpy as np

from pipeline.acquisition import SPATIAL_AXIS

from ..processed_frame import ProcessedFrame

# Constants

# Unverified starting point -- no real background/beam data has been
# used to tune this yet. Revisit once real frames are available.
SNR_THRESHOLD = 2.0

# Classes

# Functions


def apply_signal_threshold(frame: ProcessedFrame, background_sigma: float) -> ProcessedFrame:

    '''
    Flags each spectral column as valid or not based on whether its
    total spatial-axis signal clears SNR_THRESHOLD against the
    background noise floor.

    Sums each column over the spatial axis (frame.py's SPATIAL_AXIS/
    SPECTRAL_AXIS convention -- axis 0 is spatial, axis 1 is spectral) to
    get one total_intensity per spectral column. Under a no-signal null
    hypothesis, that sum's noise floor is sqrt(n_spatial_pixels) *
    background_sigma (n_spatial_pixels independent noise pixels summed
    in quadrature). A column is valid if
    total_intensity / noise_floor >= SNR_THRESHOLD.

    Does NOT modify frame.image -- this step only computes and attaches
    valid_columns, unlike bad_pixel_map/roi which zero pixels. Known
    accepted imprecision: bad-pixel-masked pixels (already zeroed by
    this point in the pipeline) are still counted in n_spatial_pixels,
    making the noise floor slightly conservative near bad-pixel
    clusters -- acceptable, not something to "fix" here.

    Parameters
    ----------
    frame
        The frame to flag -- expected to already be baseline-subtracted,
        flat-field-divided, and bad-pixel-masked, and to NOT have been
        ROI-masked yet (see module docstring for why ROI must come
        after this step, not before).
    background_sigma
        Per-pixel background noise standard deviation, in the same
        units as frame.image (ADU) -- the same quantity as
        analysis.noise_model.SensorNoiseModel.background_sigma.

    Returns
    -------
    ProcessedFrame
        Same image, frame_id, timestamp, exposure_us, and gain_db as the
        input -- only valid_columns is populated (or replaced).

    Raises
    ------
    ValueError
        If background_sigma is not positive -- a real, positive noise
        measurement is required; there's no meaningful SNR without one.
    '''

    if background_sigma <= 0:
        raise ValueError(f"background_sigma must be positive, got {background_sigma}")

    n_spatial_pixels = frame.image.shape[SPATIAL_AXIS]
    total_intensity = frame.image.sum(axis=SPATIAL_AXIS)

    noise_floor = np.sqrt(n_spatial_pixels) * background_sigma
    valid_columns = (total_intensity / noise_floor) >= SNR_THRESHOLD

    return ProcessedFrame(
        image=frame.image, frame_id=frame.frame_id, timestamp=frame.timestamp,
        exposure_us=frame.exposure_us, gain_db=frame.gain_db, valid_columns=valid_columns,
    )


__all__ = ["apply_signal_threshold", "SNR_THRESHOLD"]
