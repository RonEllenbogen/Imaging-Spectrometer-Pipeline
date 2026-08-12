"""
Applies a GeometricTiltResult (calibration/spectral/geometric_tilt.py) to
one frame: resamples each row so a column shift of row_shift[row] (plus,
optionally, the smaller per-column residual term) is undone -- the
"stretch the image until the lamp lines are vertical" correction that
calibration module's own docstring describes measuring.

This is the one preprocessing step that resamples rather than just
rescaling/masking pixel values in place, which has two consequences worth
being explicit about:

  - Interpolating across columns mixes neighbouring pixels' noise, which
    is not independent afterward -- a real, currently undocumented-further-
    than-here departure from the per-pixel-independent-noise assumption
    baked into analysis/centroiding.py's Thompson-Larson-Webb formula. Not
    solved here; flagged the same way other known approximations are
    tracked in this codebase (see docs/project_state.md).
  - A bad pixel zeroed by apply_bad_pixel_map() gets linearly interpolated
    across like any other value, rather than excluded from the
    interpolation -- acceptable given bad-pixel masking's current sparse,
    isolated-pixel design, but a simplification, not a validated non-issue.
    Run this step after apply_bad_pixel_map() (not before), so at least the
    zeroed value itself -- not an un-masked hot/dead pixel -- is what gets
    smeared.
"""

# Imports

import numpy as np
from scipy.ndimage import map_coordinates

from pipeline.calibration.spectral.geometric_tilt import GeometricTiltResult

from ..processed_frame import ProcessedFrame

# Constants

# Classes

# Functions


def apply_geometric_tilt_correction(
    frame: ProcessedFrame, tilt: GeometricTiltResult, include_residual: bool = False,
) -> ProcessedFrame:

    '''
    Resamples frame so that a truly row-independent (vertical) spectral
    line no longer drifts in column with row.

    Parameters
    ----------
    frame
        The frame to correct -- expected to already be baseline-
        subtracted, flat-field-divided, and bad-pixel-masked (see module
        docstring for why bad-pixel masking specifically should run
        first).
    tilt
        The GeometricTiltResult to apply.
    include_residual
        Whether to also apply the smaller, sparsely-sampled per-column
        residual term GeometricTiltResult.column_shift() supports.
        Defaults to False: that term is built from far fewer points, a
        model already shown (see scripts/measure_spectrometer_tilt.py) not
        to be a simple straight line, and holds its edge value rather than
        extrapolating outside the columns it was actually measured at --
        real, but on comparatively soft footing next to the dominant
        shared row_shift this defaults to applying alone.

    Returns
    -------
    ProcessedFrame
        Same shape, frame_id, timestamp, exposure_us, gain_db, and
        valid_columns as the input. Pixels shifted from outside the
        frame's column range are filled with 0, same convention as
        steps/roi.py's masking.
    '''

    n_rows, n_cols = frame.image.shape
    row_grid, col_grid = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

    shift = tilt.column_shift(row_grid, col_grid, include_residual=include_residual)
    source_columns = col_grid + shift

    import sys, time as _time
    print(
        f"[DIAG] frame.image shape={frame.image.shape} dtype={frame.image.dtype} "
        f"c_contig={frame.image.flags['C_CONTIGUOUS']} "
        f"finite={np.isfinite(frame.image).all()} "
        f"shift min={shift.min()} max={shift.max()} nan={np.isnan(shift).any()} "
        f"source_columns dtype={source_columns.dtype} c_contig={source_columns.flags['C_CONTIGUOUS']}",
        file=sys.stderr, flush=True,
    )
    _t0 = _time.time()

    corrected = map_coordinates(
        frame.image, [row_grid, source_columns], order=1, mode="constant", cval=0.0,
    )

    print(f"[DIAG] map_coordinates took {_time.time() - _t0:.4f}s", file=sys.stderr, flush=True)

    return ProcessedFrame(
        image=corrected, frame_id=frame.frame_id, timestamp=frame.timestamp,
        exposure_us=frame.exposure_us, gain_db=frame.gain_db, valid_columns=frame.valid_columns,
    )


__all__ = ["apply_geometric_tilt_correction"]
