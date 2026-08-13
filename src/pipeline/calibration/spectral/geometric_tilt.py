"""
Measures and represents the imaging spectrometer's geometric tilt: a
row-dependent shift of the observed spectral column, at fixed true
wavelength, found by tracking several lamp lines' column centroid as a
function of spatial row (see scripts/measure_spectrometer_tilt.py, the
exploratory analysis this module productionizes).

That analysis found the row-dependence is NOT a simple shear (a single
px/row slope): every detected line traces the same non-monotonic,
jump-containing curve as a function of row, reproducible across repeat
shots -- a shared, wavelength-independent distortion, dominant by roughly
an order of magnitude over line-to-line differences. build_geometric_tilt()
below measures that shared curve directly (row_shift, one value per row)
by averaging every detected line's row-anchored displacement, plus a much
smaller residual: each line's own leftover slope after the shared curve
is subtracted, sampled only at the columns where a line was actually
detected (residual_slope_columns/residual_slope_values) -- interpolated,
not extrapolated, by GeometricTiltResult.column_shift().

Line identification here does NOT need wavelength -- unlike
line_matching.py's match_lines() (built for wavelength calibration), this
module only needs "a well-separated, sufficiently bright column-summed
peak" to serve as a positional fiducial, so it detects peaks directly via
scipy.signal.find_peaks rather than matching against reference_lines.py.

Per-row centroiding duplicates analysis/centroiding.py's intensity-
weighted-moment-plus-Thompson-Larson-Webb-uncertainty formula as a
private function here, rather than importing it, for the same reason
shared/fitting.py reimplements total-least-squares instead of importing
analysis/dispersion_fitting.py's version: calibration/ and analysis/ must
not depend on each other in either direction (see shared/fitting.py's own
module docstring). gain_e_per_adu/background_sigma are accepted as plain
floats rather than analysis/noise_model.py's SensorNoiseModel for the same
boundary reason.
"""

# Imports

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from pipeline.acquisition import CANONICAL_SHAPE, FrameData, SPATIAL_AXIS

from ..exceptions import LineMatchingError
from ..shared.fitting import PolynomialFitter, TotalLeastSquaresFit
from ..shared.io import load_artifact, save_artifact
from ..shared.metadata import CalibrationRecord

# Constants

logger = logging.getLogger(__name__)

# Column-summed-spectrum smoothing width, used only to locate peaks and
# their window bounds -- wide enough to ignore single-pixel noise wiggles
# without washing out genuine neighbouring lines. Matches
# scripts/measure_spectrometer_tilt.py's WINDOW_SMOOTHING_WIDTH.
SPECTRUM_SMOOTHING_WIDTH = 5

# A candidate peak must reach this fraction of the column-summed
# spectrum's maximum to be treated as a usable line -- unverified
# starting point, same treatment as line_matching.py's
# PEAK_PROMINENCE_FRACTION (flagged for review once more lamp data exists
# to tune against).
PEAK_HEIGHT_FRACTION = 0.15

# Minimum column separation between detected peaks, and the hard cap on
# how far a line's window can extend from its peak in either direction --
# both matching scripts/measure_spectrometer_tilt.py's tuned values for
# this spectrometer's line density/width.
MIN_PEAK_SEPARATION_PX = 15
MAX_WINDOW_HALF_WIDTH = 40

# Radius searched around a candidate peak for the column-summed
# spectrum's true local maximum before window-bound-walking starts --
# see scripts/measure_spectrometer_tilt.py's PEAK_SNAP_RADIUS for why
# this matters.
PEAK_SNAP_RADIUS = 5

# Number of columns at each end of a line's window used to estimate that
# row's local background level, subtracted before centroiding.
BACKGROUND_EDGE_COLUMNS = 2

# A row is excluded from a line's centroid-vs-row fit if its
# background-subtracted window intensity falls below this.
MIN_ROW_INTENSITY = 50.0

# Half-height of the row window (centred on the frame's middle row) every
# detected line is anchored to before averaging into the shared row-shape
# -- the frame's vertical centre is the most defensible fixed reference
# absent any per-dataset knowledge of which rows have the best coverage.
REFERENCE_ROW_HALF_WIDTH = 10

# Fewer than this many usable lines can't support both a meaningful
# shared-shape average and a per-line residual estimate.
MIN_LINES_REQUIRED = 3

# Row (x) uncertainty in the row-vs-centroid fits below carries no real
# measurement error of its own -- see analyze_raw_shot.py's
# PIXEL_COLUMN_SIGMA for the same convention.
PLACEHOLDER_ROW_SIGMA = 1e-3

# Placeholder Thompson-Larson-Webb noise parameters -- mirrors
# analysis/noise_model.py's own PLACEHOLDER_GAIN_E_PER_ADU/
# PLACEHOLDER_BACKGROUND_SIGMA (duplicated, not imported -- see module
# docstring). No real conversion-gain/baseline calibration exists yet for
# every camera this might run against; pass real values once one does.
PLACEHOLDER_GAIN_E_PER_ADU = 1.0
PLACEHOLDER_BACKGROUND_SIGMA = 0.0

# TLW's discretization term uses the detector's pixel size, "a". Every
# position here is already in pixel-index units, so a = 1 exactly -- see
# analysis/centroiding.py's own PIXEL_SIZE for the identical reasoning.
_PIXEL_SIZE = 1.0

# Classes

@dataclass(frozen=True, slots=True, eq=False)
class GeometricTiltResult:

    '''
    build_geometric_tilt()'s output. See module docstring for the
    shared-curve-plus-residual model this represents.

    Parameters
    ----------
    row_shift
        Shape (CANONICAL_SHAPE[0],). The shared row-dependent column
        shift, relative to reference_row (row_shift[reference_row] == 0
        by construction). Any gaps (rows no detected line had enough
        signal at) are filled by linear interpolation over the
        row index -- see build_geometric_tilt().
    reference_row
        The row row_shift is anchored to zero at.
    residual_slope_columns, residual_slope_values
        Each detected line's own leftover row-slope (px/row) after the
        shared curve is subtracted, at the column it was detected at --
        sorted ascending by column. Sparse (as few as MIN_LINES_REQUIRED
        points): only meant to be consumed through
        column_shift()/GeometricTiltResult's own interpolation, which
        holds the boundary value rather than extrapolating past the
        measured column range.
    record
        Tags this artifact with the settings/frame count of the lamp
        frames it was built from.
    '''

    row_shift: np.ndarray
    reference_row: int
    residual_slope_columns: np.ndarray
    residual_slope_values: np.ndarray
    record: CalibrationRecord

    def __post_init__(self) -> None:
        if self.row_shift.shape != (CANONICAL_SHAPE[SPATIAL_AXIS],):
            raise ValueError(
                f"row_shift must have shape ({CANONICAL_SHAPE[SPATIAL_AXIS]},), "
                f"got {self.row_shift.shape}"
            )
        if not (0 <= self.reference_row < CANONICAL_SHAPE[SPATIAL_AXIS]):
            raise ValueError(f"reference_row {self.reference_row} is outside the frame")
        if self.residual_slope_columns.shape != self.residual_slope_values.shape:
            raise ValueError(
                "residual_slope_columns and residual_slope_values must have the same shape, "
                f"got {self.residual_slope_columns.shape}, {self.residual_slope_values.shape}"
            )
        self.row_shift.flags.writeable = False
        self.residual_slope_columns.flags.writeable = False
        self.residual_slope_values.flags.writeable = False

    def column_shift(self, row: np.ndarray, col: np.ndarray, include_residual: bool = False) -> np.ndarray:

        '''
        The column shift to apply at (row, col): a true wavelength at
        this row is observed row_shift[row] (+ the interpolated residual
        term) columns away from where it would appear at reference_row.

        Parameters
        ----------
        row, col
            Broadcastable arrays of row/column indices (row need not be
            integer-valued if a caller wants a sub-row estimate; it's
            used only to index/interpolate, not as an array index
            directly).
        include_residual
            Whether to add the smaller, sparsely-sampled per-column
            residual term (interpolated across residual_slope_columns,
            held constant -- not extrapolated -- outside the measured
            range) on top of the shared row_shift. See module/class
            docstring for why this defaults to False.

        Returns
        -------
        np.ndarray
        '''

        rows_int = np.rint(np.asarray(row)).astype(int)
        shift = self.row_shift[rows_int]
        if include_residual:
            residual_slope = np.interp(col, self.residual_slope_columns, self.residual_slope_values)
            shift = shift + residual_slope * (row - self.reference_row)
        return shift


# Functions

def _find_window_bounds(smoothed_spectrum: np.ndarray, peak_col: int) -> tuple[int, int]:

    '''Same adaptive local-minimum walk as scripts/measure_spectrometer_tilt.py's find_window_bounds.'''

    n_columns = smoothed_spectrum.shape[0]

    snap_lo = max(0, peak_col - PEAK_SNAP_RADIUS)
    snap_hi = min(n_columns, peak_col + PEAK_SNAP_RADIUS + 1)
    peak_col = snap_lo + int(np.argmax(smoothed_spectrum[snap_lo:snap_hi]))

    col = peak_col
    while (
        col > 0 and col > peak_col - MAX_WINDOW_HALF_WIDTH
        and smoothed_spectrum[col - 1] <= smoothed_spectrum[col]
    ):
        col -= 1
    col_lo = col

    col = peak_col
    while (
        col < n_columns - 1 and col < peak_col + MAX_WINDOW_HALF_WIDTH
        and smoothed_spectrum[col + 1] <= smoothed_spectrum[col]
    ):
        col += 1
    col_hi = col + 1

    return col_lo, col_hi


def _intensity_weighted_centroid(
    column_intensities: np.ndarray, positions: np.ndarray,
    gain_e_per_adu: float, background_sigma: float,
) -> tuple[float, float]:

    '''
    Intensity-weighted first moment for position, full 3-term
    Thompson-Larson-Webb (2002) uncertainty -- duplicated from
    analysis/centroiding.py's IntensityWeightedMoment/
    _thompson_larson_webb_sigma rather than imported; see module
    docstring for why.
    '''

    total_intensity = column_intensities.sum()
    x0 = np.sum(positions * column_intensities) / total_intensity
    sigma_psf = np.sqrt(np.sum(column_intensities * (positions - x0) ** 2) / total_intensity)

    n_photons = total_intensity * gain_e_per_adu
    b_photons = background_sigma * gain_e_per_adu
    shot_term = sigma_psf ** 2 / n_photons
    pixelation_term = _PIXEL_SIZE ** 2 / (12 * n_photons)
    background_term = (
        (8 * np.pi * sigma_psf ** 4 * b_photons ** 2) / (_PIXEL_SIZE ** 2 * n_photons ** 2)
    )
    sigma_x0 = np.sqrt(shot_term + pixelation_term + background_term)

    return float(x0), float(sigma_x0)


def _row_centroids(
    image: np.ndarray, col_lo: int, col_hi: int, gain_e_per_adu: float, background_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''Same background-subtracted per-row centroiding as measure_spectrometer_tilt.py's row_centroids.'''

    col_indices = np.arange(col_lo, col_hi, dtype=np.float64)

    rows, x0_values, sigma_values = [], [], []
    for row in range(image.shape[0]):
        window = image[row, col_lo:col_hi]
        background = np.mean(
            np.concatenate((window[:BACKGROUND_EDGE_COLUMNS], window[-BACKGROUND_EDGE_COLUMNS:]))
        )
        subtracted = np.clip(window - background, 0.0, None)

        if subtracted.sum() < MIN_ROW_INTENSITY:
            continue

        x0, sigma_x0 = _intensity_weighted_centroid(
            subtracted, col_indices, gain_e_per_adu, background_sigma,
        )
        rows.append(row)
        x0_values.append(x0)
        sigma_values.append(sigma_x0)

    return (
        np.array(rows, dtype=np.float64),
        np.array(x0_values, dtype=np.float64),
        np.array(sigma_values, dtype=np.float64),
    )


def _fit_slope(
    rows: np.ndarray, x0: np.ndarray, sigma_x0: np.ndarray, fitter: PolynomialFitter,
) -> float:

    '''Degree-1 TotalLeastSquaresFit of x0 vs. rows -- returns just the slope (px/row).'''

    sigma_rows = np.full(rows.shape, PLACEHOLDER_ROW_SIGMA, dtype=np.float64)
    fit = fitter.fit(rows, x0, sigma_rows, sigma_x0, degree=1)
    return float(fit.coefficients[1])


def _measure_line_displacements(
    frames: list[FrameData], gain_e_per_adu: float, background_sigma: float,
) -> tuple[
    list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray],
    np.ndarray, np.ndarray, int, float, float,
]:

    '''
    Shared first stage of build_geometric_tilt() and
    build_geometric_tilt_linear(): validates the frames, detects lamp
    lines, centroids each one per row, anchors each line's displacement
    to zero at the reference row, and combines every line into a single
    per-row inverse-variance-weighted mean displacement. The two public
    functions diverge only in what they do with that per-row mean --
    build_geometric_tilt() interpolates across any row no line covered;
    build_geometric_tilt_linear() fits a straight line through it.

    Returns
    -------
    tuple
        (line_columns, line_rows, line_displacement, line_sigma,
        mean_displacement, weight_sum, reference_row, reference_exposure,
        reference_gain). mean_displacement and weight_sum have shape
        (CANONICAL_SHAPE[SPATIAL_AXIS],); both are NaN at any row no line
        covered.

    Raises
    ------
    ValueError
        If frames is empty, or any frame's exposure_us/gain_db differs
        from the first frame's (mirrors build_baseline()'s check).
    LineMatchingError
        If fewer than MIN_LINES_REQUIRED lines are detected with enough
        valid rows to fit -- same failure mode line_matching.py's
        match_lines() raises this for (too few usable peaks in the lamp
        image), reused here rather than duplicated.
    '''

    if len(frames) < 1:
        raise ValueError("requires at least 1 frame")

    reference_exposure = frames[0].exposure_us
    reference_gain = frames[0].gain_db
    for f in frames[1:]:
        if f.exposure_us != reference_exposure or f.gain_db != reference_gain:
            raise ValueError(
                "all frames must share identical exposure_us and gain_db -- got a mismatch "
                "against the first frame"
            )

    n_rows = CANONICAL_SHAPE[SPATIAL_AXIS]
    stacked = np.mean([f.image.astype(np.float64) for f in frames], axis=0)
    spectrum = stacked.sum(axis=0)
    smoothed_spectrum = uniform_filter1d(spectrum, size=SPECTRUM_SMOOTHING_WIDTH)

    peaks, _ = find_peaks(
        smoothed_spectrum, height=smoothed_spectrum.max() * PEAK_HEIGHT_FRACTION,
        distance=MIN_PEAK_SEPARATION_PX,
    )

    reference_row = n_rows // 2
    reference_lo = reference_row - REFERENCE_ROW_HALF_WIDTH
    reference_hi = reference_row + REFERENCE_ROW_HALF_WIDTH

    line_columns: list[int] = []
    line_rows: list[np.ndarray] = []
    line_displacement: list[np.ndarray] = []
    line_sigma: list[np.ndarray] = []

    for peak_col in peaks:
        col_lo, col_hi = _find_window_bounds(smoothed_spectrum, int(peak_col))
        rows, x0, sigma_x0 = _row_centroids(stacked, col_lo, col_hi, gain_e_per_adu, background_sigma)

        in_reference = (rows >= reference_lo) & (rows <= reference_hi)
        if rows.shape[0] < 2 or not in_reference.any():
            continue

        anchor = float(x0[in_reference].mean())
        line_columns.append(int(peak_col))
        line_rows.append(rows)
        line_displacement.append(x0 - anchor)
        line_sigma.append(sigma_x0)

    if len(line_columns) < MIN_LINES_REQUIRED:
        raise LineMatchingError(
            f"only {len(line_columns)} usable line(s) detected, need at least {MIN_LINES_REQUIRED} "
            "to build a geometric tilt calibration"
        )

    displacement_grid = np.full((len(line_columns), n_rows), np.nan)
    sigma_grid = np.full((len(line_columns), n_rows), np.nan)
    for i, (rows, displacement, sigma_x0) in enumerate(
        zip(line_rows, line_displacement, line_sigma)
    ):
        displacement_grid[i, rows.astype(int)] = displacement
        sigma_grid[i, rows.astype(int)] = sigma_x0

    # Inverse-variance weighted mean, not a plain average -- a row covered
    # by one bright, tightly-centroided line and one dim, noisy one should
    # trust the bright line's displacement more, not split the difference
    # evenly. weight_grid/displacement_grid/sigma_grid all share the same
    # NaN-for-"this line has no data at this row" pattern, so nansum
    # naturally excludes absent lines the same way nanmean used to; a row
    # with zero coverage from every line still comes out NaN (0/0), same
    # as nanmean's own behavior there, handled below the same way it
    # always was.
    with np.errstate(invalid="ignore", divide="ignore"):
        weight_grid = 1.0 / sigma_grid ** 2
        weight_sum = np.nansum(weight_grid, axis=0)
        mean_displacement = np.nansum(weight_grid * displacement_grid, axis=0) / weight_sum

    return (
        line_columns, line_rows, line_displacement, line_sigma,
        mean_displacement, weight_sum, reference_row, reference_exposure, reference_gain,
    )


def _fit_line_residuals(
    line_columns: list[int], line_rows: list[np.ndarray], line_displacement: list[np.ndarray],
    line_sigma: list[np.ndarray], row_shift: np.ndarray, fitter: PolynomialFitter,
) -> tuple[np.ndarray, np.ndarray]:

    '''Per-line residual slope (px/row) left over after row_shift is subtracted, sorted by column.'''

    residual_columns = []
    residual_slopes = []
    for col, rows, displacement, sigma_x0 in zip(
        line_columns, line_rows, line_displacement, line_sigma
    ):
        residual = displacement - row_shift[rows.astype(int)]
        residual_columns.append(col)
        residual_slopes.append(_fit_slope(rows, residual, sigma_x0, fitter))

    order = np.argsort(residual_columns)
    residual_slope_columns = np.array(residual_columns, dtype=np.float64)[order]
    residual_slope_values = np.array(residual_slopes, dtype=np.float64)[order]
    return residual_slope_columns, residual_slope_values


def build_geometric_tilt(
    frames: list[FrameData],
    gain_e_per_adu: float = PLACEHOLDER_GAIN_E_PER_ADU,
    background_sigma: float = PLACEHOLDER_BACKGROUND_SIGMA,
    fitter: PolynomialFitter | None = None,
) -> GeometricTiltResult:

    '''
    Measures the shared row-dependent geometric tilt from several lamp
    (or any narrow-line-source) calibration frames -- see module
    docstring for the method and scripts/measure_spectrometer_tilt.py for
    the exploratory analysis this productionizes.

    Parameters
    ----------
    frames
        Lamp-only calibration frames (no other light source), all
        captured under identical exposure_us/gain_db -- same requirement
        as calibration/sensor/baseline.py's build_baseline(). At least 1
        frame is accepted (unlike build_baseline(), nothing here needs a
        sample standard deviation across frames), but more frames improve
        each line's centroid SNR via stacking.
    gain_e_per_adu, background_sigma
        Sensor noise parameters for the Thompson-Larson-Webb centroid
        uncertainty (see module docstring's note on why this duplicates
        rather than imports analysis/centroiding.py's version). Feed
        directly into two places, not just one: each line's own per-row
        sigma_x0 (used, as before, to weight that line's residual-slope
        fit), and -- since real values are now threaded through by every
        real caller (calibration/spectral/workflow.py's
        run_spectral_calibration()) -- the inverse-variance weighting of
        the shared row_shift curve itself (see below), so a wrong noise
        estimate here now measurably skews row_shift, not just each
        line's reported uncertainty. Default to this module's own
        placeholders only for a caller with no real conversion-gain/
        baseline calibration to pass (e.g. a bring-up script against a
        brand new camera); pass real measured values whenever they exist.
    fitter
        PolynomialFitter for each line's row-vs-centroid fit. Defaults to
        TotalLeastSquaresFit.

    Returns
    -------
    GeometricTiltResult

    Raises
    ------
    ValueError
        If frames is empty, or any frame's exposure_us/gain_db differs
        from the first frame's (mirrors build_baseline()'s check).
    LineMatchingError
        If fewer than MIN_LINES_REQUIRED lines are detected with enough
        valid rows to fit -- same failure mode line_matching.py's
        match_lines() raises this for (too few usable peaks in the lamp
        image), reused here rather than duplicated.
    '''

    fitter = fitter if fitter is not None else TotalLeastSquaresFit()

    (
        line_columns, line_rows, line_displacement, line_sigma,
        mean_displacement, weight_sum, reference_row, reference_exposure, reference_gain,
    ) = _measure_line_displacements(frames, gain_e_per_adu, background_sigma)

    n_rows = CANONICAL_SHAPE[SPATIAL_AXIS]
    valid_rows = np.flatnonzero(~np.isnan(mean_displacement))
    row_shift = np.interp(np.arange(n_rows), valid_rows, mean_displacement[valid_rows])

    residual_slope_columns, residual_slope_values = _fit_line_residuals(
        line_columns, line_rows, line_displacement, line_sigma, row_shift, fitter,
    )

    record = CalibrationRecord(
        exposure_us=reference_exposure, gain_db=reference_gain,
        timestamp=time.time(), source_frame_count=len(frames),
    )
    return GeometricTiltResult(
        row_shift=row_shift, reference_row=reference_row,
        residual_slope_columns=residual_slope_columns, residual_slope_values=residual_slope_values,
        record=record,
    )


def build_geometric_tilt_linear(
    frames: list[FrameData],
    gain_e_per_adu: float = PLACEHOLDER_GAIN_E_PER_ADU,
    background_sigma: float = PLACEHOLDER_BACKGROUND_SIGMA,
    fitter: PolynomialFitter | None = None,
) -> GeometricTiltResult:

    '''
    Amendment to build_geometric_tilt() that replaces its per-row shared
    curve (raw inverse-variance-weighted mean, gaps filled by
    interpolation) with a single straight-line fit through that same
    per-row mean. Trades the ability to represent the curve's documented
    non-monotonic, jump-containing shape (see module docstring) for
    immunity to shot-noise-driven row-to-row jitter -- an explicit bet
    that, for the frames this is built from, a smooth linear model is
    closer to the truth than the raw per-row curve. Everything upstream
    of the shared-curve combination (line detection, per-row centroiding,
    per-line anchoring at the reference row, inverse-variance weighting)
    is identical to build_geometric_tilt() -- both call
    _measure_line_displacements() for that shared stage.

    The fit is weighted by each row's combined inverse-variance from that
    per-line combination (a row built from more/brighter line coverage
    pulls the line harder than one built from a single dim line) and uses
    only rows at least one line actually covered -- deliberately
    excluding the gap rows build_geometric_tilt() fills by interpolation,
    so no already-interpolated value is fed back into this fit.

    Parameters
    ----------
    frames, gain_e_per_adu, background_sigma
        See build_geometric_tilt().
    fitter
        PolynomialFitter for both the shared row_shift line and each
        line's residual slope. Defaults to TotalLeastSquaresFit.

    Returns
    -------
    GeometricTiltResult
        row_shift is exactly linear in row by construction, and re-
        anchored so row_shift[reference_row] == 0 exactly (the fitted
        line's own value there is only approximately zero -- every
        line's displacement was already anchored relative to
        reference_row upstream in _measure_line_displacements(), so the
        fit is shifted by a constant to respect that same anchor rather
        than let finite-sample noise offset it).

    Raises
    ------
    ValueError, LineMatchingError
        See build_geometric_tilt().
    InsufficientDataError
        If fewer than 3 rows have real line coverage -- a straight-line
        fit needs at least degree + 2 points (see shared/fitting.py);
        vanishingly unlikely once MIN_LINES_REQUIRED lines are detected,
        since each contributes many rows, but a real path unlike
        build_geometric_tilt()'s interpolation, which has no such floor.
    '''

    fitter = fitter if fitter is not None else TotalLeastSquaresFit()

    (
        line_columns, line_rows, line_displacement, line_sigma,
        mean_displacement, weight_sum, reference_row, reference_exposure, reference_gain,
    ) = _measure_line_displacements(frames, gain_e_per_adu, background_sigma)

    n_rows = CANONICAL_SHAPE[SPATIAL_AXIS]
    valid_rows = np.flatnonzero(~np.isnan(mean_displacement))

    row_sigma = np.sqrt(1.0 / weight_sum[valid_rows])
    sigma_rows = np.full(valid_rows.shape, PLACEHOLDER_ROW_SIGMA, dtype=np.float64)
    fit = fitter.fit(
        valid_rows.astype(np.float64), mean_displacement[valid_rows], sigma_rows, row_sigma, degree=1,
    )
    row_shift = np.polynomial.polynomial.polyval(np.arange(n_rows), fit.coefficients)
    row_shift = row_shift - row_shift[reference_row]

    residual_slope_columns, residual_slope_values = _fit_line_residuals(
        line_columns, line_rows, line_displacement, line_sigma, row_shift, fitter,
    )

    record = CalibrationRecord(
        exposure_us=reference_exposure, gain_db=reference_gain,
        timestamp=time.time(), source_frame_count=len(frames),
    )
    return GeometricTiltResult(
        row_shift=row_shift, reference_row=reference_row,
        residual_slope_columns=residual_slope_columns, residual_slope_values=residual_slope_values,
        record=record,
    )


def save_geometric_tilt(path: str | Path, result: GeometricTiltResult) -> None:

    '''
    Saves a geometric tilt artifact to path, so it can be reused in a
    later session without recapturing/refitting lamp lines. Overwrites
    whatever was already at path -- current instrument alignment, not a
    history to keep.

    Parameters
    ----------
    path
        Destination file. A ".npz" suffix is added if not already
        present.
    result
        The GeometricTiltResult returned by build_geometric_tilt().

    Returns
    -------
    None
    '''

    arrays = {
        "row_shift": result.row_shift,
        "reference_row": np.array(result.reference_row),
        "residual_slope_columns": result.residual_slope_columns,
        "residual_slope_values": result.residual_slope_values,
    }
    save_artifact(path, arrays, result.record)


def load_geometric_tilt(path: str | Path) -> GeometricTiltResult:

    '''
    Loads a geometric tilt calibration previously saved via
    save_geometric_tilt().

    Parameters
    ----------
    path
        The file to load. A ".npz" suffix is added if not already
        present.

    Returns
    -------
    GeometricTiltResult

    Raises
    ------
    FileNotFoundError
        If path doesn't exist.
    '''

    arrays, record = load_artifact(path, CalibrationRecord)
    logger.info("loaded geometric tilt calibration from %s (age %.1fs)", path, record.age_seconds)
    return GeometricTiltResult(
        row_shift=arrays["row_shift"], reference_row=int(arrays["reference_row"]),
        residual_slope_columns=arrays["residual_slope_columns"],
        residual_slope_values=arrays["residual_slope_values"],
        record=record,
    )


__all__ = [
    "GeometricTiltResult", "build_geometric_tilt", "build_geometric_tilt_linear",
    "save_geometric_tilt", "load_geometric_tilt",
]
