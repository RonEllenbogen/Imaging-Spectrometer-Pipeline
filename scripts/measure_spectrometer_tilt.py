'''
Measures the imaging spectrometer's geometric tilt: for a set of identified
Argon calibration-lamp lines, fits the spectral-column centroid as a
function of spatial row, giving each line's shear slope (px/row) with a
formal uncertainty. Comparing the measured slopes across the field (the
line set spans ~750-842nm) answers whether the tilt is well described by a
single global shear or needs a column-dependent correction, before either
is subtracted from a real beam shot to isolate genuine spatial chirp.

Background
----------
data/raw/spectral_lamp_11.8.26/shot_{1,2,3}.bmp are three repeat exposures
of the Argon lamp alone (no laser). 12 known Argon lines between 750.39nm
and 842.46nm were matched by eye against the image, but automated
peak-finding on the column-summed spectrum only resolves 9 distinct bumps
in that range: three of the four closely-spaced doublets (<1.2nm apart)
blend into a single unresolved feature at this spectrometer's dispersion,
and only the widest doublet (840.82/842.46nm, 1.64nm apart) is resolved
into two separate peaks. LINE_GROUPS below reflects that -- the three
blended doublets are fit and reported as one line each, labelled with the
doublet's midpoint wavelength.

Method, per line
-----------------
- window bounds are found adaptively: starting from the peak column, walk
  outward until a (smoothed) local minimum of the column-summed spectrum,
  capped at MAX_HALF_WIDTH -- keeps each line's window clear of neighbours
  regardless of gap size (as small as 23px, for the one resolved doublet).
- for each spatial row, the window's per-row profile is background-
  subtracted (a local estimate from the window's own edge columns) and fed
  to analysis/centroiding.py's IntensityWeightedMoment, giving a centroid
  column x0 and its Thompson-Larson-Webb uncertainty sigma_x0.
  SensorNoiseModel uses the project's placeholder gain=1 e/ADU,
  background_sigma=0 (see analysis/noise_model.py) -- no real
  conversion-gain/baseline calibration exists for this camera yet, so
  sigma_x0's absolute scale is provisional, though its relative row-to-row
  weighting is still meaningful.
- rows whose background-subtracted window intensity is too low are
  dropped (MIN_ROW_INTENSITY).
- x0(row) is fit with calibration/shared/fitting.py's TotalLeastSquaresFit:
  row as x (a tiny placeholder sigma_x, since rows carry no real
  uncertainty), x0 as y (sigma_y = sigma_x0). The linear coefficient is
  the line's tilt slope, in px/row.

Cross-checks
------------
- the three shots are stacked (mean) for each line's headline slope and
  formal uncertainty -- justified since the three shots agree to ~2.7 DN
  mean pixel difference with no observed drift -- but are also fit
  independently, and the spread of the resulting 3 slopes (std/sqrt(3)) is
  reported as an empirical, model-free uncertainty alongside the formal
  one.
- a degree-2 fit is also run per line to flag any statistically
  significant curvature the linear model might be hiding.

Slope vs. column across the measured lines is then fit (degree 1) to
directly test whether the tilt depends on wavelength/column -- the
question motivating this script -- rather than eyeballing the per-line
table.

Shared row-shape
-----------------
The per-line linear fits above turn out to have reduced chi-squared as
high as ~55 and highly significant curvature (|z| up to ~49): x0(row) is
not a straight line for any of these lines. Plotting each line's
row-anchored displacement (relative to its own mean over
[REFERENCE_ROW_MIN, REFERENCE_ROW_MAX]) shows why -- a specific,
non-monotonic, jump-containing curve, not a smooth tilt. Critically, that
curve is the same shape (up to noise) for every one of the 9 lines
(spanning ~750-842nm) and reproduces pixel-for-pixel across all 3
independent shots, which rules out per-line noise or a shot-specific
fluke: it's a shared, wavelength-independent, highly reproducible
property of the imaging system.

build_common_shape() combines all 9 lines' row-anchored displacements
onto one row grid and computes their mean/std/count per row --
shared_shape_consistency() then checks whether that combined mean is
statistically consistent with each line's own data (normalized residual
z = deviation / sigma_x0; mean(z^2) ~ 1 if so). fit_residual_slope() fits
what's left of each line's displacement after subtracting the shared
mean curve, and fit_residual_column_trend() fits those leftover slopes
against column -- this, not the raw slope-vs-column fit above, is the
real test of whether a column-dependent correction is needed on top of
the dominant shared row-shape.

Usage:
    python scripts/measure_spectrometer_tilt.py
    python scripts/measure_spectrometer_tilt.py --output out.png
'''

# Imports

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from pipeline.acquisition import CANONICAL_DTYPE, CANONICAL_SHAPE
from pipeline.analysis.centroiding import IntensityWeightedMoment
from pipeline.analysis.noise_model import SensorNoiseModel
from pipeline.calibration.exceptions import InsufficientDataError
from pipeline.calibration.shared.fitting import TotalLeastSquaresFit
from pipeline.calibration.shared.result import PolynomialFitResult

# Constants

DEFAULT_SHOT_PATHS = tuple(
    Path("data/raw/spectral_lamp_11.8.26") / name
    for name in ("shot_1.bmp", "shot_2.bmp", "shot_3.bmp")
)
DEFAULT_OUTPUT_PATH = Path("data/processed/spectral_lamp_tilt.png")

# (peak column in the stacked shot's column-summed spectrum, label,
# representative wavelength_nm) for the 9 distinguishable bumps spanning
# the 12 Argon lines visually matched from 750.39nm to 842.46nm -- see
# module docstring for why 3 of the 9 are unresolved doublets, labelled
# with their midpoint wavelength.
LINE_GROUPS = [
    (236, "750.39 / 751.46 nm (blended)", 750.925),
    (456, "763.51 nm", 763.51),
    (613, "772.38 nm", 772.38),
    (1010, "794.82 nm", 794.82),
    (1126, "800.62 / 801.48 nm (blended)", 801.05),
    (1307, "810.37 / 811.53 nm (blended)", 810.95),
    (1573, "826.45 nm", 826.45),
    (1828, "840.82 nm", 840.82),
    (1851, "842.46 nm", 842.46),
]

# Column-summed-spectrum smoothing kernel width used only to pick window
# bounds (local minima either side of a peak) -- wide enough to ignore
# single-pixel noise wiggles (e.g. a small shoulder on col 613's falling
# wing, which is not a real line) without washing out genuine neighbouring
# features.
WINDOW_SMOOTHING_WIDTH = 5

# Cap on how far a line's window can extend from its peak in either
# direction, regardless of how far away the local minimum is -- keeps
# isolated lines' windows from including hundreds of columns of bare
# background.
MAX_HALF_WIDTH = 40

# Number of columns at each end of a line's window used to estimate that
# row's local background level, subtracted before centroiding.
BACKGROUND_EDGE_COLUMNS = 2

# A row is excluded from a line's centroid-vs-row fit if its
# background-subtracted window intensity falls below this -- too little
# signal for a trustworthy centroid.
MIN_ROW_INTENSITY = 50.0

# Row (x, in the per-line tilt fit) and column (x, in the slope-vs-column
# trend fit) carry no real measurement uncertainty of their own --
# placeholder small enough to be negligible next to the corresponding
# sigma_y, same convention as scripts/analyze_raw_shot.py's
# PIXEL_COLUMN_SIGMA.
PLACEHOLDER_X_SIGMA = 1e-3

# Placeholder sensor noise model (see analysis/noise_model.py) -- no real
# conversion-gain/baseline calibration exists for this camera yet.
NOISE_MODEL = SensorNoiseModel(gain_e_per_adu=1.0, background_sigma=0.0)

# Radius searched, around each LINE_GROUPS nominal column, for the actual
# local maximum of the stacked image's smoothed spectrum before walking
# outward -- the nominal columns come from shot_1's raw (unsmoothed)
# spectrum and can sit a pixel or two off the stacked/smoothed spectrum's
# true peak, which otherwise breaks find_window_bounds's outward walk on
# its very first step (a strictly-decreasing test that never gets going).
PEAK_SNAP_RADIUS = 5

# Row window every line is anchored to (displacement = 0) before comparing
# shapes across lines -- see build_common_shape(). Chosen because every
# LINE_GROUPS line has full row coverage there (checked by hand), clear of
# the low-row region where fainter lines lose rows to MIN_ROW_INTENSITY.
REFERENCE_ROW_MIN = 590
REFERENCE_ROW_MAX = 610

DEFAULT_SHAPE_OUTPUT_PATH = Path("data/processed/spectral_lamp_common_shape.png")

# Classes

@dataclass(frozen=True, slots=True, eq=False)
class LineTiltResult:

    '''
    Per-line tilt measurement: the fitted spectral-column-centroid-vs.-
    spatial-row slope, plus formal and empirical uncertainty estimates and
    the row-by-row data needed to plot it (see module docstring for
    method).

    Parameters
    ----------
    label, wavelength_nm
        Identification from LINE_GROUPS.
    peak_col
        Column the line's window was centred on.
    window
        (col_lo, col_hi), half-open, as used for every row/shot.
    rows, x0, sigma_x0
        Per-valid-row centroid data from the stacked image.
    linear_fit
        Degree-1 TotalLeastSquaresFit result of x0 vs. rows on the
        stacked image -- slope_stacked/slope_sigma_formal/
        reduced_chi_squared are read off this.
    slope_sigma_empirical
        std/sqrt(3) of the 3 independently-fit per-shot slopes.
    curvature_z
        |c2 / sigma(c2)| from a degree-2 fit on the stacked image --
        large values flag statistically significant curvature the linear
        model doesn't capture.
    per_shot_slopes
        The 3 independent per-shot slopes slope_sigma_empirical was
        derived from.
    '''

    label: str
    wavelength_nm: float
    peak_col: int
    window: tuple[int, int]
    rows: np.ndarray
    x0: np.ndarray
    sigma_x0: np.ndarray
    linear_fit: PolynomialFitResult
    slope_sigma_empirical: float
    curvature_z: float
    per_shot_slopes: tuple[float, ...]

    @property
    def n_valid_rows(self) -> int:
        return self.rows.shape[0]

    @property
    def slope_stacked(self) -> float:
        return float(self.linear_fit.coefficients[1])

    @property
    def slope_sigma_formal(self) -> float:
        return float(self.linear_fit.coefficient_sigma[1])

    @property
    def reduced_chi_squared(self) -> float:
        return self.linear_fit.reduced_chi_squared

    @property
    def slope_sigma_combined(self) -> float:
        '''Conservative combined uncertainty: the larger of the two estimates.'''
        return max(self.slope_sigma_formal, self.slope_sigma_empirical)


@dataclass(frozen=True, slots=True, eq=False)
class CommonShapeResult:

    '''
    Every measured line's row-anchored centroid displacement (see
    REFERENCE_ROW_MIN/MAX), combined onto one row grid -- tests whether a
    single shared function of row explains all of them, rather than each
    line needing its own independent tilt (see module docstring).

    Parameters
    ----------
    row_grid
        0..CANONICAL_SHAPE[0]-1.
    displacement_grid
        (n_lines, n_rows). displacement_grid[i, row] is line i's x0 at
        row minus its own reference-window anchor, or NaN where line i
        has no valid centroid at that row (see row_centroids).
    mean_displacement, std_displacement, n_lines
        Per-row nanmean / nanstd(ddof=1) / valid-line-count across
        displacement_grid's first axis. std_displacement is NaN wherever
        n_lines < 2 (undefined sample std).
    '''

    row_grid: np.ndarray
    displacement_grid: np.ndarray
    mean_displacement: np.ndarray
    std_displacement: np.ndarray
    n_lines: np.ndarray


# Functions

def load_shot(path: Path) -> np.ndarray:

    '''
    Loads one raw lamp-shot image as a float64 array, validated against
    the project's canonical frame contract.

    Raises
    ------
    ValueError
        If the image's shape or dtype doesn't match CANONICAL_SHAPE/
        CANONICAL_DTYPE.
    '''

    image = iio.imread(path)

    if image.shape != CANONICAL_SHAPE:
        raise ValueError(f"{path} has shape {image.shape}, expected {CANONICAL_SHAPE}")
    if image.dtype != CANONICAL_DTYPE:
        raise ValueError(f"{path} has dtype {image.dtype}, expected {CANONICAL_DTYPE}")

    return image.astype(np.float64)


def find_window_bounds(
    smoothed_spectrum: np.ndarray, peak_col: int, max_half_width: int
) -> tuple[int, int]:

    '''
    Walks outward from peak_col's true local maximum (within
    PEAK_SNAP_RADIUS -- see its own comment) in both directions while
    smoothed_spectrum keeps falling, stopping at the first local minimum
    (or max_half_width, or the array edge) on each side -- adapts each
    line's window to however close its actual neighbours are, rather than
    assuming a fixed half-width.

    Parameters
    ----------
    smoothed_spectrum
        Column-summed spectrum, pre-smoothed (see WINDOW_SMOOTHING_WIDTH)
        so single-pixel noise wiggles don't trigger an early stop.
    peak_col
        Nominal column to walk outward from (see PEAK_SNAP_RADIUS).
    max_half_width
        Hard cap on the distance walked in either direction, from the
        snapped peak.

    Returns
    -------
    tuple[int, int]
        (col_lo, col_hi), half-open -- suitable directly as a slice.
    '''

    n_columns = smoothed_spectrum.shape[0]

    snap_lo = max(0, peak_col - PEAK_SNAP_RADIUS)
    snap_hi = min(n_columns, peak_col + PEAK_SNAP_RADIUS + 1)
    peak_col = snap_lo + int(np.argmax(smoothed_spectrum[snap_lo:snap_hi]))

    col = peak_col
    while (
        col > 0 and col > peak_col - max_half_width
        and smoothed_spectrum[col - 1] <= smoothed_spectrum[col]
    ):
        col -= 1
    col_lo = col

    col = peak_col
    while (
        col < n_columns - 1 and col < peak_col + max_half_width
        and smoothed_spectrum[col + 1] <= smoothed_spectrum[col]
    ):
        col += 1
    col_hi = col + 1

    return col_lo, col_hi


def row_centroids(
    image: np.ndarray, col_lo: int, col_hi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''
    Background-subtracted, intensity-weighted centroid of image[:, col_lo:col_hi]
    for each row, via analysis/centroiding.py's IntensityWeightedMoment.

    Per-row local background is estimated from the window's own edge
    columns (BACKGROUND_EDGE_COLUMNS at each end) and subtracted with
    clipping at zero, mirroring preprocessing/steps/baseline.py's
    convention -- this script bypasses run_preprocessing() entirely (no
    baseline/flat-field calibration session exists yet for this camera),
    so that step has to happen here instead.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (rows, x0, sigma_x0) for rows whose background-subtracted window
        intensity is at least MIN_ROW_INTENSITY -- all other rows are
        dropped, not just masked.
    '''

    col_indices = np.arange(col_lo, col_hi, dtype=np.float64)
    estimator = IntensityWeightedMoment()

    rows, x0_values, sigma_values = [], [], []
    for row in range(image.shape[0]):
        window = image[row, col_lo:col_hi]
        background = np.mean(
            np.concatenate((window[:BACKGROUND_EDGE_COLUMNS], window[-BACKGROUND_EDGE_COLUMNS:]))
        )
        subtracted = np.clip(window - background, 0.0, None)

        if subtracted.sum() < MIN_ROW_INTENSITY:
            continue

        x0, sigma_x0 = estimator.estimate(subtracted, col_indices, NOISE_MODEL)
        rows.append(row)
        x0_values.append(x0)
        sigma_values.append(sigma_x0)

    return (
        np.array(rows, dtype=np.float64),
        np.array(x0_values, dtype=np.float64),
        np.array(sigma_values, dtype=np.float64),
    )


def fit_tilt(
    rows: np.ndarray, x0: np.ndarray, sigma_x0: np.ndarray, degree: int
) -> PolynomialFitResult:

    '''Thin wrapper around TotalLeastSquaresFit for x0(row): see module docstring.'''

    sigma_rows = np.full(rows.shape, PLACEHOLDER_X_SIGMA, dtype=np.float64)
    return TotalLeastSquaresFit().fit(rows, x0, sigma_rows, sigma_x0, degree=degree)


def analyze_line(
    stacked_image: np.ndarray, shot_images: list[np.ndarray],
    smoothed_spectrum: np.ndarray, peak_col: int, label: str, wavelength_nm: float,
) -> LineTiltResult | None:

    '''
    Full per-line measurement: window -> stacked-image fit (+ curvature
    check) -> per-shot fits -> empirical uncertainty. See module
    docstring.

    Returns
    -------
    LineTiltResult | None
        None if the stacked-image window doesn't have enough valid rows
        to fit (InsufficientDataError) -- printed as a warning by the
        caller rather than aborting the whole script over one bad line.
    '''

    col_lo, col_hi = find_window_bounds(smoothed_spectrum, peak_col, MAX_HALF_WIDTH)

    rows, x0, sigma_x0 = row_centroids(stacked_image, col_lo, col_hi)
    try:
        linear_fit = fit_tilt(rows, x0, sigma_x0, degree=1)
        quadratic_fit = fit_tilt(rows, x0, sigma_x0, degree=2)
    except InsufficientDataError as exc:
        print(f"warning: skipping {label} (col {peak_col}): {exc}")
        return None
    curvature_z = abs(quadratic_fit.coefficients[2] / quadratic_fit.coefficient_sigma[2])

    per_shot_slopes = []
    for shot_image in shot_images:
        shot_rows, shot_x0, shot_sigma = row_centroids(shot_image, col_lo, col_hi)
        shot_fit = fit_tilt(shot_rows, shot_x0, shot_sigma, degree=1)
        per_shot_slopes.append(float(shot_fit.coefficients[1]))
    slope_sigma_empirical = np.std(per_shot_slopes, ddof=1) / np.sqrt(len(per_shot_slopes))

    return LineTiltResult(
        label=label, wavelength_nm=wavelength_nm, peak_col=peak_col,
        window=(col_lo, col_hi), rows=rows, x0=x0, sigma_x0=sigma_x0,
        linear_fit=linear_fit, slope_sigma_empirical=float(slope_sigma_empirical),
        curvature_z=float(curvature_z), per_shot_slopes=tuple(per_shot_slopes),
    )


def fit_column_trend(results: list[LineTiltResult]) -> PolynomialFitResult:

    '''
    Degree-1 fit of each line's stacked-image slope against its column --
    the direct answer to "does the tilt depend on wavelength/column".
    Weighted by slope_sigma_combined (the more conservative of the formal
    and empirical per-line uncertainties), so a line whose two
    uncertainty estimates disagree doesn't get overweighted.
    '''

    columns = np.array([r.peak_col for r in results], dtype=np.float64)
    slopes = np.array([r.slope_stacked for r in results], dtype=np.float64)
    sigma_slopes = np.array([r.slope_sigma_combined for r in results], dtype=np.float64)
    sigma_columns = np.full(columns.shape, PLACEHOLDER_X_SIGMA, dtype=np.float64)

    return TotalLeastSquaresFit().fit(columns, slopes, sigma_columns, sigma_slopes, degree=1)


def reference_anchor(result: LineTiltResult) -> float:

    '''Mean x0 within [REFERENCE_ROW_MIN, REFERENCE_ROW_MAX] -- see those constants.'''

    in_reference = (result.rows >= REFERENCE_ROW_MIN) & (result.rows <= REFERENCE_ROW_MAX)
    return float(result.x0[in_reference].mean())


def build_common_shape(results: list[LineTiltResult]) -> CommonShapeResult:

    '''
    Anchors every line's x0(row) to 0 at the reference row window (see
    reference_anchor) and combines them onto one row grid -- see
    CommonShapeResult.
    '''

    n_rows = CANONICAL_SHAPE[0]
    row_grid = np.arange(n_rows, dtype=np.float64)
    displacement_grid = np.full((len(results), n_rows), np.nan)

    for i, result in enumerate(results):
        anchor = reference_anchor(result)
        row_indices = result.rows.astype(int)
        displacement_grid[i, row_indices] = result.x0 - anchor

    mean_displacement = np.nanmean(displacement_grid, axis=0)
    with np.errstate(invalid="ignore"):
        std_displacement = np.nanstd(displacement_grid, axis=0, ddof=1)
    n_lines = np.sum(~np.isnan(displacement_grid), axis=0)

    return CommonShapeResult(
        row_grid=row_grid, displacement_grid=displacement_grid,
        mean_displacement=mean_displacement, std_displacement=std_displacement,
        n_lines=n_lines,
    )


def shared_shape_consistency(
    results: list[LineTiltResult], shape: CommonShapeResult
) -> tuple[float, float]:

    '''
    Normalized residual of every line's row-anchored displacement against
    the shared mean_displacement, in units of that point's sigma_x0 --
    tests whether one common curve is statistically consistent with every
    line's actual data, not just visually similar.

    Returns
    -------
    tuple[float, float]
        (mean of z^2, fraction of points with |z| < 3) pooled across
        every (line, row) pair. mean(z^2) ~ 1 if the shared curve fully
        explains each line's scatter around it.
    '''

    z_values = []
    for result in results:
        anchor = reference_anchor(result)
        row_indices = result.rows.astype(int)
        displacement = result.x0 - anchor
        z_values.append((displacement - shape.mean_displacement[row_indices]) / result.sigma_x0)

    z_all = np.concatenate(z_values)
    return float(np.mean(z_all ** 2)), float(np.mean(np.abs(z_all) < 3.0))


def fit_residual_slope(result: LineTiltResult, shape: CommonShapeResult) -> PolynomialFitResult:

    '''
    Degree-1 fit of one line's row-anchored displacement (see
    reference_anchor) after subtracting the shared mean_displacement --
    isolates whatever's left over, per line, once the common row-shape is
    removed. A significant remaining slope here, not slope_stacked on the
    raw table, is the real test of whether a column-dependent correction
    is needed on top of the shared shape.
    '''

    anchor = reference_anchor(result)
    row_indices = result.rows.astype(int)
    residual = (result.x0 - anchor) - shape.mean_displacement[row_indices]
    return fit_tilt(result.rows, residual, result.sigma_x0, degree=1)


def fit_residual_column_trend(
    results: list[LineTiltResult], residual_fits: list[PolynomialFitResult]
) -> PolynomialFitResult:

    '''Same as fit_column_trend, but on the leftover slope after removing the shared row-shape.'''

    columns = np.array([r.peak_col for r in results], dtype=np.float64)
    slopes = np.array([f.coefficients[1] for f in residual_fits], dtype=np.float64)
    sigma_slopes = np.array([f.coefficient_sigma[1] for f in residual_fits], dtype=np.float64)
    sigma_columns = np.full(columns.shape, PLACEHOLDER_X_SIGMA, dtype=np.float64)

    return TotalLeastSquaresFit().fit(columns, slopes, sigma_columns, sigma_slopes, degree=1)


def print_report(results: list[LineTiltResult], trend_fit: PolynomialFitResult) -> None:

    '''Per-line table, then the slope-vs-column trend fit's headline numbers.'''

    header = (
        f"{'line':32} {'col':>5} {'rows':>5} {'slope (px/row)':>15} "
        f"{'formal sig':>11} {'empir. sig':>11} {'red. chi2':>10} {'|curv z|':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.label:32} {r.peak_col:5d} {r.n_valid_rows:5d} "
            f"{r.slope_stacked:15.5f} {r.slope_sigma_formal:11.5f} "
            f"{r.slope_sigma_empirical:11.5f} {r.reduced_chi_squared:10.3g} "
            f"{r.curvature_z:9.2f}"
        )
    print()

    slope_of_slope = trend_fit.coefficients[1]
    sigma_slope_of_slope = trend_fit.coefficient_sigma[1]
    z = slope_of_slope / sigma_slope_of_slope
    col_span = results[-1].peak_col - results[0].peak_col
    print("slope vs. column trend fit (degree 1):")
    print(
        f"  d(slope)/d(col) = {slope_of_slope:.4e} +/- {sigma_slope_of_slope:.4e} "
        f"(px/row)/px  (z = {z:.2f})"
    )
    print(f"  reduced chi-squared = {trend_fit.reduced_chi_squared:.3g}")
    print(
        f"  implied slope change across measured field "
        f"(col {results[0].peak_col}-{results[-1].peak_col}, {col_span}px): "
        f"{slope_of_slope * col_span:.4f} px/row"
    )


def plot_results(
    stacked_image: np.ndarray, results: list[LineTiltResult],
    trend_fit: PolynomialFitResult, output_path: Path,
) -> None:

    '''
    Top: stacked lamp image with each line's window, per-row centroids,
    and fitted line overlaid. Bottom: slope vs. column across all lines,
    with the fitted trend line.
    '''

    fig, (ax_image, ax_slope) = plt.subplots(2, 1, figsize=(12, 11), height_ratios=(2, 1))

    im = ax_image.imshow(stacked_image, cmap="viridis", origin="upper", aspect="auto")
    fig.colorbar(im, ax=ax_image, label="Stacked intensity (ADU, mean of 3 shots)")

    for r in results:
        col_lo, col_hi = r.window
        ax_image.axvspan(col_lo, col_hi, color="white", alpha=0.12, linewidth=0)
        ax_image.plot(r.x0, r.rows, ".", color="red", markersize=1.5, alpha=0.4)
        fit_rows = np.array([r.rows.min(), r.rows.max()])
        fit_x0 = np.polynomial.polynomial.polyval(fit_rows, r.linear_fit.coefficients)
        ax_image.plot(fit_x0, fit_rows, color="white", linewidth=1.2)

    ax_image.set_xlabel("Spectral pixel column")
    ax_image.set_ylabel("Spatial pixel row")
    ax_image.set_title(
        "Stacked lamp shot (mean of shot_1/2/3) -- line windows (shaded), "
        "row centroids (red), linear fits (white)"
    )

    columns = np.array([r.peak_col for r in results], dtype=np.float64)
    slopes = np.array([r.slope_stacked for r in results], dtype=np.float64)
    sigma_slopes = np.array([r.slope_sigma_combined for r in results], dtype=np.float64)

    ax_slope.errorbar(
        columns, slopes, yerr=sigma_slopes, fmt="o", color="steelblue",
        capsize=3, label="per-line slope (+/- combined sigma)",
    )
    fit_columns = np.linspace(columns.min(), columns.max(), 200)
    fit_slopes = np.polynomial.polynomial.polyval(fit_columns, trend_fit.coefficients)
    ax_slope.plot(fit_columns, fit_slopes, color="black", linewidth=1.2, label="degree-1 trend fit")
    ax_slope.axhline(
        float(np.mean(slopes)), color="gray", linestyle="--", linewidth=1, label="mean of 9 lines"
    )

    ax_slope.set_xlabel("Spectral pixel column")
    ax_slope.set_ylabel("Tilt slope (px/row)")
    ax_slope.set_title(
        f"Slope vs. column: d(slope)/d(col) = {trend_fit.coefficients[1]:.2e} "
        f"+/- {trend_fit.coefficient_sigma[1]:.2e} (px/row)/px"
    )
    ax_slope.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def print_common_shape_report(
    results: list[LineTiltResult], shape: CommonShapeResult,
    residual_fits: list[PolynomialFitResult], residual_trend_fit: PolynomialFitResult,
) -> None:

    '''Consistency of the shared row-shape across lines, then the residual-slope-vs-column test.'''

    mean_z2, frac_within_3sigma = shared_shape_consistency(results, shape)
    print("shared row-shape consistency (each line vs. the combined mean displacement):")
    print(f"  mean(z^2) = {mean_z2:.2f}  (z = per-row deviation / sigma_x0; ~1 if the shared")
    print("  curve fully explains each line's row-to-row scatter around it)")
    print(f"  fraction of points with |z| < 3 = {frac_within_3sigma * 100:.1f}%")
    print()

    header = f"{'line':32} {'col':>5} {'residual slope (px/row)':>24} {'sigma':>10} {'red. chi2':>10}"
    print("per-line residual slope, after removing the shared row-shape:")
    print(header)
    print("-" * len(header))
    for r, fit in zip(results, residual_fits):
        print(
            f"{r.label:32} {r.peak_col:5d} {fit.coefficients[1]:24.6f} "
            f"{fit.coefficient_sigma[1]:10.6f} {fit.reduced_chi_squared:10.3g}"
        )
    print()

    slope_of_slope = residual_trend_fit.coefficients[1]
    sigma_slope_of_slope = residual_trend_fit.coefficient_sigma[1]
    z = slope_of_slope / sigma_slope_of_slope
    col_span = results[-1].peak_col - results[0].peak_col
    print("residual slope vs. column trend fit (degree 1) -- whether a column-dependent")
    print("shear is needed ON TOP OF the shared row-shape:")
    print(
        f"  d(residual slope)/d(col) = {slope_of_slope:.4e} +/- {sigma_slope_of_slope:.4e} "
        f"(px/row)/px  (z = {z:.2f})"
    )
    print(f"  reduced chi-squared = {residual_trend_fit.reduced_chi_squared:.3g}")
    print(
        f"  implied residual slope change across measured field "
        f"(col {results[0].peak_col}-{results[-1].peak_col}, {col_span}px): "
        f"{slope_of_slope * col_span:.4f} px/row"
    )


def plot_common_shape(
    results: list[LineTiltResult], shape: CommonShapeResult,
    residual_fits: list[PolynomialFitResult], residual_trend_fit: PolynomialFitResult,
    output_path: Path,
) -> None:

    '''
    Top: every line's row-anchored displacement overlaid, plus the
    combined mean +/- std across lines -- visualizes whether one shared
    curve explains every wavelength. Bottom: the leftover per-line slope
    after removing that shared curve, vs. column, with its own trend fit.
    '''

    fig, (ax_shape, ax_residual) = plt.subplots(2, 1, figsize=(12, 11), height_ratios=(3, 2))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for r, color in zip(results, colors):
        anchor = reference_anchor(r)
        ax_shape.plot(
            r.rows, r.x0 - anchor, ".", color=color, markersize=1.5, alpha=0.5, label=r.label
        )

    valid = shape.n_lines >= 2
    ax_shape.plot(
        shape.row_grid[valid], shape.mean_displacement[valid],
        color="black", linewidth=1.5, label="combined mean (>=2 lines)",
    )
    ax_shape.fill_between(
        shape.row_grid[valid],
        shape.mean_displacement[valid] - shape.std_displacement[valid],
        shape.mean_displacement[valid] + shape.std_displacement[valid],
        color="black", alpha=0.15, linewidth=0, label="+/- std across lines",
    )
    ax_shape.axvspan(
        REFERENCE_ROW_MIN, REFERENCE_ROW_MAX, color="red", alpha=0.15, linewidth=0,
        label="reference window (anchor = 0)",
    )
    ax_shape.set_xlabel("Spatial pixel row")
    ax_shape.set_ylabel("Column displacement from reference row (px)")
    ax_shape.set_title(
        f"Every line's centroid trace, anchored to 0 at rows {REFERENCE_ROW_MIN}-{REFERENCE_ROW_MAX} -- "
        "overlap tests whether one shared\nrow-dependent distortion (not a simple shear) explains all wavelengths"
    )
    ax_shape.legend(fontsize=6, ncol=2, loc="upper left")

    columns = np.array([r.peak_col for r in results], dtype=np.float64)
    residual_slopes = np.array([f.coefficients[1] for f in residual_fits], dtype=np.float64)
    residual_sigma = np.array([f.coefficient_sigma[1] for f in residual_fits], dtype=np.float64)

    ax_residual.errorbar(
        columns, residual_slopes, yerr=residual_sigma, fmt="o", color="steelblue",
        capsize=3, label="per-line residual slope",
    )
    fit_columns = np.linspace(columns.min(), columns.max(), 200)
    fit_slopes = np.polynomial.polynomial.polyval(fit_columns, residual_trend_fit.coefficients)
    ax_residual.plot(fit_columns, fit_slopes, color="black", linewidth=1.2, label="degree-1 trend fit")
    ax_residual.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax_residual.set_xlabel("Spectral pixel column")
    ax_residual.set_ylabel("Residual slope (px/row)")
    ax_residual.set_title(
        "Leftover per-line slope after removing the shared row-shape -- "
        "tests for a genuine column-dependent shear on top of it"
    )
    ax_residual.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots", type=Path, nargs=3, default=list(DEFAULT_SHOT_PATHS),
        help="Three repeat lamp-only shots (default: data/raw/spectral_lamp_11.8.26/shot_{1,2,3}.bmp)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Where to save the per-line tilt-fit summary plot PNG",
    )
    parser.add_argument(
        "--shape-output", type=Path, default=DEFAULT_SHAPE_OUTPUT_PATH,
        help="Where to save the shared row-shape summary plot PNG",
    )
    args = parser.parse_args()

    shot_images = [load_shot(path) for path in args.shots]
    stacked_image = np.mean(shot_images, axis=0)

    spectrum = stacked_image.sum(axis=0)
    smoothed_spectrum = uniform_filter1d(spectrum, size=WINDOW_SMOOTHING_WIDTH)

    results = [
        result for peak_col, label, wavelength_nm in LINE_GROUPS
        if (result := analyze_line(
            stacked_image, shot_images, smoothed_spectrum, peak_col, label, wavelength_nm
        )) is not None
    ]
    trend_fit = fit_column_trend(results)

    print_report(results, trend_fit)
    plot_results(stacked_image, results, trend_fit, args.output)
    print()

    shape = build_common_shape(results)
    residual_fits = [fit_residual_slope(r, shape) for r in results]
    residual_trend_fit = fit_residual_column_trend(results, residual_fits)

    print_common_shape_report(results, shape, residual_fits, residual_trend_fit)
    plot_common_shape(results, shape, residual_fits, residual_trend_fit, args.shape_output)


if __name__ == "__main__":
    main()
