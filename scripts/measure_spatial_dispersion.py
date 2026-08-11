'''
Measures the beam's spatial dispersion (spatial pixel position x0 vs.
wavelength_nm) from data/raw/khz/11.8.26-2.bmp, using two real calibration
artifacts built earlier this session: the geometric tilt correction
(calibration/spectral/geometric_tilt.py) and the pixel -> wavelength_nm
spectral calibration (calibration/spectral/calibrate.py) built from it
(see scripts/build_geometric_tilt_calibration.py and
scripts/build_spectral_calibration.py). Fits degree 1 (linear), 2
(quadratic), and 3 (cubic) polynomials and compares their reduced
chi-squared, the same "does a higher-order term actually help" check
scripts/analyze_raw_shot.py's --degrees flag exists for -- except that
script used PixelColumnAxis (pixel column standing in for wavelength,
px/px units) because no real spectral calibration existed yet. This one
does, so the fit's independent variable is real wavelength_nm and zeta
comes out in the project's eventual px/nm convention.

Like analyze_raw_shot.py, this deliberately skips run_preprocessing()
entirely -- no baseline/flat-field/bad-pixel-map calibration session has
been run for this camera yet -- so the raw frame is wrapped directly into
a ProcessedFrame. The ONLY correction applied is geometric tilt (see
module docstring's note on why that has to happen before centroid
extraction, not after: preprocessing/steps/geometric_tilt.py's module
docstring), which this script does apply, unlike analyze_raw_shot.py.
That means sigma_x0 (and everything derived from it: reduced_chi_squared,
coefficient_sigma) is still only as good as the placeholder
SensorNoiseModel (gain=1 e/ADU, background_sigma=0) -- not a final
calibrated uncertainty, same caveat as analyze_raw_shot.py's.

Usage:
    python scripts/measure_spatial_dispersion.py
    python scripts/measure_spatial_dispersion.py --col-min 501 --col-max 1700
'''

# Imports

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from pipeline.acquisition import CANONICAL_DTYPE, CANONICAL_SHAPE
from pipeline.analysis import analyze_shot
from pipeline.analysis.results import ShotAnalysisResult
from pipeline.calibration.spectral import load_spectral_calibration
from pipeline.calibration.spectral.geometric_tilt import load_geometric_tilt
from pipeline.preprocessing import ProcessedFrame
from pipeline.preprocessing.steps import apply_geometric_tilt_correction

# Constants

DEFAULT_IMAGE_PATH = Path("data/raw/khz/11.8.26-2.bmp")
DEFAULT_TILT_PATH = Path("data/processed/spectral_lamp_geometric_tilt.npz")
DEFAULT_SPECTRAL_PATH = Path("data/processed/spectral_lamp_wavelength_calibration.npz")
DEFAULT_OUTPUT_PATH = Path("data/processed/khz_spatial_dispersion.png")
DEFAULT_DEGREES = (1, 2, 3)

# Columns outside this range are excluded from centroiding/fitting via
# ProcessedFrame.valid_columns -- same range scripts/analyze_raw_shot.py
# uses for the sibling capture 11.8.26.bmp (this file, 11.8.26-2.bmp, is
# from the same session/setup; a quick column-summed-intensity check
# confirms the real beam signal here also sits comfortably inside
# [501, 1700), well clear of both edges).
DEFAULT_VALID_COL_MIN = 501
DEFAULT_VALID_COL_MAX = 1700

# Classes

# Functions

def load_raw_processed_frame(image_path: Path, col_min: int, col_max: int) -> ProcessedFrame:

    '''
    Loads a raw image file and wraps it directly as a ProcessedFrame,
    skipping run_preprocessing() entirely -- see module docstring.

    Parameters
    ----------
    image_path
        Path to the raw image file.
    col_min, col_max
        Spectral columns outside [col_min, col_max) are marked invalid
        via valid_columns -- see analyze_raw_shot.py's load_raw_frame()
        for the same convention this mirrors.

    Returns
    -------
    ProcessedFrame

    Raises
    ------
    ValueError
        If the image's shape or dtype doesn't match CANONICAL_SHAPE/
        CANONICAL_DTYPE.
    '''

    image = iio.imread(image_path)

    if image.shape != CANONICAL_SHAPE:
        raise ValueError(f"{image_path} has shape {image.shape}, expected {CANONICAL_SHAPE}")
    if image.dtype != CANONICAL_DTYPE:
        raise ValueError(f"{image_path} has dtype {image.dtype}, expected {CANONICAL_DTYPE}")

    n_columns = image.shape[1]
    columns = np.arange(n_columns)
    valid_columns = (columns >= col_min) & (columns < col_max)

    return ProcessedFrame(
        image=image.astype(np.float64), frame_id=0, timestamp=0.0,
        exposure_us=19.0, gain_db=0.0, valid_columns=valid_columns,
    )


def print_report(result: ShotAnalysisResult, wavelength_axis) -> None:

    '''Per-degree fit coefficients/zeta/goodness-of-fit, then a reduced-chi-squared comparison.'''

    centroids = result.centroids
    print(f"Frame {result.frame_id}: {centroids.columns.shape[0]} valid columns")
    print()

    # Median column is enough to pick one representative evaluation point for
    # zeta -- it only varies meaningfully with wavelength once degree > 1.
    center_column = np.array([np.median(centroids.columns)])
    center_wavelength_nm = wavelength_axis.wavelength_nm(center_column)

    for degree, fit in sorted(result.fits.items()):
        print(f"--- degree {degree} ---")
        for k, (c, sigma_c) in enumerate(zip(fit.coefficients, fit.coefficient_sigma)):
            print(f"  c{k} = {c:.6g} +/- {sigma_c:.3g}")
        zeta = fit.zeta(center_wavelength_nm)[0]
        label = "zeta" if degree == 1 else f"zeta(lambda={center_wavelength_nm[0]:.1f}nm)"
        print(f"  {label} = {zeta:.6g} px/nm")
        print(f"  reduced chi-squared = {fit.reduced_chi_squared:.4g}")
        print()

    print("reduced chi-squared comparison:")
    for degree, fit in sorted(result.fits.items()):
        print(f"  degree {degree}: {fit.reduced_chi_squared:.4g}")


def resample_to_wavelength_grid(image: np.ndarray, wavelength_axis) -> tuple[np.ndarray, np.ndarray]:

    '''
    Resamples image's columns from pixel-index space onto a uniform
    wavelength grid via wavelength_axis's pixel -> wavelength_nm mapping.
    Needed for plot_result()'s heatmap to be honestly labelled "wavelength"
    on its x-axis -- relabelling pixel-index tick marks alone would only
    be correct if the calibration were exactly linear; this is exact
    regardless of the fitted degree, by construction.

    Parameters
    ----------
    image
        2D array, columns in pixel-index order (e.g. an already
        geometric-tilt-corrected frame).
    wavelength_axis
        Anything with a wavelength_nm(pixel) method, e.g. a loaded
        WavelengthCalibrationResult.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (wavelength_grid, resampled_image) -- wavelength_grid is
        ascending and evenly spaced; resampled_image has the same shape
        as image, with resampled_image[:, i] corresponding to
        wavelength_grid[i].
    '''

    n_rows, n_columns = image.shape
    columns = np.arange(n_columns, dtype=np.float64)
    wavelength_per_column = wavelength_axis.wavelength_nm(columns)

    # np.interp requires ascending x -- sort once here rather than assume
    # wavelength increases with column (grating orientation isn't fixed,
    # see line_matching.py's own module docstring on this same point).
    order = np.argsort(wavelength_per_column)
    sorted_wavelength = wavelength_per_column[order]
    wavelength_grid = np.linspace(sorted_wavelength[0], sorted_wavelength[-1], n_columns)

    resampled_image = np.empty_like(image)
    for row in range(n_rows):
        resampled_image[row] = np.interp(wavelength_grid, sorted_wavelength, image[row, order])

    return wavelength_grid, resampled_image


def plot_result(
    image: np.ndarray, wavelength_axis, result: ShotAnalysisResult,
    col_min: int, col_max: int, output_path: Path,
) -> None:

    '''
    Heatmap with centroids overlaid (top), then one residuals panel per
    fitted degree -- every panel shares a genuine wavelength x-axis (see
    resample_to_wavelength_grid()), so the heatmap, centroids, and
    residuals are all directly comparable, not just the bottom panels.
    '''

    centroids = result.centroids
    degrees = sorted(result.fits)

    wavelength_grid, wavelength_image = resample_to_wavelength_grid(image, wavelength_axis)
    centroid_wavelength_nm = wavelength_axis.wavelength_nm(centroids.columns)
    col_min_wavelength_nm, col_max_wavelength_nm = sorted(
        wavelength_axis.wavelength_nm(np.array([col_min, col_max])).tolist()
    )

    fig, axes = plt.subplots(
        1 + len(degrees), 1, figsize=(11, 6 + 2.2 * len(degrees)),
        height_ratios=(3,) + (1,) * len(degrees), sharex=False,
    )
    ax_image = axes[0]

    extent = (wavelength_grid[0], wavelength_grid[-1], image.shape[0], 0)
    im = ax_image.imshow(wavelength_image, cmap="viridis", origin="upper", aspect="auto", extent=extent)
    fig.colorbar(im, ax=ax_image, label="Intensity (ADU, tilt-corrected)")

    if col_min > 0:
        ax_image.axvspan(
            wavelength_grid[0], col_min_wavelength_nm, color="black", alpha=0.35, linewidth=0, zorder=2.5
        )
    if col_max < image.shape[1]:
        ax_image.axvspan(
            col_max_wavelength_nm, wavelength_grid[-1], color="black", alpha=0.35, linewidth=0, zorder=2.5
        )

    ax_image.errorbar(
        centroid_wavelength_nm, centroids.x0, yerr=centroids.sigma_x0,
        fmt="o", color="red", markersize=2, elinewidth=0.5, alpha=0.6,
        label="Centroid (x0 +/- sigma_x0)",
    )
    ax_image.set_xlim(wavelength_grid[0], wavelength_grid[-1])
    ax_image.set_ylabel("Spatial pixel row")
    ax_image.set_xlabel("Wavelength (nm)")
    ax_image.set_title(
        f"{DEFAULT_IMAGE_PATH} -- geometric-tilt-corrected, no baseline/flat-field\n"
        f"shaded = excluded from analysis (cols < {col_min} or >= {col_max})"
    )
    ax_image.legend(loc="upper right", fontsize=8)

    for ax, degree in zip(axes[1:], degrees):
        fit = result.fits[degree]
        ax.errorbar(
            centroid_wavelength_nm, fit.residuals, yerr=centroids.sigma_x0,
            fmt=".", color="steelblue", markersize=3, elinewidth=0.5, alpha=0.6,
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Residual (px)")
        ax.set_title(f"degree-{degree} residuals (reduced chi-sq = {fit.reduced_chi_squared:.3g})")

    axes[-1].set_xlabel("Wavelength (nm)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, nargs="?", default=DEFAULT_IMAGE_PATH,
                         help="Path to the raw beam-shot image file")
    parser.add_argument("--tilt", type=Path, default=DEFAULT_TILT_PATH,
                         help="Geometric tilt calibration .npz")
    parser.add_argument("--spectral", type=Path, default=DEFAULT_SPECTRAL_PATH,
                         help="Spectral (pixel -> wavelength_nm) calibration .npz")
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES),
                         help="Polynomial degrees to fit and compare (default: 1 2 3)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                         help="Where to save the plot PNG")
    parser.add_argument("--col-min", type=int, default=DEFAULT_VALID_COL_MIN,
                         help="First spectral column included in analysis (default: 501)")
    parser.add_argument("--col-max", type=int, default=DEFAULT_VALID_COL_MAX,
                         help="First spectral column excluded again onward (default: 1700)")
    parser.add_argument("--include-residual", action="store_true",
                         help="Also apply the geometric tilt calibration's smaller per-column "
                              "residual term (default: off -- see calibration/spectral/geometric_tilt.py)")
    args = parser.parse_args()

    tilt = load_geometric_tilt(args.tilt)
    spectral = load_spectral_calibration(args.spectral)

    raw_frame = load_raw_processed_frame(args.image_path, col_min=args.col_min, col_max=args.col_max)
    frame = apply_geometric_tilt_correction(raw_frame, tilt, include_residual=args.include_residual)

    result = analyze_shot(frame, spectral, degrees=tuple(args.degrees))

    print_report(result, spectral)

    plot_result(
        frame.image, spectral, result,
        col_min=args.col_min, col_max=args.col_max, output_path=args.output,
    )


if __name__ == "__main__":
    main()
