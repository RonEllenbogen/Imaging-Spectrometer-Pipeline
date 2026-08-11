'''
Runs a single manually-captured raw frame through analysis/ only (no
preprocessing/calibration/) and plots the result: raw heatmap with
per-column centroids overlaid, plus the fitted spatial-dispersion line(s)
and a residuals panel.

Deliberately skips run_preprocessing() -- there is no baseline/flat-field/
bad-pixel-map calibration session finished yet (see docs/project_state.md
Sec.0), so this script wraps the raw frame directly into a ProcessedFrame
and hands it straight to analyze_shot(). That means no baseline
subtraction, no flat-fielding, no bad-pixel masking, and the placeholder
SensorNoiseModel (gain=1 e/ADU, background_sigma=0) -- so sigma_x0 and
anything derived from it (reduced chi-squared, coefficient sigmas) is only
as good as those placeholders, not a final calibrated uncertainty.

There is also no spectral (wavelength) calibration yet, so the fit's
independent variable is pixel column, not wavelength_nm -- zeta comes out
in px/px (spatial pixels per spectral-column pixel), the same
"slope_px_per_col" quantity SyntheticBackend injects for tests, not the
project's eventual px/nm convention.

Usage:
    python scripts/analyze_raw_shot.py data/raw/khz/11.8.26.bmp
    python scripts/analyze_raw_shot.py data/raw/khz/11.8.26.bmp --degrees 1 2 3 --output out.png
    python scripts/analyze_raw_shot.py data/raw/khz/11.8.26.bmp --col-min 501 --col-max 1700
'''

# Imports

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from pipeline.acquisition import CANONICAL_DTYPE, CANONICAL_SHAPE
from pipeline.analysis import analyze_shot
from pipeline.analysis.results import ShotAnalysisResult
from pipeline.preprocessing import ProcessedFrame
from pipeline.utils.helpers import load_config

# Constants

DEFAULT_DEGREES = (1, 2, 3)
DEFAULT_OUTPUT_PATH = Path("data/processed/khz_analysis.png")

# Columns outside this range are excluded from centroiding/fitting via
# ProcessedFrame.valid_columns -- 11.8.26.bmp's edges (cols 0-500 and
# 1700 onward) carry no real beam signal, just uncalibrated background,
# and were visibly dragging the centroid trace off the beam.
DEFAULT_VALID_COL_MIN = 501
DEFAULT_VALID_COL_MAX = 1700

# analyze_shot()'s wavelength-axis abstraction requires a strictly
# positive sigma on both axes (scipy.odr divides by it). Pixel column has
# no real uncertainty of its own here -- this is a placeholder small
# enough to be negligible next to sigma_x0, not a measured quantity.
PIXEL_COLUMN_SIGMA = 1e-3

# Classes

class PixelColumnAxis:

    '''
    Stand-in WavelengthAxis for use before calibration/spectral/ has a
    real wavelength calibration: treats pixel column index itself as the
    fit's independent variable, in place of wavelength_nm. See module
    docstring for the resulting unit change (px/px, not px/nm).
    '''

    def wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:
        return pixel.astype(np.float64)

    def sigma_wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:
        return np.full(pixel.shape, PIXEL_COLUMN_SIGMA, dtype=np.float64)


# Functions

def load_raw_frame(image_path: Path, col_min: int, col_max: int) -> ProcessedFrame:

    '''
    Loads a raw image file and wraps it directly as a ProcessedFrame,
    skipping run_preprocessing() entirely (see module docstring).

    Parameters
    ----------
    image_path
        Path to the raw image file (e.g. a .bmp capture).
    col_min, col_max
        Spectral columns outside [col_min, col_max) are marked invalid
        via valid_columns, so extract_centroids() (analysis/centroiding.py)
        skips them entirely -- the same gate signal-threshold masking
        would normally populate, here set by hand since that preprocessing
        step hasn't run.

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
        raise ValueError(
            f"{image_path} has shape {image.shape}, expected {CANONICAL_SHAPE}"
        )
    if image.dtype != CANONICAL_DTYPE:
        raise ValueError(
            f"{image_path} has dtype {image.dtype}, expected {CANONICAL_DTYPE}"
        )

    config = load_config("configs/default.yaml")

    n_columns = image.shape[1]
    columns = np.arange(n_columns)
    valid_columns = (columns >= col_min) & (columns < col_max)

    # exposure_us/gain_db are metadata ProcessedFrame requires but this
    # frame never carried -- it was captured manually, outside
    # CameraStream. Placeholders only (config's default exposure, 0dB
    # gain), not the frame's real acquisition settings.
    return ProcessedFrame(
        image=image.astype(np.float64),
        frame_id=0,
        timestamp=time.monotonic(),
        exposure_us=float(config["camera"]["exposure_time"]),
        gain_db=0.0,
        valid_columns=valid_columns,
    )


def print_report(result: ShotAnalysisResult) -> None:

    '''Prints per-degree fit coefficients, zeta, and goodness-of-fit.'''

    centroids = result.centroids
    print(f"Frame {result.frame_id}: {centroids.columns.shape[0]} valid columns")
    print()

    for degree, fit in sorted(result.fits.items()):
        print(f"--- degree {degree} ---")
        for k, (c, sigma_c) in enumerate(zip(fit.coefficients, fit.coefficient_sigma)):
            print(f"  c{k} = {c:.6g} +/- {sigma_c:.3g}")
        center_column = np.array([np.median(centroids.columns)])
        zeta = fit.zeta(center_column)[0]
        sigma_zeta = fit.sigma_zeta(center_column)[0]
        label = "zeta" if degree == 1 else f"zeta(col={center_column[0]:.0f})"
        print(f"  {label} = {zeta:.6g} +/- {sigma_zeta:.3g} px/px")
        print(f"  reduced chi-squared = {fit.reduced_chi_squared:.4g}")
        print()


def plot_result(
    image: np.ndarray, result: ShotAnalysisResult, degree_for_line: int,
    col_min: int, col_max: int, output_path: Path,
) -> None:

    '''
    Heatmap of the raw frame with centroids overlaid (top), residuals of
    the requested degree's fit (bottom). Columns excluded from analysis
    (outside [col_min, col_max)) are shaded on both panels.

    Parameters
    ----------
    image
        Raw 2D frame, CANONICAL_SHAPE.
    result
        analyze_shot() output.
    degree_for_line
        Which fit's line/residuals to draw.
    col_min, col_max
        Same range passed to load_raw_frame() -- drawn as shaded bands
        for context, not recomputed from result (which only ever sees the
        already-excluded columns).
    output_path
        Where to save the figure (PNG).
    '''

    centroids = result.centroids
    fit = result.fits[degree_for_line]

    fig, (ax_image, ax_resid) = plt.subplots(
        2, 1, figsize=(11, 8), height_ratios=(3, 1), sharex=True
    )

    im = ax_image.imshow(image, cmap="viridis", origin="upper", aspect="auto")
    fig.colorbar(im, ax=ax_image, label="Raw intensity (ADU)")

    n_columns = image.shape[1]
    for ax in (ax_image, ax_resid):
        if col_min > 0:
            ax.axvspan(0, col_min, color="black", alpha=0.35, linewidth=0, zorder=2.5)
        if col_max < n_columns:
            ax.axvspan(col_max, n_columns, color="black", alpha=0.35, linewidth=0, zorder=2.5)

    ax_image.errorbar(
        centroids.columns, centroids.x0, yerr=centroids.sigma_x0,
        fmt="o", color="red", markersize=2, elinewidth=0.5, alpha=0.6,
        label="Centroid (x0 +/- sigma_x0)",
    )

    fit_columns = np.linspace(centroids.columns.min(), centroids.columns.max(), 200)
    fit_x0 = np.polynomial.polynomial.polyval(fit_columns, fit.coefficients)
    ax_image.plot(
        fit_columns, fit_x0, color="white", linewidth=1.5,
        label=f"degree-{degree_for_line} fit",
    )

    ax_image.set_ylabel("Spatial pixel row")
    ax_image.set_title(
        "data/raw/khz/11.8.26.bmp -- raw heatmap + centroids (no calibration applied)\n"
        f"shaded = excluded from analysis (cols < {col_min} or >= {col_max})"
    )
    ax_image.legend(loc="upper right", fontsize=8)

    ax_resid.errorbar(
        centroids.columns, fit.residuals, yerr=centroids.sigma_x0,
        fmt=".", color="steelblue", markersize=3, elinewidth=0.5, alpha=0.6,
    )
    ax_resid.axhline(0.0, color="black", linewidth=0.8)
    ax_resid.set_xlabel("Spectral pixel column")
    ax_resid.set_ylabel("Residual (px)")
    ax_resid.set_title(f"degree-{degree_for_line} fit residuals (reduced chi-sq = {fit.reduced_chi_squared:.3g})")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="Path to the raw image file")
    parser.add_argument(
        "--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES),
        help="Polynomial degrees to fit (default: 1 2 3)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Where to save the plot PNG",
    )
    parser.add_argument(
        "--col-min", type=int, default=DEFAULT_VALID_COL_MIN,
        help="First spectral column included in analysis (default: 501)",
    )
    parser.add_argument(
        "--col-max", type=int, default=DEFAULT_VALID_COL_MAX,
        help="First spectral column excluded again onward (default: 1700)",
    )
    args = parser.parse_args()

    frame = load_raw_frame(args.image_path, col_min=args.col_min, col_max=args.col_max)
    axis = PixelColumnAxis()
    result = analyze_shot(frame, axis, degrees=tuple(args.degrees))

    print_report(result)
    plot_result(
        frame.image, result, degree_for_line=1,
        col_min=args.col_min, col_max=args.col_max, output_path=args.output,
    )


if __name__ == "__main__":
    main()
