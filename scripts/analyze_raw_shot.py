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

def load_raw_frame(image_path: Path) -> ProcessedFrame:

    '''
    Loads a raw image file and wraps it directly as a ProcessedFrame,
    skipping run_preprocessing() entirely (see module docstring).

    Parameters
    ----------
    image_path
        Path to the raw image file (e.g. a .bmp capture).

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
    image: np.ndarray, result: ShotAnalysisResult, degree_for_line: int, output_path: Path
) -> None:

    '''
    Heatmap of the raw frame with centroids overlaid (top), residuals of
    the requested degree's fit (bottom).

    Parameters
    ----------
    image
        Raw 2D frame, CANONICAL_SHAPE.
    result
        analyze_shot() output.
    degree_for_line
        Which fit's line/residuals to draw.
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
    ax_image.set_title("data/raw/khz/11.8.26.bmp -- raw heatmap + centroids (no calibration applied)")
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
    args = parser.parse_args()

    frame = load_raw_frame(args.image_path)
    axis = PixelColumnAxis()
    result = analyze_shot(frame, axis, degrees=tuple(args.degrees))

    print_report(result)
    plot_result(frame.image, result, degree_for_line=1, output_path=args.output)


if __name__ == "__main__":
    main()
