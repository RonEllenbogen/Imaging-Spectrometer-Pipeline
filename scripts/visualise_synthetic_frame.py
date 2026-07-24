"""
Generates a sample frame using the real SyntheticBackend and plots it
alongside its per-column weighted centroid and recovered linear fit.

Useful for:
- A quick visual sanity check after changing SyntheticBackend's parameters
- Seeing the injected-slope-recovery test visually, outside pytest

Usage:
    python scripts/visualise_synthetic_frame.py
    python scripts/visualise_synthetic_frame.py --slope 0.05 --seed 7
    python scripts/visualise_synthetic_frame.py --pixel-format Mono12 --output out.png
"""

# Imports
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pipeline.acquisition import (
    SyntheticBackend,
    CameraConfigurationError,
    CameraTimeoutError,
)

# Constants
# Some default values
DEFAULT_SLOPE_PX_PER_COL = 0.03
DEFAULT_SEED = 42
DEFAULT_PIXEL_FORMAT = "Mono8"
DEFAULT_TIMEOUT_MS = 5000
DEFAULT_OUTPUT_PATH = Path("assets/images/synthetic_frame_sample.png")


def weighted_centroid_per_column(frame: np.ndarray) -> np.ndarray:

    '''
    Minimal standalone intensity-weighted centroid calculation, for
    visualization purposes only -- NOT a substitute for
    analysis/computations.py, which will also compute per-column
    uncertainty and handle low-signal columns and background subtraction.

    Parameters
    ----------
    frame
        A 2D array with the spatial axis as rows and the spectral axis as
        columns, matching CANONICAL_SHAPE.

    Returns
    -------
    np.ndarray
        1D array of length frame.shape[1], the intensity-weighted centroid
        (in row/spatial-pixel units) for each column.
    '''

    rows = np.arange(frame.shape[0]).reshape(-1, 1)
    weights = frame.astype(float)
    col_sums = weights.sum(axis=0)
    return (rows * weights).sum(axis=0) / col_sums


def generate_sample_frame(
    slope_px_per_col: float, seed: int, pixel_format: str, timeout_ms: int
) -> np.ndarray:

    '''
    Connects to a real SyntheticBackend, grabs one frame, and cleans up.

    Parameters
    ----------
    slope_px_per_col
        Injected centroid slope (px per column) for the synthetic beam.
    seed
        RNG seed, for reproducible frames.
    pixel_format
        PixelFormat string passed to configure(), e.g. "Mono8", "Mono12".
    timeout_ms
        Timeout passed to grab_one().

    Returns
    -------
    np.ndarray
        The generated frame, canonical shape and dtype.
    '''

    backend = SyntheticBackend(seed=seed, slope_px_per_col=slope_px_per_col)
    backend.connect()
    try:
        backend.configure(exposure_us=2000, gain_db=0.0, pixel_format=pixel_format)
        frame = backend.grab_one(timeout_ms=timeout_ms)
    finally:
        # close() is safe to call even if configure()/grab_one() raised --
        # this is the same guarantee TestBackendLifecycle checks
        backend.close()
    return frame


def plot_frame(frame: np.ndarray, injected_slope: float, output_path: Path) -> None:

    '''
    Plots the frame alongside its per-column centroid and linear fit,
    saves the figure, and prints the injected vs recovered slope.

    Parameters
    ----------
    frame
        2D array, spatial axis as rows, spectral axis as columns.
    injected_slope
        The slope injected into the frame, used for plot labels.
    output_path
        Path to save the resulting PNG. Parent directories are created
        if they don't already exist.
    '''

    cols = np.arange(frame.shape[1])
    centroids = weighted_centroid_per_column(frame)
    fitted_slope, fitted_intercept = np.polyfit(cols, centroids, 1)

    fig, (ax_img, ax_fit) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    im = ax_img.imshow(frame, aspect="auto", cmap="inferno", origin="upper")
    ax_img.plot(cols, centroids, color="cyan", linewidth=1, label="weighted centroid")
    ax_img.set_xlabel("Spectral axis (column index)")
    ax_img.set_ylabel("Spatial axis (row index)")
    ax_img.set_title(f"SyntheticBackend frame -- injected slope = {injected_slope} px/col")
    ax_img.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax_img, label="counts")

    ax_fit.scatter(cols[::20], centroids[::20], s=6, alpha=0.5, label="centroid (every 20th col)")
    ax_fit.plot(
        cols, fitted_slope * cols + fitted_intercept, color="red",
        label=f"fit: slope={fitted_slope:.4f} px/col",
    )
    ax_fit.set_xlabel("Spectral axis (column index)")
    ax_fit.set_ylabel("Centroid position (row, px)")
    ax_fit.set_title(f"Recovered vs injected slope\n(injected = {injected_slope} px/col)")
    ax_fit.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=130)

    print(f"injected slope:  {injected_slope}")
    print(f"recovered slope: {fitted_slope:.5f}")
    print(f"saved figure to: {output_path}")

    plt.show()


def main() -> None:

    '''
    Parses CLI arguments, generates a sample frame, and plots it.
    Exits with status 1 and a clear message on CameraConfigurationError
    or CameraTimeoutError, rather than a raw traceback.
    '''

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slope", type=float, default=DEFAULT_SLOPE_PX_PER_COL,
        help="Injected centroid slope, px per column",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="RNG seed for reproducible frames",
    )
    parser.add_argument(
        "--pixel-format", type=str, default=DEFAULT_PIXEL_FORMAT,
        help="PixelFormat string, e.g. Mono8, Mono10, Mono12, Mono16",
    )
    parser.add_argument(
        "--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS,
        help="Timeout passed to grab_one()",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Where to save the resulting PNG",
    )
    args = parser.parse_args()

    try:
        frame = generate_sample_frame(args.slope, args.seed, args.pixel_format, args.timeout_ms)
    except CameraConfigurationError as e:
        print(f"Configuration error: {e}")
        raise SystemExit(1)
    except CameraTimeoutError as e:
        print(f"Grab timed out: {e}")
        raise SystemExit(1)

    plot_frame(frame, args.slope, args.output)


if __name__ == "__main__":
    main()