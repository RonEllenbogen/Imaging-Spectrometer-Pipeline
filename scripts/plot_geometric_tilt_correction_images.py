'''
Visualizes the actual effect of both geometric tilt corrections
(preprocessing/steps/geometric_tilt.py's apply_geometric_tilt_correction())
on a real calibration lamp image -- pairs with
compare_geometric_tilt_methods.py's row_shift-curve comparison by showing
what the two calibrations (calibration/spectral/geometric_tilt.py's
build_geometric_tilt() and build_geometric_tilt_linear()) look like
applied to the image itself, not just their shared curve.

Runs correction against the stacked (mean) lamp image rather than a
single shot, for the same SNR reason build_geometric_tilt() itself stacks
frames before detecting lines. No baseline/flat-field/bad-pixel
correction is applied first (this script has no calibration artifacts for
those steps to draw on) -- fine for visualizing tilt/straightness, but
the displayed images are not fully preprocessed frames.

exposure_us/gain_db are NOT the frames' real captured settings -- same
placeholder convention as compare_geometric_tilt_methods.py.

Usage:
    python scripts/plot_geometric_tilt_correction_images.py
    python scripts/plot_geometric_tilt_correction_images.py --shots a.bmp b.bmp c.bmp --output out.png
'''

# Imports

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from pipeline.acquisition import CANONICAL_DTYPE, CANONICAL_SHAPE, FrameData
from pipeline.calibration.spectral import (
    GeometricTiltResult, build_geometric_tilt, build_geometric_tilt_linear,
)
from pipeline.preprocessing import ProcessedFrame
from pipeline.preprocessing.steps import apply_geometric_tilt_correction
from pipeline.utils.helpers import load_config

# Constants

DEFAULT_SHOT_PATHS = tuple(
    Path("data/diagnostic/grating_rotation") / name
    for name in ("l2.1.bmp", "l2.2.bmp", "l2.3.bmp")
)
DEFAULT_OUTPUT_PATH = Path("data/diagnostic/geometric_tilt_correction_images.png")
DEFAULT_PER_SHOT_OUTPUT_PATH = Path("data/diagnostic/geometric_tilt_correction_per_shot.png")
DEFAULT_GAIN_DB = 0.0

# Bright, narrow lines against near-zero background -- clipping the
# display range to a low percentile of the stacked image (rather than its
# max) keeps the lines visible instead of the colormap being dominated by
# a handful of saturated-looking pixels.
DISPLAY_PERCENTILE = 99.9

# Default zoom window for the second (per-line) row of panels: centered
# on this spectrometer's brightest detected line (see
# compare_geometric_tilt_methods.py's printed residual_slope_columns),
# wide enough either side to hold the full measured row_shift excursion
# (-34 to +41 px, see geometric_tilt_linear_comparison.png) without
# clipping it out of frame.
ZOOM_COLUMN_CENTER = 1241
ZOOM_COLUMN_HALF_WIDTH = 60

# Classes

# Functions

def load_frame(path: Path, frame_id: int, exposure_us: float, gain_db: float) -> FrameData:

    '''Loads one raw image file as a FrameData, validated against the canonical frame contract.'''

    image = iio.imread(path)
    if image.shape != CANONICAL_SHAPE:
        raise ValueError(f"{path} has shape {image.shape}, expected {CANONICAL_SHAPE}")
    if image.dtype != CANONICAL_DTYPE:
        raise ValueError(f"{path} has dtype {image.dtype}, expected {CANONICAL_DTYPE}")

    return FrameData(
        image=image, frame_id=frame_id, timestamp=float(frame_id),
        exposure_us=exposure_us, gain_db=gain_db,
    )


def plot_corrected_images(
    uncorrected: np.ndarray, pointwise_corrected: np.ndarray, linear_corrected: np.ndarray,
    output_path: Path, zoom_column_center: int, zoom_column_half_width: int,
) -> None:

    '''
    Top row: uncorrected vs. both tilt-corrected versions of the full
    stacked lamp image, same color scale. Bottom row: the same three
    images cropped tightly around one bright line -- the full-frame view
    barely shows the correction (the measured shift is only a few percent
    of the 1920px-wide frame), so the crop is what actually makes the
    straightening, and the two methods' disagreement at the row_shift
    curve's noisy edges/jumps, visible.
    '''

    vmax = float(np.percentile(uncorrected, DISPLAY_PERCENTILE))
    col_lo = zoom_column_center - zoom_column_half_width
    col_hi = zoom_column_center + zoom_column_half_width

    fig, axes = plt.subplots(2, 3, figsize=(18, 13))
    titles = ["Uncorrected", "build_geometric_tilt (pointwise)", "build_geometric_tilt_linear"]
    images = [uncorrected, pointwise_corrected, linear_corrected]

    for ax, title, image in zip(axes[0], titles, images):
        ax.imshow(image, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Spectral pixel column")
    axes[0, 0].set_ylabel("Spatial pixel row")

    for ax, image in zip(axes[1], images):
        ax.imshow(
            image[:, col_lo:col_hi], cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto",
            extent=(col_lo, col_hi, image.shape[0], 0),
        )
        ax.set_xlabel("Spectral pixel column (zoomed)")
    axes[1, 0].set_ylabel("Spatial pixel row")

    fig.suptitle("Geometric tilt correction applied to the stacked lamp image (l2.1-3.bmp)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved comparison image to {output_path}")


def plot_per_shot_corrected_images(
    frames: list[FrameData], pointwise: GeometricTiltResult, linear: GeometricTiltResult,
    exposure_us: float, gain_db: float, output_path: Path,
    zoom_column_center: int, zoom_column_half_width: int,
) -> None:

    '''
    One row per individual lamp shot (not stacked): uncorrected vs. both
    tilt corrections, zoomed to the same line window as
    plot_corrected_images(). Both calibrations are still built from all
    shots stacked together (build_geometric_tilt()/
    build_geometric_tilt_linear() need that stack for per-row centroid
    SNR) -- only what the correction is *applied to* changes here, to
    check whether what the stacked comparison shows (doublet-splitting,
    a noisy top edge) is a real feature of every individual shot or an
    artifact introduced by averaging the three together first.
    '''

    col_lo = zoom_column_center - zoom_column_half_width
    col_hi = zoom_column_center + zoom_column_half_width
    titles = ["Uncorrected", "build_geometric_tilt (pointwise)", "build_geometric_tilt_linear"]

    fig, axes = plt.subplots(len(frames), 3, figsize=(18, 6.5 * len(frames)), squeeze=False)

    for row_idx, frame in enumerate(frames):
        image = frame.image.astype(np.float64)
        processed = ProcessedFrame(
            image=image, frame_id=frame.frame_id, timestamp=frame.timestamp,
            exposure_us=exposure_us, gain_db=gain_db,
        )
        pointwise_corrected = apply_geometric_tilt_correction(processed, pointwise).image
        linear_corrected = apply_geometric_tilt_correction(processed, linear).image
        vmax = float(np.percentile(image, DISPLAY_PERCENTILE))

        for col_idx, (title, panel_image) in enumerate(
            zip(titles, [image, pointwise_corrected, linear_corrected])
        ):
            ax = axes[row_idx, col_idx]
            ax.imshow(
                panel_image[:, col_lo:col_hi], cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto",
                extent=(col_lo, col_hi, panel_image.shape[0], 0),
            )
            if row_idx == 0:
                ax.set_title(title, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"shot {frame.frame_id + 1}: spatial pixel row")
            if row_idx == len(frames) - 1:
                ax.set_xlabel("Spectral pixel column (zoomed)")

    fig.suptitle("Geometric tilt correction applied to each individual lamp shot (not stacked)")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved per-shot comparison image to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots", type=Path, nargs="+", default=list(DEFAULT_SHOT_PATHS),
        help="Lamp-only frame files (default: data/diagnostic/grating_rotation/l2.{1,2,3}.bmp)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save the stacked comparison image",
    )
    parser.add_argument(
        "--per-shot-output", type=Path, default=DEFAULT_PER_SHOT_OUTPUT_PATH,
        help="Where to save the per-shot (not stacked) comparison image",
    )
    parser.add_argument(
        "--gain-db", type=float, default=DEFAULT_GAIN_DB,
        help="Placeholder gain_db tagged on both results (default: 0.0) -- see module docstring",
    )
    parser.add_argument(
        "--zoom-column-center", type=int, default=ZOOM_COLUMN_CENTER,
        help=f"Spectral column the bottom-row zoomed panels are centered on (default: {ZOOM_COLUMN_CENTER})",
    )
    parser.add_argument(
        "--zoom-column-half-width", type=int, default=ZOOM_COLUMN_HALF_WIDTH,
        help=f"Half-width in columns of the zoomed panels (default: {ZOOM_COLUMN_HALF_WIDTH})",
    )
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    exposure_us = float(config["camera"]["exposure_time"])

    frames = [
        load_frame(path, frame_id=i, exposure_us=exposure_us, gain_db=args.gain_db)
        for i, path in enumerate(args.shots)
    ]

    stacked = np.mean([f.image.astype(np.float64) for f in frames], axis=0)
    processed = ProcessedFrame(
        image=stacked, frame_id=0, timestamp=0.0, exposure_us=exposure_us, gain_db=args.gain_db,
    )

    pointwise = build_geometric_tilt(frames)
    linear = build_geometric_tilt_linear(frames)

    pointwise_corrected = apply_geometric_tilt_correction(processed, pointwise)
    linear_corrected = apply_geometric_tilt_correction(processed, linear)

    plot_corrected_images(
        stacked, pointwise_corrected.image, linear_corrected.image, args.output,
        args.zoom_column_center, args.zoom_column_half_width,
    )
    plot_per_shot_corrected_images(
        frames, pointwise, linear, exposure_us, args.gain_db, args.per_shot_output,
        args.zoom_column_center, args.zoom_column_half_width,
    )


if __name__ == "__main__":
    main()
