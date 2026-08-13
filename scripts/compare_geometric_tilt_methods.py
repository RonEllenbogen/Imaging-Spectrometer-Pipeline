'''
Compares build_geometric_tilt() (the per-row inverse-variance-weighted
shared curve, gaps filled by interpolation) against
build_geometric_tilt_linear() (the same per-row curve reduced to a
single weighted straight-line fit) on the same set of lamp frames --
one-off diagnostic script for evaluating the linear-fit amendment
against real calibration lamp data before deciding whether to adopt it
(see calibration/spectral/geometric_tilt.py's module docstring for what
both methods measure and why they differ).

exposure_us/gain_db are NOT the frames' real captured settings -- these
.bmp files carry no acquisition metadata, so configs/default.yaml's
configured exposure_time is used as a placeholder and gain_db defaults to
0.0, same convention (and same caveat) as
scripts/build_geometric_tilt_calibration.py.

Usage:
    python scripts/compare_geometric_tilt_methods.py
    python scripts/compare_geometric_tilt_methods.py --shots a.bmp b.bmp c.bmp --output out.png
'''

# Imports

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from pipeline.acquisition import CANONICAL_DTYPE, CANONICAL_SHAPE, FrameData
from pipeline.calibration.spectral import build_geometric_tilt, build_geometric_tilt_linear
from pipeline.utils.helpers import load_config

# Constants

DEFAULT_SHOT_PATHS = tuple(
    Path("data/diagnostic/grating_rotation") / name
    for name in ("l2.1.bmp", "l2.2.bmp", "l2.3.bmp")
)
DEFAULT_OUTPUT_PATH = Path("data/diagnostic/geometric_tilt_linear_comparison.png")
DEFAULT_GAIN_DB = 0.0

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


def plot_comparison(
    pointwise_row_shift: np.ndarray, linear_row_shift: np.ndarray, reference_row: int, output_path: Path,
) -> None:

    '''Top: both row_shift curves overlaid. Bottom: their difference, linear minus pointwise.'''

    rows = np.arange(pointwise_row_shift.shape[0])
    diff = linear_row_shift - pointwise_row_shift

    fig, (ax_curve, ax_diff) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=(2, 1), sharex=True)

    ax_curve.plot(
        rows, pointwise_row_shift, color="steelblue", linewidth=1.2,
        label="build_geometric_tilt (pointwise)",
    )
    ax_curve.plot(
        rows, linear_row_shift, color="darkorange", linewidth=1.5, label="build_geometric_tilt_linear",
    )
    ax_curve.axvline(reference_row, color="gray", linestyle="--", linewidth=1, label="reference_row")
    ax_curve.set_ylabel("row_shift (px)")
    ax_curve.set_title("Geometric tilt shared curve: pointwise-weighted-mean vs. linear fit")
    ax_curve.legend(fontsize=8)

    ax_diff.plot(rows, diff, color="firebrick", linewidth=1.0)
    ax_diff.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax_diff.set_xlabel("Spatial pixel row")
    ax_diff.set_ylabel("linear - pointwise (px)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots", type=Path, nargs="+", default=list(DEFAULT_SHOT_PATHS),
        help="Lamp-only frame files (default: data/diagnostic/grating_rotation/l2.{1,2,3}.bmp)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save the comparison plot",
    )
    parser.add_argument(
        "--gain-db", type=float, default=DEFAULT_GAIN_DB,
        help="Placeholder gain_db tagged on both results (default: 0.0) -- see module docstring",
    )
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    exposure_us = float(config["camera"]["exposure_time"])

    frames = [
        load_frame(path, frame_id=i, exposure_us=exposure_us, gain_db=args.gain_db)
        for i, path in enumerate(args.shots)
    ]

    pointwise = build_geometric_tilt(frames)
    linear = build_geometric_tilt_linear(frames)

    diff = linear.row_shift - pointwise.row_shift

    print(f"reference_row: {pointwise.reference_row}")
    print(
        f"pointwise row_shift range: [{pointwise.row_shift.min():.3f}, {pointwise.row_shift.max():.3f}] px"
    )
    print(
        f"linear    row_shift range: [{linear.row_shift.min():.3f}, {linear.row_shift.max():.3f}] px"
    )
    print(f"max |linear - pointwise|: {np.max(np.abs(diff)):.3f} px")
    print(f"rms  |linear - pointwise|: {np.sqrt(np.mean(diff ** 2)):.3f} px")

    plot_comparison(pointwise.row_shift, linear.row_shift, pointwise.reference_row, args.output)


if __name__ == "__main__":
    main()
