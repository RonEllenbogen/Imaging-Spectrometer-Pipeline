'''
Visualizes the effect of both geometric tilt corrections
(preprocessing/steps/geometric_tilt.py's apply_geometric_tilt_correction())
on a real beam image, rather than on the calibration lamp image itself --
pairs with plot_geometric_tilt_correction_images.py (which corrects the
same lamp frames the calibration was built from) by instead applying both
calibrations to a *different* stacked image: the beam shots at
data/diagnostic/grating_rotation/2.{1,2,3}.bmp.

Both calibrations (calibration/spectral/geometric_tilt.py's
build_geometric_tilt() and build_geometric_tilt_linear()) are still built
from the stacked lamp shots (default: data/diagnostic/grating_rotation/
l2.{1,2,3}.bmp) -- only what the correction is *applied to* differs from
compare_geometric_tilt_methods.py/plot_geometric_tilt_correction_images.py.

No baseline/flat-field/bad-pixel correction is applied first (this script
has no calibration artifacts for those steps to draw on) -- fine for
visualizing tilt/straightness, but the displayed images are not fully
preprocessed frames.

exposure_us/gain_db are NOT the frames' real captured settings -- same
placeholder convention as the other geometric-tilt scripts.

Usage:
    python scripts/plot_geometric_tilt_correction_beam_image.py
    python scripts/plot_geometric_tilt_correction_beam_image.py \\
        --lamp-shots a.bmp b.bmp c.bmp --beam-shots d.bmp e.bmp f.bmp
'''

# Imports

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from pipeline.acquisition import CANONICAL_DTYPE, CANONICAL_SHAPE, FrameData
from pipeline.calibration.spectral import build_geometric_tilt, build_geometric_tilt_linear
from pipeline.preprocessing import ProcessedFrame
from pipeline.preprocessing.steps import apply_geometric_tilt_correction
from pipeline.utils.helpers import load_config

# Constants

DEFAULT_LAMP_SHOT_PATHS = tuple(
    Path("data/diagnostic/grating_rotation") / name
    for name in ("l2.1.bmp", "l2.2.bmp", "l2.3.bmp")
)
DEFAULT_BEAM_SHOT_PATHS = tuple(
    Path("data/diagnostic/grating_rotation") / name
    for name in ("2.1.bmp", "2.2.bmp", "2.3.bmp")
)
DEFAULT_OUTPUT_PATH = Path("data/diagnostic/geometric_tilt_correction_beam_image.png")
DEFAULT_GAIN_DB = 0.0

# Beam signal is broad rather than a handful of narrow lines -- a lower
# display percentile than the lamp scripts use keeps the fainter parts of
# the beam visible instead of only its brightest core.
DISPLAY_PERCENTILE = 99.5

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


def plot_beam_comparison(
    uncorrected: np.ndarray, pointwise_corrected: np.ndarray, linear_corrected: np.ndarray,
    output_path: Path,
) -> None:

    '''
    Top row: uncorrected vs. both tilt-corrected versions of the full
    beam image, same color scale. Unlike the lamp images (a handful of
    narrow lines), this beam is one broad diffuse blob spanning almost
    the full frame height, so a row-band crop looks the same as the full
    frame and shows nothing -- the correction's effect here is a several-
    percent-of-frame-width column shift applied differently to each row,
    not a visible change in the blob's outline. The bottom row makes that
    visible directly as a difference image instead: pointwise-minus-
    uncorrected, linear-minus-uncorrected, and pointwise-minus-linear (the
    two methods' disagreement about where the beam belongs), each on a
    diverging scale centered at zero.
    '''

    vmax = float(np.percentile(uncorrected, DISPLAY_PERCENTILE))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    titles = ["Uncorrected", "build_geometric_tilt (pointwise)", "build_geometric_tilt_linear"]
    images = [uncorrected, pointwise_corrected, linear_corrected]

    for ax, title, image in zip(axes[0], titles, images):
        ax.imshow(image, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Spectral pixel column")
    axes[0, 0].set_ylabel("Spatial pixel row")

    diffs = [
        ("pointwise - uncorrected", pointwise_corrected - uncorrected),
        ("linear - uncorrected", linear_corrected - uncorrected),
        ("pointwise - linear", pointwise_corrected - linear_corrected),
    ]
    diff_vmax = float(max(np.percentile(np.abs(diff), DISPLAY_PERCENTILE) for _, diff in diffs))
    for ax, (title, diff) in zip(axes[1], diffs):
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Spectral pixel column")
    axes[1, 0].set_ylabel("Spatial pixel row")
    fig.colorbar(im, ax=axes[1, :], fraction=0.02, pad=0.01, label="intensity difference")

    fig.suptitle(
        "Geometric tilt correction (calibrated from the stacked lamp image) applied to "
        "the stacked beam image (2.1-3.bmp)"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved beam comparison image to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lamp-shots", type=Path, nargs="+", default=list(DEFAULT_LAMP_SHOT_PATHS),
        help="Lamp frames the calibration is built from "
             "(default: data/diagnostic/grating_rotation/l2.{1,2,3}.bmp)",
    )
    parser.add_argument(
        "--beam-shots", type=Path, nargs="+", default=list(DEFAULT_BEAM_SHOT_PATHS),
        help="Beam frames the calibration is applied to "
             "(default: data/diagnostic/grating_rotation/2.{1,2,3}.bmp)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save the comparison image",
    )
    parser.add_argument(
        "--gain-db", type=float, default=DEFAULT_GAIN_DB,
        help="Placeholder gain_db tagged on both results (default: 0.0) -- see module docstring",
    )
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    exposure_us = float(config["camera"]["exposure_time"])

    lamp_frames = [
        load_frame(path, frame_id=i, exposure_us=exposure_us, gain_db=args.gain_db)
        for i, path in enumerate(args.lamp_shots)
    ]
    beam_frames = [
        load_frame(path, frame_id=i, exposure_us=exposure_us, gain_db=args.gain_db)
        for i, path in enumerate(args.beam_shots)
    ]

    pointwise = build_geometric_tilt(lamp_frames)
    linear = build_geometric_tilt_linear(lamp_frames)

    stacked_beam = np.mean([f.image.astype(np.float64) for f in beam_frames], axis=0)
    processed_beam = ProcessedFrame(
        image=stacked_beam, frame_id=0, timestamp=0.0, exposure_us=exposure_us, gain_db=args.gain_db,
    )

    pointwise_corrected = apply_geometric_tilt_correction(processed_beam, pointwise)
    linear_corrected = apply_geometric_tilt_correction(processed_beam, linear)

    plot_beam_comparison(
        stacked_beam, pointwise_corrected.image, linear_corrected.image, args.output,
    )


if __name__ == "__main__":
    main()
