'''
Builds and saves a geometric tilt calibration artifact
(calibration/spectral/geometric_tilt.py) from already-captured lamp
frames -- the "load files, build, save" glue this codebase's
run_*_calibration() workflow functions provide for a live CameraStream,
which doesn't apply here since these frames were captured manually
outside CameraStream (same situation scripts/analyze_raw_shot.py's
module docstring describes).

build_geometric_tilt() itself never touches disk (see its own module
docstring: every build_*() in this codebase is pure, save_*()/load_*()
are separate) -- this script is the one-off/session-level entry point
that calls both, for reuse whenever the lamp is recaptured.

exposure_us/gain_db are NOT the frames' real captured settings -- these
.bmp files carry no acquisition metadata, so configs/default.yaml's
configured exposure_time is used as a placeholder and gain_db defaults to
0.0, same convention (and same caveat) as analyze_raw_shot.py's
load_raw_frame().

Usage:
    python scripts/build_geometric_tilt_calibration.py
    python scripts/build_geometric_tilt_calibration.py --output data/processed/my_tilt.npz
'''

# Imports

import argparse
from pathlib import Path

import imageio.v3 as iio

from pipeline.acquisition import CANONICAL_DTYPE, CANONICAL_SHAPE, FrameData
from pipeline.calibration.spectral import build_geometric_tilt, save_geometric_tilt
from pipeline.utils.helpers import load_config

# Constants

DEFAULT_SHOT_PATHS = tuple(
    Path("data/raw/spectral_lamp_11.8.26") / name
    for name in ("shot_1.bmp", "shot_2.bmp", "shot_3.bmp")
)
DEFAULT_OUTPUT_PATH = Path("data/processed/spectral_lamp_geometric_tilt.npz")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots", type=Path, nargs="+", default=list(DEFAULT_SHOT_PATHS),
        help="Lamp-only frame files (default: data/raw/spectral_lamp_11.8.26/shot_{1,2,3}.bmp)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Where to save the geometric tilt calibration .npz",
    )
    parser.add_argument(
        "--gain-db", type=float, default=DEFAULT_GAIN_DB,
        help="Placeholder gain_db tagged on the saved record (default: 0.0) -- see module docstring",
    )
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    exposure_us = float(config["camera"]["exposure_time"])

    frames = [
        load_frame(path, frame_id=i, exposure_us=exposure_us, gain_db=args.gain_db)
        for i, path in enumerate(args.shots)
    ]

    result = build_geometric_tilt(frames)

    print(f"reference_row: {result.reference_row}")
    print(f"row_shift range: [{result.row_shift.min():.3f}, {result.row_shift.max():.3f}] px")
    print(f"{len(result.residual_slope_columns)} lines used for the residual term:")
    for col, slope in zip(result.residual_slope_columns, result.residual_slope_values):
        print(f"  col {col:6.0f}: residual slope {slope:+.5f} px/row")

    save_geometric_tilt(args.output, result)
    print(f"Saved geometric tilt calibration to {args.output}")


if __name__ == "__main__":
    main()
