'''
Builds and saves a pixel -> wavelength_nm spectral calibration from the
Argon lamp shots, with geometric tilt correction applied but explicitly
NO baseline or flat-field correction -- no baseline/flat-field calibration
session exists yet for this camera (see scripts/analyze_raw_shot.py's own
module docstring for the same situation), so each raw frame is wrapped
directly into a ProcessedFrame rather than run through run_preprocessing(),
same as that script does. This is a deliberate, narrower correction chain
than calibration/spectral/workflow.py's run_spectral_calibration() (which
requires a full CalibrationSet and a live CameraStream, neither available
here), assembled from the same building blocks by hand.

Pipeline: load raw frames -> wrap as ProcessedFrame (no baseline/flat-
field/bad-pixel correction) -> apply_geometric_tilt_correction() per frame
-> average -> line_matching.match_lines() -> calibrate.calibrate_spectral()
-> save_spectral_calibration().

Usage:
    python scripts/build_spectral_calibration.py
    python scripts/build_spectral_calibration.py --degree 2 --output out.npz
'''

# Imports

import argparse
import time
from pathlib import Path

import imageio.v3 as iio

from pipeline.acquisition import CANONICAL_DTYPE, CANONICAL_SHAPE, FrameData
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spectral import calibrate_spectral, match_lines, save_spectral_calibration
from pipeline.calibration.spectral.geometric_tilt import load_geometric_tilt
from pipeline.preprocessing import ProcessedFrame
from pipeline.preprocessing.steps import apply_geometric_tilt_correction
from pipeline.utils.helpers import load_config

# Constants

DEFAULT_SHOT_PATHS = tuple(
    Path("data/raw/spectral_lamp_11.8.26") / name
    for name in ("shot_1.bmp", "shot_2.bmp", "shot_3.bmp")
)
DEFAULT_TILT_PATH = Path("data/processed/spectral_lamp_geometric_tilt.npz")
DEFAULT_OUTPUT_PATH = Path("data/processed/spectral_lamp_wavelength_calibration.npz")
DEFAULT_GAIN_DB = 0.0

# Classes

# Functions

def load_raw_frame(path: Path, frame_id: int, exposure_us: float, gain_db: float) -> FrameData:

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


def print_report(result) -> None:

    '''Prints the fitted coefficients and the matched lines the fit was built from.'''

    fit = result.fit
    print("pixel -> wavelength_nm fit:")
    for k, (c, sigma_c) in enumerate(zip(fit.coefficients, fit.coefficient_sigma)):
        print(f"  c{k} = {c:.6g} +/- {sigma_c:.3g}")
    print(f"  reduced chi-squared = {fit.reduced_chi_squared:.4g}")
    print(f"  {fit.residuals.shape[0]} matched lines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots", type=Path, nargs="+", default=list(DEFAULT_SHOT_PATHS),
        help="Lamp-only frame files (default: data/raw/spectral_lamp_11.8.26/shot_{1,2,3}.bmp)",
    )
    parser.add_argument(
        "--tilt", type=Path, default=DEFAULT_TILT_PATH,
        help="Geometric tilt calibration .npz to apply before line matching",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Where to save the spectral calibration .npz",
    )
    parser.add_argument(
        "--degree", type=int, default=1,
        help="Polynomial degree for the pixel -> wavelength_nm fit (default: 1)",
    )
    parser.add_argument(
        "--gain-db", type=float, default=DEFAULT_GAIN_DB,
        help="Placeholder gain_db tagged on the saved record (default: 0.0)",
    )
    parser.add_argument(
        "--include-residual", action="store_true",
        help="Also apply the geometric tilt calibration's smaller per-column residual term "
             "(default: off -- see calibration/spectral/geometric_tilt.py)",
    )
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    exposure_us = float(config["camera"]["exposure_time"])

    tilt = load_geometric_tilt(args.tilt)

    corrected_images = []
    for i, path in enumerate(args.shots):
        raw_frame = load_raw_frame(path, frame_id=i, exposure_us=exposure_us, gain_db=args.gain_db)
        # No baseline/flat-field/bad-pixel correction -- see module docstring.
        processed = ProcessedFrame(
            image=raw_frame.image.astype(float), frame_id=raw_frame.frame_id,
            timestamp=raw_frame.timestamp, exposure_us=raw_frame.exposure_us, gain_db=raw_frame.gain_db,
        )
        corrected = apply_geometric_tilt_correction(processed, tilt, include_residual=args.include_residual)
        corrected_images.append(corrected.image)

    averaged_image = sum(corrected_images) / len(corrected_images)

    pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm = match_lines(averaged_image)

    record = CalibrationRecord(
        exposure_us=exposure_us, gain_db=args.gain_db,
        timestamp=time.time(), source_frame_count=len(args.shots),
    )
    result = calibrate_spectral(
        pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm, record, degree=args.degree,
    )

    print_report(result)
    save_spectral_calibration(args.output, result)
    print(f"Saved spectral calibration to {args.output}")


if __name__ == "__main__":
    main()
