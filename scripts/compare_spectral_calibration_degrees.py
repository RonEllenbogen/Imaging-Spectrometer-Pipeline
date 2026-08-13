'''
Compares the pixel -> wavelength_nm spectral calibration
(calibration/spectral/calibrate.py's calibrate_spectral()) at degree 1
(linear), 2 (quadratic), and 3 (cubic) against the same matched lamp-line
data, from the same stacked/geometric-tilt-corrected lamp image
plot_beam_spectrum.py builds its own calibration from -- same lamp shots
(data/diagnostic/grating_rotation/l2.{1,2,3}.bmp), same stack-before-tilt-
correction convention, same build_geometric_tilt_linear() default (see
docs/project_state.md, "Switch default geometric tilt method to the
linear-fit amendment"), same no-baseline/flat-field caveat (no such
calibration session exists for this dataset -- the stacked image is
wrapped directly into a ProcessedFrame).

line_matching.py's match_lines() has no degree parameter -- it only
detects and identifies lines against the reference Argon list, it doesn't
fit a polynomial -- so it's run exactly once, and its matched (pixel,
wavelength_nm, sigma_pixel, sigma_wavelength_nm) arrays are reused for all
three calibrate_spectral() calls below. This means any difference between
the three fits is purely the polynomial degree, not different matched-
line input -- the direct model-adequacy comparison calibrate_spectral()'s
own docstring describes (higher degrees are a diagnostic on whether a
linear dispersion model is adequate, not usually the production choice).

Saves a two-panel plot (top: matched-line scatter with all three fit
curves overlaid; bottom: each fit's residuals vs. pixel column) to
--output, plus a plain-text summary (coefficients, per-coefficient
uncertainty, reduced chi-squared, matched-line count for each degree) to
the same path with a .txt suffix -- printed to stdout too.

Usage:
    python scripts/compare_spectral_calibration_degrees.py
    python scripts/compare_spectral_calibration_degrees.py --lamp-shots a.bmp b.bmp c.bmp --output out.png
'''

# Imports

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from pipeline.acquisition import CANONICAL_DTYPE, CANONICAL_SHAPE, FrameData
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spectral import (
    WavelengthCalibrationResult, build_geometric_tilt_linear, calibrate_spectral, match_lines,
)
from pipeline.preprocessing import ProcessedFrame
from pipeline.preprocessing.steps import apply_geometric_tilt_correction
from pipeline.utils.helpers import load_config

# Constants

DEFAULT_LAMP_SHOT_PATHS = tuple(
    Path("data/diagnostic/grating_rotation") / name
    for name in ("l2.1.bmp", "l2.2.bmp", "l2.3.bmp")
)
DEFAULT_OUTPUT_PATH = Path("data/diagnostic/spectral_calibration_degree_comparison.png")
DEFAULT_GAIN_DB = 0.0

DEGREES = (1, 2, 3)
DEGREE_LABELS = {1: "linear", 2: "quadratic", 3: "cubic"}
DEGREE_COLORS = {1: "steelblue", 2: "darkorange", 3: "firebrick"}

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


def stack_frames(frames: list[FrameData]) -> np.ndarray:

    '''Averages a list of frames' images into a single float64 array.'''

    return np.mean([f.image.astype(np.float64) for f in frames], axis=0)


def format_report(results: dict[int, WavelengthCalibrationResult], n_matched: int) -> str:

    '''Plain-text comparison of every degree's coefficients, uncertainty, and reduced chi-squared.'''

    lines = [f"{n_matched} matched lamp lines, shared across every degree below", ""]
    for degree in DEGREES:
        fit = results[degree].fit
        lines.append(f"degree {degree} ({DEGREE_LABELS[degree]}):")
        for k, (c, sigma_c) in enumerate(zip(fit.coefficients, fit.coefficient_sigma)):
            lines.append(f"  c{k} = {c:.6g} +/- {sigma_c:.3g}")
        lines.append(f"  reduced chi-squared = {fit.reduced_chi_squared:.4g}")
        lines.append("")
    return "\n".join(lines)


def plot_comparison(
    pixel: np.ndarray, wavelength_nm: np.ndarray, sigma_pixel: np.ndarray, sigma_wavelength_nm: np.ndarray,
    results: dict[int, WavelengthCalibrationResult], output_path: Path,
) -> None:

    '''
    Top: matched (pixel, wavelength_nm) points with error bars, all three
    fit curves overlaid. Bottom: each fit's own residuals (observed minus
    predicted wavelength, at the matched pixel positions) vs. pixel column
    -- a systematic residual trend (rather than scatter around zero) is
    the visual sign a lower degree is inadequate.
    '''

    fig, (ax_fit, ax_resid) = plt.subplots(2, 1, figsize=(11, 10), height_ratios=(3, 2))

    ax_fit.errorbar(
        pixel, wavelength_nm, xerr=sigma_pixel, yerr=sigma_wavelength_nm,
        fmt="o", color="black", markersize=4, capsize=3, label="matched lamp lines", zorder=5,
    )
    pixel_range = np.linspace(pixel.min(), pixel.max(), 400)
    for degree in DEGREES:
        fit = results[degree].fit
        curve = results[degree].wavelength_nm(pixel_range)
        ax_fit.plot(
            pixel_range, curve, color=DEGREE_COLORS[degree], linewidth=1.5,
            label=f"{DEGREE_LABELS[degree]} (degree {degree}), reduced chi2 = {fit.reduced_chi_squared:.3g}",
        )
    ax_fit.set_xlabel("Spectral pixel column")
    ax_fit.set_ylabel("Wavelength (nm)")
    ax_fit.set_title("Pixel -> wavelength calibration: linear/quadratic/cubic fits")
    ax_fit.legend(fontsize=9)

    for degree in DEGREES:
        fit = results[degree].fit
        ax_resid.plot(
            pixel, fit.residuals, "o", color=DEGREE_COLORS[degree], markersize=5, alpha=0.8,
            label=f"{DEGREE_LABELS[degree]} residuals",
        )
    ax_resid.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax_resid.set_xlabel("Spectral pixel column")
    ax_resid.set_ylabel("Residual (nm)")
    ax_resid.set_title("Fit residuals (observed - predicted wavelength)")
    ax_resid.legend(fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lamp-shots", type=Path, nargs="+", default=list(DEFAULT_LAMP_SHOT_PATHS),
        help="Lamp frames the calibration is built from "
             "(default: data/diagnostic/grating_rotation/l2.{1,2,3}.bmp)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Where to save the comparison plot -- a same-named .txt stats summary is saved alongside it",
    )
    parser.add_argument(
        "--gain-db", type=float, default=DEFAULT_GAIN_DB,
        help="Placeholder gain_db tagged on the calibration (default: 0.0) -- see module docstring",
    )
    parser.add_argument(
        "--include-residual", action="store_true",
        help="Also apply the geometric tilt calibration's smaller per-column residual term "
             "(default: off -- see calibration/spectral/geometric_tilt.py)",
    )
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    exposure_us = float(config["camera"]["exposure_time"])

    lamp_frames = [
        load_frame(path, frame_id=i, exposure_us=exposure_us, gain_db=args.gain_db)
        for i, path in enumerate(args.lamp_shots)
    ]

    tilt = build_geometric_tilt_linear(lamp_frames)

    stacked_lamp = ProcessedFrame(
        image=stack_frames(lamp_frames), frame_id=0, timestamp=0.0,
        exposure_us=exposure_us, gain_db=args.gain_db,
    )
    lamp_corrected = apply_geometric_tilt_correction(stacked_lamp, tilt, include_residual=args.include_residual)

    pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm = match_lines(lamp_corrected.image)
    record = CalibrationRecord(
        exposure_us=exposure_us, gain_db=args.gain_db,
        timestamp=time.time(), source_frame_count=len(args.lamp_shots),
    )

    results = {
        degree: calibrate_spectral(pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm, record, degree=degree)
        for degree in DEGREES
    }

    report = format_report(results, n_matched=pixel.shape[0])
    print(report)

    stats_path = args.output.with_suffix(".txt")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(report)
    print(f"Saved stats summary to {stats_path}")

    plot_comparison(pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm, results, args.output)


if __name__ == "__main__":
    main()
