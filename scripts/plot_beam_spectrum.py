'''
End-to-end diagnostic: calibrates both the geometric tilt correction and
the pixel -> wavelength_nm spectral calibration from the grating-rotation
lamp shots (data/diagnostic/grating_rotation/l2.{1,2,3}.bmp), applies both
to the real beam shots from the same session (data/diagnostic/
grating_rotation/2.{1,2,3}.bmp), and plots the resulting intensity-vs-
wavelength spectrum -- letting the measured beam spectrum be eyeballed
against its known shape (e.g. Ti:Sapphire, typically centered near 800nm)
using the same physically-realigned dataset docs/project_state.md's
geometric-tilt comparison work already validated.

Tilt: build_geometric_tilt_linear() (the weighted-straight-line-fit
method), not build_geometric_tilt() (the pointwise method) -- matches the
default this codebase's build_geometric_tilt_calibration.py and
run_spectral_calibration() now build with (see docs/project_state.md
"Switch default geometric tilt method to the linear-fit amendment" for
why: smoother/more robust at the frame's noisy edge rows, at the cost of
being unable to represent the two small non-monotonic jumps
build_geometric_tilt() can).

Both the lamp and beam frames are stacked (averaged) BEFORE tilt
correction is applied -- one correction of one stacked image each, same
convention as plot_geometric_tilt_correction_beam_image.py -- rather than
correcting each shot individually and averaging afterward (the convention
build_spectral_calibration.py uses instead). No baseline/flat-field/bad-
pixel correction is applied first (no such calibration session exists for
this dataset): both stacked images are wrapped directly into a
ProcessedFrame, same caveat as build_spectral_calibration.py and
plot_geometric_tilt_correction_beam_image.py.

Spectral calibration defaults to a cubic (degree=3) pixel -> wavelength_nm
fit, rather than build_spectral_calibration.py's default degree=1 --
caller-selected here since this script's purpose is plotting a beam
spectrum, not evaluating fit order itself.

Neither calibration artifact is saved to disk -- both are one-off,
built and consumed within this script, same convention as
compare_geometric_tilt_methods.py and plot_geometric_tilt_correction_
beam_image.py. Use build_geometric_tilt_calibration.py / build_spectral_
calibration.py directly if a reusable saved artifact is needed instead.

exposure_us/gain_db are NOT the frames' real captured settings -- these
.bmp files carry no acquisition metadata, so configs/default.yaml's
configured exposure_time is used as a placeholder and gain_db defaults to
0.0, same convention as every other grating_rotation/ script.

The plot also overlays an independently-measured "true" reference
spectrum (default: data/reference/beam_spectra/post_regen_1.txt, a raw
export from a separate, standalone spectrometer -- not this imaging
spectrometer -- covering the beam's full range post-regen). That file's
intensity is in that instrument's own raw counts, on a completely
different scale from this script's column-summed ADU (different
instrument, different aperture/collection efficiency, not photometrically
comparable), so both spectra are plotted peak-normalized (each divided by
its own maximum) rather than in their native units -- this plot compares
spectral *shape*, not absolute intensity.

Usage:
    python scripts/plot_beam_spectrum.py
    python scripts/plot_beam_spectrum.py --degree 2 --output out.png
    python scripts/plot_beam_spectrum.py --reference-spectrum data/reference/beam_spectra/oscillator_1.txt
'''

# Imports

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from pipeline.acquisition import CANONICAL_DTYPE, CANONICAL_SHAPE, FrameData, SPATIAL_AXIS
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spectral import build_geometric_tilt_linear, calibrate_spectral, match_lines
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
DEFAULT_REFERENCE_SPECTRUM_PATH = Path("data/reference/beam_spectra/post_regen_1.txt")
DEFAULT_OUTPUT_PATH = Path("data/diagnostic/grating_rotation_beam_spectrum.png")
DEFAULT_GAIN_DB = 0.0
DEFAULT_SPECTRAL_DEGREE = 3

# Marks the end of the metadata header in the reference spectrometer's raw
# export format (see load_reference_spectrum()) -- spectral data starts
# on the line after this one.
REFERENCE_SPECTRUM_DATA_MARKER = ">>>>>Begin Spectral Data<<<<<"

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


def load_reference_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:

    '''
    Parses a raw spectrometer export: a metadata header terminated by
    REFERENCE_SPECTRUM_DATA_MARKER, followed by tab-separated
    (wavelength_nm, intensity) rows -- see module docstring for what
    instrument/scale this intensity is in.

    Raises
    ------
    ValueError
        If path contains no REFERENCE_SPECTRUM_DATA_MARKER line.
    '''

    lines = path.read_text().splitlines()
    try:
        data_start = lines.index(REFERENCE_SPECTRUM_DATA_MARKER) + 1
    except ValueError:
        raise ValueError(f"{path} has no '{REFERENCE_SPECTRUM_DATA_MARKER}' marker")

    data = np.array([line.split() for line in lines[data_start:] if line.strip()], dtype=np.float64)
    return data[:, 0], data[:, 1]


def print_tilt_report(tilt) -> None:

    '''Prints the fitted geometric tilt curve and residual terms, same format as
    build_geometric_tilt_calibration.py.'''

    print("geometric tilt (linear-fit method):")
    print(f"  reference_row: {tilt.reference_row}")
    print(f"  row_shift range: [{tilt.row_shift.min():.3f}, {tilt.row_shift.max():.3f}] px")
    print(f"  {len(tilt.residual_slope_columns)} lines used for the residual term:")
    for col, slope in zip(tilt.residual_slope_columns, tilt.residual_slope_values):
        print(f"    col {col:6.0f}: residual slope {slope:+.5f} px/row")


def print_spectral_report(spectral) -> None:

    '''Prints the fitted pixel -> wavelength_nm coefficients, same format as
    build_spectral_calibration.py.'''

    fit = spectral.fit
    print("pixel -> wavelength_nm fit:")
    for k, (c, sigma_c) in enumerate(zip(fit.coefficients, fit.coefficient_sigma)):
        print(f"  c{k} = {c:.6g} +/- {sigma_c:.3g}")
    print(f"  reduced chi-squared = {fit.reduced_chi_squared:.4g}")
    print(f"  {fit.residuals.shape[0]} matched lines")


def plot_beam_spectrum(
    wavelength_nm: np.ndarray, spectrum: np.ndarray,
    reference_wavelength_nm: np.ndarray, reference_spectrum: np.ndarray, reference_label: str,
    output_path: Path,
) -> None:

    '''
    Plots the stacked, tilt- and spectrally-corrected beam spectrum against
    wavelength_nm, overlaid with an independently-measured reference
    spectrum. Both are peak-normalized before plotting -- see module
    docstring for why they can't be compared in native units.
    '''

    peak_nm = float(wavelength_nm[np.argmax(spectrum)])
    normalized_spectrum = spectrum / spectrum.max()
    normalized_reference = reference_spectrum / reference_spectrum.max()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelength_nm, normalized_spectrum, linewidth=1.5, color="steelblue", label="measured (this script)")
    ax.plot(
        reference_wavelength_nm, normalized_reference, linewidth=1.2, color="darkorange", alpha=0.8,
        label=f"reference: {reference_label}",
    )
    ax.axvline(peak_nm, color="firebrick", linestyle="--", linewidth=1, label=f"measured peak: {peak_nm:.2f} nm")

    # The reference instrument's wavelength range is much wider than the
    # imaging spectrometer's -- zoom to where the measured spectrum
    # actually has signal, rather than stretching the axis to fit the
    # reference's mostly-flat noise floor either side of it.
    margin_nm = 0.1 * (wavelength_nm.max() - wavelength_nm.min())
    ax.set_xlim(wavelength_nm.min() - margin_nm, wavelength_nm.max() + margin_nm)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized intensity (peak = 1)")
    ax.set_title("Beam spectrum vs. reference (geometric-tilt- and spectrally-corrected)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved beam spectrum plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lamp-shots", type=Path, nargs="+", default=list(DEFAULT_LAMP_SHOT_PATHS),
        help="Lamp frames both calibrations are built from "
             "(default: data/diagnostic/grating_rotation/l2.{1,2,3}.bmp)",
    )
    parser.add_argument(
        "--beam-shots", type=Path, nargs="+", default=list(DEFAULT_BEAM_SHOT_PATHS),
        help="Beam frames the calibrations are applied to "
             "(default: data/diagnostic/grating_rotation/2.{1,2,3}.bmp)",
    )
    parser.add_argument(
        "--degree", type=int, default=DEFAULT_SPECTRAL_DEGREE,
        help=f"Polynomial degree for the pixel -> wavelength_nm fit (default: {DEFAULT_SPECTRAL_DEGREE})",
    )
    parser.add_argument(
        "--reference-spectrum", type=Path, default=DEFAULT_REFERENCE_SPECTRUM_PATH,
        help="Independently-measured 'true' spectrum to overlay, in the raw spectrometer-export "
             f"format (default: {DEFAULT_REFERENCE_SPECTRUM_PATH})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save the beam spectrum plot",
    )
    parser.add_argument(
        "--gain-db", type=float, default=DEFAULT_GAIN_DB,
        help="Placeholder gain_db tagged on both calibrations (default: 0.0) -- see module docstring",
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
    beam_frames = [
        load_frame(path, frame_id=i, exposure_us=exposure_us, gain_db=args.gain_db)
        for i, path in enumerate(args.beam_shots)
    ]

    tilt = build_geometric_tilt_linear(lamp_frames)
    print_tilt_report(tilt)

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
    spectral = calibrate_spectral(
        pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm, record, degree=args.degree,
    )
    print_spectral_report(spectral)

    stacked_beam = ProcessedFrame(
        image=stack_frames(beam_frames), frame_id=0, timestamp=0.0,
        exposure_us=exposure_us, gain_db=args.gain_db,
    )
    beam_corrected = apply_geometric_tilt_correction(stacked_beam, tilt, include_residual=args.include_residual)

    columns = np.arange(beam_corrected.image.shape[1])
    beam_wavelength_nm = spectral.wavelength_nm(columns)
    beam_spectrum = beam_corrected.image.sum(axis=SPATIAL_AXIS)

    peak_nm = beam_wavelength_nm[np.argmax(beam_spectrum)]
    print(f"beam spectrum: peak at {peak_nm:.2f} nm, total intensity = {beam_spectrum.sum():.4g} ADU")

    reference_wavelength_nm, reference_spectrum = load_reference_spectrum(args.reference_spectrum)
    reference_peak_nm = reference_wavelength_nm[np.argmax(reference_spectrum)]
    print(f"reference spectrum ({args.reference_spectrum}): peak at {reference_peak_nm:.2f} nm")

    plot_beam_spectrum(
        beam_wavelength_nm, beam_spectrum, reference_wavelength_nm, reference_spectrum,
        args.reference_spectrum.name, args.output,
    )


if __name__ == "__main__":
    main()
