'''
Diagnostic: turns a set of preprocessed frames into an intensity-vs-
wavelength spectrum, by summing each frame over the spatial axis (axis 0)
to get one total intensity per spectral column, then mapping column ->
wavelength_nm via a saved spectral calibration. Lets the measured laser
spectrum be eyeballed against its known spectrum (e.g. Ti:Sapphire,
typically centered near 800nm) as a sanity check on the whole acquisition
+ preprocessing + spectral-calibration chain, independent of the spatial-
chirp measurement itself.

Input frames: data/diagnostic/geometric_tilt_correction/frame_NNNN_*.npy,
produced by scripts/save_tilt_diagnostic_frames.py. Each frame there was
saved three ways -- raw, corrected (full preprocessing INCLUDING
geometric-tilt correction), and uncorrected (same preprocessing, tilt
correction skipped). This script defaults to "corrected": that variant is
already the output of run_preprocessing() with a real geometric_tilt.npz
applied -- exactly what live view/extended measurement would have shown
for these frames -- so there is no reason to re-run preprocessing here
from the raw frames (and doing so is not straightforward as of this
writing besides: the raw frames were captured at gain_db=39.5, but
data/processed/calibration_artifacts_12.8.26/baseline.npz and
flat_field.npz were built at gain_db=25.0, so replaying them through
run_preprocessing() with that artifact dir would raise
SettingsMismatchError. That artifact dir's spectral.npz and
geometric_tilt.npz WERE built at gain_db=39.5, matching these frames --
only baseline/flat-field/bad-pixel-map are the mismatched ones, which is
why the already-preprocessed "corrected" frames are used directly instead
of rebuilding them here).

Geometric-tilt correction shifts pixels along the spatial axis per
column, so it should leave each column's spatial-axis sum roughly
unchanged (up to sub-pixel interpolation effects at the edges) --
--frame-variant uncorrected is available to check that directly.

Usage:
    python scripts/plot_column_spectrum.py
    python scripts/plot_column_spectrum.py --frame-variant uncorrected
    python scripts/plot_column_spectrum.py --frames-dir data/diagnostic/geometric_tilt_correction \\
        --spectral-calibration data/processed/calibration_artifacts_12.8.26/spectral.npz
'''

# Imports

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline.acquisition import SPATIAL_AXIS
from pipeline.calibration.spectral import load_spectral_calibration

# Constants

DEFAULT_FRAMES_DIR = Path("data/diagnostic/geometric_tilt_correction")
DEFAULT_SPECTRAL_CALIBRATION = Path("data/processed/calibration_artifacts_12.8.26/spectral.npz")
DEFAULT_OUTPUT_PATH = Path("data/processed/column_spectrum.png")
DEFAULT_FRAME_VARIANT = "corrected"

# Classes

# Functions


def load_frame_spectra(frames_dir: Path, variant: str) -> tuple[list[str], np.ndarray]:

    '''
    Loads every frame_NNNN_{variant}.npy in frames_dir and collapses each
    to a per-column intensity spectrum by summing over the spatial axis.

    Parameters
    ----------
    frames_dir
        Directory containing frame_NNNN_{variant}.npy files (see module
        docstring for the layout scripts/save_tilt_diagnostic_frames.py
        produces).
    variant
        Which saved variant to load: "raw", "corrected", or
        "uncorrected".

    Returns
    -------
    tuple[list[str], np.ndarray]
        Sorted frame labels (e.g. "frame_0000"), and an array of shape
        (n_frames, n_columns) of column-summed intensity.

    Raises
    ------
    FileNotFoundError
        If no matching files are found in frames_dir.
    '''

    paths = sorted(frames_dir.glob(f"frame_*_{variant}.npy"))
    if not paths:
        raise FileNotFoundError(f"no frame_*_{variant}.npy files found in {frames_dir}")

    labels = [path.stem.removesuffix(f"_{variant}") for path in paths]
    spectra = np.stack([np.load(path).sum(axis=SPATIAL_AXIS) for path in paths])
    return labels, spectra


def plot_spectra(
    labels: list[str], spectra: np.ndarray, wavelength_nm: np.ndarray,
    variant: str, frames_dir: Path, output_path: Path,
) -> None:

    '''
    Plots each frame's spectrum (thin lines) plus the mean across frames
    (thick line) against wavelength_nm, and saves the figure.

    Parameters
    ----------
    labels
        One label per row of spectra, for the legend.
    spectra
        Shape (n_frames, n_columns), column-summed intensity per frame.
    wavelength_nm
        Shape (n_columns,), wavelength for each column.
    variant
        Which frame variant was plotted -- included in the title.
    frames_dir
        Source directory -- included in the title for provenance.
    output_path
        Where to save the figure (PNG).
    '''

    mean_spectrum = spectra.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, spectrum in zip(labels, spectra):
        ax.plot(wavelength_nm, spectrum, linewidth=0.8, alpha=0.4, label=label)
    ax.plot(wavelength_nm, mean_spectrum, linewidth=2.0, color="black", label="mean")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Column-integrated intensity (ADU)")
    ax.set_title(f"Column-integrated spectrum -- {variant} frames from {frames_dir}")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def print_report(labels: list[str], spectra: np.ndarray, wavelength_nm: np.ndarray) -> None:

    '''Prints, per frame and for the mean, the peak wavelength and total integrated intensity.'''

    mean_spectrum = spectra.mean(axis=0)

    print(f"{len(labels)} frame(s), {wavelength_nm.shape[0]} spectral columns, "
          f"wavelength range [{wavelength_nm.min():.2f}, {wavelength_nm.max():.2f}] nm")
    print()
    for label, spectrum in zip(labels, spectra):
        peak_nm = wavelength_nm[np.argmax(spectrum)]
        print(f"{label}: peak at {peak_nm:.2f} nm, total intensity = {spectrum.sum():.4g} ADU")
    peak_nm = wavelength_nm[np.argmax(mean_spectrum)]
    print(f"mean: peak at {peak_nm:.2f} nm, total intensity = {mean_spectrum.sum():.4g} ADU")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR,
        help=f"Directory of frame_NNNN_{{variant}}.npy files (default: {DEFAULT_FRAMES_DIR})",
    )
    parser.add_argument(
        "--frame-variant", choices=("raw", "corrected", "uncorrected"), default=DEFAULT_FRAME_VARIANT,
        help=f"Which saved frame variant to use (default: {DEFAULT_FRAME_VARIANT})",
    )
    parser.add_argument(
        "--spectral-calibration", type=Path, default=DEFAULT_SPECTRAL_CALIBRATION,
        help=f"Path to a saved spectral calibration .npz (default: {DEFAULT_SPECTRAL_CALIBRATION})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"Where to save the plot PNG (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    calibration = load_spectral_calibration(args.spectral_calibration)
    labels, spectra = load_frame_spectra(args.frames_dir, args.frame_variant)

    n_columns = spectra.shape[1]
    columns = np.arange(n_columns)
    wavelength_nm = calibration.fit.evaluate(columns)

    print_report(labels, spectra, wavelength_nm)
    plot_spectra(labels, spectra, wavelength_nm, args.frame_variant, args.frames_dir, args.output)


if __name__ == "__main__":
    main()
