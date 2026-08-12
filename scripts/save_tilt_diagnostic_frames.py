'''
Diagnostic tool: grabs real frames from the camera and saves each one
three ways -- raw, preprocessed WITH geometric-tilt correction applied
(exactly what live view/extended measurement actually see), and
preprocessed WITHOUT it (same baseline/flat-field/bad-pixel-map/signal-
threshold steps, geometric tilt skipped) -- plus a corrected-minus-
uncorrected diff, so the effect of the correction on a real beam shot can
be inspected directly rather than inferred from code alone.

Motivating question: does a near-horizontal streak visible in live view,
running parallel to the fitted spatial-dispersion line, represent real
spatial chirp, or is it residual spectrometer-induced tilt that the
geometric-tilt correction isn't fully removing? run_preprocessing() DOES
conditionally apply apply_geometric_tilt_correction() whenever
CalibrationSet.geometric_tilt is not None, and both LiveViewWidget and
ExtendedMeasurementScreen reuse one CalibrationSet built once at startup
(see gui/app.py) rather than rebuilding their own -- so the correction is
confirmed to run every tick, given a real geometric_tilt.npz on disk.
This script doesn't test whether the correction runs (already confirmed
by that code path); it's for inspecting whether the *measured* row_shift
curve is accurate enough to actually straighten a real beam feature, by
comparing corrected vs. uncorrected output side by side.

Loads the full CalibrationSet from --artifact-dir exactly the way
gui/calibration_screen.py's _attempt_load_existing_calibrations() does
(same load_*() functions, same default filenames), so this exercises the
identical artifacts a live session would use -- not a re-derived or
approximated calibration.

Output layout, under --output-dir (default: a fresh timestamped folder
under data/diagnostics/):
    frame_0000_raw.npy / .png
    frame_0000_corrected.npy / .png     (geometric tilt applied)
    frame_0000_uncorrected.npy / .png   (geometric tilt skipped)
    frame_0000_diff.npy / .png          (corrected - uncorrected)
    ... one set per captured frame ...
    summary.txt

.npy files are the numerically authoritative record (float64, unclamped);
.png files are a percentile-stretched uint8 rendering for quick visual
inspection only.

Usage:
    python scripts/save_tilt_diagnostic_frames.py --gain-db 39.5 --exposure-us 1000
    python scripts/save_tilt_diagnostic_frames.py --gain-db 39.5 --exposure-us 1000 --n-frames 10
    python scripts/save_tilt_diagnostic_frames.py --gain-db 39.5 --auto-exposure
'''

# Imports

import argparse
import dataclasses
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pipeline.calibration.sensor import load_baseline, load_bad_pixel_map, load_flat_field
from pipeline.calibration.spectral import load_geometric_tilt
from pipeline.cli.calibration import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BAD_PIXEL_MAP_FILENAME,
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_FLAT_FIELD_FILENAME,
    DEFAULT_GEOMETRIC_TILT_FILENAME,
    build_camera_stream,
)
from pipeline.preprocessing import CalibrationSet, run_preprocessing

# Constants

DEFAULT_N_FRAMES = 5
DEFAULT_DIAGNOSTICS_ROOT = Path("data/diagnostics")

# Classes

# Functions


def _load_calibration_set(artifact_dir: Path) -> CalibrationSet:

    '''
    Loads baseline/flat-field/bad-pixel-map/geometric-tilt from
    artifact_dir, exactly the artifacts (and default filenames)
    gui/calibration_screen.py's WelcomePage/CreatePage load from. Unlike
    that loader, geometric_tilt.npz is REQUIRED here, not optional --
    this script exists specifically to inspect its effect, so a missing
    file is a setup error, not a silently-accepted "correction skipped"
    default.

    Parameters
    ----------
    artifact_dir
        Directory containing the saved calibration artifacts.

    Returns
    -------
    CalibrationSet
        With geometric_tilt populated (never None).

    Raises
    ------
    FileNotFoundError
        If any required artifact -- including geometric_tilt.npz -- is
        missing from artifact_dir.
    '''

    baseline_result, baseline_record = load_baseline(artifact_dir / DEFAULT_BASELINE_FILENAME)
    flat_field, flat_field_record = load_flat_field(artifact_dir / DEFAULT_FLAT_FIELD_FILENAME)
    bad_pixel_mask, _ = load_bad_pixel_map(artifact_dir / DEFAULT_BAD_PIXEL_MAP_FILENAME)
    geometric_tilt = load_geometric_tilt(artifact_dir / DEFAULT_GEOMETRIC_TILT_FILENAME)

    return CalibrationSet(
        baseline=baseline_result.baseline,
        baseline_record=baseline_record,
        flat_field=flat_field,
        flat_field_record=flat_field_record,
        bad_pixel_mask=bad_pixel_mask,
        background_sigma=baseline_result.background_sigma,
        geometric_tilt=geometric_tilt,
    )


def _to_uint8(image: np.ndarray) -> np.ndarray:

    '''
    Percentile-stretches a float image to uint8 for a quick-look PNG --
    display only, never the numerically authoritative record (that's the
    matching .npy file). A flat (zero-range) image maps to mid-grey
    rather than dividing by zero.

    Parameters
    ----------
    image
        Any real-valued 2D array.

    Returns
    -------
    np.ndarray
        Same shape, dtype uint8.
    '''

    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi <= lo:
        return np.full(image.shape, 128, dtype=np.uint8)
    stretched = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def _save_pair(path_stem: Path, image: np.ndarray) -> None:
    '''Saves image as both path_stem.npy (float64, authoritative) and
    path_stem.png (uint8, percentile-stretched, for quick viewing).'''
    np.save(path_stem.with_suffix(".npy"), image)
    iio.imwrite(path_stem.with_suffix(".png"), _to_uint8(image))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gain-db", type=float, required=True, help="Sensor gain in dB.")
    exposure_group = parser.add_mutually_exclusive_group(required=True)
    exposure_group.add_argument("--exposure-us", type=float, help="Fixed exposure time in microseconds.")
    exposure_group.add_argument(
        "--auto-exposure", action="store_true", help="Run one-time auto-exposure convergence instead."
    )
    parser.add_argument(
        "--n-frames", type=int, default=DEFAULT_N_FRAMES,
        help=f"Number of frames to capture and save (default: {DEFAULT_N_FRAMES}).",
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR,
        help=f"Directory to load baseline/flat-field/bad-pixel-map/geometric-tilt from "
             f"(default: {DEFAULT_ARTIFACT_DIR}, same as the GUI).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save the diagnostic frames (default: a fresh timestamped folder "
             f"under {DEFAULT_DIAGNOSTICS_ROOT}).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (DEFAULT_DIAGNOSTICS_ROOT / time.strftime("tilt_check_%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_with_tilt = _load_calibration_set(args.artifact_dir)
    calibration_without_tilt = dataclasses.replace(calibration_with_tilt, geometric_tilt=None)

    camera_stream = build_camera_stream(
        args.gain_db, exposure_us=args.exposure_us, auto_exposure=args.auto_exposure,
    )
    camera_stream.start()
    try:
        frames = camera_stream.collect_n_frames(args.n_frames)
    finally:
        camera_stream.stop()

    print(f"captured {len(frames)} frame(s) at exposure_us={frames[0].exposure_us}, "
          f"gain_db={frames[0].gain_db}")
    print(f"geometric_tilt loaded from {args.artifact_dir / DEFAULT_GEOMETRIC_TILT_FILENAME}: "
          f"reference_row={calibration_with_tilt.geometric_tilt.reference_row}, "
          f"row_shift range=[{calibration_with_tilt.geometric_tilt.row_shift.min():.3f}, "
          f"{calibration_with_tilt.geometric_tilt.row_shift.max():.3f}] px")

    summary_lines = [
        f"gain_db={args.gain_db}",
        f"exposure_us={args.exposure_us if args.exposure_us is not None else 'auto'}",
        f"artifact_dir={args.artifact_dir}",
        f"n_frames={len(frames)}",
        "",
    ]

    for i, frame in enumerate(frames):
        corrected, _ = run_preprocessing(frame, calibration_with_tilt)
        uncorrected, _ = run_preprocessing(frame, calibration_without_tilt)
        diff = corrected.image - uncorrected.image

        stem = output_dir / f"frame_{i:04d}"
        _save_pair(stem.with_name(stem.name + "_raw"), frame.image.astype(np.float64))
        _save_pair(stem.with_name(stem.name + "_corrected"), corrected.image)
        _save_pair(stem.with_name(stem.name + "_uncorrected"), uncorrected.image)
        _save_pair(stem.with_name(stem.name + "_diff"), diff)

        summary_lines.append(
            f"frame {i}: frame_id={frame.frame_id}, timestamp={frame.timestamp:.3f}, "
            f"max|diff|={np.max(np.abs(diff)):.3f}"
        )
        print(f"saved frame {i} (frame_id={frame.frame_id}), max|corrected-uncorrected|="
              f"{np.max(np.abs(diff)):.3f}")

    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"\nsaved {len(frames)} frame set(s) to {output_dir}")


if __name__ == "__main__":
    main()
