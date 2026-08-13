'''
Saves a complete, self-contained record of one ExtendedMeasurementScreen
"Run Measurement" result to the repo -- raw preprocessed frames (a subset,
see below), every shot's centroid/fit data, the combined result at every
fit degree, the calibration artifacts in effect, the ROI actually used,
and a journal-quality plot -- so a measurement can be inspected,
reproduced, or dropped into a report later, not just displayed once and
discarded. Triggered by "Save Record", a deliberately separate, explicit
action from "Run Measurement" (see extended_measurement.py's module
docstring) -- not every trial run should permanently commit a record.

Pure Python (matplotlib's Agg backend only, no Qt import at all) so this
is directly unit-testable without a QApplication -- same separation-of-
concerns already used for roi_control.py/formatting.py.

Frame storage: a stacked (mean) frame, plus the first/middle/last
individual shots by acquisition order -- not every shot (a default run is
20 shots, up to 1000; each preprocessed frame is ~18MB uncompressed). The
stack is a cheap, at-a-glance sanity check but isn't a quantity this
codebase's analysis ever actually computes (each shot is fit
individually, never pixel-averaged first) -- it can't help debug a
specific odd shot. The three individual frames are real, traceable
primary data (tied to a specific frame_id/ShotAnalysisResult) that reveal
within-run drift by direct comparison. Both together cost only 4 files
regardless of n_shots, so there's no reason to pick only one.

Per-degree combined results/fit lines are computed via
extended_measurement.py's compute_combined_result_for_degree()/
compute_fit_line_and_residuals() -- the exact same functions
ExtendedMeasurementScreen itself calls -- so every number in the saved
record is guaranteed identical to what the live GUI would show for that
degree, never independently re-derived logic that could drift from it.
Imported lazily by extended_measurement.py (inside
_on_save_record_clicked(), not at module scope) specifically so this
module can import those two functions back from extended_measurement.py
at its own module scope without a circular-import failure -- by the time
"Save Record" is ever clicked, extended_measurement.py has already
finished defining them (mirrors calibration/spectral/workflow.py's own
"see module docstring's import note" local-import pattern for the same
reason).

Calibration artifacts are copied byte-for-byte from artifact_dir (via
shutil.copy2()) rather than reconstructed from the in-memory
CalibrationSet/wavelength_axis/position_calibration objects -- simpler,
and exact regardless of whether every field needed to reconstruct a given
artifact type is actually retained in memory by the caller (e.g.
conversion gain's full PolynomialFitResult isn't -- only the derived
gain_e_per_adu float is, via SensorNoiseModel).
'''

# Imports

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; see module docstring (no Qt/display here)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pipeline.analysis import CombinedSpatialDispersionResult, ShotAnalysisResult  # noqa: E402
from pipeline.analysis.interfaces import WavelengthAxis  # noqa: E402
from pipeline.calibration.spatial import ScaleFactorPositionCalibration  # noqa: E402
from pipeline.cli.calibration import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BAD_PIXEL_MAP_FILENAME,
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_CONVERSION_GAIN_FILENAME,
    DEFAULT_FLAT_FIELD_FILENAME,
    DEFAULT_GEOMETRIC_TILT_FILENAME,
    DEFAULT_SCALE_FACTOR_FILENAME,
    DEFAULT_SPECTRAL_FILENAME,
)
from pipeline.preprocessing import CalibrationSet, ProcessedFrame  # noqa: E402

from .extended_measurement import (  # noqa: E402
    FIT_CURVE_N_POINTS,
    compute_combined_result_for_degree,
    compute_fit_line_and_residuals,
)
from .formatting import format_value_with_uncertainty, microns_to_mm  # noqa: E402
from .live_view import DEGREE_CHOICES, DEGREE_LABELS  # noqa: E402

# Constants

DEFAULT_MEASUREMENTS_DIR = Path("data/measurements")

# Display-only percentile clip for the quick-look frame PNGs -- same
# convention as scripts/save_tilt_diagnostic_frames.py's _to_uint8().
FRAME_DISPLAY_PERCENTILE = 99.5

# Journal-quality rcParams: serif/Computer-Modern mathtext (no real LaTeX
# install required -- usetex stays False), no gridlines, ticks inward on
# all four spines (standard PRL/APS look), no per-axes title anywhere
# (see _plot_journal_figure() -- axis labels carry the meaning instead).
# Applied via plt.rc_context() around figure creation only, so it never
# leaks into any other matplotlib usage in the same process.
JOURNAL_RC_PARAMS = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": False,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "font.size": 11,
    "legend.fontsize": 9,
    "legend.frameon": False,
}

# Distinguished by both color and linestyle (grayscale/colorblind-safe),
# same convention as scripts/compare_spectral_calibration_degrees.py's
# DEGREE_COLORS.
DEGREE_PLOT_COLORS = {1: "#1b7837", 2: "#2166ac", 3: "#b2182b"}
DEGREE_PLOT_LINESTYLES = {1: "-", 2: "--", 3: ":"}

# Every calibration artifact type this record copies, keyed by the label
# used in manifest.txt/the calibrations/ subdirectory -- values are the
# same DEFAULT_*_FILENAME constants cli/calibration.py and
# gui/calibration_screen.py already treat as canonical.
CALIBRATION_ARTIFACT_FILENAMES = {
    "baseline": DEFAULT_BASELINE_FILENAME,
    "flat_field": DEFAULT_FLAT_FIELD_FILENAME,
    "bad_pixel_map": DEFAULT_BAD_PIXEL_MAP_FILENAME,
    "conversion_gain": DEFAULT_CONVERSION_GAIN_FILENAME,
    "spectral": DEFAULT_SPECTRAL_FILENAME,
    "geometric_tilt": DEFAULT_GEOMETRIC_TILT_FILENAME,
    "scale_factor": DEFAULT_SCALE_FACTOR_FILENAME,
}

# Classes

@dataclass(frozen=True)
class _DegreeResult:

    '''One degree's combined result plus its drawn fit-line/residual arrays (pixel units) --
    bundles compute_combined_result_for_degree()'s and compute_fit_line_and_residuals()'s
    return values together so the rest of this module only has to pass one object around.'''

    combined: CombinedSpatialDispersionResult
    intercept_px: float
    intercept_sigma_px: float
    fit_x: np.ndarray
    fit_y_px: np.ndarray
    residual_px: np.ndarray


# Functions

def _convert_to_mm(
    value_px: np.ndarray, sigma_px: np.ndarray, position_calibration: ScaleFactorPositionCalibration,
) -> tuple[np.ndarray, np.ndarray]:

    '''Same conversion as ExtendedMeasurementScreen._convert_to_mm() -- position_calibration.convert()
    returns microns, everything in this module displays/reports in mm.'''

    value_um, sigma_um = position_calibration.convert(value_px, sigma_px)
    return microns_to_mm(value_um), microns_to_mm(sigma_um)


def _reduced_chi_squared(combined: CombinedSpatialDispersionResult) -> float:

    '''Same recovery combine_shots() itself doesn't expose directly -- see
    ExtendedMeasurementScreen._refresh_measurement_display()'s identical comment for the derivation.'''

    return (combined.sigma_external / combined.sigma_internal) ** 2


def _to_uint8(image: np.ndarray) -> np.ndarray:

    '''Percentile-stretches a float image to uint8 for a quick-look PNG -- display only, never the
    numerically authoritative record (that's the matching .npz). Same convention as
    scripts/save_tilt_diagnostic_frames.py's _to_uint8(), including its flat-image fallback.'''

    lo, hi = np.percentile(image, [1.0, FRAME_DISPLAY_PERCENTILE])
    if hi <= lo:
        return np.full(image.shape, 128, dtype=np.uint8)
    stretched = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def _save_frame(path_stem: Path, image: np.ndarray) -> None:

    '''Saves image as both path_stem.npz (float64, compressed, authoritative) and path_stem.png
    (uint8, percentile-stretched, quick-look only) -- compressed rather than raw .npy given how many
    of these images are mostly near-zero (see module docstring's frame-storage note).'''

    np.savez_compressed(path_stem.with_suffix(".npz"), image=image)
    iio.imwrite(path_stem.with_suffix(".png"), _to_uint8(image))


def _select_representative_frames(
    processed_frames: list[ProcessedFrame],
) -> list[tuple[str, ProcessedFrame]]:

    '''
    First, middle, and last frame by acquisition order -- see module
    docstring for why these three (not every shot). Collapses to fewer
    entries when there are fewer than 3 distinct indices (e.g. a 2-shot
    run) rather than saving the same frame twice under different labels.
    '''

    n = len(processed_frames)
    candidates = [("first", 0), ("middle", n // 2), ("last", n - 1)]
    seen_indices: set[int] = set()
    selected = []
    for label, index in candidates:
        if index in seen_indices:
            continue
        seen_indices.add(index)
        selected.append((label, processed_frames[index]))
    return selected


def _save_frames(record_dir: Path, processed_frames: list[ProcessedFrame]) -> None:

    frames_dir = record_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    stacked_image = np.mean([frame.image for frame in processed_frames], axis=0)
    _save_frame(frames_dir / "stacked", stacked_image)

    for label, frame in _select_representative_frames(processed_frames):
        _save_frame(frames_dir / f"shot_{label}_{frame.frame_id}", frame.image)


def _save_centroids(path: Path, shot_results: list[ShotAnalysisResult]) -> None:

    '''One .npz, keyed shot{i}_{field} -- see shared/io.py's own multi-array convention for the
    same "one file, several named arrays" shape used throughout calibration/.'''

    arrays: dict[str, np.ndarray] = {}
    for i, result in enumerate(shot_results):
        arrays[f"shot{i}_frame_id"] = np.array(result.frame_id)
        arrays[f"shot{i}_columns"] = result.centroids.columns
        arrays[f"shot{i}_x0"] = result.centroids.x0
        arrays[f"shot{i}_sigma_x0"] = result.centroids.sigma_x0
    np.savez_compressed(path, **arrays)


def _save_fits(path: Path, shot_results: list[ShotAnalysisResult]) -> None:

    '''One .npz, keyed shot{i}_degree{d}_{field} -- every per-shot, per-degree
    SpatialDispersionFitResult field, for every degree actually fit (result.fits.keys()).'''

    arrays: dict[str, np.ndarray] = {}
    for i, result in enumerate(shot_results):
        for degree, fit in result.fits.items():
            prefix = f"shot{i}_degree{degree}"
            arrays[f"{prefix}_coefficients"] = fit.coefficients
            arrays[f"{prefix}_coefficient_sigma"] = fit.coefficient_sigma
            arrays[f"{prefix}_coefficient_covariance"] = fit.coefficient_covariance
            arrays[f"{prefix}_reduced_chi_squared"] = np.array(fit.reduced_chi_squared)
            arrays[f"{prefix}_residuals"] = fit.residuals
            arrays[f"{prefix}_normalized_residuals"] = fit.normalized_residuals
    np.savez_compressed(path, **arrays)


def _copy_calibration_artifacts(artifact_dir: Path, destination_dir: Path) -> dict[str, bool]:

    '''Copies whichever calibration artifact files actually exist in artifact_dir into
    destination_dir, byte-for-byte (see module docstring for why a copy, not a reconstruction).
    Returns {artifact_name: was_copied}, for manifest.txt to report which ones are genuinely
    present versus absent/left at a default (e.g. no manual scale-factor override saved, or no
    geometric-tilt/spectral calibration built yet).'''

    destination_dir.mkdir(parents=True, exist_ok=True)
    present: dict[str, bool] = {}
    for name, filename in CALIBRATION_ARTIFACT_FILENAMES.items():
        source = artifact_dir / filename
        if source.exists():
            shutil.copy2(source, destination_dir / filename)
            present[name] = True
        else:
            present[name] = False
    return present


def _write_manifest(
    path: Path,
    timestamp_text: str,
    n_shots: int,
    exposure_us: float,
    gain_db: float,
    roi_bounds_px: tuple[int, int],
    roi_bounds_mm: tuple[float, float],
    spectral_column_bounds: tuple[int, int] | None,
    calibration_present: dict[str, bool],
) -> None:

    lines = [
        f"timestamp: {timestamp_text}",
        f"n_shots: {n_shots}",
        f"exposure_us: {exposure_us}",
        f"gain_db: {gain_db}",
        "",
        f"spatial ROI (px): [{roi_bounds_px[0]}, {roi_bounds_px[1]})",
        f"spatial ROI (mm): [{roi_bounds_mm[0]:.4f}, {roi_bounds_mm[1]:.4f})",
    ]
    if spectral_column_bounds is not None:
        lines.append(
            f"spectral ROI (manual override, as used at run time): "
            f"[{spectral_column_bounds[0]}, {spectral_column_bounds[1]})"
        )
    else:
        lines.append(
            "spectral ROI: automatic (signal-threshold SNR gate, no manual override was in effect) "
            "-- the realized per-shot column set is recorded exactly in centroids.npz regardless"
        )
    lines.append("")
    lines.append("calibration artifacts:")
    for name, filename in CALIBRATION_ARTIFACT_FILENAMES.items():
        status = f"copied ({filename})" if calibration_present[name] else "not present / default"
        lines.append(f"  {name}: {status}")

    path.write_text("\n".join(lines) + "\n")


def _write_combined_results(
    path: Path, per_degree: dict[int, _DegreeResult], position_calibration: ScaleFactorPositionCalibration,
) -> None:

    lines = []
    for degree in DEGREE_CHOICES:
        result = per_degree[degree]
        zeta_mm, zeta_sigma_mm = _convert_to_mm(
            np.array([result.combined.zeta_combined]), np.array([result.combined.sigma_zeta_combined]),
            position_calibration,
        )
        c0_mm, c0_sigma_mm = _convert_to_mm(
            np.array([result.intercept_px]), np.array([result.intercept_sigma_px]), position_calibration,
        )
        lines.append(f"degree {degree} ({DEGREE_LABELS[degree]}):")
        lines.append(f"  n_shots = {result.combined.n_shots}")
        lines.append(
            f"  c0 (mm) = {format_value_with_uncertainty(float(c0_mm[0]), float(c0_sigma_mm[0]))}"
        )
        lines.append(
            f"  c1 = spatial dispersion (mm/nm) = "
            f"{format_value_with_uncertainty(float(zeta_mm[0]), float(zeta_sigma_mm[0]))}"
        )
        lines.append(f"  reduced chi-squared = {_reduced_chi_squared(result.combined):.4g}")
        lines.append("")

    path.write_text("\n".join(lines))


def _plot_journal_figure(
    output_stem: Path,
    wavelength_nm: np.ndarray,
    y_values_mm: np.ndarray,
    y_sigma_mm: np.ndarray,
    x_sigma_nm: np.ndarray | None,
    per_degree: dict[int, _DegreeResult],
    position_calibration: ScaleFactorPositionCalibration,
) -> None:

    '''
    Top row: one subplot per degree (scatter + error bars + that degree's
    combined-zeta fit line). Bottom row: one combined residual panel, all
    three degrees overlaid. Journal styling per JOURNAL_RC_PARAMS -- no
    titles (axis labels only), no gridlines, colorblind/grayscale-safe
    color+linestyle pairing per degree. Saved as both .png (quick view)
    and .pdf (vector, for actual publication use).
    '''

    with plt.rc_context(JOURNAL_RC_PARAMS):
        fig = plt.figure(figsize=(12, 8))
        grid = fig.add_gridspec(2, 3, height_ratios=(2, 1), hspace=0.32, wspace=0.38)

        fit_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        residual_ax = fig.add_subplot(grid[1, :])

        for ax, degree in zip(fit_axes, DEGREE_CHOICES):
            result = per_degree[degree]

            ax.errorbar(
                wavelength_nm, y_values_mm, xerr=x_sigma_nm, yerr=y_sigma_mm,
                fmt="o", color="black", markersize=3, elinewidth=0.6, capsize=0, alpha=0.6, zorder=2,
            )

            fit_y_mm, _ = _convert_to_mm(result.fit_y_px, np.zeros_like(result.fit_y_px), position_calibration)
            zeta_mm, zeta_sigma_mm = _convert_to_mm(
                np.array([result.combined.zeta_combined]), np.array([result.combined.sigma_zeta_combined]),
                position_calibration,
            )
            ax.plot(
                result.fit_x, fit_y_mm, color=DEGREE_PLOT_COLORS[degree],
                linestyle=DEGREE_PLOT_LINESTYLES[degree], linewidth=1.5, zorder=3,
            )

            ax.set_xlabel(r"$\lambda$ (nm)")
            ax.set_ylabel(r"$x_0$ (mm)")

            # A steep, full-range-spanning fit line leaves no corner
            # reliably clear of data on both sides -- only the two
            # corners OFF the line's own diagonal are (e.g. top-left and
            # bottom-right for a positive slope). Picking the corner from
            # the sign of the fitted slope, rather than a fixed one,
            # keeps the annotation readable regardless of which way this
            # instrument's actual dispersion happens to run. A white
            # background patch is a second line of defense either way.
            annotation_in_upper_left = float(zeta_mm[0]) >= 0
            annotation_text = (
                f"{DEGREE_LABELS[degree]}\n"
                rf"$\zeta$ = {format_value_with_uncertainty(float(zeta_mm[0]), float(zeta_sigma_mm[0]))} mm/nm"
                "\n"
                rf"$\chi^2_\nu$ = {_reduced_chi_squared(result.combined):.3g}"
            )
            ax.text(
                0.04 if annotation_in_upper_left else 0.96, 0.96, annotation_text,
                transform=ax.transAxes, ha="left" if annotation_in_upper_left else "right", va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.35", facecolor="white",
                    edgecolor=DEGREE_PLOT_COLORS[degree], linewidth=0.8, alpha=0.92,
                ),
            )

        for degree in DEGREE_CHOICES:
            result = per_degree[degree]
            residual_mm, _ = _convert_to_mm(
                result.residual_px, np.zeros_like(result.residual_px), position_calibration,
            )
            residual_ax.plot(
                wavelength_nm, residual_mm, "o", color=DEGREE_PLOT_COLORS[degree], markersize=3, alpha=0.6,
                label=DEGREE_LABELS[degree],
            )
        residual_ax.axhline(0.0, color="black", linewidth=0.8)
        residual_ax.set_xlabel(r"$\lambda$ (nm)")
        residual_ax.set_ylabel(r"$x_0 - x_0^{\mathrm{fit}}$ (mm)")
        # frameon=True overrides JOURNAL_RC_PARAMS's legend.frameon=False
        # just for this legend -- residual scatter is noise-like rather
        # than a full-range diagonal, so there's no single corner
        # guaranteed clear of points the way there is for the fit
        # subplots above; a background patch is the simpler fix here.
        residual_ax.legend(
            loc="upper right", ncol=3, frameon=True, framealpha=0.92, facecolor="white", edgecolor="none",
        )

        fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def save_measurement_record(
    shot_results: list[ShotAnalysisResult],
    processed_frames: list[ProcessedFrame],
    calibration_set: CalibrationSet,
    wavelength_axis: WavelengthAxis | None,
    position_calibration: ScaleFactorPositionCalibration,
    axis_for_fit: WavelengthAxis,
    evaluated_at_wavelength_nm: float,
    roi_bounds_px: tuple[int, int],
    spectral_column_bounds: tuple[int, int] | None,
    exposure_us: float,
    gain_db: float,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    output_dir: Path = DEFAULT_MEASUREMENTS_DIR,
) -> Path:

    '''
    Writes one complete measurement record to
    output_dir/extended_measurement_<timestamp>/ -- see module docstring
    for the full rationale and directory layout.

    Parameters
    ----------
    shot_results, processed_frames
        One "Run Measurement" result -- same order/length, straight from
        ExtendedMeasurementScreen._shot_results/_processed_frames.
    calibration_set
        Only consulted for calibration_set.geometric_tilt's presence (via
        artifact_dir's file check below) -- the artifacts themselves are
        copied from disk, not from this object (see module docstring).
    wavelength_axis
        The real spectral calibration, or None if none is loaded (pixel-
        column fallback in effect) -- used only to report per-column
        wavelength uncertainty on the plot, when available.
    position_calibration
        Pixel -> physical-position conversion, for every mm-unit value
        this record reports.
    axis_for_fit
        Whichever WavelengthAxis analyze_shot() was actually fit against
        (wavelength_axis itself, or ExtendedMeasurementScreen's
        _PixelColumnWavelengthAxis fallback) -- used to recompute the
        flattened (shot, column) wavelength array
        compute_fit_line_and_residuals() needs, the same way
        ExtendedMeasurementScreen._set_measurement_data() does.
    evaluated_at_wavelength_nm
        The reference wavelength combine_combined_result_for_degree()
        evaluates each shot's degree > 1 zeta at before combining --
        ExtendedMeasurementScreen._evaluated_at_wavelength_nm()'s current
        value at save time.
    roi_bounds_px, spectral_column_bounds
        The spatial/spectral ROI actually used to produce shot_results --
        must be values captured at "Run Measurement" time (see
        extended_measurement.py's _set_measurement_data()), not read
        fresh from the live ROI controls, since either could have changed
        since the run this record is describing.
    exposure_us, gain_db
        The settings this measurement was acquired under.
    artifact_dir
        Where the calibration artifacts currently on disk live --
        defaults to the same DEFAULT_ARTIFACT_DIR every other real caller
        in this codebase treats as canonical. Exposed as a parameter
        (rather than hardcoded) so tests can point it at a tmp_path
        instead.
    output_dir
        Where the timestamped record directory is created.

    Returns
    -------
    Path
        The created record directory.
    '''

    timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
    record_dir = output_dir / f"extended_measurement_{time.strftime('%Y%m%d_%H%M%S')}"
    record_dir.mkdir(parents=True, exist_ok=True)

    _save_frames(record_dir, processed_frames)
    _save_centroids(record_dir / "centroids.npz", shot_results)
    _save_fits(record_dir / "fits.npz", shot_results)

    columns = np.concatenate([result.centroids.columns for result in shot_results])
    x0_px = np.concatenate([result.centroids.x0 for result in shot_results])
    sigma_x0_px = np.concatenate([result.centroids.sigma_x0 for result in shot_results])
    wavelength_nm = axis_for_fit.wavelength_nm(columns)
    x_range = (float(wavelength_nm.min()), float(wavelength_nm.max()))

    per_degree: dict[int, _DegreeResult] = {}
    for degree in DEGREE_CHOICES:
        combined = compute_combined_result_for_degree(shot_results, degree, evaluated_at_wavelength_nm)
        intercept_px, intercept_sigma_px, fit_x, fit_y_px, residual_px = compute_fit_line_and_residuals(
            x0_px, sigma_x0_px, wavelength_nm, combined.zeta_combined, x_range, FIT_CURVE_N_POINTS,
        )
        per_degree[degree] = _DegreeResult(
            combined=combined, intercept_px=intercept_px, intercept_sigma_px=intercept_sigma_px,
            fit_x=fit_x, fit_y_px=fit_y_px, residual_px=residual_px,
        )

    _write_combined_results(record_dir / "combined_results.txt", per_degree, position_calibration)

    calibration_present = _copy_calibration_artifacts(artifact_dir, record_dir / "calibrations")

    roi_bounds_mm_arr, _ = _convert_to_mm(
        np.array([roi_bounds_px[0], roi_bounds_px[1]], dtype=np.float64), np.zeros(2), position_calibration,
    )
    _write_manifest(
        record_dir / "manifest.txt", timestamp_text, len(shot_results), exposure_us, gain_db,
        roi_bounds_px, (float(roi_bounds_mm_arr[0]), float(roi_bounds_mm_arr[1])),
        spectral_column_bounds, calibration_present,
    )

    y_values_mm, y_sigma_mm = _convert_to_mm(x0_px, sigma_x0_px, position_calibration)
    x_sigma_nm = wavelength_axis.sigma_wavelength_nm(columns) if wavelength_axis is not None else None
    _plot_journal_figure(
        record_dir / "plot", wavelength_nm, y_values_mm, y_sigma_mm, x_sigma_nm, per_degree,
        position_calibration,
    )

    return record_dir


__all__ = ["save_measurement_record", "DEFAULT_MEASUREMENTS_DIR"]
