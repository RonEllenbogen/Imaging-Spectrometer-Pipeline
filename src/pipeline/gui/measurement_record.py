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

This function takes stacked_image/representative_frames already reduced
(a running mean plus up to 3 selected frames -- see
extended_measurement.py's _representative_shot_indices()), NOT a full
list of every shot's ProcessedFrame, and does not (any longer) reduce a
frame list itself. An earlier version took the full list and reduced it
here; that meant ExtendedMeasurementScreen had to retain every shot's
full-resolution image for the widget's entire lifetime just in case Save
Record was later clicked -- ~3.6GB resident for a real 200-shot run, with
a further multi-GB spike averaging that list in one call -- which caused
real, reported multi-minute UI freezes (here, and very likely a second,
unrelated-looking one switching the degree selector afterward too, from
lingering memory/GC pressure rather than a bug in that code path).
Reducing during acquisition instead, in extended_measurement.py, bounds
memory at a small constant regardless of n_shots.

Spatial dispersion (zeta = dx0/dwavelength_nm) at every degree is
computed via extended_measurement.py's compute_combined_result_for_degree()
-- the exact same function ExtendedMeasurementScreen itself calls -- so
that number is guaranteed identical to what the live GUI would show for
that degree (given the same reference wavelength), never independently
re-derived logic that could drift from it. It is evaluated at the
spectral ROI's center (see save_measurement_record()'s own docstring),
not the GUI's live "Evaluate At" value.

The polynomial *coefficients* reported alongside it are a different
computation, deliberately: zeta is a single derivative at one reference
point and only equals the polynomial's c1 coefficient when degree == 1
(see _DegreeResult's docstring for why conflating the two, as an earlier
version of this module did, is a real labeling bug, not just a cosmetic
one). Degree 1's coefficients reuse compute_fit_line_and_residuals()
(also imported from extended_measurement.py, unchanged, so its output
still matches the GUI exactly). Degree 2/3 use
extended_measurement.py's compute_combined_polynomial_for_degree()
instead -- combining every coefficient across shots the same
inverse-variance way, giving a genuine combined quadratic/cubic curve for
the plot to draw, rather than only ever a locally-linear tangent line at
degree > 1 (which is what an earlier version of this module drew, and
which is easy to mistake for a real bug since it makes every degree's
panel look like a straight line). ExtendedMeasurementScreen's own live
display now draws from the exact same function for degree > 1, so the
saved record's curve and the live GUI's curve are guaranteed identical,
not independently re-derived logic that could drift apart.

compute_combined_result_for_degree()/compute_combined_polynomial_for_degree()/
compute_fit_line_and_residuals() are imported lazily by
extended_measurement.py (inside _on_save_record_clicked(), not at module
scope) specifically so this module can import them back from
extended_measurement.py at its own module scope without a
circular-import failure -- by the time "Save Record" is ever clicked,
extended_measurement.py has already finished defining them (mirrors
calibration/spectral/workflow.py's own "see module docstring's import
note" local-import pattern for the same reason).

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
    aggregate_centroids_by_column,
    compute_combined_polynomial_for_degree,
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

    '''
    One degree's combined polynomial coefficients, its separately-reported
    spatial dispersion, and the drawn fit-curve/residual arrays (pixel
    units) -- bundles everything the rest of this module needs into one
    object per degree.

    coefficients_px/coefficient_sigma_px are the full combined polynomial
    (length degree + 1: c0..c_degree) -- NOT the same thing as
    spatial_dispersion below. spatial_dispersion is dx0/dwavelength_nm,
    the polynomial's *derivative* evaluated at one reference wavelength;
    it only equals c1 when degree == 1 (where the polynomial's derivative
    is a constant, c1, everywhere) -- at degree > 1 the two are genuinely
    different quantities and must not be conflated (an earlier version of
    this module's combined_results.txt labeled zeta_combined "c1" even at
    degree > 1, which was simply wrong).

    residual_px is every raw (shot, column) point's residual -- what the
    reduced chi-squared and every other statistic in combined_results.txt
    is computed from. display_residual_px is the same residuals reduced
    to one mean value per spectral column (see
    aggregate_centroids_by_column()), purely for _plot_journal_figure()'s
    residual panel -- plotting all ~10^5 raw residuals overplots into a
    solid mass with no visible points (see that function's docstring).
    '''

    coefficients_px: np.ndarray
    coefficient_sigma_px: np.ndarray
    spatial_dispersion: CombinedSpatialDispersionResult
    fit_x: np.ndarray
    fit_y_px: np.ndarray
    residual_px: np.ndarray
    display_residual_px: np.ndarray


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


def _save_frames(
    record_dir: Path, stacked_image: np.ndarray, representative_frames: dict[str, ProcessedFrame],
) -> None:

    '''
    Saves the already-reduced frame data extended_measurement.py's
    _on_run_clicked()/_representative_shot_indices() produced -- see
    module docstring for why this function no longer reduces a full
    per-shot frame list itself.
    '''

    frames_dir = record_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    _save_frame(frames_dir / "stacked", stacked_image)
    for label, frame in representative_frames.items():
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


# Unit label for polynomial coefficient c_k in x0_px = c0 + c1*wavelength_nm +
# c2*wavelength_nm**2 + ... -- position-like (mm) scaled by 1/nm**k.
def _coefficient_unit(k: int) -> str:
    if k == 0:
        return "mm"
    if k == 1:
        return "mm/nm"
    return f"mm/nm^{k}"


def _write_combined_results(
    path: Path, per_degree: dict[int, _DegreeResult], roi_center_wavelength_nm: float,
    position_calibration: ScaleFactorPositionCalibration,
) -> None:

    lines = [f"spatial dispersion below is evaluated at the spectral ROI's center, {roi_center_wavelength_nm:.3f} nm", ""]
    for degree in DEGREE_CHOICES:
        result = per_degree[degree]
        coefficients_mm, coefficient_sigma_mm = _convert_to_mm(
            result.coefficients_px, result.coefficient_sigma_px, position_calibration,
        )
        zeta_mm, zeta_sigma_mm = _convert_to_mm(
            np.array([result.spatial_dispersion.zeta_combined]),
            np.array([result.spatial_dispersion.sigma_zeta_combined]),
            position_calibration,
        )

        lines.append(f"degree {degree} ({DEGREE_LABELS[degree]}):")
        lines.append(f"  n_shots = {result.spatial_dispersion.n_shots}")
        lines.append("  combined polynomial coefficients (x0 = c0 + c1*wavelength_nm + c2*wavelength_nm^2 + ...):")
        for k, (c, sigma_c) in enumerate(zip(coefficients_mm, coefficient_sigma_mm)):
            lines.append(
                f"    c{k} ({_coefficient_unit(k)}) = {format_value_with_uncertainty(float(c), float(sigma_c))}"
            )
        # NOT the same as c1 above except at degree 1 -- see _DegreeResult's
        # own docstring for why these are reported as two distinct things.
        lines.append(
            f"  spatial dispersion at ROI center (mm/nm) = "
            f"{format_value_with_uncertainty(float(zeta_mm[0]), float(zeta_sigma_mm[0]))}"
        )
        lines.append(
            f"  reduced chi-squared (of the spatial-dispersion combination across shots) = "
            f"{_reduced_chi_squared(result.spatial_dispersion):.4g}"
        )
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
    combined fit curve). Bottom row: that SAME degree's own residual panel,
    directly underneath -- not one combined panel with all three degrees
    overlaid (an earlier version of this function did that; three series
    overlaid in one panel look like a single solid blob wherever the
    degrees roughly agree, which is often, since transparency alone can't
    separate near-identical overlapping distributions no matter how low
    it's set). All three residual panels share one y-axis range, so
    residual *magnitude* is still directly comparable by eye across
    degrees despite being in separate panels. Journal styling per
    JOURNAL_RC_PARAMS -- no titles (axis labels only), no gridlines,
    colorblind/grayscale-safe color+linestyle pairing per degree. Saved as
    both .png (quick view) and .pdf (vector, for actual publication use).

    wavelength_nm/y_values_mm/y_sigma_mm/x_sigma_nm and every
    per_degree[degree].display_residual_px are already reduced to one
    point per spectral column (aggregate_centroids_by_column(), called by
    save_measurement_record() before this function is reached) -- not the
    full raw (shot, column) data every statistic in combined_results.txt
    is computed from. A real 200-shot measurement has on the order of
    10^5 (shot, column) points per panel; plotting all of them individually
    overplots into a solid mass with every point and error bar completely
    indistinguishable from its neighbors (this function used to do exactly
    that). Reducing to one point per column bounds what's drawn to the
    column count (order 10^3, fixed by the ROI) regardless of shot count.

    The scatter/error-bar/residual layers are still rasterized in the PDF
    (via set_rasterization_zorder(), matplotlib's documented technique for
    mixed vector+raster figures -- everything below the given zorder
    rasterizes, everything at or above stays vector) while the fit lines,
    axes, text, and legend stay vector -- cheap insurance against a large
    point count even though, post-aggregation, generation time/file size
    is no longer the concern it was when this plotted every raw point (an
    earlier version's 10^5-point-per-panel PDFs took minutes to generate
    and were the direct cause of a reported Adobe Acrobat crash opening
    one). PNG output is already fully raster regardless, but gets the same
    zorder split for consistency -- it costs nothing there.
    '''

    with plt.rc_context(JOURNAL_RC_PARAMS):
        fig = plt.figure(figsize=(12, 9))
        # Nested gridspec, one per degree: an outer 1x3 row (normal
        # spacing between the three degrees) each split into its own
        # 2-row inner grid with hspace=0 (the fit panel and its own
        # residual panel touch, sharing the x-axis below) -- a single
        # flat hspace across the whole 2x3 grid can't express "these two
        # touch, but these two don't" at the same time.
        outer_grid = fig.add_gridspec(1, 3, wspace=0.4)
        fit_axes = []
        residual_axes = []
        for i in range(3):
            inner_grid = outer_grid[0, i].subgridspec(2, 1, height_ratios=(2, 1), hspace=0.0)
            fit_ax = fig.add_subplot(inner_grid[0])
            residual_ax = fig.add_subplot(inner_grid[1], sharex=fit_ax)
            fit_axes.append(fit_ax)
            residual_axes.append(residual_ax)

        residual_mm_by_degree = {
            degree: _convert_to_mm(
                per_degree[degree].display_residual_px,
                np.zeros_like(per_degree[degree].display_residual_px),
                position_calibration,
            )[0]
            for degree in DEGREE_CHOICES
        }
        # Shared y-range across all three residual panels (with a small
        # margin) -- separate panels solve the overlap problem, but
        # letting each auto-scale independently would defeat comparing
        # residual *magnitude* across degrees at a glance, which is the
        # main reason to look at all three together in the first place.
        all_residuals_mm = np.concatenate(list(residual_mm_by_degree.values()))
        residual_margin_mm = 0.05 * (all_residuals_mm.max() - all_residuals_mm.min())
        residual_ylim = (
            all_residuals_mm.min() - residual_margin_mm, all_residuals_mm.max() + residual_margin_mm,
        )

        for ax, residual_ax, degree in zip(fit_axes, residual_axes, DEGREE_CHOICES):
            result = per_degree[degree]

            # alpha kept modest even at this (post-aggregation) point
            # count -- adjacent columns' error bars still overlap along
            # the fit curve, and a lighter fill keeps that overlap's
            # density visible rather than a filled band.
            ax.errorbar(
                wavelength_nm, y_values_mm, xerr=x_sigma_nm, yerr=y_sigma_mm,
                fmt="o", color="black", markersize=3, elinewidth=0.6, capsize=0, alpha=0.5, zorder=1,
            )

            fit_y_mm, _ = _convert_to_mm(result.fit_y_px, np.zeros_like(result.fit_y_px), position_calibration)
            zeta_mm, zeta_sigma_mm = _convert_to_mm(
                np.array([result.spatial_dispersion.zeta_combined]),
                np.array([result.spatial_dispersion.sigma_zeta_combined]),
                position_calibration,
            )
            ax.plot(
                result.fit_x, fit_y_mm, color=DEGREE_PLOT_COLORS[degree],
                linestyle=DEGREE_PLOT_LINESTYLES[degree], linewidth=1.5, zorder=3,
            )
            # Rasterizes the errorbar layer (zorder=1) only -- the fit
            # line (zorder=3) and the annotation/text below (default
            # zorder, well above 2) stay vector.
            ax.set_rasterization_zorder(2)

            # No x-label/tick-labels here -- shares an x-axis with
            # residual_ax directly below (touching, sharex=ax above), so
            # the residual panel's own label/ticks below already cover it.
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_ylabel(r"$x_0$ (mm)")

            # A steep, full-range-spanning fit line/curve leaves no corner
            # reliably clear of data on both sides -- only the two
            # corners OFF the curve's own diagonal are (e.g. top-left and
            # bottom-right for a positive slope). Picking the corner from
            # the sign of the fitted dispersion, rather than a fixed one,
            # keeps the annotation readable regardless of which way this
            # instrument's actual dispersion happens to run. A white
            # background patch is a second line of defense either way.
            # zeta (not literally c1 except at degree 1 -- see
            # _DegreeResult's docstring) is reported separately from the
            # coefficients, which aren't summarized on the plot itself at
            # all (they're in combined_results.txt/fits.npz) -- there
            # isn't room here for up to 4 coefficients per panel, and zeta
            # is the one number that's directly comparable across degrees.
            annotation_in_upper_left = float(zeta_mm[0]) >= 0
            annotation_text = (
                f"{DEGREE_LABELS[degree]}\n"
                rf"$\zeta$ = {format_value_with_uncertainty(float(zeta_mm[0]), float(zeta_sigma_mm[0]))} mm/nm"
                "\n"
                rf"$\chi^2_\nu$ = {_reduced_chi_squared(result.spatial_dispersion):.3g}"
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

            # A single series per panel now (no more overlap to fight),
            # so a higher alpha than the old combined-panel version is
            # both fine and preferable -- shows point density better.
            residual_ax.plot(
                wavelength_nm, residual_mm_by_degree[degree], "o", color=DEGREE_PLOT_COLORS[degree],
                markersize=2.5, alpha=0.4, zorder=1,
            )
            residual_ax.axhline(0.0, color="black", linewidth=0.8, zorder=2)
            residual_ax.set_ylim(residual_ylim)
            # Rasterizes the residual-scatter layer (zorder=1) only -- the
            # zero line stays vector.
            residual_ax.set_rasterization_zorder(1.5)
            residual_ax.set_xlabel(r"$\lambda$ (nm)")
            residual_ax.set_ylabel(r"$x_0 - x_0^{\mathrm{fit}}$ (mm)")

        # dpi is passed explicitly to both -- for the PDF specifically,
        # this controls the resolution of the rasterized layers above
        # (matplotlib's savefig.dpi rcParam default otherwise applies,
        # which isn't guaranteed to match); 300 keeps them sharp without
        # the file size a much higher value would cost.
        fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_measurement_record(
    shot_results: list[ShotAnalysisResult],
    stacked_image: np.ndarray,
    representative_frames: dict[str, ProcessedFrame],
    calibration_set: CalibrationSet,
    wavelength_axis: WavelengthAxis | None,
    position_calibration: ScaleFactorPositionCalibration,
    axis_for_fit: WavelengthAxis,
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
    shot_results
        One "Run Measurement" result, straight from
        ExtendedMeasurementScreen._shot_results.
    stacked_image, representative_frames
        The already-reduced frame data for that same run, straight from
        ExtendedMeasurementScreen._measurement_stacked_image/
        _measurement_representative_frames -- a running mean and up to 3
        selected shots, not a full per-shot frame list (see module
        docstring for why this function takes them pre-reduced).
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
        ExtendedMeasurementScreen._set_measurement_data() does. Spatial
        dispersion (see combined_results.txt/the plot) is evaluated at
        the center of the realized spectral ROI -- (columns.min() +
        columns.max()) / 2 converted via this axis -- NOT at
        ExtendedMeasurementScreen's "Evaluate At" spin box value; that
        reference point is a live-GUI-only concept (a user-editable
        pixel column, defaulting to a fixed value unrelated to any given
        run's actual ROI) that doesn't carry over meaningfully to a
        saved record describing one specific measurement.
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

    _save_frames(record_dir, stacked_image, representative_frames)
    _save_centroids(record_dir / "centroids.npz", shot_results)
    _save_fits(record_dir / "fits.npz", shot_results)

    columns = np.concatenate([result.centroids.columns for result in shot_results])
    x0_px = np.concatenate([result.centroids.x0 for result in shot_results])
    sigma_x0_px = np.concatenate([result.centroids.sigma_x0 for result in shot_results])
    wavelength_nm = axis_for_fit.wavelength_nm(columns)
    x_range = (float(wavelength_nm.min()), float(wavelength_nm.max()))

    roi_center_column = (float(columns.min()) + float(columns.max())) / 2.0
    roi_center_wavelength_nm = float(axis_for_fit.wavelength_nm(np.array([roi_center_column]))[0])

    per_degree: dict[int, _DegreeResult] = {}
    for degree in DEGREE_CHOICES:
        spatial_dispersion = compute_combined_result_for_degree(
            shot_results, degree, roi_center_wavelength_nm,
        )
        if degree == 1:
            # Unchanged from before: the single global weighted-least-
            # squares line, anchored by the already-combined slope --
            # identical to what ExtendedMeasurementScreen itself draws
            # for degree 1 (see compute_fit_line_and_residuals()).
            intercept_px, intercept_sigma_px, fit_x, fit_y_px, residual_px = compute_fit_line_and_residuals(
                x0_px, sigma_x0_px, wavelength_nm, spatial_dispersion.zeta_combined, x_range, FIT_CURVE_N_POINTS,
            )
            coefficients_px = np.array([intercept_px, spatial_dispersion.zeta_combined])
            coefficient_sigma_px = np.array([intercept_sigma_px, spatial_dispersion.sigma_zeta_combined])
        else:
            # A genuine combined polynomial (see
            # compute_combined_polynomial_for_degree()'s own docstring) --
            # a real quadratic/cubic curve, not the single-reference-point
            # tangent line degree 1 uses above.
            coefficients_px, coefficient_sigma_px = compute_combined_polynomial_for_degree(shot_results, degree)
            fit_x = np.linspace(x_range[0], x_range[1], FIT_CURVE_N_POINTS)
            fit_y_px = np.polynomial.polynomial.polyval(fit_x, coefficients_px)
            residual_px = x0_px - np.polynomial.polynomial.polyval(wavelength_nm, coefficients_px)

        _, display_residual_px, _ = aggregate_centroids_by_column(columns, residual_px)
        per_degree[degree] = _DegreeResult(
            coefficients_px=coefficients_px, coefficient_sigma_px=coefficient_sigma_px,
            spatial_dispersion=spatial_dispersion, fit_x=fit_x, fit_y_px=fit_y_px, residual_px=residual_px,
            display_residual_px=display_residual_px,
        )

    _write_combined_results(
        record_dir / "combined_results.txt", per_degree, roi_center_wavelength_nm, position_calibration,
    )

    calibration_present = _copy_calibration_artifacts(artifact_dir, record_dir / "calibrations")

    roi_bounds_mm_arr, _ = _convert_to_mm(
        np.array([roi_bounds_px[0], roi_bounds_px[1]], dtype=np.float64), np.zeros(2), position_calibration,
    )
    _write_manifest(
        record_dir / "manifest.txt", timestamp_text, len(shot_results), exposure_us, gain_db,
        roi_bounds_px, (float(roi_bounds_mm_arr[0]), float(roi_bounds_mm_arr[1])),
        spectral_column_bounds, calibration_present,
    )

    # _plot_journal_figure() draws one point per spectral column (mean/std
    # of x0 across whichever shots reported that column), not every raw
    # (shot, column) point above -- see aggregate_centroids_by_column()'s
    # own docstring for why. Every statistic in combined_results.txt
    # (already written above) still comes from the full raw x0_px/
    # columns/wavelength_nm.
    display_columns, display_x0_px, display_x0_sigma_px = aggregate_centroids_by_column(columns, x0_px)
    display_wavelength_nm = axis_for_fit.wavelength_nm(display_columns)
    display_y_mm, display_y_sigma_mm = _convert_to_mm(display_x0_px, display_x0_sigma_px, position_calibration)
    display_x_sigma_nm = (
        wavelength_axis.sigma_wavelength_nm(display_columns) if wavelength_axis is not None else None
    )
    _plot_journal_figure(
        record_dir / "plot", display_wavelength_nm, display_y_mm, display_y_sigma_mm, display_x_sigma_nm,
        per_degree, position_calibration,
    )

    return record_dir


__all__ = ["save_measurement_record", "DEFAULT_MEASUREMENTS_DIR"]
