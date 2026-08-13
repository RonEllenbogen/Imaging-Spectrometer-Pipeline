'''
Test suite for measurement_record.py -- pure module tests, no Qt/
QApplication involved at all (see that module's own docstring for why),
against real shot_results/stacked_image/representative_frames acquired
the same collect_n_frames() -> run_preprocessing() -> analyze_shot() ->
running-mean-plus-representative-frames way
extended_measurement.py's _on_run_clicked() does, using
build_realistic_calibration_bundle() (tests/gui_fixture_helpers.py) for
the in-memory calibration state and a small set of freshly-saved
artifact files (not necessarily numerically identical to that bundle --
see _save_realistic_artifacts()) for save_measurement_record()'s
artifact_dir copying step.
'''

# Imports

import dataclasses
import time

import numpy as np
import pytest

from pipeline.acquisition import CameraStream, SyntheticBackend
from pipeline.analysis import SensorNoiseModel, ShotAnalysisResult, analyze_shot
from pipeline.calibration.sensor import (
    BaselineResult,
    build_conversion_gain,
    save_bad_pixel_map,
    save_baseline,
    save_conversion_gain,
    save_flat_field,
)
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.calibration.spectral import save_geometric_tilt, save_spectral_calibration
from pipeline.gui.extended_measurement import DEGREE_CHOICES, _representative_shot_indices
from pipeline.gui.measurement_record import CALIBRATION_ARTIFACT_FILENAMES, save_measurement_record
from pipeline.preprocessing import ProcessedFrame, run_preprocessing

from gui_fixture_helpers import (
    FIXTURE_EXPOSURE_US,
    FIXTURE_GAIN_DB,
    _conversion_gain_frames_by_exposure,
    build_realistic_calibration_bundle,
)

# Constants

N_SHOTS = 3

# Functions

def _acquire_real_measurement(
    bundle, n_shots: int = N_SHOTS, slope_px_per_col: float = 0.01,
) -> tuple[list[ShotAnalysisResult], np.ndarray, dict[str, ProcessedFrame]]:

    '''
    Real acquire -> preprocess -> analyze -> reduce loop, same shape as
    ExtendedMeasurementScreen._on_run_clicked() (running sum + selected
    representative frames, never a full per-shot frame list -- see that
    method's docstring for why) -- deliberately not going through that
    widget (this module has no Qt dependency to exercise). A small
    injected chirp (slope_px_per_col) keeps every degree's fit non-
    degenerate, same convention as
    tests/test_extended_measurement.py's _realistic_running_camera_stream().
    '''

    calibration_set = dataclasses.replace(bundle.calibration_set, background_sigma=1.0)
    stream = CameraStream(
        exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
        pixel_format="Mono8", timeout_ms=5000,
        backend=SyntheticBackend(seed=7, slope_px_per_col=slope_px_per_col, peak_counts=180.0, noise_std=3.0),
    )
    stream.start()
    try:
        frames = stream.collect_n_frames(n_shots)
    finally:
        stream.stop()

    representative_indices = _representative_shot_indices(len(frames))
    representative_frames: dict[str, ProcessedFrame] = {}
    frame_sum = None

    shot_results = []
    for i, frame in enumerate(frames):
        processed, _saturation = run_preprocessing(frame, calibration_set)
        shot_results.append(
            analyze_shot(
                processed, bundle.wavelength_axis, noise_model=bundle.noise_model, degrees=DEGREE_CHOICES,
            )
        )
        frame_sum = processed.image if frame_sum is None else frame_sum + processed.image
        for label, index in representative_indices.items():
            if i == index:
                representative_frames[label] = processed

    stacked_image = frame_sum / len(frames)
    return shot_results, stacked_image, representative_frames


def _save_realistic_artifacts(bundle, artifact_dir) -> None:

    '''
    Saves a small set of real (not necessarily numerically matching
    bundle's own calibration_set) calibration artifact files to
    artifact_dir -- exercises save_measurement_record()'s file-copying
    step against genuine save_*() output. scale_factor.npz is
    deliberately never written here, so tests can also check the
    "not present / default" manifest path for at least one artifact type.
    '''

    artifact_dir.mkdir(parents=True, exist_ok=True)
    record = CalibrationRecord(
        exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
        timestamp=time.time(), source_frame_count=3,
    )

    baseline_result = BaselineResult(
        baseline=bundle.calibration_set.baseline, background_sigma=1.0,
    )
    save_baseline(artifact_dir / CALIBRATION_ARTIFACT_FILENAMES["baseline"], baseline_result, record)
    save_flat_field(artifact_dir / CALIBRATION_ARTIFACT_FILENAMES["flat_field"], bundle.calibration_set.flat_field, record)
    save_bad_pixel_map(artifact_dir / CALIBRATION_ARTIFACT_FILENAMES["bad_pixel_map"], bundle.calibration_set.bad_pixel_mask, record)

    conversion_gain_result, conversion_gain_record = build_conversion_gain(_conversion_gain_frames_by_exposure())
    save_conversion_gain(
        artifact_dir / CALIBRATION_ARTIFACT_FILENAMES["conversion_gain"],
        conversion_gain_result, conversion_gain_record,
    )

    save_spectral_calibration(artifact_dir / CALIBRATION_ARTIFACT_FILENAMES["spectral"], bundle.wavelength_axis)
    save_geometric_tilt(artifact_dir / CALIBRATION_ARTIFACT_FILENAMES["geometric_tilt"], bundle.calibration_set.geometric_tilt)


# ---------------------------------------------------------------------------
# measurement_record.py -- save_measurement_record()
# ---------------------------------------------------------------------------

class TestSaveMeasurementRecord:

    def test_creates_expected_directory_structure(self, tmp_path):
        bundle = build_realistic_calibration_bundle()
        shot_results, stacked_image, representative_frames = _acquire_real_measurement(bundle)
        artifact_dir = tmp_path / "artifacts"
        _save_realistic_artifacts(bundle, artifact_dir)

        record_dir = save_measurement_record(
            shot_results, stacked_image, representative_frames, bundle.calibration_set, bundle.wavelength_axis,
            bundle.position_calibration, axis_for_fit=bundle.wavelength_axis,
            roi_bounds_px=(0, 1200), spectral_column_bounds=None,
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            artifact_dir=artifact_dir, output_dir=tmp_path / "measurements",
        )

        assert record_dir.is_dir()
        assert record_dir.parent == tmp_path / "measurements"

        assert (record_dir / "manifest.txt").is_file()
        assert (record_dir / "centroids.npz").is_file()
        assert (record_dir / "fits.npz").is_file()
        assert (record_dir / "combined_results.txt").is_file()
        assert (record_dir / "plot.png").is_file()
        assert (record_dir / "plot.pdf").is_file()
        assert (record_dir / "plot.png").stat().st_size > 0
        assert (record_dir / "plot.pdf").stat().st_size > 0

        frames_dir = record_dir / "frames"
        assert (frames_dir / "stacked.npz").is_file()
        assert (frames_dir / "stacked.png").is_file()
        # first/middle/last are distinct at N_SHOTS=3.
        shot_frame_files = sorted(p.name for p in frames_dir.glob("shot_*.npz"))
        assert len(shot_frame_files) == 3

        calibrations_dir = record_dir / "calibrations"
        for name in ("baseline", "flat_field", "bad_pixel_map", "conversion_gain", "spectral", "geometric_tilt"):
            assert (calibrations_dir / CALIBRATION_ARTIFACT_FILENAMES[name]).is_file()
        # scale_factor.npz was never written to artifact_dir -- confirms the
        # "absent artifact" path doesn't fabricate a file or crash.
        assert not (calibrations_dir / CALIBRATION_ARTIFACT_FILENAMES["scale_factor"]).exists()

    def test_absent_artifact_noted_in_manifest_not_copied(self, tmp_path):
        bundle = build_realistic_calibration_bundle()
        shot_results, stacked_image, representative_frames = _acquire_real_measurement(bundle)
        artifact_dir = tmp_path / "artifacts"
        _save_realistic_artifacts(bundle, artifact_dir)

        record_dir = save_measurement_record(
            shot_results, stacked_image, representative_frames, bundle.calibration_set, bundle.wavelength_axis,
            bundle.position_calibration, axis_for_fit=bundle.wavelength_axis,
            roi_bounds_px=(0, 1200), spectral_column_bounds=None,
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            artifact_dir=artifact_dir, output_dir=tmp_path / "measurements",
        )

        manifest_text = (record_dir / "manifest.txt").read_text()
        assert "scale_factor: not present / default" in manifest_text
        assert "baseline: copied" in manifest_text

    def test_manifest_reports_captured_roi_not_a_different_value(self, tmp_path):
        bundle = build_realistic_calibration_bundle()
        shot_results, stacked_image, representative_frames = _acquire_real_measurement(bundle)
        artifact_dir = tmp_path / "artifacts"
        _save_realistic_artifacts(bundle, artifact_dir)

        record_dir = save_measurement_record(
            shot_results, stacked_image, representative_frames, bundle.calibration_set, bundle.wavelength_axis,
            bundle.position_calibration, axis_for_fit=bundle.wavelength_axis,
            roi_bounds_px=(100, 900), spectral_column_bounds=(300, 1000),
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            artifact_dir=artifact_dir, output_dir=tmp_path / "measurements",
        )

        manifest_text = (record_dir / "manifest.txt").read_text()
        assert "spatial ROI (px): [100, 900)" in manifest_text
        assert "spectral ROI (manual override, as used at run time): [300, 1000)" in manifest_text

    def test_spectral_roi_none_reports_automatic_in_manifest(self, tmp_path):
        bundle = build_realistic_calibration_bundle()
        shot_results, stacked_image, representative_frames = _acquire_real_measurement(bundle)
        artifact_dir = tmp_path / "artifacts"
        _save_realistic_artifacts(bundle, artifact_dir)

        record_dir = save_measurement_record(
            shot_results, stacked_image, representative_frames, bundle.calibration_set, bundle.wavelength_axis,
            bundle.position_calibration, axis_for_fit=bundle.wavelength_axis,
            roi_bounds_px=(0, 1200), spectral_column_bounds=None,
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            artifact_dir=artifact_dir, output_dir=tmp_path / "measurements",
        )

        manifest_text = (record_dir / "manifest.txt").read_text()
        assert "spectral ROI: automatic" in manifest_text

    def test_centroids_round_trip(self, tmp_path):
        bundle = build_realistic_calibration_bundle()
        shot_results, stacked_image, representative_frames = _acquire_real_measurement(bundle)
        artifact_dir = tmp_path / "artifacts"
        _save_realistic_artifacts(bundle, artifact_dir)

        record_dir = save_measurement_record(
            shot_results, stacked_image, representative_frames, bundle.calibration_set, bundle.wavelength_axis,
            bundle.position_calibration, axis_for_fit=bundle.wavelength_axis,
            roi_bounds_px=(0, 1200), spectral_column_bounds=None,
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            artifact_dir=artifact_dir, output_dir=tmp_path / "measurements",
        )

        loaded = np.load(record_dir / "centroids.npz")
        for i, result in enumerate(shot_results):
            assert np.array_equal(loaded[f"shot{i}_columns"], result.centroids.columns)
            assert np.array_equal(loaded[f"shot{i}_x0"], result.centroids.x0)
            assert np.array_equal(loaded[f"shot{i}_sigma_x0"], result.centroids.sigma_x0)
            assert int(loaded[f"shot{i}_frame_id"]) == result.frame_id

    def test_fits_round_trip(self, tmp_path):
        bundle = build_realistic_calibration_bundle()
        shot_results, stacked_image, representative_frames = _acquire_real_measurement(bundle)
        artifact_dir = tmp_path / "artifacts"
        _save_realistic_artifacts(bundle, artifact_dir)

        record_dir = save_measurement_record(
            shot_results, stacked_image, representative_frames, bundle.calibration_set, bundle.wavelength_axis,
            bundle.position_calibration, axis_for_fit=bundle.wavelength_axis,
            roi_bounds_px=(0, 1200), spectral_column_bounds=None,
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            artifact_dir=artifact_dir, output_dir=tmp_path / "measurements",
        )

        loaded = np.load(record_dir / "fits.npz")
        for i, result in enumerate(shot_results):
            for degree, fit in result.fits.items():
                prefix = f"shot{i}_degree{degree}"
                assert np.array_equal(loaded[f"{prefix}_coefficients"], fit.coefficients)
                assert np.array_equal(loaded[f"{prefix}_coefficient_covariance"], fit.coefficient_covariance)
                assert float(loaded[f"{prefix}_reduced_chi_squared"]) == pytest.approx(fit.reduced_chi_squared)
                assert np.array_equal(loaded[f"{prefix}_residuals"], fit.residuals)

    def test_combined_results_match_shared_helper_directly(self, tmp_path):
        # The whole point of extracting compute_combined_result_for_degree()
        # (see extended_measurement.py) is that the saved record's numbers
        # can never drift from what the live GUI would show -- confirm the
        # saved combined_results.txt actually reflects that function's own
        # output (evaluated at the spectral ROI's center, as
        # save_measurement_record() itself computes it -- see its
        # docstring), not an independently re-derived value.
        from pipeline.gui.extended_measurement import compute_combined_result_for_degree
        from pipeline.gui.formatting import format_value_with_uncertainty, microns_to_mm

        bundle = build_realistic_calibration_bundle()
        shot_results, stacked_image, representative_frames = _acquire_real_measurement(bundle)
        artifact_dir = tmp_path / "artifacts"
        _save_realistic_artifacts(bundle, artifact_dir)

        record_dir = save_measurement_record(
            shot_results, stacked_image, representative_frames, bundle.calibration_set, bundle.wavelength_axis,
            bundle.position_calibration, axis_for_fit=bundle.wavelength_axis,
            roi_bounds_px=(0, 1200), spectral_column_bounds=None,
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            artifact_dir=artifact_dir, output_dir=tmp_path / "measurements",
        )

        columns = np.concatenate([result.centroids.columns for result in shot_results])
        roi_center_column = (float(columns.min()) + float(columns.max())) / 2.0
        roi_center_wavelength_nm = float(bundle.wavelength_axis.wavelength_nm(np.array([roi_center_column]))[0])

        combined_1 = compute_combined_result_for_degree(shot_results, 1, roi_center_wavelength_nm)
        zeta_mm, zeta_sigma_mm = bundle.position_calibration.convert(
            np.array([combined_1.zeta_combined]), np.array([combined_1.sigma_zeta_combined]),
        )
        text = (record_dir / "combined_results.txt").read_text()
        assert f"{roi_center_wavelength_nm:.3f} nm" in text
        assert f"n_shots = {combined_1.n_shots}" in text
        # The exact formatted string is reproduced independently rather than
        # substring-matched against a hand-typed number, to actually
        # exercise format_value_with_uncertainty() the same way the saved
        # file does.
        expected = format_value_with_uncertainty(
            microns_to_mm(float(zeta_mm[0])), microns_to_mm(float(zeta_sigma_mm[0])),
        )
        assert f"spatial dispersion at ROI center (mm/nm) = {expected}" in text
        # c0/c1 are listed as combined polynomial coefficients too, distinct
        # from the "spatial dispersion" line above.
        assert "combined polynomial coefficients" in text
        assert "c0 (mm) = " in text
        assert "c1 (mm/nm) = " in text
        # Degree 2/3 must list their own higher-order coefficients too --
        # this was the actual bug report (only c0/c1 were ever shown).
        assert "c2 (mm/nm^2) = " in text
        assert "c3 (mm/nm^3) = " in text

    def test_saved_stack_matches_the_stacked_image_passed_in(self, tmp_path):
        # stacked_image is now computed by the caller (a running mean
        # during acquisition -- see _acquire_real_measurement()/
        # ExtendedMeasurementScreen._on_run_clicked()), not reduced from a
        # frame list inside save_measurement_record() itself -- this just
        # confirms _save_frame() persists exactly what it was given.
        bundle = build_realistic_calibration_bundle()
        shot_results, stacked_image, representative_frames = _acquire_real_measurement(bundle)
        artifact_dir = tmp_path / "artifacts"
        _save_realistic_artifacts(bundle, artifact_dir)

        record_dir = save_measurement_record(
            shot_results, stacked_image, representative_frames, bundle.calibration_set, bundle.wavelength_axis,
            bundle.position_calibration, axis_for_fit=bundle.wavelength_axis,
            roi_bounds_px=(0, 1200), spectral_column_bounds=None,
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            artifact_dir=artifact_dir, output_dir=tmp_path / "measurements",
        )

        loaded_stack = np.load(record_dir / "frames" / "stacked.npz")["image"]
        assert np.allclose(loaded_stack, stacked_image)

    def test_two_shots_collapses_duplicate_representative_frame(self, tmp_path):
        # extended_measurement.py's _representative_shot_indices() dedup
        # path: with only 2 shots, "last" collides with "middle" (index
        # n-1 == n//2 at n=2) and should be dropped, not saved twice
        # under two different labels.
        bundle = build_realistic_calibration_bundle()
        shot_results, stacked_image, representative_frames = _acquire_real_measurement(bundle, n_shots=2)
        artifact_dir = tmp_path / "artifacts"
        _save_realistic_artifacts(bundle, artifact_dir)

        record_dir = save_measurement_record(
            shot_results, stacked_image, representative_frames, bundle.calibration_set, bundle.wavelength_axis,
            bundle.position_calibration, axis_for_fit=bundle.wavelength_axis,
            roi_bounds_px=(0, 1200), spectral_column_bounds=None,
            exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
            artifact_dir=artifact_dir, output_dir=tmp_path / "measurements",
        )

        shot_frame_files = sorted(p.name for p in (record_dir / "frames").glob("shot_*.npz"))
        assert len(shot_frame_files) == 2
