"""
Test suite for the analysis package. Every test operates on synthetic
ProcessedFrame objects (or bare arrays) with known, injected ground
truth -- no camera or real calibration data involved -- following the
same "prove against known ground truth first" principle used throughout
acquisition/'s and preprocessing/'s test suites.

Synthetic frames are built directly with numpy (mirroring
SyntheticBackend's generative model: Gaussian beam + injected linear
spatial chirp + optional Gaussian noise) rather than routed through
SyntheticBackend itself -- that avoids Mono8's 8-bit/255 ceiling, which
has nothing to do with what these tests are checking and would just add
clipping artifacts to work around.

Kept as a single file, mirroring test_preprocessing.py's convention.
"""

# Imports

import numpy as np
import pytest

from pipeline.acquisition import CANONICAL_SHAPE
from pipeline.preprocessing import ProcessedFrame

from pipeline.analysis import (
    analyze_shot, DEFAULT_DEGREES, combine_shots,
    CentroidEstimator, IntensityWeightedMoment, extract_centroids,
    SpatialDispersionFitter, TotalLeastSquaresFit,
    SensorNoiseModel,
    CentroidResult, SpatialDispersionFitResult,
    ShotAnalysisResult, CombinedSpatialDispersionResult,
    AnalysisError, InsufficientDataError,
)

# Constants

FIXTURE_EXPOSURE_US = 2000.0
FIXTURE_GAIN_DB = 0.0
FIXTURE_FRAME_ID = 0
FIXTURE_TIMESTAMP = 0.0


# Functions

def _synthetic_frame(
    centroid0_px=600.0, slope_px_per_col=0.0, beam_sigma_px=15.0,
    peak_counts=3000.0, noise_std=0.0, seed=0, frame_id=FIXTURE_FRAME_ID,
) -> ProcessedFrame:
    '''
    Builds a ProcessedFrame with a known injected Gaussian beam + linear
    spatial chirp, mirroring SyntheticBackend's generative model directly
    in float64 -- see module docstring for why this isn't routed through
    SyntheticBackend itself.
    '''
    rows, cols = CANONICAL_SHAPE
    row_axis = np.arange(rows, dtype=np.float64).reshape(-1, 1)
    col_axis = np.arange(cols, dtype=np.float64).reshape(1, -1)
    x0 = centroid0_px + slope_px_per_col * col_axis

    signal = peak_counts * np.exp(-((row_axis - x0) ** 2) / (2 * beam_sigma_px ** 2))
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_std, size=CANONICAL_SHAPE)
    else:
        noise = 0.0

    image = np.clip(signal + noise, 0, None)
    return ProcessedFrame(
        image=image, frame_id=frame_id, timestamp=FIXTURE_TIMESTAMP,
        exposure_us=FIXTURE_EXPOSURE_US, gain_db=FIXTURE_GAIN_DB,
    )


def _noise_model(gain=1.0, background_sigma=0.0) -> SensorNoiseModel:
    return SensorNoiseModel(gain_e_per_adu=gain, background_sigma=background_sigma)


class _LinearWavelengthAxis:
    '''Minimal WavelengthAxis stub: wavelength_nm = wavelength0_nm + dwavelength_nm_dpixel * pixel.'''

    def __init__(self, wavelength0_nm: float, dwavelength_nm_dpixel: float, sigma_wavelength_nm_value: float):
        self.wavelength0_nm = wavelength0_nm
        self.dwavelength_nm_dpixel = dwavelength_nm_dpixel
        self.sigma_wavelength_nm_value = sigma_wavelength_nm_value

    def wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:
        return self.wavelength0_nm + self.dwavelength_nm_dpixel * pixel.astype(np.float64)

    def sigma_wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:
        return np.full(pixel.shape, self.sigma_wavelength_nm_value, dtype=np.float64)


# Classes

# ---------------------------------------------------------------------------
# noise_model.py
# ---------------------------------------------------------------------------

class TestSensorNoiseModel:

    def test_valid_construction(self):
        model = _noise_model(gain=2.0, background_sigma=1.5)
        assert model.gain_e_per_adu == 2.0
        assert model.background_sigma == 1.5

    def test_non_positive_gain_raises(self):
        with pytest.raises(ValueError):
            _noise_model(gain=0.0)

    def test_negative_background_sigma_raises(self):
        with pytest.raises(ValueError):
            _noise_model(background_sigma=-1.0)


# ---------------------------------------------------------------------------
# centroiding.py
# ---------------------------------------------------------------------------

class TestIntensityWeightedMoment:

    def test_recovers_known_centroid_noiseless(self):
        positions = np.arange(1200, dtype=np.float64)
        true_x0 = 437.0
        sigma = 12.0
        column = 1000.0 * np.exp(-((positions - true_x0) ** 2) / (2 * sigma ** 2))

        x0, sigma_x0 = IntensityWeightedMoment().estimate(column, positions, _noise_model())

        assert x0 == pytest.approx(true_x0, abs=1e-6)
        assert np.isfinite(sigma_x0)
        assert sigma_x0 > 0

    def test_sigma_x0_decreases_with_more_signal(self):
        positions = np.arange(1200, dtype=np.float64)
        column = 100.0 * np.exp(-((positions - 600.0) ** 2) / (2 * 15.0 ** 2))
        noise_model = _noise_model()

        x0_weak, sigma_weak = IntensityWeightedMoment().estimate(column, positions, noise_model)
        x0_strong, sigma_strong = IntensityWeightedMoment().estimate(4.0 * column, positions, noise_model)

        # Scaling intensity by a constant factor leaves the moment ratio
        # (hence x0) unchanged, but more photon-equivalent signal must
        # tighten the TLW uncertainty.
        assert x0_strong == pytest.approx(x0_weak, abs=1e-9)
        assert sigma_strong < sigma_weak

    def test_higher_gain_tightens_uncertainty(self):
        positions = np.arange(1200, dtype=np.float64)
        column = 500.0 * np.exp(-((positions - 600.0) ** 2) / (2 * 15.0 ** 2))

        _, sigma_low_gain = IntensityWeightedMoment().estimate(column, positions, _noise_model(gain=1.0))
        _, sigma_high_gain = IntensityWeightedMoment().estimate(column, positions, _noise_model(gain=10.0))

        assert sigma_high_gain < sigma_low_gain

    def test_background_sigma_increases_uncertainty(self):
        positions = np.arange(1200, dtype=np.float64)
        column = 500.0 * np.exp(-((positions - 600.0) ** 2) / (2 * 15.0 ** 2))

        _, sigma_no_background = IntensityWeightedMoment().estimate(column, positions, _noise_model(background_sigma=0.0))
        _, sigma_with_background = IntensityWeightedMoment().estimate(column, positions, _noise_model(background_sigma=5.0))

        assert sigma_with_background > sigma_no_background


class TestExtractCentroids:

    def test_constant_centroid_recovered_across_all_columns(self):
        frame = _synthetic_frame(centroid0_px=600.0, slope_px_per_col=0.0)
        result = extract_centroids(frame, IntensityWeightedMoment(), _noise_model())

        assert isinstance(result, CentroidResult)
        assert result.x0.shape == (CANONICAL_SHAPE[1],)
        assert np.allclose(result.x0, 600.0, atol=0.05)
        assert np.all(np.isfinite(result.sigma_x0))
        assert np.all(result.sigma_x0 > 0)

    def test_linear_slope_recovered_across_columns(self):
        slope = 0.05
        frame = _synthetic_frame(centroid0_px=600.0, slope_px_per_col=slope)
        result = extract_centroids(frame, IntensityWeightedMoment(), _noise_model())

        expected = 600.0 + slope * result.columns
        assert np.allclose(result.x0, expected, atol=0.05)

    def test_result_arrays_are_read_only(self):
        frame = _synthetic_frame()
        result = extract_centroids(frame, IntensityWeightedMoment(), _noise_model())
        with pytest.raises(ValueError):
            result.x0[0] = 0.0

    def test_valid_columns_none_is_unchanged_behavior(self):
        # Regression test: a frame with valid_columns=None (its default)
        # must behave exactly as before this gate was added -- every
        # column processed, none skipped.
        frame = _synthetic_frame(centroid0_px=600.0, slope_px_per_col=0.0)
        assert frame.valid_columns is None

        result = extract_centroids(frame, IntensityWeightedMoment(), _noise_model())

        assert result.columns.shape == (CANONICAL_SHAPE[1],)
        assert np.array_equal(result.columns, np.arange(CANONICAL_SHAPE[1]))
        assert np.allclose(result.x0, 600.0, atol=0.05)

    def test_valid_columns_mask_shrinks_result_to_valid_subset(self):
        frame = _synthetic_frame(centroid0_px=600.0, slope_px_per_col=0.0)

        n_columns = CANONICAL_SHAPE[1]
        valid_columns = np.zeros(n_columns, dtype=bool)
        valid_columns[100:200] = True   # only columns [100, 200) are valid
        masked_frame = ProcessedFrame(
            image=frame.image, frame_id=frame.frame_id, timestamp=frame.timestamp,
            exposure_us=frame.exposure_us, gain_db=frame.gain_db, valid_columns=valid_columns,
        )

        result = extract_centroids(masked_frame, IntensityWeightedMoment(), _noise_model())

        assert result.columns.shape == (100,)
        assert np.array_equal(result.columns, np.arange(100, 200))
        assert result.x0.shape == (100,)
        assert result.sigma_x0.shape == (100,)
        assert np.allclose(result.x0, 600.0, atol=0.05)


# ---------------------------------------------------------------------------
# results.py
# ---------------------------------------------------------------------------

class TestCentroidResult:

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            CentroidResult(columns=np.arange(5), x0=np.zeros(4), sigma_x0=np.ones(5))


class TestSpatialDispersionFitResult:

    def test_linear_zeta_is_constant(self):
        result = SpatialDispersionFitResult(
            degree=1,
            coefficients=np.array([10.0, 0.5]),
            coefficient_sigma=np.array([0.1, 0.01]),
            reduced_chi_squared=1.0,
            residuals=np.zeros(3),
            normalized_residuals=np.zeros(3),
        )
        wavelength_nm = np.array([0.0, 10.0, 100.0])
        assert np.allclose(result.zeta(wavelength_nm), 0.5)

    def test_quadratic_zeta_varies_with_wavelength(self):
        # x0 = 10 + 0.5*wavelength_nm + 2*wavelength_nm^2
        # -> dx0/dwavelength_nm = 0.5 + 4*wavelength_nm
        result = SpatialDispersionFitResult(
            degree=2,
            coefficients=np.array([10.0, 0.5, 2.0]),
            coefficient_sigma=np.array([0.1, 0.01, 0.01]),
            reduced_chi_squared=1.0,
            residuals=np.zeros(3),
            normalized_residuals=np.zeros(3),
        )
        wavelength_nm = np.array([0.0, 1.0, 2.0])
        assert np.allclose(result.zeta(wavelength_nm), 0.5 + 4.0 * wavelength_nm)

    def test_degree_coefficient_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            SpatialDispersionFitResult(
                degree=1,
                coefficients=np.array([1.0, 2.0, 3.0]),
                coefficient_sigma=np.array([0.1, 0.1, 0.1]),
                reduced_chi_squared=1.0,
                residuals=np.zeros(3),
                normalized_residuals=np.zeros(3),
            )

    def test_degree_below_one_raises(self):
        with pytest.raises(ValueError):
            SpatialDispersionFitResult(
                degree=0,
                coefficients=np.array([1.0]),
                coefficient_sigma=np.array([0.1]),
                reduced_chi_squared=1.0,
                residuals=np.zeros(3),
                normalized_residuals=np.zeros(3),
            )


class TestShotAnalysisResultImmutability:

    def test_fits_dict_is_read_only(self):
        centroids = CentroidResult(columns=np.arange(3), x0=np.zeros(3), sigma_x0=np.ones(3))
        fit = SpatialDispersionFitResult(
            degree=1, coefficients=np.array([0.0, 1.0]),
            coefficient_sigma=np.array([0.1, 0.1]), reduced_chi_squared=1.0,
            residuals=np.zeros(3), normalized_residuals=np.zeros(3),
        )
        result = ShotAnalysisResult(frame_id=0, centroids=centroids, fits={1: fit})
        with pytest.raises(TypeError):
            result.fits[2] = fit


# ---------------------------------------------------------------------------
# dispersion_fitting.py
# ---------------------------------------------------------------------------

class TestTotalLeastSquaresFit:

    def test_recovers_known_linear_slope(self):
        rng = np.random.default_rng(1)
        n = 200
        true_intercept, true_slope = 500.0, 2.0
        wavelength_nm = np.linspace(10.0, 20.0, n)
        sigma_wavelength_nm = np.full(n, 0.01)
        sigma_x0 = np.full(n, 0.05)
        x0 = (
            true_intercept + true_slope * wavelength_nm
            + rng.normal(0.0, sigma_x0)
            + rng.normal(0.0, sigma_wavelength_nm) * true_slope
        )

        result = TotalLeastSquaresFit().fit(wavelength_nm, x0, sigma_wavelength_nm, sigma_x0, degree=1)

        assert isinstance(result, SpatialDispersionFitResult)
        assert result.coefficients[1] == pytest.approx(true_slope, abs=5 * result.coefficient_sigma[1])
        assert result.reduced_chi_squared < 3.0   # generous -- single noisy realization

    def test_insufficient_data_raises(self):
        wavelength_nm = np.array([1.0])
        x0 = np.array([1.0])
        sigma = np.array([0.1])
        with pytest.raises(InsufficientDataError):
            TotalLeastSquaresFit().fit(wavelength_nm, x0, sigma, sigma, degree=1)

    def test_residuals_shape_matches_input(self):
        wavelength_nm = np.linspace(0.0, 10.0, 20)
        x0 = 5.0 + 0.3 * wavelength_nm
        sigma = np.full(20, 0.1)

        result = TotalLeastSquaresFit().fit(wavelength_nm, x0, sigma, sigma, degree=1)

        assert result.residuals.shape == wavelength_nm.shape
        assert result.normalized_residuals.shape == wavelength_nm.shape


# ---------------------------------------------------------------------------
# combination.py
# ---------------------------------------------------------------------------

class TestCombineShots:

    def test_consistent_shots_combine_near_true_value(self):
        rng = np.random.default_rng(2)
        true_zeta = 0.015
        sigma = 0.001
        zeta_values = rng.normal(true_zeta, sigma, size=50)
        sigma_values = np.full(50, sigma)

        result = combine_shots(zeta_values, sigma_values)

        assert isinstance(result, CombinedSpatialDispersionResult)
        assert result.zeta_combined == pytest.approx(true_zeta, abs=5 * result.sigma_zeta_combined)
        assert result.n_shots == 50
        # Consistent shots -> internal and external error should be
        # comparable, not wildly different.
        assert result.sigma_external / result.sigma_internal < 3.0

    def test_jittery_shots_widen_reported_uncertainty(self):
        # Each shot claims a tiny uncertainty, but the shots themselves
        # disagree far more than that -- simulated pointing jitter.
        zeta_values = np.array([0.010, 0.030, 0.005, 0.040, 0.012])
        sigma_values = np.full(5, 0.0005)

        result = combine_shots(zeta_values, sigma_values)

        assert result.sigma_external > result.sigma_internal
        assert result.sigma_zeta_combined == pytest.approx(result.sigma_external)

    def test_single_shot_falls_back_to_internal(self):
        result = combine_shots(np.array([0.02]), np.array([0.001]))
        assert result.sigma_external == pytest.approx(result.sigma_internal)
        assert result.sigma_zeta_combined == pytest.approx(result.sigma_internal)
        assert result.n_shots == 1

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            combine_shots(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# analysis_pipeline.py
# ---------------------------------------------------------------------------

class TestAnalyzeShot:

    def test_end_to_end_recovers_known_spatial_dispersion(self):
        slope_px_per_col = 0.03
        dwavelength_nm_dpixel = 2.0
        expected_zeta = slope_px_per_col / dwavelength_nm_dpixel

        frame = _synthetic_frame(
            centroid0_px=600.0, slope_px_per_col=slope_px_per_col,
            noise_std=1.0, seed=42,
        )
        wavelength_axis = _LinearWavelengthAxis(
            wavelength0_nm=100.0, dwavelength_nm_dpixel=dwavelength_nm_dpixel, sigma_wavelength_nm_value=0.01,
        )

        result = analyze_shot(frame, wavelength_axis, degrees=(1,))

        assert isinstance(result, ShotAnalysisResult)
        assert result.frame_id == frame.frame_id
        fit = result.fits[1]
        # 1% relative tolerance rather than the fit's own formal
        # coefficient_sigma -- the injected synthetic noise (flat
        # additive Gaussian) doesn't match TLW's assumed noise model
        # exactly, so the two aren't expected to agree to the formal
        # uncertainty; this checks the pipeline recovers the true
        # injected ground truth to good precision, not that TLW's noise
        # model matches this specific synthetic injection (a separate,
        # deeper question -- see project_state.md's MC/bootstrap item).
        assert fit.coefficients[1] == pytest.approx(expected_zeta, rel=0.01)

    def test_multiple_degrees_all_present(self):
        frame = _synthetic_frame(centroid0_px=600.0, slope_px_per_col=0.02, noise_std=0.5, seed=7)
        wavelength_axis = _LinearWavelengthAxis(wavelength0_nm=50.0, dwavelength_nm_dpixel=1.0, sigma_wavelength_nm_value=0.01)

        result = analyze_shot(frame, wavelength_axis, degrees=(1, 2, 3))

        assert set(result.fits.keys()) == {1, 2, 3}
        for degree, fit in result.fits.items():
            assert fit.degree == degree

    def test_default_degrees_is_linear_only(self):
        assert DEFAULT_DEGREES == (1,)
