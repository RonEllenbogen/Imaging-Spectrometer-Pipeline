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

from pathlib import Path

import numpy as np
import pytest

from pipeline.acquisition import CANONICAL_SHAPE
from pipeline.preprocessing import ProcessedFrame

from pipeline.analysis import (
    analyze_shot, DEFAULT_DEGREES, combine_shots,
    moving_block_bootstrap_sigma_external, sample_acf, select_block_length,
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

# A real recorded 200-shot extended-measurement run, used only as a
# sanity check that the block-bootstrap machinery lands in the same
# ballpark against real data as the independent analytic AR(1) check it
# was validated against (see TestCombineShotsRealDataset below). Skipped,
# not failed, if this data isn't present in a given checkout.
REAL_MEASUREMENT_FITS_PATH = Path(
    "data/measurements/extended_measurement_20260813_171209/fits.npz"
)


# Functions

def _ar1_series(rho: float, n: int, innovation_sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    '''
    Generates a length-n AR(1) series x[t] = rho*x[t-1] + eps[t], eps ~
    N(0, innovation_sigma) -- a known, controllable correlation structure
    (autocorrelation at lag k is exactly rho**k in the population) to
    check select_block_length()/moving_block_bootstrap_sigma_external()
    respond correctly to real shot-to-shot correlation, mirroring the
    lag-1 ~0.89 autocorrelation actually measured in a real recorded
    extended-measurement run (see module docstring's
    REAL_MEASUREMENT_FITS_PATH).
    '''
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, innovation_sigma, size=n)
    series = np.empty(n)
    series[0] = innovations[0]
    for t in range(1, n):
        series[t] = rho * series[t - 1] + innovations[t]
    return series

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
            coefficient_covariance=np.diag([0.1, 0.01]) ** 2,
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
            coefficient_covariance=np.diag([0.1, 0.01, 0.01]) ** 2,
            reduced_chi_squared=1.0,
            residuals=np.zeros(3),
            normalized_residuals=np.zeros(3),
        )
        wavelength_nm = np.array([0.0, 1.0, 2.0])
        assert np.allclose(result.zeta(wavelength_nm), 0.5 + 4.0 * wavelength_nm)

    def test_linear_sigma_zeta_matches_coefficient_sigma(self):
        # At degree 1, zeta = c1 alone -- sigma_zeta should collapse to
        # coefficient_sigma[1] exactly, with no dependence on wavelength.
        coefficient_sigma = np.array([0.2, 0.03])
        result = SpatialDispersionFitResult(
            degree=1,
            coefficients=np.array([10.0, 0.5]),
            coefficient_sigma=coefficient_sigma,
            coefficient_covariance=np.diag(coefficient_sigma ** 2),
            reduced_chi_squared=1.0,
            residuals=np.zeros(3),
            normalized_residuals=np.zeros(3),
        )
        wavelength_nm = np.array([0.0, 10.0, 100.0])
        assert np.allclose(result.sigma_zeta(wavelength_nm), coefficient_sigma[1])

    def test_quadratic_sigma_zeta_uses_full_covariance(self):
        # zeta(lambda) = c1 + 2*c2*lambda, so
        # Var[zeta] = Var[c1] + 4*lambda^2*Var[c2] + 4*lambda*Cov[c1, c2].
        # Off-diagonal covariance is nonzero here specifically to check
        # that sigma_zeta uses the full matrix, not just its diagonal.
        var_c1, var_c2, cov_c1_c2 = 0.04, 0.0009, 0.002
        coefficient_covariance = np.array([
            [1.0, 0.0, 0.0],
            [0.0, var_c1, cov_c1_c2],
            [0.0, cov_c1_c2, var_c2],
        ])
        result = SpatialDispersionFitResult(
            degree=2,
            coefficients=np.array([10.0, 0.5, 2.0]),
            coefficient_sigma=np.sqrt(np.diag(coefficient_covariance)),
            coefficient_covariance=coefficient_covariance,
            reduced_chi_squared=1.0,
            residuals=np.zeros(3),
            normalized_residuals=np.zeros(3),
        )
        wavelength_nm = np.array([0.0, 5.0, 10.0])
        expected = np.sqrt(
            var_c1 + 4 * wavelength_nm ** 2 * var_c2 + 4 * wavelength_nm * cov_c1_c2
        )
        assert np.allclose(result.sigma_zeta(wavelength_nm), expected)
        # At lambda=0, cross-terms vanish -- sigma_zeta must NOT just
        # report sqrt(var_c1) here if it (incorrectly) ignored covariance
        # at other wavelengths, so this confirms the varying part above
        # rather than repeating it.
        assert result.sigma_zeta(0.0) == pytest.approx(np.sqrt(var_c1))

    def test_degree_coefficient_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            SpatialDispersionFitResult(
                degree=1,
                coefficients=np.array([1.0, 2.0, 3.0]),
                coefficient_sigma=np.array([0.1, 0.1, 0.1]),
                coefficient_covariance=np.eye(3),
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
                coefficient_covariance=np.eye(1),
                reduced_chi_squared=1.0,
                residuals=np.zeros(3),
                normalized_residuals=np.zeros(3),
            )

    def test_coefficient_covariance_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            SpatialDispersionFitResult(
                degree=1,
                coefficients=np.array([1.0, 2.0]),
                coefficient_sigma=np.array([0.1, 0.1]),
                coefficient_covariance=np.eye(3),
                reduced_chi_squared=1.0,
                residuals=np.zeros(3),
                normalized_residuals=np.zeros(3),
            )


class TestShotAnalysisResultImmutability:

    def test_fits_dict_is_read_only(self):
        centroids = CentroidResult(columns=np.arange(3), x0=np.zeros(3), sigma_x0=np.ones(3))
        fit = SpatialDispersionFitResult(
            degree=1, coefficients=np.array([0.0, 1.0]),
            coefficient_sigma=np.array([0.1, 0.1]),
            coefficient_covariance=np.diag([0.1, 0.1]) ** 2,
            reduced_chi_squared=1.0,
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

        # coefficient_covariance's diagonal must reproduce coefficient_sigma
        # (scipy.odr's own sd_beta = sqrt(diag(cov_beta) * res_var)), and
        # sigma_zeta must collapse to coefficient_sigma[1] at degree 1.
        assert np.allclose(
            np.sqrt(np.diag(result.coefficient_covariance)), result.coefficient_sigma
        )
        assert result.sigma_zeta(wavelength_nm) == pytest.approx(
            np.full(n, result.coefficient_sigma[1])
        )

    def test_insufficient_data_raises(self):
        wavelength_nm = np.array([1.0])
        x0 = np.array([1.0])
        sigma = np.array([0.1])
        with pytest.raises(InsufficientDataError):
            TotalLeastSquaresFit().fit(wavelength_nm, x0, sigma, sigma, degree=1)

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_exactly_degree_plus_one_points_raises(self, degree):
        # Regression test: degree + 1 points is enough to solve for the
        # coefficients (an exact interpolation), but leaves zero residual
        # degrees of freedom -- scipy.odr then reports coefficient_sigma
        # as (near-)zero rather than a real number, which crashes anything
        # downstream that requires a strictly positive sigma (e.g.
        # gui/formatting.py's format_value_with_uncertainty(), which is
        # exactly how this was first caught: a real live-view session hit
        # a tick with precisely degree + 1 valid columns and crashed on
        # display). InsufficientDataError must fire here instead, at
        # degree + 1, not just below it.
        n = degree + 1
        rng = np.random.default_rng(0)
        wavelength_nm = np.sort(rng.uniform(700.0, 900.0, size=n))
        x0 = np.polynomial.polynomial.polyval(wavelength_nm, rng.normal(size=degree + 1))
        sigma_wavelength_nm = np.full(n, 0.05)
        sigma_x0 = np.full(n, 0.5)

        with pytest.raises(InsufficientDataError):
            TotalLeastSquaresFit().fit(wavelength_nm, x0, sigma_wavelength_nm, sigma_x0, degree=degree)

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_exactly_degree_plus_two_points_succeeds_with_positive_sigma(self, degree):
        # The smallest point count with at least one residual degree of
        # freedom -- must succeed, and every coefficient_sigma must be
        # finite and strictly positive (a real, non-degenerate uncertainty
        # estimate), matching format_value_with_uncertainty()'s own
        # contract.
        n = degree + 2
        rng = np.random.default_rng(0)
        wavelength_nm = np.sort(rng.uniform(700.0, 900.0, size=n))
        true_coefficients = rng.normal(size=degree + 1)
        x0 = np.polynomial.polynomial.polyval(wavelength_nm, true_coefficients)
        x0 += rng.normal(scale=0.01, size=n)   # tiny noise -- avoids an exactly-zero residual by chance
        sigma_wavelength_nm = np.full(n, 0.05)
        sigma_x0 = np.full(n, 0.5)

        result = TotalLeastSquaresFit().fit(
            wavelength_nm, x0, sigma_wavelength_nm, sigma_x0, degree=degree
        )

        assert np.all(np.isfinite(result.coefficient_sigma))
        assert np.all(result.coefficient_sigma > 0)

    def test_residuals_shape_matches_input(self):
        wavelength_nm = np.linspace(0.0, 10.0, 20)
        x0 = 5.0 + 0.3 * wavelength_nm
        sigma = np.full(20, 0.1)

        result = TotalLeastSquaresFit().fit(wavelength_nm, x0, sigma, sigma, degree=1)

        assert result.residuals.shape == wavelength_nm.shape
        assert result.normalized_residuals.shape == wavelength_nm.shape


# ---------------------------------------------------------------------------
# block_bootstrap.py
# ---------------------------------------------------------------------------

class TestSelectBlockLength:

    def test_short_block_length_for_iid_data(self):
        rng = np.random.default_rng(11)
        series = rng.normal(0.0, 1.0, size=200)

        block_length, first_crossing_lag, lag1_autocorrelation = select_block_length(series)

        # I.i.d. noise has no real correlation to preserve -- the ACF
        # should cross the white-noise bound essentially immediately
        # (lag 1), collapsing the block length to its minimum.
        assert first_crossing_lag == 1
        assert block_length == 2
        assert abs(lag1_autocorrelation) < 0.3

    def test_longer_block_length_for_correlated_ar1_data(self):
        series = _ar1_series(rho=0.85, n=200, seed=3)

        block_length, first_crossing_lag, lag1_autocorrelation = select_block_length(series)

        # A strongly autocorrelated series must select a meaningfully
        # longer block than the i.i.d. case above -- this is the same
        # correlation regime as the real 200-shot dataset this bootstrap
        # was validated against (lag-1 ~0.89, see
        # TestCombineShotsRealDataset).
        assert first_crossing_lag > 1
        assert block_length > 2
        assert lag1_autocorrelation == pytest.approx(0.85, abs=0.1)

    def test_block_length_never_exceeds_series_length(self):
        # A short, strongly-correlated series (e.g. n=3) is a degenerate
        # edge case -- the clip must never let block_length exceed n,
        # regardless of what the correlation structure alone would imply.
        series = np.array([1.0, 1.01, 1.02])
        block_length, _, _ = select_block_length(series)
        assert block_length <= series.shape[0]

    def test_constant_series_has_zero_autocorrelation(self):
        # sample_acf()'s documented zero-variance fallback -- a constant
        # series must not raise (divide by zero) and must read as
        # "no detectable correlation".
        series = np.full(50, 3.0)
        block_length, first_crossing_lag, lag1_autocorrelation = select_block_length(series)
        assert lag1_autocorrelation == 0.0
        assert block_length == 2


class TestMovingBlockBootstrapSigmaExternal:

    def test_close_to_naive_scatter_for_independent_data(self):
        rng = np.random.default_rng(21)
        true_zeta = 0.02
        sigma = 0.001
        n = 100
        zeta_values = rng.normal(true_zeta, sigma, size=n)
        sigma_values = np.full(n, sigma)

        weights = 1.0 / sigma_values ** 2
        zeta_combined = np.sum(weights * zeta_values) / np.sum(weights)
        weighted_scatter = np.sum(weights * (zeta_values - zeta_combined) ** 2)
        naive_sigma_external = np.sqrt(weighted_scatter / ((n - 1) * np.sum(weights)))

        block_length, _, _ = select_block_length(zeta_values)
        bootstrap_sigma_external = moving_block_bootstrap_sigma_external(
            zeta_values, sigma_values, block_length,
            n_resamples=3000, rng=np.random.default_rng(0),
        )

        # Independent data -> the block bootstrap and the naive weighted
        # scatter should be in the same ballpark (bootstraps are
        # stochastic, so this is a ratio check, not an exact match).
        assert bootstrap_sigma_external == pytest.approx(naive_sigma_external, rel=0.5)

    def test_larger_than_naive_scatter_for_correlated_data(self):
        # Same construction as the real dataset's sanity check
        # (TestCombineShotsRealDataset): a genuinely autocorrelated shot
        # series makes the naive weighted-scatter sigma_external an
        # underestimate, which the block bootstrap should correct upward.
        rho = 0.85
        n = 200
        sigma_value = 0.05
        zeta_values = _ar1_series(rho=rho, n=n, innovation_sigma=1.0, seed=5) * 0.01 + 0.02
        sigma_values = np.full(n, sigma_value)

        weights = 1.0 / sigma_values ** 2
        zeta_combined = np.sum(weights * zeta_values) / np.sum(weights)
        weighted_scatter = np.sum(weights * (zeta_values - zeta_combined) ** 2)
        naive_sigma_external = np.sqrt(weighted_scatter / ((n - 1) * np.sum(weights)))

        block_length, _, _ = select_block_length(zeta_values)
        bootstrap_sigma_external = moving_block_bootstrap_sigma_external(
            zeta_values, sigma_values, block_length,
            n_resamples=3000, rng=np.random.default_rng(0),
        )

        assert bootstrap_sigma_external > 1.3 * naive_sigma_external

    def test_reproducible_with_same_seed(self):
        rng_data = np.random.default_rng(7)
        zeta_values = rng_data.normal(0.02, 0.001, size=30)
        sigma_values = np.full(30, 0.001)

        first = moving_block_bootstrap_sigma_external(
            zeta_values, sigma_values, block_length=4, n_resamples=500, rng=np.random.default_rng(42)
        )
        second = moving_block_bootstrap_sigma_external(
            zeta_values, sigma_values, block_length=4, n_resamples=500, rng=np.random.default_rng(42)
        )
        assert first == second


class TestSampleAcf:

    def test_lag1_matches_known_ar1_correlation(self):
        series = _ar1_series(rho=0.7, n=500, seed=9)
        acf = sample_acf(series, max_lag=5)
        assert acf[0] == pytest.approx(0.7, abs=0.1)
        # Autocorrelation must decay with increasing lag for a
        # mean-reverting AR(1) process.
        assert acf[0] > acf[1] > acf[2]


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
        # No scatter/correlation information exists at n_shots == 1 -- no
        # bootstrap runs, and the diagnostics read as "not applicable".
        assert result.block_length == 0
        assert result.first_crossing_lag == 0
        assert result.lag1_autocorrelation == 0.0

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            combine_shots(np.zeros(3), np.zeros(4))

    def test_diagnostics_populated_for_multiple_shots(self):
        series = _ar1_series(rho=0.8, n=60, seed=13) * 0.001 + 0.02
        result = combine_shots(series, np.full(60, 0.0005))

        assert result.block_length >= 2
        assert result.first_crossing_lag >= 1
        assert result.lag1_autocorrelation == pytest.approx(0.8, abs=0.15)

    def test_reproducible_without_explicit_rng(self):
        # Scientific measurement record -- repeated calls against the same
        # input must reproduce the same sigma_external by default (see
        # block_bootstrap.DEFAULT_BOOTSTRAP_SEED), not draw fresh
        # randomness each time.
        zeta_values = _ar1_series(rho=0.7, n=40, seed=17) * 0.001 + 0.02
        sigma_values = np.full(40, 0.0005)

        first = combine_shots(zeta_values, sigma_values)
        second = combine_shots(zeta_values, sigma_values)
        assert first.sigma_external == second.sigma_external

    def test_n_resamples_override_is_honored(self):
        # A smaller n_resamples (as extended_measurement.py's live preview
        # path uses) must still return a valid, finite result -- just
        # exercising the parameter is enough here, the statistical
        # validity of the bootstrap itself is covered by
        # TestMovingBlockBootstrapSigmaExternal above.
        zeta_values = _ar1_series(rho=0.6, n=50, seed=23) * 0.001 + 0.02
        sigma_values = np.full(50, 0.0005)

        result = combine_shots(zeta_values, sigma_values, n_resamples=50)
        assert np.isfinite(result.sigma_external)
        assert result.sigma_external > 0


class TestCombineShotsRealDataset:

    '''
    Sanity check against a real recorded 200-shot extended-measurement
    run (data/measurements/extended_measurement_20260813_171209/): a
    naive weighted-scatter sigma_external on this dataset's degree-1
    per-shot zeta series measures ~9.15 nm/nm (in the codebase's final
    reporting units, after ScaleFactorPositionCalibration's px -> nm/nm
    conversion), while an independent analytic AR(1) correction
    (sigma x sqrt((1+rho)/(1-rho)), rho ~ 0.7815 from an ARIMA(1,0,2) fit)
    gives ~26.1 nm/nm and a hand-tuned moving-block bootstrap (block
    length 8) gives ~21.1 nm/nm -- both ~2.3-2.9x the naive value. This
    confirms combine_shots()'s own automatic block-length selection lands
    in the same ballpark, not just that the machinery runs without error.
    '''

    def test_bootstrap_sigma_in_validated_ballpark(self):
        if not REAL_MEASUREMENT_FITS_PATH.is_file():
            pytest.skip(f"real measurement data not present: {REAL_MEASUREMENT_FITS_PATH}")

        # Same px -> nm/nm conversion factor
        # gui/calibration_spatial.py's ScaleFactorPositionCalibration
        # applies (PIXEL_PITCH_UM=3.45 x DEFAULT_SCALE_FACTOR=1.5 x 1000
        # nm/um) -- reproduced here as a bare constant rather than
        # importing gui/ into analysis/'s own test suite, which would
        # invert this codebase's layering (gui/ depends on analysis/,
        # never the reverse).
        px_to_nm_per_nm = 3.45 * 1.5 * 1000.0

        with np.load(REAL_MEASUREMENT_FITS_PATH) as data:
            n_shots = 0
            while f"shot{n_shots}_degree1_coefficients" in data.files:
                n_shots += 1
            zeta_values = np.array(
                [data[f"shot{i}_degree1_coefficients"][1] for i in range(n_shots)]
            )
            sigma_zeta_values = np.array(
                [data[f"shot{i}_degree1_coefficient_sigma"][1] for i in range(n_shots)]
            )

        result = combine_shots(zeta_values, sigma_zeta_values)
        sigma_external_nm = result.sigma_external * px_to_nm_per_nm

        assert result.n_shots == 200
        # Real shot-to-shot correlation genuinely exists in this dataset.
        assert result.lag1_autocorrelation == pytest.approx(0.89, abs=0.05)
        # Validated range from the independent analytic AR(1)/hand-tuned
        # bootstrap checks above: roughly 20-26 nm/nm.
        assert 18.0 < sigma_external_nm < 28.0


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
