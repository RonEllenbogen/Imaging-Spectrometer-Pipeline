"""
Result objects analysis/ produces. Bundled in one module rather than
split one-per-file since all four are small and tightly coupled by what
consumes what (ShotAnalysisResult wraps CentroidResult and
SpatialDispersionFitResult; CombinedSpatialDispersionResult summarizes
many shots' fits), and none has enough independent behavior to warrant
its own file the way FrameData/ProcessedFrame do.

Array-holding dataclasses lock their arrays' writeability in
__post_init__, same as FrameData/ProcessedFrame -- frozen=True alone
stops reassigning a field, not mutating an array's contents in place.
ShotAnalysisResult's `fits` dict is similarly wrapped read-only via
MappingProxyType for the same reason.
"""

# Imports

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

# Constants

# Classes

@dataclass(frozen=True, slots=True, eq=False)
class CentroidResult:

    '''
    Per-frame centroid extraction output: one pixel-column bin per
    spectral column (docs/project_state.md #5/#6), covering the full
    spatial axis (#3/#4).

    Parameters
    ----------
    columns
        Spectral pixel-column index of each entry.
    x0
        Spatial centroid position, in pixels, per column.
    sigma_x0
        Thompson-Larson-Webb uncertainty on x0, in pixels, per column.
    '''

    columns: np.ndarray
    x0: np.ndarray
    sigma_x0: np.ndarray

    def __post_init__(self) -> None:
        if not (self.columns.shape == self.x0.shape == self.sigma_x0.shape):
            raise ValueError(
                "columns, x0, and sigma_x0 must all have the same shape, got "
                f"{self.columns.shape}, {self.x0.shape}, {self.sigma_x0.shape}"
            )
        for array in (self.columns, self.x0, self.sigma_x0):
            array.flags.writeable = False


@dataclass(frozen=True, slots=True, eq=False)
class SpatialDispersionFitResult:

    '''
    Result of fitting x0 = c0 + c1*wavelength_nm + c2*wavelength_nm^2 + ...
    to one shot's centroid data, at one polynomial degree (see "Result
    object shapes" in docs/project_state.md).

    Parameters
    ----------
    degree
        Polynomial degree fitted (1 = linear, 2 = quadratic, 3 = cubic).
    coefficients
        Fitted coefficients, ascending order (c0, c1, c2, ...), length
        degree + 1.
    coefficient_sigma
        1-sigma uncertainty on each coefficient (the covariance matrix's
        diagonal, marginalized over the others), same length/order.
    coefficient_covariance
        Full (degree + 1, degree + 1) covariance matrix of the fitted
        coefficients -- scipy.odr's cov_beta scaled by res_var (see
        reduced_chi_squared below), since cov_beta alone is normalized and
        not yet the real covariance. Needed for sigma_zeta: zeta(lambda) is
        an exact linear combination of the coefficients, so its variance
        depends on their covariances, not just their individual variances
        (coefficient_sigma's diagonal alone is only enough at degree == 1,
        where zeta is a single coefficient with no cross-terms).
    reduced_chi_squared
        scipy.odr's residual variance -- the errors-in-both-variables
        generalization of reduced chi-squared; ~1 indicates a fit
        consistent with the input uncertainties.
    residuals
        x0_observed - x0_fit(wavelength_nm), per column.
    normalized_residuals
        residuals divided by the effective combined sigma
        sqrt(sigma_x0^2 + zeta^2 * sigma_wavelength_nm^2), per column.
    '''

    degree: int
    coefficients: np.ndarray
    coefficient_sigma: np.ndarray
    coefficient_covariance: np.ndarray
    reduced_chi_squared: float
    residuals: np.ndarray
    normalized_residuals: np.ndarray

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError(f"degree must be at least 1, got {self.degree}")
        if self.coefficients.shape != self.coefficient_sigma.shape:
            raise ValueError(
                "coefficients and coefficient_sigma must have the same shape, got "
                f"{self.coefficients.shape}, {self.coefficient_sigma.shape}"
            )
        if self.coefficients.shape[0] != self.degree + 1:
            raise ValueError(
                f"expected {self.degree + 1} coefficients for degree {self.degree}, "
                f"got {self.coefficients.shape[0]}"
            )
        n = self.coefficients.shape[0]
        if self.coefficient_covariance.shape != (n, n):
            raise ValueError(
                f"expected coefficient_covariance shape ({n}, {n}), got "
                f"{self.coefficient_covariance.shape}"
            )
        if self.residuals.shape != self.normalized_residuals.shape:
            raise ValueError(
                "residuals and normalized_residuals must have the same shape, got "
                f"{self.residuals.shape}, {self.normalized_residuals.shape}"
            )
        for array in (
            self.coefficients, self.coefficient_sigma, self.coefficient_covariance,
            self.residuals, self.normalized_residuals,
        ):
            array.flags.writeable = False

    def zeta(self, wavelength_nm: np.ndarray) -> np.ndarray:

        '''
        Local spatial dispersion dx0/dwavelength_nm (px/nm), evaluated at
        wavelength_nm -- the fitted polynomial's derivative. Collapses to
        the familiar single constant (coefficients[1]) whenever
        degree == 1 (see "Result object shapes" in docs/project_state.md).
        '''

        derivative_coefficients = np.polynomial.polynomial.polyder(self.coefficients)
        return np.polynomial.polynomial.polyval(wavelength_nm, derivative_coefficients)

    def sigma_zeta(self, wavelength_nm: np.ndarray) -> np.ndarray:

        '''
        1-sigma uncertainty on zeta(wavelength_nm), propagated from the
        fit's full coefficient_covariance rather than just the marginal
        coefficient_sigma -- see coefficient_covariance's docstring above
        for why the marginal alone isn't enough once degree > 1.

        zeta(lambda) = sum_{k=1}^{degree} k * c_k * lambda^(k-1) is an
        exact linear function of the coefficients c, i.e. zeta(lambda) =
        g(lambda) @ c with g_k(lambda) = k * lambda^(k-1) (k >= 1) and
        g_0(lambda) = 0. So, exactly (not just to first order):

            Var[zeta(lambda)] = g(lambda) @ coefficient_covariance @ g(lambda)

        Collapses to coefficient_sigma[1] at degree == 1, where g = (0, 1)
        and the formula reduces to sqrt(coefficient_covariance[1, 1]).
        '''

        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        powers = np.arange(self.coefficients.shape[0])
        gradient = powers * wavelength_nm[..., np.newaxis] ** np.maximum(powers - 1, 0)
        variance = np.einsum("...i,ij,...j->...", gradient, self.coefficient_covariance, gradient)
        return np.sqrt(variance)


@dataclass(frozen=True, slots=True)
class ShotAnalysisResult:

    '''
    Full analysis output for one shot: its centroid data, plus one
    SpatialDispersionFitResult per requested polynomial degree, so
    linear/quadratic/cubic fits can be compared side by side
    (docs/project_state.md #18).

    Parameters
    ----------
    frame_id
        Traces this result back to the originating frame.
    centroids
        This shot's per-column centroid extraction.
    fits
        Fit results keyed by degree.
    '''

    frame_id: int
    centroids: CentroidResult
    fits: dict[int, SpatialDispersionFitResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fits", MappingProxyType(dict(self.fits)))


@dataclass(frozen=True, slots=True)
class CombinedSpatialDispersionResult:

    '''
    Inverse-variance combination of the linear (degree-1) spatial
    dispersion across N shots (docs/project_state.md #19/#20). Only the
    linear fit is combined this way -- quadratic/cubic fits stay per-shot
    model-adequacy diagnostics, not something aggregated across shots.

    Parameters
    ----------
    zeta_combined
        Inverse-variance-weighted mean of each shot's degree-1 zeta.
    sigma_internal
        1/sqrt(sum of inverse variances) -- the uncertainty implied
        purely by propagating each shot's own claimed sigma_zeta.
    sigma_external
        Weighted scatter of the per-shot zeta values around
        zeta_combined -- the empirical uncertainty implied by how much
        the shots actually disagree with each other.
    sigma_zeta_combined
        max(sigma_internal, sigma_external) -- reports the internal
        (propagated) uncertainty when the shots are mutually consistent,
        and the larger, empirically-scattered external uncertainty when
        they aren't, without double-counting by summing both in
        quadrature.
    n_shots
        Number of shots combined.
    '''

    zeta_combined: float
    sigma_internal: float
    sigma_external: float
    sigma_zeta_combined: float
    n_shots: int

    def __post_init__(self) -> None:
        if self.n_shots < 1:
            raise ValueError(f"n_shots must be at least 1, got {self.n_shots}")


# Functions


__all__ = [
    "CentroidResult", "SpatialDispersionFitResult",
    "ShotAnalysisResult", "CombinedSpatialDispersionResult",
]
