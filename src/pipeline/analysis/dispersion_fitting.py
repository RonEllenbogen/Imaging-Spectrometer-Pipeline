"""
Fits x0 = c0 + c1*wavelength_nm + c2*wavelength_nm^2 + ... (ascending-order
coefficients) to one shot's centroid data via total least squares /
orthogonal distance regression, required rather than ordinary weighted
least squares because both axes carry real uncertainty here -- sigma_x0
from centroid extraction, sigma_wavelength_nm from whatever eventually
implements interfaces.WavelengthAxis (see "Dispersion interface" in
docs/project_state.md).

Supporting degree > 1 isn't about redefining the physical quantity of
interest -- it's a model-adequacy diagnostic (see "Result object shapes"
in docs/project_state.md): comparing reduced chi-squared across
linear/quadratic/cubic fits of the same data tells you whether a single
constant spatial dispersion is actually an adequate description, or
whether there's real curvature a straight line is averaging over.

Requires sigma_wavelength_nm and sigma_x0 to be strictly positive
everywhere -- scipy.odr weights internally by their inverse, so an exact
zero (e.g. from a not-yet-built WavelengthAxis placeholder returning
sigma_wavelength_nm=0) will fail or misbehave. No zero-guard/clipping is
added here deliberately -- surfacing that requirement loudly at whatever
supplies the WavelengthAxis is better than silently papering over it.
"""

# Imports

from typing import Protocol

import numpy as np
from scipy import odr

from .exceptions import InsufficientDataError
from .results import SpatialDispersionFitResult

# Constants

# Classes

class SpatialDispersionFitter(Protocol):

    '''
    Structural interface every spatial-dispersion fit method must match.
    '''

    def fit(
        self,
        wavelength_nm: np.ndarray,
        x0: np.ndarray,
        sigma_wavelength_nm: np.ndarray,
        sigma_x0: np.ndarray,
        degree: int,
    ) -> SpatialDispersionFitResult:

        '''
        Parameters
        ----------
        wavelength_nm, x0
            Per-column wavelength (nm) and centroid position.
        sigma_wavelength_nm, sigma_x0
            Per-column 1-sigma uncertainties on wavelength_nm and x0. Must
            be strictly positive -- see module docstring.
        degree
            Polynomial degree to fit (1 = linear, 2 = quadratic, 3 = cubic).

        Returns
        -------
        SpatialDispersionFitResult

        Raises
        ------
        InsufficientDataError
            If fewer than degree + 1 columns are supplied.
        '''

        ...


class TotalLeastSquaresFit:

    '''
    Default SpatialDispersionFitter: orthogonal distance regression via
    scipy.odr, which minimizes a properly-weighted distance accounting
    for uncertainty in both wavelength_nm and x0 simultaneously -- see the
    module docstring for why ordinary (y-only) weighted least squares
    isn't enough here.
    '''

    def fit(
        self,
        wavelength_nm: np.ndarray,
        x0: np.ndarray,
        sigma_wavelength_nm: np.ndarray,
        sigma_x0: np.ndarray,
        degree: int = 1,
    ) -> SpatialDispersionFitResult:

        '''See SpatialDispersionFitter.fit for parameters/returns/raises.'''

        if wavelength_nm.shape[0] < degree + 1:
            raise InsufficientDataError(degree, wavelength_nm.shape[0])

        # Seeded with an ordinary (uncertainty-blind) polynomial fit --
        # standard practice to help ODR's iterative solver converge,
        # rather than starting from an arbitrary guess.
        initial_guess = np.polynomial.polynomial.polyfit(wavelength_nm, x0, degree)

        model = odr.Model(_polynomial_model)
        data = odr.RealData(wavelength_nm, x0, sx=sigma_wavelength_nm, sy=sigma_x0)
        result = odr.ODR(data, model, beta0=initial_guess).run()

        coefficients = result.beta
        coefficient_sigma = result.sd_beta
        reduced_chi_squared = result.res_var
        # result.cov_beta is normalized -- scipy.odr's own sd_beta is
        # sqrt(diag(cov_beta) * res_var), so the real covariance matrix
        # needs the same res_var scaling applied across the whole matrix.
        coefficient_covariance = result.cov_beta * reduced_chi_squared

        x0_fit = np.polynomial.polynomial.polyval(wavelength_nm, coefficients)
        residuals = x0 - x0_fit

        local_zeta = np.polynomial.polynomial.polyval(
            wavelength_nm, np.polynomial.polynomial.polyder(coefficients)
        )
        effective_sigma = np.sqrt(sigma_x0 ** 2 + (local_zeta * sigma_wavelength_nm) ** 2)
        normalized_residuals = residuals / effective_sigma

        return SpatialDispersionFitResult(
            degree=degree,
            coefficients=coefficients,
            coefficient_sigma=coefficient_sigma,
            coefficient_covariance=coefficient_covariance,
            reduced_chi_squared=float(reduced_chi_squared),
            residuals=residuals,
            normalized_residuals=normalized_residuals,
        )


# Functions

def _polynomial_model(coefficients: np.ndarray, wavelength_nm: np.ndarray) -> np.ndarray:

    '''
    scipy.odr's expected model signature: model(beta, x) -> y. Thin
    wrapper around numpy's ascending-order polynomial evaluation, kept
    module-level (not a lambda/closure) so scipy.odr.Model can pickle it
    if ever run in a multiprocessing context.
    '''

    return np.polynomial.polynomial.polyval(wavelength_nm, coefficients)


__all__ = ["SpatialDispersionFitter", "TotalLeastSquaresFit"]
