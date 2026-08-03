"""
Fits x0 = c0 + c1*omega + c2*omega^2 + ... (ascending-order coefficients)
to one shot's centroid data via total least squares / orthogonal distance
regression, required rather than ordinary weighted least squares because
both axes carry real uncertainty here -- sigma_x0 from centroid
extraction, sigma_omega from whatever eventually implements
interfaces.FrequencyAxis (docs/project_state.md #16).

Supporting degree > 1 isn't about redefining the physical quantity of
interest -- it's a model-adequacy diagnostic (docs/project_state.md #18):
comparing reduced chi-squared across linear/quadratic/cubic fits of the
same data tells you whether a single constant spatial dispersion is
actually an adequate description, or whether there's real curvature a
straight line is averaging over.

Requires sigma_omega and sigma_x0 to be strictly positive everywhere --
scipy.odr weights internally by their inverse, so an exact zero (e.g. from
a not-yet-built FrequencyAxis placeholder returning sigma_omega=0) will
fail or misbehave. No zero-guard/clipping is added here deliberately --
surfacing that requirement loudly at whatever supplies the FrequencyAxis
is better than silently papering over it.
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
        omega: np.ndarray,
        x0: np.ndarray,
        sigma_omega: np.ndarray,
        sigma_x0: np.ndarray,
        degree: int,
    ) -> SpatialDispersionFitResult:

        '''
        Parameters
        ----------
        omega, x0
            Per-column angular frequency and centroid position.
        sigma_omega, sigma_x0
            Per-column 1-sigma uncertainties on omega and x0. Must be
            strictly positive -- see module docstring.
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
    for uncertainty in both omega and x0 simultaneously -- see the module
    docstring for why ordinary (y-only) weighted least squares isn't
    enough here.
    '''

    def fit(
        self,
        omega: np.ndarray,
        x0: np.ndarray,
        sigma_omega: np.ndarray,
        sigma_x0: np.ndarray,
        degree: int = 1,
    ) -> SpatialDispersionFitResult:

        '''See SpatialDispersionFitter.fit for parameters/returns/raises.'''

        if omega.shape[0] < degree + 1:
            raise InsufficientDataError(degree, omega.shape[0])

        # Seeded with an ordinary (uncertainty-blind) polynomial fit --
        # standard practice to help ODR's iterative solver converge,
        # rather than starting from an arbitrary guess.
        initial_guess = np.polynomial.polynomial.polyfit(omega, x0, degree)

        model = odr.Model(_polynomial_model)
        data = odr.RealData(omega, x0, sx=sigma_omega, sy=sigma_x0)
        result = odr.ODR(data, model, beta0=initial_guess).run()

        coefficients = result.beta
        coefficient_sigma = result.sd_beta
        reduced_chi_squared = result.res_var

        x0_fit = np.polynomial.polynomial.polyval(omega, coefficients)
        residuals = x0 - x0_fit

        local_zeta = np.polynomial.polynomial.polyval(
            omega, np.polynomial.polynomial.polyder(coefficients)
        )
        effective_sigma = np.sqrt(sigma_x0 ** 2 + (local_zeta * sigma_omega) ** 2)
        normalized_residuals = residuals / effective_sigma

        return SpatialDispersionFitResult(
            degree=degree,
            coefficients=coefficients,
            coefficient_sigma=coefficient_sigma,
            reduced_chi_squared=float(reduced_chi_squared),
            residuals=residuals,
            normalized_residuals=normalized_residuals,
        )


# Functions

def _polynomial_model(coefficients: np.ndarray, omega: np.ndarray) -> np.ndarray:

    '''
    scipy.odr's expected model signature: model(beta, x) -> y. Thin
    wrapper around numpy's ascending-order polynomial evaluation, kept
    module-level (not a lambda/closure) so scipy.odr.Model can pickle it
    if ever run in a multiprocessing context.
    '''

    return np.polynomial.polynomial.polyval(omega, coefficients)


__all__ = ["SpatialDispersionFitter", "TotalLeastSquaresFit"]
