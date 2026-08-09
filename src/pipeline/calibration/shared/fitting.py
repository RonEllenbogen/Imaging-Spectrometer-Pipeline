"""
Generic weighted polynomial fit via total least squares / orthogonal
distance regression (scipy.odr) -- accounts for uncertainty in both x and
y simultaneously, needed whenever both axes carry real measurement
uncertainty (e.g. calibration/spectral/calibrate.py's pixel/wavelength
fit: sigma on the matched pixel position from line detection, sigma on
the reference wavelength from atomic transition tables).

Generalizes analysis/dispersion_fitting.py's TotalLeastSquaresFit to
generic x/y rather than wavelength_nm/x0 -- kept as a structurally
separate implementation (not imported from analysis/), since calibration/
and analysis/ must not depend on each other in either direction (see
analysis/interfaces.py's own docstring on this boundary).
"""

# Imports

from typing import Protocol

import numpy as np
from scipy import odr

from ..exceptions import InsufficientDataError
from .result import PolynomialFitResult

# Constants

# Classes

class PolynomialFitter(Protocol):

    '''
    Structural interface every polynomial fit method must match -- same
    structural-typing pattern as CameraBackend/SpatialDispersionFitter
    elsewhere in this codebase.
    '''

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma_x: np.ndarray,
        sigma_y: np.ndarray,
        degree: int,
    ) -> PolynomialFitResult:

        '''
        Parameters
        ----------
        x, y
            Independent/dependent variable values.
        sigma_x, sigma_y
            Per-point 1-sigma uncertainties on x and y, same shape as x/y.
            Must be strictly positive -- scipy.odr weights internally by
            their inverse.
        degree
            Polynomial degree to fit (1 = linear, 2 = quadratic, 3 = cubic).

        Returns
        -------
        PolynomialFitResult

        Raises
        ------
        InsufficientDataError
            If fewer than degree + 1 points are supplied.
        '''

        ...


class TotalLeastSquaresFit:

    '''
    Default PolynomialFitter: orthogonal distance regression via
    scipy.odr, which minimizes a properly-weighted distance accounting
    for uncertainty in both x and y simultaneously -- see the module
    docstring for why ordinary (y-only) weighted least squares isn't
    enough here.
    '''

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma_x: np.ndarray,
        sigma_y: np.ndarray,
        degree: int = 1,
    ) -> PolynomialFitResult:

        '''See PolynomialFitter.fit for parameters/returns/raises.'''

        if x.shape[0] < degree + 1:
            raise InsufficientDataError(degree, x.shape[0])

        # Seeded with an ordinary (uncertainty-blind) polynomial fit --
        # standard practice to help ODR's iterative solver converge,
        # rather than starting from an arbitrary guess.
        initial_guess = np.polynomial.polynomial.polyfit(x, y, degree)

        model = odr.Model(_polynomial_model)
        data = odr.RealData(x, y, sx=sigma_x, sy=sigma_y)
        result = odr.ODR(data, model, beta0=initial_guess).run()

        coefficients = result.beta
        coefficient_sigma = result.sd_beta
        reduced_chi_squared = result.res_var

        y_fit = np.polynomial.polynomial.polyval(x, coefficients)
        residuals = y - y_fit

        local_slope = np.polynomial.polynomial.polyval(
            x, np.polynomial.polynomial.polyder(coefficients)
        )
        effective_sigma = np.sqrt(sigma_y ** 2 + (local_slope * sigma_x) ** 2)
        normalized_residuals = residuals / effective_sigma

        return PolynomialFitResult(
            degree=degree,
            coefficients=coefficients,
            coefficient_sigma=coefficient_sigma,
            reduced_chi_squared=float(reduced_chi_squared),
            residuals=residuals,
            normalized_residuals=normalized_residuals,
        )


# Functions

def _polynomial_model(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:

    '''
    scipy.odr's expected model signature: model(beta, x) -> y. Thin
    wrapper around numpy's ascending-order polynomial evaluation, kept
    module-level (not a lambda/closure) so scipy.odr.Model can pickle it
    if ever run in a multiprocessing context.
    '''

    return np.polynomial.polynomial.polyval(x, coefficients)


__all__ = ["PolynomialFitter", "TotalLeastSquaresFit"]
