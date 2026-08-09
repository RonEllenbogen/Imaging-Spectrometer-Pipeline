"""
Generic polynomial fit result, produced by shared/fitting.py's
PolynomialFitter implementations. Mirrors analysis/results.py's
SpatialDispersionFitResult shape and validation, under generic x/y naming
rather than wavelength_nm/x0 -- kept as a structurally separate class
rather than importing analysis/'s version, since analysis/ and
calibration/ must not depend on each other in either direction (see
shared/fitting.py's module docstring).
"""

# Imports

from dataclasses import dataclass

import numpy as np

# Constants

# Classes

@dataclass(frozen=True, slots=True, eq=False)
class PolynomialFitResult:

    '''
    Result of fitting y = c0 + c1*x + c2*x^2 + ... via total least
    squares / orthogonal distance regression.

    Parameters
    ----------
    degree
        Polynomial degree fitted (1 = linear, 2 = quadratic, 3 = cubic).
    coefficients
        Fitted coefficients, ascending order (c0, c1, c2, ...), length
        degree + 1.
    coefficient_sigma
        1-sigma uncertainty on each coefficient, same length/order.
    reduced_chi_squared
        scipy.odr's residual variance -- the errors-in-both-variables
        generalization of reduced chi-squared; ~1 indicates a fit
        consistent with the input uncertainties.
    residuals
        y_observed - y_fit(x), per point.
    normalized_residuals
        residuals divided by the effective combined sigma
        sqrt(sigma_y^2 + slope^2 * sigma_x^2), per point.
    '''

    degree: int
    coefficients: np.ndarray
    coefficient_sigma: np.ndarray
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
        if self.residuals.shape != self.normalized_residuals.shape:
            raise ValueError(
                "residuals and normalized_residuals must have the same shape, got "
                f"{self.residuals.shape}, {self.normalized_residuals.shape}"
            )
        for array in (
            self.coefficients, self.coefficient_sigma,
            self.residuals, self.normalized_residuals,
        ):
            array.flags.writeable = False

    def evaluate(self, x: np.ndarray) -> np.ndarray:

        '''Evaluates the fitted polynomial at x.'''

        return np.polynomial.polynomial.polyval(x, self.coefficients)

    def evaluate_derivative(self, x: np.ndarray) -> np.ndarray:

        '''Evaluates the fitted polynomial's derivative at x.'''

        derivative_coefficients = np.polynomial.polynomial.polyder(self.coefficients)
        return np.polynomial.polynomial.polyval(x, derivative_coefficients)


# Functions


__all__ = ["PolynomialFitResult"]
