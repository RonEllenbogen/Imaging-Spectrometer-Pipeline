"""
Fits pixel -> wavelength_nm from matched (pixel, wavelength_nm) pairs
produced by line_matching.py (not yet implemented -- see its own module
docstring), via calibration/shared/fitting.py's generic polynomial
fitter. The result implements analysis.interfaces.WavelengthAxis directly
(wavelength_nm()/sigma_wavelength_nm() live on the result object itself),
the same "result IS the interface" pattern analysis/results.py's
SpatialDispersionFitResult.zeta() uses.
"""

# Imports

from dataclasses import dataclass

import numpy as np

from ..shared.fitting import PolynomialFitter, TotalLeastSquaresFit
from ..shared.metadata import CalibrationRecord
from ..shared.result import PolynomialFitResult

# Constants

# Classes

@dataclass(frozen=True, slots=True)
class WavelengthCalibrationResult:

    '''
    Implements analysis.interfaces.WavelengthAxis. Wraps a generic
    PolynomialFitResult (pixel -> wavelength_nm) plus the CalibrationRecord
    tagging the lamp frame(s) it was built from.

    sigma_wavelength_nm() propagates coefficient_sigma treating each
    coefficient's uncertainty as independent -- it ignores their
    covariance, so it's an approximation of the fit's true (correlated)
    uncertainty, not an exact propagation. scipy.odr exposes a full
    covariance matrix (cov_beta) that could replace this if the
    approximation ever proves too coarse against real lamp data. Flagged
    for review once real calibration data exists, consistent with other
    placeholder/approximation choices already tracked in this codebase
    (docs/project_state.md, "Implementation decisions flagged for review").
    '''

    fit: PolynomialFitResult
    record: CalibrationRecord

    def wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:

        '''See analysis.interfaces.WavelengthAxis.wavelength_nm.'''

        return self.fit.evaluate(pixel)

    def sigma_wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:

        '''See analysis.interfaces.WavelengthAxis.sigma_wavelength_nm --
        see class docstring for the independent-coefficient approximation
        used here.'''

        pixel = np.asarray(pixel, dtype=np.float64)
        powers = pixel[..., np.newaxis] ** np.arange(self.fit.coefficients.shape[0])
        variance = np.sum((powers * self.fit.coefficient_sigma) ** 2, axis=-1)
        return np.sqrt(variance)


# Functions

def calibrate_spectral(
    pixel: np.ndarray,
    wavelength_nm: np.ndarray,
    sigma_pixel: np.ndarray,
    sigma_wavelength_nm: np.ndarray,
    record: CalibrationRecord,
    degree: int = 1,
    fitter: PolynomialFitter | None = None,
) -> WavelengthCalibrationResult:

    '''
    Fits pixel -> wavelength_nm from matched line data.

    Parameters
    ----------
    pixel, wavelength_nm
        Matched line positions (pixel-column index) and their known
        reference wavelengths (nm), from line_matching.py.
    sigma_pixel, sigma_wavelength_nm
        Per-line 1-sigma uncertainties, strictly positive -- see
        shared/fitting.py.
    record
        Tags the resulting calibration with the settings/timing of the
        lamp frame(s) it was built from. CalibrationRecord is reused here
        exactly as calibration/sensor/'s artifacts use it, even though
        exposure/gain aren't physically load-bearing for a wavelength
        calibration the way they are for a flat field -- kept for
        provenance/logging consistency across artifact types.
    degree
        Polynomial degree to fit (1 = linear, matching a first-order
        grating-dispersion approximation; higher degrees are a
        model-adequacy diagnostic, same role as analysis/'s zeta fit).
    fitter
        PolynomialFitter to use. Defaults to TotalLeastSquaresFit.

    Returns
    -------
    WavelengthCalibrationResult

    Raises
    ------
    InsufficientDataError
        Propagated from the fitter if fewer than degree + 1 lines matched.
    '''

    fitter = fitter if fitter is not None else TotalLeastSquaresFit()
    fit = fitter.fit(pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm, degree)
    return WavelengthCalibrationResult(fit=fit, record=record)


__all__ = ["WavelengthCalibrationResult", "calibrate_spectral"]
