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

def build_manual_spectral_calibration(
    coefficients: np.ndarray,
    coefficient_sigma: np.ndarray,
    record: CalibrationRecord,
) -> WavelengthCalibrationResult:

    '''
    Builds a WavelengthCalibrationResult directly from a pixel->wavelength_nm
    polynomial the user measured independently (e.g. via Pylon Viewer,
    reading off pixel positions of known spectral lines by hand) --
    bypassing match_lines()/calibrate_spectral()'s automatic fit entirely.
    Same coefficient convention as everywhere else in this module:
    wavelength_nm = c0 + c1*pixel + c2*pixel^2 + ... (ascending order).

    coefficient_sigma must be supplied by the caller, not defaulted or
    derived here -- there is no fit residual to estimate it from, and
    analysis/dispersion_fitting.py's TotalLeastSquaresFit hard-requires
    sigma_wavelength_nm strictly positive everywhere downstream, so
    inventing a placeholder precision nobody actually measured would
    silently misrepresent the calibration's real uncertainty. If the
    user doesn't have a real uncertainty estimate, entering a
    deliberately conservative (large) value is their call to make, not
    this function's.

    Parameters
    ----------
    coefficients
        Polynomial coefficients, ascending order (c0, c1, c2, ...).
    coefficient_sigma
        1-sigma uncertainty on each coefficient, same shape as
        coefficients, strictly positive.
    record
        Tags the resulting calibration with provenance -- same role as in
        calibrate_spectral(), though exposure_us/gain_db are even less
        physically load-bearing here (no frame was captured at all for a
        manually-entered calibration); kept for logging consistency.
        CalibrationRecord.source_frame_count must be >= 1 even though no
        frames were actually captured -- callers constructing this record
        for a manual entry should pass source_frame_count=1 as a
        convention meaning "not applicable", not 0 (which the class
        rejects outright).

    Returns
    -------
    WavelengthCalibrationResult

    Raises
    ------
    ValueError
        If coefficient_sigma isn't the same shape as coefficients, or any
        entry isn't strictly positive (mirrors PolynomialFitResult's own
        shape validation, plus the strict-positivity check
        TotalLeastSquaresFit would otherwise only catch much later, at
        the first downstream per-shot fit).
    '''

    coefficients = np.asarray(coefficients, dtype=np.float64)
    coefficient_sigma = np.asarray(coefficient_sigma, dtype=np.float64)
    if coefficient_sigma.shape != coefficients.shape:
        raise ValueError(
            "coefficient_sigma must have the same shape as coefficients, got "
            f"{coefficient_sigma.shape}, {coefficients.shape}"
        )
    if np.any(coefficient_sigma <= 0):
        raise ValueError(f"coefficient_sigma must be strictly positive, got {coefficient_sigma!r}")

    fit = PolynomialFitResult(
        degree=coefficients.shape[0] - 1,
        coefficients=coefficients,
        coefficient_sigma=coefficient_sigma,
        reduced_chi_squared=float("nan"),   # not applicable -- no fit was run
        residuals=np.array([]),
        normalized_residuals=np.array([]),
    )
    return WavelengthCalibrationResult(fit=fit, record=record)


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


__all__ = ["WavelengthCalibrationResult", "calibrate_spectral", "build_manual_spectral_calibration"]
