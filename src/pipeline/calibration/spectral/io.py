"""
Persists a spectral calibration artifact (WavelengthCalibrationResult) to
a single .npz file via shared/io.py. degree and reduced_chi_squared --
scalars belonging to the fit itself, not to CalibrationRecord -- are
packed as 0-d arrays alongside the fit's coefficient/residual arrays,
since shared/io.py's `record` slot is reserved for CalibrationRecord
(reused here per the decision that a spectral calibration is tagged with
the lamp frame(s)' settings, same as any other frame-built artifact).
"""

# Imports

import logging
from pathlib import Path

import numpy as np

from ..shared.io import save_artifact, load_artifact
from ..shared.metadata import CalibrationRecord
from ..shared.result import PolynomialFitResult
from .calibrate import WavelengthCalibrationResult

# Constants

logger = logging.getLogger(__name__)

# Classes

# Functions

def save_spectral_calibration(path: str | Path, result: WavelengthCalibrationResult) -> None:

    '''
    Saves a spectral calibration artifact to path, so it can be reused in
    a later session without recapturing/rematching lamp lines. Overwrites
    whatever was already at path -- current instrument state, not a
    history to keep.

    Parameters
    ----------
    path
        Destination file. A ".npz" suffix is added if not already
        present.
    result
        The WavelengthCalibrationResult returned by calibrate_spectral().

    Returns
    -------
    None
    '''

    fit = result.fit
    arrays = {
        "coefficients": fit.coefficients,
        "coefficient_sigma": fit.coefficient_sigma,
        "residuals": fit.residuals,
        "normalized_residuals": fit.normalized_residuals,
        "degree": np.array(fit.degree),
        "reduced_chi_squared": np.array(fit.reduced_chi_squared),
    }
    save_artifact(path, arrays, result.record)


def load_spectral_calibration(path: str | Path) -> WavelengthCalibrationResult:

    '''
    Loads a spectral calibration previously saved via
    save_spectral_calibration().

    Parameters
    ----------
    path
        The file to load. A ".npz" suffix is added if not already
        present.

    Returns
    -------
    WavelengthCalibrationResult

    Raises
    ------
    FileNotFoundError
        If path doesn't exist.
    '''

    arrays, record = load_artifact(path, CalibrationRecord)
    fit = PolynomialFitResult(
        degree=int(arrays["degree"]),
        coefficients=arrays["coefficients"],
        coefficient_sigma=arrays["coefficient_sigma"],
        reduced_chi_squared=float(arrays["reduced_chi_squared"]),
        residuals=arrays["residuals"],
        normalized_residuals=arrays["normalized_residuals"],
    )
    logger.info("loaded spectral calibration from %s (age %.1fs)", path, record.age_seconds)
    return WavelengthCalibrationResult(fit=fit, record=record)


__all__ = ["save_spectral_calibration", "load_spectral_calibration"]
