"""
Detects spectral line peaks in a preprocessed, averaged lamp-calibration
image and matches them to known reference wavelengths, producing the
(pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm) arrays
calibrate.py's calibrate_spectral() fits.

BLOCKED: not yet implemented. Requires two pieces of information not yet
available:
  - which reference lamp will be used (e.g. Ne, Ar, HgAr) and its known
    reference line wavelengths;
  - an approximate prior pixel->wavelength dispersion (from the grating
    equation / optical design), needed to turn "N detected peaks" into
    "peak i is reference line j" via nearest-neighbor matching + iterative
    outlier rejection, rather than a blind/ambiguous assignment problem
    (docs/project_handover.md §5).
Both are lamp-hardware decisions outside this codebase's scope until a
lamp is chosen -- see docs/project_state.md for the current status of
this decision.
"""

# Imports

import numpy as np

# Constants

# Classes

# Functions

def match_lines(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    '''
    Detects and identifies spectral lines in a preprocessed lamp image.

    Parameters
    ----------
    image
        Preprocessed, averaged lamp-calibration image (spatial x
        spectral, matching CANONICAL_SHAPE) -- see
        calibration/spectral/workflow.py.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm) -- one
        entry per matched line, ready for calibrate.calibrate_spectral().

    Raises
    ------
    NotImplementedError
        Always -- see module docstring.
    '''

    raise NotImplementedError(
        "spectral line matching is blocked on reference lamp selection -- see module docstring"
    )


__all__ = ["match_lines"]
