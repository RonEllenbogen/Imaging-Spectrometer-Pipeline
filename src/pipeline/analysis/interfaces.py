"""
Boundary contracts between analysis and the not-yet-built calibration
package (docs/project_state.md: calibration/spectral/ and
calibration/spatial/ are both "designed, not built"). analysis/ depends on
these interfaces, never on calibration/ directly -- keeping analysis
usable and testable today, before either calibration subpackage exists,
and decoupled from however they end up being implemented.
"""

# Imports

from typing import Protocol

import numpy as np

# Constants

# Classes

class WavelengthAxis(Protocol):

    '''
    Supplies wavelength, in nanometres (see "Wavelength convention" in
    docs/project_state.md), for a given set of pixel-column indices, plus
    its uncertainty. Implemented, eventually, by calibration/spectral/,
    whose pixel->wavelength(nm) polynomial fit is used directly --
    analysis performs no further unit conversion of its own.
    '''

    def wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:

        '''Wavelength, in nm, at each given pixel-column index.'''

        ...

    def sigma_wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:

        '''
        1-sigma uncertainty on wavelength_nm() at each given pixel-column
        index. Must be strictly positive everywhere -- TotalLeastSquaresFit
        (dispersion_fitting.py) is backed by scipy.odr, which requires
        positive input standard deviations on both axes.
        '''

        ...


class PositionCalibration(Protocol):

    '''
    Optional conversion from pixel-index position to physical position at
    the input plane. Implemented, eventually, by calibration/spatial/.
    Nothing in analysis/ requires this -- results stay in pixel units
    (docs/project_state.md #21) whenever it's omitted, which is always,
    until calibration/spatial/ exists.
    '''

    def convert(
        self, x0: np.ndarray, sigma_x0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        '''
        Converts pixel-unit (x0, sigma_x0) arrays to physical units. The
        returned uncertainty reflects both x0's own input uncertainty and
        the position calibration's own uncertainty (e.g. a measured
        scale factor's sigma) -- not x0's uncertainty alone -- so it may
        be larger than a naive rescaling of sigma_x0 would suggest.
        '''

        ...

# Functions


__all__ = ["WavelengthAxis", "PositionCalibration"]
