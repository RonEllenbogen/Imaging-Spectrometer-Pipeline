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

class FrequencyAxis(Protocol):

    '''
    Supplies angular frequency (never ordinary frequency -- see
    docs/project_state.md #14) for a given set of pixel-column indices,
    plus its uncertainty. Implemented, eventually, by
    calibration/spectral/ -- analysis never performs any lambda->omega
    unit conversion itself; that conversion happens once, at
    calibration-build time, on whatever pixel->lambda dispersion relation
    calibration fits (docs/project_state.md #13/#15).
    '''

    def omega(self, pixel: np.ndarray) -> np.ndarray:

        '''Angular frequency at each given pixel-column index.'''

        ...

    def sigma_omega(self, pixel: np.ndarray) -> np.ndarray:

        '''
        1-sigma uncertainty on omega() at each given pixel-column index.
        Must be strictly positive everywhere -- TotalLeastSquaresFit
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

        '''Converts pixel-unit (x0, sigma_x0) arrays to physical units.'''

        ...

# Functions


__all__ = ["FrequencyAxis", "PositionCalibration"]
