"""
Pixel -> physical-position conversion at the spectrometer's input slit,
via a fixed scale factor -- the ratio of the imaging spectrometer's two
relay-lens focal lengths (f1/f2), which sets the relay optics'
magnification between the detector plane and the slit plane. Camera pixel
pitch alone is NOT sufficient here, since it only converts pixels to
distance AT THE DETECTOR, not at the slit/input plane where the
physically meaningful spatial chirp lives (see docs/project_handover.md
§5).

Full per-point translation-stage calibration (matching detector pixel
displacement to a known physical displacement at the slit) is out of
scope for this project -- DEFAULT_SCALE_FACTOR is precise enough on its
own, since the relay lenses' focal lengths are known to high precision.
The only real source of error (optical misalignment, incorrect component
spacing) manifests as blur/aberration in the image, not as a quantifiable
scale-factor uncertainty -- so no uncertainty is tracked on scale_factor
itself (docs/project_state.md). DEFAULT_SCALE_FACTOR is used unless the
GUI user has entered a better-measured value, which is then persisted via
io.py and reused in future sessions.
"""

# Imports

from dataclasses import dataclass

import numpy as np

# Constants

# f1/f2: ratio of the imaging spectrometer's first relay lens focal
# length to its second. Fixed by the optical design -- see module
# docstring for why no uncertainty is tracked on this value.
DEFAULT_SCALE_FACTOR = 1.5

# Classes

@dataclass(frozen=True, slots=True)
class ScaleFactorPositionCalibration:

    '''
    Implements analysis.interfaces.PositionCalibration by scaling pixel-
    unit positions by a fixed factor. No per-point fit -- this project's
    spatial calibration is a single known (or user-measured) ratio, not a
    pixel->position mapping built from a translation-stage measurement
    session (see module docstring).

    Parameters
    ----------
    scale_factor
        Physical-position units per detector pixel. DEFAULT_SCALE_FACTOR
        unless the GUI user has entered a manually calibrated value (see
        io.py).
    '''

    scale_factor: float = DEFAULT_SCALE_FACTOR

    def __post_init__(self) -> None:
        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")

    def convert(
        self, x0: np.ndarray, sigma_x0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        '''
        See analysis.interfaces.PositionCalibration.convert. sigma_x0
        scales by exactly the same factor as x0 -- scale_factor itself
        carries no uncertainty of its own (see module docstring), so
        there's no additional term to propagate.
        '''

        return self.scale_factor * x0, self.scale_factor * sigma_x0


# Functions


__all__ = ["ScaleFactorPositionCalibration", "DEFAULT_SCALE_FACTOR"]
