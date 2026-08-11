"""
Pixel -> physical-position conversion at the spectrometer's input slit.
Two multiplicative steps, both needed: PIXEL_PITCH_UM converts a
pixel-index displacement to a physical distance AT THE DETECTOR (a fixed
sensor hardware spec, a2A1920-51gmBAS datasheet, loaded from
configs/default.yaml); scale_factor -- the ratio of the imaging
spectrometer's two relay-lens focal lengths (f1/f2) -- then converts that
detector-plane distance to the slit-plane distance where the physically
meaningful spatial chirp actually lives (see docs/project_handover.md
§5). Pixel pitch alone is NOT sufficient on its own for this reason --
it only gets as far as the detector plane.

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

convert() returns physical position in MICRONS, matching PIXEL_PITCH_UM's
own unit -- a deliberate choice so the unit is self-evident from reading
this file, rather than an implicit mm conversion buried in the
calibration math. Converting to mm (or any other display unit) for a
human-readable axis/label is the caller's job, e.g. gui/'s.
"""

# Imports

from dataclasses import dataclass

import numpy as np

from ...utils.helpers import load_config

# Constants

_config = load_config("configs/default.yaml")

# Sensor pixel pitch, in microns -- a2A1920-51gmBAS datasheet spec, not a
# per-instrument measurement, so it's config-driven the same way
# canonical_shape/pixel_format are, not user-measured/persisted like
# scale_factor below.
PIXEL_PITCH_UM = float(_config["camera"]["pixel_pitch_um"])

# f1/f2: ratio of the imaging spectrometer's first relay lens focal
# length to its second. Fixed by the optical design -- see module
# docstring for why no uncertainty is tracked on this value.
DEFAULT_SCALE_FACTOR = 1.5

# Classes

@dataclass(frozen=True, slots=True)
class ScaleFactorPositionCalibration:

    '''
    Implements analysis.interfaces.PositionCalibration by converting
    pixel-unit positions to physical microns at the slit: pixel index x
    PIXEL_PITCH_UM (pixel -> detector-plane distance) x scale_factor
    (detector-plane -> slit-plane distance). No per-point fit -- this
    project's spatial calibration is a single known (or user-measured)
    ratio on top of a fixed hardware pixel pitch, not a pixel->position
    mapping built from a translation-stage measurement session (see
    module docstring). to_pixels() is the inverse of convert(), used by
    the GUI's manual ROI entry.

    Parameters
    ----------
    scale_factor
        Relay-optics magnification (detector-plane to slit-plane).
        DEFAULT_SCALE_FACTOR unless the GUI user has entered a manually
        calibrated value (see io.py). PIXEL_PITCH_UM (detector hardware
        spec, not user-measured) is applied separately in convert().
    '''

    scale_factor: float = DEFAULT_SCALE_FACTOR

    def __post_init__(self) -> None:
        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")

    def convert(
        self, x0: np.ndarray, sigma_x0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        '''
        See analysis.interfaces.PositionCalibration.convert. Returns
        physical position in microns (see module docstring for why).
        sigma_x0 scales by the same combined factor as x0 -- neither
        PIXEL_PITCH_UM nor scale_factor carries its own uncertainty (see
        module docstring), so there's no additional term to propagate.
        '''

        combined_factor = PIXEL_PITCH_UM * self.scale_factor
        return combined_factor * x0, combined_factor * sigma_x0

    def to_pixels(self, physical_position_um: np.ndarray) -> np.ndarray:

        '''
        Inverse of convert(): physical position (microns, at the slit
        plane) back to a pixel-index position (at the detector). No
        uncertainty term -- unlike convert(), this exists only for the
        GUI's manual ROI entry (gui/live_view.py, gui/roi_control.py), where
        the input is an exact user-entered bound, not a measurement with
        its own sigma.
        '''

        combined_factor = PIXEL_PITCH_UM * self.scale_factor
        return physical_position_um / combined_factor


# Functions


__all__ = ["ScaleFactorPositionCalibration", "DEFAULT_SCALE_FACTOR", "PIXEL_PITCH_UM"]
