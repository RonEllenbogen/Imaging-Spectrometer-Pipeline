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
scope for this project -- a single scale factor is precise enough on its
own. That scale factor is now a directly measured quantity rather than a
theoretical design value: DEFAULT_SCALE_FACTOR = 1.5 is the rounded
result of a real measurement (raw data:
data/diagnostic/spatial_calibration_14.8.26/, a 3x3 grid of captures --
measured 1.504 +/- 0.008), and DEFAULT_SIGMA_SCALE_FACTOR = 0.01 is that
measurement's 1-sigma uncertainty, rounded the same way. This supersedes
the project's earlier stance (docs/project_state.md) that no uncertainty
needed tracking because the relay lenses' nominal focal lengths were
"known to high precision" and misalignment would only show up as
blur/aberration, never a quantifiable number -- a real measurement now
exists, carries real uncertainty, and convert() propagates it into every
returned position's own uncertainty (see below). Both defaults are used
unless the GUI/CLI user has entered a manually-measured override (value
AND its own uncertainty -- see calibration_dialogs.py/cli/calibration.py),
which is then persisted via io.py and reused in future sessions.

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
# length to its second. Directly measured (see module docstring for
# provenance and DEFAULT_SIGMA_SCALE_FACTOR below for its uncertainty).
DEFAULT_SCALE_FACTOR = 1.5

# 1-sigma uncertainty on DEFAULT_SCALE_FACTOR. Measured 1.504 +/- 0.008
# (data/diagnostic/spatial_calibration_14.8.26/); rounded to the same
# precision as the rounded default above.
DEFAULT_SIGMA_SCALE_FACTOR = 0.01

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
        DEFAULT_SCALE_FACTOR unless the GUI/CLI user has entered a
        manually calibrated value (see io.py). PIXEL_PITCH_UM (detector
        hardware spec, not user-measured) is applied separately in
        convert().
    sigma_scale_factor
        1-sigma uncertainty on scale_factor. DEFAULT_SIGMA_SCALE_FACTOR
        unless the GUI/CLI user has entered a manually calibrated value
        alongside their own scale_factor override. A single, session-wide
        value applied identically to every point converted -- a
        systematic, fully-correlated uncertainty, not an independent
        per-point one (see convert()).
    '''

    scale_factor: float = DEFAULT_SCALE_FACTOR
    sigma_scale_factor: float = DEFAULT_SIGMA_SCALE_FACTOR

    def __post_init__(self) -> None:
        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")
        if self.sigma_scale_factor < 0:
            raise ValueError(
                f"sigma_scale_factor must be non-negative, got {self.sigma_scale_factor}"
            )

    def convert(
        self, x0: np.ndarray, sigma_x0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        '''
        See analysis.interfaces.PositionCalibration.convert. Returns
        physical position in microns (see module docstring for why).

        Propagates two independent 1-sigma error sources into the
        returned uncertainty: x0's own uncertainty (sigma_x0, scaled by
        the same combined factor as x0 itself), and scale_factor's own
        uncertainty (sigma_scale_factor). For y = k*x0 with both k
        (= PIXEL_PITCH_UM * scale_factor) and x0 uncertain and
        independent, standard error propagation gives
        sigma_y**2 = (k * sigma_x0)**2 + (x0 * PIXEL_PITCH_UM * sigma_scale_factor)**2
        -- the second term requires x0 itself (not just sigma_x0), since
        it's proportional to how far from zero the point being converted
        is. At sigma_scale_factor=0 this collapses exactly to the old
        single-term "just rescale sigma_x0" behavior.

        sigma_scale_factor must only ever be applied here, on an already
        final, already pixel-domain-combined quantity (a combined
        zeta/coefficient, or one shot's own fitted result) -- never
        earlier, on a per-column/per-shot value that then gets averaged,
        since it is a single systematic uncertainty shared across every
        point converted in a session, not an independent per-point one;
        injecting it earlier would let it incorrectly shrink under
        sqrt(N)-style averaging.
        '''

        combined_factor = PIXEL_PITCH_UM * self.scale_factor
        converted_x0 = combined_factor * x0
        variance_from_x0 = (combined_factor * sigma_x0) ** 2
        variance_from_scale_factor = (x0 * PIXEL_PITCH_UM * self.sigma_scale_factor) ** 2
        converted_sigma = np.sqrt(variance_from_x0 + variance_from_scale_factor)
        return converted_x0, converted_sigma

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


__all__ = [
    "ScaleFactorPositionCalibration",
    "DEFAULT_SCALE_FACTOR",
    "DEFAULT_SIGMA_SCALE_FACTOR",
    "PIXEL_PITCH_UM",
]
