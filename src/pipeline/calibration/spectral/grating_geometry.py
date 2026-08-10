"""
Predicts relative spectral-line spacing on the detector from the
spectrometer's transmission-grating geometry, for line_matching.py's
peak-to-reference-line matching search. Deliberately does NOT predict an
absolute pixel column for a given wavelength -- that depends on the
camera's precise physical translation/alignment, which is not known here
(see line_matching.py's module docstring). What IS predictable from
geometry alone is the relative pixel separation between any two
wavelengths, which is what the matching search actually needs: a
"fingerprint" pattern of expected spacings to compare against detected
peak spacings.

Grating equation (m=-1, fixed for this hardware): m*lambda*rho =
sin(theta_m) - sin(theta_i), giving sin(theta_m(lambda)) = sin(theta_i) -
lambda*rho. incidence_angle_deg (theta_i) is only approximately known
("roughly 15 degrees") -- this is fine for THIS module's purpose, since
theta_i's imprecision affects how well the predicted spacing pattern
matches reality (and therefore the matching search's confidence), not the
final wavelength calibration's accuracy, which comes entirely from
calibrate.py's fit to whichever (pixel, wavelength) pairs the matching
search actually identifies.

The second relay lens (focal length lens_focal_length_mm, positioned one
focal length from the grating along the central wavelength's optical
axis, and one focal length from the detector along the same axis) converts
angular deviation from that optical axis to a transverse displacement at
the detector plane via the small-angle relation delta_y = f * delta_theta.
Converting that detector-plane displacement to a PIXEL displacement reuses
calibration.spatial.PIXEL_PITCH_UM directly (no relay-optics scale factor
here, unlike calibration/spatial/calibrate.py's spatial-axis conversion --
this is a physically different axis of the same sensor, measured directly
at the detector plane, not at the spectrometer's slit plane).
"""

# Imports

import math

from ...utils.helpers import load_config
from ..spatial import PIXEL_PITCH_UM

# Constants

_config = load_config("configs/default.yaml")
_spectrometer_config = _config["spectrometer"]

# rho: grating groove density, lines/mm in the config but used here as
# lines/nm (wavelength is always handled in nm throughout calibration/spectral/,
# see calibrate.py) -- converted once at import time.
GRATING_LINES_PER_MM = float(_spectrometer_config["grating_lines_per_mm"])
_GRATING_LINES_PER_NM = GRATING_LINES_PER_MM / 1.0e6

# theta_i: angle of incidence on the grating, degrees in config (human-
# readable), converted to radians once at import time for direct use in
# diffraction_angle_rad().
INCIDENCE_ANGLE_DEG = float(_spectrometer_config["incidence_angle_deg"])
_INCIDENCE_ANGLE_RAD = math.radians(INCIDENCE_ANGLE_DEG)

# f: second relay lens focal length, mm.
LENS_FOCAL_LENGTH_MM = float(_spectrometer_config["lens_focal_length_mm"])

# Diffraction order -- fixed for this hardware's optical design, not a
# free parameter (see module docstring).
_DIFFRACTION_ORDER = -1

# Classes

# Functions


def diffraction_angle_rad(wavelength_nm: float) -> float:

    '''
    theta_m(wavelength_nm): the diffracted-beam angle for a given
    wavelength, per the grating equation with m=-1 -- see module
    docstring for the full derivation and its limitations.

    Parameters
    ----------
    wavelength_nm
        Wavelength, in nanometres.

    Returns
    -------
    float
        Diffraction angle, in radians.

    Raises
    ------
    ValueError
        If the grating equation has no real solution at this wavelength
        (|sin(theta_i) - lambda*rho| > 1) -- physically, this wavelength
        cannot be diffracted at all at this incidence angle/groove
        density.
    '''

    sin_theta_m = math.sin(_INCIDENCE_ANGLE_RAD) + _DIFFRACTION_ORDER * wavelength_nm * _GRATING_LINES_PER_NM
    if abs(sin_theta_m) > 1.0:
        raise ValueError(
            f"no real diffraction angle for wavelength_nm={wavelength_nm!r} "
            f"(sin(theta_m)={sin_theta_m!r} out of [-1, 1])"
        )
    return math.asin(sin_theta_m)


def predicted_pixel_separation(wavelength_a_nm: float, wavelength_b_nm: float) -> float:

    '''
    Predicted signed pixel displacement between two wavelengths' positions
    on the detector -- positive if wavelength_a_nm falls at a larger pixel
    column than wavelength_b_nm, negative otherwise. Purely a difference
    (see module docstring for why this is predictable while an absolute
    pixel position is not): the central wavelength's own diffraction angle
    cancels out entirely, so this needs no assumption about what the
    central wavelength actually is.

    Parameters
    ----------
    wavelength_a_nm, wavelength_b_nm
        Wavelengths to compare, in nanometres.

    Returns
    -------
    float
        Predicted pixel separation (can be fractional -- this is a
        continuous physical prediction, not a detected integer pixel).
    '''

    delta_theta_rad = diffraction_angle_rad(wavelength_a_nm) - diffraction_angle_rad(wavelength_b_nm)
    delta_y_mm = LENS_FOCAL_LENGTH_MM * delta_theta_rad
    delta_y_um = delta_y_mm * 1000.0
    return delta_y_um / PIXEL_PITCH_UM


__all__ = [
    "diffraction_angle_rad",
    "predicted_pixel_separation",
    "GRATING_LINES_PER_MM",
    "INCIDENCE_ANGLE_DEG",
    "LENS_FOCAL_LENGTH_MM",
]
