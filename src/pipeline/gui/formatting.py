'''
Pure, non-Qt display-formatting helpers, shared across gui/ screens.
Kept free of any Qt import (unlike live_view.py's widget class) so it can
be unit-tested directly, the same way live_view.py's own
wavelength_axis_label()/heatmap_x_extent() are split out from the widget
class that calls them.
'''

# Imports

import math
from decimal import ROUND_HALF_UP, Decimal

# Constants

MICRONS_PER_MM = 1000.0
NM_PER_MICRON = 1000.0

# Classes

# Functions


def format_value_with_uncertainty(value: float, sigma: float) -> str:

    '''
    Formats "value ± sigma" as one string, per the standard
    uncertainty-rounding convention: sigma is rounded to 1 significant
    figure, unless its leading significant digit is 1 -- checked AFTER
    rounding, so a sigma that rounds *up into* a leading "1" (e.g. 0.95,
    which rounds to 1.0 at 1 significant figure) is caught too, not just
    a sigma whose un-rounded leading digit already reads "1" (e.g.
    0.0163) -- in which case 2 significant figures are used instead, to
    avoid a misleadingly large relative rounding step right at the "1"
    boundary. value is then rounded to the SAME decimal precision as the
    rounded sigma (never independently rounded to its own precision).

    Parameters
    ----------
    value
        Central value.
    sigma
        Uncertainty (standard deviation) on value. Must be finite and
        strictly positive -- this rounding convention has no meaning for
        a zero, negative, or non-finite uncertainty.

    Returns
    -------
    str
        "value ± sigma", both rounded/decimal-aligned per the
        convention above, using the Unicode "±" character (never
        the two-character ASCII "+/-" approximation).

    Raises
    ------
    ValueError
        If sigma is not finite and strictly positive, or value is not
        finite.
    '''

    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be finite and positive, got {sigma!r}")
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")

    sigma_decimal = Decimal(str(sigma))
    leading_digit_before = _leading_significant_digit(sigma_decimal)

    if leading_digit_before == 1:
        # Already reads "1..." before any rounding -- 2 significant
        # figures directly, no need to check a provisional 1-sig-fig
        # rounding first.
        significant_figures = 2
    else:
        provisional = _round_to_significant_figures(sigma_decimal, 1)
        if _leading_significant_digit(provisional) == 1:
            # Rounding to 1 significant figure carried into the next
            # power of ten (e.g. 0.95 -> 1.0) -- re-round from the
            # original value at 2 significant figures instead.
            significant_figures = 2
        else:
            significant_figures = 1

    sigma_rounded = _round_to_significant_figures(sigma_decimal, significant_figures)

    # value is rounded to the exact same decimal precision as the
    # rounded sigma -- i.e. quantized to the same power-of-ten step,
    # never independently rounded to its own precision.
    quantum = Decimal(1).scaleb(sigma_rounded.as_tuple().exponent)
    value_rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)

    return f"{format(value_rounded, 'f')} ± {format(sigma_rounded, 'f')}"


def _leading_significant_digit(magnitude: Decimal) -> int:

    '''Leading (most significant) digit of a positive Decimal, as an int 1-9.'''

    exponent = magnitude.adjusted()
    return int(magnitude.scaleb(-exponent))


def _round_to_significant_figures(magnitude: Decimal, significant_figures: int) -> Decimal:

    '''Rounds a positive Decimal to significant_figures figures, half-away-from-zero.'''

    exponent = magnitude.adjusted()
    quantum = Decimal(1).scaleb(exponent - significant_figures + 1)
    return magnitude.quantize(quantum, rounding=ROUND_HALF_UP)


def microns_to_mm(value_um: float) -> float:

    '''Converts microns to mm, since calibration/spatial/calibrate.py's ScaleFactorPositionCalibration.convert() returns microns but gui/ screens display in mm.'''

    return value_um / MICRONS_PER_MM


def mm_to_microns(value_mm: float) -> float:

    '''Inverse of microns_to_mm().'''

    return value_mm * MICRONS_PER_MM


def microns_to_nm(value_um: float) -> float:

    '''
    Converts microns to nm -- this codebase's convention for *quoted*
    spatial-dispersion/polynomial-coefficient values (spatial dispersion
    and every coefficient c0..c_degree are reported in nm-based units,
    e.g. c1 in nm/nm), kept deliberately separate from microns_to_mm()
    (still used for every plotted graph's position axis, which stays in
    mm regardless of this convention -- see gui/extended_measurement.py's
    module docstring).
    '''

    return value_um * NM_PER_MICRON


def nm_to_microns(value_nm: float) -> float:

    '''Inverse of microns_to_nm().'''

    return value_nm / NM_PER_MICRON


def coefficient_unit(k: int) -> str:

    '''
    Unit label for polynomial coefficient c_k in x0 = c0 + c1*wavelength_nm
    + c2*wavelength_nm**2 + ... , with position expressed in nm (this
    codebase's quoted-value convention -- see microns_to_nm()): c0 is a
    plain position (nm); c1 is a position-per-wavelength ratio (nm/nm,
    reported with matching numerator/denominator units rather than
    simplified to a dimensionless number, so the wavelength normalization
    stays explicit); c2 and above follow the same pattern (nm/nm^k).
    Shared by every gui/ screen and measurement_record.py that displays a
    combined or per-shot polynomial fit, so the unit convention can't
    drift between them.
    '''

    if k == 0:
        return "nm"
    if k == 1:
        return "nm/nm"
    return f"nm/nm^{k}"


__all__ = [
    "format_value_with_uncertainty",
    "MICRONS_PER_MM",
    "NM_PER_MICRON",
    "microns_to_mm",
    "mm_to_microns",
    "microns_to_nm",
    "nm_to_microns",
    "coefficient_unit",
]
