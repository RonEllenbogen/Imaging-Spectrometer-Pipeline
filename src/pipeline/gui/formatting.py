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


__all__ = ["format_value_with_uncertainty"]
