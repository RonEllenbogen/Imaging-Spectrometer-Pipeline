'''
Test suite for gui/formatting.py's format_value_with_uncertainty() --
plain arithmetic, no Qt/display involved, so (unlike tests/test_gui.py)
this module needs no PySide6/pyqtgraph/pytest-qt and always runs.
'''

# Imports

import pytest

from pipeline.gui.formatting import format_value_with_uncertainty

# Constants

# Classes

# Functions


class TestOrdinaryRounding:

    def test_leading_digit_not_one_rounds_sigma_to_one_sig_fig(self):
        # sigma=2.1e-5, leading digit 2 -> 1 significant figure -> 2e-5,
        # value aligned to the same (5) decimal places.
        assert format_value_with_uncertainty(1.6234e-3, 2.1e-5) == "0.00162 ± 0.00002"

    def test_leading_digit_four_rounds_up_to_nearest_ten(self):
        # sigma=45, leading digit 4 -> 1 significant figure -> 50 (not a
        # boundary case -- 4 stays 4, no carry into a leading "1").
        # value rounds to the same (nearest-ten) precision.
        assert format_value_with_uncertainty(123.0, 45.0) == "120 ± 50"

    def test_negative_value_rounds_correctly(self):
        assert format_value_with_uncertainty(-3.456, 0.21) == "-3.5 ± 0.2"


class TestLeadingDigitOneUsesTwoSigFigs:

    def test_sigma_already_reading_one_uses_two_sig_figs(self):
        # sigma=0.0163, leading digit 1 before any rounding -> 2
        # significant figures -> 0.016; value shown to the same (3)
        # decimal places.
        assert format_value_with_uncertainty(0.012345, 0.0163) == "0.012 ± 0.016"


class TestRoundingBoundary:

    '''
    The easiest place to get this convention subtly wrong: a sigma whose
    un-rounded leading digit is NOT 1, but which rounds UP into a
    leading "1" at 1 significant figure. The leading-digit check must be
    made on the rounded value, not the original one, or this case
    silently reports the same (too-coarse) precision as an ordinary
    1-sig-fig sigma.
    '''

    def test_sigma_at_exact_boundary_escalates_to_two_sig_figs(self):
        # 0.95 rounds to 1.0 at 1 significant figure -- its rounded
        # leading digit is 1, so this must escalate to 2 significant
        # figures instead, leaving 0.95 itself unchanged (it already had
        # 2 significant figures) rather than coarsening it to 1.0.
        assert format_value_with_uncertainty(1.0, 0.95) == "1.00 ± 0.95"

    def test_sigma_that_carries_past_a_power_of_ten_escalates(self):
        # 0.096 rounds to 0.10 at 1 significant figure -- leading digit
        # 1 after rounding -- so this escalates and re-rounds the
        # ORIGINAL value at 2 significant figures (0.096, unchanged),
        # not the coarser carried-over 0.10 a naive 1-sig-fig-only
        # implementation would report.
        assert format_value_with_uncertainty(1.0, 0.096) == "1.000 ± 0.096"

    def test_sigma_just_below_the_carry_does_not_escalate(self):
        # 0.09 does NOT round up into a leading "1" at 1 significant
        # figure (it already is one) -- no escalation, ordinary 1-sig-fig
        # behavior applies.
        assert format_value_with_uncertainty(1.0, 0.09) == "1.00 ± 0.09"


class TestInvalidInputs:

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            format_value_with_uncertainty(1.0, 0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            format_value_with_uncertainty(1.0, -0.5)

    def test_non_finite_sigma_raises(self):
        with pytest.raises(ValueError):
            format_value_with_uncertainty(1.0, float("nan"))

    def test_non_finite_value_raises(self):
        with pytest.raises(ValueError):
            format_value_with_uncertainty(float("inf"), 0.1)
