"""
Exceptions for the analysis package. AnalysisError is the base class for
anything that goes wrong inside analysis/; InsufficientDataError is the
one specific subclass, reserved for a hard mathematical impossibility (not
enough columns to fit the requested polynomial degree) rather than a
policy judgment call -- unlike preprocessing's threshold-driven exceptions,
analysis deliberately has no minimum-valid-column or minimum-signal gate
(see docs/project_state.md, decisions #8/#18); this exception exists only
because a fit genuinely cannot be solved with fewer points than free
parameters.
"""

# Imports

# Constants

# Classes

class AnalysisError(Exception):
    """Base class for all analysis-related errors."""


class InsufficientDataError(AnalysisError):
    """Raised when a spatial-dispersion fit is asked to solve for more
    polynomial coefficients than it has data points to usefully estimate
    them from -- a hard mathematical requirement (degree + 2 points
    minimum, not degree + 1), not a configurable threshold. degree + 1
    points is enough to solve for the coefficients themselves (an exact
    interpolation), but leaves zero residual degrees of freedom -- no
    excess data to estimate a reduced chi-squared or a coefficient
    uncertainty FROM, so scipy.odr reports both as (near-)zero rather
    than a real number, and that zero uncertainty then blows up
    downstream anywhere it's displayed or propagated (e.g.
    gui/formatting.py's format_value_with_uncertainty(), which requires a
    strictly positive sigma). degree + 2 is the smallest point count with
    at least one residual degree of freedom, so a fit's reported
    uncertainty is always statistically meaningful, never degenerate."""

    def __init__(self, degree: int, n_points: int):
        super().__init__(
            f"cannot fit degree-{degree} polynomial with only {n_points} "
            f"point(s); need at least {degree + 2} for a meaningful "
            f"uncertainty estimate (degree + 1 alone is an exact "
            f"interpolation with no residual degrees of freedom)"
        )
        self.degree = degree
        self.n_points = n_points

# Functions


__all__ = ["AnalysisError", "InsufficientDataError"]
