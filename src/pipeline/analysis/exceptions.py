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
    polynomial coefficients than it has data points for -- a hard
    mathematical requirement (degree + 1 points minimum), not a
    configurable threshold."""

    def __init__(self, degree: int, n_points: int):
        super().__init__(
            f"cannot fit degree-{degree} polynomial with only {n_points} "
            f"point(s); need at least {degree + 1}"
        )
        self.degree = degree
        self.n_points = n_points

# Functions


__all__ = ["AnalysisError", "InsufficientDataError"]
