"""
Combines the linear (degree-1) spatial dispersion across N shots via
inverse-variance weighting (docs/project_state.md #19), reporting
whichever of the internal (propagated) or external (empirical-scatter)
uncertainty is larger (docs/project_state.md #20) -- the standard
internal/external error comparison (Bevington & Robinson; also the
Particle Data Group's method for averaging a set of measurements), which
protects the reported uncertainty against shot-to-shot jitter without
double-counting by quadrature-summing both estimates.

Deliberately a plain function, not a Protocol -- unlike the centroid
estimator and fit method, this hasn't been asked to be swappable, and the
internal/external method is a fixed statistical procedure rather than a
"which algorithm" choice.

Agnostic to N (docs/project_state.md #19) -- no minimum shot count is
enforced; how many shots to combine, and when, is an orchestration-layer
decision (live mode: every N_default shots; batch mode: user-specified
N), not something this function decides for its caller.
"""

# Imports

import numpy as np

from .results import CombinedSpatialDispersionResult

# Constants

# Classes

# Functions

def combine_shots(
    zeta_values: np.ndarray, sigma_zeta_values: np.ndarray
) -> CombinedSpatialDispersionResult:

    '''
    Parameters
    ----------
    zeta_values
        Each shot's fitted degree-1 spatial dispersion.
    sigma_zeta_values
        Each shot's coefficient_sigma[1] (the linear fit's slope
        uncertainty) -- the per-shot uncertainty being combined.

    Returns
    -------
    CombinedSpatialDispersionResult

    Raises
    ------
    ValueError
        If zeta_values and sigma_zeta_values don't have the same shape,
        or are empty.
    '''

    if zeta_values.shape != sigma_zeta_values.shape:
        raise ValueError(
            "zeta_values and sigma_zeta_values must have the same shape, got "
            f"{zeta_values.shape}, {sigma_zeta_values.shape}"
        )
    n_shots = zeta_values.shape[0]
    if n_shots < 1:
        raise ValueError("combine_shots() requires at least one shot")

    weights = 1.0 / sigma_zeta_values ** 2
    zeta_combined = np.sum(weights * zeta_values) / np.sum(weights)
    sigma_internal = 1.0 / np.sqrt(np.sum(weights))

    if n_shots > 1:
        weighted_scatter = np.sum(weights * (zeta_values - zeta_combined) ** 2)
        sigma_external = np.sqrt(weighted_scatter / ((n_shots - 1) * np.sum(weights)))
    else:
        # No scatter information from a single shot -- "external" is
        # undefined for N=1, so fall back to internal error only.
        sigma_external = sigma_internal

    sigma_zeta_combined = max(sigma_internal, sigma_external)

    return CombinedSpatialDispersionResult(
        zeta_combined=float(zeta_combined),
        sigma_internal=float(sigma_internal),
        sigma_external=float(sigma_external),
        sigma_zeta_combined=float(sigma_zeta_combined),
        n_shots=n_shots,
    )


__all__ = ["combine_shots"]
