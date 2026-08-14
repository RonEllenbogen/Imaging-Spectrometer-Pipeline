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

NOT order-blind, unlike an earlier version of this module: sigma_external
is now estimated via a moving-block bootstrap (analysis/block_bootstrap.py),
which needs shot order preserved to detect and resample real shot-to-shot
correlation. A real recorded 200-shot measurement
(data/measurements/extended_measurement_20260813_171209/) showed the
previous, order-blind "weighted scatter around the combined mean"
sigma_external implicitly assumed independent shots -- false in practice
(lag-1 autocorrelation ~0.89, a genuine slow shot-to-shot wander, not
noise), which made it a real underestimate. zeta_combined and
sigma_internal are still exactly what they were -- pure inverse-variance
arithmetic, insensitive to order either way -- only sigma_external (and
therefore sigma_zeta_combined, whenever it's the larger of the two) reads
zeta_values/sigma_zeta_values as an ordered, shot-acquisition-order
series now. Callers must pass shots in acquisition order.
"""

# Imports

import numpy as np

from .block_bootstrap import DEFAULT_N_RESAMPLES, moving_block_bootstrap_sigma_external, select_block_length
from .results import CombinedSpatialDispersionResult

# Constants

# Classes

# Functions

def combine_shots(
    zeta_values: np.ndarray,
    sigma_zeta_values: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> CombinedSpatialDispersionResult:

    '''
    Parameters
    ----------
    zeta_values
        Each shot's fitted degree-1 spatial dispersion, in acquisition
        order -- see module docstring for why order now matters (the
        moving-block bootstrap below needs it to detect/preserve
        shot-to-shot correlation).
    sigma_zeta_values
        Each shot's coefficient_sigma[1] (the linear fit's slope
        uncertainty) -- the per-shot uncertainty being combined, same
        order as zeta_values.
    n_resamples
        Number of moving-block-bootstrap resamples used to estimate
        sigma_external (see analysis/block_bootstrap.py's
        moving_block_bootstrap_sigma_external()). Ignored at n_shots == 1
        (no bootstrap runs -- see below). block_bootstrap.DEFAULT_N_RESAMPLES
        is precise enough for an offline/persisted use; a caller on a
        tighter interactive time budget (e.g. a live GUI preview) should
        pass a smaller value explicitly.
    rng
        Source of randomness for the bootstrap. None constructs a fresh
        seeded generator (see block_bootstrap.DEFAULT_BOOTSTRAP_SEED) --
        reproducible by default, since this is a scientific measurement
        record where repeatability matters more than true randomness.

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
        block_length, first_crossing_lag, lag1_autocorrelation = select_block_length(zeta_values)
        sigma_external = moving_block_bootstrap_sigma_external(
            zeta_values, sigma_zeta_values, block_length, n_resamples=n_resamples, rng=rng,
        )
    else:
        # No scatter/correlation information from a single shot --
        # "external" is undefined for N=1, so fall back to internal error
        # only, and skip the bootstrap entirely (nothing to resample).
        sigma_external = sigma_internal
        block_length, first_crossing_lag, lag1_autocorrelation = 0, 0, 0.0

    sigma_zeta_combined = max(sigma_internal, sigma_external)

    return CombinedSpatialDispersionResult(
        zeta_combined=float(zeta_combined),
        sigma_internal=float(sigma_internal),
        sigma_external=float(sigma_external),
        sigma_zeta_combined=float(sigma_zeta_combined),
        n_shots=n_shots,
        block_length=block_length,
        first_crossing_lag=first_crossing_lag,
        lag1_autocorrelation=lag1_autocorrelation,
    )


__all__ = ["combine_shots"]
