"""
Autocorrelation-aware moving-block bootstrap for combination.py's
sigma_external. Split out of combination.py as its own module since this
is a real, independently-testable statistical procedure (autocorrelation
estimation + block-length selection + a weighted block bootstrap), not an
implementation detail of combine_shots() itself -- the same "structurally
separate, generically useful machinery" rationale calibration/shared/
fitting.py already uses for its own generalized fit machinery.

Why this exists: combine_shots()'s original sigma_external (the weighted
scatter of shots around the combined mean) implicitly assumes each shot is
an independent draw. A real recorded extended measurement
(data/measurements/extended_measurement_20260813_171209/, 200 shots)
showed that assumption is false -- the per-shot zeta series has lag-1
autocorrelation ~0.89 (a slow, mean-reverting shot-to-shot wander, not
white noise), which makes the naive scatter-based sigma_external a real
underestimate of the true uncertainty (confirmed against an independent
analytic AR(1) correction on that same dataset). A moving-block bootstrap
recovers a trustworthy sigma_external without needing to assume any
particular parametric correlation structure (AR(1) or otherwise) -- it
only needs a block length long enough to keep each resampled block's
internal correlation intact.

Block-length selection is itself data-driven, not hardcoded: the sample
ACF's first crossing below the standard ~95% white-noise significance
bound (1.96/sqrt(N)) estimates how many lags of real correlation exist,
and doubling that (a common rule of thumb for moving-block bootstrap block
length -- enough slack that a block's own boundary effects don't cut into
the correlation length being preserved) sets the block length. This
adapts correctly to whatever correlation structure a given measurement
run actually has, from i.i.d. shots (block length collapses to the
minimum) to strongly correlated ones (block length grows accordingly, up
to a capped fraction of the series so a degenerate/short run can't demand
an unreasonably large block).
"""

# Imports

import numpy as np

# Constants

# Sample ACF is only estimated out to this many lags -- a cap, not a
# per-series computation, since a lag close to N is both statistically
# unreliable (too few pairs contribute to it) and not useful for a block
# length that must stay well under N anyway (see MAX_BLOCK_LENGTH_FRACTION
# below).
MAX_LAG_CAP = 40

# The standard ~95% two-sided significance bound for a sample
# autocorrelation of a white-noise series of length N (Box & Jenkins) --
# used to find the first lag where the series' own autocorrelation is no
# longer distinguishable from what pure noise would produce.
WHITE_NOISE_Z = 1.96

# Block length is 2x the first-crossing lag (a common moving-block-
# bootstrap rule of thumb -- enough slack past the estimated correlation
# length that a block's own boundary doesn't cut into it), clipped to
# these bounds so a degenerate series (very short, or so strongly
# correlated the first crossing never happens within MAX_LAG_CAP) can't
# produce a nonsensical block length: MIN_BLOCK_LENGTH keeps at least some
# resampling entropy, MAX_BLOCK_LENGTH_FRACTION caps it well under N so
# there are still multiple distinct blocks to resample from.
MIN_BLOCK_LENGTH = 2
MAX_BLOCK_LENGTH_FRACTION = 0.2

# Default number of moving-block resamples -- precise enough for a
# persisted record (measurement_record.py's synchronous save path is not
# on any interactive time budget). Callers on a tighter budget (e.g.
# extended_measurement.py's live "Evaluate At" preview, debounced and run
# off the Qt main thread) pass a smaller value explicitly.
DEFAULT_N_RESAMPLES = 2000

# Default seed for the bootstrap's random number generator when a caller
# doesn't supply their own np.random.Generator. This is a scientific
# measurement record -- reproducible output across repeated runs on the
# same saved data matters more than true randomness, so every call that
# doesn't pass its own rng reconstructs the identical resampling sequence
# for the same input.
DEFAULT_BOOTSTRAP_SEED = 0

# Classes

# Functions

def sample_acf(series: np.ndarray, max_lag: int) -> np.ndarray:

    '''
    Sample autocorrelation of series at lags 1..max_lag (lag 0, always 1
    by definition, is not included).

    Parameters
    ----------
    series
        1-D, shot-ordered values (e.g. each shot's fitted zeta).
    max_lag
        Highest lag to compute -- caller's responsibility to keep this
        well under len(series) (see select_block_length()).

    Returns
    -------
    np.ndarray
        Length max_lag, acf[k - 1] = autocorrelation at lag k.

    Notes
    -----
    A series with zero variance (every value identical, e.g. a
    degenerate/constant synthetic fixture) has no well-defined
    autocorrelation -- returned as all-zero rather than dividing by a
    zero denominator, which reads as "no detectable correlation" and lets
    select_block_length() fall through to the minimum block length, the
    same outcome a genuinely uncorrelated series would produce.
    '''

    n = series.shape[0]
    centered = series - series.mean()
    denominator = np.sum(centered ** 2)
    if denominator == 0.0:
        return np.zeros(max_lag)

    acf = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = np.sum(centered[: n - lag] * centered[lag:]) / denominator
    return acf


def select_block_length(series: np.ndarray) -> tuple[int, int, float]:

    '''
    Picks a moving-block-bootstrap block length from series' own measured
    autocorrelation -- see module docstring for the method and rationale.

    Parameters
    ----------
    series
        1-D, shot-ordered values whose autocorrelation structure the
        block length should adapt to (e.g. each shot's fitted zeta at a
        given degree/reference wavelength).

    Returns
    -------
    tuple[int, int, float]
        (block_length, first_crossing_lag, lag1_autocorrelation).
        first_crossing_lag is the first lag (>= 1) at which |ACF| drops
        below the white-noise significance bound -- MAX_LAG_CAP (or
        len(series) // 4 if smaller) if the bound is never crossed within
        range, i.e. the series is correlated out to the full lag window
        considered. lag1_autocorrelation is reported separately as the
        single most interpretable diagnostic number, even though the
        block length itself is driven by first_crossing_lag, not lag 1
        alone.
    '''

    n = series.shape[0]
    max_lag = max(1, min(n // 4, MAX_LAG_CAP))
    acf = sample_acf(series, max_lag)

    threshold = WHITE_NOISE_Z / np.sqrt(n)
    first_crossing_lag = max_lag
    for lag in range(1, max_lag + 1):
        if abs(acf[lag - 1]) < threshold:
            first_crossing_lag = lag
            break

    max_block_length = max(MIN_BLOCK_LENGTH, int(n * MAX_BLOCK_LENGTH_FRACTION))
    block_length = int(round(2 * first_crossing_lag))
    block_length = int(np.clip(block_length, MIN_BLOCK_LENGTH, max_block_length))
    # A block length can never exceed the series itself, regardless of
    # what the clip above allowed -- only reachable for a very short
    # series (n < MIN_BLOCK_LENGTH), where nothing about a meaningful
    # block structure applies anyway.
    block_length = min(block_length, n)

    return block_length, first_crossing_lag, float(acf[0])


def moving_block_bootstrap_sigma_external(
    values: np.ndarray,
    sigma_values: np.ndarray,
    block_length: int,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> float:

    '''
    Moving-block bootstrap estimate of the external (empirical-scatter)
    uncertainty on the inverse-variance-weighted mean of values, honoring
    each shot's own sigma_values weight -- combine_shots()'s own
    weighting, not an unweighted resample -- so the bootstrap distribution
    matches what combine_shots() actually reports as the combined value.

    Resamples n_blocks contiguous blocks of length block_length (with
    replacement, block start positions drawn uniformly from every valid
    in-range start) to reconstruct a sequence of the same length as
    values, recomputes the weighted mean for each resample, and returns
    the standard deviation of that resampled-mean distribution.

    Parameters
    ----------
    values, sigma_values
        Each shot's estimate and its own uncertainty -- same shape,
        combine_shots()'s own zeta_values/sigma_zeta_values.
    block_length
        From select_block_length() -- contiguous run length resampled as
        one unit, preserving whatever correlation structure exists within
        a block.
    n_resamples
        Number of bootstrap resamples.
    rng
        Source of randomness. None constructs
        np.random.default_rng(DEFAULT_BOOTSTRAP_SEED) -- see that
        constant's docstring for why a fixed default matters here.

    Returns
    -------
    float
        The resampled weighted-mean distribution's standard deviation
        (population, ddof=0 -- the bootstrap distribution is the full
        object of interest here, not itself a sample estimating some
        larger population).
    '''

    if rng is None:
        rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)

    n = values.shape[0]
    block_length = min(max(block_length, 1), n)
    weights = 1.0 / sigma_values ** 2

    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length  # last index a block can start at, inclusive

    starts = rng.integers(0, max_start + 1, size=(n_resamples, n_blocks))
    offsets = np.arange(block_length)
    # (n_resamples, n_blocks, block_length) -> (n_resamples, n_blocks * block_length),
    # then trimmed back to exactly n columns (the last block may overshoot
    # when block_length doesn't evenly divide n).
    indices = (starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :])
    indices = indices.reshape(n_resamples, -1)[:, :n]

    resampled_values = values[indices]
    resampled_weights = weights[indices]
    resampled_means = (
        np.sum(resampled_weights * resampled_values, axis=1) / np.sum(resampled_weights, axis=1)
    )

    return float(np.std(resampled_means, ddof=0))


__all__ = [
    "sample_acf",
    "select_block_length",
    "moving_block_bootstrap_sigma_external",
    "MIN_BLOCK_LENGTH",
    "MAX_BLOCK_LENGTH_FRACTION",
    "MAX_LAG_CAP",
    "WHITE_NOISE_Z",
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
]
