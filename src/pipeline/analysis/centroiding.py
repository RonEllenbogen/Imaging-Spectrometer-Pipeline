"""
Per-column centroid extraction: the spatial (axis 0) intensity-weighted
position for each spectral (axis 1) column of an already-cleaned frame.

Operates over the FULL spatial axis and performs no background
subtraction of its own -- both windowing (ROI) and background handling
are preprocessing's job (see docs/project_state.md, decisions #3/#4). A
pixel preprocessing has zeroed (by ROI or bad-pixel masking) simply
contributes nothing to the weighted moment, exactly as if it didn't
exist. This deliberately does NOT guard against an all-zero column
(0/0 -> NaN) -- filtering negligible-signal columns is explicitly
preprocessing's responsibility, not this module's (docs/project_state.md
#8).

Bin width is one pixel column, no grouping (docs/project_state.md
#5/#6) -- sub-resolution spectral blur convolves a locally-linear x0(omega)
with a roughly symmetric kernel, which shouldn't bias the linear
spatial-dispersion fit, only add scatter that the multi-shot combination
in combination.py is trusted to average out.

Centroid uncertainty is the full 3-term Thompson-Larson-Webb (2002)
formula, working entirely in pixel-index units throughout (matching the
project's pixels-until-spatial-calibration-exists convention, see
interfaces.py) -- so the formula's pixel-size term "a" is simply 1, and
no physical pixel pitch is needed.
"""

# Imports

from typing import Protocol

import numpy as np

from pipeline.acquisition import SPATIAL_AXIS, SPECTRAL_AXIS
from pipeline.preprocessing import ProcessedFrame

from .noise_model import SensorNoiseModel
from .results import CentroidResult

# Constants

# TLW's discretization term uses the detector's pixel size, "a". Every
# position in this module (x0, sigma_PSF) is already expressed in pixel
# index units, so a = 1 pixel exactly, by construction -- not a
# measured/tunable value.
PIXEL_SIZE = 1.0

# Classes

class CentroidEstimator(Protocol):

    '''
    Structural interface every centroid estimator must match. Operates on
    one spectral column at a time -- deliberately not vectorized across
    an entire frame, since a future alternative estimator (e.g. a Gaussian
    fit) would need per-column fitting anyway and couldn't share a
    whole-frame vectorized code path with this one. extract_centroids()
    below pays a Python-level loop over columns as the cost of that
    swappability -- worth revisiting as a possible fast path if profiling
    ever shows it's a real bottleneck for live display.

    positions is threaded through as an explicit parameter (rather than
    each estimator inventing its own 0..N-1 pixel-index array internally)
    for two reasons: it removes a hidden assumption from the interface,
    and it lets extract_centroids() compute it once per frame instead of
    once per column -- identical every call, since every column spans the
    same full spatial axis (profiled: recomputing it 1920 times per frame
    was a measurable, easily avoidable cost).
    '''

    def estimate(
        self, column_intensities: np.ndarray, positions: np.ndarray,
        noise_model: SensorNoiseModel,
    ) -> tuple[float, float]:

        '''
        Parameters
        ----------
        column_intensities
            1D array, the spatial-axis intensity profile of one spectral
            column, already preprocessed (baseline-subtracted, flat-field
            divided, bad-pixel/ROI masked).
        positions
            Pixel-index position of each entry in column_intensities,
            same shape -- computed once per frame by extract_centroids()
            and passed unchanged to every column (see class docstring).
        noise_model
            Sensor noise parameters needed to turn the profile's shape
            into a position uncertainty.

        Returns
        -------
        tuple[float, float]
            (x0, sigma_x0), both in pixel-index units.
        '''

        ...


class IntensityWeightedMoment:

    '''
    Default CentroidEstimator: intensity-weighted first moment for
    position, full 3-term Thompson-Larson-Webb (2002) formula for
    uncertainty -- see the module docstring for why pixel units let the
    formula's "a" term collapse to 1.
    '''

    def estimate(
        self, column_intensities: np.ndarray, positions: np.ndarray,
        noise_model: SensorNoiseModel,
    ) -> tuple[float, float]:

        '''See CentroidEstimator.estimate for parameters/returns.'''

        total_intensity = column_intensities.sum()

        x0 = np.sum(positions * column_intensities) / total_intensity
        sigma_psf = np.sqrt(
            np.sum(column_intensities * (positions - x0) ** 2) / total_intensity
        )

        sigma_x0 = _thompson_larson_webb_sigma(sigma_psf, total_intensity, noise_model)
        return float(x0), float(sigma_x0)


# Functions

def _thompson_larson_webb_sigma(
    sigma_psf: float, total_intensity: float, noise_model: SensorNoiseModel
) -> float:

    '''
    The 3-term Thompson-Larson-Webb (2002) centroid localization
    precision formula:

        sigma_x0^2 = sigma_psf^2/N + a^2/(12N) + 8*pi*sigma_psf^4*b^2/(a^2*N^2)

    where N is total signal in photon-equivalent units, b is background
    noise in the same units, and a is the detector pixel size -- here
    fixed at PIXEL_SIZE = 1, since every position in this module is
    already in pixel-index units (see module docstring).

    Parameters
    ----------
    sigma_psf
        Standard deviation of the column's spatial intensity profile, in
        pixels.
    total_intensity
        Sum of the column's raw (ADU) intensity values.
    noise_model
        Supplies the ADU->photon-equivalent conversion gain and the
        background noise standard deviation.

    Returns
    -------
    float
        sigma_x0, in pixels.
    '''

    n_photons = total_intensity * noise_model.gain_e_per_adu
    b_photons = noise_model.background_sigma * noise_model.gain_e_per_adu

    shot_term = sigma_psf ** 2 / n_photons
    pixelation_term = PIXEL_SIZE ** 2 / (12 * n_photons)
    background_term = (
        (8 * np.pi * sigma_psf ** 4 * b_photons ** 2) / (PIXEL_SIZE ** 2 * n_photons ** 2)
    )

    return np.sqrt(shot_term + pixelation_term + background_term)


def extract_centroids(
    frame: ProcessedFrame,
    estimator: CentroidEstimator,
    noise_model: SensorNoiseModel,
) -> CentroidResult:

    '''
    Applies estimator to every spectral column of frame, one pixel column
    per bin (docs/project_state.md #5/#6), over the full spatial axis.

    Two things are hoisted out of the per-column loop rather than
    recomputed on every iteration (profiled as measurable overhead at
    1920 columns/frame): the pixel-index positions array, identical for
    every column since each spans the same full spatial axis; and one
    bulk, contiguous transpose of frame.image with the spectral axis
    first, done once. That second one is less obvious than it looks --
    a per-column np.take() copies out a small array every iteration, but
    a *view* alone (e.g. a bare np.moveaxis with no copy) is actually
    slower in practice: each column is strided in the original row-major
    layout, and every element-wise op in estimate() then pays for
    non-contiguous memory access 1920 times over. Paying one bulk copy
    up front (np.ascontiguousarray) and slicing genuinely contiguous
    views out of it afterward beat both alternatives when profiled.

    Parameters
    ----------
    frame
        An already-preprocessed frame -- windowing/background handling
        must already have happened (see module docstring).
    estimator
        The CentroidEstimator to apply to each column.
    noise_model
        Passed through to estimator unchanged for every column.

    Returns
    -------
    CentroidResult
        Arrays of length frame.image.shape[SPECTRAL_AXIS] (the number of
        spectral columns), one entry per column.
    '''

    n_columns = frame.image.shape[SPECTRAL_AXIS]
    positions = np.arange(frame.image.shape[SPATIAL_AXIS], dtype=np.float64)
    image_by_column = np.ascontiguousarray(np.moveaxis(frame.image, SPECTRAL_AXIS, 0))

    x0 = np.empty(n_columns, dtype=np.float64)
    sigma_x0 = np.empty(n_columns, dtype=np.float64)

    for column in range(n_columns):
        column_intensities = image_by_column[column]
        x0[column], sigma_x0[column] = estimator.estimate(column_intensities, positions, noise_model)

    return CentroidResult(columns=np.arange(n_columns), x0=x0, sigma_x0=sigma_x0)


__all__ = ["CentroidEstimator", "IntensityWeightedMoment", "extract_centroids", "PIXEL_SIZE"]
