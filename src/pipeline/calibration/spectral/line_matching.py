"""
Detects spectral line peaks in a preprocessed, averaged lamp-calibration
image and matches them to known Argon reference wavelengths, producing the
(pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm) arrays
calibrate.py's calibrate_spectral() fits.

Two steps: peak detection on the spatial-axis-collapsed 1D spectrum
(_detect_peaks), then matching detected peaks to reference_lines.py's
curated Argon window (_match_peaks_to_lines), using
grating_geometry.py's predicted relative pixel spacing as a search prior.
Absolute pixel position is NOT predictable (depends on the camera's
precise physical translation -- see grating_geometry.py's module
docstring), so the matching search does not assume where any single line
will fall; it only assumes the *pattern* of spacing between lines matches
the physics, and searches over which detected peaks could plausibly be
which reference lines to find a self-consistent identification.

The search deliberately does not assume a fixed sign for "does pixel
column increase or decrease with wavelength" -- this project has hit
sensor/optics orientation flips before, so both orderings are tried and
scored on equal footing.
"""

# Imports

import numpy as np
from scipy.signal import find_peaks, peak_widths

from ..exceptions import LineMatchingError
from . import grating_geometry
from .reference_lines import (
    ARGON_LAMP_NAME,
    ARGON_MAX_WAVELENGTH_NM,
    ARGON_MIN_WAVELENGTH_NM,
    load_reference_lines,
)

# Constants

# Matches acquisition/frame.py's CANONICAL_SHAPE convention (spatial axis, spectral axis).
_SPATIAL_AXIS = 0

# Peak-detection thresholds -- unverified starting points, same treatment
# as SNR_THRESHOLD (preprocessing/steps/signal_threshold.py) and
# SIGMA_THRESHOLD (calibration/sensor/bad_pixel_map.py) elsewhere in this
# codebase: a reasonable first guess, flagged for review once real lamp
# data exists to tune against.
PEAK_PROMINENCE_FRACTION = 0.1   # fraction of the spectrum's peak value
MIN_PEAK_SEPARATION_PX = 5

# Minimum matched lines to trust a candidate identification. 2 lines pin
# the affine pixel<->diffraction-angle relationship exactly, with no
# residual left to check it against -- requiring at least one more line
# beyond that gives the search something to actually validate a candidate
# against, rather than accepting any arbitrary 2-line guess.
MIN_MATCHED_LINES = 3

# How close (in predicted pixels) a reference line's prediction must land
# to a detected peak to count as a match, once an anchor pair has been
# chosen -- another unverified starting point (see above).
MATCH_TOLERANCE_PX = 5.0

# Placeholder for the known reference wavelength's own uncertainty. These
# are tabulated atomic-transition wavelengths, known to far better than
# 0.01nm precision in reality -- negligible next to detected-peak pixel
# uncertainty -- but shared/fitting.py's TotalLeastSquaresFit requires
# sigma_wavelength_nm strictly positive, so this can't be exactly zero.
SIGMA_WAVELENGTH_NM = 0.01

# Classes

# Functions


def match_lines(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    '''
    Detects and identifies spectral lines in a preprocessed lamp image.

    Parameters
    ----------
    image
        Preprocessed, averaged lamp-calibration image (spatial x
        spectral, matching CANONICAL_SHAPE) -- see
        calibration/spectral/workflow.py.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm) -- one
        entry per matched line, sorted by ascending pixel, ready for
        calibrate.calibrate_spectral().

    Raises
    ------
    LineMatchingError
        If fewer than MIN_MATCHED_LINES peaks are detected at all, or no
        candidate identification scores well enough against the
        predicted geometry pattern.
    '''

    spectrum = image.sum(axis=_SPATIAL_AXIS)
    detected_pixel, detected_sigma_pixel = _detect_peaks(spectrum)

    if detected_pixel.size < MIN_MATCHED_LINES:
        raise LineMatchingError(
            f"only {detected_pixel.size} peak(s) detected; need at least {MIN_MATCHED_LINES}"
        )

    reference_wavelength_nm = load_reference_lines(
        ARGON_LAMP_NAME, ARGON_MIN_WAVELENGTH_NM, ARGON_MAX_WAVELENGTH_NM
    )

    pixel, wavelength_nm, sigma_pixel = _match_peaks_to_lines(
        detected_pixel, detected_sigma_pixel, reference_wavelength_nm
    )
    sigma_wavelength_nm = np.full_like(wavelength_nm, SIGMA_WAVELENGTH_NM)

    return pixel, wavelength_nm, sigma_pixel, sigma_wavelength_nm


def _detect_peaks(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    '''
    Finds spectral line peaks in a 1D intensity-vs-pixel-column profile.
    Integer peak locations come from scipy.signal.find_peaks; each is
    then sub-pixel-refined via an intensity-weighted centroid over a
    small window around it -- the same weighted-first-moment approach
    analysis/centroiding.py's IntensityWeightedMoment uses for spatial
    centroiding, reimplemented locally rather than imported (calibration/
    must not depend on analysis/, see shared/fitting.py's module
    docstring for the same rule applied elsewhere).

    Parameters
    ----------
    spectrum
        1D intensity profile, one value per spectral pixel column.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (pixel, sigma_pixel) for every detected peak, in ascending pixel
        order. Both empty if no peaks clear the prominence threshold.
    '''

    if spectrum.max() <= 0:
        return np.array([]), np.array([])

    prominence = PEAK_PROMINENCE_FRACTION * spectrum.max()
    integer_peaks, _ = find_peaks(spectrum, prominence=prominence, distance=MIN_PEAK_SEPARATION_PX)
    if integer_peaks.size == 0:
        return np.array([]), np.array([])

    fwhm_px, *_ = peak_widths(spectrum, integer_peaks, rel_height=0.5)

    n_columns = spectrum.shape[0]
    refined_pixel = np.empty(integer_peaks.size)
    sigma_pixel = np.empty(integer_peaks.size)

    for i, (peak, fwhm) in enumerate(zip(integer_peaks, fwhm_px)):
        half_window = max(int(np.ceil(fwhm)), 2)
        lo = max(peak - half_window, 0)
        hi = min(peak + half_window + 1, n_columns)
        window_columns = np.arange(lo, hi)
        window_intensity = spectrum[lo:hi]

        refined_pixel[i] = np.sum(window_columns * window_intensity) / window_intensity.sum()
        # FWHM -> Gaussian-equivalent sigma, as a placeholder pixel-
        # position uncertainty (see PEAK_PROMINENCE_FRACTION's docstring
        # note above -- same "flagged for review" treatment).
        sigma_pixel[i] = fwhm / 2.3548

    return refined_pixel, sigma_pixel


def _match_peaks_to_lines(
    detected_pixel: np.ndarray,
    detected_sigma_pixel: np.ndarray,
    reference_wavelength_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''
    Searches for the best correspondence between detected_pixel and
    reference_wavelength_nm. For every ordered pair of detected peaks and
    every ordered pair of reference lines, solves the affine relationship
    pixel = C + D*diffraction_angle_rad(wavelength) implied by treating
    that pair as a correct identification, then checks how many of the
    remaining reference lines land within MATCH_TOLERANCE_PX of a
    (still-unclaimed) detected peak under that relationship. The
    highest-scoring candidate (most matched lines, then lowest residual)
    wins. Trying every ordered pair (not just increasing-pixel-with-
    increasing-wavelength) is what makes this robust to an unknown sensor/
    optics orientation -- see module docstring.

    Parameters
    ----------
    detected_pixel, detected_sigma_pixel
        Output of _detect_peaks().
    reference_wavelength_nm
        Candidate reference-line wavelengths (nm), any order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (pixel, wavelength_nm, sigma_pixel) for the winning
        identification, sorted by ascending pixel.

    Raises
    ------
    LineMatchingError
        If no candidate identification reaches MIN_MATCHED_LINES matched
        lines.
    '''

    n_detected = detected_pixel.size
    n_reference = reference_wavelength_nm.size
    reference_theta_m = np.array(
        [grating_geometry.diffraction_angle_rad(w) for w in reference_wavelength_nm]
    )

    best_score = None
    best_indices: tuple[list[int], list[int]] | None = None   # (detected_indices, reference_indices)

    # i<j only (not every ordered pair): swapping (i,j,a,b) -> (j,i,b,a)
    # gives the identical affine (slope, intercept), so this halves the
    # detected-pair search space with no loss of candidates -- a,b still
    # range over every ordered pair, which is what covers both possible
    # orientations (wavelength increasing or decreasing with pixel).
    for i in range(n_detected):
        for j in range(i + 1, n_detected):
            for a in range(n_reference):
                for b in range(n_reference):
                    if a == b:
                        continue
                    theta_a, theta_b = reference_theta_m[a], reference_theta_m[b]
                    if np.isclose(theta_a, theta_b):
                        continue

                    slope = (detected_pixel[i] - detected_pixel[j]) / (theta_a - theta_b)
                    intercept = detected_pixel[i] - slope * theta_a
                    predicted_pixel = intercept + slope * reference_theta_m   # (n_reference,)

                    # Bulk-vectorized nearest-detected-peak lookup for
                    # every reference line at once, then a small
                    # closest-first greedy claim over just n_reference
                    # entries (no detected peak claimed by two reference
                    # lines) -- see docstring below for why this replaces
                    # a much slower per-reference-line Python loop that
                    # each recomputed a full distance array.
                    distance_matrix = np.abs(predicted_pixel[:, None] - detected_pixel[None, :])
                    nearest_detected_idx = np.argmin(distance_matrix, axis=1)
                    nearest_distance = distance_matrix[np.arange(n_reference), nearest_detected_idx]

                    matched_detected_idx: list[int] = []
                    matched_reference_idx: list[int] = []
                    claimed_detected: set[int] = set()

                    for k in np.argsort(nearest_distance):
                        if nearest_distance[k] > MATCH_TOLERANCE_PX:
                            continue
                        d_idx = int(nearest_detected_idx[k])
                        if d_idx in claimed_detected:
                            continue
                        claimed_detected.add(d_idx)
                        matched_detected_idx.append(d_idx)
                        matched_reference_idx.append(int(k))

                    if len(matched_detected_idx) < MIN_MATCHED_LINES:
                        continue

                    matched_pixel = detected_pixel[matched_detected_idx]
                    matched_theta = reference_theta_m[matched_reference_idx]
                    residual = float(np.sum((matched_pixel - (intercept + slope * matched_theta)) ** 2))
                    score = (len(matched_detected_idx), -residual)

                    if best_score is None or score > best_score:
                        best_score = score
                        best_indices = (matched_detected_idx, matched_reference_idx)

    if best_indices is None:
        raise LineMatchingError(
            f"no candidate identification matched at least {MIN_MATCHED_LINES} reference lines"
        )

    detected_idx, reference_idx = best_indices
    pixel = detected_pixel[detected_idx]
    sigma_pixel = detected_sigma_pixel[detected_idx]
    wavelength_nm = reference_wavelength_nm[reference_idx]

    order = np.argsort(pixel)
    return pixel[order], wavelength_nm[order], sigma_pixel[order]


__all__ = ["match_lines"]
