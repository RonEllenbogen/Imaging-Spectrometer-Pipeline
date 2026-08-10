"""
Measures the sensor's conversion gain (e-/ADU, "gain_e_per_adu" in
analysis/noise_model.py's SensorNoiseModel) via a photon transfer curve:
uniform illumination at fixed brightness, captured at a sweep of exposure
times, giving a range of mean signal levels. For a shot-noise-limited
linear sensor, pixel variance (ADU^2) and mean signal (ADU) are related by

    variance_ADU = mean_ADU / gain + read_noise_ADU^2

so a linear fit of variance against mean gives gain = 1 / slope, with the
intercept as a bonus cross-check against build_baseline()'s
background_sigma (both estimate read noise, from different data).

Variance at each exposure level is computed TEMPORALLY (per-pixel, across
repeat frames at that level), not spatially (across pixels within one
frame) -- this sensor has real PRNU (see flat_field.py's docstring), which
would inflate a spatial variance estimate and bias the gain low. Computed
temporally, PRNU cancels out with no dependency on the flat field at all,
the same reasoning build_baseline()'s background_sigma already uses.

Applying the measured gain (constructing a real SensorNoiseModel from it)
is left to whatever future orchestration/GUI layer calls analyze_shot() --
that layer doesn't exist yet, and analyze_shot()'s noise_model parameter
already accepts an externally-built SensorNoiseModel with no code changes
needed here.
"""

# Imports

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.acquisition import FrameData
from ..exceptions import InvalidConversionGainError, SettingsMismatchError
from ..shared.fitting import PolynomialFitter, TotalLeastSquaresFit
from ..shared.io import save_artifact, load_artifact
from ..shared.metadata import CalibrationRecord, GAIN_MATCH_TOLERANCE_ABS
from ..shared.result import PolynomialFitResult
from .saturation import check_saturation

# Constants

logger = logging.getLogger(__name__)

# Minimum frames per exposure level -- a temporal sample variance (ddof=1)
# is undefined at n=1, same reasoning as build_baseline()'s minimum.
MIN_FRAMES_PER_LEVEL = 2

# Minimum exposure levels -- a degree-1 fit needs at least 2 points.
MIN_ILLUMINATION_LEVELS = 2

# Classes

@dataclass(frozen=True, slots=True)
class ConversionGainRecord:

    '''
    Tags a persisted conversion-gain artifact with the settings and sweep
    it was measured under. Deliberately does NOT include exposure_us
    (unlike calibration/shared/metadata.py's CalibrationRecord) -- exposure
    is the swept variable here, not a fixed setting to tag the whole
    artifact with. gain_db IS included: conversion gain is a property of
    the camera's amplifier gain setting, so a real change in gain_db
    invalidates a previously-measured value.

    Parameters
    ----------
    gain_db
        Camera gain setting the sweep was captured under.
    timestamp
        time.time() when this record was created.
    n_illumination_levels
        Number of distinct exposure levels the measurement was built from.
    '''

    gain_db: float
    timestamp: float
    n_illumination_levels: int

    def __post_init__(self) -> None:
        if self.n_illumination_levels < MIN_ILLUMINATION_LEVELS:
            raise ValueError(
                f"n_illumination_levels must be at least {MIN_ILLUMINATION_LEVELS}, "
                f"got {self.n_illumination_levels}"
            )

    @property
    def age_seconds(self) -> float:

        '''Seconds elapsed since this record was created.'''

        return time.time() - self.timestamp


@dataclass(frozen=True, slots=True)
class ConversionGainResult:

    '''
    build_conversion_gain()'s output: the variance-vs-mean fit, plus the
    derived conversion gain itself. Kept separate from
    ConversionGainRecord, which stays scoped to settings/provenance
    metadata, matching the same split calibration/sensor/baseline.py's
    BaselineResult/CalibrationRecord already uses.

    Parameters
    ----------
    fit
        The degree-1 PolynomialFitResult of variance_ADU (y) against
        mean_ADU (x) across the swept exposure levels.
    '''

    fit: PolynomialFitResult

    @property
    def gain_e_per_adu(self) -> float:

        '''
        The measured conversion gain, in electrons per ADU count --
        1 / slope of the variance-vs-mean fit. fit.coefficients[0] (the
        intercept, in ADU^2) is available directly on .fit for anyone who
        wants to cross-check it against build_baseline()'s
        background_sigma^2, without a dedicated property here.
        '''

        return 1.0 / self.fit.coefficients[1]


# Functions


def build_conversion_gain(
    frames_by_exposure: dict[float, list[FrameData]],
    fitter: PolynomialFitter | None = None,
) -> tuple[ConversionGainResult, ConversionGainRecord]:

    '''
    Builds a conversion-gain measurement from a photon transfer curve
    sweep: uniform illumination at fixed brightness, captured at several
    exposure times.

    Parameters
    ----------
    frames_by_exposure
        Maps each swept exposure_us value to the frames captured at it
        (at least MIN_FRAMES_PER_LEVEL each, all sharing that exposure_us
        and a common gain_db across every level). At least
        MIN_ILLUMINATION_LEVELS keys required.
    fitter
        PolynomialFitter to use for the variance-vs-mean fit. Defaults to
        TotalLeastSquaresFit.

    Returns
    -------
    tuple[ConversionGainResult, ConversionGainRecord]

    Raises
    ------
    ValueError
        If fewer than MIN_ILLUMINATION_LEVELS exposure levels are given,
        if any level has fewer than MIN_FRAMES_PER_LEVEL frames, or if any
        frame's exposure_us/gain_db doesn't match its level/the other
        levels.
    InvalidConversionGainError
        If any frame is saturated, or the fitted variance-vs-mean slope
        isn't positive (see the exception's own docstring for why that's
        physically invalid rather than just an unlucky fit).
    InsufficientDataError
        Propagated from the fitter -- should not occur given the
        MIN_ILLUMINATION_LEVELS check above, kept as a defense-in-depth
        guarantee from shared/fitting.py itself.
    '''

    if len(frames_by_exposure) < MIN_ILLUMINATION_LEVELS:
        raise ValueError(
            f"build_conversion_gain() requires at least {MIN_ILLUMINATION_LEVELS} "
            f"exposure levels, got {len(frames_by_exposure)}"
        )

    reference_gain = next(iter(frames_by_exposure.values()))[0].gain_db

    mean_adu: list[float] = []
    variance_adu: list[float] = []
    sigma_mean_adu: list[float] = []
    sigma_variance_adu: list[float] = []

    for exposure_us, frames in sorted(frames_by_exposure.items()):
        if len(frames) < MIN_FRAMES_PER_LEVEL:
            raise ValueError(
                f"exposure level {exposure_us}us has {len(frames)} frame(s); "
                f"at least {MIN_FRAMES_PER_LEVEL} are required"
            )

        for frame in frames:
            if frame.exposure_us != exposure_us:
                raise ValueError(
                    f"frame {frame.frame_id} has exposure_us={frame.exposure_us}, "
                    f"but was passed under the {exposure_us}us level"
                )
            if frame.gain_db != reference_gain:
                raise ValueError(
                    "all frames must share identical gain_db across every exposure "
                    f"level -- got a mismatch against the first level's {reference_gain}"
                )

            result = check_saturation(frame)
            if result.is_saturated:
                raise InvalidConversionGainError(
                    f"frame {frame.frame_id} at exposure {exposure_us}us is saturated "
                    f"({result.n_saturated_pixels} px at {result.peak_value}); "
                    f"reduce the exposure sweep's upper bound and recapture"
                )

        n = len(frames)
        stacked = np.stack([f.image.astype(np.float64) for f in frames], axis=0)

        level_mean = float(stacked.mean())
        level_variance = float(np.median(stacked.var(axis=0, ddof=1)))

        # Standard error of the mean and of the sample variance itself,
        # both from n (frames at this level, not n_pixels*n -- a
        # deliberately conservative choice that doesn't presume pixels are
        # independent samples of one global mean; flagged for review once
        # real sweep data exists, same as other approximations in this
        # codebase, e.g. spectral/calibrate.py's coefficient-covariance one.
        mean_adu.append(level_mean)
        variance_adu.append(level_variance)
        sigma_mean_adu.append(np.sqrt(level_variance / n))
        sigma_variance_adu.append(level_variance * np.sqrt(2.0 / (n - 1)))

    fitter = fitter if fitter is not None else TotalLeastSquaresFit()
    fit = fitter.fit(
        np.array(mean_adu), np.array(variance_adu),
        np.array(sigma_mean_adu), np.array(sigma_variance_adu),
        degree=1,
    )

    if fit.coefficients[1] <= 0:
        raise InvalidConversionGainError(
            f"fitted variance-vs-mean slope is non-positive ({fit.coefficients[1]}); "
            f"check for illumination drift, non-linearity, or insufficient dynamic "
            f"range across the exposure sweep"
        )

    record = ConversionGainRecord(
        gain_db=reference_gain,
        timestamp=time.time(),
        n_illumination_levels=len(frames_by_exposure),
    )
    return ConversionGainResult(fit=fit), record


def check_conversion_gain_matches_baseline(
    baseline_record: CalibrationRecord, conversion_gain_record: ConversionGainRecord
) -> None:

    '''
    Checks that a loaded baseline and a loaded conversion-gain artifact
    were captured at the same gain_db.

    A ConversionGainRecord has no exposure_us to compare against
    CalibrationRecord's -- exposure is the swept variable in a
    conversion-gain measurement, not a fixed setting (see
    ConversionGainRecord's own docstring), so only gain_db is checked
    here.

    Unlike build_flat_field()'s dark/illuminated check and
    run_spectral_calibration()'s N-frame check (both of which compare
    frames captured back-to-back in one session and require exact
    equality), this compares two independently-built artifacts that may
    come from different sessions entirely -- so it reuses
    check_settings_match()'s existing GAIN_MATCH_TOLERANCE_ABS tolerance
    rather than requiring exact equality.

    Parameters
    ----------
    baseline_record
        The CalibrationRecord a loaded baseline artifact was tagged with.
    conversion_gain_record
        The ConversionGainRecord a loaded conversion-gain artifact was
        tagged with.

    Returns
    -------
    None

    Raises
    ------
    SettingsMismatchError
        If gain_db differs between the two records by more than
        GAIN_MATCH_TOLERANCE_ABS.
    '''

    gain_diff_abs = abs(baseline_record.gain_db - conversion_gain_record.gain_db)
    if gain_diff_abs > GAIN_MATCH_TOLERANCE_ABS:
        raise SettingsMismatchError("gain_db", baseline_record.gain_db, conversion_gain_record.gain_db)


def save_conversion_gain(path: str | Path, result: ConversionGainResult, record: ConversionGainRecord) -> None:

    '''
    Saves a conversion-gain artifact to path, so it can be reused in a
    later session without recapturing the exposure sweep. Overwrites
    whatever was already at path -- current instrument state, not a
    history to keep.

    Parameters
    ----------
    path
        Destination file. A ".npz" suffix is added if not already
        present.
    result
        The ConversionGainResult returned by build_conversion_gain().
    record
        The ConversionGainRecord returned alongside it.

    Returns
    -------
    None
    '''

    fit = result.fit
    arrays = {
        "coefficients": fit.coefficients,
        "coefficient_sigma": fit.coefficient_sigma,
        "residuals": fit.residuals,
        "normalized_residuals": fit.normalized_residuals,
        "degree": np.array(fit.degree),
        "reduced_chi_squared": np.array(fit.reduced_chi_squared),
    }
    save_artifact(path, arrays, record)


def load_conversion_gain(path: str | Path) -> tuple[ConversionGainResult, ConversionGainRecord]:

    '''
    Loads a conversion-gain artifact previously saved via
    save_conversion_gain().

    Parameters
    ----------
    path
        The file to load. A ".npz" suffix is added if not already
        present.

    Returns
    -------
    tuple[ConversionGainResult, ConversionGainRecord]

    Raises
    ------
    FileNotFoundError
        If path doesn't exist.
    '''

    arrays, record = load_artifact(path, ConversionGainRecord)
    fit = PolynomialFitResult(
        degree=int(arrays["degree"]),
        coefficients=arrays["coefficients"],
        coefficient_sigma=arrays["coefficient_sigma"],
        reduced_chi_squared=float(arrays["reduced_chi_squared"]),
        residuals=arrays["residuals"],
        normalized_residuals=arrays["normalized_residuals"],
    )
    result = ConversionGainResult(fit=fit)
    logger.info(
        "loaded conversion gain from %s (age %.1fs, gain=%.4f e-/ADU)",
        path, record.age_seconds, result.gain_e_per_adu,
    )
    return result, record


__all__ = [
    "ConversionGainRecord", "ConversionGainResult",
    "build_conversion_gain", "check_conversion_gain_matches_baseline",
    "save_conversion_gain", "load_conversion_gain",
    "MIN_FRAMES_PER_LEVEL", "MIN_ILLUMINATION_LEVELS",
]
