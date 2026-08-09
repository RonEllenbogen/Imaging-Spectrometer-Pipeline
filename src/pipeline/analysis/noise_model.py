"""
Sensor noise parameters needed by the Thompson-Larson-Webb centroid
uncertainty formula (see centroiding.py). Both fields are currently
unverified placeholders -- see docs/project_state.md's to-do list --
pending two real calibration measurements:

  - gain_e_per_adu: a photon transfer curve (pixel variance vs. mean
    across a range of illumination levels), needed to put TLW's total
    signal term in true photon-equivalent units rather than raw ADU. Now
    measured for real by calibration/sensor/conversion_gain.py's
    build_conversion_gain() (ConversionGainResult.gain_e_per_adu) and
    persisted via save_conversion_gain()/load_conversion_gain() -- nothing
    yet constructs a SensorNoiseModel from a loaded result, same as
    background_sigma below.
  - background_sigma: the per-pixel background noise standard deviation.
    Now measured for real by calibration/sensor/baseline.py's
    build_baseline() (BaselineResult.background_sigma, the median
    per-pixel sample standard deviation across the source frames) and
    persisted alongside the baseline itself -- nothing yet constructs a
    SensorNoiseModel from a loaded baseline, since that's the future
    orchestration/GUI layer's job, not this module's.

Bundled into one object, rather than two bare module constants, so the
whole noise model can be swapped out (e.g. loaded from a future
calibration artifact) without changing every call site that needs it.
"""

# Imports

from dataclasses import dataclass

# Constants

# Placeholder gain -- treats raw ADU counts as if they were photon counts.
# A constant gain factor cancels out of the *relative* weights used
# downstream in the spatial-dispersion fit even though it doesn't get the
# absolute uncertainty scale right -- good enough to build and test the
# rest of the module against, wrong to trust for a final reported sigma.
# A real measurement now exists (calibration/sensor/conversion_gain.py's
# ConversionGainResult.gain_e_per_adu) -- this constant remains
# analyze_shot()'s default only for callers that don't supply a real
# SensorNoiseModel.
PLACEHOLDER_GAIN_E_PER_ADU = 1.0

# Placeholder background noise -- assumes preprocessing's baseline
# subtraction has already suppressed background to negligible levels.
# A real measurement now exists (calibration/sensor/baseline.py's
# BaselineResult.background_sigma) -- this constant remains analyze_shot()'s
# default only for callers that don't supply a real SensorNoiseModel.
PLACEHOLDER_BACKGROUND_SIGMA = 0.0

# Classes

@dataclass(frozen=True, slots=True)
class SensorNoiseModel:

    '''
    The two sensor-noise quantities the Thompson-Larson-Webb formula
    needs beyond what's derivable from a single column's own data.

    Parameters
    ----------
    gain_e_per_adu
        Conversion gain, in electrons per ADU count. Used to convert a
        column's summed ADU intensity into the photon-equivalent N that
        shot noise actually scales with.
    background_sigma
        Per-pixel background noise standard deviation ("b" in TLW), in
        ADU. Represents residual noise (read noise, dark current
        fluctuation) after preprocessing's baseline subtraction -- not a
        background level to subtract, just its scatter.
    '''

    gain_e_per_adu: float
    background_sigma: float

    def __post_init__(self) -> None:
        if self.gain_e_per_adu <= 0:
            raise ValueError(f"gain_e_per_adu must be positive, got {self.gain_e_per_adu}")
        if self.background_sigma < 0:
            raise ValueError(f"background_sigma must be non-negative, got {self.background_sigma}")

# Functions


__all__ = ["SensorNoiseModel", "PLACEHOLDER_GAIN_E_PER_ADU", "PLACEHOLDER_BACKGROUND_SIGMA"]
