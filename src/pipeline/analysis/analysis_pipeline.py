"""
Single public entry point for per-shot analysis. Encodes the order as a
property of the code: centroid extraction -> optional position-
calibration conversion -> wavelength-axis lookup -> spatial-dispersion
fit(s), one per requested polynomial degree.

combine_shots() (combination.py) deliberately sits outside this function
-- combining is an orchestration concern across multiple analyze_shot()
calls (see docs/project_state.md's to-do list), not something a
single-shot entry point does itself.
"""

# Imports

from pipeline.preprocessing import ProcessedFrame

from .centroiding import CentroidEstimator, IntensityWeightedMoment, extract_centroids
from .dispersion_fitting import SpatialDispersionFitter, TotalLeastSquaresFit
from .interfaces import WavelengthAxis, PositionCalibration
from .noise_model import SensorNoiseModel, PLACEHOLDER_GAIN_E_PER_ADU, PLACEHOLDER_BACKGROUND_SIGMA
from .results import ShotAnalysisResult

# Constants

DEFAULT_DEGREES = (1,)

# Classes

# Functions

def analyze_shot(
    frame: ProcessedFrame,
    wavelength_axis: WavelengthAxis,
    estimator: CentroidEstimator | None = None,
    fitter: SpatialDispersionFitter | None = None,
    noise_model: SensorNoiseModel | None = None,
    degrees: tuple[int, ...] = DEFAULT_DEGREES,
    position_calibration: PositionCalibration | None = None,
) -> ShotAnalysisResult:

    '''
    Runs one already-preprocessed frame through the full analysis
    pipeline: centroid extraction, optional physical-position conversion,
    wavelength-axis lookup, and a spatial-dispersion fit for each
    requested polynomial degree.

    Parameters
    ----------
    frame
        An already-preprocessed frame -- see centroiding.py's module
        docstring for why windowing/background must already be handled.
    wavelength_axis
        Supplies wavelength (nm) and its uncertainty per pixel column --
        implemented, eventually, by calibration/spectral/.
    estimator
        Defaults to IntensityWeightedMoment().
    fitter
        Defaults to TotalLeastSquaresFit().
    noise_model
        Defaults to the placeholder SensorNoiseModel (gain=1, b=0) -- see
        noise_model.py for why, and docs/project_state.md's to-do list
        for what replaces it.
    degrees
        Polynomial degrees to fit, e.g. (1,) for just the linear spatial
        dispersion, or (1, 2, 3) to also check model adequacy (see
        "Result object shapes" in docs/project_state.md).
    position_calibration
        Optional pixel->physical-position conversion, applied
        immediately after centroid extraction so every downstream step
        (fitting, residuals) operates in one consistent unit. Omitted
        (the default) until calibration/spatial/ exists, per the
        "Position units" note in docs/project_state.md.

    Returns
    -------
    ShotAnalysisResult
    '''

    if estimator is None:
        estimator = IntensityWeightedMoment()
    if fitter is None:
        fitter = TotalLeastSquaresFit()
    if noise_model is None:
        noise_model = SensorNoiseModel(
            gain_e_per_adu=PLACEHOLDER_GAIN_E_PER_ADU,
            background_sigma=PLACEHOLDER_BACKGROUND_SIGMA,
        )

    centroids = extract_centroids(frame, estimator, noise_model)

    if position_calibration is not None:
        x0_for_fit, sigma_x0_for_fit = position_calibration.convert(
            centroids.x0, centroids.sigma_x0
        )
    else:
        x0_for_fit, sigma_x0_for_fit = centroids.x0, centroids.sigma_x0

    wavelength_nm = wavelength_axis.wavelength_nm(centroids.columns)
    sigma_wavelength_nm = wavelength_axis.sigma_wavelength_nm(centroids.columns)

    fits = {
        degree: fitter.fit(wavelength_nm, x0_for_fit, sigma_wavelength_nm, sigma_x0_for_fit, degree)
        for degree in degrees
    }

    return ShotAnalysisResult(frame_id=frame.frame_id, centroids=centroids, fits=fits)


__all__ = ["analyze_shot", "DEFAULT_DEGREES"]
