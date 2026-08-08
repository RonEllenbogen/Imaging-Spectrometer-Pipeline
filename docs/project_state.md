# Project State & Roadmap — Imaging Spectrometer Pipeline

Living tracker for current project status, active design decisions, and the
running to-do list. This file is actively updated as work and discussion
progress. `docs/project_handover.md` is a separate, frozen document seeded
from earlier Claude Desktop discussions covering `acquisition/` and
`preprocessing/` in depth (design rationale, bugs found and fixed,
deliberate cuts) — it is not maintained going forward. Consult it for that
historical detail; consult this file for where things currently stand and
what's next.

---

## 1. Status by package

| Package | Status |
|---|---|
| `acquisition/` | Complete, tested (synthetic + real hardware) |
| `preprocessing/` | Complete, tested (synthetic only) |
| `calibration/sensor/` | Complete, tested (synthetic only) — moved out of `preprocessing/sensor_calibration/`, see §3 |
| `calibration/spectral/`, `calibration/spatial/` | Designed, not built |
| `analysis/` | Built, tested (synthetic only) -- see §2 for design and file layout. |
| `gui/`, `main.py` | Not started |

---

## 2. `analysis/` — design and file layout

Built as:

```
src/pipeline/analysis/
├── exceptions.py          AnalysisError, InsufficientDataError
├── noise_model.py          SensorNoiseModel (+ placeholder gain/b constants)
├── centroiding.py           CentroidEstimator protocol, IntensityWeightedMoment, extract_centroids()
├── interfaces.py            WavelengthAxis, PositionCalibration protocols (calibration/ boundary contracts)
├── dispersion_fitting.py    SpatialDispersionFitter protocol, TotalLeastSquaresFit
├── combination.py           combine_shots()
├── results.py                CentroidResult, SpatialDispersionFitResult, ShotAnalysisResult, CombinedSpatialDispersionResult
└── analysis_pipeline.py      analyze_shot() -- single public entry point
```

`tests/test_analysis.py` -- single file, synthetic-only, 26 tests, all passing.

Decisions locked in below; anything decided only during implementation
(not discussed explicitly beforehand) is marked **(flagged)**.

**Centroid estimator**: intensity-weighted first moment, behind a
`CentroidEstimator` `typing.Protocol` (same structural-typing pattern as
`CameraBackend` elsewhere in the codebase), so it can be swapped for a
Gaussian/parabolic fit later without touching callers. Weighting is by raw
intensity (not intensity² or inverse-variance).

**Windowing and background**: entirely preprocessing's responsibility.
Analysis centroids over the *full* spatial axis of the already-cleaned
frame it receives, trusting that pixels zeroed by ROI masking or bad-pixel
masking contribute zero weight to the moment and therefore don't bias it.
Analysis performs no background subtraction of its own.

**Spectral binning**: 1 pixel column per bin, no grouping, for the first
version — no `bin_width` parameter yet. Rationale: sub-resolution spectral
blur (grating resolving power, slit image) convolves a locally-linear
x0(λ) with a roughly symmetric kernel, which shouldn't bias the linear ζ
fit, only add scatter. The resulting correlated noise between neighboring
columns is real, but is sidestepped by trusting the empirical shot-to-shot
scatter in fitted ζ across repeat frames (see N-shot combination below)
over a single-shot fit's internal, likely-optimistic error bar, rather than
trying to model the correlation analytically.

**Centroid uncertainty**: full 3-term Thompson-Larson-Webb (2002) formula
(photon shot noise + pixel discretization + background terms) — see
`σ_x0² = σ_PSF²/N + a²/(12N) + 8π σ_PSF⁴ b² / (a² N²)`. Chosen over
Monte Carlo/bootstrap resampling for the live/production path (MC is more
robust to violated assumptions — non-Gaussian beam profile, masking,
non-uniform background — but too slow for real-time use). MC/bootstrap is
an optional future addition (time permitting) as an *offline validation*
tool for the non-real-time spatial-dispersion measurement path: run it
once against real data to confirm TLW's error bars are trustworthy for
this instrument's actual beam profile, and derive an empirical correction
factor for the live TLW estimate if a systematic discrepancy shows up.

**Wavelength convention**: wavelength λ, in nanometres, always — never
angular frequency ω. ζ = dx0/dλ throughout (units: px/nm). (This
supersedes an earlier ω-based convention; see the addendum note at the
top of `docs/project_handover.md` for the historical design.)

**Dispersion interface**: analysis accepts an injected `WavelengthAxis`
object that exposes `wavelength_nm(pixel)` and `sigma_wavelength_nm(pixel)`
*directly* — analysis contains no physical constants, no unit-conversion
logic at all. `calibration/spectral/calibrate.py`'s pixel→λ fit (physically
expected to be close to linear, or whatever low-order polynomial
`calibration/shared/fitting.py`'s generic weighted-polynomial-fit produces)
is used as-is, with no further conversion step — since the fit variable
(λ) and the analysis variable (λ) are now the same quantity, there's no
chain-rule uncertainty propagation to do at calibration-build time the way
the old ω convention required (`dω/dλ = 2πc/λ²`).

**Fit method for ζ**: true total least squares / orthogonal distance
regression (e.g. `scipy.odr`), which accounts for uncertainty in both λ
and x0 simultaneously, behind a `SpatialDispersionFitter` `typing.Protocol`
(default: `TotalLeastSquaresFit`) so ordinary weighted least squares or the
effective-variance iterative method could be substituted later without
touching callers.

**Architectural pattern for the module**: every "which algorithm" choice
(centroid estimator, fit method, and likely future ones) is a
`typing.Protocol` with one concrete default implementation — consistent
with the `CameraBackend` pattern already established elsewhere in this
codebase. Treat this as the organizing principle for `analysis/` overall.

**Outlier handling**: none. Assumes preprocessing has already excluded bad
columns; no sigma-clipping in the fit. Revisit only if real data shows
otherwise.

**Goodness-of-fit**: no minimum-valid-column gate. Reduced χ² is always
computed (via ODR's residual variance, or equivalent) as the primary
goodness-of-fit signal, alongside residuals and normalized residuals
(`x0_observed − x0_fit(λ)`, normalized by the effective combined sigma
`√(σ_x0² + ζ²σ_λ²)`) for plotting. `SpatialDispersionFitter.fit()` takes a
`degree` parameter (1/2/3) — same total-least-squares machinery fits a
linear/quadratic/cubic polynomial in λ, each with its own reduced χ² and
residuals, letting degree be compared directly as a check on whether a
linear chirp model is adequate. ζ generalizes to a `zeta(wavelength_nm)`
method (the fitted polynomial's derivative, evaluated at any λ) rather
than a single constant once degree > 1; it collapses to the familiar
constant for the linear case.

**N per measurement**: no minimum enforced. `analysis/`'s own inverse-
variance combination function is agnostic to N — it just combines
whatever (ζ, σζ) pairs it's given. Live mode aggregates/plots every
N_default shots; batch/careful mode uses a user-specified N. Both are
orchestration-layer concerns, not `analysis/`'s.

**Jitter vs. chirp**: no dedicated discrimination mechanism, but the
combined uncertainty is protected against it directly, via the standard
internal/external error comparison (Bevington & Robinson; also the
Particle Data Group's method for averaging a set of measurements).
Averaging correctly handles the central ζ̄ value even in the presence of
zero-mean jitter, but combining each shot's own σζ alone (the internal
error) could understate the true uncertainty if jitter adds real
shot-to-shot scatter beyond what individual-shot uncertainties predict.
Combination reports `max(σ_internal, σ_external)`:
  - `σ_internal = 1/√(Σ wᵢ)`, `wᵢ = 1/σζ,ᵢ²` — the inverse-variance-
    propagated uncertainty.
  - `σ_external = √[ Σ wᵢ(ζᵢ − ζ̄)² / ((N−1)Σ wᵢ) ]` — weighted scatter of
    the per-shot ζᵢ around their combined mean; reduces to the familiar
    `std/√N` in the equal-weight case.
Reports `σ_internal` when the shots are mutually consistent, and the
larger, empirically-scattered `σ_external` when they aren't — without
double-counting by quadrature-summing both.

**Position units**: pixels by default throughout `analysis/`, since
spatial calibration doesn't exist yet. Handled the same way as the
dispersion/λ interface — an optional injectable position-calibration
object (scale factor or callable, pixel→physical position, with its own
uncertainty) converts to physical units only if supplied; until
`calibration/spatial/` exists, it never is.

**Result object shapes**:
- `CentroidResult` (per frame) — arrays over valid columns: `columns`,
  `x0` (pixels), `sigma_x0` (TLW).
- `SpatialDispersionFitResult` (per shot, per requested degree) — `degree`,
  `coefficients`, `coefficient_sigma`, `reduced_chi_squared`, `residuals`,
  `normalized_residuals`, `zeta(wavelength_nm)` method.
- `ShotAnalysisResult` (per shot) — bundles a `CentroidResult` with a
  `dict[int, SpatialDispersionFitResult]` keyed by degree, so linear/
  quadratic/cubic fits sit side by side for the same shot.
- `CombinedSpatialDispersionResult` (across N shots) — `zeta_combined`,
  `sigma_internal`, `sigma_external`, `sigma_zeta_combined` (=
  `max(sigma_internal, sigma_external)`), `n_shots`. Only the linear ζ is
  combined across shots; quadratic/cubic fits stay per-shot diagnostics
  for model-adequacy checking, not aggregated across the ensemble.

### Implementation decisions flagged for review

Made while writing the code, not discussed beforehand -- all easily
reversible, none locks in an architecture that would be costly to undo:

- TLW's pixel-size term `a` is fixed at 1, since every position in
  `centroiding.py` is already in pixel-index units (consequence of
  decision #21, not a new choice, but the resulting formula is worth a
  look).
- `IntensityWeightedMoment` does not guard against an all-zero column
  (division by zero -> NaN) -- consistent with #8 being preprocessing's
  job, but this is a slightly different edge case (exact zero vs. "low
  signal") worth confirming is the intended behavior.
- `extract_centroids()` loops over spectral columns in Python (one
  `estimator.estimate()` call per column) rather than vectorizing the
  whole frame at once, to keep `CentroidEstimator` swappable for a future
  estimator (e.g. a Gaussian fit) that couldn't be vectorized this way
  regardless. Profiled at ~13ms/frame (1920 columns) initially; two
  internal optimizations applied (positions array hoisted out of the
  per-column call instead of recomputed 1920x; one bulk contiguous
  transpose-copy of the frame up front instead of either a per-column
  np.take copy or a bare view -- a bare moveaxis view was tried first and
  was actually *slower*, since strided access penalized every
  element-wise op in estimate() more than the copy it avoided) bring this
  to ~10.4ms/frame, ~11.7ms for a full analyze_shot() call (single linear
  fit) -- an ~85 fps ceiling for single-shot processing alone. The
  per-column Python loop itself remains; whether that's ever worth
  removing depends on N_default (below) and the real laser rep rate,
  neither pinned down yet.
- `TotalLeastSquaresFit` requires `sigma_wavelength_nm`/`sigma_x0` strictly
  positive (scipy.odr weights by their inverse) and does not clip/guard
  against zero -- whatever eventually supplies a placeholder
  `WavelengthAxis` (before `calibration/spectral/` exists) must return a
  small positive `sigma_wavelength_nm`, not exactly zero.
- `analyze_shot()` applies an optional `PositionCalibration` conversion
  immediately after centroid extraction, so every downstream step (fit,
  residuals) operates in one consistent unit -- an alternative would be
  converting only the final combined zeta, leaving intermediate
  diagnostics in pixel units.
- `CentroidResult`/`SpatialDispersionFitResult`'s array fields, and
  `ShotAnalysisResult.fits`, are locked read-only in `__post_init__`
  (`.flags.writeable = False` / `MappingProxyType`) -- extending the same
  immutability discipline `FrameData`/`ProcessedFrame` already use, to
  the new result types.
- `analysis_pipeline.py`'s empty `computations.py` stub (pre-existing,
  unused) was removed as part of writing the real files.

---

## 3. To-do list

- ~~**Split `sensor_calibration` out of `preprocessing/`.**~~ **Done.**
  `build_baseline`, `build_flat_field`, `build_bad_pixel_map`, plus
  `metadata.py` (`CalibrationRecord`/`check_settings_match()`), moved into
  `calibration/sensor/`. `apply_baseline`, `apply_flat_field`,
  `apply_bad_pixel_map`, `CalibrationSet`, and `run_preprocessing` stayed in
  `preprocessing/` (now in `preprocessing/steps/`, one file per artifact
  type, mirroring `calibration/sensor/`'s layout) — a one-directional
  dependency, preprocessing depends on calibration, never the reverse.
  One consequence not anticipated when this item was written:
  `check_saturation`/`SaturationCheckResult` (previously
  `preprocessing/steps/saturation.py`) also moved to
  `calibration/sensor/saturation.py`, since `build_flat_field()` needs it
  to reject a saturated calibration source, and it depends only on
  `pipeline.acquisition` — leaving it in `preprocessing/` would have made
  `calibration/` depend on `preprocessing/`, violating the one-directional
  rule. `SettingsMismatchError`/`InvalidFlatFieldError` moved to a new
  `calibration/exceptions.py` (`CalibrationError` base) for the same
  reason; `preprocessing/` re-exports `SettingsMismatchError` for caller
  convenience. `tests/test_preprocessing.py` split: `build_*()`/
  `check_saturation()`/`CalibrationRecord`/`check_settings_match()`
  coverage moved to new `tests/test_calibration.py`; `apply_*()` and the
  end-to-end `run_preprocessing()` test stayed. Full suite still at 102
  passed, 17 skipped, now split across the two files.
- **Conversion gain measurement (e⁻/ADU)**, via a photon transfer curve
  (pixel variance vs. mean across a range of illumination levels). Needed
  so TLW's `N` term is in true photon-equivalent units rather than raw ADU.
  Not yet built. Until it exists, `analysis/` will use a placeholder
  gain = 1.
- **Background noise `b` measurement** (per-pixel variance, not just mean).
  Likely obtainable by extending `build_baseline()` (or a sibling function)
  to record variance across the same baseline frames it already consumes.
  Not yet built. Until it exists, `analysis/` will use a placeholder `b = 0`.
- **Preprocessing: spatial ROI cropping.** Currently a simple min/max row
  mask; may need more sophisticated cropping logic in the future. Low
  priority, no current evidence it's needed.
- **Preprocessing: minimum-signal / spectral-axis-cropping threshold.** No
  mechanism yet to filter spectral columns with negligible signal near the
  beam's edges before they reach `analysis/` (a near-zero-intensity column
  breaks the centroid's weighted-moment division and would bias the linear
  ζ fit if left in uncaught). Deferred back to `preprocessing/`.
- **Monte Carlo / bootstrap uncertainty validation** for `analysis/`
  (optional, time permitting) — see centroid uncertainty note in §2.
