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
| `preprocessing/` | Complete, tested (synthetic only) — now includes mandatory signal-threshold masking, see §6 |
| `calibration/sensor/` | Complete, tested (synthetic only) — moved out of `preprocessing/sensor_calibration/`, see §6 |
| `calibration/shared/`, `calibration/spatial/` | Complete, tested (synthetic only) — see §3 |
| `calibration/spectral/` | Complete except `line_matching.py`, blocked on reference-lamp selection — see §3 |
| `analysis/` | Built, tested (synthetic only) -- see §2 for design and file layout. |
| `cli/` | Calibration subcommands complete, import/argparse-tested — real hardware paths unverified — see §4 |
| `gui/` | Designed in detail, not built — see §5 |
| `main.py` | Not started |

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

*Why no bias*: convolving a linear function with any symmetric, normalized
kernel leaves its value unchanged at the kernel's own center of mass --
∫(a + bλ)K(λ − λⱼ)dλ = a + bλⱼ + b·[first moment of K about its own
center], and that moment is exactly zero when K is symmetric, regardless
of the kernel's width or shape. So as long as x0(λ) really is linear over
the wavelength span spanned by a few adjacent pixels, the blurred
measurement at column j equals the unblurred x0(λⱼ) exactly -- no
systematic shift is introduced into the individual x0 values or into the
fitted ζ. Only real curvature in x0(λ) (a nonzero second derivative at the
scale of the kernel) would produce an actual bias, proportional to the
kernel's variance -- a separate failure mode from binning itself, and one
that should surface as excess residual / degree-comparison disagreement
(the `degrees=(1, 2, 3)` diagnostic path) rather than be absorbed into an
inflated per-point uncertainty.

*Why scatter, then*: the same blur mixes each column's photon statistics
with its neighbors' -- light nominally belonging to column j's wavelength
band partly lands in columns j−1/j+1 and vice versa -- so the shot-noise
fluctuations in adjacent columns' centroids are correlated, not
independent. TLW's per-column formula has no notion of this correlation,
so a single-shot fit's own internally-computed uncertainty on ζ
under-reports the true variance. Rather than derive an analytic
correction for the correlation structure (fragile, PSF-shape-dependent),
this correlated noise is left to show up as real shot-to-shot scatter in
fitted ζ, which repeat-shot combination (N-shot combination below) is
relied on to average down -- the empirical spread across shots is trusted
over any single shot's optimistic internal error bar.

This is also why `calibration/spectral/`'s `sigma_wavelength_nm()` should
be purely calibration-fit-propagated (from the pixel→λ polynomial's
`coefficient_sigma`), not inflated with an added "wavelength spread across
a pixel" term: under the local-linearity assumption above, a pixel's
assigned wavelength is exactly its response kernel's weighted-mean
wavelength, with no leftover ambiguity to add a sigma for -- unlike TLW's
genuine spatial-quantization term, where a photon's true position within
a pixel really is unknown. The finite-bandpass-per-pixel effect is already
accounted for above, on the x0 side, via shot-to-shot scatter; folding it
into `sigma_wavelength_nm` too would double-count the same underlying
effect through two different mechanisms rather than filling a real gap.

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

## 3. `calibration/shared/`, `calibration/spatial/`, `calibration/spectral/` — design and file layout

Built as:

```
src/pipeline/calibration/
├── shared/
│   ├── io.py            save_artifact()/load_artifact() -- one or more named arrays + one record
│   ├── metadata.py        CalibrationRecord, check_settings_match() (moved from sensor/, see below)
│   ├── result.py           PolynomialFitResult
│   └── fitting.py          PolynomialFitter protocol, TotalLeastSquaresFit
├── spatial/
│   ├── calibrate.py        ScaleFactorPositionCalibration, DEFAULT_SCALE_FACTOR
│   └── io.py                ScaleFactorRecord, save_scale_factor(), load_scale_factor()
└── spectral/
    ├── calibrate.py        calibrate_spectral(), WavelengthCalibrationResult
    ├── io.py                 save_spectral_calibration(), load_spectral_calibration()
    ├── line_matching.py    match_lines() -- NotImplementedError stub, blocked (see below)
    └── workflow.py          run_spectral_calibration()
```

`tests/test_calibration.py` covers everything above except the blocked parts of `line_matching.py`
itself (its own stub-raises-`NotImplementedError` behavior is tested; the real detection/matching logic
isn't, since it doesn't exist yet).

**`CalibrationRecord` moved from `sensor/metadata.py` to `shared/metadata.py`.** Originally
sensor-only; `spectral/`'s wavelength calibration is also frame-built (from lamp frames) and reuses it
for the same reason `sensor/`'s artifacts do — tagging with the settings/timing it was captured under.
`spatial/`'s scale factor is NOT frame-built (see below) and does not use `CalibrationRecord` at all.
`sensor/__init__.py` still re-exports `CalibrationRecord`/`check_settings_match` for discoverability
(callers working only with `sensor/` artifacts don't need to know it now lives in `shared/`), but
`preprocessing/` and `spectral/` import it directly from `calibration.shared`.

**`shared/io.py` extended to persist multiple named arrays**, not just one. `save_artifact()`/
`load_artifact()` now take/return a `dict[str, np.ndarray]` instead of a single positional array —
needed because a fit result (`spectral/`'s artifact) has at least two arrays (`coefficients`,
`coefficient_sigma`), not one. `sensor/`'s three artifact types (baseline, flat field, bad-pixel map)
were updated to the new call shape (`{"baseline": ...}` etc.) with no change in behavior. Array key
names are chosen freely by each artifact type's own `save_*()`/`load_*()`; the one constraint is no key
may start with `"record__"` (reserved for record fields), enforced with a `ValueError`.

**`spatial/` is a fixed scale factor, not a per-point calibration.** Superseding the original design
sketched in `docs/project_handover.md` §5 (a `SpatialCalibrationSession` built from translation-stage
measurements) — decided out of scope for this project. Pixel→physical-position conversion at the
spectrometer's slit is the ratio of the imaging spectrometer's two relay-lens focal lengths (f1/f2),
which measures the relay optics' magnification (camera pixel pitch alone only converts to distance AT
THE DETECTOR, not at the slit). `DEFAULT_SCALE_FACTOR = 1.5` in `calibrate.py`. No uncertainty is
tracked on the scale factor: the lenses' focal lengths are known precisely, and the only real error
source (misalignment, incorrect component spacing) manifests as blur/aberration in the image, not a
quantifiable scale-factor uncertainty. `ScaleFactorPositionCalibration.convert()` implements
`analysis.interfaces.PositionCalibration` directly, scaling both `x0` and `sigma_x0` by the same
factor. The GUI can enter a manually better-measured value, which `io.py` persists (tagged `source=
"manual"` vs. `"default"` via `ScaleFactorRecord`) and reuses in future sessions.
`load_scale_factor()` is the one `load_*()` in this package that does NOT raise `FileNotFoundError` on
a missing file -- it falls back to `DEFAULT_SCALE_FACTOR`, since (unlike a baseline or flat field) the
scale factor always has a physically valid default; a fresh instrument with no saved override is the
expected common case, not an error. `spatial/session.py`, from the original design, was deleted as no
longer needed.

**`shared/fitting.py`/`shared/result.py` generalize `analysis/dispersion_fitting.py`'s total-least-
squares machinery** (`TotalLeastSquaresFit` via `scipy.odr`) to generic x/y naming (`PolynomialFitter`
protocol, `PolynomialFitResult`), for `spectral/calibrate.py` to reuse. Kept as a structurally separate
implementation from `analysis/`'s own copy, not an import of it -- `calibration/` and `analysis/` must
not depend on each other in either direction (`analysis/interfaces.py`'s `WavelengthAxis`/
`PositionCalibration` are the only boundary between them). `calibration/exceptions.py` gained its own
`InsufficientDataError(CalibrationError)` for the same reason, mirroring but not reusing
`analysis/exceptions.py`'s version.

**`spectral/calibrate.py`'s `WavelengthCalibrationResult` implements `WavelengthAxis` directly** (the
same "result IS the interface" pattern as `SpatialDispersionFitResult.zeta()`), wrapping a
`PolynomialFitResult` plus a `CalibrationRecord`. `sigma_wavelength_nm()` propagates
`coefficient_sigma` treating each coefficient's uncertainty as independent of the others (ignores their
covariance) -- an approximation, not the fit's true correlated uncertainty. `scipy.odr` exposes a full
covariance matrix (`cov_beta`) that could replace this if the approximation proves too coarse against
real lamp data; flagged for review once that data exists, same as the other placeholder/approximation
decisions in this file.

**`spectral/line_matching.py`'s `match_lines()` is genuinely blocked, not just unwritten.** It needs
two things that don't exist yet: which reference lamp will be used (and its known reference-line
wavelengths), and an approximate prior pixel→wavelength dispersion (from the grating equation /
optical design) to make peak-to-line matching tractable rather than a blind assignment problem. Both
are lamp-hardware decisions outside this codebase's scope until a lamp is chosen. The stub raises
`NotImplementedError` with an explanatory message. Everything downstream of it is still built and
tested against synthetic already-matched (pixel, wavelength_nm) data:

- `spectral/calibrate.py`'s `calibrate_spectral()` — fits matched line data, degree defaults to 1
  (first-order grating-dispersion approximation; higher degrees remain a model-adequacy diagnostic,
  same role as `analysis/`'s degree-1/2/3 comparison).
- `spectral/io.py` — persists a `WavelengthCalibrationResult` via `shared/io.py`'s extended multi-array
  support; `degree`/`reduced_chi_squared` (scalars belonging to the fit, not to `CalibrationRecord`)
  are packed as 0-d arrays alongside the fit's coefficient/residual arrays.
- `spectral/workflow.py`'s `run_spectral_calibration()` — fully wired: acquires `n_frames` lamp frames,
  preprocesses each individually via a caller-supplied `CalibrationSet` (dark/baseline subtraction,
  flat-field division, bad-pixel masking -- the full existing preprocessing pipeline, needed because a
  lamp frame is preprocessed the same as any other science frame), averages the N preprocessed images
  for better line-detection SNR, then calls `match_lines()` → `calibrate_spectral()` → save. Frames are
  preprocessed individually and averaged afterward, NOT averaged as raw frames first the way
  `build_baseline()` averages background frames -- averaging raw lamp frames before dark/flat-field
  correction would reintroduce the DSNU-contamination problem `build_flat_field()` avoids by
  dark-subtracting before normalizing. Mirrors `calibration/sensor/workflow.py`'s pattern of the caller
  supplying already-built pieces rather than this module loading anything from a hardcoded path.
  Currently raises `NotImplementedError` end-to-end (propagated from `match_lines()`) until
  `line_matching.py` is filled in; `tests/test_calibration.py` verifies the wiring itself with a
  monkeypatched `match_lines()`.

---

## 4. `cli/` — headless calibration CLI

Built as `src/pipeline/cli/calibration.py` (+ minimal `__init__.py`), a `python -m pipeline.cli.calibration
<subcommand>` entry point wiring `calibration/sensor/`'s existing workflow functions to a real
`CameraStream`. Subcommands: `baseline`, `flat-field` (interactively prompts the user between its two
physical-setup phases — block the beam, then set up uniform illumination — mirroring why
`sensor/workflow.py`'s flat-field functions are split the way they are), `bad-pixel-map` (loads a
flat-field artifact and derives the mask; no camera involved, matching `build_bad_pixel_map()` itself),
`conversion-gain` (`--exposure-min-us`/`--exposure-max-us`/`--n-levels` are required CLI arguments with
no defaults, matching the caller-supplied-not-auto-probed decision in §6), and a bonus `noise-model`
subcommand that loads a saved baseline + conversion-gain artifact and prints the `SensorNoiseModel` they'd
produce together.

Camera settings come from `configs/default.yaml` via `load_config()`, except `gain_db` — not present in
the YAML config at all, so it's a required flag on every camera-touching subcommand. Artifacts default to
a `calibration_artifacts/` directory (added to `.gitignore` — captured instrument data, not source),
overridable per-subcommand via `--path`/`--output-dir`.

Built from a self-contained written spec (no access to the design conversation, repo access only), then
independently reviewed against the real APIs it calls rather than trusting the initial report: full test
suite re-run, every subcommand's `--help` re-exercised, `load_config()`'s actual signature cross-checked.
Four minor issues found (docstring quote-style convention, a missing blank line before docstrings, a dead
`--path` argument on `noise-model` — which produces no output artifact, so the flag did nothing but was
still listed — and `resolve_artifact_path()` silently dropping a relative path's subdirectory when
combined with `--output-dir`) — all fixed in a follow-up pass and re-verified independently the same way.

No hardware-connected code path has actually been exercised against a real camera yet — everything above
was verified via imports, argparse `--help`, and the existing synthetic-only test suite.

---

## 5. `gui/` — designed in detail, not built

Nothing in code yet. Extensively designed in discussion first, the same way `calibration/spectral/` and
`spatial/` were before being built — recorded here in full so a future implementation pass (or session)
has the actual decisions, not just the fact that a GUI is planned.

**Overall structure.** On launch: choose between loading existing calibration artifacts, or creating new
ones (per-type — the user picks which one, not an all-or-nothing choice). Calibration and live view are
mutually exclusive phases, never simultaneous, which happens to line up exactly with the one-camera-
connection-at-a-time hardware constraint without needing to special-case it.

**The calibration screen is not uniform across the five types:**
- **Spatial** isn't a camera measurement at all — just a text field for a manually-measured scale-factor
  override (or accept `DEFAULT_SCALE_FACTOR`), a fundamentally different UI element from the other four.
- **Spectral** is still blocked (`line_matching.py`, §3) and must be shown as unavailable, not offered as
  if it works.
- **Bad-pixel-map has no manual "create" option at all** — it runs automatically immediately after every
  flat-field capture (`build_bad_pixel_map()` + `save_bad_pixel_map()` chained onto
  `finish_flat_field_calibration()`), since it's derived purely from the flat field with no camera
  involvement of its own.
- **Flat-field's two-phase capture needs an explicit UI pause** between "block the beam" and "set up
  uniform illumination", mirroring `cli/`'s `input()` prompts (§4).
- **Conversion-gain's exposure range is user-entered, not auto-probed** — a deliberate decision (see §6),
  not a placeholder gap; its thresholds can't be tuned without real hardware to test against.

**Main live-view interface:**
- A scatter plot: x = wavelength, y = physical position along the slit, one point per valid spectral
  column, with error bars on both axes (`sigma_x0` always; `sigma_wavelength_nm` only once real wavelength
  calibration exists — a pixel-column index has no meaningful "uncertainty" of its own, so an x-error-bar
  isn't a well-defined thing to draw against the interim pixel-column axis below).
- **x-axis fallback until spectral calibration exists: pixel column, clearly labeled as such, not
  wavelength.** Chosen specifically so the graph is buildable and testable now, with a small, contained
  change later (supplying a real `WavelengthAxis`) rather than a rebuild once the lamp is chosen.
- The fitted curve (linear/quadratic/cubic, user-selectable) drawn over the scatter — exact, no
  approximation needed, since the fit already operates directly in whatever the x-axis's units are.
- **Raw image displayed underneath as a heatmap, sharing the same axis** — deliberately NOT resampled/
  warped to true wavelength spacing (real per-pixel interpolation, every frame, at the target refresh
  rate, for a live-monitoring tool that doesn't need publication-grade precision). Instead: `pyqtgraph`'s
  `ImageItem.setRect()` stretches the image's displayed extent linearly between `wavelength_nm(first
  column)` and `wavelength_nm(last column)` — a one-time-per-frame axis transform, not a per-pixel one.
  This means the heatmap is only an approximation of true column-to-wavelength positioning whenever the
  real dispersion relationship is non-linear; the scatter points and fit curve above it remain exact
  regardless. Severity of the approximation is unknown until real spectral calibration data exists to
  check against; true per-column resampling is a documented, deferred upgrade path if it turns out to
  matter.
- **Target refresh rate: 5Hz, or the fastest achievable if not.** Computationally very achievable —
  `analyze_shot()` profiles at ~11.7ms/call (~85fps ceiling, §2), and the camera itself tops out around
  51fps — so the plotting library's redraw overhead, not the science pipeline, is the actual constraint.
  This is the deciding factor behind the framework choice below.
- **Skip-frame handling**: a live frame without enough valid columns to fit the selected degree
  (`InsufficientDataError`) is skipped, not shown as an error. After ~10 consecutive skips (an unverified
  starting constant, same treatment as `SNR_THRESHOLD`/`SIGMA_THRESHOLD` elsewhere — see §6), the display
  switches to an explicit "insufficient signal" state rather than silently freezing on the last good
  frame.
- **Side info panel**: reduced χ² and per-coefficient values + uncertainties are already fully available
  at any degree today (`SpatialDispersionFitResult.coefficients`/`coefficient_sigma`/
  `reduced_chi_squared`, §2) — no new `analysis/` work needed for these. For degree 1, ζ is a single
  number with a direct uncertainty (`coefficients[1]`/`coefficient_sigma[1]`). For degree > 1, the panel
  shows ζ evaluated at the central wavelength of the currently-valid columns — but see the uncertainty gap
  below.
- **A rolling strip chart** (ζ vs. time, last N seconds) alongside the current-frame numbers — added
  because a single number overwritten 5 times a second doesn't show the trend the live view's whole
  stated purpose depends on (watching how upstream laser adjustments affect dispersion in real time).

**The degree > 1 uncertainty gap** (real, currently-missing statistics, not a UI question). ζ(λ) is an
exactly linear function of the fit coefficients, so a *proper* uncertainty on it needs the coefficients'
full covariance matrix, not just their marginal `coefficient_sigma` (the diagonal). `scipy.odr`'s result
already exposes this (`cov_beta`, combined with `res_var` = `reduced_chi_squared`) but
`TotalLeastSquaresFit` currently discards it. The real fix belongs in `analysis/dispersion_fitting.py` +
`analysis/results.py` (`SpatialDispersionFitResult` gaining a stored covariance matrix and a
`sigma_zeta(wavelength_nm)` method) — NOT `calibration/shared/`'s structurally-separate copy of the same
machinery (§3), since the GUI's live/extended-measurement path runs through `analyze_shot()` →
`analysis/`, not calibration's. Scoped as its own follow-on task (see §6), not required before the GUI's
first version.

**Interim decision for degree > 1, until that's built**: report only the *external* (empirical,
cross-shot-scatter) uncertainty for extended measurements at degree > 1, with an explicit caveat note
displayed next to it in the GUI stating it only accounts for external uncertainty — rather than
fabricating an internal number that isn't statistically sound yet.

**Extended measurement:**
- User picks a number of frames, optionally overriding exposure/other camera parameters — which forces
  the same stop/reconfigure/restart cycle as conversion-gain calibration (§6), visibly freezing live view
  for the duration; an accepted, expected interruption, not a bug.
- Degree selection (linear/quadratic/cubic) applies here too, which runs into an existing, deliberate
  design boundary: `analysis/combination.py`'s `combine_shots()` only combines the linear ζ across shots
  by design (§2 already documents this — quadratic/cubic fits were explicitly kept per-shot-only, not
  aggregated, when `analysis/` was designed). For degree 1, `combine_shots()` covers this exactly as
  built. For the central-value combination at degree > 1: evaluating each shot's own
  `zeta(wavelength_ref)` first (a well-defined scalar regardless of degree, using each shot's already-
  computed fit) and then combining *those scalars* the same way `combine_shots()` already combines linear
  ζ is mathematically sound for the point estimate — averaging coefficients first and evaluating
  afterward gives an identical answer by linearity, as long as it's inverse-variance-weighted, not naive.
  The internal-uncertainty half of that combination hits the same covariance gap described above.
- **Static graph**: NOT a re-fit of per-column-averaged centroid positions — that's exactly the
  alternative combination methodology considered and rejected when `analysis/`'s N-shot combination was
  originally designed (§2: "not per-column combination... reusing existing single-shot code"), and would
  risk the drawn line's slope visually disagreeing with the reported ζ_combined number sitting right next
  to it. Instead: plot the full scatter (or column averages as a lighter reference layer) and draw the
  line from the actually-reported combined result (slope = ζ_combined, anchored through the data), so the
  picture and the number can never visually contradict each other.

**Framework: PyQt/PySide with `pyqtgraph`** specifically (not matplotlib, not Tkinter) — chosen for
genuine high-frequency live-plotting performance at the target refresh rate, where matplotlib's redraw
overhead becomes the practical bottleneck and Tkinter has no comparable live-plotting story.

---

## 6. To-do list

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
- **`calibration/spectral/line_matching.py`'s `match_lines()`** — blocked on choosing a reference lamp
  (and its known reference-line wavelengths) and an approximate prior pixel→wavelength dispersion from
  the grating equation / optical design. See §3 for what's already built around it.
- ~~**Conversion gain measurement (e⁻/ADU).**~~ **Done.** New
  `calibration/sensor/conversion_gain.py`: `build_conversion_gain()` takes a
  photon transfer curve sweep -- uniform illumination at *fixed brightness*,
  captured at several *exposure times* (not a variable-intensity source;
  varying exposure needs no new hardware) -- as `frames_by_exposure:
  dict[exposure_us, list[FrameData]]`, and fits `variance_ADU` against
  `mean_ADU` via `shared/fitting.py`'s `TotalLeastSquaresFit` (degree 1);
  `gain_e_per_adu = 1 / slope` is exposed as a derived property on
  `ConversionGainResult`, which wraps the full `PolynomialFitResult` (kept
  around as a diagnostic, same pattern as `spectral/`'s
  `WavelengthCalibrationResult`) rather than collapsing straight to a bare
  float. Variance at each exposure level is computed *temporally* (per-pixel
  across ≥2 repeat frames at that level, median-reduced across pixels), not
  *spatially* (across pixels within one frame) -- this sensor has real PRNU
  (see `flat_field.py`), which would inflate a spatial variance estimate and
  bias the gain low; computed temporally, PRNU cancels out with no
  dependency on the flat field, the same reasoning `background_sigma`
  already uses. A new `ConversionGainRecord` (gain_db + timestamp +
  n_illumination_levels -- no `exposure_us`, which is the swept variable
  here, not a fixed setting) tags the artifact, kept separate from
  `CalibrationRecord`. A new `InvalidConversionGainError` catches a
  saturated sweep frame or a fitted slope that isn't positive (physically
  impossible -- noise variance can't fall as signal rises).

  **Exposure range is caller-supplied, not auto-probed.** The workflow
  function (`run_conversion_gain_calibration()`) takes `exposure_min_us`/
  `exposure_max_us`/`n_levels` and linearly spaces the sweep itself, but the
  operator picks the range (informed by a quick look at the live view) --
  building a runtime auto-probing search (step exposure until near
  saturation, etc.) was considered and explicitly deferred: its thresholds
  ("close to saturation," "above the noise floor") can't be properly tuned
  without real hardware to test against. This is the one workflow function
  in `calibration/sensor/` that stops/reconfigures/starts `camera_stream`
  itself (`CameraStream` has no way to change `exposure_us` while running --
  `configure()` only runs once, inside `start()`), repeatedly, once per
  swept level, restoring the stream's original `exposure_us` (and leaving it
  running) once the sweep finishes, successfully or not. This interrupts
  live view on that stream for the sweep's duration -- unlike every other
  workflow function here, which never touches the stream's settings.
  `SyntheticBackend` doesn't scale its signal by `exposure_us` (see
  `TestFlatFieldCalibrationWorkflow`'s test comment), so the workflow
  function's own test stubs `build_conversion_gain()` to verify the
  stop/reconfigure/start/collect sequencing and exposure-restoration
  behavior, rather than exercising real PTC physics end-to-end through a
  backend that can't produce them.

  Applying the measured gain (constructing a real `SensorNoiseModel` from a
  loaded `ConversionGainResult`) is left to whatever future orchestration/
  GUI layer eventually calls `analyze_shot()` -- same as `background_sigma`,
  that layer doesn't exist yet, and `analyze_shot()`'s `noise_model`
  parameter already accepts an externally-built `SensorNoiseModel` with no
  code changes needed here.
- ~~**Background noise `b` measurement.**~~ **Done.** `build_baseline()` now
  returns a `BaselineResult` (`baseline` + `background_sigma`) instead of a
  bare array, computed from the same stacked frames the mean is built from.
  `background_sigma` is the *median* (not mean) per-pixel sample standard
  deviation (`ddof=1`) across the source frames -- median chosen for
  robustness against a handful of hot/dead pixels skewing the one scalar
  `analysis/noise_model.py`'s `SensorNoiseModel.background_sigma` expects,
  since no bad-pixel mask exists yet at the point `build_baseline()` runs
  (it's built later, from the flat field, so there's nothing to exclude
  outliers with at this stage). `build_baseline()` now requires **at least
  2 frames** (was 1) -- a sample standard deviation is undefined at n=1, and
  a silently-returned `0.0` there would be indistinguishable from a real
  "no noise measured" result rather than "not enough frames to measure it
  at all". Consequence: `build_flat_field()`'s internal reuse of
  `build_baseline()` to average its illuminated/dark frames now inherits
  this same 2-frame-per-phase minimum (was 1) -- accepted rather than
  splitting out a separate no-stats averaging helper, since nobody
  realistically flat-field-calibrates from a single frame anyway.
  `save_baseline()`/`load_baseline()` persist `background_sigma` alongside
  `baseline` in the same `.npz` artifact via `shared/io.py`'s multi-array
  support. Nothing yet constructs a `SensorNoiseModel` from a loaded
  baseline (`analyze_shot()`'s `noise_model` parameter already accepts one
  built externally, so no new glue code is needed there) -- that's left to
  whatever future orchestration/GUI layer eventually calls `analyze_shot()`,
  since it doesn't exist yet. Until gain is also measured (see above),
  `analysis/` still effectively runs with `gain = 1` even once a real `b`
  is supplied.
- **Preprocessing: spatial ROI cropping.** Currently a simple min/max row
  mask; may need more sophisticated cropping logic in the future. Low
  priority, no current evidence it's needed.
- ~~**Preprocessing: minimum-signal / spectral-axis-cropping threshold.**~~
  **Done.** New `preprocessing/steps/signal_threshold.py`: `apply_signal_threshold()`
  flags a spectral column as valid only if its total spatial-axis intensity clears
  `SNR_THRESHOLD = 2.0` (an unverified starting point, per the codebase's usual
  treatment of such thresholds) against the noise floor `sqrt(n_spatial_pixels) *
  background_sigma` — i.e. roughly "2x the noise floor counts as part of the beam,"
  the physical criterion this was designed around. Validity is carried as a new
  `ProcessedFrame.valid_columns: np.ndarray | None = None` field (`None` = every
  column valid, keeping every pre-existing `preprocessing/steps/*.py` function's
  `ProcessedFrame(...)` construction unchanged) and consumed by
  `analysis/centroiding.py`'s `extract_centroids()`, which now skips invalid columns
  entirely (not even calling `estimator.estimate()` on them) rather than guarding
  against the near-zero-`total_intensity` division centroiding.py's own docstring
  flags as deliberately left to preprocessing/ — `CentroidResult`'s `columns`/`x0`/
  `sigma_x0` arrays shrink to the valid subset rather than carrying NaN placeholders,
  since nothing downstream (`TotalLeastSquaresFit` in particular) guards against NaN.
  `CalibrationSet` gained a required `background_sigma: float` field (no default) to
  drive this, mandatory in `run_preprocessing()`'s correction order (not optional
  like ROI) — inserted after bad-pixel masking but *before* ROI masking specifically,
  since ROI zeroing rows first would make the noise-floor calculation's
  `n_spatial_pixels` an overestimate of how many real noisy pixels remain in a
  column, silently biasing the SNR calculation. `run_preprocessing()` stashes
  `valid_columns` before calling `apply_roi()` (unmodified) and reattaches it
  afterward, rather than modifying `roi.py` itself — a spatial (row) mask can't
  invalidate a spectral (column) classification, so this is safe. Full suite at 159
  passed, 17 skipped after this change. Built in a dispatched pass, reviewed and
  independently re-verified (diffs read in full, suite re-run) rather than trusting
  the initial report.
- **Build the GUI** per the design recorded in §5. Nothing started in code yet.
- **`analysis/`: proper internal uncertainty on ζ for degree > 1.** Currently only
  `coefficient_sigma` (marginal, per-coefficient) is stored; a statistically sound
  uncertainty on `zeta(wavelength_nm)` needs the fit's full coefficient covariance
  matrix (`scipy.odr`'s `cov_beta` × `res_var`, not currently kept) propagated through
  ζ's derivative formula. Surfaced while designing the GUI's live/extended-measurement
  degree > 1 display (§5); belongs in `analysis/dispersion_fitting.py` +
  `analysis/results.py`, not `calibration/shared/`'s separate copy of the same fitting
  machinery. Not required before a first GUI version — external (empirical)
  uncertainty alone is the agreed interim for degree > 1 until this is built.
- **Monte Carlo / bootstrap uncertainty validation** for `analysis/`
  (optional, time permitting) — see centroid uncertainty note in §2.
