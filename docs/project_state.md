# Project State & Roadmap — Imaging Spectrometer Pipeline

Living tracker for current project status, active design decisions, and the
running to-do list. This file is actively updated as work and discussion
progress. `docs/project_handover.md` is a separate, frozen document seeded
from earlier Claude Desktop discussions covering `acquisition/` and
`preprocessing/` in depth (design rationale, bugs found and fixed,
deliberate cuts) — it is not maintained going forward. Consult it for that
historical detail; consult this file for where things currently stand and
what's next.

## 0. Current focus (as of 2026-08-13)

**The first real calibration session against the actual instrument is done.** Baseline, flat-field,
bad-pixel-map, conversion-gain, and spectral-capture (which also built and saved geometric tilt, from
the same lamp frames — see §3/§4) all ran for real on the lab PC and are saved to
`data/calibration_artifacts_12.8.26/` (spatial stayed a manual value entry, not part of this session, per
its own design — see §3). This is real captured instrument data, not synthetic — every package's test
suite is still synthetic-only (§1), so this is the project's first evidence of how the real optics/sensor
behave, not a replacement for the test suite. The **second** `baseline` run at whatever exposure/gain the
actual spatial-chirp measurement will use (`run_preprocessing()`'s `check_settings_match()` rejects a
baseline tagged with the wrong settings, so this remains required, not optional, and belongs at a
different path than the first) has not happened yet. A published guide walks through the exact commands
and physical setup for each step; ask the user for the link if it's needed again, since artifact URLs
aren't recorded in this repo — it predates geometric tilt's CLI integration and should be checked/updated
to include it.

**Follow-up real-camera captures, once the session above was on disk.** `scripts/save_tilt_diagnostic_frames.py`
(new — see §3) was run to inspect whether the newly-built geometric-tilt calibration actually straightens
a real beam feature; its output (raw/corrected/uncorrected/diff frames, `.npy` + percentile-stretched
`.png`) is saved under `data/diagnostic/geometric_tilt_correction/`. A separate batch of 15 raw `.bmp`
frames (5 positions × 3 repeats, unprocessed — no script in this repo reads them yet) was also captured to
`data/diagnostic/spatial_calibration_12.8.26/`; purpose and next step (if any) not yet recorded here, ask
the user before assuming what it's for.

The repo is pushed to a **public** GitHub remote (`origin` = `RonEllenbogen/Imaging-Spectrometer-
Pipeline`) — the tooling-mention policy documented in `CLAUDE.md`'s "Conventions" section applies to
everything that reaches it: commit messages, PR text, code comments, docs (see that section for the
exact restricted wordlist and which two files are exempt — this file is not one of them, so don't repeat
the list here either). Before any future push, sweep unpushed commit messages *and* diff content for
restricted-list matches first (`git log --format="%B" origin/main..HEAD`, `git diff origin/main..HEAD`)
— this has caught real violations before, including auto-generated merge-commit messages that embedded a
background task's internal branch name. Local git identity may need one-off `GIT_AUTHOR_NAME`/
`GIT_AUTHOR_EMAIL`/`GIT_COMMITTER_NAME`/`GIT_COMMITTER_EMAIL` env vars on a commit if the machine's
hostname-based auto-detection fails — never `git config`, and never `git rebase -i` for history fixes
(use `git reset --soft`/`--hard` + replay, or `git filter-branch --msg-filter` restricted to the unpushed
range, for anything not yet pushed).

`pyproject.toml` now declares real dependencies (previously empty — see §4/CLAUDE.md) so `pip install -e
.` works on a fresh checkout; verify this still holds before telling anyone to rely on it, since it was
only tested via a throwaway venv, not a real second machine, until this lab session confirms it for real.

**Geometric tilt: a linear-fit amendment was built, compared against the original method, and adopted as
the default — driven by a real physical diagnosis, not the software comparison alone.**
`build_geometric_tilt_linear()` (new, alongside the pre-existing `build_geometric_tilt()` — full design
and comparison findings in §3) replaces the per-row inverse-variance-weighted shared curve with a single
weighted straight-line fit through the same per-row data, trading the ability to represent genuine
non-monotonic structure for immunity to row-to-row shot noise. Three new diagnostic scripts
(`compare_geometric_tilt_methods.py`, `plot_geometric_tilt_correction_images.py`,
`plot_geometric_tilt_correction_beam_image.py`) compared both methods against real lamp frames
(`data/diagnostic/grating_rotation/l2.{1,2,3}.bmp`) and real beam frames
(`data/diagnostic/grating_rotation/2.{1,2,3}.bmp`) — see §3 for what they found. Separately, and more
decisively: a physical diagnosis (credited to Simon) identified the actual cause of the row-dependent
tilt — dust on the slit (visible as dark streaks that should be horizontal, since a dust speck's height
doesn't depend on wavelength) revealed the grating was misaligned relative to the camera, and the lamp
lines' own tilt (each line is an image of the slit at one wavelength) revealed the slit itself is tilted
relative to the camera, with no physical adjustment available for that second one — see §3 for the full
reasoning and the physical fix performed (grating rotated, post-grating mirror re-adjusted). Both `l2.*`
and `2.*` datasets used in this session's comparisons were captured **after** that physical fix, and
geometric tilt correction (calibrated on the lamp frames, applied to the beam frames) is the intended fix
for the slit-tilt part the physical adjustment can't reach. Given the physical fix plus this session's
method-comparison findings, **the default geometric tilt method used throughout the app
(`run_spectral_calibration()`, and therefore every GUI/CLI caller of it) has been switched from
`build_geometric_tilt()` to `build_geometric_tilt_linear()`.** Spectral calibration needs to be redone
under the new default and fresh spatial-chirp measurements taken to see whether the two fixes together
(grating rotation + linear tilt correction) actually deliver a chirp-free (or accurately-measured-chirp)
beam — not yet done as of this writing.

**Spectral calibration degree comparison: linear, quadratic, and cubic pixel→wavelength_nm fits were
compared against real lamp data, and linear adopted as the default everywhere it wasn't already.**
`scripts/compare_spectral_calibration_degrees.py` (new) ran `calibrate_spectral()` at degree 1
(linear), 2 (quadratic), and 3 (cubic) against the same 9 matched lamp lines (same stacked, geometric-
tilt-corrected lamp image `plot_beam_spectrum.py` builds its own calibration from,
`data/diagnostic/grating_rotation/l2.{1,2,3}.bmp`), so any difference between the three fits is purely
the polynomial degree, not different input data. Reduced chi-squared was similar across all three
(linear 0.086, quadratic 0.045, cubic 0.050) — no degree fits obviously better or worse than another.
Decisively, though: the quadratic coefficient in the quadratic fit (`c2 = -4.74e-07 ± 1.76e-07`, ~2.7σ
from zero) and *both* the quadratic and cubic coefficients in the cubic fit (`c2 = 5.76e-07 ± 1.62e-06`,
`c3 = -3.21e-10 ± 4.93e-10`, both <1σ from zero) are statistically indistinguishable from zero — neither
higher-order fit resolves real curvature in this dataset, only noise dressed up as one. A linear
pixel→wavelength_nm model is therefore the right default, not just an arbitrary simpler choice.
`calibration/spectral/calibrate.py`'s `calibrate_spectral()` already defaulted `degree` to 1 (§3), and so
did `cli/calibration.py`'s `--degree` flag (`DEFAULT_SPECTRAL_DEGREE = 1`) — only
`gui/calibration_dialogs.py`'s `SpectralCalibrationDialog` (both its "Capture from Lamp" and "Manual
Entry" pages) was out of step, defaulting its degree selector to `DEFAULT_DEGREE = 3` on the theory that
a spectral fit needed a higher baseline degree than `live_view.py`'s spatial-dispersion fit (already 1)
— a theory this comparison does not support. `calibration_dialogs.py`'s `DEFAULT_DEGREE` is now `1`,
matching every other default in the app. Higher degrees remain selectable in both dialogs
(`DEGREE_CHOICES`/`DEGREE_LABELS`) and useful as a model-adequacy diagnostic — this finding says the
*default* should be linear, not that the quadratic/cubic options should be removed.

---

## 1. Status by package

| Package | Status |
|---|---|
| `acquisition/` | Complete, tested (synthetic + real hardware) |
| `preprocessing/` | Complete, tested (synthetic only) — now includes mandatory signal-threshold masking, see §6 |
| `calibration/sensor/` | Complete, tested (synthetic only) — moved out of `preprocessing/sensor_calibration/`, see §6 |
| `calibration/shared/`, `calibration/spatial/` | Complete, tested (synthetic only) — see §3 |
| `calibration/spectral/` | Complete, tested (synthetic only) — `line_matching.py` now built (Argon lamp), see §3 |
| `analysis/` | Built, tested (synthetic only) -- see §2 for design and file layout. |
| `cli/` | Calibration subcommands complete, import/argparse-tested — now exercised end-to-end against real hardware too, see §4 |
| `gui/` | Fully wired to real acquisition/calibration/analysis calls, tested offscreen (`SyntheticBackend`) — see §5 |
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

**Bug found and fixed, from real GUI usage — `InsufficientDataError`'s point-count threshold was one
point too low.** `TotalLeastSquaresFit.fit()` (both this package's copy and `calibration/shared/
fitting.py`'s structurally-separate one) originally required only `degree + 1` points before attempting
a fit — exactly enough to solve for the polynomial's coefficients (an exact interpolation), but leaving
**zero residual degrees of freedom**. With no excess data to estimate a variance from, `scipy.odr`
reports both the reduced chi-squared and every `coefficient_sigma` as (near-)zero rather than a real
number — confirmed empirically, and in the cubic case observed as *exactly* `0.0`, not just small.
That zero then crashed downstream: `gui/formatting.py`'s `format_value_with_uncertainty()` requires a
strictly positive sigma by design (the rounding convention has no meaning for one that isn't) and raised
`ValueError`, uncaught, out of `live_view.py`'s `_on_timer_tick()` — a real QTimer slot in a live,
unattended-capable lab tool. Reported as a recurring terminal traceback during real GUI testing. Fixed by
raising the threshold to `degree + 2` (the smallest point count with at least one residual degree of
freedom) in both `InsufficientDataError`-raising call sites, with matching updates to both
`InsufficientDataError` classes' (`analysis/exceptions.py` and `calibration/exceptions.py`) messages and
docstrings, and to `calibration/sensor/conversion_gain.py`'s own `MIN_ILLUMINATION_LEVELS` (2 → 3, since
its degree-1 photon-transfer-curve fit needs the same +1 point to stay non-degenerate — its previous
value of 2 would now always hit the corrected threshold and fail). `calibration/spectral/line_matching.py`'s
`MIN_MATCHED_LINES = 3` already happened to match the corrected threshold exactly for `calibrate_spectral()`'s
default degree (1), so no change was needed there. `live_view.py`'s `_on_timer_tick()` also gained a
last-resort `try`/`except Exception` around `_display_shot_result()` specifically (logged via
`logging`, not silently dropped) — defense-in-depth against whatever the *next* unforeseen edge case
turns out to be, fulfilling that method's own docstring, which already promised to swallow every
failure mode rather than let one escape a Qt slot but didn't fully live up to it before this fix.
`extended_measurement.py`'s `_on_run_clicked()` had the identical unguarded `analyze_shot()` call (a
button click away from the same crash, just synchronous rather than in a background timer) and gained
matching `NoSignalError`/`SettingsMismatchError`/`InsufficientDataError` handling — reported via
`QMessageBox`, aborting the run cleanly with no partial `shot_results` update, mirroring
`_enter_drifted_state()`'s existing message-box convention in the same file.

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

`calibration/spatial/` exists now (§3), but `analyze_shot()` still
defaults to pixel units by design — `position_calibration` stays an
opt-in parameter, not a new default, so every non-GUI caller
(`scripts/`, `analysis/` tests) is unaffected. The GUI is the one caller
that now opts in, but only for the "Spatial Dispersion" (ζ) display
specifically, not the whole fit: `live_view.py`/`extended_measurement.py`
each gained a `_zeta_to_mm()` helper that converts a fitted ζ (px/nm) to
physical units (mm/nm) via the same `ScaleFactorPositionCalibration` the
scatter/fit-curve y-values already went through — valid for a slope
because `.convert()` is a pure linear scale with no additive offset, so
scaling a derivative this way is exactly as correct as scaling a
position. Everything ζ feeds into internally (`_recompute_fit_and_
residuals()`'s redrawn fit line/residuals in `extended_measurement.py`,
`combine_shots()`'s inputs) still uses the raw px/nm value — only the
two on-screen labels (`live_view.py`'s side-panel field + rolling strip
chart, `extended_measurement.py`'s combined-result field) show the
converted one, each now suffixed "(mm/nm)" so the unit is explicit
on-screen rather than implicit.

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
    ├── calibrate.py           calibrate_spectral(), build_manual_spectral_calibration(), WavelengthCalibrationResult
    ├── grating_geometry.py    diffraction_angle_rad(), predicted_pixel_separation()
    ├── reference_lines.py     load_reference_lines(), Argon lamp constants
    ├── io.py                  save_spectral_calibration(), load_spectral_calibration()
    ├── line_matching.py       match_lines() -- built (see below)
    └── workflow.py            run_spectral_calibration()
```

`tests/test_calibration.py` covers everything above, including `line_matching.py`'s real peak-detection
and matching-search logic against synthetic lamp images built from `grating_geometry.py`'s own
predictions (not hand-picked numbers) -- exact match, missing lines, extra spurious peaks, and both
possible pixel/wavelength orientations are each their own test.

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
spectrometer's slit needs TWO multiplicative steps, both required: `PIXEL_PITCH_UM` (a fixed sensor
hardware spec, a2A1920-51gmBAS datasheet, 3.45 -- config-driven via `configs/default.yaml`'s
`camera.pixel_pitch_um`, the same way `canonical_shape`/`pixel_format` are, since it's a hardware fact
about whatever camera is connected, not a per-session measurement) converts a pixel-index displacement
to distance AT THE DETECTOR; `scale_factor` (the ratio of the imaging spectrometer's two relay-lens
focal lengths, f1/f2, measuring the relay optics' magnification) then converts that detector-plane
distance to the slit-plane distance where the physically meaningful spatial chirp actually lives.
`DEFAULT_SCALE_FACTOR = 1.5` in `calibrate.py`. No uncertainty is tracked on either factor: the pixel
pitch is a fixed datasheet spec, the lenses' focal lengths are known precisely, and the only real error
source (misalignment, incorrect component spacing) manifests as blur/aberration in the image, not a
quantifiable uncertainty on either quantity. `ScaleFactorPositionCalibration.convert()` implements
`analysis.interfaces.PositionCalibration` directly, scaling both `x0` and `sigma_x0` by
`PIXEL_PITCH_UM * scale_factor`, and returns the result in **microns** (matching `PIXEL_PITCH_UM`'s own
unit -- a deliberate choice so the unit is self-evident from the code rather than an implicit mm
conversion buried in the calibration math; converting to mm for a human-readable display is the
caller's/GUI's job). The GUI can enter a manually better-measured `scale_factor`, which `io.py` persists
(tagged `source="manual"` vs. `"default"` via `ScaleFactorRecord`) and reuses in future sessions --
`PIXEL_PITCH_UM` has no equivalent manual-override path, since remeasuring a camera's own pixel pitch
isn't a realistic user action the way remeasuring relay-lens magnification is.
`load_scale_factor()` is the one `load_*()` in this package that does NOT raise `FileNotFoundError` on
a missing file -- it falls back to `DEFAULT_SCALE_FACTOR`, since (unlike a baseline or flat field) the
scale factor always has a physically valid default; a fresh instrument with no saved override is the
expected common case, not an error. `spatial/session.py`, from the original design, was deleted as no
longer needed.

**Bug fixed during GUI review**: `convert()` originally applied `scale_factor` alone directly to the raw
pixel index, skipping the `PIXEL_PITCH_UM` step entirely -- present in this file's own design rationale
from the start (pixel pitch alone being insufficient was always documented) but never actually wired
into the formula. Caught by inspecting the live-view GUI's physical-position axis against known hardware
numbers (1200 spatial pixels x 3.45 um/pixel = 4140 um at the detector; x 1.5 scale factor = 6210 um /
6.21 mm at the slit) rather than by a test, since every existing test only checked *relative* scaling
behavior (e.g. "does doubling scale_factor double the output"), never an absolute real-world value.

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

**`spectral/line_matching.py`'s `match_lines()` is now built.** It was genuinely blocked (not just
unwritten) until a reference lamp was chosen and a real optical-geometry prior became available. Both
now exist: reference lamp is Argon, a lab-wide lamp reference table lives at
`data/reference/oriel_spectral_calibration_lamps.csv`, and the spectrometer's transmission-grating
equation (`m*lambda*rho = sin(theta_m) - sin(theta_i)`, `m=-1` fixed for this hardware) plus the second
relay lens's `delta_y = f*delta_theta` mapping were supplied and verified numerically against hand
derivations (θ_m(800nm)≈−12.8°, full 1920-column sensor spans ~108nm).

- **`spectral/grating_geometry.py`** (new) — `diffraction_angle_rad(wavelength_nm)` (θ_m(λ) per the
  grating equation) and `predicted_pixel_separation(wavelength_a_nm, wavelength_b_nm)` (signed pixel
  displacement between two wavelengths, reusing `calibration.spatial.PIXEL_PITCH_UM` directly -- same
  physical sensor, same pixel pitch on both axes, no relay-optics scale factor involved here since this
  measures displacement AT THE DETECTOR, not the slit plane). `grating_lines_per_mm`/
  `incidence_angle_deg`/`lens_focal_length_mm` are config-driven (`configs/default.yaml`'s new
  `spectrometer:` section), matching `pixel_pitch_um`'s existing hardware-constant pattern. Deliberately
  predicts only *relative* spacing between two wavelengths, never an absolute pixel position -- that
  depends on the camera's precise physical translation, not known here; the central wavelength's own
  θ_m cancels out of any two-wavelength difference, so imprecision in `incidence_angle_deg` ("roughly
  15°") affects matching confidence, not the final fit's accuracy (which comes entirely from
  `calibrate_spectral()`'s ODR on whatever `match_lines()` actually identifies).
- **`spectral/reference_lines.py`** (new) — `load_reference_lines(lamp, wavelength_min_nm,
  wavelength_max_nm, path=...)` reads the CSV (Python's `csv` module, no pandas dependency added).
  `ARGON_MIN_WAVELENGTH_NM = 751.46`/`ARGON_MAX_WAVELENGTH_NM = 842.46` curate the CSV's full Argon line
  list (which has no intensity data, and clusters in two disjoint UV/red bands with nothing in between)
  down to 11 lines, roughly symmetric about the 800nm central wavelength (Ti:Sapphire) -- 842.46nm is
  the largest-wavelength Ar line available, 751.46nm chosen equally spaced below 800nm.
- **`spectral/line_matching.py`**'s `match_lines()` — collapses the preprocessed/averaged 2D image to a
  1D spectrum (sum over the spatial axis), detects peaks (`scipy.signal.find_peaks`, prominence +
  minimum-separation thresholds -- unverified starting points, same treatment as
  `preprocessing/steps/signal_threshold.py`'s `SNR_THRESHOLD`/`calibration/sensor/bad_pixel_map.py`'s
  `SIGMA_THRESHOLD` elsewhere in this codebase), sub-pixel-refines each via an intensity-weighted
  centroid over a small window (the same weighted-first-moment approach
  `analysis/centroiding.py`'s `IntensityWeightedMoment` uses for spatial centroiding, reimplemented
  locally rather than imported -- `calibration/` must not depend on `analysis/`, same rule
  `shared/fitting.py`'s separate TLS-machinery copy already follows). Matching searches every ordered
  pair of detected peaks against every ordered pair of curated reference lines, solving the implied
  affine `pixel = C + D*theta_m(wavelength)` for each candidate pair and scoring by how many remaining
  reference lines land within tolerance of a (still-unclaimed) detected peak -- deliberately not
  assuming a fixed sign for "does pixel increase with wavelength," since this project has hit sensor/
  optics orientation flips before. Originally implemented as an unvectorized four-nested-Python-loop
  search that took **over two minutes** on the full 11-detected/11-reference case; rewritten to
  restrict the detected-pair loop to `i<j` (halves the search -- swapping both pairs simultaneously
  gives an identical candidate, so nothing is lost) and vectorize the inner per-reference-line nearest-
  detected-peak lookup via numpy broadcasting instead of a per-line Python loop -- now **~0.06 seconds**
  on the same input. A new `LineMatchingError(CalibrationError)` (`calibration/exceptions.py`) is raised
  when fewer than `MIN_MATCHED_LINES` (3) peaks are detected, or no candidate identification scores well
  enough. `sigma_wavelength_nm` for matched lines uses a small fixed placeholder (0.01nm) reflecting
  these tabulated atomic-line wavelengths' real (far tighter) precision -- negligible next to detected-
  peak pixel uncertainty, but strictly positive as `shared/fitting.py` requires.
- **`spectral/calibrate.py`'s `build_manual_spectral_calibration()`** (new) — a second, independent path
  into `WavelengthCalibrationResult` for a wavelength calibration the user measures entirely separately
  (e.g. via Pylon Viewer, reading off pixel positions of known lines by hand), bypassing
  `match_lines()`/`calibrate_spectral()`'s automatic fit. Takes `coefficients` **and**
  `coefficient_sigma` directly (same `wavelength_nm = c0 + c1*pixel + ...` convention as everywhere
  else) -- `coefficient_sigma` must be supplied by the caller, not defaulted or derived, since there's
  no fit residual to estimate it from and `analysis/dispersion_fitting.py`'s `TotalLeastSquaresFit`
  hard-requires `sigma_wavelength_nm` strictly positive downstream; inventing a placeholder precision
  nobody measured would silently misrepresent the calibration's real uncertainty. Note for CLI/GUI
  callers: `CalibrationRecord.source_frame_count` must be >= 1 even though a manual entry captured zero
  frames -- pass `1` as a "not applicable" convention, since the class rejects `0` outright.
- `spectral/calibrate.py`'s `calibrate_spectral()` — fits matched line data, degree defaults to 1
  (first-order grating-dispersion approximation; higher degrees remain a model-adequacy diagnostic,
  same role as `analysis/`'s degree-1/2/3 comparison).
- `spectral/io.py` — persists a `WavelengthCalibrationResult` via `shared/io.py`'s extended multi-array
  support; `degree`/`reduced_chi_squared` (scalars belonging to the fit, not to `CalibrationRecord`)
  are packed as 0-d arrays alongside the fit's coefficient/residual arrays.
- **`spectral/geometric_tilt.py`** (new) — `build_geometric_tilt(frames, gain_e_per_adu=..., background_sigma=..., fitter=...)`
  measures the spectrometer's row-dependent geometric tilt (a shared, non-monotonic, wavelength-
  independent column shift found across every detected lamp line -- productionizes
  `scripts/measure_spectrometer_tilt.py`'s exploratory analysis). Auto-detects usable lines via
  `scipy.signal.find_peaks` on the column-summed spectrum (no wavelength/reference-line matching needed
  -- unlike `line_matching.py`, this only needs positional fiducials), fits each one's row-vs-centroid
  trace, and returns a `GeometricTiltResult` (dominant shared `row_shift` array + a sparser per-column
  `residual_slope_columns`/`residual_slope_values` term, consumed via `column_shift()`). Raises
  `LineMatchingError` (reused from `line_matching.py`'s error, not duplicated) if fewer than
  `MIN_LINES_REQUIRED` (3) lines are detected. Duplicates analysis/centroiding.py's intensity-weighted-
  centroid-plus-Thompson-Larson-Webb formula locally rather than importing it (`calibration/` must not
  depend on `analysis/`, same rule `shared/fitting.py` follows) -- accepts plain
  `gain_e_per_adu`/`background_sigma` floats rather than a `SensorNoiseModel` for the same reason.
  The shared `row_shift` curve is an inverse-variance-weighted mean across lines at each row (each
  line's own per-row Thompson-Larson-Webb `sigma_x0`, already computed for the residual-slope fits,
  now also weights this step) -- previously a plain `np.nanmean()`, which gave a dim/noisy line exactly
  as much say as a bright, tightly-centroided one. `gain_e_per_adu`/`background_sigma` were, until a
  real lab session existed to supply them, always this module's own placeholders (`gain=1.0`,
  `background_sigma=0.0`) -- `run_spectral_calibration()` now threads real values through
  unconditionally for `background_sigma` (a required `CalibrationSet` field, always available) and via
  a new `gain_e_per_adu` parameter for the caller-supplied conversion gain; both real GUI/CLI callers
  (`SpectralCalibrationDialog`, `spectral-capture`) now load a `conversion_gain.npz` alongside
  baseline/flat-field/bad-pixel-map (a new required artifact for spectral capture, where it was
  previously not needed) and pass its `gain_e_per_adu` through. Found empirically not to be a small
  effect: reproducibility between two independent lamp captures (diffing `row_shift`) showed up to 46px
  of disagreement against the curve's own ~52px total range -- prompted by a real lab-PC session where
  live view's measured "spatial dispersion" dropped by roughly half once tilt correction was applied at
  all, but still didn't reproduce cleanly between two independently-measured tilt calibrations.
  `save_geometric_tilt()`/`load_geometric_tilt()` persist it via `shared/io.py`. Applied by
  `preprocessing/steps/geometric_tilt.py`'s `apply_geometric_tilt_correction()` (a resample via
  `scipy.ndimage.map_coordinates`, not just an intensity correction -- the one preprocessing step that
  warps rather than rescales/masks pixel values, with the noise-correlation and bad-pixel-interpolation
  caveats that implies, documented in its own module docstring), wired into `CalibrationSet` as an
  optional `geometric_tilt: GeometricTiltResult | None = None` field (`None` = skipped, so every
  pre-existing `CalibrationSet` construction site keeps working unchanged) and applied in
  `run_preprocessing()`'s correction order right after bad-pixel masking, before signal-threshold
  masking.
- **`scripts/save_tilt_diagnostic_frames.py`** (new) — a real-camera diagnostic tool, written to answer
  the reproducibility question above directly against a real beam rather than by comparing two tilt
  calibrations to each other: grabs real frames and preprocesses each one twice, once through the full
  pipeline with `apply_geometric_tilt_correction()` applied (`CalibrationSet.geometric_tilt` populated)
  and once with it skipped (`geometric_tilt=None`, everything else identical), saving raw/corrected/
  uncorrected/diff images (`.npy` float64 + a percentile-stretched `.png` for quick viewing) plus a
  `summary.txt`. Loads baseline/flat-field/bad-pixel-map/geometric-tilt the same way
  `gui/calibration_screen.py`'s `WelcomePage` does (same `load_*()` calls, same default filenames), so it
  exercises the identical on-disk artifacts a real session would use, not a re-derived approximation. It
  doesn't test whether the correction *runs* (already confirmed by `run_preprocessing()`'s own code path)
  — it's for inspecting whether the measured `row_shift` curve is accurate enough to actually straighten a
  real beam feature. First real output is saved to `data/diagnostic/geometric_tilt_correction/` (§0);
  those specific findings were never folded back into this section, but the general question they were
  asking (is the correction actually straightening a real beam feature?) is now answered by the newer
  work below, against a different real dataset.
- **Physical root cause identified (credited to Simon) and a physical fix applied, ahead of/alongside the
  software work below.** The lamp/beam images captured for geometric-tilt work show dark streaks and
  bright lines; both are diagnostic of the spectrometer's physical alignment. Dark streaks are dust specks
  on the slit — physically fixed features of the slit itself, so their image should sit at the same row
  regardless of wavelength (i.e. perfectly horizontal); a streak that ISN'T horizontal means the grating
  is misaligned relative to the camera. Separately, each bright lamp line is an image of the slit itself
  at one wavelength (an Argon emission line illuminates the whole slit, imaged through the spectrometer at
  that line's dispersed position) — a tilted lamp line means the slit is physically tilted relative to the
  camera, not a grating problem. The grating angle can be adjusted; the slit's own tilt relative to the
  camera cannot. Fix applied: the grating was rotated and the post-grating mirror re-adjusted at the same
  time, iterating until the dark (dust) streaks were as close to horizontal as achievable. Effect
  confirmed visually: the beam image's bright vertical lines, previously tilted with a slight *positive*
  gradient, are now tilted with a slight *negative* gradient — matching the slight negative gradient still
  present in the lamp images (the slit-tilt component the grating rotation can't reach). `data/diagnostic/
  grating_rotation/l2.{1,2,3}.bmp` (lamp) and `2.{1,2,3}.bmp` (beam) are both captured **after** this
  physical fix — every geometric-tilt comparison in this section and the one below uses this
  already-grating-rotated data, not the original mis-aligned setup.
- **`spectral/geometric_tilt.py`'s `build_geometric_tilt_linear()`** (new) — an alternative to
  `build_geometric_tilt()` for the one remaining distortion the grating rotation above can't fix (the
  slit's own tilt relative to the camera): instead of `build_geometric_tilt()`'s per-row inverse-
  variance-weighted shared curve (interpolated across any row no line covers), this fits one straight line
  through the same per-row combined data — weighted by each row's combined inverse-variance, using only
  rows with real line coverage (deliberately excluding `build_geometric_tilt()`'s own interpolated-fill
  rows, so no already-interpolated value feeds back into the fit), then re-anchored so `row_shift[reference_row]
  == 0` exactly (the raw fitted line only satisfies that approximately; every line's displacement was
  already anchored to `reference_row` upstream, so the fit is shifted by a constant to respect the same
  anchor rather than let finite-sample noise offset it). Shares all line-detection/per-row-centroiding/
  per-line-anchoring logic with `build_geometric_tilt()` via a new private `_measure_line_displacements()`
  helper both call, rather than duplicating it. Trades the ability to represent genuine non-monotonic
  structure (this module's own docstring documents the tilt curve as non-monotonic and jump-containing,
  reproducible across shots) for immunity to row-to-row shot noise — an explicit bet, not a strict
  improvement, on whether a given dataset's row_shift curve is closer to "real structure" or "noise" at
  any given row.

  **Compared against real data via three new scripts** (`scripts/compare_geometric_tilt_methods.py`,
  `scripts/plot_geometric_tilt_correction_images.py`, `scripts/plot_geometric_tilt_correction_beam_image.py`),
  against the lamp frames `data/diagnostic/grating_rotation/l2.{1,2,3}.bmp` (calibration source for both
  methods throughout) and the beam frames `data/diagnostic/grating_rotation/2.{1,2,3}.bmp` (correction
  target, using the lamp-built calibration — not self-calibrated). Findings:
  - The measured `row_shift` curve (7 lines detected, columns 180-1773) spans roughly -34 to +41px. In the
    frame's core (rows ~150-1100) the two methods agree closely, except for two sharp, narrow features
    around row ~480 and row ~700-750 (~5px deviation each) that only `build_geometric_tilt()` can
    represent — a straight line can't reproduce a discontinuity by construction.
  - At the frame's edges (rows 0-150, 1100-1200), `build_geometric_tilt()`'s curve is genuinely noisy
    (std ≈4.4-4.7px, excursions up to ±15-20px) while `build_geometric_tilt_linear()` stays smooth there
    by construction. Applying both corrections to each of the 3 individual lamp shots (not just their
    stack) confirmed this isn't a stacking artifact: `build_geometric_tilt()`'s correction is visibly
    ragged/broken in the top ~150 rows of every one of the 3 shots individually, while
    `build_geometric_tilt_linear()` stays clean through the same rows in all 3.
  - Applying both corrections to the real beam image (rather than the lamp image the calibration was
    built from) confirmed the same picture at the pixel level: a difference image between each correction
    and the uncorrected beam shows the expected dipole pattern (sign-flipping at `reference_row`) with
    magnitude up to ~20 intensity units; the *difference between the two methods'* corrections is smaller
    (~1/4 that magnitude) and concentrated almost entirely in narrow row bands matching the same two
    row_shift discrepancy locations above — confirming the two views (row_shift curve, pixel-level
    correction) are self-consistent, though this is expected math, not independent new evidence that the
    jumps are physically real.
  - Two open questions, still unresolved by either method: (1) the lamp image's brightest line (~col 1241)
    splits into two closely-spaced tracks once straightened, identically across all 3 individual shots —
    most likely two closely-spaced Argon lines that the peak-detection step (`find_peaks` on the
    column-summed spectrum) merges into one peak because the pre-correction tilt smears them together
    across the full row range; neither correction method resolves this, since it's upstream of where they
    diverge. (2) whether the edge-row noise above is genuine shot noise (which `build_geometric_tilt_linear()`
    correctly averages down) or a `MAX_WINDOW_HALF_WIDTH` (40px) window-clipping bias (given the ~75px
    total measured shift is uncomfortably close to that limit) — if the latter, "smoother at the edges"
    isn't the same as "more accurate there" for either method. Neither has been investigated further yet.

  **Given the above, plus the physical fix documented above it, the default geometric tilt method used
  throughout the app has been switched from `build_geometric_tilt()` to `build_geometric_tilt_linear()`**
  — `run_spectral_calibration()` (below) now builds the tilt calibration via the linear method, so every
  real caller (`SpectralCalibrationDialog`'s "Capture from Lamp" mode, the `spectral-capture` CLI
  subcommand) picks it up automatically. `build_geometric_tilt()` (pointwise) remains available and fully
  tested, just no longer the default any production code path calls.
- `spectral/workflow.py`'s `run_spectral_calibration()` — fully wired: acquires `n_frames` lamp frames,
  builds a `GeometricTiltResult` from those same raw frames (before any other preprocessing --
  `build_geometric_tilt_linear()` does its own per-line background handling, inherited from the shared
  `_measure_line_displacements()` helper it and `build_geometric_tilt()` both call) and saves it to a
  caller-supplied `geometric_tilt_path`, then preprocesses each frame individually via a caller-supplied
  `CalibrationSet` (dark/baseline subtraction, flat-field division, bad-pixel masking, and now the tilt
  correction just built -- the full existing preprocessing pipeline, needed because a lamp frame is
  preprocessed the same as any other science frame), averages the N preprocessed images for better
  line-detection SNR, then calls `match_lines()` → `calibrate_spectral()` → save. Geometric tilt is built
  here rather than requiring a separate caller-driven capture, since (unlike baseline/flat-field/bad-
  pixel-map) it needs nothing but a lamp exposure -- no reason to make the caller run a second physical
  setup for it; tilt-correcting the averaged image before `match_lines()` also matters for line-detection
  quality, since an un-corrected image's lines are row-tilt-smeared (broadened, less resolved) when
  collapsed to a 1D spectrum. Frames are preprocessed individually and averaged afterward, NOT averaged as
  raw frames first the way `build_baseline()` averages background frames -- averaging raw lamp frames
  before dark/flat-field correction would reintroduce the DSNU-contamination problem `build_flat_field()`
  avoids by dark-subtracting before normalizing. Mirrors `calibration/sensor/workflow.py`'s pattern of the
  caller supplying already-built pieces rather than this module loading anything from a hardcoded path.
  Returns `tuple[WavelengthCalibrationResult, GeometricTiltResult]`. Runs end-to-end against a real lamp
  frame; `tests/test_calibration.py` covers both the full wiring (monkeypatched `match_lines()` and
  `build_geometric_tilt_linear()`) and a real-detection case (a `SyntheticBackend` frame's smooth Gaussian
  beam profile correctly has no usable lines for either `build_geometric_tilt_linear()` or `match_lines()`,
  raising `LineMatchingError`). This function called `build_geometric_tilt()` (the pointwise method) until
  the default-method switch documented above it.

---

## 4. `cli/` — headless calibration CLI

Built as `src/pipeline/cli/calibration.py` (+ minimal `__init__.py`), a `python -m pipeline.cli.calibration
<subcommand>` entry point wiring `calibration/sensor/`'s existing workflow functions to a real
`CameraStream`. Subcommands: `baseline`, `flat-field` (interactively prompts the user between its two
physical-setup phases — block the beam, then set up uniform illumination — mirroring why
`sensor/workflow.py`'s flat-field functions are split the way they are), `bad-pixel-map` (loads a
flat-field artifact and derives the mask; no camera involved, matching `build_bad_pixel_map()` itself),
`spatial` (sets or reports the scale factor via `calibration/spatial/`'s `save_scale_factor()`/
`load_scale_factor()` — no camera involved either; added later than the other four, see below),
`conversion-gain` (`--exposure-min-us`/`--exposure-max-us`/`--n-levels` are required CLI arguments with
no defaults, matching the caller-supplied-not-auto-probed decision in §6), `spectral-capture`/
`spectral-manual` (two flat subcommands, not one nested one — this file has no nested subparsers
anywhere, so two top-level commands fit the existing style better; `spectral-capture` wires
`run_spectral_calibration()` to a real `CameraStream`, loading baseline/flat-field/bad-pixel-map inputs
the same optional-default way `noise-model` resolves its own inputs, plus a `--geometric-tilt-path`
flag (default `calibration_artifacts/geometric_tilt.npz`, same `--output-dir`-relative-path rules as
every other artifact flag) for where the geometric tilt calibration `run_spectral_calibration()` now
builds automatically gets saved -- there is no separate `geometric-tilt` subcommand, since it needs
nothing but the same lamp frames `spectral-capture` already captures, see §3;
`spectral-manual` takes
`--coefficients`/`--coefficient-sigma` as `nargs="+"` lists and calls
`build_manual_spectral_calibration()` -- no camera involved, so it's the one subcommand in this file
actually exercised end-to-end by its own tests rather than only argument-parsing-tested), and a bonus
`noise-model` subcommand that loads a saved baseline + conversion-gain artifact and prints the
`SensorNoiseModel` they'd produce together.

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

**Exposure/gain audit, ahead of real lab use.** A later audit (prompted by a real calibration session
being imminent) found two real gaps, both fixed: no CLI coverage at all for the spatial scale factor
(added as the `spatial` subcommand above), and no way to override `exposure_us` per invocation — every
camera-touching subcommand silently used `configs/default.yaml`'s fixed `camera.exposure_time`. Fixed by
adding a mutually-exclusive `--auto-exposure`/`--exposure-us` pair to `baseline`/`flat-field`
(`conversion-gain` deliberately excluded — it already sweeps its own exposure range, a fixed override
wouldn't mean anything there); `build_camera_stream()` gained matching `exposure_us`/`auto_exposure`
parameters, both optional, defaulting to the pre-existing config-driven behavior when omitted (no change
for existing usage that doesn't pass either flag). Everything else — every call from `cli/calibration.py`
into `calibration/sensor/workflow.py` — was checked param-for-param against that module's real signatures
and found to match exactly; no other drift found.

**`--auto-exposure` required fixing real acquisition-layer gaps first, not just adding a flag.**
`PylonBackend._converge_auto_exposure()` (genuine pypylon `ExposureAuto="Once"` GenICam calls) already
existed but was fully disconnected — nothing ever constructed `PylonBackend(auto_exposure=True)`, and
`configs/default.yaml`'s own `auto_exposure` field was read only by a standalone bring-up script
(`scripts/camera_testing.py`), never by the pipeline package itself. Fixed: `CameraStream` gained
`auto_exposure`/`auto_timeout_ms` constructor parameters, threaded to `PylonBackend`'s construction.
Separately, and just as important: nothing read back the real converged exposure value after
auto-exposure finished, which would have left every captured frame's `FrameData.exposure_us` (and any
`CalibrationRecord` built from it) silently wrong whenever auto-exposure was actually used.
`CameraBackend.configure()`'s contract changed to *return* the applied exposure (µs) —
`PylonBackend` reads back `ExposureTime.GetValue()` after convergence; the manual path and
`SyntheticBackend` both just echo back what they were given — and `CameraStream.start()` now adopts that
return value into `self.exposure_us` before the grab thread starts, so every frame afterward carries the
true applied exposure regardless of which path configured it.

`build_camera_stream()`'s `--gain-db` stays a required CLI flag in both auto and manual exposure modes —
deliberately no silent default, unlike the GUI's own "Auto" preset (§5), which is a convenience default
for a different, less explicit context.

**Bug found and fixed: `CameraStream.start()` could hand back a frame grabbed under the OLD settings
after a stop/reconfigure/restart cycle.** Surfaced as exposure-sweep leakage — `run_conversion_gain_calibration()`'s
per-level stop/reconfigure/start loop (this section, above) calls `collect_n_frames()` immediately after
each `start()`, and `get_latest_frame()` was returning whatever `FrameData` the background thread had
grabbed *just before* the preceding `stop()` took effect, still sitting in `_latest_frame` from the
previous exposure level — silently mislabeling a frame as belonging to the new settings when it was
actually captured under the old ones. `ExtendedMeasurementScreen`'s own reconfigure-then-run cycle (§5)
and this diagnostic script's `--auto-exposure` capture (§0) go through the identical `stop()`/`start()`
sequence, so they were equally exposed even though this was found via the conversion-gain sweep
specifically. Fixed in `CameraStream.start()`: `_latest_frame` is now explicitly cleared (under
`_frame_lock`) as part of the same "clear state left over from a previous start()/stop() cycle" step that
already reset `_stop_event`/`_last_error` — so every caller's first `get_latest_frame()`/`collect_n_frames()`
poll after `start()` now genuinely blocks until a fresh frame under the new settings has been grabbed,
rather than risking a stale one leaking through.

`tests/test_cli.py` (new) covers argument parsing for every subcommand (including the mutual-exclusion
rule on `baseline`/`flat-field`'s new flags, and confirming `conversion-gain` does NOT accept
`--exposure-us`) — the CLI previously had no dedicated test file at all; the "argparse-tested" status in
this section's own history was verified manually via `--help`, not via committed tests.

**Every hardware-connected code path has now been exercised against the real camera, not just imports and
argparse `--help`.** `baseline`/`flat-field`/`bad-pixel-map`/`conversion-gain`/`spectral-capture` all ran
for real on the lab PC and produced the artifacts saved to `data/calibration_artifacts_12.8.26/` (§0) —
the first time any of this CLI's real-hardware paths (as opposed to `SyntheticBackend`) had run at all.
No CLI-layer bugs were found in the process; the two real issues that session did surface —
`CameraStream.start()` leaking a stale frame across an exposure-sweep reconfigure, and
`check_settings_match()` rejecting an auto-exposure-converged lamp frame against a fixed-exposure
baseline — were both acquisition/GUI-layer, not CLI-layer, and are documented where they were fixed (this
section, above, and §5's `SpectralCalibrationDialog` note respectively).

`pyproject.toml` (previously empty) now declares `numpy`/`scipy`/`pyyaml`/`pypylon` as real install
dependencies, so `pip install -e .` alone is enough to run this CLI on a fresh checkout — no more
installing each dependency by hand first. `PySide6`/`pyqtgraph` (gui extra) and `pytest`/`pytest-qt` (dev
extra) are optional, so a CLI-only checkout doesn't need to install Qt at all.

---

## 5. `gui/` — fully wired to real backend calls

Extensively designed in discussion first (recorded in full below, still the reference for *why* things
are shaped the way they are), built as a tested Phase-1 visual skeleton, then — in a dedicated
multi-agent wiring pass (Team-Lead-orchestrated: one agent per screen, each in its own worktree, merged
sequentially with a full-suite check after each merge, plus a Team-Lead-only foundation stage before and
integration/review/end-to-end-smoke-test stage after) — wired end-to-end to the real
`acquisition`/`calibration`/`preprocessing`/`analysis` backend. Every screen now calls the real
`build_*()`/`run_preprocessing()`/`analyze_shot()`/`combine_shots()` functions; nothing left in
`src/pipeline/gui/` generates random placeholder data as its primary behavior (the one deliberate
exception — `LiveViewWidget._populate_placeholder_data()` — is documented below, since it's a genuine
design choice, not leftover scaffolding).

Built as:

```
src/pipeline/gui/
├── app.py                    MainWindow -- CalibrationScreen -> LiveViewWidget -> ExtendedMeasurementScreen
│                              navigation, owns the one shared CameraStream both downstream screens use
├── theme.py                  Shared dark-palette color/spacing/font constants + load_bundled_font()
├── assets/fonts/              Bundled Latin Modern Roman .otf files (OFL-licensed)
├── calibration_screen.py      CalibrationScreen (WelcomePage -> CreatePage), CalibrationBundle
├── calibration_dialogs.py     BaselineDialog, FlatFieldDialog, ConversionGainDialog,
│                              SpatialCalibrationDialog, SpectralCalibrationDialog, error dialogs
├── live_view.py               LiveViewWidget -- real-time QTimer polling loop (scatter/fit/heatmap +
│                              strip chart + side panel) plus several pure/non-Qt helper functions
├── extended_measurement.py    ExtendedMeasurementScreen -- N-shot acquire + combine workflow
└── roi_control.py             SpatialROIControl, shared by live_view.py and extended_measurement.py
```

`tests/test_calibration_screen.py`/`test_live_view.py`/`test_extended_measurement.py` (split out of one
originally-monolithic `tests/test_gui.py`, one file per screen, so the wiring agents below could work
with zero merge-conflict risk) cover each screen's real backend calls as pytest-qt offscreen tests —
`SyntheticBackend`-driven `CameraStream`s, real `build_*()`/`save_*()` calls against synthetic frame data
wherever practical, mocked only where a real call needs preconditions `SyntheticBackend` can't produce
(e.g. `SyntheticBackend`'s smooth Gaussian beam has no discrete lamp-line peaks for spectral-capture
line-matching or geometric-tilt detection). `tests/gui_fixture_helpers.py` provides one shared, real
(`build_*()`-constructed, not hand-typed) `CalibrationBundle` fixture for the live-view/extended-
measurement test files. `tests/test_gui.py` now covers only `app.py`'s cross-cutting screen navigation.
`tests/test_gui_end_to_end.py` (new) drives the full `MainWindow` flow — real calibration artifacts
saved to disk, loaded via `WelcomePage`'s real load path, through a real polling `LiveViewWidget` and a
real N-shot `ExtendedMeasurementScreen` run — specifically to catch cross-screen integration bugs
isolated per-screen tests can't; it caught one for real (see the camera-lifecycle bug below).
`scripts/demo_live_view.py`/`scripts/demo_app.py` remain useful for interactive/screenshot exploration
against a placeholder (not real) bundle — see each script's own docstring.

**Every screen's real flow, in outline:**
- **`WelcomePage`'s "Load Existing Calibrations"** loads baseline/flat-field/bad-pixel-map/conversion-
  gain/spectral (all hard-required — missing any shows "No existing calibrations found. Please create
  new calibrations." and stays on `WelcomePage`) plus scale-factor and geometric-tilt (both soft-optional
  — a physically valid default / "skipped" respectively) from `calibration_artifacts/`, builds a
  `CalibrationBundle`, and emits `calibration_ready`.
- **`CreatePage`**'s five dialogs each call their real backend function on Start/Save
  (`run_baseline_calibration`, `capture_dark_frames`/`capture_illuminated_frames`/
  `finish_flat_field_calibration` + a real `build_bad_pixel_map()`/`save_bad_pixel_map()` chained on,
  `run_conversion_gain_calibration`, `save_scale_factor`, `run_spectral_calibration`/
  `build_manual_spectral_calibration`), saving to the same `calibration_artifacts/` paths `WelcomePage`
  reads from. `CameraError` routes to `show_camera_error_dialog()`; calibration-specific failures
  (`SettingsMismatchError`/`InvalidFlatFieldError`/`InvalidConversionGainError`/`NoSignalError`/
  `LineMatchingError`) route to `show_calibration_error_dialog()`, leaving the dialog open to retry rather
  than closing on failure. `CreatePage` tracks per-type completion and enables "Continue to Main Window"
  once baseline/flat-field/conversion-gain/spectral are all done (spatial isn't gated — it always has a
  valid default); "Continue" reuses the exact same `_attempt_load_existing_calibrations()` re-read-from-
  disk path `WelcomePage` uses, rather than threading each dialog's in-memory result through a second code
  path. Its per-type "done" flags now also **pre-mark** any type whose artifact is already found on disk
  at construction time (`_mark_existing_calibrations()`, probing `DEFAULT_ARTIFACT_DIR` with the same
  `load_*()` calls `WelcomePage` uses, each type independent — one missing file doesn't block detecting
  the others), revealing an "Existing calibration found on disk." note on that type's card
  (`_CalibrationTypeCard.mark_existing_on_disk()`) without changing the card's action button — re-running
  it is still how the user recalibrates. Fixes a real gap: before this, the four gated flags were
  session-only, so a user who'd already calibrated in an earlier app run, then closed and relaunched into
  `CreatePage` (rather than `WelcomePage`'s "Load Existing Calibrations"), would have had to redo
  perfectly good captures just to re-enable "Continue to Main Window" — the artifacts themselves always
  persisted across restarts in `DEFAULT_ARTIFACT_DIR`, only the page's own flags didn't know that.
- **`LiveViewWidget`** runs a real `QTimer` (`DEFAULT_UPDATE_INTERVAL_MS = 200`, ~5Hz, overridable) polling
  loop: each tick pulls `camera_stream.get_latest_frame()`, runs it through `run_preprocessing()` (using
  the live-updated ROI bounds from `SpatialROIControl`) and `analyze_shot()`, and pushes the result into
  the scatter/error-bars/fit-curve/heatmap/strip-chart/side-panel. `_populate_placeholder_data()` still
  seeds the widget's very first paint (before the first real tick has anything to show) — the one
  legitimate remaining placeholder use in this package, not a gap.
- **`ExtendedMeasurementScreen`** needs no timer — "Run Measurement" is a synchronous acquire-then-analyze
  flow: optionally reconfigures the shared camera stream (stop/set exposure+gain/restart, mirroring
  conversion-gain calibration's own cycle) if the entered settings differ from the stream's current ones,
  blocks on `camera_stream.collect_n_frames(n_shots)`, runs each frame through `run_preprocessing()` then
  `analyze_shot()` at every selectable degree at once (so switching the degree dropdown afterward needs no
  re-acquisition), and combines the results (see below).
- **`app.py`'s `MainWindow`** owns the one shared `CameraStream` both downstream screens are built around:
  builds it via `build_camera_stream()` and **starts it** in `_on_calibration_ready()` (see the bug note
  below), stops it again in `closeEvent()` -- and also in `_on_back_to_calibration_requested()`, wired to
  `LiveViewWidget`'s "Back to Calibration" button, which fully tears down both downstream screens and
  returns to `CalibrationScreen` (see the to-do list entry below for why "fully," not just hidden).

**Bug found and fixed during the end-to-end integration pass**: `build_camera_stream()` explicitly does
not start the stream it returns ("callers own stream lifecycle" — see its own docstring), and
`MainWindow._on_calibration_ready()` built the shared `CameraStream` and handed it straight to
`LiveViewWidget`/`ExtendedMeasurementScreen` without ever calling `.start()` on it. Neither Agent B's
(live view) nor Agent C's (extended measurement) own tests caught this, since both start their own
`CameraStream` directly rather than going through `MainWindow` — a live camera stream that's never
started would have meant `get_latest_frame()` returning `None` forever and `collect_n_frames()` hanging
indefinitely, in the real app, despite every screen's own isolated tests passing. Caught specifically by
`tests/test_gui_end_to_end.py`'s cross-screen drive-through, which is the reason that test file exists
as a distinct thing from the three per-screen files above.

**Three more real-camera-error gaps found in a pre-lab-session audit, all now fixed.** Every screen's
tests run exclusively against `SyntheticBackend`, which never raises `CameraError` -- so three call sites
into the real camera had accumulated with zero exception handling, none of them caught by any existing
test:
- `app.py`'s `_on_calibration_ready()` -- `camera_stream.start()`, right as calibration hands off to live
  view. A real `CameraConnectionError` here (no device found, already open elsewhere, camera not powered
  on yet) would have crashed that transition outright, before live view was shown even once.
- `extended_measurement.py`'s `_maybe_reconfigure_camera_stream()` -- the `.start()` half of its
  stop/reconfigure/restart cycle, if the camera fails to come back up after being stopped to apply new
  exposure/gain.
- `extended_measurement.py`'s `_on_run_clicked()` -- `collect_n_frames()`, if the camera drops mid-
  acquisition (a real, plausible lab event: a cable wiggle, a GigE hiccup).

All three now route through `show_camera_error_dialog()` -- the same convention every camera-touching
call in `calibration_dialogs.py` already used from the original wiring pass -- catching `CameraError`
(and, for `collect_n_frames()`, the `RuntimeError` it also documents for a stream that stops without
`last_error` set). Both `extended_measurement.py` fixes leave `shot_results`/the display exactly as they
were before the click, same as the `InsufficientDataError` fix above. Verified against the real,
unmocked `build_camera_stream()`/`PylonBackend` with no camera attached (not just mocked in tests): the
real failure path genuinely raises `CameraConnectionError` after PylonBackend's documented 3-attempt/
0.5s-interval device-enumeration retry, `app.py`'s fix catches it cleanly with no crash, and a full sweep
of every remaining `.start()`/`.stop()`/`collect_n_frames()`/`build_camera_stream()` call in `gui/` found
no further gaps -- `calibration_dialogs.py`'s four dialogs already had this from the original wiring
pass, and every `.stop()` call is safe by design regardless of stream state (`PylonBackend.close()`'s own
docstring: "safe to call in any state, including if connect() was never called or failed partway
through").

**A fourth real-camera gap, distinct from the three above: a disconnect mid-session while already on live
view, not just at connection time.** Those three fixes cover *starting* the camera; nothing covered the
camera dying *while already streaming*. `CameraStream`'s background thread, on a fatal `CameraError`
(anything other than a tolerated transient `CameraTimeoutError`), sets `last_error` and exits -- but never
clears `_latest_frame`. `get_latest_frame()` therefore keeps returning the same stale `FrameData` forever,
and `LiveViewWidget._on_timer_tick()` had no check for this: it would have kept re-running that one stale
frame through `run_preprocessing()`/`analyze_shot()` and redrawing it every tick, indefinitely --
indistinguishable from a genuinely live but momentarily static feed, with only a small status-label text
change (`_update_status_label()` already correctly showed `"Status: Camera error -- ..."`) as any
indication. Fixed with a third explicit state, `_camera_disconnected`, mirroring `_settings_drifted`/
`_insufficient_signal`'s existing pattern: checked every tick, before ever calling `get_latest_frame()`;
hides the fit overlay and "N/A"s the diagnostics like the insufficient-signal state; pops a message box
like the drift state, naming the "Back to Calibration" button as the actual recovery path (nothing
restarts a dead stream from inside `LiveViewWidget` itself). The raw heatmap is deliberately left showing
its last frame as a reference, not blanked -- same reasoning as the insufficient-signal state.

**Bug caught and fixed while building the fix above, before it ever reached `main`.** The first version
keyed this new state on `camera_stream.is_running` alone -- but `is_running` is *also* `False` for the
ordinary, harmless "never started yet" case (true at construction, and for several existing tests'
deliberately-unstarted streams), which must keep falling through to the pre-existing "no frame yet, skip
silently" handling, not trigger a disconnect warning. Running the full suite with this bug present
crashed the test process outright: an existing test's widget ticked past 200ms with a never-started
stream, wrongly entered the new "disconnected" state, and popped a real, unmocked `QMessageBox.warning()`
-- fatal in offscreen mode, but this time as a genuine process crash during Qt teardown (a malloc
corruption error), not the usual silent hang this project's other `QMessageBox`/`QDialog.exec()` mishaps
produce. Fixed by keying the check on `camera_stream.last_error is not None` instead -- the same signal
`_update_status_label()` already used to distinguish "genuine fatal error" from "not currently running"
-- which only a real fatal `CameraError` ever sets, never a stream that's simply never been started.

**Bug found and fixed post-merge, from real usage**: `LiveViewWidget._on_roi_changed()` (the
`SpatialROIControl.roi_changed` handler) unconditionally called `_apply_roi_bounds()`, which repaints the
scatter/error-bars/fit-curve/heatmap from `self._placeholder_*` — stale, randomly-generated,
construction-time-only arrays. Since nothing gated this on whether real data was already on screen,
narrowing/widening the spatial ROI after the real polling loop had already started showing genuine
`analyze_shot()` results would replace them with fake placeholder data for one tick's worth of time (until
the next real tick overwrote it again) — a visible flash/jump to an unrelated, uncalibrated-looking beam
pattern every time the ROI control was touched. Fixed with a `self._displayed_real_data` flag, flipped
permanently `False -> True` the first time `_display_shot_result()` draws a real result: once set,
`_on_roi_changed()` only rescales the plot's y-range (`setRange(...)`) and leaves the data-bearing items
alone, letting the next real tick reflect the new ROI naturally rather than repainting fake data as an
intermediate step. `_apply_roi_bounds()` itself is unchanged and still correct for its one remaining
caller — before any real tick has ever landed (construction, or a stream that hasn't produced a frame
yet), there's nothing real to preserve, so cropping the placeholder to the new bounds is still the right
behavior.

**The same bug class, found twice more in the same real-usage report**: `_on_degree_changed()` (the Fit
Degree combo box's handler) and `_exit_drifted_state()` (restoring from a settings-drift episode) both
unconditionally called `_update_fit_panel()` — the placeholder counterpart to
`_update_fit_panel_from_result()`, reading `self._placeholder_fits` rather than a real result. Concretely
reported as: switching to a quadratic/cubic fit degree while live view was already showing real data made
the side panel immediately (and, if no further tick landed for any reason, indefinitely) show "Uncertainty
not available for degree > 1 in live view" — a note that can *only* come from the placeholder path, since
real data always has `sigma_zeta()` available at every degree (see the "degree > 1 uncertainty gap" note
above). Also separately verified, per the same report: the plotted fit curve genuinely does update to a
real quadratic/cubic shape on the next real tick (3/4 real coefficients, curve data changed) — that part
was never broken, only the side-panel note was. Fixed the same way as `_on_roi_changed()`: both call sites
now check `self._displayed_real_data` first and skip the placeholder repaint once real data exists,
letting the next real tick's `_update_fit_panel_from_result()` supply genuine numbers instead. `_update_fit_panel()`
itself is unchanged and still correct for construction time and for a still-drifted/never-yet-real state.

**Exposure/gain consistency between calibration and live view, now built** (the gap: nothing used to
record what exposure/gain a calibration was captured under anywhere the live-view screen could see it,
so a live setting drifting from the calibrated one would silently produce wrong results):
- `BaselineDialog`/`FlatFieldDialog` gained an Auto/Manual exposure choice (`QComboBox`,
  `EXPOSURE_MODE_AUTO`/`EXPOSURE_MODE_MANUAL` in `calibration_dialogs.py`), mirroring the CLI's
  `--auto-exposure`/`--exposure-us` pair (§4). Auto: `exposure_us_spin` disabled (real auto-exposure
  convergence decides it at capture time), `gain_db_spin` reset to 0.0 as a starting suggestion,
  re-applied every time Auto is reselected. Manual: `exposure_us_spin` enabled, gain left alone.
  `auto_exposure()`/`exposure_us()` getters mirror `args.auto_exposure`/`args.exposure_us`'s semantics
  exactly (`None` = "let auto-exposure or the config default decide").
- `CalibrationBundle` gained `conversion_gain_record: ConversionGainRecord | None = None`, populated by
  `_attempt_load_existing_calibrations()` — previously only the derived `gain_e_per_adu` float was kept,
  discarding the `gain_db` the sweep was actually measured at.
- `LiveViewWidget` gained an "Acquisition Settings" `QGroupBox` (exposure_us/gain_db spin boxes) at the
  top of the side panel, pre-filled from `calibration_set.baseline_record`, plus a new constructor
  parameter `conversion_gain_record: ConversionGainRecord | None = None`. Editing either field
  **deliberately, permanently** never touches `camera_stream` — unlike `ExtendedMeasurementScreen`'s own
  Acquisition Settings panel, which does reconfigure the camera on Run, live view's continuous polling
  loop has no natural point to pause for a mid-stream reconfiguration, so this field only ever drives the
  drift check: `exposure_has_drifted()`/`gain_has_drifted()` (pure, non-Qt, reusing
  `calibration.shared.metadata.EXPOSURE_MATCH_TOLERANCE_REL`/`GAIN_MATCH_TOLERANCE_ABS` exactly — the
  same tolerances `check_settings_match()` enforces at the preprocessing layer) compare the entered value
  against `baseline_record` (exposure and gain) and independently against `conversion_gain_record`'s
  `gain_db` (if supplied), since the two artifacts can drift apart from each other. On drift, a
  `QMessageBox` prompt names the specific artifact and both values, asking to recalibrate, and the real
  polling loop skips its per-tick work entirely (raw display frozen, fit overlay hidden) until the entered
  values are back in tolerance; confirming emits `recalibration_requested = Signal(str)` (`"baseline"` or
  `"conversion_gain"`) — still deliberately left unconnected even now that `app.py` exists, since wiring
  it would need a "go back to calibration screen" navigation path `MainWindow` doesn't have yet; noted as
  an open item in §6, not silently dropped.

**Overall structure.** On launch: choose between loading existing calibration artifacts, or creating new
ones (per-type — the user picks which one, not an all-or-nothing choice). Calibration and live view are
mutually exclusive phases, never simultaneous, which happens to line up exactly with the one-camera-
connection-at-a-time hardware constraint without needing to special-case it.

**Fixed layout, not responsive.** Every page's layout is designed to look best at a fixed default window
size, not to reflow as the window is resized. Resizing the window should not rearrange widgets, change
spacing, or restretch cards/panels — every page still uses stretch factors and layouts that reflow on
resize, which needs revisiting on a future pass across `calibration_screen.py`, `live_view.py`,
`extended_measurement.py`, and any future page, likely by fixing the window size (disabling resize) or
constraining layouts to hold their proportions rather than expand into extra space.

**The calibration screen is not uniform across the five types:**
- **Spatial** isn't a camera measurement at all — just a text field for a manually-measured scale-factor
  override (or accept `DEFAULT_SCALE_FACTOR`), a fundamentally different UI element from the other four.
- **Spectral** — like spatial, offers a choice between two flows rather than one, via a two-mode
  `SpectralCalibrationDialog`: "Capture from Lamp" (mirrors `BaselineDialog`'s single-phase form,
  including its Auto/Manual exposure choice — added after a real lamp capture on the lab PC hit
  `check_settings_match()` rejecting an auto-exposure-converged lamp frame (picked ~39ms for the dim
  Argon lamp) against a baseline captured at a fixed 1000us, with no way to force the lamp exposure to
  match short of dropping to the CLI; uses the curated Argon 751.46-842.46nm window, §3; needs an
  already-saved baseline+flat-field on disk first, showing "Missing Sensor Calibration" if either is
  absent, and — as a side effect of calling `run_spectral_calibration()` — builds and saves a real
  geometric-tilt calibration too, for free) and "Manual Entry" (mirrors `SpatialCalibrationDialog`'s
  manual-value style, but for a variable-length coefficient+sigma list, rebuilt per chosen degree, that
  calls `build_manual_spectral_calibration()`). Both modes' Start/Save buttons call their real backend
  function, same as every other dialog in this screen.
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

**The degree > 1 uncertainty gap — closed at both the statistics layer and the GUI layer.** ζ(λ) is an
exactly linear function of the fit coefficients, so a *proper* uncertainty on it needs the coefficients'
full covariance matrix, not just their marginal `coefficient_sigma` (the diagonal). `SpatialDispersionFitResult`
carries `coefficient_covariance` (`scipy.odr`'s `cov_beta × res_var`) and a `sigma_zeta(wavelength_nm)`
method implementing the exact propagation, built in `analysis/dispersion_fitting.py` +
`analysis/results.py` — NOT `calibration/shared/`'s structurally-separate copy of the same machinery
(§3), since the GUI's live/extended-measurement path runs through `analyze_shot()` → `analysis/`, not
calibration's. Both `LiveViewWidget` and `ExtendedMeasurementScreen` now call `sigma_zeta()` for real at
every degree — the old "external-uncertainty-only, with a caveat note" interim behavior for degree > 1 is
gone; the caveat note is retired and both screens report a genuine internal uncertainty at every degree.

**Extended measurement — built as designed:**
- User picks a number of frames, optionally overriding exposure/gain — which forces the same
  stop/reconfigure/restart cycle as conversion-gain calibration (§6) around the shared `CameraStream`,
  visibly freezing live view for the duration; an accepted, expected interruption, not a bug. Only
  triggers when the entered values actually differ from the stream's current ones.
- Degree selection (linear/quadratic/cubic) runs into the existing, deliberate design boundary that
  `analysis/combination.py`'s `combine_shots()` only combines the linear ζ across shots by design (§2 —
  quadratic/cubic fits were explicitly kept per-shot-only, not aggregated, when `analysis/` was designed).
  For degree 1, `combine_shots()` is called directly. For degree > 1: each shot's own
  `zeta(wavelength_ref)`/`sigma_zeta(wavelength_ref)` (evaluated at the "Evaluate At" reference point,
  using each shot's own fit) are combined via the same `combine_shots()` call — it's generic over any
  (value, sigma) pairs, so no separate weighting scheme was needed, and the internal-uncertainty half of
  the combination is now real too (see above), not just the point estimate.
- **Static graph**: NOT a re-fit of per-column-averaged centroid positions — that's exactly the
  alternative combination methodology considered and rejected when `analysis/`'s N-shot combination was
  originally designed (§2: "not per-column combination... reusing existing single-shot code"), and would
  risk the drawn line's slope visually disagreeing with the reported ζ_combined number sitting right next
  to it. Built as one straight line of slope `zeta_combined`, anchored through all raw (shot, column)
  points via a plain weighted-least-squares intercept — never a per-column-averaged refit — so the drawn
  line's slope can't visually disagree with the reported number.
- **Residual plot beneath the graph.** Built: a residual subplot underneath the main scatter/fit-curve
  graph (observed x0 minus the fitted combined curve, per column/shot, sharing the same x-axis) — a
  standard fit-quality check, and a natural fit for extended measurement's already-static
  (non-live-updating) graph. Not needed for the live single-shot view, which redraws too fast for a
  residual panel to be readable.
- **"Save Record" — a complete, git-committed snapshot of one Run Measurement result.** New
  `gui/measurement_record.py`: `save_measurement_record()` writes a stacked frame plus first/middle/last
  individual shots (compressed `.npz` + quick-look `.png` — not every shot, given a default run is 20 and
  up to 1000; a stack is cheap but isn't a quantity this codebase's analysis ever actually computes, so
  the three individual frames are kept too, for real traceable per-shot data), every shot's centroid data,
  every shot's fit at every degree (coefficients/sigma/covariance/reduced chi-squared/residuals), the
  combined polynomial and spatial dispersion at every degree (not just whichever is on screen — see
  below), the spatial+spectral ROI actually used, and the calibration artifacts in effect (copied
  byte-for-byte from `DEFAULT_ARTIFACT_DIR`, not reconstructed from in-memory objects — simpler and exact
  regardless of which fields a given in-memory object happens to retain) — to
  `data/measurements/extended_measurement_<timestamp>/`. Deliberately a separate, explicit button from
  "Run Measurement" (disabled until a measurement exists) — not every trial run should be permanently
  saved. Pure Python, no Qt import at all (`matplotlib`'s `Agg` backend only), directly unit-testable
  without a `QApplication`, matching `roi_control.py`/`formatting.py`'s existing separation of pure
  logic from Qt wiring. Imported *locally* inside `_on_save_record_clicked()`, not at module scope —
  `measurement_record.py` itself imports functions back from `extended_measurement.py` at its own
  module scope (see below), so a module-scope import in the other direction would be circular; mirrors
  `calibration/spectral/workflow.py`'s existing local-import pattern for the identical reason.

  **A correctness fix this surfaced**: neither ROI's value was previously stashed anywhere —
  `_on_run_clicked()` read both controls fresh and used them immediately, with nothing remembering what
  was actually used once the run finished. Since "Save Record" is a separate, later click, re-reading
  the live controls at save time could report the *wrong* ROI for an already-completed measurement if
  either control was edited in between (worse for the spectral one, which actually gates which columns
  get analyzed — a changed value wouldn't even match the columns really present in `shot_results`).
  Fixed by capturing both (`self._measurement_roi_bounds_px`/`_measurement_column_bounds`) into instance
  state inside `_set_measurement_data()`, at the same time as `shot_results` itself — "Save Record" reads
  only those captured values, never the live controls.

  **A real reported multi-minute UI freeze, root-caused and fixed.** A 200-shot Run Measurement followed
  by Save Record showed "Application Not Responding" for several minutes, and switching the degree
  selector afterward froze too. Cause (confirmed by benchmarking against a real 200-shot synthetic run):
  the first version of this feature stored every shot's full-resolution `ProcessedFrame` in
  `self._processed_frames` for the widget's entire lifetime (so Save Record, run later, could pick from
  them) — ~18MB/frame × 200 shots ≈ 3.6GB resident, with a further multi-GB spike averaging that list in
  one call, and the memory pressure lingered into unrelated later actions like the degree-selector
  freeze. Fixed by never retaining the full list at all: `_on_run_clicked()` now folds each frame into a
  running sum (→ a mean at the end) and checks it against `_representative_shot_indices()` (a new free
  function) as it's produced, keeping only a small constant number of full-resolution frames in memory
  (`self._measurement_stacked_image`/`_measurement_representative_frames`) regardless of `n_shots`.
  `measurement_record.py`'s `_save_frames()` takes this pre-reduced data directly rather than reducing a
  frame list itself.

  **A second, independent bottleneck in the plot itself, also fixed.** Even after the memory fix, saving
  still took minutes: a real 200-shot measurement produces on the order of 10^5 (shot, column) points per
  panel, and fully-vector scatter/error-bar rendering at that density made PDF generation alone slow
  (matching a separately reported Adobe Acrobat crash opening the resulting file — a `.pdf` that came out
  56.8MB for one measurement) and the file enormous. Fixed via `ax.set_rasterization_zorder()`
  (matplotlib's documented technique for mixed vector+raster figures): the dense scatter/error-bar/
  residual layers rasterize into a single embedded image while fit curves, axis text, and annotations
  stay vector. Brought a real 200-shot record down from several minutes to ~30 seconds end-to-end and the
  PDF from 56.8MB to under 1MB. The remaining ~30s is still synchronous on the Qt main thread (a
  background-thread version was considered but not built, since this wasn't asked for and the
  measured improvement — from unusable to a single noticeable pause — was judged sufficient for now).

  **Refactor to guarantee the saved numbers can never drift from the display**:
  `_compute_combined_result()`'s and `_recompute_fit_and_residuals()`'s core computations were pulled out
  into two module-level free functions in `extended_measurement.py`
  (`compute_combined_result_for_degree()`, `compute_fit_line_and_residuals()`), with the original bound
  methods becoming thin wrappers (behavior-preserving — full pre-existing test suite still passes
  unmodified). `measurement_record.py` calls the *same* two functions for degree 1, so that degree's
  numbers are guaranteed identical to what the live GUI would show, not independently re-derived logic
  that could silently drift from it.

  **A real labeling bug, found from real use and fixed: spatial dispersion (ζ) is not the same thing as
  the polynomial's c1 coefficient, except at degree 1.** ζ = dx0/dwavelength_nm is a *derivative*
  evaluated at one reference point; at degree > 1 it depends on c2/c3 too, and only collapses to c1 when
  the polynomial's derivative is constant everywhere (degree 1). An earlier version of
  `combined_results.txt` labeled `combine_shots()`'s combined-zeta value "c1" at every degree, and never
  reported c2/c3 at all — both wrong. Fixed: `compute_combined_polynomial_for_degree()` (new, in
  `extended_measurement.py`) combines *every*
  coefficient across shots via the same inverse-variance `combine_shots()` weighting
  `compute_combined_result_for_degree()` already used for zeta alone (a real generalization, not an ad
  hoc addition — each shot's `coefficients[k]` is an independent estimate of the same physical quantity
  regardless of k) — used for degree 2/3 (degree 1 keeps its existing intercept+zeta_combined pair,
  unchanged, still GUI-matching). `combined_results.txt` now lists every combined coefficient (c0..c_degree,
  with per-degree units, e.g. `mm/nm^2`) as its own section, separate from spatial dispersion — which is
  now evaluated at the center of the realized spectral ROI (`(columns.min() + columns.max()) / 2`,
  converted via `axis_for_fit`) rather than the live GUI's "Evaluate At" spin box value, since that
  reference point is a live-only concept (a user-editable pixel column with a fixed default, unrelated to
  any specific run's actual ROI) that doesn't carry over meaningfully to a saved record describing one
  specific measurement. `save_measurement_record()` no longer takes an `evaluated_at_wavelength_nm`
  parameter at all as a result.

  **The plot's degree 2/3 curves were tangent lines, not real curves — also fixed, using the same
  coefficient combination above.** The original version drew every degree's "fit curve" from a single
  reference-point slope (`compute_fit_line_and_residuals()`), so quadratic/cubic panels always rendered
  as straight lines — visually indistinguishable from degree 1, which is what prompted the bug report
  ("I am not sure the curves of best fit have been drawn correctly"). Now degree 2/3 evaluate the real
  combined polynomial (`np.polynomial.polynomial.polyval` against `compute_combined_polynomial_for_degree()`'s
  output) across the fit range, and residuals are computed against that same curve — genuine curvature
  shows up when the data actually has any (this synthetic-data test case doesn't, so its curves still
  look straight, correctly, since the injected chirp really is linear there).

  **The live GUI had the exact same tangent-line issue, fixed the same way.** Until this pass,
  `ExtendedMeasurementScreen._refresh_measurement_display()` drew degree 2/3's main-plot curve/residuals
  from `_recompute_fit_and_residuals()` at every degree — the same single-reference-point tangent line
  degree 1 uses, recomputed (both slope *and* intercept) every time "Evaluate At" changed, so the whole
  curve visibly moved on every edit even though nothing about the underlying combined fit had changed
  (the bug report that prompted this: "when I switch to quadratic or cubic fits, if I change the pixel
  at which the spatial dispersion is evaluated at, the curve of best fit on the plot seems to move, and
  so do the residuals"). `_combine_polynomial_coefficients()` was moved out of `measurement_record.py`
  and into `extended_measurement.py` as the shared `compute_combined_polynomial_for_degree()`, so the
  live GUI and the saved record draw from the identical function — no independently re-derived logic to
  drift apart. A new `_recompute_combined_polynomial_fit_and_residuals()` (degree 2/3's counterpart to
  `_recompute_fit_and_residuals()`) builds the curve/residuals from that combined polynomial instead,
  taking no reference-point argument at all, since the combined polynomial doesn't depend on it — only
  `_spatial_dispersion_label` (a scalar evaluated *at* the reference point, via
  `compute_combined_result_for_degree()`, unchanged) still moves when "Evaluate At" is edited. The
  "Coefficients" row was also generalized from its previous hardcoded c0/c1-only display to list every
  combined coefficient at any degree, fixing the same c1-labeled-as-spatial-dispersion issue in the live
  GUI that `combined_results.txt` already had fixed for the saved record. Verified with a new regression
  test asserting the drawn fit curve/residuals/coefficients are bit-identical before and after an
  "Evaluate At" edit at degree 2/3, and a second test confirming the live GUI's coefficients match
  `compute_combined_polynomial_for_degree()` called directly.

  **The saved plot** is deliberately not styled like anything else in this codebase — these may end up in
  reports/presentations, so it follows physics-journal convention instead of this app's own dark theme:
  serif font, Computer-Modern mathtext (no real LaTeX install required), no gridlines, ticks inward on all
  four spines, no per-axes titles (labels only), uncertainties shown as error bars, degrees distinguished
  by both color and linestyle (grayscale/colorblind-safe). Saved as both `.png` (quick view) and `.pdf`
  (vector, for actual publication use). Layout: three side-by-side fit subplots (one per degree — scatter
  + that degree's own combined curve) each with its **own** residual panel directly underneath (not one
  combined panel with all three degrees overlaid, which an earlier version drew — three near-identical
  overlapping series in one panel are indistinguishable at any alpha when the underlying dispersion is
  close to linear across degrees, which is common; reducing alpha further only ever thinned the visible
  blend, it never separated it). All three residual panels share one y-axis range so residual *magnitude*
  stays directly comparable across degrees despite being in separate panels now. Each fit subplot's ζ/χ²ᵥ
  annotation is placed in whichever top corner the fitted dispersion's sign leaves clear of the data (a
  steep, full-range-spanning curve means the *other* diagonal's two corners are never clear regardless of
  slope direction), with a white background patch as a second line of defense either way — a plain
  `ax.legend()` was tried first and rejected: a long label string anchored at a nominally-empty corner
  still stretches back across most of the axes width, unavoidably overlapping the diagonal.

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
- ~~**`calibration/spectral/line_matching.py`'s `match_lines()`.**~~ **Done.** Argon reference lamp
  chosen, grating-equation geometry supplied, peak-detection + matching search built (with a
  physics-driven synthetic-lamp-image test suite) and verified fast (~0.06s, after fixing an initially
  unvectorized version that took minutes) -- see §3. GUI enablement (`SpectralCalibrationDialog`, both
  capture and manual-entry modes) and the CLI `spectral-capture`/`spectral-manual` subcommands are also
  now done -- see §4/§5. Both dialog modes' accept paths remain UI-only placeholders, same as every
  other calibration dialog, pending the follow-up pass that wires all of them to real build_*() calls.
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
- ~~**Build the GUI** per the design recorded in §5.~~ **Done**, in two passes: a tested Phase-1 visual
  skeleton first, then a full real-backend wiring pass (Team-Lead-orchestrated multi-agent effort — one
  agent per screen, Team-Lead-only sequential merges with a full-suite check after each, a shared
  end-to-end smoke test at the end). Every screen now calls real `build_*()`/`run_preprocessing()`/
  `analyze_shot()`/`combine_shots()` — see §5 for the full breakdown, including the camera-lifecycle bug
  the end-to-end pass found and fixed. Remaining GUI work is the more specific items below (fixed layout,
  spectral-axis manual ROI entry, input validation, `recalibration_requested` routing), not "wire it up."
- ~~**GUI live view: fit curve doesn't actually change shape with the selected degree.**~~ **Done** — was
  a genuine bug in the Phase-1 placeholder path (`_populate_placeholder_data()` drew `_fit_curve` from a
  single fixed "true" trend function, independent of `self._current_degree`); moot now that the real
  polling loop draws `_fit_curve` from `analyze_shot()`'s actual per-degree fit each tick.
- **GUI: fixed (non-responsive) layout across all pages.** Currently the calibration screen and live
  view both reflow on window resize (stretch factors, expanding layouts) — needs a pass to make every
  page hold a fixed layout tuned for its default window size regardless of resizing. See §5.
- ~~**GUI live view: no way to navigate back to CalibrationScreen.**~~ **Done.** `LiveViewWidget` gained a
  "Back to Calibration" button (`back_to_calibration_requested = Signal()`, placed in the side panel below
  "Extended Measurement...", mirroring `ExtendedMeasurementScreen`'s own `back_requested`/`_back_button`
  pattern). `MainWindow._on_back_to_calibration_requested()` stops the shared `CameraStream` (freeing it
  for `CalibrationScreen`'s own dialogs — the one-camera-connection-at-a-time constraint means both can't
  be connected at once) and **fully tears down** `LiveViewWidget`/`ExtendedMeasurementScreen` (removed from
  `self._stack`, `deleteLater()`'d, references dropped) rather than just hiding them, since
  `_on_calibration_ready()` can now genuinely fire more than once per `MainWindow` lifetime — completing
  calibration again after navigating back builds a **fresh** `LiveViewWidget` and `CameraStream`, never
  reusing or resurrecting the torn-down ones. `recalibration_requested` (emitted by the drift-detection
  logic in §5) is still NOT auto-connected to this new back-navigation — that remains a deliberate, open
  question (should a drift warning also auto-navigate the user back, or is the manual button enough now
  that one exists?), not an oversight.
- **Input validation pass**, GUI and CLI both -- not yet done anywhere. Numeric
  fields (exposure_us, gain_db, scale factor, ROI bounds once built, etc.) and
  CLI arguments currently rely on Qt's own spin-box range clamping or accept
  whatever argparse's type= coercion allows, with no deliberate validation
  layer of this codebase's own (rejecting nonsensical combinations, clearer
  error messages, etc.). Noted for a future pass once more of the real
  input surfaces (dialogs, CLI flags) exist to validate.
- ~~**GUI live view: manual ROI entry -- spatial axis.**~~ **Done.** New
  `gui/roi_control.py`: `SpatialROIControl(QGroupBox)`, a self-contained widget
  (deliberately no dependency on `live_view.py`, so a future Extended Measurement
  dialog can embed it too) with min/max `QDoubleSpinBox` fields in mm (matching
  the live-view plot's y-axis units), a "Reset to Full Range" button, and inline
  validation (`min >= max` is rejected with an error label, reverting to the last
  valid pair). Session default is the full spatial extent. `roi_bounds_px()`
  converts back to pixel-row bounds via a new inverse method,
  `calibration/spatial/calibrate.py`'s `ScaleFactorPositionCalibration.to_pixels()`
  -- shaped exactly like `preprocessing/steps/roi.py`'s `apply_roi()`/
  `run_preprocessing()`'s `roi_bounds` parameter, ready for whenever real camera
  wiring happens (not yet -- see below). Wired into `live_view.py`'s side panel;
  entering new bounds narrows the plot's y-axis to exactly `[min, max]`
  (`setYRange(..., padding=0)`, which also disables pyqtgraph's autorange on that
  axis) and crops the displayed scatter/fit-curve/heatmap to match, zeroing
  heatmap rows outside the window rather than resizing the image -- mirroring
  `apply_roi()`'s real zero-not-crop behavior for visual consistency. Now that
  `live_view.py`'s polling loop is real (see above), this control's bounds are
  genuinely fed into every tick's `run_preprocessing(roi_bounds=...)` call, not
  just used for display cropping.
- ~~**GUI live view: manual ROI entry -- spectral axis.**~~ **Done.** New
  `preprocessing/steps/spectral_roi.py`: `apply_spectral_roi(frame, column_min,
  column_max)`, shaped like `apply_signal_threshold()` rather than `apply_roi()`
  -- it only ever *overrides* `ProcessedFrame.valid_columns` (every column in
  `[column_min, column_max)` forced valid regardless of its actual SNR, every
  column outside forced invalid regardless of signal strength), never zeroes
  `frame.image`, matching the "overriding the automatic SNR gate" framing this
  item was originally scoped around. `run_preprocessing()` gained an optional
  `column_bounds` parameter, applied right after `apply_signal_threshold()`;
  `None` (the default) leaves the automatic gate as the sole word on validity,
  unchanged from before this parameter existed. `gui/roi_control.py` gained
  `SpectralROIControl(QGroupBox)`, the sibling this item anticipated -- min/max
  `QSpinBox` fields, but in raw **pixel-column** units rather than nm/mm:
  unlike the spatial scale factor, `analysis.interfaces.WavelengthAxis` is a
  general degree-N polynomial fit with no inverse method, and may not exist at
  all yet, so pixel-column bounds are the only representation always
  well-defined; a read-only wavelength hint is shown alongside when a
  `WavelengthAxis` is loaded, computed via the one direction that's always
  defined (`wavelength_nm()`). Distinguishes `column_window()` (always the
  current bounds, for viewport zooming) from `column_bounds()` (`None` at the
  full-range default) -- the full range must map to `None`, not the literal
  `(0, n_columns)` tuple, since passing that through as an explicit override
  would force every column valid and silently disable the SNR gate instead of
  leaving it in place. Wired into both `live_view.py` and
  `extended_measurement.py`'s side panels, alongside the existing
  `SpatialROIControl`. In `live_view.py`, narrowing the window actually zooms
  the plot's x-axis to match (`_current_x_extent()`, the spectral counterpart
  to the spatial control's y-axis zoom) and drops out-of-window scatter/fit
  points on every tick, both for the placeholder feed (before any real frame
  has landed) and for real per-tick data (via `column_bounds=` on
  `run_preprocessing()`), mirroring `SpatialROIControl`'s existing before/
  after-real-data handling exactly. In `extended_measurement.py`, the current
  value is read once per "Run Measurement" click (same timing as the spatial
  control's `roi_bounds`) -- deliberately NOT wired to a live re-render like
  the spatial control is: since a spectral-ROI change alters which columns are
  even analyzed (`valid_columns`, gating `extract_centroids()` itself) rather
  than just zeroing pixels post hoc, there is no way to retroactively apply a
  change to an already-completed run's `shot_results` without either
  fabricating data for columns never analyzed (widening) or misrepresenting
  how many columns went into the combined result (narrowing) -- a change here
  takes effect on the *next* Run Measurement, not immediately.
- ~~**`analysis/`: proper internal uncertainty on ζ ("spatial dispersion" in the GUI) for
  degree > 1.**~~ **Done.** `SpatialDispersionFitResult` gained a `coefficient_covariance`
  field (the full (degree+1, degree+1) matrix, not just `coefficient_sigma`'s diagonal)
  and a `sigma_zeta(wavelength_nm)` method. Since ζ(λ) = Σ k·c_k·λ^(k-1) is an *exact*
  linear function of the coefficients c (not an approximation), its variance is exactly
  `g(λ)ᵀ · coefficient_covariance · g(λ)` where `g_k(λ) = k·λ^(k-1)` (k ≥ 1, `g_0 = 0`) --
  standard linear error propagation, applied exactly rather than as a first-order
  approximation. `TotalLeastSquaresFit` now populates `coefficient_covariance` as
  `scipy.odr`'s `cov_beta × res_var` (`cov_beta` alone is normalized -- this is the same
  scaling `odr` already applies internally to get `sd_beta` from `cov_beta`'s diagonal,
  just kept for the full matrix instead of collapsed to it). Collapses to the existing
  `coefficient_sigma[1]` at degree == 1 (verified by test, since `g = (0, 1)` there and
  cross terms vanish). Built in `analysis/dispersion_fitting.py` + `analysis/results.py`
  only, not `calibration/shared/`'s separate copy of the fitting machinery, per the
  no-cross-dependency rule. The GUI wiring pass still pending in §5 (live view's degree
  > 1 side panel, extended measurement's degree > 1 combination) can now report a real
  internal uncertainty instead of "uncertainty not available" -- that wiring itself is
  unchanged/still pending, only the missing statistics underneath it are now built.
- **Monte Carlo / bootstrap uncertainty validation** for `analysis/`
  (optional, time permitting) — see centroid uncertainty note in §2.
- **Calibration dialogs: no persistent "finished" confirmation.** Each of the five
  `calibration_dialogs.py` dialogs currently `accept()`s (closing itself) the instant its
  underlying `build_*()`/`save_*()` call succeeds, with only a status-label text change (e.g.
  `"Baseline calibration complete."`) visible for a moment before the dialog disappears —
  there's nothing the user can look at afterward to be sure the capture actually finished
  successfully, as opposed to the dialog just closing for some other reason. Requested: a
  clearer "calibration complete" indication (e.g. holding the dialog open on an explicit
  success state with its own "Close" button, or a brief confirmation message) so the user
  knows it's safe to close the dialog and move on.
- **Redo spectral calibration under the new geometric-tilt default, then take real spatial-chirp
  measurements.** Now that `run_spectral_calibration()` builds its geometric-tilt calibration via
  `build_geometric_tilt_linear()` (§3), the spectral calibration on disk needs to be re-captured so its
  geometric-tilt artifact reflects the new default rather than the old pointwise one it was built with.
  Once redone, take real spatial-chirp measurements (live view and/or extended measurement) to see whether
  the grating-rotation physical fix and the linear tilt correction together (§0/§3) produce a genuinely
  chirp-free beam, or a real, accurately-measured nonzero chirp. Needs the real camera connected — not
  something to run unattended.
- **Investigate the two open geometric-tilt questions flagged in §3**: whether the lamp image's brightest
  line is really two unresolved close Argon lines (would need either less total tilt per exposure, or a
  wavelength-informed match against the known line list to split them), and whether the edge-row row_shift
  noise is genuine shot noise or a `MAX_WINDOW_HALF_WIDTH` window-clipping artifact (test by widening it
  and re-running the comparison). Neither blocks the default-method switch above, but both affect how much
  to trust either method's row_shift curve at its extremes.
