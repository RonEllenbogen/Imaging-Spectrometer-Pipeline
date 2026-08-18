# Architecture

This document describes how the imaging-spectrometer software is put together: the packages under
`src/pipeline/`, how data flows between them, and the contracts that keep them decoupled. For *why*
particular design decisions were made (bugs found, alternatives rejected, deliberate cuts), see
`project_handover.md`. For current status and open items, see `project_state.md`.

## System overview

The software turns raw camera frames into a spatial-chirp estimate (ζ) through five layers, each with
a narrow, one-directional dependency on the layer(s) below it:

```
acquisition  →  calibration  →  preprocessing  →  analysis
     │                                                │
     └───────────────────────┬────────────────────────┘
                              │
                        gui / cli
```

- **`acquisition/`** talks to the camera (or a synthetic stand-in) and produces raw frames.
- **`calibration/`** *builds* correction artifacts (baseline, flat field, bad-pixel map, conversion
  gain, spatial scale factor, spectral wavelength axis) from batches of raw frames.
- **`preprocessing/`** *applies* those artifacts to a single science frame, in a fixed, code-enforced
  order, producing a cleaned float frame.
- **`analysis/`** turns a cleaned frame into a per-shot spatial-dispersion (ζ) estimate, and combines
  many shots into one result with a correlation-aware uncertainty.
- **`gui/`** and **`cli/`** are orchestration layers on top of all of the above — the GUI for
  interactive/live use, the CLI for headless calibration sessions.

`calibration/` depends on `acquisition/` (it drives the camera to collect calibration frames) and is
depended on by `preprocessing/` (which applies its artifacts). `analysis/` depends on neither
`calibration/` nor `preprocessing/` directly — it talks to two small `Protocol` interfaces instead (see
[Analysis](#analysis-srcpipelineanalysis)), so the dispersion math can be developed, tested, and
reasoned about without pulling in the calibration machinery at all.

## The canonical frame contract

`src/pipeline/acquisition/frame.py` derives `CANONICAL_SHAPE`, `CANONICAL_DTYPE`, and
`CANONICAL_MAX_VALUE` from `configs/default.yaml` at import time. Every frame type in the pipeline
validates against these in `__post_init__`, so a malformed frame fails immediately at construction
rather than propagating downstream:

- **`FrameData`** (`acquisition/`) — a raw integer frame plus capture metadata (`frame_id`,
  `timestamp`, `exposure_us`, `gain_db`). Frozen and slotted; the wrapped array is also marked
  read-only (`.flags.writeable = False`) since `frozen=True` alone doesn't stop in-place mutation of a
  numpy array.
- **`ProcessedFrame`** (`preprocessing/`) — the float counterpart used from baseline subtraction
  onward, once pixel values need to go negative before being clipped. Same shape contract, relaxed to
  any floating dtype.

Axis convention throughout the pipeline: axis 0 is spatial (rows), axis 1 is spectral (columns) —
`SPATIAL_AXIS`/`SPECTRAL_AXIS` in `frame.py`.

## Acquisition (`src/pipeline/acquisition/`)

`CameraStream` runs a background thread that continuously calls a `CameraBackend.grab_one()` and
publishes only the newest frame behind a lock; `get_latest_frame()` is a non-blocking read for
consumers like the GUI's live view. It tolerates up to `max_consecutive_timeouts` transient
`CameraTimeoutError`s before treating the stream as fatally broken; any other `CameraError` is fatal
immediately; anything else is left uncaught so a real bug surfaces as a traceback rather than
vanishing into `last_error`.

`CameraBackend` is a structurally-typed `Protocol` with two implementations:

- **`PylonBackend`** wraps the real pypylon/pylon SDK for the Basler `a2A1920-51gmBAS`. Construction
  never touches hardware; `connect()` opens the device, `configure()` applies settings and starts
  grabbing.
- **`SyntheticBackend`** generates synthetic frames (Gaussian beam + injected chirp + noise), no
  hardware involved — what the test suite and the GUI's demo scripts run against.

Hardware-only tests are gated behind `SPECTROMETER_HARDWARE_TESTS=1` rather than always requiring a
connected camera.

## Calibration (`src/pipeline/calibration/`)

Owns *building* calibration artifacts; `preprocessing/` owns *applying* them — a one-directional
dependency enforced in practice throughout the package.

- **`shared/`** — cross-cutting machinery used by more than one artifact type. `io.py`'s
  `save_artifact()`/`load_artifact()` persist one or more named arrays plus a record to a single
  `.npz` file (never pickle, so loading an artifact never executes arbitrary code); every subpackage's
  `save_*()`/`load_*()` is a thin wrapper around this. `metadata.py`'s `CalibrationRecord` tags a
  frame-built artifact with the exposure/gain/timestamp/frame-count it was built under, so a science
  frame's actual settings can be checked against it before the artifact is applied. `fitting.py`'s
  `PolynomialFitter` protocol (default `TotalLeastSquaresFit`, via `scipy.odr`) and `result.py`'s
  `PolynomialFitResult` generalize `analysis/dispersion_fitting.py`'s total-least-squares machinery to
  generic x/y — a structurally separate implementation, not a shared import, since `calibration/` and
  `analysis/` must not depend on each other.
- **`sensor/`** — `baseline.py` builds a per-session background average (`BaselineResult`), which also
  measures `background_sigma` — the per-pixel noise scatter used both for saturation-adjacent QA and
  as the "b" term in `analysis/noise_model.py`'s uncertainty formula. `flat_field.py` builds a PRNU
  correction from uniform-illumination frames (dark-subtracted first). `bad_pixel_map.py` flags
  dead/hot pixels from flat-field outliers. `conversion_gain.py` measures `gain_e_per_adu` via a
  photon-transfer curve (uniform illumination swept across exposure times, variance computed
  temporally so PRNU doesn't bias it low). `saturation.py`'s `check_saturation()` checks a raw frame
  against `CANONICAL_MAX_VALUE` and returns a result for the caller to act on rather than raising.
  `workflow.py` glues `acquisition/`'s `CameraStream` to these `build_*()`/`save_*()` functions:
  single-phase calibrations (baseline) run acquire→build→save in one call; flat-field calibration is
  split into `capture_dark_frames()`/`capture_illuminated_frames()`/`finish_flat_field_calibration()`
  since its physical setup changes mid-capture; conversion-gain calibration repeatedly stops/
  reconfigures/restarts the camera stream to sweep exposure, restoring the original exposure
  afterward.
- **`spatial/`** — pixel→physical-position conversion at the spectrometer's slit is a *fixed scale
  factor* (`DEFAULT_SCALE_FACTOR = 1.5`, the ratio of the imaging spectrometer's two relay-lens focal
  lengths), not a per-point fit — the project's scope only needs displacement along the detector's
  spatial axis, and the focal-length ratio is precise enough on its own.
  `ScaleFactorPositionCalibration` implements `analysis.interfaces.PositionCalibration` directly.
  `load_scale_factor()` returns `DEFAULT_SCALE_FACTOR` rather than raising if no override file exists.
  A GUI-entered manual override is persisted via `save_scale_factor()` and reused in later sessions.
- **`spectral/`** — pixel→wavelength calibration from a spectral-lamp image (reference: Argon).
  `calibrate.py`'s `calibrate_spectral()` fits matched (pixel, wavelength_nm) pairs; the returned
  `WavelengthCalibrationResult` implements `analysis.interfaces.WavelengthAxis` directly.
  `grating_geometry.py` predicts relative pixel spacing between wavelengths from the spectrometer's
  transmission-grating equation (constants in `configs/default.yaml`'s `spectrometer:` section).
  `line_matching.py`'s `match_lines()` detects spectral peaks and matches them to reference lines via
  a search over the geometry-predicted spacing pattern. `geometric_tilt.py` builds a correction for
  detector/spectrometer misalignment (the `geometric_tilt` field on `CalibrationSet`, applied by
  `preprocessing/` when present). `workflow.py`'s `run_spectral_calibration()` runs the full path
  end-to-end: acquire → preprocess each frame via a caller-supplied `CalibrationSet` → average →
  `match_lines()` → `calibrate_spectral()` → save.
- Exceptions derive from `CalibrationError`: `SettingsMismatchError`, `InvalidFlatFieldError`,
  `InsufficientDataError`, `LineMatchingError`.

## Preprocessing (`src/pipeline/preprocessing/`)

`run_preprocessing()` in `preprocessing_pipeline.py` is the single public entry point and encodes the
correction order **in code**, not documentation:

1. `check_frame_sanity()` — rejects an all-zero raw frame.
2. `check_saturation()` — checked against the raw frame, returned (not raised) for the caller to act
   on.
3. `apply_baseline()` — subtract, clipped at zero.
4. `apply_flat_field()` — divide, floored so it can never produce inf/nan.
5. `apply_bad_pixel_map()` — zero flagged pixels (not interpolated — exact for a weighted centroid).
6. `apply_geometric_tilt_correction()` — only if `calibration.geometric_tilt` was supplied; skipped
   cleanly otherwise.
7. `apply_signal_threshold()` — an automatic SNR gate against `background_sigma` that marks which
   spectral columns actually carry signal (`valid_columns`), consumed downstream by `analysis/`.
8. `apply_spectral_roi()` — optional, overrides the automatic column gate with an explicit
   `column_bounds` range.
9. `apply_roi()` — optional, zeroes spatial rows outside `roi_bounds` while keeping `CANONICAL_SHAPE`
   intact (masks, doesn't crop).

`CalibrationSet` (in `preprocessing_pipeline.py`) bundles everything step 3–7 need — built once per
session from `calibration/`'s artifacts and reused across every science frame.

Exceptions derive from `PreprocessingError`: `SaturationError`, `NoSignalError`.
(`SettingsMismatchError`/`InvalidFlatFieldError` are raised from `calibration/`, not here, but a
caller of `preprocessing/` can still see them — `pipeline.preprocessing` re-exports
`SettingsMismatchError` for convenience.)

## Analysis (`src/pipeline/analysis/`)

Computes per-shot spatial dispersion (ζ) from an already-preprocessed frame and combines it across
shots. Deliberately has no import of `pipeline.calibration` anywhere — it depends only on two
`Protocol`s in `interfaces.py`:

- **`WavelengthAxis`** — `wavelength_nm(pixel)` / `sigma_wavelength_nm(pixel)` (must be strictly
  positive, since the fit weights by inverse sigma). Implemented directly by
  `calibration.spectral`'s `WavelengthCalibrationResult`.
- **`PositionCalibration`** — `convert(x0, sigma_x0)`, pixel→physical with uncertainty propagation.
  Implemented directly by `calibration.spatial`'s `ScaleFactorPositionCalibration`.

`analyze_shot()` in `analysis_pipeline.py` is analysis's equivalent of `run_preprocessing()` — the
single per-shot orchestrating entry point:

1. `centroiding.extract_centroids()` loops over the frame's valid spectral columns and calls
   `IntensityWeightedMoment.estimate()` per column: an intensity-weighted first moment for the
   centroid `x0`, and the full three-term Thompson-Larson-Webb formula (via `noise_model.py`'s
   `SensorNoiseModel`, bundling `gain_e_per_adu` and `background_sigma`) for its uncertainty
   `sigma_x0`.
2. An optional `PositionCalibration` converts `(x0, sigma_x0)` from pixels to physical units.
3. The `WavelengthAxis` supplies `wavelength_nm`/`sigma_wavelength_nm` per column.
4. For each requested polynomial degree (default: linear), `TotalLeastSquaresFit` (via `scipy.odr`)
   fits centroid vs. wavelength, producing a `SpatialDispersionFitResult`. Its `.zeta(wavelength_nm)`
   (the polynomial derivative) and `.sigma_zeta()` (full-covariance error propagation) are the
   dispersion estimate itself — the result object *is* the interface, mirroring
   `calibration/spectral`'s pattern.

Combining many shots is a deliberately separate step, not folded into `analyze_shot()`:
`combination.combine_shots()` inverse-variance-weights each shot's ζ and estimates the run-to-run
scatter (`sigma_external`) via `block_bootstrap.py`'s autocorrelation-aware moving-block bootstrap
(block length chosen from the series' own measured autocorrelation) rather than naive scatter — real
multi-shot data showed lag-1 autocorrelation around 0.89, which would make a naive scatter estimate an
underestimate. The reported uncertainty is `max(sigma_internal, sigma_external)`.

`exceptions.py` defines `AnalysisError` and `InsufficientDataError`, raised only when fewer than
`degree + 2` columns are available to fit (`degree + 1` would be an exact interpolation with zero
residual degrees of freedom).

## GUI (`src/pipeline/gui/`)

Entry point: `python3 -m pipeline.gui.app` — builds a `QApplication` and shows `MainWindow`, a
`QStackedWidget` wiring three screens in sequence:

1. **`CalibrationScreen`** (shown first) — either loads existing artifacts from
   `calibration_artifacts/`, or walks the user through building new ones via modal dialogs in
   `calibration_dialogs.py` (baseline, flat-field, conversion-gain, spatial, spectral — bad-pixel-map
   has no dialog of its own, since it's derived automatically from flat-field). Both paths converge on
   emitting `calibration_ready` with a `CalibrationBundle` (a GUI-level wrapper around the same
   `CalibrationSet` `preprocessing/` and the CLI both use), which `MainWindow` uses to build one shared
   `CameraStream`.
2. **`LiveViewWidget`** — polls the camera at roughly 5 Hz, showing a live centroid-vs-wavelength
   scatter with fit overlay, a raw heatmap, and a rolling trend chart; flags if the live frame's
   settings drift from the loaded baseline's.
3. **`ExtendedMeasurementScreen`** (reached from LiveView) — runs a blocking N-shot measurement
   (default 20, up to 1000), combines shots via `combination.combine_shots()`, and can save a full
   record via `measurement_record.py`: a mean stack frame, first/middle/last individual shots,
   per-shot and combined fit results at every degree, the calibration artifacts in effect, the ROI
   used, and a journal-quality plot.

`roi_control.py`'s `SpatialROIControl` (bounds in mm) and `SpectralROIControl` (bounds in pixel
columns) are reusable crop widgets embedded in both LiveView and ExtendedMeasurement.

## CLI (`src/pipeline/cli/`)

`calibration.py` is a headless, argparse-based counterpart to the GUI's calibration dialogs — same
`build_*()`/`save_*()` functions, same default artifact paths. Invoked as
`python3 -m pipeline.cli.calibration <subcommand> [flags]`; subcommands are `baseline`, `flat-field`,
`bad-pixel-map`, `spatial`, `conversion-gain`, `noise-model`, `spectral-capture`, `spectral-manual`.
See `docs/user_guide.md` for usage.

## Configuration

`configs/default.yaml` holds camera settings (serial number, exposure, gain, pixel format, canonical
shape, pixel pitch) and preprocessing/spectrometer-geometry defaults (ROI bounds, grating lines/mm,
incidence angle, relay-lens focal length). Loaded via `pipeline.utils.helpers.load_config()`. Note
that `frame.py` reads this file at *import time* to derive `CANONICAL_*` — changing
`camera.pixel_format` or `camera.canonical_shape` changes what every frame in the pipeline validates
against.
