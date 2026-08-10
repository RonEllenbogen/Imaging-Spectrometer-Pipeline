# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Current status, active design decisions, and the project to-do list:
@docs/project_state.md. Historical rationale for `acquisition/` and
`preprocessing/` (design decisions, bugs found/fixed, deliberate cuts):
@docs/project_handover.md.

## Project

Software for an imaging spectrometer built to detect and quantify *spatial chirp* (correlation between
optical frequency and transverse beam position) in a Ti:sapphire laser system, for the Laser Plasma
Accelerator Group (LPAG) at the University of Oxford. It acquires frames from a Basler camera
(`a2A1920-51gmBAS`, via pypylon), corrects them, estimates spatial dispersion, and will eventually
present results live through a GUI.

Build order (see `docs/notes.md` for the full roadmap): acquisition → preprocessing → analysis →
validation → headless CLI integration → GUI → full integration. `acquisition/`, `preprocessing/`,
`calibration/sensor/`, `calibration/shared/`, `calibration/spatial/`, `calibration/spectral/`, and
`analysis/` are built and tested (synthetic data only, beyond `acquisition/`).
`gui/app.py` and `main.py` are still empty stubs or not yet started.

## Commands

Run everything from the repo root. The `pipeline` package (`src/pipeline`) is already installed in
editable mode, so `import pipeline` and pytest both work without any path hacks.

```bash
# Run the full test suite
python3 -m pytest

# Run one file / one test
python3 -m pytest tests/test_preprocessing.py
python3 -m pytest tests/test_preprocessing.py::test_name -q

# Run hardware-dependent tests too (camera must be connected and powered on;
# otherwise they're skipped automatically)
export SPECTROMETER_HARDWARE_TESTS=1
python3 -m pytest
```

`pyproject.toml` declares real install dependencies (`numpy`/`scipy`/`pyyaml`/`pypylon` as core,
`PySide6`/`pyqtgraph` under the `gui` extra, `pytest`/`pytest-qt` under `dev`) so `pip install -e .` works
end-to-end on a fresh checkout — it was an empty placeholder earlier in the project and is not to be
reverted to that state. There is still no configured lint/format step (no `ruff`/`black`/`mypy`) —
don't assume one is wired up unless you check first. `requirements.txt` remains an empty placeholder;
`pyproject.toml` is the real source of truth for dependencies.

## Git

Commits and pushes to this repo must be attributed to the `RonEllenbogen` GitHub account. That means
the commit author's (and committer's) git email must be one verified on that account
(`ron.ellenbogen84@gmail.com`) — GitHub links a commit to a profile by matching the git email against
the account's verified emails, not by matching the name string. A commit authored with an unverified or
machine-local email (e.g. a `user@hostname.local` address from an unconfigured `git config user.email`)
will show the right name on GitHub but won't link to the profile. Check `git config user.email` (and
`user.name`) resolve correctly before committing if there's any doubt.

## Architecture

### Layering and the canonical frame contract

`src/pipeline/acquisition/frame.py` defines `CANONICAL_SHAPE`, `CANONICAL_DTYPE`, and
`CANONICAL_MAX_VALUE`, derived at import time from `configs/default.yaml`. Every frame type in the
pipeline validates against these in `__post_init__`:

- `FrameData` (acquisition) — raw integer frame + capture metadata (`frame_id`, `timestamp`,
  `exposure_us`, `gain_db`). Immutable (`frozen=True, slots=True`); `__post_init__` also flips
  `image.flags.writeable = False` since `frozen` alone doesn't stop in-place array mutation.
- `ProcessedFrame` (preprocessing) — the float-image counterpart, used from baseline subtraction
  onward once values need to go negative before clipping. Same shape contract, relaxed dtype
  (any floating type instead of the strict integer `CANONICAL_DTYPE`).

Axis convention: axis 0 is spatial, axis 1 is spectral (`SPATIAL_AXIS`/`SPECTRAL_AXIS` in `frame.py`).

### Acquisition (`src/pipeline/acquisition/`)

`CameraStream` runs a background thread that continuously calls a `CameraBackend.grab_one()` and
publishes only the newest frame behind a lock (`get_latest_frame()` is a non-blocking read). It
tolerates up to `max_consecutive_timeouts` transient `CameraTimeoutError`s before treating the stream
as fatally broken and exiting; any other `CameraError` is fatal immediately; anything else (a real bug)
is deliberately left uncaught so it surfaces as a traceback instead of vanishing into `last_error`.

`CameraBackend` is a `Protocol` (structural typing, not inheritance) with two implementations:

- `PylonBackend` — wraps real pypylon/pylon SDK calls for the Basler camera. Construction never
  touches hardware; `connect()` opens the device, `configure()` applies settings and starts grabbing.
- `SyntheticBackend` — generates synthetic frames (Gaussian beam + injected `slope_px_per_col` chirp +
  noise), no hardware involved. This is what `CameraStream` should be constructed with in tests and
  what most of the test suite runs against.

Hardware-only tests are gated behind `SPECTROMETER_HARDWARE_TESTS=1` (see `HARDWARE_AVAILABLE` in
`tests/test_acquisition.py`) rather than always requiring a connected camera.

### Calibration (`src/pipeline/calibration/`)

Owns *building* calibration artifacts; `preprocessing/` owns *applying* them. This is a
one-directional dependency (`preprocessing/` imports from `calibration/`, never the reverse) —
enforced in practice by `calibration/sensor/saturation.py` depending only on `pipeline.acquisition`,
never on `preprocessing/`, even though both `calibration/sensor/flat_field.py` and
`preprocessing/`'s per-frame pipeline need a saturation check.

- `shared/io.py` — `save_artifact()`/`load_artifact()`: generic persistence of one or more named
  arrays plus one record to a single `.npz` file, `np.savez` only (never pickle, so loading an artifact
  never executes arbitrary code). Every subpackage's per-type `save_*()`/`load_*()` are thin wrappers
  around this. `shared/metadata.py`'s `CalibrationRecord` tags a frame-built artifact with the
  exposure/gain/timestamp/frame-count it was built under (`check_settings_match()` enforces a science
  frame's actual settings match one, within tolerance, before it's applied) — lives in `shared/`, not
  `sensor/`, since both `sensor/` and `spectral/` artifacts are frame-built and need it; `spatial/`'s
  scale factor is not frame-built and uses its own lighter `ScaleFactorRecord` instead (see below).
  `shared/fitting.py`'s `PolynomialFitter` protocol (default `TotalLeastSquaresFit`, via `scipy.odr`)
  and `shared/result.py`'s `PolynomialFitResult` generalize `analysis/dispersion_fitting.py`'s
  total-least-squares machinery to generic x/y, for `spectral/calibrate.py` to reuse — kept as a
  structurally separate implementation rather than importing `analysis/`'s, since `calibration/` and
  `analysis/` must not depend on each other in either direction (see `analysis/interfaces.py`).
- `sensor/` — each artifact type owns its own `build_*()`/`save_*()`/`load_*()`: `baseline.py`
  (per-session background average, returned as a `BaselineResult` bundling the averaged `baseline`
  with `background_sigma` — the median per-pixel sample standard deviation across the source frames,
  "b" in `analysis/noise_model.py`'s Thompson-Larson-Webb formula, measured for free from the same
  stacked frames the mean is built from; requires at least 2 frames, since a sample standard deviation
  is undefined at n=1), `flat_field.py` (PRNU correction from uniform-illumination frames,
  dark-subtracted first so DSNU isn't baked in; rejects a saturated source outright via
  `saturation.py`), `bad_pixel_map.py` (dead/hot pixels flagged from flat-field outliers beyond
  `SIGMA_THRESHOLD`), `conversion_gain.py` (`gain_e_per_adu`, the other `SensorNoiseModel` quantity —
  a photon transfer curve: uniform illumination at *fixed brightness*, swept across *exposure times*;
  variance at each level is computed temporally, across repeat frames, not spatially across pixels in
  one frame, so PRNU doesn't bias the gain low; `1/slope` of a `shared/fitting.py` linear fit of
  variance against mean, wrapped in `ConversionGainResult` alongside the full fit for diagnostics).
  `saturation.py`'s `check_saturation()` checks the *raw* frame against `CANONICAL_MAX_VALUE` before
  flat-field division changes the numeric domain, and returns a result rather than raising (the caller
  decides whether to discard/log/escalate). `workflow.py` is the "press start" layer gluing
  `acquisition/`'s `CameraStream.collect_n_frames()` to `build_*()`/`save_*()`: `run_baseline_calibration()`
  does acquire→build→save in one call (baseline is single-phase); flat-field calibration needs its
  physical setup changed mid-capture (dark, then uniformly illuminated), so it's split across
  `capture_dark_frames()`/`capture_illuminated_frames()`/`finish_flat_field_calibration()` instead of
  one blocking call — the caller (eventually `gui/`) sequences them at its own pace.
  `run_conversion_gain_calibration()` is the one exception to "never touches the stream's settings":
  since `CameraStream` can't change `exposure_us` while running, it repeatedly stops/reconfigures/
  restarts `camera_stream` itself while sweeping exposure (interrupting live view for the sweep's
  duration), restoring the original `exposure_us` afterward. The exposure range is caller-supplied
  (`exposure_min_us`/`exposure_max_us`/`n_levels`, linearly spaced), not auto-probed — deliberately, per
  `docs/project_state.md`.
- `spatial/` — pixel→physical-position conversion at the spectrometer's slit is a *fixed scale factor*
  (the ratio of the imaging spectrometer's two relay-lens focal lengths, `DEFAULT_SCALE_FACTOR = 1.5`
  in `calibrate.py`), not a per-point fit — no translation-stage measurement session exists in this
  codebase; the project's scope only needs displacement along the detector's spatial axis, and the
  focal-length ratio is precise enough on its own (misalignment shows up as blur/aberration, not a
  quantifiable scale-factor uncertainty). `ScaleFactorPositionCalibration` implements
  `analysis.interfaces.PositionCalibration` directly. `io.py`'s `load_scale_factor()` differs from
  every other `load_*()` in this package: a missing file returns `DEFAULT_SCALE_FACTOR` rather than
  raising `FileNotFoundError`, since (unlike a baseline or flat field) the scale factor always has a
  physically valid default. A GUI-entered manual override is persisted via `save_scale_factor()` and
  reused in future sessions.
- `spectral/` — pixel→wavelength calibration from a spectral-lamp image (reference lamp: Argon).
  `calibrate.py`'s `calibrate_spectral()` fits matched (pixel, wavelength_nm) pairs via
  `shared/fitting.py`; the returned `WavelengthCalibrationResult` implements
  `analysis.interfaces.WavelengthAxis` directly (`wavelength_nm()`/`sigma_wavelength_nm()` live on the
  result itself, the same "result IS the interface" pattern as `analysis/results.py`'s
  `SpatialDispersionFitResult.zeta()`) and reuses `CalibrationRecord` for provenance.
  `sigma_wavelength_nm()` propagates `coefficient_sigma` treating coefficients as uncorrelated — a
  documented approximation, flagged for review once real lamp data exists. `calibrate.py`'s
  `build_manual_spectral_calibration()` is a second path into the same result type, for a wavelength
  calibration measured entirely outside this codebase (e.g. via Pylon Viewer) — takes coefficients and
  their uncertainty directly, no fit involved. `grating_geometry.py` predicts relative pixel spacing
  between two wavelengths from the spectrometer's transmission-grating equation (config-driven
  constants in `configs/default.yaml`'s `spectrometer:` section); `reference_lines.py` loads Argon's
  known line wavelengths from `data/reference/oriel_spectral_calibration_lamps.csv`.
  `line_matching.py`'s `match_lines()` detects spectral peaks (1D collapse + `scipy.signal.find_peaks`
  + sub-pixel intensity-weighted refinement) and matches them to reference lines via a search over the
  geometry-predicted spacing pattern, raising a new `LineMatchingError` if too few peaks are found or no
  candidate identification scores well enough. `workflow.py`'s `run_spectral_calibration()` is fully
  wired (acquire → preprocess each frame individually via a caller-supplied `CalibrationSet` → average →
  `match_lines()` → `calibrate_spectral()` → save) and runs end-to-end.
- Exceptions derive from `CalibrationError` (see `exceptions.py`): `SettingsMismatchError`,
  `InvalidFlatFieldError`, `InsufficientDataError` (mirrors `analysis/exceptions.py`'s version, kept
  separate for the same no-cross-dependency reason as `shared/fitting.py`), `LineMatchingError`
  (spectral peak-to-reference-line matching failed). `pipeline.preprocessing`
  re-exports `SettingsMismatchError` for caller convenience, since `preprocessing/`'s `apply_baseline()`
  can raise it too — but it is a `CalibrationError`, not a `PreprocessingError`; catch both explicitly
  if a caller needs to handle anything either package can raise.

### Preprocessing (`src/pipeline/preprocessing/`)

`run_preprocessing()` in `preprocessing_pipeline.py` is the single public entry point and *encodes the
correction order in code, not documentation*: frame sanity check → saturation check → baseline
subtraction → flat-field division → bad-pixel masking → optional ROI masking. Callers pass a
pre-built `CalibrationSet` (baseline, flat field, bad-pixel mask + records); building those artifacts
is `calibration/`'s job, not this function's.

- `steps/` — each artifact type's `apply_*()` counterpart to its `calibration/sensor/` `build_*()`:
  `baseline.py` (subtracted with clipping at zero), `flat_field.py` (divided, floored at
  `MIN_FLAT_FIELD_VALUE` so division can never produce inf/nan), `bad_pixel_map.py` (zeroed rather
  than interpolated — exact for a weighted centroid, no bias risk near sharp gradients). `roi.py`
  zeroes rows outside the spatial ROI rather than cropping, keeping `CANONICAL_SHAPE` intact.
- `validation/frame_checks.py` — `check_frame_sanity()` only checks whether a raw frame has *any*
  signal (rejects all-zero frames); structural validity is already guaranteed by `FrameData`'s own
  constructor.
- Exceptions derive from `PreprocessingError` (see `exceptions.py`): `SaturationError`,
  `NoSignalError`. (`SettingsMismatchError`/`InvalidFlatFieldError` are `calibration/`'s — see above.)

### Config

`configs/default.yaml` holds camera settings (serial number, exposure, gain, pixel format, canonical
shape) and preprocessing defaults (ROI bounds). Loaded via `pipeline.utils.helpers.load_config()`.
Note `frame.py` loads this at *import time* to derive `CANONICAL_*` constants — changing
`configs/default.yaml`'s `camera.pixel_format`/`canonical_shape` changes what every frame in the
pipeline validates against.

## Conventions used throughout this codebase

- Every module is laid out under the same section comments, in order: `# Imports`, `# Constants`,
  `# Classes`, `# Functions`. Keep new code under the matching section rather than introducing new
  structure.
- Frame-like value objects are `@dataclass(frozen=True, slots=True)` and enforce their invariants in
  `__post_init__` (shape, dtype, non-negativity, finiteness), including explicitly locking any
  wrapped numpy array with `.flags.writeable = False`.
- Docstrings are NumPy-style (`Parameters`/`Returns`/`Raises` sections) and are used to explain *why*
  a function behaves the way it does (ordering constraints, what was deliberately left out, what's
  still an open empirical question) — read them before changing behavior in acquisition/preprocessing,
  they often record a design decision that isn't obvious from the code alone.
- Backends/interchangeable implementations use `typing.Protocol` (structural typing) rather than an
  abstract base class.
- Functions that detect an abnormal-but-recoverable condition (e.g. saturation) return a result object
  for the caller to act on instead of raising; functions that detect a genuinely invalid state (e.g. no
  signal at all, a settings mismatch) raise a specific exception.
- Test files mirror `src/pipeline/<package>/` one-to-one (e.g. `tests/test_preprocessing.py`), and
  default to synthetic data/backends so the suite runs with no hardware attached.
- No mention of Claude, AI, agents/agentic, Anthropic, Copilot, or similar AI-tooling keywords anywhere
  in this repository except `CLAUDE.md` and `docs/project_handover.md` — not in source files,
  docstrings, comments, other docs, config, or anything else. This applies just as strictly to anything 
  that makes it to GitHub without living in the repo itself — commit messages, PR descriptions, issue 
  text, code review comments. In particular, never add a `Co-Authored-By: Claude` (or similar) trailer
  to a commit, and don't reference AI assistance in commit bodies.
