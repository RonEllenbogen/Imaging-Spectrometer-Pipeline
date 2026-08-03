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
validation → headless CLI integration → GUI → full integration. `acquisition/` and `preprocessing/`
are built and tested; `analysis/`, `gui/app.py`, and `main.py` are still empty stubs.

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

There is no configured lint/format/build step (`pyproject.toml`, `requirements.txt` are currently
empty placeholders) — don't assume `ruff`/`black`/`mypy` are wired up unless you check first.

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

### Preprocessing (`src/pipeline/preprocessing/`)

`run_preprocessing()` in `preprocessing_pipeline.py` is the single public entry point and *encodes the
correction order in code, not documentation*: frame sanity check → saturation check → baseline
subtraction → flat-field division → bad-pixel masking → optional ROI masking. Callers pass a
pre-built `CalibrationSet` (baseline, flat field, bad-pixel mask + records); building those artifacts
is not this function's job.

- `sensor_calibration/` — each artifact type owns its own `build_*()`/`apply_*()` pair:
  `baseline.py` (per-session background average, subtracted with clipping at zero),
  `flat_field.py` (PRNU correction from uniform-illumination frames, dark-subtracted first so DSNU
  isn't baked in), `bad_pixel_map.py` (dead/hot pixels flagged from flat-field outliers beyond
  `SIGMA_THRESHOLD`, zeroed rather than interpolated — exact for a weighted centroid, no bias risk
  near sharp gradients). `metadata.py`'s `CalibrationRecord` tags every artifact with the
  exposure/gain it was built under; `check_settings_match()` enforces that a science frame's actual
  settings match (relative tolerance for exposure, absolute for gain) before an artifact is applied.
- `steps/` — `saturation.py` checks the *raw* frame against `CANONICAL_MAX_VALUE` before flat-field
  division changes the numeric domain, and returns a result rather than raising (the caller decides
  whether to discard/log/escalate). `roi.py` zeroes rows outside the spatial ROI rather than cropping,
  keeping `CANONICAL_SHAPE` intact.
- `validation/frame_checks.py` — `check_frame_sanity()` only checks whether a raw frame has *any*
  signal (rejects all-zero frames); structural validity is already guaranteed by `FrameData`'s own
  constructor.
- Exceptions all derive from `PreprocessingError` (see `exceptions.py`); catch that broadly, or a
  specific subclass (`SettingsMismatchError`, `SaturationError`, `InvalidFlatFieldError`,
  `NoSignalError`) to act differently per failure.

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
