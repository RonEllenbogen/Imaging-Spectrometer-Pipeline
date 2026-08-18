# User Guide

Practical instructions for installing the software, calibrating the instrument, and taking a
measurement — either through the GUI or headlessly through the CLI. For how the software is built,
see `architecture.md`.

## Installation

From the repo root, with a Python environment active:

```bash
pip install -e .            # core: numpy, scipy, pyyaml, pypylon
pip install -e ".[gui]"     # + PySide6, pyqtgraph, for the GUI
pip install -e ".[dev]"     # + pytest, pytest-qt, for the test suite
```

`configs/default.yaml` holds the camera's serial number and default exposure/gain — check it matches
your hardware before connecting. The camera is the Basler `a2A1920-51gmBAS`, expected at a static IP
on the local network (see `notes.md` for the specific network settings used during development);
pylon/pypylon must be installed and able to see the camera (`pylon Viewer` is the easiest way to
confirm connectivity first).

## Running on the lab PC

Lab PC `JAIWTDAQ02` already has a conda environment named `tango` with all dependencies
installed and the camera network-configured, so day-to-day use doesn't need a fresh `pip install` —
just activate the environment, pull the latest code, and launch the GUI:

```bash
conda activate tango
cd Imaging-Spectrometer-Pipeline
git fetch
git pull
python -m pipeline.gui.app
```

`git pull` only updates tracked source files; `calibration_artifacts/` is git-ignored, so any
baseline/flat-field/spectral calibrations already saved on this machine are untouched by it. If
`git pull` ever reports local changes in the way, check `git status` before discarding anything — it
likely means an edit was made directly on this machine rather than through a worktree elsewhere.

## Running the test suite

```bash
python3 -m pytest                              # synthetic backend only, no hardware needed
SPECTROMETER_HARDWARE_TESTS=1 python3 -m pytest # also runs hardware-dependent tests (camera required)
```

## Using the GUI

Launch it with:

```bash
python3 -m pipeline.gui.app
```

This opens a 1400×900 window that walks through three screens in sequence.

### 1. Calibration screen

On first launch you have two options:

- **Load Existing Calibrations** — reads whatever artifacts already exist under
  `calibration_artifacts/` (the CLI and GUI share this directory and file layout). If everything
  needed is present, this proceeds straight to live view.
- **Create new** — a card per artifact type, each opening a modal dialog that drives the camera
  through the same acquisition/build/save steps the CLI's subcommands use. Build them in this order,
  since each later step either depends on or preprocesses through the earlier ones:

  1. **Baseline** — background frames with the beam blocked/lamp off.
  2. **Flat field** — dark frames, then uniformly-illuminated frames; the bad-pixel map is derived
     from this automatically (no separate dialog).
  3. **Conversion gain** — a fixed-brightness, swept-exposure sequence.
  4. **Spatial** — enter a manual scale-factor override, or accept the default
     (relay-lens focal-length ratio).
  5. **Baseline, recomputed** — the Argon lamp needs its own exposure/gain, different from the live
     beam's, so re-run baseline at the lamp's settings before capturing it (see
     [Calibration consistency](#calibration-consistency)).
  6. **Spectral** — Argon lamp frames, fit into a pixel→wavelength calibration (also builds the
     geometric-tilt correction from the same frames).
  7. **Baseline, recomputed again** — back to the live beam's exposure/gain before continuing to live
     view, for the same reason as step 5.

  Click **Continue to Main Window** once you're satisfied; this emits the same `calibration_ready`
  signal as the "load" path.

### 2. Live view

Shows the camera at roughly 2 Hz: a live centroid-vs-wavelength scatter with the current fit overlaid,
the raw frame as a heatmap, and a rolling trend chart. If the live frame's exposure/gain drifts from
what the baseline was built under, this is flagged on screen — rebuild the baseline (and flat field)
rather than trusting a mismatched correction. Use the ROI controls
(`SpatialROIControl`/`SpectralROIControl`) here to crop to the beam's actual footprint. A button here
opens extended measurement.

### 3. Extended measurement

Runs a blocking N-shot acquisition (default 20, up to 1000), combines the shots into one
inverse-variance-weighted ζ with a correlation-aware uncertainty, and displays the result. **Save
Record** persists everything needed to reconstruct the measurement later: a mean stack frame,
first/middle/last individual raw shots, per-shot and combined fits at every requested polynomial
degree, the calibration artifacts that were in effect, the ROI used, and a journal-quality plot.

### Reviewing the GUI without hardware

Two scripts under `scripts/` open the real GUI with synthetic data, useful for UI review or when no
camera is attached:

```bash
python3 scripts/demo_app.py         # full MainWindow with a placeholder calibration bundle
python3 scripts/demo_live_view.py   # CalibrationScreen + LiveView as separate windows
```

## Calibration consistency

The calibration artifacts aren't interchangeable across arbitrary sessions — several of them are
checked for consistency, either against the live camera or against each other, and a mismatch raises
`SettingsMismatchError` rather than silently applying a wrong correction:

- **Baseline vs. the live frame** — every frame run through `run_preprocessing()` has its exposure
  and gain checked against the loaded baseline's `CalibrationRecord`, to within 1% (exposure) and
  0.05 dB (gain). This is strict: **if you change exposure or gain on the camera, the baseline (and
  therefore the whole loaded calibration set) is no longer valid and must be rebuilt** before you can
  process another frame. The same check applies to lamp frames during spectral calibration.
- **Flat field's dark vs. illuminated phases** — the two capture phases of a single flat-field session
  must match each other's settings exactly, but the flat field itself is *not* re-checked against the
  live frame's exposure/gain when applied (PRNU is treated as exposure/gain-independent within the
  camera's linear regime).
- **Conversion gain vs. baseline** — building a `SensorNoiseModel` (e.g. via the CLI's `noise-model`
  command) requires the loaded conversion-gain and baseline artifacts to share the same `gain_db`
  (same 0.05 dB tolerance as above); conversion-gain has no single exposure to compare, since exposure
  is the swept variable in that measurement.
- **Spatial and bad-pixel-map** are exempt from all of this — the scale factor isn't built from a
  captured frame at any particular setting, and the bad-pixel map is derived from the flat field
  rather than re-checked independently.

Practically: changing **gain** on the camera means rebuilding baseline, flat field, and conversion
gain together (gain feeds all three consistency checks above); changing only **exposure** means
rebuilding just the baseline (and, if you also want the geometric-tilt/wavelength calibration to stay
valid against the new setting, the spectral calibration too).

**Rebuild baseline every session, and spectral calibration after any physical adjustment to the
spectrometer.** The checks above only catch an exposure/gain mismatch — they can't detect a change
to the optical setup itself:

- **Baseline** — rebuild it at the start of every session, even if exposure/gain are unchanged.
  Background level can drift session to session (ambient light, thermal drift), and nothing enforces
  a maximum calibration age, so this is on the operator, not the software.
- **Spectral calibration** — the pixel→wavelength mapping is only valid for the spectrometer's
  physical alignment at the time it was built. Rebuild it any time something inside the instrument is
  adjusted — slit width changed, grating rotated, a mirror's orientation changed, and so on. None of
  these change `exposure_us`/`gain_db`, so `SettingsMismatchError` won't fire and won't warn you —
  it's on the operator to know to recapture.

## Using the CLI

`src/pipeline/cli/calibration.py` is a headless equivalent of the calibration dialogs — same
`build_*()`/`save_*()` functions and default artifact paths (`calibration_artifacts/<name>.npz`,
overridable per-command with `--path`/`--output-dir`). Every command that acquires frames requires
`--gain-db` and exactly one of `--auto-exposure` / `--exposure-us`.

```bash
# Baseline: 50 background frames (default), fixed exposure
python3 -m pipeline.cli.calibration baseline --gain-db 0 --exposure-us 100

# Flat field: interactive dark/illuminated prompts
python3 -m pipeline.cli.calibration flat-field --gain-db 0 --auto-exposure

# Bad-pixel map, derived from a saved flat-field artifact
python3 -m pipeline.cli.calibration bad-pixel-map

# Spatial: report the currently active scale factor
python3 -m pipeline.cli.calibration spatial
# ...or set a manually-measured override
python3 -m pipeline.cli.calibration spatial --scale-factor 1.52 --sigma-scale-factor 0.01

# Conversion gain: sweep exposure at fixed gain
python3 -m pipeline.cli.calibration conversion-gain --gain-db 0 \
    --exposure-min-us 50 --exposure-max-us 500 --n-levels 8 --n-frames-per-level 20

# Print the SensorNoiseModel built from saved baseline + conversion-gain artifacts
python3 -m pipeline.cli.calibration noise-model

# Spectral, from a live Argon lamp capture (also builds geometric-tilt correction)
python3 -m pipeline.cli.calibration spectral-capture --gain-db 0 --exposure-us 200

# Spectral, entering coefficients measured externally (e.g. via Pylon Viewer), bypassing capture
python3 -m pipeline.cli.calibration spectral-manual \
    --coefficients 400.0 0.05 --coefficient-sigma 0.5 0.001
```

`--n-frames` (default 50) controls averaging depth for `baseline`/`flat-field`/`spectral-capture`;
`--degree` (default 1) controls the pixel→wavelength polynomial degree for `spectral-capture`.

## Offline / diagnostic scripts

The rest of `scripts/` are one-off tools used during instrument bring-up and calibration validation,
not part of the normal operate-the-instrument flow:

- `camera_testing.py` — minimal single-frame grab, for checking camera connectivity.
- `plot_raw_image.py` — plots one raw captured image.
- `visualise_synthetic_frame.py` — sanity-checks `SyntheticBackend` output against `analysis/`.
- `analyze_raw_shot.py` — runs one manually-captured raw frame through `analysis/` only (no
  calibration applied).
- `build_geometric_tilt_calibration.py` / `build_spectral_calibration.py` — build and save those
  artifacts from already-captured lamp frames on disk, as an offline alternative to
  `spectral-capture`.
- `measure_spectrometer_tilt.py` / `measure_spatial_dispersion.py` — one-off physical measurements
  against real captured data.
- `compare_geometric_tilt_methods.py` / `compare_spectral_calibration_degrees.py` — diagnostic
  comparisons across fitting methods/polynomial degrees.
- `plot_beam_spectrum.py`, `plot_column_spectrum.py`, `plot_geometric_tilt_correction_images.py`,
  `plot_geometric_tilt_correction_beam_image.py` — diagnostic plots against real captured data.
- `save_tilt_diagnostic_frames.py` — grabs live frames and saves raw/corrected variants for
  inspection.

`scripts/run_app.py` and `scripts/capture_sample_data.py` are currently empty stubs — use
`python3 -m pipeline.gui.app` to launch the GUI, not `run_app.py`.

## Troubleshooting

- **"Settings mismatch" errors** — a science/lamp frame's exposure or gain no longer matches what a
  loaded baseline (or flat field) artifact was built under. Rebuild the affected calibration at the
  current settings, or restore the original settings.
- **Saturation warnings** — reduce exposure or gain and recapture; `check_saturation()` reports rather
  than blocks, so a saturated frame can still be processed, but the resulting centroid/ζ will be
  biased.
- **No signal / all-zero frame** — check the beam is present and not blocked, and that exposure isn't
  set so low the signal is below the noise floor.
