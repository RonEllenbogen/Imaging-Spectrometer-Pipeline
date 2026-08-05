# Project Handover — Imaging Spectrometer Pipeline

Detailed companion to `CLAUDE.md`. Written to bring a fresh Claude Code
session up to speed on everything already decided, built, tested, found, and
deliberately cut — so none of it needs re-deriving or accidentally reversing.

---

## 1. Project goal

Design, build, and characterise a 4f imaging spectrometer measuring spatial
chirp (ζ = ∂x₀/∂ω — how a beam's transverse centroid position shifts with
wavelength) in a Ti:sapphire laser system, as an 8-week Durham MPhys placement
at OPAL/John Adams Institute. Camera: Basler ace 2 a2A1920-51gmBAS (Sony
IMX392, GigE Vision, 1920×1200 px, 3.45µm pitch, 8/10/12-bit selectable,
currently run at PixelFormat=Mono8 for full 51fps without needing jumbo
frames). Optical design already completed separately (Optiland ray-trace
model, 600 l/mm VPH grating, folded 4f "Option C" geometry) — not part of this
codebase.

---

## 2. Repository structure

```
Imaging-Spectrometer-Pipeline/
├── configs/default.yaml
├── data/{raw,interim,processed}/
├── docs/{architecture.md, notes.md}
├── src/pipeline/
│   ├── acquisition/        COMPLETE, tested (synthetic + real hardware)
│   ├── preprocessing/      COMPLETE, tested (synthetic only)
│   ├── calibration/        DESIGNED, not built
│   ├── analysis/           NOT STARTED
│   ├── gui/                NOT STARTED
│   └── utils/
├── tests/
└── scripts/
```

---

## 3. `acquisition/` — complete, fully tested

### Files
- `exceptions.py` — `CameraError` base; `CameraConnectionError`,
  `CameraTimeoutError` (carries `timeout_ms`), `CameraConfigurationError`
  (carries `parameter`, `value`, `reason`).
- `frame.py` — `FrameData` (frozen dataclass: `image`, `timestamp`
  [`time.monotonic()`], `frame_id`, `exposure_us`, `gain_db`). Validates shape
  == `CANONICAL_SHAPE`, dtype == `CANONICAL_DTYPE`, `frame_id >= 0`,
  `exposure_us > 0`, `gain_db` finite. Sets `image.flags.writeable = False`.
  `frozen=True, slots=True, eq=False` (eq=False because numpy array
  comparison would otherwise raise/behave wrong in a dataclass-generated
  `__eq__`). Also defines `CANONICAL_SHAPE`, `CANONICAL_DTYPE`,
  `CANONICAL_MAX_VALUE`, `SPATIAL_AXIS=0`, `SPECTRAL_AXIS=1`, all derived from
  `configs/default.yaml` via `pixel_formats.py`.
- `pixel_formats.py` — `PIXEL_FORMAT_INFO` dict mapping `"Mono8"/"Mono10"/
  "Mono12"/"Mono16"` → `(numpy dtype, true max value)`. `dtype_for_pixel_format()`,
  `max_value_for_pixel_format()`. Critical: Mono10/Mono12 share the `uint16`
  container but have TRUE ceilings of 1023/4095, not 65535 — always use
  `max_value_for_pixel_format()`, never `np.iinfo(dtype).max`, when clipping.
- `backends.py` — `CameraBackend` (a `typing.Protocol`, not ABC — structural
  typing, no inheritance needed since there's no shared implementation to
  inherit). `SyntheticBackend` (generates a Gaussian beam with injectable
  `slope_px_per_col` for known-ground-truth testing, `noise_std`,
  `timeout_probability` for testing failure paths). `PylonBackend` (real
  hardware: `connect()` finds device by `serial_number` via
  `EnumerateDevices()`, NOT `CreateFirstDevice()` — multiple devices exist on
  the network. `configure()` sets PixelFormat/Gain always, and either
  `ExposureAuto="Once"` convergence or manual `ExposureTime`, ending with
  `StartGrabbing(GrabStrategy_LatestImageOnly)`. `grab_one()` uses
  `RetrieveResult()` + `.Array.copy()` + `.Release()` — copying is mandatory,
  pylon's buffer is reclaimed on Release. `close()` resets `_configured` flag
  too, not just the camera handle, or a post-close `grab_one()` won't
  correctly raise).
- `camera.py` — `CameraStream`: wraps a `CameraBackend` in a background
  `threading.Thread`. Lock + plain variable for frame hand-off (NOT
  `queue.Queue` — considered and rejected, see §6). `start()`/`stop()` are
  idempotent no-ops if called out of order. `stop()` checks
  `self._thread is None`, NOT `self.is_running` — a thread that died on its
  own from a fatal error still needs `stop()` to close the backend (see bug
  list below). `last_error` property surfaces why the thread died.
  `max_consecutive_timeouts` (default 5) before a timeout becomes fatal.
  Requires `serial_number` (forwarded to a default-constructed
  `PylonBackend`) since `PylonBackend()` can no longer be built with zero
  arguments.

### Key design decisions worth knowing
- Thread buffer: Lock + variable chosen over `Queue` because `Queue.get()` is
  destructive — would require a redundant cache to make repeated
  `get_latest_frame()` calls return the same frame, defeating the point of
  using `Queue` at all.
- `PylonBackend`'s `auto_exposure`/`auto_timeout` are constructor-only args,
  deliberately NOT part of the shared `CameraBackend` Protocol (no
  `SyntheticBackend` equivalent exists).

### Bugs found and fixed during development (do not reintroduce)
1. `CameraTimeoutError.__init__` was typo'd `__innit__` — silently became a
   dead method, `.timeout_ms` attribute never set.
2. `SyntheticBackend.grab_one()`'s saturation clip used
   `np.iinfo(self._dtype).max` (container ceiling) instead of the true
   per-format ceiling — silently let Mono10/Mono12 values exceed their real
   bit depth.
3. `CameraStream.stop()` originally checked `self.is_running` before
   join/close — meant a thread that died naturally from a fatal error would
   never get its backend closed by a subsequent `stop()` call, since
   `is_running` was already `False`.
4. `PylonBackend.configure()`'s `self._configured = True` was mis-indented
   inside the `else:` branch — meant the `auto_exposure=True` path never
   marked the backend configured, so `grab_one()` always refused with
   "called before connect()/configure()" for that path specifically.

### Test files
`tests/test_acquisition.py` (backend contract + lifecycle, parametrized
across SyntheticBackend/PylonBackend via `SPECTROMETER_HARDWARE_TESTS=1`;
35 tests, all passing against real hardware). `tests/test_camera_stream.py`
(14 tests, no hardware needed — CameraStream logic proven entirely against
SyntheticBackend).

---

## 4. `preprocessing/` — complete, synthetic-tested only

Contract: raw `FrameData` + calibration artifacts in → cleaned `ProcessedFrame`
out. **No real calibration data has been captured yet** — every threshold
below is an unverified placeholder pending real baseline/flat-field captures.

### Files
- `exceptions.py` — `PreprocessingError` base; `SettingsMismatchError`
  (hard failure, always — the one thing this package never downgrades to a
  warning), `SaturationError` (now unused internally — see below),
  `InvalidFlatFieldError`, `NoSignalError`.
- `processed_frame.py` — `ProcessedFrame`: float-image counterpart to
  `FrameData`. Same shape/metadata contract, but validates
  `np.issubdtype(dtype, np.floating)` instead of an exact integer dtype.
  Exists specifically because baseline subtraction and flat-field division
  both require float precision `FrameData` can't hold.
- `sensor_calibration/metadata.py` — `CalibrationRecord` (`exposure_us`,
  `gain_db`, `timestamp` [`time.time()`, NOT `time.monotonic()` — records
  must survive across process restarts], `source_frame_count`).
  `check_settings_match(frame, record)` — tolerance-based (1% relative
  exposure, 0.05dB absolute gain — unverified placeholders), raises
  `SettingsMismatchError` on mismatch.
- `validation/frame_checks.py` — `check_frame_sanity(frame)`: raises
  `NoSignalError` if `frame.image.max() == 0`. That's the ONLY check here —
  shape/dtype/bit-depth are already guaranteed by `FrameData`'s own
  construction (it's the only place `FrameData` is ever built), and a "no
  NaN" check is architecturally impossible since `CANONICAL_DTYPE` is always
  an integer type. Do NOT call this on background/baseline capture frames —
  near-zero signal there is correct, not an error.
- `steps/saturation.py` — `check_saturation(frame, bad_pixel_mask=None)` →
  `SaturationCheckResult` (`is_saturated`, `peak_value`, `n_saturated_pixels`,
  `threshold`). **Returns, does not raise** — caller decides whether to
  discard/log/escalate. Runs on the RAW frame, early in the pipeline (NOT at
  handover's originally-described "step 6" position) — flat-field division
  changes the numeric domain (float, no longer bounded by
  `CANONICAL_MAX_VALUE`), so a late check on corrected data wouldn't detect
  genuine ADC clipping at all.
- `steps/roi.py` — `apply_roi(frame: ProcessedFrame, row_min, row_max) →
  ProcessedFrame`. **Masks (zeroes) rather than crops** — a true crop would
  produce a smaller-than-`CANONICAL_SHAPE` array incompatible with
  `ProcessedFrame`'s validation, and masking is mathematically exact for a
  weighted centroid anyway (zeroed pixel contributes nothing to numerator or
  denominator). Row indices stay absolute, never offset.
  **`linearity.py` was deliberately cut** — only two EMVA summary bounds
  exist (±0.376%/+0.598%), not a real correction curve; not enough to build
  an actual correction. **ROI's necessity for this setup is still an open,
  unresolved question** — pending a real background-frame histogram check
  for genuine zero-clipping bias (see §6). `run_preprocessing()` treats it as
  fully optional (`roi_bounds=None` skips it).
- `sensor_calibration/baseline.py` — `build_baseline(frames) → (baseline:
  float ndarray, CalibrationRecord)`: averages N background frames, REQUIRES
  identical exposure_us/gain_db across all input frames (raises otherwise).
  `apply_baseline(frame, baseline, record) → ProcessedFrame`: calls
  `check_settings_match()` first (hard requirement — dark current genuinely
  scales with exposure/gain), subtracts, clips at 0.
- `sensor_calibration/flat_field.py` — `build_flat_field(illuminated_frames,
  dark_frames)`: dark-subtracts illuminated frames (using `build_baseline()`
  internally — reused, not reimplemented) BEFORE normalizing to mean=1.0
  (PRNU is multiplicative, DSNU is additive — normalizing without
  dark-subtracting first would contaminate the flat field with the wrong
  kind of correction). Raises `InvalidFlatFieldError` and rejects the WHOLE
  build if any illuminated source frame is saturated (a user decision — not
  silently excluding bad frames). `apply_flat_field()` divides by the flat
  field, clipped at `MIN_FLAT_FIELD_VALUE=0.01` to guarantee no inf/nan
  regardless of whether bad-pixel masking has run yet. **Deliberately does
  NOT check settings match** — PRNU treated as exposure/gain-independent
  within the linear regime, unlike baseline.
- `sensor_calibration/bad_pixel_map.py` — `build_bad_pixel_map(flat_field,
  record)`: flags pixels > `SIGMA_THRESHOLD=5.0` standard deviations from the
  flat field's mean. **Uses std, NOT MAD** — MAD (median absolute deviation)
  is structurally blind to sparse outliers, since the median of deviations is
  exactly zero whenever fewer than half the population deviates (true by
  definition for rare bad pixels). This was a real bug caught by an actual
  test failure, not just theoretical — see bug list. `apply_bad_pixel_map()`
  masks (zeroes) flagged pixels — chosen over interpolation specifically
  because interpolation risks bias near a sharp gradient (the beam's edge is
  exactly that), where masking is exact for a weighted centroid and
  interpolation isn't.
- `preprocessing_pipeline.py` — `CalibrationSet` (bundles baseline+record,
  flat_field+record, bad_pixel_mask — no separate bad-pixel record, nothing
  consumes it). `run_preprocessing(frame, calibration, roi_bounds=None) →
  (ProcessedFrame, SaturationCheckResult)`. Order: sanity check → saturation
  check (raw) → baseline subtract → flat-field divide → bad-pixel mask →
  optional ROI mask. This order is NOT arbitrary: baseline-before-flat-field
  is mathematically required (undoing `offset + gain×signal` must happen in
  reverse — subtract offset, then divide by gain); bad-pixel masking's exact
  position among the later steps is NOT load-bearing given masking-to-zero
  is a fixed point under both subtraction-then-clip and division.

### Deliberately cut / deferred (do not silently re-add without a new reason)
- `linearity.py` / any per-pixel linearity correction — insufficient source
  data (only 2 summary bounds, not a curve).
- Dark-frame-specific hot-pixel detection (separate from the flat-field-
  derived check) — deferred until real data shows the flat-field-derived
  check alone is inadequate.
- `PRNU_FRACTION`/SNR-adequacy check on flat fields — user explicitly decided
  a flat field should be used regardless of measured SNR.
- Calibration staleness thresholds that RAISE — user explicitly chose
  log-the-age-only, no hard failure, given the thresholds were unverified
  guesses with real cost if wrong in either direction.
- Disk persistence for `CalibrationSet` artifacts — explicitly deferred,
  in-memory only for now.
- A whole standalone `config.py` for preprocessing thresholds — was written,
  then almost entirely deleted after discovering most constants had no real
  consumer (see extensive back-and-forth; the few survivors are inline
  constants in the one file that actually uses each one).

### Test file
`tests/test_preprocessing.py` — 43 tests, single file (deliberately, despite
covering many source files), all synthetic data with known injected
ground truth. Notably includes an end-to-end pipeline test that builds real
calibration artifacts via the actual `build_*()` functions (not
hand-constructed) and hand-verifies the expected recovered signal.

---

## 5. `calibration/` — designed, not yet built

Split into two independent subpackages under `src/pipeline/calibration/`,
sharing a `shared/fitting.py` (generic weighted polynomial fit + residual
validation) and `shared/result.py`:

- **`spectral/`** — pixel(column)→wavelength calibration from a spectral
  lamp image. Genuinely hard step: matching detected line peaks to known
  reference wavelengths (`line_matching.py`) — tractable because an
  approximate prior dispersion already exists from the grating-equation
  design work, turning blind matching into nearest-neighbor + iterative
  outlier rejection.
- **`spatial/`** — pixel(row)→physical position calibration, via a laser on
  a translation stage, measuring pixel displacement per known physical
  displacement at the slit. This measures the RELAY OPTICS' MAGNIFICATION —
  camera pixel pitch (3.45µm) alone is NOT sufficient, since it only converts
  pixels to distance AT THE DETECTOR, not at the slit/input plane where the
  physically meaningful spatial chirp lives. Designed to be agnostic to
  whether the translation stage is manual or motorized (undecided) — a
  `SpatialCalibrationSession.add_point(known_position, centroid_px,
  uncertainty_px)` API accepts a plain float regardless of source.
  No `line_matching.py` equivalent needed — each point's true position is
  known by construction (whatever the stage was set to), no ambiguity to
  resolve.

Neither subpackage has been started. `preprocessing/`'s public API needed to
be stable before `calibration/spectral/calibrate.py` could be finished — it
now is.

---

## 6. Open / unresolved items

- **ROI necessity**: unresolved. Plan is to capture a real background frame,
  check its histogram for a pile-up at zero (evidence of real ADC-level
  clipping bias), and decide based on that rather than the synthetic model's
  simplified noise assumptions.
- **All preprocessing thresholds are placeholders**: `SIGMA_THRESHOLD=5.0`,
  `EXPOSURE_MATCH_TOLERANCE_REL=0.01`, `GAIN_MATCH_TOLERANCE_ABS=0.05`,
  `MIN_FLAT_FIELD_VALUE=0.01` — all need tuning against real calibration
  captures once they exist.
- **Frame averaging strategy for N-shot measurements**: resolved at the
  concept level, NOT yet implemented — decided to analyze each shot
  independently through the full pipeline, THEN combine the resulting
  (ζ, σζ) pairs via inverse-variance weighting (not average raw frames
  first, and not per-column combination for now — chosen as the pragmatic
  starting point, reusing existing single-shot code with zero new
  combination logic). This decision belongs to whatever orchestrates
  `analysis/`, not to `preprocessing/` itself.
- ~~**Batch/live dual capture**: `CameraStream.collect_n_frames(n)`~~ **Done.**
  Implemented in `acquisition/camera.py` as designed — polls
  `get_latest_frame()`, keeping only frames whose `frame_id` hasn't been
  seen yet, so live view and batch/calibration capture can share one
  running `CameraStream` (GigE's one-connection-per-camera limit) without
  double-counting a slow-arriving frame. Raises `RuntimeError` if called
  before `start()` or if the stream stops without a recorded error while
  waiting, and re-raises `last_error` if the background thread dies from a
  genuine `CameraError` mid-collection. Tested in
  `tests/test_camera_stream.py::TestCollectNFrames` (synthetic only).
- **`analysis/`**: not started. Will need centroid extraction (intensity-
  weighted, per spectral column) with Thompson-Larson-Webb-style uncertainty,
  and a weighted linear fit for ζ = dx0/dω with proper λ→ω chain-rule
  uncertainty propagation. A known, real bias exists if centroiding is done
  over the full spatial axis without background subtraction/ROI restriction
  — demonstrated directly with synthetic data (clipped, non-negative sensor
  noise has nonzero mean, biasing centroids toward the frame's geometric
  center).
- **`gui/`**: not started. Established that live view and batch/calibration
  workflows should reuse a single running `CameraStream` via polling, not
  separate connections. Spectral calibration is a genuine single-button
  "run silently, confirm when done" workflow; spatial calibration (manual
  stage) is inherently a multi-step, human-paced interaction — these need
  different GUI patterns, not one generic "calibrate" button.