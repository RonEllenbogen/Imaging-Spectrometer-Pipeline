"""
Modal dialogs used by calibration_screen.py's "create new calibration"
flow -- one per calibration type that needs its own form or step sequence,
plus a shared error dialog for camera-connection and calibration-specific
failures.

Every dialog's Start/Continue/Save path is wired to the real
calibration/sensor/, calibration/spatial/, calibration/spectral/
functions referenced in their docstrings below, mirroring
src/pipeline/cli/calibration.py's own call sequence for each calibration
type -- build_camera_stream() + start()/stop() bracketing the acquisition
calls, real build_*()/save_*() underneath, no placeholder state left in
any accept path. CameraError (and subclasses) route to
show_camera_error_dialog(); SettingsMismatchError/InvalidFlatFieldError/
InvalidConversionGainError/NoSignalError/LineMatchingError route to
show_calibration_error_dialog() -- both leave the originating dialog open
(no accept()/reject()) so the user can correct inputs and retry, per
their own docstrings below.

Kept separate from calibration_screen.py so the "which forms/dialogs
exist for which calibration type" concern doesn't get lost inside the
top-level screen's page-navigation code.
"""

# Imports

import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pipeline.acquisition import CameraError, CameraStream, FrameData
from pipeline.calibration.exceptions import (
    InvalidConversionGainError,
    InvalidFlatFieldError,
    LineMatchingError,
    SettingsMismatchError,
)
from pipeline.calibration.sensor import (
    build_bad_pixel_map,
    capture_dark_frames,
    capture_illuminated_frames,
    finish_flat_field_calibration,
    load_bad_pixel_map,
    load_baseline,
    load_conversion_gain,
    load_flat_field,
    run_baseline_calibration,
    run_conversion_gain_calibration,
    save_bad_pixel_map,
)
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spatial import ScaleFactorPositionCalibration, save_scale_factor
from pipeline.calibration.spectral import (
    build_manual_spectral_calibration,
    run_spectral_calibration,
    save_spectral_calibration,
)
from pipeline.cli.calibration import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BAD_PIXEL_MAP_FILENAME,
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_CONVERSION_GAIN_FILENAME,
    DEFAULT_FLAT_FIELD_FILENAME,
    DEFAULT_GEOMETRIC_TILT_FILENAME,
    DEFAULT_SCALE_FACTOR_FILENAME,
    DEFAULT_SPECTRAL_FILENAME,
    MANUAL_SPECTRAL_EXPOSURE_US,
    MANUAL_SPECTRAL_GAIN_DB,
    build_camera_stream,
)
from pipeline.preprocessing import CalibrationSet, NoSignalError
from pipeline.gui.live_view import DEGREE_CHOICES, DEGREE_LABELS
from pipeline.gui.theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_ALT,
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_ERROR,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    MARGIN_DEFAULT,
    SPACING_MEDIUM,
    SPACING_SMALL,
    load_bundled_font,
)

# Constants

DEFAULT_N_FRAMES = 50
# Deliberately its own constant rather than reusing live_view.DEFAULT_DEGREE
# (1): a spectral calibration's pixel->wavelength_nm fit needs a higher
# baseline degree than the live view's default spatial dispersion fit.
DEFAULT_DEGREE = 3

# Exposure-mode choice strings, shared by BaselineDialog and FlatFieldDialog's
# exposure_mode_combo below -- mirrors cli/calibration.py's mutually-exclusive
# --auto-exposure/--exposure-us flag pair, minus the CLI's "give neither" case
# (a combo box always has one of the two selected).
EXPOSURE_MODE_AUTO = "Auto"
EXPOSURE_MODE_MANUAL = "Manual"

# Calibration-specific failures every camera-driven dialog below must route
# to show_calibration_error_dialog() rather than show_camera_error_dialog()
# (CameraError and subclasses) -- see module docstring.
_CALIBRATION_ERROR_TYPES = (
    SettingsMismatchError,
    InvalidFlatFieldError,
    InvalidConversionGainError,
    NoSignalError,
)

_DIALOG_STYLE = f"""
QDialog {{
    background-color: {COLOR_BACKGROUND};
}}
QLabel {{
    color: {COLOR_TEXT_PRIMARY};
}}
QLabel[role="hint"] {{
    color: {COLOR_TEXT_SECONDARY};
}}
QLabel[role="phase"] {{
    color: {COLOR_ACCENT};
    font-weight: 600;
}}
QLabel[role="formula"] {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 8px 10px;
}}
QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 4px 6px;
}}
QPushButton {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    border-color: {COLOR_ACCENT};
}}
QPushButton[role="primary"] {{
    background-color: {COLOR_ACCENT};
    color: #10101a;
    font-weight: 600;
    border: none;
}}
"""

# Classes


class BaselineDialog(QDialog):

    '''
    Form for a single-phase baseline calibration: number of background
    frames to average, an Auto/Manual exposure choice, plus gain_db (a
    required numeric input, same as the CLI -- not in configs/default.yaml
    since it varies per session).

    Auto/Manual mirrors cli/calibration.py's mutually-exclusive
    --auto-exposure/--exposure-us flags: "Auto" leaves exposure_us() as
    None (real auto-exposure convergence decides it at capture time) and
    resets gain_db_spin to 0.0 as a starting suggestion; "Manual" enables
    exposure_us_spin for a fixed exposure_us and leaves gain_db_spin alone.

    Mirrors cli/calibration.py's _cmd_baseline / run_baseline_calibration().
    '''

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Baseline Calibration")
        self.setStyleSheet(_DIALOG_STYLE)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*([MARGIN_DEFAULT] * 4))
        layout.setSpacing(SPACING_MEDIUM)

        heading = QLabel("Baseline Calibration")
        heading.setFont(load_bundled_font(14, bold=True))
        layout.addWidget(heading)

        hint = QLabel(
            "Block the beam and configure room lighting to the level "
            "used in spatial chirp measurement, then click Start Capture."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(SPACING_SMALL)

        self.n_frames_spin = QSpinBox()
        self.n_frames_spin.setRange(2, 1000)
        self.n_frames_spin.setValue(DEFAULT_N_FRAMES)
        form.addRow("Number of frames:", self.n_frames_spin)

        self.exposure_mode_combo = QComboBox()
        self.exposure_mode_combo.addItems([EXPOSURE_MODE_AUTO, EXPOSURE_MODE_MANUAL])
        form.addRow("Exposure:", self.exposure_mode_combo)

        self.exposure_us_spin = QDoubleSpinBox()
        self.exposure_us_spin.setRange(1.0, 1_000_000.0)
        self.exposure_us_spin.setSuffix(" us")
        form.addRow("Exposure (exposure_us):", self.exposure_us_spin)

        self.gain_db_spin = QDoubleSpinBox()
        self.gain_db_spin.setRange(0.0, 48.0)
        self.gain_db_spin.setSingleStep(0.1)
        self.gain_db_spin.setSuffix(" dB")
        form.addRow("Gain (db):", self.gain_db_spin)

        self.exposure_mode_combo.currentTextChanged.connect(self._on_exposure_mode_changed)
        self._on_exposure_mode_changed(self.exposure_mode_combo.currentText())

        layout.addLayout(form)

        self.status_label = QLabel("Not started.")
        self.status_label.setProperty("role", "hint")
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.start_button = QPushButton("Start Capture")
        self.start_button.setProperty("role", "primary")
        buttons.addButton(self.start_button, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.start_button.clicked.connect(self._on_start_clicked)

    def _on_exposure_mode_changed(self, mode: str) -> None:

        '''
        "Auto": disables exposure_us_spin (its value is ignored -- real
        auto-exposure convergence decides exposure_us at capture time) and
        resets gain_db_spin to 0.0 as a starting suggestion, re-applied
        every time Auto is (re-)selected, not just once.

        "Manual": enables exposure_us_spin; gain_db_spin is left as-is.
        '''

        is_auto = mode == EXPOSURE_MODE_AUTO
        self.exposure_us_spin.setEnabled(not is_auto)
        if is_auto:
            self.gain_db_spin.setValue(0.0)

    def auto_exposure(self) -> bool:
        '''True if "Auto" exposure is currently selected.'''
        return self.exposure_mode_combo.currentText() == EXPOSURE_MODE_AUTO

    def exposure_us(self) -> float | None:
        '''Manually-entered exposure_us, or None if "Auto" is selected.'''
        if self.auto_exposure():
            return None
        return self.exposure_us_spin.value()

    def _on_start_clicked(self) -> None:

        '''
        Builds a real CameraStream and runs run_baseline_calibration()
        against it -- mirrors cli/calibration.py's _cmd_baseline. Accepts
        the dialog only once the artifact is actually built and saved;
        any CameraError/calibration failure shows an error dialog and
        leaves this dialog open (with Start Capture re-enabled) so the
        user can retry.
        '''

        self.start_button.setEnabled(False)
        self.status_label.setText("Capturing...")
        try:
            camera_stream = build_camera_stream(
                self.gain_db_spin.value(),
                exposure_us=self.exposure_us(),
                auto_exposure=self.auto_exposure(),
            )
            camera_stream.start()
            try:
                run_baseline_calibration(
                    camera_stream,
                    self.n_frames_spin.value(),
                    DEFAULT_ARTIFACT_DIR / DEFAULT_BASELINE_FILENAME,
                )
            finally:
                if camera_stream.is_running:
                    camera_stream.stop()
        except CameraError as error:
            self.status_label.setText("Not started.")
            self.start_button.setEnabled(True)
            show_camera_error_dialog(self, str(error))
            return
        except _CALIBRATION_ERROR_TYPES as error:
            self.status_label.setText("Not started.")
            self.start_button.setEnabled(True)
            show_calibration_error_dialog(self, "Baseline Calibration Failed", str(error))
            return

        self.status_label.setText("Baseline calibration complete.")
        self.accept()


class FlatFieldDialog(QDialog):

    '''
    Two-phase flat-field calibration: dark frames, then uniformly
    illuminated frames, with an explicit UI pause (this dialog's phase
    stepper) between them rather than one blocking call -- the physical
    setup changes between phases (block the beam, then illuminate it).

    Also carries an Auto/Manual exposure choice (see auto_exposure()/
    exposure_us() below), same semantics as BaselineDialog's -- applies to
    both phases of this capture, matching how a single build_camera_stream()
    call configures the whole session on the CLI side.

    Mirrors cli/calibration.py's _cmd_flat_field (input() prompts there
    become this dialog's phase transitions), calling capture_dark_frames()
    -> capture_illuminated_frames() -> finish_flat_field_calibration().

    Bad-pixel-map has no manual "create" entry point anywhere in this
    screen -- build_bad_pixel_map()/save_bad_pixel_map() chain directly
    onto this dialog's completion (phase 3 below), automatically, once
    real logic is wired.
    '''

    PHASE_DARK = 0
    PHASE_ILLUMINATED = 1
    PHASE_FINISHING = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Flat-Field Calibration")
        self.setStyleSheet(_DIALOG_STYLE)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*([MARGIN_DEFAULT] * 4))
        layout.setSpacing(SPACING_MEDIUM)

        heading = QLabel("Flat-Field Calibration")
        heading.setFont(load_bundled_font(14, bold=True))
        layout.addWidget(heading)

        form = QFormLayout()
        form.setSpacing(SPACING_SMALL)
        self.n_frames_spin = QSpinBox()
        self.n_frames_spin.setRange(2, 1000)
        self.n_frames_spin.setValue(DEFAULT_N_FRAMES)
        form.addRow("Frames per phase:", self.n_frames_spin)
        self.exposure_mode_combo = QComboBox()
        self.exposure_mode_combo.addItems([EXPOSURE_MODE_AUTO, EXPOSURE_MODE_MANUAL])
        form.addRow("Exposure:", self.exposure_mode_combo)
        self.exposure_us_spin = QDoubleSpinBox()
        self.exposure_us_spin.setRange(1.0, 1_000_000.0)
        self.exposure_us_spin.setSuffix(" us")
        form.addRow("Exposure (exposure_us):", self.exposure_us_spin)
        self.gain_db_spin = QDoubleSpinBox()
        self.gain_db_spin.setRange(0.0, 48.0)
        self.gain_db_spin.setSingleStep(0.1)
        self.gain_db_spin.setSuffix(" dB")
        form.addRow("Gain (gain_db):", self.gain_db_spin)
        self.exposure_mode_combo.currentTextChanged.connect(self._on_exposure_mode_changed)
        self._on_exposure_mode_changed(self.exposure_mode_combo.currentText())
        layout.addLayout(form)

        self.phase_label = QLabel()
        self.phase_label.setProperty("role", "phase")
        layout.addWidget(self.phase_label)

        self.instruction_label = QLabel()
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)

        self.status_label = QLabel()
        self.status_label.setProperty("role", "hint")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.continue_button = QPushButton("Continue")
        self.continue_button.setProperty("role", "primary")
        buttons.addButton(self.continue_button, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._phase = self.PHASE_DARK
        self._camera_stream: CameraStream | None = None
        self._dark_frames: list[FrameData] | None = None
        self.continue_button.clicked.connect(self._on_continue_clicked)
        self._render_phase()

    def _render_phase(self) -> None:
        if self._phase == self.PHASE_DARK:
            self.phase_label.setText("Phase 1 of 2 -- Baseline frames")
            self.instruction_label.setText("Block the beam and configure room lighting to the level used in spatial chirp measurement, then click Continue.")
            self.status_label.setText("Not started.")
            self.continue_button.setText("Continue")
        elif self._phase == self.PHASE_ILLUMINATED:
            self.phase_label.setText("Phase 2 of 2 -- Illuminated frames")
            self.instruction_label.setText(
                "Set up uniform illumination, then click Continue."
            )
            self.status_label.setText("Baseline frames captured.")
            self.continue_button.setText("Continue")
        else:
            self.phase_label.setText("Finishing")
            self.instruction_label.setText(
                "Building flat field and bad-pixel map from captured frames..."
            )
            self.status_label.setText(
                "Illuminated frames captured. Bad-pixel map builds "
                "automatically once the flat field is saved."
            )
            self.continue_button.setText("Done")

    def _advance_phase(self) -> None:
        '''Visual-only phase step, kept for tests that only need to exercise
        the phase stepper's own rendering (see _render_phase()) without
        driving a real capture -- _on_continue_clicked() below is what the
        Continue button is actually wired to.'''
        if self._phase == self.PHASE_FINISHING:
            self.accept()
            return
        self._phase += 1
        self._render_phase()

    def _reset_to_dark_phase(self) -> None:
        '''Discards any in-progress capture state and returns to phase 1 --
        used on failure so a retry starts the two-phase sequence over
        rather than resuming from a stream/frames left in an unknown
        state.'''
        if self._camera_stream is not None and self._camera_stream.is_running:
            self._camera_stream.stop()
        self._camera_stream = None
        self._dark_frames = None
        self._phase = self.PHASE_DARK
        self._render_phase()

    def _on_continue_clicked(self) -> None:
        if self._phase == self.PHASE_DARK:
            self._capture_dark_phase()
        elif self._phase == self.PHASE_ILLUMINATED:
            self._capture_illuminated_and_finish()
        else:
            self.accept()

    def _capture_dark_phase(self) -> None:

        '''
        Phase 1 -> 2: builds a real CameraStream, starts it, and captures
        the dark frames -- mirrors cli/calibration.py's _cmd_flat_field
        up through its first input() prompt. The stream is left running
        (not stopped) on success, since phase 2 captures from the same
        stream/physical setup.
        '''

        self.continue_button.setEnabled(False)
        self.status_label.setText("Capturing dark frames...")
        try:
            camera_stream = build_camera_stream(
                self.gain_db_spin.value(),
                exposure_us=self.exposure_us(),
                auto_exposure=self.auto_exposure(),
            )
            camera_stream.start()
            self._camera_stream = camera_stream
            self._dark_frames = capture_dark_frames(camera_stream, self.n_frames_spin.value())
        except CameraError as error:
            self._reset_to_dark_phase()
            self.continue_button.setEnabled(True)
            show_camera_error_dialog(self, str(error))
            return

        self.continue_button.setEnabled(True)
        self._phase = self.PHASE_ILLUMINATED
        self._render_phase()

    def _capture_illuminated_and_finish(self) -> None:

        '''
        Phase 2 -> 3: captures illuminated frames from the same stream
        _capture_dark_phase() started, then builds and saves the flat
        field. finish_flat_field_calibration() itself only builds/saves
        the flat field (build_flat_field() + save_flat_field(), per its
        own docstring) -- it does NOT chain bad-pixel-map building, so
        that's done explicitly here instead, the same load-then-build
        sequence cli/calibration.py's separate `bad-pixel-map` subcommand
        (_cmd_bad_pixel_map) uses. This is what makes bad-pixel-map have
        no manual "create" entry point of its own anywhere in this screen
        (see class docstring). Mirrors cli/calibration.py's
        _cmd_flat_field from its second input() prompt onward, plus that
        extra bad-pixel-map step. On any failure this resets back to
        phase 1 (see _reset_to_dark_phase()) rather than leaving a
        stopped stream/stale frames behind for a phase-2-only retry.
        '''

        self.continue_button.setEnabled(False)
        self.status_label.setText("Capturing illuminated frames...")
        camera_stream = self._camera_stream
        try:
            illuminated_frames = capture_illuminated_frames(
                camera_stream, self.n_frames_spin.value()
            )
        except CameraError as error:
            self._reset_to_dark_phase()
            self.continue_button.setEnabled(True)
            show_camera_error_dialog(self, str(error))
            return

        try:
            flat_field_path = DEFAULT_ARTIFACT_DIR / DEFAULT_FLAT_FIELD_FILENAME
            finish_flat_field_calibration(illuminated_frames, self._dark_frames, flat_field_path)
            flat_field, flat_field_record = load_flat_field(flat_field_path)
            bad_pixel_mask, bad_pixel_record = build_bad_pixel_map(flat_field, flat_field_record)
            save_bad_pixel_map(
                DEFAULT_ARTIFACT_DIR / DEFAULT_BAD_PIXEL_MAP_FILENAME, bad_pixel_mask, bad_pixel_record
            )
        except _CALIBRATION_ERROR_TYPES as error:
            self._reset_to_dark_phase()
            self.continue_button.setEnabled(True)
            show_calibration_error_dialog(self, "Flat-Field Calibration Failed", str(error))
            return
        finally:
            if camera_stream.is_running:
                camera_stream.stop()

        self.continue_button.setEnabled(True)
        self._camera_stream = None
        self._dark_frames = None
        self._phase = self.PHASE_FINISHING
        self._render_phase()

    def _on_exposure_mode_changed(self, mode: str) -> None:

        '''
        "Auto": disables exposure_us_spin (its value is ignored -- real
        auto-exposure convergence decides exposure_us at capture time) and
        resets gain_db_spin to 0.0 as a starting suggestion, re-applied
        every time Auto is (re-)selected, not just once.

        "Manual": enables exposure_us_spin; gain_db_spin is left as-is.
        '''

        is_auto = mode == EXPOSURE_MODE_AUTO
        self.exposure_us_spin.setEnabled(not is_auto)
        if is_auto:
            self.gain_db_spin.setValue(0.0)

    def auto_exposure(self) -> bool:
        '''True if "Auto" exposure is currently selected.'''
        return self.exposure_mode_combo.currentText() == EXPOSURE_MODE_AUTO

    def exposure_us(self) -> float | None:
        '''Manually-entered exposure_us, or None if "Auto" is selected.'''
        if self.auto_exposure():
            return None
        return self.exposure_us_spin.value()


class ConversionGainDialog(QDialog):

    '''
    Form for conversion-gain (photon transfer curve) calibration: exposure
    sweep bounds, level count, frames per level, and gain_db -- all
    required with no defaults, matching run_conversion_gain_calibration()'s
    own contract (the right exposure range depends on the physical setup
    and can't be guessed at).
    '''

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Conversion-Gain Calibration")
        self.setStyleSheet(_DIALOG_STYLE)
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*([MARGIN_DEFAULT] * 4))
        layout.setSpacing(SPACING_MEDIUM)

        heading = QLabel("Conversion-Gain Calibration")
        heading.setFont(load_bundled_font(14, bold=True))
        layout.addWidget(heading)

        hint = QLabel(
            "Block the beam and configure room lighting to the level used "
            "in spatial chirp measurement. Set up uniform illumination. "
            "Fill in required fields, then click Start Sweep."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(SPACING_SMALL)

        # Every field below starts at its range minimum with
        # setSpecialValueText("required") -- all four are required with no
        # sane default (see class docstring), so each reads as an
        # unfilled placeholder rather than a real suggested value until
        # the user changes it.
        self.exposure_min_spin = QDoubleSpinBox()
        self.exposure_min_spin.setRange(0.0, 1_000_000.0)
        self.exposure_min_spin.setSuffix(" us")
        self.exposure_min_spin.setSpecialValueText("required")
        form.addRow("Exposure min (--exposure-min-us):", self.exposure_min_spin)

        self.exposure_max_spin = QDoubleSpinBox()
        self.exposure_max_spin.setRange(0.0, 1_000_000.0)
        self.exposure_max_spin.setSuffix(" us")
        self.exposure_max_spin.setSpecialValueText("required")
        form.addRow("Exposure max (--exposure-max-us):", self.exposure_max_spin)

        self.n_levels_spin = QSpinBox()
        self.n_levels_spin.setRange(0, 100)
        self.n_levels_spin.setSpecialValueText("required")
        form.addRow("Number of levels (--n-levels):", self.n_levels_spin)

        self.n_frames_per_level_spin = QSpinBox()
        self.n_frames_per_level_spin.setRange(0, 1000)
        self.n_frames_per_level_spin.setSpecialValueText("required")
        form.addRow("Frames per level (--n-frames-per-level):", self.n_frames_per_level_spin)

        self.gain_db_spin = QDoubleSpinBox()
        self.gain_db_spin.setRange(0.0, 48.0)
        self.gain_db_spin.setSingleStep(0.1)
        self.gain_db_spin.setSuffix(" dB")
        form.addRow("Gain (gain_db):", self.gain_db_spin)

        layout.addLayout(form)

        self.status_label = QLabel("Not started.")
        self.status_label.setProperty("role", "hint")
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.start_button = QPushButton("Start Sweep")
        self.start_button.setProperty("role", "primary")
        buttons.addButton(self.start_button, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.start_button.clicked.connect(self._on_start_clicked)

    def _on_start_clicked(self) -> None:

        '''
        Builds a real CameraStream (seeded at exposure_min_us -- see
        cli/calibration.py's _cmd_conversion_gain for why that's exact,
        not a guess) and runs run_conversion_gain_calibration() against
        it. ValueError (e.g. a required field left at its unfilled
        minimum) is routed the same way as the calibration-specific
        exceptions below, since it's just as retryable by correcting the
        form's inputs.
        '''

        self.start_button.setEnabled(False)
        self.status_label.setText("Sweeping exposure...")
        try:
            camera_stream = build_camera_stream(
                self.gain_db_spin.value(), exposure_us=self.exposure_min_spin.value(),
            )
            camera_stream.start()
            try:
                run_conversion_gain_calibration(
                    camera_stream,
                    self.exposure_min_spin.value(),
                    self.exposure_max_spin.value(),
                    self.n_levels_spin.value(),
                    self.n_frames_per_level_spin.value(),
                    DEFAULT_ARTIFACT_DIR / DEFAULT_CONVERSION_GAIN_FILENAME,
                )
            finally:
                if camera_stream.is_running:
                    camera_stream.stop()
        except CameraError as error:
            self.status_label.setText("Not started.")
            self.start_button.setEnabled(True)
            show_camera_error_dialog(self, str(error))
            return
        except (ValueError, *_CALIBRATION_ERROR_TYPES) as error:
            self.status_label.setText("Not started.")
            self.start_button.setEnabled(True)
            show_calibration_error_dialog(self, "Conversion-Gain Calibration Failed", str(error))
            return

        self.status_label.setText("Conversion-gain calibration complete.")
        self.accept()


class SpatialCalibrationDialog(QDialog):

    '''
    Spatial calibration is NOT a camera measurement -- just a numeric
    override for the fixed pixel->physical-position scale factor (see
    calibration/spatial/calibrate.py's DEFAULT_SCALE_FACTOR). This dialog
    is deliberately styled distinctly from the four acquire-driven dialogs
    above (COLOR_ACCENT_ALT instead of COLOR_ACCENT, "Save" instead of
    "Start Capture", no frame-count/gain fields, no phase stepper) since
    it has no acquisition step at all.

    Saves a ScaleFactorPositionCalibration built from the entered value
    via save_scale_factor() -- mirrors cli/calibration.py's _cmd_spatial.
    '''

    def __init__(self, current_scale_factor: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spatial Calibration -- Scale Factor")
        self.setStyleSheet(
            _DIALOG_STYLE
            + f"""
            QPushButton[role="primary"] {{
                background-color: {COLOR_ACCENT_ALT};
                color: #14101a;
            }}
            QLabel[role="phase"] {{
                color: {COLOR_ACCENT_ALT};
            }}
            """
        )
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*([MARGIN_DEFAULT] * 4))
        layout.setSpacing(SPACING_MEDIUM)

        heading = QLabel("Spatial Calibration")
        heading.setFont(load_bundled_font(14, bold=True))
        layout.addWidget(heading)

        '''
        badge = QLabel("No camera interaction -- manual value entry only")
        badge.setProperty("role", "phase")
        layout.addWidget(badge)
        '''

        hint = QLabel(
            "Scale factor converting distances along the camera's "
            "spatial axis to distances on the slit-plane. Theoretical "
            "default is the ratio of the focal length of the first lens "
            "to that of the second. Enter a manually measured value, or leave the "
            "default in place."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(SPACING_SMALL)
        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setRange(0.001, 1000.0)
        self.scale_factor_spin.setDecimals(4)
        self.scale_factor_spin.setValue(current_scale_factor)
        form.addRow("Scale factor:", self.scale_factor_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.save_button = QPushButton("Save")
        self.save_button.setProperty("role", "primary")
        buttons.addButton(self.save_button, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.save_button.clicked.connect(self._on_save_clicked)

    def _on_save_clicked(self) -> None:
        '''Builds a ScaleFactorPositionCalibration from the entered value
        and persists it with source="manual" -- no camera/build_*() step,
        so nothing here can raise a CameraError or CalibrationError.'''
        calibration = ScaleFactorPositionCalibration(scale_factor=self.scale_factor_spin.value())
        save_scale_factor(
            DEFAULT_ARTIFACT_DIR / DEFAULT_SCALE_FACTOR_FILENAME, calibration, source="manual"
        )
        self.accept()


class _SpectralCaptureWorker(QThread):

    '''
    Runs build_camera_stream() -> start() -> run_spectral_calibration() ->
    stop() on a background Qt thread, mirroring exactly what
    SpectralCalibrationDialog._on_start_clicked() used to run directly on
    the GUI thread. That synchronous version froze the whole window for
    the entire capture (tens of frames' worth of grabs, plus line-matching
    and fitting) -- worse, a fatal camera error (e.g. a GigE buffer
    underrun) partway through left the window looking hung until the error
    dialog finally appeared. Running it here keeps Qt's event loop free to
    paint/process input the whole time; results come back via the
    succeeded/failed signals rather than a return value or raised
    exception, since neither crosses a thread boundary safely.

    failed carries the raised exception object itself (Signal(object), not
    a message string) so the connected slot can reproduce the original
    isinstance-based dispatch to show_camera_error_dialog() vs.
    show_calibration_error_dialog().
    '''

    succeeded = Signal()
    failed = Signal(object)

    def __init__(
        self,
        gain_db: float,
        exposure_us: float | None,
        auto_exposure: bool,
        n_frames: int,
        sensor_calibration: CalibrationSet,
        spectral_path: Path,
        geometric_tilt_path: Path,
        degree: int,
        gain_e_per_adu: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._gain_db = gain_db
        self._exposure_us = exposure_us
        self._auto_exposure = auto_exposure
        self._n_frames = n_frames
        self._sensor_calibration = sensor_calibration
        self._spectral_path = spectral_path
        self._geometric_tilt_path = geometric_tilt_path
        self._degree = degree
        self._gain_e_per_adu = gain_e_per_adu

    def run(self) -> None:

        '''
        QThread's entry point -- never called directly, only ever invoked
        by start(). Every exception (CameraError, ValueError,
        LineMatchingError, the calibration-specific error types) is caught
        here and handed to `failed` rather than left to propagate, since an
        exception raised on this thread would otherwise be silently lost
        instead of reaching the GUI thread at all.
        '''

        try:
            camera_stream = build_camera_stream(
                self._gain_db,
                exposure_us=self._exposure_us,
                auto_exposure=self._auto_exposure,
            )
            camera_stream.start()
            try:
                run_spectral_calibration(
                    camera_stream,
                    self._n_frames,
                    self._sensor_calibration,
                    self._spectral_path,
                    self._geometric_tilt_path,
                    degree=self._degree,
                    gain_e_per_adu=self._gain_e_per_adu,
                )
            finally:
                if camera_stream.is_running:
                    camera_stream.stop()
        except Exception as error:
            self.failed.emit(error)
        else:
            self.succeeded.emit()


class SpectralCalibrationDialog(QDialog):

    '''
    Pixel -> wavelength_nm calibration, offering two mutually exclusive
    entry points via a mode selector (a QRadioButton pair) at the top of
    the dialog, each showing its own section of a QStackedWidget below:

    "Capture from Lamp" mirrors BaselineDialog's single-phase capture
    form (frame count + Auto/Manual exposure choice + gain_db), plus a
    fit-degree selector, and calls calibration/spectral/workflow.py's
    run_spectral_calibration(camera_stream, n_frames, sensor_calibration,
    path, geometric_tilt_path, degree, gain_e_per_adu) against a
    CalibrationSet + real gain_e_per_adu loaded fresh from
    DEFAULT_ARTIFACT_DIR (baseline + flat field + bad-pixel map +
    conversion gain -- see _on_start_clicked()'s docstring for what
    happens if any of those don't exist yet; conversion gain specifically
    is what lets the geometric-tilt calibration built alongside spectral
    use real centroid-uncertainty weighting instead of a placeholder).
    Manual exposure matters here specifically because
    the lamp frames get preprocessed through that loaded baseline before
    line-matching -- check_settings_match() rejects a lamp frame whose
    actual exposure_us drifts from the baseline artifact's tagged value,
    which auto-exposure convergence (picking whatever the dim lamp needs)
    has no way to guarantee; entering the same exposure_us the baseline
    was captured at avoids the mismatch outright.

    "Manual Entry" mirrors SpatialCalibrationDialog's no-camera,
    direct-value style, but for a variable-length pixel->wavelength_nm
    polynomial (wavelength_nm = c0 + c1*pixel + c2*pixel^2 + ...,
    ascending order) plus each coefficient's REQUIRED 1-sigma
    uncertainty -- see calibration/spectral/calibrate.py's
    build_manual_spectral_calibration() docstring for why
    coefficient_sigma can't be optional or defaulted (no fit residual
    exists to estimate it from). Changing the degree selector rebuilds
    the coefficient/sigma row list to match. Calls
    build_manual_spectral_calibration() + save_spectral_calibration().
    '''

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Spectral Calibration")
        self.setStyleSheet(_DIALOG_STYLE)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*([MARGIN_DEFAULT] * 4))
        layout.setSpacing(SPACING_MEDIUM)

        heading = QLabel("Spectral Calibration")
        heading.setFont(load_bundled_font(14, bold=True))
        layout.addWidget(heading)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(SPACING_MEDIUM)
        self.mode_group = QButtonGroup(self)
        self.capture_mode_radio = QRadioButton("Automatic")
        self.capture_mode_radio.setChecked(True)
        self.manual_mode_radio = QRadioButton("Manual Entry")
        self.mode_group.addButton(self.capture_mode_radio)
        self.mode_group.addButton(self.manual_mode_radio)
        mode_row.addWidget(self.capture_mode_radio)
        mode_row.addWidget(self.manual_mode_radio)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self._mode_stack = QStackedWidget()
        layout.addWidget(self._mode_stack)

        self._mode_stack.addWidget(self._build_capture_page())
        self._mode_stack.addWidget(self._build_manual_page())

        self.capture_mode_radio.toggled.connect(self._on_mode_toggled)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.cancel_button = buttons.button(QDialogButtonBox.Cancel)
        self.start_button = QPushButton("Start Capture")
        self.start_button.setProperty("role", "primary")
        buttons.addButton(self.start_button, QDialogButtonBox.ActionRole)
        self.save_button = QPushButton("Save")
        self.save_button.setProperty("role", "primary")
        buttons.addButton(self.save_button, QDialogButtonBox.ActionRole)
        self.save_button.setVisible(False)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.start_button.clicked.connect(self._on_start_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)

        # Set only while a _SpectralCaptureWorker is running (see
        # _on_start_clicked()) -- kept alive via this reference and via Qt
        # parent ownership (parent=self), so it isn't garbage-collected
        # mid-capture.
        self._capture_worker: _SpectralCaptureWorker | None = None

    def _build_capture_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(SPACING_MEDIUM)

        hint = QLabel(
            "Place the lamp in front of the imaging spectrometer's closed "
            "entrance slit then perform Baseline calibration. Turn off the "
            "lamp and perform phase 1 of Flat Field calibration. Illuminate "
            "the sensor uniformly (with the lamp if you wish) and perform "
            "phase 2 of Flat Field calibration. Turn on the lamp and place it "
            "in front of the imaging spectrometer's entrance slit, widened to "
            "the correct width, then perform Spectral Calibration by clicking "
            "Start Capture. Once this is done, turn off the lamp and recalibrate "
            "the Baseline with the same room-lighting configuration used in spatial "
            "chirp measurement."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        page_layout.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(SPACING_SMALL)

        self.n_frames_spin = QSpinBox()
        self.n_frames_spin.setRange(2, 1000)
        self.n_frames_spin.setValue(DEFAULT_N_FRAMES)
        form.addRow("Number of frames:", self.n_frames_spin)

        self.exposure_mode_combo = QComboBox()
        self.exposure_mode_combo.addItems([EXPOSURE_MODE_AUTO, EXPOSURE_MODE_MANUAL])
        form.addRow("Exposure:", self.exposure_mode_combo)

        self.exposure_us_spin = QDoubleSpinBox()
        self.exposure_us_spin.setRange(1.0, 1_000_000.0)
        self.exposure_us_spin.setSuffix(" us")
        form.addRow("Exposure (exposure_us):", self.exposure_us_spin)

        self.gain_db_spin = QDoubleSpinBox()
        self.gain_db_spin.setRange(0.0, 48.0)
        self.gain_db_spin.setSingleStep(0.1)
        self.gain_db_spin.setSuffix(" dB")
        form.addRow("Gain (gain_db):", self.gain_db_spin)

        self.exposure_mode_combo.currentTextChanged.connect(self._on_exposure_mode_changed)
        self._on_exposure_mode_changed(self.exposure_mode_combo.currentText())

        self.capture_degree_selector = QComboBox()
        for degree in DEGREE_CHOICES:
            self.capture_degree_selector.addItem(DEGREE_LABELS[degree], userData=degree)
        self.capture_degree_selector.setCurrentIndex(DEGREE_CHOICES.index(DEFAULT_DEGREE))
        form.addRow("Fit degree:", self.capture_degree_selector)

        page_layout.addLayout(form)

        self.status_label = QLabel("Not started.")
        self.status_label.setProperty("role", "hint")
        page_layout.addWidget(self.status_label)

        page_layout.addStretch(1)
        return page

    def _build_manual_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(SPACING_MEDIUM)

        hint = QLabel(
            "Enter the coefficients of a user-calibrated polynomial mapping "
            "of sensor-column to wavelength."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        page_layout.addWidget(hint)

        degree_row = QHBoxLayout()
        degree_row.setSpacing(SPACING_SMALL)
        degree_row.addWidget(QLabel("Polynomial degree:"))
        self.manual_degree_selector = QComboBox()
        for degree in DEGREE_CHOICES:
            self.manual_degree_selector.addItem(DEGREE_LABELS[degree], userData=degree)
        self.manual_degree_selector.setCurrentIndex(DEGREE_CHOICES.index(DEFAULT_DEGREE))
        degree_row.addWidget(self.manual_degree_selector)
        degree_row.addStretch(1)
        page_layout.addLayout(degree_row)

        self.formula_label = QLabel()
        self.formula_label.setTextFormat(Qt.RichText)
        self.formula_label.setAlignment(Qt.AlignCenter)
        self.formula_label.setProperty("role", "formula")
        self.formula_label.setFont(load_bundled_font(12))
        page_layout.addWidget(self.formula_label)

        self._coefficient_form = QFormLayout()
        self._coefficient_form.setSpacing(SPACING_SMALL)
        page_layout.addLayout(self._coefficient_form)

        self._coefficient_rows: list[tuple[QDoubleSpinBox, QDoubleSpinBox]] = []
        self.manual_degree_selector.currentIndexChanged.connect(self._rebuild_coefficient_rows)
        self._rebuild_coefficient_rows()

        page_layout.addStretch(1)
        return page

    def _rebuild_coefficient_rows(self) -> None:

        '''
        Rebuilds self._coefficient_form to have one (value, sigma) row per
        coefficient of the currently selected degree -- c0..c(degree),
        ascending order matching build_manual_spectral_calibration()'s own
        coefficient convention. Called once at construction and again
        every time manual_degree_selector's selection changes.
        '''

        while self._coefficient_form.rowCount():
            self._coefficient_form.removeRow(0)
        self._coefficient_rows = []

        degree = self.manual_degree_selector.currentData()
        self.formula_label.setText(manual_spectral_formula_html(degree))
        for i in range(degree + 1):
            value_spin = QDoubleSpinBox()
            value_spin.setRange(-1_000_000.0, 1_000_000.0)
            value_spin.setDecimals(6)

            sigma_spin = QDoubleSpinBox()
            sigma_spin.setRange(1e-6, 1_000_000.0)
            sigma_spin.setDecimals(6)
            sigma_spin.setValue(1e-6)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(SPACING_SMALL)
            row_layout.addWidget(QLabel("value:"))
            row_layout.addWidget(value_spin)
            row_layout.addWidget(QLabel("+/- sigma:"))
            row_layout.addWidget(sigma_spin)

            self._coefficient_form.addRow(f"c{i}:", row)
            self._coefficient_rows.append((value_spin, sigma_spin))

    def _on_mode_toggled(self, capture_checked: bool) -> None:
        self._mode_stack.setCurrentIndex(0 if capture_checked else 1)
        self.start_button.setVisible(capture_checked)
        self.save_button.setVisible(not capture_checked)

    def coefficients(self) -> list[float]:

        '''Current manual-mode coefficient values, ascending order (c0, c1, ...).'''

        return [value_spin.value() for value_spin, _ in self._coefficient_rows]

    def coefficient_sigma(self) -> list[float]:

        '''Current manual-mode per-coefficient 1-sigma uncertainties, same order as coefficients().'''

        return [sigma_spin.value() for _, sigma_spin in self._coefficient_rows]

    def _load_sensor_calibration(self) -> tuple[CalibrationSet, float] | None:

        '''
        Loads baseline + flat field + bad-pixel map + conversion gain
        from DEFAULT_ARTIFACT_DIR -- the same paths CalibrationScreen's
        own "load existing calibrations" flow reads from -- and bundles
        the first three into a CalibrationSet for run_spectral_
        calibration() to preprocess lamp frames with. conversion_gain
        is returned separately (as gain_e_per_adu, not folded into
        CalibrationSet, which has no field for it) for the caller to pass
        into run_spectral_calibration() directly -- it's what lets the
        geometric-tilt calibration built alongside spectral use real
        Thompson-Larson-Webb centroid weighting instead of
        build_geometric_tilt()'s own placeholder (see that function's
        docstring). Returns None (having already shown an explanatory
        error dialog) if any of the four haven't been created yet, rather
        than letting the resulting FileNotFoundError propagate.
        '''

        try:
            baseline_result, baseline_record = load_baseline(
                DEFAULT_ARTIFACT_DIR / DEFAULT_BASELINE_FILENAME
            )
            flat_field, flat_field_record = load_flat_field(
                DEFAULT_ARTIFACT_DIR / DEFAULT_FLAT_FIELD_FILENAME
            )
            bad_pixel_mask, _ = load_bad_pixel_map(
                DEFAULT_ARTIFACT_DIR / DEFAULT_BAD_PIXEL_MAP_FILENAME
            )
            conversion_gain_result, _ = load_conversion_gain(
                DEFAULT_ARTIFACT_DIR / DEFAULT_CONVERSION_GAIN_FILENAME
            )
        except FileNotFoundError:
            show_calibration_error_dialog(
                self,
                "Missing Sensor Calibration",
                "Baseline, flat-field, and conversion-gain calibrations must be created "
                "before spectral calibration can run. Create them first, then retry.",
            )
            return None

        sensor_calibration = CalibrationSet(
            baseline=baseline_result.baseline,
            baseline_record=baseline_record,
            flat_field=flat_field,
            flat_field_record=flat_field_record,
            bad_pixel_mask=bad_pixel_mask,
            background_sigma=baseline_result.background_sigma,
        )
        return sensor_calibration, conversion_gain_result.gain_e_per_adu

    def _on_exposure_mode_changed(self, mode: str) -> None:

        '''
        "Auto": disables exposure_us_spin (its value is ignored -- real
        auto-exposure convergence decides exposure_us at capture time) and
        resets gain_db_spin to 0.0 as a starting suggestion, re-applied
        every time Auto is (re-)selected, not just once.

        "Manual": enables exposure_us_spin; gain_db_spin is left as-is.
        '''

        is_auto = mode == EXPOSURE_MODE_AUTO
        self.exposure_us_spin.setEnabled(not is_auto)
        if is_auto:
            self.gain_db_spin.setValue(0.0)

    def auto_exposure(self) -> bool:
        '''True if "Auto" exposure is currently selected.'''
        return self.exposure_mode_combo.currentText() == EXPOSURE_MODE_AUTO

    def exposure_us(self) -> float | None:
        '''Manually-entered exposure_us, or None if "Auto" is selected.'''
        if self.auto_exposure():
            return None
        return self.exposure_us_spin.value()

    def _on_start_clicked(self) -> None:

        '''
        Capture-mode accept path: loads the already-built sensor
        CalibrationSet + real gain_e_per_adu (see
        _load_sensor_calibration()), then hands a CameraStream build +
        run_spectral_calibration() call off to a _SpectralCaptureWorker
        (see its own docstring for why this runs on a background thread
        rather than here directly) -- mirrors cli/calibration.py's
        _cmd_spectral_capture. Manual exposure (see
        auto_exposure()/exposure_us() above) matters here specifically so
        the lamp frames' actual exposure_us can be made to match the
        loaded baseline's -- see class docstring. run_spectral_
        calibration() also builds and saves the geometric tilt
        calibration as a side effect (see its own docstring) -- not
        duplicated here.

        The Cancel button is disabled for the capture's duration: closing
        this dialog while _SpectralCaptureWorker still owns a running
        CameraStream would leave that stream (and its background grab
        thread) orphaned, and risks the worker's succeeded/failed signals
        firing into a dialog that's already been destroyed.
        '''

        loaded = self._load_sensor_calibration()
        if loaded is None:
            return
        sensor_calibration, gain_e_per_adu = loaded

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Capturing...")

        self._capture_worker = _SpectralCaptureWorker(
            self.gain_db_spin.value(),
            self.exposure_us(),
            self.auto_exposure(),
            self.n_frames_spin.value(),
            sensor_calibration,
            DEFAULT_ARTIFACT_DIR / DEFAULT_SPECTRAL_FILENAME,
            DEFAULT_ARTIFACT_DIR / DEFAULT_GEOMETRIC_TILT_FILENAME,
            self.capture_degree_selector.currentData(),
            gain_e_per_adu,
            parent=self,
        )
        self._capture_worker.succeeded.connect(self._on_capture_succeeded)
        self._capture_worker.failed.connect(self._on_capture_failed)
        self._capture_worker.start()

    def _on_capture_succeeded(self) -> None:

        '''
        _SpectralCaptureWorker.succeeded slot -- mirrors the tail of the
        old synchronous _on_start_clicked()'s success path exactly.
        '''

        self._capture_worker = None
        self.status_label.setText("Spectral calibration complete.")
        self.accept()

    def _on_capture_failed(self, error: Exception) -> None:

        '''
        _SpectralCaptureWorker.failed slot -- reproduces the isinstance
        dispatch the old synchronous _on_start_clicked() did inline via
        `except CameraError` / `except (ValueError, LineMatchingError,
        *_CALIBRATION_ERROR_TYPES)`, since a background thread can't raise
        into the GUI thread's except blocks directly.
        '''

        self._capture_worker = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Not started.")

        if isinstance(error, CameraError):
            show_camera_error_dialog(self, str(error))
        elif isinstance(error, (ValueError, LineMatchingError, *_CALIBRATION_ERROR_TYPES)):
            show_calibration_error_dialog(self, "Spectral Calibration Failed", str(error))
        else:
            raise error

    def _on_save_clicked(self) -> None:

        '''
        Manual-mode accept path: builds a WavelengthCalibrationResult
        directly from the entered coefficients/coefficient_sigma via
        build_manual_spectral_calibration() and saves it -- mirrors
        cli/calibration.py's _cmd_spectral_manual. MANUAL_SPECTRAL_
        EXPOSURE_US/MANUAL_SPECTRAL_GAIN_DB are the same "not applicable"
        CalibrationRecord placeholders the CLI uses (no frame was
        actually captured).
        '''

        record = CalibrationRecord(
            exposure_us=MANUAL_SPECTRAL_EXPOSURE_US,
            gain_db=MANUAL_SPECTRAL_GAIN_DB,
            timestamp=time.time(),
            source_frame_count=1,
        )
        try:
            result = build_manual_spectral_calibration(
                self.coefficients(), self.coefficient_sigma(), record
            )
        except ValueError as error:
            show_calibration_error_dialog(self, "Spectral Calibration Failed", str(error))
            return

        save_spectral_calibration(DEFAULT_ARTIFACT_DIR / DEFAULT_SPECTRAL_FILENAME, result)
        self.accept()


# Functions


def manual_spectral_formula_html(degree: int) -> str:

    '''
    Rich-text (HTML) general form of the pixel -> wavelength_nm polynomial
    at the given degree, e.g. "λ = c<sub>0</sub> + c<sub>1</sub>x +
    c<sub>2</sub>x<sup>2</sup>" for degree 2, with real sub/superscripts
    for QLabel's built-in rich-text support -- shown above
    SpectralCalibrationDialog's manual-entry coefficient rows and kept in
    sync with manual_degree_selector by _rebuild_coefficient_rows().
    Mirrors live_view.py's fit_formula_html() convention, but for the
    pixel->wavelength_nm direction used by manual entry (fit_formula_html()
    only covers the reverse wavelength_nm->pixel direction).

    Parameters
    ----------
    degree
        Polynomial degree (1, 2, or 3 -- see DEGREE_CHOICES).

    Returns
    -------
    str
        HTML fragment suitable for a QLabel with rich text enabled.
    '''

    terms = ["c<sub>0</sub>"]
    for power in range(1, degree + 1):
        x_part = "x" if power == 1 else f"x<sup>{power}</sup>"
        terms.append(f"c<sub>{power}</sub>{x_part}")

    return "λ = " + " + ".join(terms)


def show_camera_error_dialog(parent: QWidget | None, message: str) -> None:

    '''
    Clearly visible error state for camera-connection-level failures
    (CameraError and subclasses -- the acquisition stream itself has
    died). Distinct from show_calibration_error_dialog() below: this one
    does not offer a "retry the same form" framing, since the stream
    itself needs to be reconnected first.

    Parameters
    ----------
    parent
        Parent widget for the modal dialog.
    message
        The underlying CameraError's str(), shown verbatim.
    '''

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle("Camera Connection Error")
    box.setText("The camera stream has stopped responding.")
    box.setInformativeText(message)
    box.setStyleSheet(_DIALOG_STYLE)
    box.exec()


def show_calibration_error_dialog(parent: QWidget | None, title: str, message: str) -> None:

    '''
    Clear, retryable error dialog for calibration-specific failures --
    SettingsMismatchError, InvalidFlatFieldError, InvalidConversionGainError,
    NoSignalError. Shows the actual exception message rather than a
    generic "something went wrong", and leaves the originating form/dialog
    open so the user can correct inputs (e.g. re-check illumination) and
    retry, instead of losing their in-progress entries.

    Parameters
    ----------
    parent
        Parent widget for the modal dialog.
    title
        Short description of which calibration step failed.
    message
        The underlying exception's str(), shown verbatim.
    '''

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(title)
    box.setText(message)
    box.setStyleSheet(_DIALOG_STYLE)
    box.exec()


__all__ = [
    "BaselineDialog",
    "FlatFieldDialog",
    "ConversionGainDialog",
    "SpatialCalibrationDialog",
    "SpectralCalibrationDialog",
    "manual_spectral_formula_html",
    "show_camera_error_dialog",
    "show_calibration_error_dialog",
    "DEFAULT_N_FRAMES",
    "DEFAULT_DEGREE",
    "EXPOSURE_MODE_AUTO",
    "EXPOSURE_MODE_MANUAL",
]
