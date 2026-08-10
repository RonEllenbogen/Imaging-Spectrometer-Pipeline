"""
Modal dialogs used by calibration_screen.py's "create new calibration"
flow -- one per calibration type that needs its own form or step sequence,
plus a shared error dialog for camera-connection and calibration-specific
failures.

PHASE 1 (VISUAL SKELETON) NOTE: every dialog here is laid out and styled
but not wired to real calibration/acquisition calls yet -- "Start"/
"Continue"/"Save" buttons currently just advance the dialog's own visual
state (e.g. FlatFieldDialog's phase indicator) or accept()/reject() the
dialog, with no CameraStream, build_*(), or save_*() call underneath. A
follow-up pass wires each dialog's accept path to the real
calibration/sensor/, calibration/spatial/ functions referenced in their
docstrings below (see src/pipeline/cli/calibration.py for the reference
call sequence each one mirrors).

Kept separate from calibration_screen.py so the "which forms/dialogs
exist for which calibration type" concern doesn't get lost inside the
top-level screen's page-navigation code.
"""

# Imports

from PySide6.QtCore import Qt
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
        self.continue_button.clicked.connect(self._advance_phase)
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
        if self._phase == self.PHASE_FINISHING:
            self.accept()
            return
        self._phase += 1
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


class SpatialCalibrationDialog(QDialog):

    '''
    Spatial calibration is NOT a camera measurement -- just a numeric
    override for the fixed pixel->physical-position scale factor (see
    calibration/spatial/calibrate.py's DEFAULT_SCALE_FACTOR). This dialog
    is deliberately styled distinctly from the four acquire-driven dialogs
    above (COLOR_ACCENT_ALT instead of COLOR_ACCENT, "Save" instead of
    "Start Capture", no frame-count/gain fields, no phase stepper) since
    it has no acquisition step at all.

    Will eventually call save_scale_factor()/load_scale_factor() and
    construct a ScaleFactorPositionCalibration from the entered value.
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


class SpectralCalibrationDialog(QDialog):

    '''
    Pixel -> wavelength_nm calibration, offering two mutually exclusive
    entry points via a mode selector (a QRadioButton pair) at the top of
    the dialog, each showing its own section of a QStackedWidget below:

    "Capture from Lamp" mirrors BaselineDialog's single-phase capture
    form (frame count + gain_db), plus a fit-degree selector -- will
    eventually call calibration/spectral/workflow.py's
    run_spectral_calibration(camera_stream, n_frames, sensor_calibration,
    path, degree) against an already-built CalibrationSet.

    "Manual Entry" mirrors SpatialCalibrationDialog's no-camera,
    direct-value style, but for a variable-length pixel->wavelength_nm
    polynomial (wavelength_nm = c0 + c1*pixel + c2*pixel^2 + ...,
    ascending order) plus each coefficient's REQUIRED 1-sigma
    uncertainty -- see calibration/spectral/calibrate.py's
    build_manual_spectral_calibration() docstring for why
    coefficient_sigma can't be optional or defaulted (no fit residual
    exists to estimate it from). Changing the degree selector rebuilds
    the coefficient/sigma row list to match.

    Like every other dialog in this file, both modes' accept paths are
    still UI-only placeholders (see module docstring) -- neither
    run_spectral_calibration() nor build_manual_spectral_calibration() is
    actually called yet.
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
        self.start_button = QPushButton("Start Capture")
        self.start_button.setProperty("role", "primary")
        buttons.addButton(self.start_button, QDialogButtonBox.ActionRole)
        self.save_button = QPushButton("Save")
        self.save_button.setProperty("role", "primary")
        buttons.addButton(self.save_button, QDialogButtonBox.ActionRole)
        self.save_button.setVisible(False)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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

        self.gain_db_spin = QDoubleSpinBox()
        self.gain_db_spin.setRange(0.0, 48.0)
        self.gain_db_spin.setSingleStep(0.1)
        self.gain_db_spin.setSuffix(" dB")
        form.addRow("Gain (gain_db):", self.gain_db_spin)

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
