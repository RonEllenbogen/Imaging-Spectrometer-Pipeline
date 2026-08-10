"""
Calibration screen -- the first thing shown on launch. Lets the user
choose between loading existing calibration artifacts from disk, or
creating new ones, per calibration type (not all-or-nothing), then hands
off the resulting calibration objects to whatever opens the main window
next (see CalibrationScreen's class docstring for the hand-off contract).

PHASE 1 (VISUAL SKELETON) NOTE: this module currently lays out and styles
every screen/page/card described below with placeholder/dummy data. No
camera or calibration package call is made anywhere in this file yet --
CreatePage's cards open the real dialogs from calibration_dialogs.py, and
LoadPage's rows show static placeholder status, but nothing acquires a
frame, builds an artifact, or touches calibration_artifacts/ on disk. A
follow-up pass wires:
  - LoadPage's "Browse"/row logic to calibration/sensor's load_baseline()/
    load_flat_field()/load_bad_pixel_map()/load_conversion_gain() and
    calibration/spatial's load_scale_factor(), plus constructing a real
    SensorNoiseModel from the loaded baseline + conversion-gain (see
    cli/calibration.py's _cmd_noise_model for the exact call sequence).
  - CreatePage's dialog accept-paths to cli/calibration.py's
    build_camera_stream() + calibration/sensor's run_baseline_calibration()/
    capture_dark_frames()/capture_illuminated_frames()/
    finish_flat_field_calibration() (chaining build_bad_pixel_map()+
    save_bad_pixel_map() automatically onto the latter) /
    run_conversion_gain_calibration(), calibration/spatial's
    save_scale_factor(), and calibration/spectral's
    run_spectral_calibration()/build_manual_spectral_calibration().
  - Camera-connection failures (CameraError and subclasses) to
    calibration_dialogs.show_camera_error_dialog(); calibration-specific
    failures (SettingsMismatchError, InvalidFlatFieldError,
    InvalidConversionGainError, NoSignalError) to
    calibration_dialogs.show_calibration_error_dialog(), leaving the
    originating dialog open so the user can retry.

Bad-pixel-map is deliberately absent from CreatePage's card list -- it has
no manual "create new" entry point anywhere in this screen (see
calibration_dialogs.FlatFieldDialog's docstring).
"""

# Imports

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pipeline.analysis import SensorNoiseModel, WavelengthAxis
from pipeline.calibration.spatial import DEFAULT_SCALE_FACTOR, ScaleFactorPositionCalibration
from pipeline.gui.calibration_dialogs import (
    BaselineDialog,
    ConversionGainDialog,
    FlatFieldDialog,
    SpatialCalibrationDialog,
    SpectralCalibrationDialog,
)
from pipeline.gui.theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_ALT,
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_ERROR,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_SURFACE_ALT,
    COLOR_TEXT_DISABLED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    MARGIN_DEFAULT,
    SPACING_LARGE,
    SPACING_MEDIUM,
    SPACING_SMALL,
    load_bundled_font,
)
from pipeline.preprocessing.preprocessing_pipeline import CalibrationSet

# Constants

_PAGE_WELCOME = 0
_PAGE_LOAD = 1
_PAGE_CREATE = 2

# Placeholder rows shown on LoadPage: (display name, artifact description).
# Status/paths are dummy data in Phase 1 -- see module docstring.
_LOAD_ROWS = [
    ("Baseline", "background_sigma + per-pixel dark offset"),
    ("Flat field", "PRNU correction"),
    ("Bad-pixel map", "dead/hot pixel mask"),
    ("Conversion gain", "gain_e_per_adu"),
    ("Spatial scale factor", "pixel -> physical position"),
]

_SCREEN_STYLE = f"""
QWidget#CalibrationScreen {{
    background-color: {COLOR_BACKGROUND};
}}
QLabel {{
    color: {COLOR_TEXT_PRIMARY};
}}
QLabel[role="subtitle"] {{
    color: {COLOR_TEXT_SECONDARY};
}}
QLabel[role="hint"] {{
    color: {COLOR_TEXT_SECONDARY};
}}
QFrame[role="card"] {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
}}
QFrame[role="card-disabled"] {{
    background-color: {COLOR_SURFACE_ALT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
}}
QFrame[role="row"] {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
}}
QPushButton {{
    background-color: {COLOR_SURFACE_ALT};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    border-color: {COLOR_ACCENT};
}}
QPushButton:disabled {{
    color: {COLOR_TEXT_DISABLED};
}}
QPushButton[role="primary"] {{
    background-color: {COLOR_ACCENT};
    color: #10101a;
    font-weight: 600;
    border: none;
}}
QPushButton[role="primary"]:disabled {{
    background-color: {COLOR_SURFACE_ALT};
    color: {COLOR_TEXT_DISABLED};
}}
QPushButton[role="primary-alt"] {{
    background-color: {COLOR_ACCENT_ALT};
    color: #14101a;
    font-weight: 600;
    border: none;
}}
QPushButton[role="primary-alt"]:disabled {{
    background-color: {COLOR_SURFACE_ALT};
    color: {COLOR_TEXT_DISABLED};
}}
QPushButton[role="nav"] {{
    background-color: transparent;
    border: none;
    color: {COLOR_TEXT_SECONDARY};
}}
QPushButton[role="nav"]:hover {{
    color: {COLOR_TEXT_PRIMARY};
}}
"""

# Classes


@dataclass(frozen=True, slots=True)
class CalibrationBundle:

    '''
    Everything CalibrationScreen hands off once the user is done loading
    or creating calibrations. Retrieve it either by connecting to
    CalibrationScreen.calibration_ready (emitted once, when the user
    clicks "Continue to Main Window") or by calling
    CalibrationScreen.get_calibration_bundle() afterward -- both give the
    same object.

    Parameters
    ----------
    calibration_set
        Baseline + flat field + bad-pixel mask (+ background_sigma), for
        pipeline.preprocessing.run_preprocessing(). None until the
        relevant artifacts are loaded/built.
    noise_model
        SensorNoiseModel(gain_e_per_adu=..., background_sigma=...), built
        from the loaded baseline + conversion-gain artifacts. None until
        both exist.
    position_calibration
        ScaleFactorPositionCalibration wrapping either DEFAULT_SCALE_FACTOR
        or a manually-entered override. Always constructible (unlike the
        other fields, a physically valid default always exists -- see
        calibration/spatial/calibrate.py).
    wavelength_axis
        Always None for now -- SpectralCalibrationDialog's accept path is
        still a UI-only placeholder (see calibration_dialogs.py's module
        docstring), so no real WavelengthCalibrationResult is built yet
        to construct one from.
    '''

    calibration_set: CalibrationSet | None
    noise_model: SensorNoiseModel | None
    position_calibration: ScaleFactorPositionCalibration
    wavelength_axis: WavelengthAxis | None = None


class _CalibrationTypeCard(QFrame):

    '''
    One selectable calibration type on CreatePage: title, one-line
    description, and either an action button (enabled types) or a
    disabled placeholder with an explanatory note (spectral only).

    Parameters
    ----------
    title
        Calibration type name, e.g. "Baseline".
    description
        One-line summary of what it measures/produces.
    action_text
        Button label, e.g. "Configure...". Ignored when enabled=False.
    accent
        Button/border accent color -- COLOR_ACCENT for the four
        camera-driven types, COLOR_ACCENT_ALT for spatial (see
        calibration_dialogs.SpatialCalibrationDialog's docstring).
    enabled
        False renders a greyed-out card with an "Unavailable" button
        instead of a real action button. No card currently uses this --
        all five calibration types now have their own dialog -- but the
        option is kept for a future calibration type that isn't ready
        yet.
    '''

    def __init__(
        self,
        title: str,
        description: str,
        action_text: str,
        accent: str,
        enabled: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("role", "card" if enabled else "card-disabled")
        self.setFrameShape(QFrame.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_MEDIUM, SPACING_MEDIUM, SPACING_MEDIUM, SPACING_MEDIUM)
        layout.setSpacing(SPACING_SMALL)

        title_label = QLabel(title)
        title_label.setFont(load_bundled_font(13, bold=True))
        if not enabled:
            title_label.setStyleSheet(f"color: {COLOR_TEXT_DISABLED};")
        layout.addWidget(title_label)

        self.description_label = QLabel(description if enabled else "Unavailable.")
        self.description_label.setProperty("role", "hint")
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        layout.addStretch(1)

        self.action_button = QPushButton(action_text if enabled else "Unavailable")
        self.action_button.setEnabled(enabled)
        if enabled:
            role = "primary" if accent == COLOR_ACCENT else "primary-alt"
            self.action_button.setProperty("role", role)
        layout.addWidget(self.action_button)


class WelcomePage(QWidget):

    '''First page shown: choose "load existing" or "create new".'''

    load_requested = Signal()
    create_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_LARGE, SPACING_LARGE, SPACING_LARGE, SPACING_LARGE)
        layout.setSpacing(SPACING_LARGE)
        layout.addStretch(1)

        title = QLabel("Spectrometer Calibration")
        title.setFont(load_bundled_font(22, bold=True))
        layout.addWidget(title)

        subtitle = QLabel(
            "Load calibration artifacts from a previous session, or create "
            "new ones before starting acquisition."
        )
        subtitle.setProperty("role", "subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        button_row = QHBoxLayout()
        button_row.setSpacing(SPACING_MEDIUM)

        self.load_button = QPushButton("Load Existing Calibrations")
        self.load_button.setProperty("role", "primary")
        self.load_button.setMinimumHeight(48)
        self.load_button.clicked.connect(self.load_requested)
        button_row.addWidget(self.load_button)

        self.create_button = QPushButton("Create New Calibrations")
        self.create_button.setProperty("role", "primary-alt")
        self.create_button.setMinimumHeight(48)
        self.create_button.clicked.connect(self.create_requested)
        button_row.addWidget(self.create_button)

        layout.addLayout(button_row)
        layout.addStretch(2)


class LoadPage(QWidget):

    '''
    Lists the five loadable calibration artifacts (baseline, flat field,
    bad-pixel mask, conversion gain, scale factor) with per-row status and
    a browse action, plus a summary of the resulting SensorNoiseModel.
    Phase 1: all status/values are placeholder dummy data (see module
    docstring) -- rows do not yet call any load_*() function.
    '''

    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_LARGE, SPACING_LARGE, SPACING_LARGE, SPACING_LARGE)
        layout.setSpacing(SPACING_MEDIUM)

        back_button = QPushButton("< Back")
        back_button.setProperty("role", "nav")
        back_button.clicked.connect(self.back_requested)
        layout.addWidget(back_button)

        title = QLabel("Load Existing Calibrations")
        title.setFont(load_bundled_font(18, bold=True))
        layout.addWidget(title)

        subtitle = QLabel("From calibration_artifacts/ (default location).")
        subtitle.setProperty("role", "subtitle")
        layout.addWidget(subtitle)

        for name, description in _LOAD_ROWS:
            layout.addWidget(self._build_row(name, description))

        layout.addSpacing(SPACING_SMALL)

        summary = QFrame()
        summary.setProperty("role", "row")
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(
            SPACING_MEDIUM, SPACING_MEDIUM, SPACING_MEDIUM, SPACING_MEDIUM
        )
        summary_heading = QLabel("Sensor Noise Model")
        summary_heading.setFont(load_bundled_font(12, bold=True))
        summary_layout.addWidget(summary_heading)
        self.noise_model_label = QLabel("gain_e_per_adu: --      background_sigma: --")
        self.noise_model_label.setProperty("role", "hint")
        summary_layout.addWidget(self.noise_model_label)
        layout.addWidget(summary)

        layout.addStretch(1)

        self.continue_button = QPushButton("Continue to Main Window")
        self.continue_button.setProperty("role", "primary")
        self.continue_button.setEnabled(False)
        self.continue_button.setToolTip("Load at least baseline and flat field first.")
        layout.addWidget(self.continue_button)

    def _build_row(self, name: str, description: str) -> QFrame:
        row = QFrame()
        row.setProperty("role", "row")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(
            SPACING_MEDIUM, SPACING_SMALL, SPACING_MEDIUM, SPACING_SMALL
        )
        row_layout.setSpacing(SPACING_MEDIUM)

        text_col = QVBoxLayout()
        name_label = QLabel(name)
        name_label.setFont(load_bundled_font(11, bold=True))
        text_col.addWidget(name_label)
        desc_label = QLabel(description)
        desc_label.setProperty("role", "hint")
        text_col.addWidget(desc_label)
        row_layout.addLayout(text_col, stretch=1)

        status_label = QLabel("Not loaded")
        status_label.setStyleSheet(f"color: {COLOR_TEXT_DISABLED};")
        row_layout.addWidget(status_label)

        browse_button = QPushButton("Browse...")
        row_layout.addWidget(browse_button)

        return row


class CreatePage(QWidget):

    '''
    Per-type "create new calibration" entry points. Baseline, flat field,
    conversion gain, spatial, and spectral each open their own dialog from
    calibration_dialogs.py. Bad-pixel-map is intentionally not listed (see
    module docstring).
    '''

    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_LARGE, SPACING_LARGE, SPACING_LARGE, SPACING_LARGE)
        layout.setSpacing(SPACING_MEDIUM)

        back_button = QPushButton("< Back")
        back_button.setProperty("role", "nav")
        back_button.clicked.connect(self.back_requested)
        layout.addWidget(back_button)

        title = QLabel("Create New Calibrations")
        title.setFont(load_bundled_font(18, bold=True))
        layout.addWidget(title)

        subtitle = QLabel(
            "Each type is created independently -- pick as many as you need."
        )
        subtitle.setProperty("role", "subtitle")
        layout.addWidget(subtitle)

        cards_row_1 = QHBoxLayout()
        cards_row_1.setSpacing(SPACING_MEDIUM)

        self.baseline_card = _CalibrationTypeCard(
            "Baseline",
            "Single-phase: averages background frames for dark offset "
            "and background_sigma.",
            "Configure...",
            COLOR_ACCENT,
        )
        self.baseline_card.action_button.clicked.connect(self._open_baseline_dialog)
        cards_row_1.addWidget(self.baseline_card)

        self.flat_field_card = _CalibrationTypeCard(
            "Flat Field",
            "Two-phase: dark, then uniformly illuminated frames. "
            "Bad-pixel map builds automatically afterward.",
            "Configure...",
            COLOR_ACCENT,
        )
        self.flat_field_card.action_button.clicked.connect(self._open_flat_field_dialog)
        cards_row_1.addWidget(self.flat_field_card)

        self.conversion_gain_card = _CalibrationTypeCard(
            "Conversion Gain",
            "Exposure sweep at fixed illumination to fit gain_e_per_adu.",
            "Configure...",
            COLOR_ACCENT,
        )
        self.conversion_gain_card.action_button.clicked.connect(
            self._open_conversion_gain_dialog
        )
        cards_row_1.addWidget(self.conversion_gain_card)

        layout.addLayout(cards_row_1)

        cards_row_2 = QHBoxLayout()
        cards_row_2.setSpacing(SPACING_MEDIUM)

        self.spatial_card = _CalibrationTypeCard(
            "Spatial",
            "No camera interaction -- manually enter a scale-factor "
            "override, or use the physical default.",
            "Enter Value...",
            COLOR_ACCENT_ALT,
        )
        self.spatial_card.action_button.clicked.connect(self._open_spatial_dialog)
        cards_row_2.addWidget(self.spatial_card)

        self.spectral_card = _CalibrationTypeCard(
            "Spectral",
            "Fits pixel-to-wavelength from an Argon lamp capture, or "
            "accepts a manually measured polynomial instead.",
            "Configure...",
            COLOR_ACCENT,
        )
        self.spectral_card.action_button.clicked.connect(self._open_spectral_dialog)
        cards_row_2.addWidget(self.spectral_card)

        cards_row_2.addStretch(1)
        layout.addLayout(cards_row_2)

        layout.addStretch(1)

        self.continue_button = QPushButton("Continue to Main Window")
        self.continue_button.setProperty("role", "primary")
        self.continue_button.setEnabled(False)
        self.continue_button.setToolTip("Create at least baseline and flat field first.")
        layout.addWidget(self.continue_button)

    def _open_baseline_dialog(self) -> None:
        dialog = BaselineDialog(self)
        dialog.exec()

    def _open_flat_field_dialog(self) -> None:
        dialog = FlatFieldDialog(self)
        dialog.exec()

    def _open_conversion_gain_dialog(self) -> None:
        dialog = ConversionGainDialog(self)
        dialog.exec()

    def _open_spatial_dialog(self) -> None:
        dialog = SpatialCalibrationDialog(DEFAULT_SCALE_FACTOR, self)
        dialog.exec()

    def _open_spectral_dialog(self) -> None:
        dialog = SpectralCalibrationDialog(self)
        dialog.exec()


class CalibrationScreen(QWidget):

    '''
    Top-level calibration screen -- the first thing shown on launch.
    Owns page navigation between WelcomePage, LoadPage, and CreatePage.

    Hand-off contract for whatever opens the main window next
    -----------------------------------------------------------
    Once the user finishes (loads existing artifacts, or creates whatever
    new calibrations they wanted) and clicks "Continue to Main Window" on
    either LoadPage or CreatePage, this screen emits `calibration_ready`
    exactly once, carrying a CalibrationBundle. A caller that would rather
    poll than connect to the signal can call get_calibration_bundle()
    afterward -- it returns the same object, or None if the user hasn't
    reached "Continue" yet.

    PHASE 1: get_calibration_bundle() always returns None right now, and
    calibration_ready is never emitted -- both continue buttons are
    disabled placeholders (see LoadPage/CreatePage). Wiring them up is a
    follow-up pass, once real load_*()/build_*() calls replace the dummy
    data described in this module's docstring.
    '''

    calibration_ready = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CalibrationScreen")
        self.setStyleSheet(_SCREEN_STYLE)

        self._bundle: CalibrationBundle | None = None

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        outer_layout.addWidget(self._stack)

        self.welcome_page = WelcomePage()
        self.load_page = LoadPage()
        self.create_page = CreatePage()

        self._stack.addWidget(self.welcome_page)
        self._stack.addWidget(self.load_page)
        self._stack.addWidget(self.create_page)

        self.welcome_page.load_requested.connect(lambda: self._stack.setCurrentIndex(_PAGE_LOAD))
        self.welcome_page.create_requested.connect(
            lambda: self._stack.setCurrentIndex(_PAGE_CREATE)
        )
        self.load_page.back_requested.connect(
            lambda: self._stack.setCurrentIndex(_PAGE_WELCOME)
        )
        self.create_page.back_requested.connect(
            lambda: self._stack.setCurrentIndex(_PAGE_WELCOME)
        )

    def get_calibration_bundle(self) -> CalibrationBundle | None:

        '''
        Returns the CalibrationBundle handed off when the user finished
        this screen, or None if they haven't reached "Continue" yet. See
        class docstring for the full hand-off contract.
        '''

        return self._bundle


__all__ = [
    "CalibrationScreen",
    "CalibrationBundle",
    "WelcomePage",
    "LoadPage",
    "CreatePage",
]
