"""
Calibration screen -- the first thing shown on launch. Lets the user
choose between loading existing calibration artifacts from disk, or
creating new ones, per calibration type (not all-or-nothing), then hands
off the resulting calibration objects to whatever opens the main window
next (see CalibrationScreen's class docstring for the hand-off contract).

PHASE 1 (VISUAL SKELETON) NOTE: CreatePage's cards open the real dialogs
from calibration_dialogs.py, but those dialogs' own accept-paths are still
placeholder -- nothing acquires a frame, builds an artifact, or touches
calibration_artifacts/ on disk yet. A follow-up pass wires:
  - CreatePage's dialog accept-paths to cli/calibration.py's
    build_camera_stream() + calibration/sensor's run_baseline_calibration()/
    capture_dark_frames()/capture_illuminated_frames()/
    finish_flat_field_calibration() (chaining build_bad_pixel_map()+
    save_bad_pixel_map() automatically onto the latter) /
    run_conversion_gain_calibration(), and calibration/spatial's
    save_scale_factor().
  - Camera-connection failures (CameraError and subclasses) to
    calibration_dialogs.show_camera_error_dialog(); calibration-specific
    failures (SettingsMismatchError, InvalidFlatFieldError,
    InvalidConversionGainError, NoSignalError) to
    calibration_dialogs.show_calibration_error_dialog(), leaving the
    originating dialog open so the user can retry.

WelcomePage's "Load Existing Calibrations" flow is fully wired, by
contrast: clicking it immediately attempts to load every artifact from
DEFAULT_ARTIFACT_DIR (baseline, flat field, bad-pixel map, conversion
gain, spatial scale factor) via CalibrationScreen._attempt_load_existing_
calibrations(), with no intermediate review page. Success builds a
CalibrationBundle and emits calibration_ready right away; failure (any of
baseline/flat-field/bad-pixel-map/conversion-gain missing -- spatial's
scale factor always falls back to a physically valid default, so it alone
missing doesn't count) shows an error dialog and leaves the user on
WelcomePage.

Bad-pixel-map is deliberately absent from CreatePage's card list -- it has
no manual "create new" entry point anywhere in this screen (see
calibration_dialogs.FlatFieldDialog's docstring).
"""

# Imports

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
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
from pipeline.calibration.sensor import (
    ConversionGainRecord,
    load_bad_pixel_map,
    load_baseline,
    load_conversion_gain,
    load_flat_field,
)
from pipeline.calibration.spatial import (
    DEFAULT_SCALE_FACTOR,
    ScaleFactorPositionCalibration,
    load_scale_factor,
)
from pipeline.cli.calibration import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BAD_PIXEL_MAP_FILENAME,
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_CONVERSION_GAIN_FILENAME,
    DEFAULT_FLAT_FIELD_FILENAME,
)
from pipeline.gui.calibration_dialogs import (
    BaselineDialog,
    ConversionGainDialog,
    FlatFieldDialog,
    SpatialCalibrationDialog,
    show_calibration_error_dialog,
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
_PAGE_CREATE = 1

# Spatial's scale factor has no filename constant in cli/calibration.py
# (that module only ever built the four camera-driven artifacts) -- named
# here to match its sibling DEFAULT_*_FILENAME constants imported above.
_DEFAULT_SCALE_FACTOR_FILENAME = "scale_factor.npz"

_NO_EXISTING_CALIBRATIONS_TITLE = "No Existing Calibrations"
_NO_EXISTING_CALIBRATIONS_MESSAGE = (
    "No existing calibrations found. Please create new calibrations."
)

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
    CalibrationScreen.calibration_ready (emitted once, either immediately
    after a successful automatic load from WelcomePage or when the user
    clicks "Continue to Main Window" on CreatePage) or by calling
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
        Always None for now -- calibration/spectral/line_matching.py is an
        intentional NotImplementedError stub (see calibration_screen's
        SPECTRAL_UNAVAILABLE_NOTE), so no WavelengthAxis implementation
        exists yet to construct one from.
    conversion_gain_record
        The ConversionGainRecord tagging the loaded conversion-gain
        artifact (gain_db/timestamp/n_illumination_levels it was measured
        under -- see calibration/sensor/conversion_gain.py; deliberately
        has no exposure_us, since exposure is the swept variable, not a
        fixed setting) -- kept separately from noise_model, which only
        retains the derived gain_e_per_adu float. None until a
        conversion-gain artifact is loaded. Exists so a caller (e.g. the
        live-view screen) can compare the sensor's currently-configured
        gain against what the noise model was actually measured under,
        the same way calibration_set.baseline_record already lets
        baseline settings be compared.
    '''

    calibration_set: CalibrationSet | None
    noise_model: SensorNoiseModel | None
    position_calibration: ScaleFactorPositionCalibration
    wavelength_axis: WavelengthAxis | None = None
    conversion_gain_record: ConversionGainRecord | None = None


SPECTRAL_UNAVAILABLE_NOTE = (
    "Unavailable -- calibration/spectral/line_matching.py is not yet "
    "implemented (blocked on reference-lamp selection)."
)


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
        False only for spectral -- renders a greyed-out card with
        SPECTRAL_UNAVAILABLE_NOTE instead of an action button.
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

        self.description_label = QLabel(description if enabled else SPECTRAL_UNAVAILABLE_NOTE)
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

        title = QLabel("Spatial Chirp Diagnostic - Imaging Spectrometer")
        title.setFont(load_bundled_font(22, bold=True))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("LPAG - Ron Ellenbogen")
        subtitle.setProperty("role", "subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
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


class CreatePage(QWidget):

    '''
    Per-type "create new calibration" entry points. Baseline, flat field,
    conversion gain, and spatial each open their own dialog from
    calibration_dialogs.py; spectral renders disabled. Bad-pixel-map is
    intentionally not listed (see module docstring).
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

        cards_row_1 = QHBoxLayout()
        cards_row_1.setSpacing(SPACING_MEDIUM)

        self.baseline_card = _CalibrationTypeCard(
            "Baseline",
            "Compute a baseline frame for background + dark"
            " noise correction.",
            "Configure...",
            COLOR_ACCENT,
        )
        self.baseline_card.action_button.clicked.connect(self._open_baseline_dialog)
        cards_row_1.addWidget(self.baseline_card)

        self.flat_field_card = _CalibrationTypeCard(
            "Flat Field",
            "Compute a flat field for PRNU correction.",
            "Configure...",
            COLOR_ACCENT,
        )
        self.flat_field_card.action_button.clicked.connect(self._open_flat_field_dialog)
        cards_row_1.addWidget(self.flat_field_card)

        self.conversion_gain_card = _CalibrationTypeCard(
            "Conversion Gain",
            "Compute the sensor's conversion gain for uncertainty"
            " calculations.",
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
            "Input an externally calibrated value for the Imaging"
            " Spectrometer's inverse spatial magnification, or use the"
            " theoretical default.",
            "Enter Value...",
            COLOR_ACCENT_ALT,
        )
        self.spatial_card.action_button.clicked.connect(self._open_spatial_dialog)
        cards_row_2.addWidget(self.spatial_card)

        self.spectral_card = _CalibrationTypeCard(
            "Spectral",
            "",
            "",
            COLOR_ACCENT,
            enabled=False,
        )
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


class CalibrationScreen(QWidget):

    '''
    Top-level calibration screen -- the first thing shown on launch.
    Owns page navigation between WelcomePage and CreatePage, plus the
    "load existing calibrations" flow triggered from WelcomePage (which
    has no page of its own -- see below).

    Hand-off contract for whatever opens the main window next
    -----------------------------------------------------------
    This screen emits `calibration_ready` exactly once, carrying a
    CalibrationBundle, either:
      - immediately, if WelcomePage's "Load Existing Calibrations" finds
        every required artifact in DEFAULT_ARTIFACT_DIR and loads
        successfully (see _attempt_load_existing_calibrations()), or
      - when the user clicks "Continue to Main Window" on CreatePage
        after creating whatever new calibrations they wanted.
    A caller that would rather poll than connect to the signal can call
    get_calibration_bundle() afterward -- it returns the same object, or
    None if neither path above has completed yet.

    PHASE 1: CreatePage's "Continue to Main Window" is still a disabled
    placeholder (see CreatePage) -- wiring it up is a follow-up pass, once
    real build_*() calls replace CreatePage's dialogs' placeholder accept
    paths (see module docstring). The load path above is fully wired.
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
        self.create_page = CreatePage()

        self._stack.addWidget(self.welcome_page)
        self._stack.addWidget(self.create_page)

        self.welcome_page.load_requested.connect(self._attempt_load_existing_calibrations)
        self.welcome_page.create_requested.connect(
            lambda: self._stack.setCurrentIndex(_PAGE_CREATE)
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

    def _attempt_load_existing_calibrations(self) -> None:

        '''
        Handler for WelcomePage.load_requested: loads every calibration
        artifact from DEFAULT_ARTIFACT_DIR immediately, with no
        intermediate review page -- reuses each subpackage's own
        load_*() function (the same calls cli/calibration.py's
        _cmd_noise_model sequences) rather than duplicating any loading
        logic here.

        On success, builds a CalibrationBundle (constructing a
        SensorNoiseModel from the loaded baseline + conversion gain,
        exactly as _cmd_noise_model does) and emits calibration_ready.

        On failure -- baseline, flat field, bad-pixel map, or conversion
        gain missing -- shows an error dialog and leaves the user on
        WelcomePage. Spatial's scale factor is not part of this check:
        load_scale_factor() always falls back to DEFAULT_SCALE_FACTOR
        rather than raising, so it alone missing is never a failure.
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
            conversion_gain_result, conversion_gain_record = load_conversion_gain(
                DEFAULT_ARTIFACT_DIR / DEFAULT_CONVERSION_GAIN_FILENAME
            )
        except FileNotFoundError:
            show_calibration_error_dialog(
                self, _NO_EXISTING_CALIBRATIONS_TITLE, _NO_EXISTING_CALIBRATIONS_MESSAGE
            )
            return

        position_calibration, _ = load_scale_factor(
            DEFAULT_ARTIFACT_DIR / _DEFAULT_SCALE_FACTOR_FILENAME
        )

        calibration_set = CalibrationSet(
            baseline=baseline_result.baseline,
            baseline_record=baseline_record,
            flat_field=flat_field,
            flat_field_record=flat_field_record,
            bad_pixel_mask=bad_pixel_mask,
            background_sigma=baseline_result.background_sigma,
        )
        noise_model = SensorNoiseModel(
            gain_e_per_adu=conversion_gain_result.gain_e_per_adu,
            background_sigma=baseline_result.background_sigma,
        )
        self._bundle = CalibrationBundle(
            calibration_set=calibration_set,
            noise_model=noise_model,
            position_calibration=position_calibration,
            conversion_gain_record=conversion_gain_record,
        )
        self.calibration_ready.emit(self._bundle)


__all__ = [
    "CalibrationScreen",
    "CalibrationBundle",
    "WelcomePage",
    "CreatePage",
    "SPECTRAL_UNAVAILABLE_NOTE",
]
