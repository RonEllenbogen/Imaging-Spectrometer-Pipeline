"""
Calibration screen -- the first thing shown on launch. Lets the user
choose between loading existing calibration artifacts from disk, or
creating new ones, per calibration type (not all-or-nothing), then hands
off the resulting calibration objects to whatever opens the main window
next (see CalibrationScreen's class docstring for the hand-off contract).

CreatePage's cards open the real dialogs from calibration_dialogs.py, each
of which now acquires real frames and builds/saves a real artifact under
DEFAULT_ARTIFACT_DIR on its own accept path (see calibration_dialogs.py's
module docstring) -- CreatePage itself only tracks, per calibration type,
whether that dialog has ever been accepted (see CreatePage's own
docstring), to gate "Continue to Main Window".

WelcomePage's "Load Existing Calibrations" flow and CreatePage's "Continue
to Main Window" both end up calling the same method --
CalibrationScreen._attempt_load_existing_calibrations() -- since every
dialog in calibration_dialogs.py already saves to the exact paths that
method reads from: re-reading from disk is simpler and more consistent
than threading each dialog's in-memory result through a second code path.
It attempts to load every artifact from DEFAULT_ARTIFACT_DIR (baseline,
flat field, bad-pixel map, conversion gain, spectral, spatial scale
factor, geometric tilt), with no intermediate review page. Success builds
a CalibrationBundle and emits calibration_ready right away; failure (any
of baseline/flat-field/bad-pixel-map/conversion-gain/spectral missing) or
a baseline/conversion-gain gain_db mismatch shows an error dialog and
leaves the user wherever they were (WelcomePage, or CreatePage for the
"Continue" path). Two artifacts are allowed to be missing without failing
the load: spatial's scale factor always falls back to a physically valid
default, and geometric tilt (built alongside spectral only when it was
captured from the lamp, not entered manually) is simply left unset on
CalibrationSet -- see CalibrationBundle's own docstring.

Bad-pixel-map is deliberately absent from CreatePage's card list -- it has
no manual "create new" entry point anywhere in this screen (see
calibration_dialogs.FlatFieldDialog's docstring).
"""

# Imports

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
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
from pipeline.calibration.exceptions import SettingsMismatchError
from pipeline.calibration.sensor import (
    ConversionGainRecord,
    check_conversion_gain_matches_baseline,
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
from pipeline.calibration.spectral import load_geometric_tilt, load_spectral_calibration
from pipeline.cli.calibration import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BAD_PIXEL_MAP_FILENAME,
    DEFAULT_BASELINE_FILENAME,
    DEFAULT_CONVERSION_GAIN_FILENAME,
    DEFAULT_FLAT_FIELD_FILENAME,
    DEFAULT_GEOMETRIC_TILT_FILENAME,
    DEFAULT_SPECTRAL_FILENAME,
)
from pipeline.gui.calibration_dialogs import (
    BaselineDialog,
    ConversionGainDialog,
    FlatFieldDialog,
    SpatialCalibrationDialog,
    SpectralCalibrationDialog,
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

_SETTINGS_MISMATCH_TITLE = "Calibration Settings Mismatch"
_SETTINGS_MISMATCH_MESSAGE = (
    "Baseline and conversion-gain calibrations were captured at different "
    "gain settings. Recalibrate baseline and/or conversion gain so both "
    "use the same gain before continuing."
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
        relevant artifacts are loaded/built. Its geometric_tilt field is
        populated when a geometric_tilt.npz exists alongside the other
        artifacts, and left None otherwise (see
        _attempt_load_existing_calibrations()) -- unlike the four
        camera-driven artifacts above, a missing geometric tilt
        calibration is not itself a load failure: run_preprocessing()
        already treats CalibrationSet.geometric_tilt=None as "skip that
        correction" cleanly, and a manually-entered spectral calibration
        (build_manual_spectral_calibration(), bypassing lamp capture)
        never produces one at all.
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
        The loaded WavelengthCalibrationResult (implements WavelengthAxis
        directly). Stays Optional at the type level, matching
        calibration_set/noise_model above, but in practice
        _attempt_load_existing_calibrations() treats a missing
        spectral.npz as a load failure the same way it does baseline/
        flat-field/bad-pixel-map/conversion-gain -- a real wavelength
        axis is the whole point of getting this calibration in the first
        place, so it's never silently left None by that flow.
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


class _CalibrationTypeCard(QFrame):

    '''
    One selectable calibration type on CreatePage: title, one-line
    description, and either an action button (enabled types) or a
    disabled placeholder with an explanatory note (see enabled below --
    no card currently uses this).

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
    conversion gain, spatial, and spectral each open their own dialog from
    calibration_dialogs.py. Bad-pixel-map is intentionally not listed (see
    module docstring).

    Tracks, per calibration type, whether its dialog has ever been
    accepted this session (_baseline_done/_flat_field_done/
    _conversion_gain_done/_spectral_done -- a dialog only calls accept()
    once the real acquire-and-save call underneath it actually succeeded,
    see calibration_dialogs.py) and enables continue_button once
    baseline, flat field, conversion gain, and spectral are all done.
    Spatial is deliberately not part of this gate -- it always has a
    physically valid default (DEFAULT_SCALE_FACTOR), so there's nothing
    the user is required to configure before continuing (see
    calibration/spatial/calibrate.py's module docstring). Bad-pixel-map
    is likewise not gated on directly -- it's built automatically as part
    of flat-field calibration (see FlatFieldDialog's docstring).

    continue_requested is emitted when the user clicks "Continue to Main
    Window" (only reachable once every gated type is done) -- see
    CalibrationScreen's own docstring for why its handler is the exact
    same _attempt_load_existing_calibrations() WelcomePage uses, rather
    than a second code path built from these dialogs' in-memory results.
    '''

    back_requested = Signal()
    continue_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._baseline_done = False
        self._flat_field_done = False
        self._conversion_gain_done = False
        self._spectral_done = False

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
            "Create a sensor-column to wavelength mapping using an Argon lamp. "
            "Measure spectral smile.",
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
        self.continue_button.setToolTip(
            "Create baseline, flat field, conversion gain, and spectral calibrations first."
        )
        self.continue_button.clicked.connect(self.continue_requested)
        layout.addWidget(self.continue_button)

    def _update_continue_button(self) -> None:
        '''Enables continue_button once every gated calibration type has
        been created this session -- see class docstring for which types
        gate it and why.'''
        ready = (
            self._baseline_done
            and self._flat_field_done
            and self._conversion_gain_done
            and self._spectral_done
        )
        self.continue_button.setEnabled(ready)

    def _open_baseline_dialog(self) -> None:
        dialog = BaselineDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._baseline_done = True
            self._update_continue_button()

    def _open_flat_field_dialog(self) -> None:
        dialog = FlatFieldDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._flat_field_done = True
            self._update_continue_button()

    def _open_conversion_gain_dialog(self) -> None:
        dialog = ConversionGainDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._conversion_gain_done = True
            self._update_continue_button()

    def _open_spatial_dialog(self) -> None:
        # Spatial is not part of the completion gate (see class docstring)
        # -- no bookkeeping needed on top of the dialog's own save.
        dialog = SpatialCalibrationDialog(DEFAULT_SCALE_FACTOR, self)
        dialog.exec()

    def _open_spectral_dialog(self) -> None:
        dialog = SpectralCalibrationDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._spectral_done = True
            self._update_continue_button()


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

    CreatePage's "Continue to Main Window" reaches calibration_ready via
    the exact same _attempt_load_existing_calibrations() call the
    WelcomePage load path uses (see module docstring) -- CreatePage.
    continue_requested is connected straight to it below, rather than to
    a separate handler.
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
        self.create_page.continue_requested.connect(self._attempt_load_existing_calibrations)
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

        On failure -- baseline, flat field, bad-pixel map, conversion
        gain, or spectral missing -- shows an error dialog and leaves the
        user on WelcomePage. Two artifacts are NOT part of this hard-
        failure check: spatial's scale factor (load_scale_factor() always
        falls back to DEFAULT_SCALE_FACTOR rather than raising) and
        geometric tilt (load_geometric_tilt() raising FileNotFoundError is
        caught separately and just leaves CalibrationSet.geometric_tilt
        as None -- see CalibrationBundle's own docstring for why that's
        the right default, not a failure).

        A second, separate check runs once the four camera-driven
        artifacts load successfully: check_conversion_gain_matches_baseline()
        cross-checks the two independently-recorded gain_db values
        (baseline's and conversion-gain's) against each other, since
        nothing else in this method compares them directly. If they've
        drifted apart by more than GAIN_MATCH_TOLERANCE_ABS, this also
        shows an error dialog and leaves the user on WelcomePage, without
        building or emitting a CalibrationBundle from the
        (settings-inconsistent) artifacts.
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
            wavelength_axis = load_spectral_calibration(
                DEFAULT_ARTIFACT_DIR / DEFAULT_SPECTRAL_FILENAME
            )
        except FileNotFoundError:
            show_calibration_error_dialog(
                self, _NO_EXISTING_CALIBRATIONS_TITLE, _NO_EXISTING_CALIBRATIONS_MESSAGE
            )
            return

        try:
            check_conversion_gain_matches_baseline(baseline_record, conversion_gain_record)
        except SettingsMismatchError:
            show_calibration_error_dialog(self, _SETTINGS_MISMATCH_TITLE, _SETTINGS_MISMATCH_MESSAGE)
            return

        position_calibration, _ = load_scale_factor(
            DEFAULT_ARTIFACT_DIR / _DEFAULT_SCALE_FACTOR_FILENAME
        )
        try:
            geometric_tilt = load_geometric_tilt(
                DEFAULT_ARTIFACT_DIR / DEFAULT_GEOMETRIC_TILT_FILENAME
            )
        except FileNotFoundError:
            geometric_tilt = None

        calibration_set = CalibrationSet(
            baseline=baseline_result.baseline,
            baseline_record=baseline_record,
            flat_field=flat_field,
            flat_field_record=flat_field_record,
            bad_pixel_mask=bad_pixel_mask,
            background_sigma=baseline_result.background_sigma,
            geometric_tilt=geometric_tilt,
        )
        noise_model = SensorNoiseModel(
            gain_e_per_adu=conversion_gain_result.gain_e_per_adu,
            background_sigma=baseline_result.background_sigma,
        )
        self._bundle = CalibrationBundle(
            calibration_set=calibration_set,
            noise_model=noise_model,
            position_calibration=position_calibration,
            wavelength_axis=wavelength_axis,
            conversion_gain_record=conversion_gain_record,
        )
        self.calibration_ready.emit(self._bundle)


__all__ = [
    "CalibrationScreen",
    "CalibrationBundle",
    "WelcomePage",
    "CreatePage",
]
