'''
Extended-measurement interface: the N-shot combination workflow reached
from live_view.py's "Extended Measurement..." button. Static scatter plot
of per-shot centroid positions vs. wavelength (or pixel column, if no
wavelength calibration exists yet) with an overlaid fit line, a residual
subplot underneath it, a side info panel, and a combined-result summary --
in place of live_view.py's rolling strip chart, since this screen shows
one already-combined result rather than a live-updating trend
(docs/project_state.md #5's "Extended measurement" notes).

"Run Measurement" is a single synchronous, blocking operation -- click,
wait, get a result -- unlike live_view.py's QTimer-driven polling loop
(the 5Hz target refresh rate in docs/project_state.md is a live_view.py
concern, not this screen's). It acquires
camera_stream.collect_n_frames(n_shots), runs every frame through
run_preprocessing()/analyze_shot() at every DEGREE_CHOICES degree at once
(so switching the degree selector afterward is free -- no re-acquisition
needed), and combines the per-shot results via
pipeline.analysis.combination.combine_shots(). If the Acquisition Settings
panel's exposure/gain no longer match what camera_stream is actually
configured with, this first runs the same stop -> mutate exposure_us/
gain_db -> start cycle calibration/sensor/conversion_gain.py's
run_conversion_gain_calibration() uses to sweep exposure -- freezing live
view (if it shares this stream) for the acquisition's duration, an
accepted, expected interruption (see docs/project_state.md's "Extended
measurement" notes), not something to work around.

Analysis is always kept in pixel units -- analyze_shot() is never given a
position_calibration (see analysis/interfaces.py's PositionCalibration
docstring: results stay in pixel units whenever it's omitted, the
convention every real caller in this codebase already follows, e.g.
scripts/analyze_raw_shot.py/scripts/measure_spatial_dispersion.py).
self._position_calibration/_convert_to_mm() only convert already-computed
pixel-unit results to mm for display, the same duality live_view.py uses.
zeta_combined specifically -- the one value with its own side-panel
display -- goes through the analogous _zeta_to_mm() at that one display
call site only; every internal consumer (_recompute_fit_and_residuals(),
which redraws the fit line/residuals against self._measurement_x0_px)
keeps using the raw px/nm value returned by combine_shots().
When no real wavelength calibration is loaded (self._wavelength_axis is
None), _PixelColumnWavelengthAxis stands in for it so analyze_shot() still
has something to fit against -- pixel column itself, mirroring this
screen's existing pixel-column x-axis fallback (wavelength_axis_label())
and scripts/analyze_raw_shot.py's own PixelColumnAxis.

Degree 1 combination uses combine_shots() directly on each shot's fitted
linear zeta (coefficients[1]/coefficient_sigma[1]) -- exactly what it was
built for. combine_shots() deliberately does NOT aggregate quadratic/cubic
fits (see combination.py's module docstring): there is no single "combined
quadratic/cubic curve", since only a linear zeta is combined by design.
For degree > 1, each shot's own zeta(wavelength_ref)/
sigma_zeta(wavelength_ref) (wavelength_ref = self._evaluated_at_spin's
value; a well-defined scalar at any degree, using that shot's own
already-computed fit and its full coefficient_covariance -- see
SpatialDispersionFitResult.sigma_zeta()) is combined across shots via that
exact same combine_shots() inverse-variance weighting, since that function
is generic over any (value, sigma) pairs, not specific to the linear-fit
coefficients it's usually called with.

The main-plot fit curve/residual subplot are built from that same combined
zeta -- never a refit of per-column-averaged centroid positions. This
codebase already considered and rejected that combination methodology when
analysis/combination.py was designed (see its module docstring): refitting
column averages could visually disagree with the reported zeta_combined
sitting right next to it. Instead, every raw (shot, column) centroid point
is plotted/residualed against one straight line of slope zeta_combined,
anchored through the data by a plain weighted-least-squares intercept (the
best-fit offset given that fixed, already-combined slope) -- so the drawn
line's slope and the reported number can never visually contradict each
other, at any degree.

The Acquisition Settings side-panel section (exposure_us/gain_db spin
boxes, pre-filled from the loaded baseline's capture settings) feeds the
same drift check (exposure_has_drifted()/gain_has_drifted(), imported from
live_view.py) against calibration_set.baseline_record (and
conversion_gain_record, if supplied). Crossing into the drifted state N/As
the Combined Result group and hides the scatter/error-bar/fit-curve/
residual overlay, pops one informational message, and emits
recalibration_requested -- mirroring LiveViewWidget's
_recompute_settings_drift()/_enter_drifted_state()/_exit_drifted_state()
with this screen's own target widgets swapped in. This drift check is
independent of (and does not block) the exposure/gain reconfigure cycle
above -- one governs whether the loaded calibrations are still trustworthy
for the entered settings, the other just gets the camera itself onto those
settings before acquiring.
'''

# Imports

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pipeline.acquisition import CameraError, CameraStream, CANONICAL_SHAPE, SPATIAL_AXIS, SPECTRAL_AXIS
from pipeline.analysis import (
    analyze_shot, combine_shots, CombinedSpatialDispersionResult,
    InsufficientDataError, SensorNoiseModel, ShotAnalysisResult,
)
from pipeline.analysis.interfaces import WavelengthAxis
from pipeline.calibration.sensor import ConversionGainRecord
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.preprocessing import (
    CalibrationSet, NoSignalError, SettingsMismatchError, run_preprocessing,
)

from pipeline.gui.calibration_dialogs import show_camera_error_dialog
from pipeline.gui.formatting import format_value_with_uncertainty, microns_to_mm
from pipeline.gui.live_view import (
    DEFAULT_DEGREE,
    DEGREE_CHOICES,
    DEGREE_LABELS,
    EVALUATED_AT_COLUMN,
    FIT_CURVE_COLOR,
    FIT_CURVE_WIDTH,
    SIDE_PANEL_WIDTH,
    evaluated_at_text,
    exposure_has_drifted,
    fit_formula_html,
    gain_has_drifted,
    wavelength_axis_label,
)
from pipeline.gui.roi_control import SpatialROIControl
from pipeline.gui.theme import (
    COLOR_ACCENT as ACCENT_COLOR,
    COLOR_BACKGROUND as BACKGROUND_COLOR,
    COLOR_TEXT_PRIMARY as FOREGROUND_COLOR,
    COLOR_PLOT_GRID as GRID_COLOR,
    combo_box_stylesheet,
    group_box_stylesheet,
    load_bundled_font,
)


# Constants

# Default shot count pre-filled into the "Number of Shots" spin box.
# Deliberately a new, separate constant from calibration/'s
# DEFAULT_N_FRAMES (cli/calibration.py, gui/calibration_dialogs.py) --
# "shots" here means independently fit-then-combined measurements
# (analysis/combination.py's combine_shots() input), a different concept
# from the frames averaged together into one calibration artifact.
DEFAULT_N_SHOTS = 20

MIN_N_SHOTS = 2
MAX_N_SHOTS = 1000

# Residual subplot color -- distinct from the main scatter's white/
# COLOR_ACCENT so a glance at the two plots doesn't read as one repeated
# dataset.
RESIDUAL_SCATTER_COLOR = "#ff9d5c"

# Number of points sampled along the drawn fit-curve line -- a display
# resolution choice only, unrelated to the number of shots/columns
# actually combined.
FIT_CURVE_N_POINTS = 200

# analyze_shot()'s wavelength-axis abstraction requires a strictly
# positive sigma on both axes (scipy.odr divides by it). Pixel column has
# no real calibrated uncertainty of its own in the no-wavelength-
# calibration fallback -- this is a placeholder small enough to be
# negligible next to sigma_x0, not a measured quantity. Same value/
# rationale as scripts/analyze_raw_shot.py's PIXEL_COLUMN_SIGMA.
PIXEL_COLUMN_SIGMA = 1e-3

# Static caveat shown next to the Combined Result panel for degree > 1:
# explains the combination *methodology* (per-shot scalar evaluation, not
# a joint polynomial fit) -- not an uncertainty caveat. The previous
# "external uncertainty only" interim caveat (docs/project_state.md) no
# longer applies now that sigma_zeta_combined is built from a real
# sigma_zeta() call (the full covariance-propagated uncertainty, combined
# internal/external the same way as degree 1), so there is nothing left
# to disclaim about the number itself.
DEGREE_GT_ONE_NOTE = (
    "Degree > 1: each shot's spatial dispersion is evaluated at "
    "the reference point above, then those values are combined "
    "-- not a joint polynomial fit across shots."
)


# Classes

class _PixelColumnWavelengthAxis:

    '''
    Stand-in WavelengthAxis used whenever no real spectral calibration is
    loaded (self._wavelength_axis is None): treats pixel-column index
    itself as analyze_shot()'s fit independent variable, matching this
    screen's existing pixel-column x-axis fallback
    (wavelength_axis_label()/_x_axis_label()). Structurally identical to
    scripts/analyze_raw_shot.py's PixelColumnAxis -- duplicated rather
    than imported, since that script lives outside pipeline/ and importing
    from scripts/ into src/pipeline/ would invert the project's layering.
    '''

    def wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:
        return pixel.astype(np.float64)

    def sigma_wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:
        return np.full(pixel.shape, PIXEL_COLUMN_SIGMA, dtype=np.float64)


class ExtendedMeasurementScreen(QWidget):

    '''
    Extended-measurement widget: scatter + fit overlay + residual subplot
    on top, a side info panel summarizing an N-shot combined result, and a
    "Back to Live View" button -- wired around (not responsible for
    producing) an already-assembled calibration set, noise model, position
    calibration, optional wavelength axis, and a running camera stream.

    Parameters
    ----------
    calibration_set
        Preprocessing artifacts (baseline, flat field, bad-pixel mask,
        background_sigma) -- see pipeline.preprocessing.CalibrationSet.
        Building/loading it is the calibration screen's job, not this
        widget's.
    noise_model
        Sensor noise parameters (gain, background sigma) for
        analyze_shot()'s Thompson-Larson-Webb centroid uncertainty.
    position_calibration
        Pixel -> physical-position conversion at the spectrometer's slit
        (always real -- calibration/spatial/'s fixed scale factor has a
        physically valid default even with no manual override). Used only
        to convert already-computed pixel-unit results to mm for display
        -- never fed into analyze_shot() itself (see module docstring).
    wavelength_axis
        Pixel -> wavelength(nm) conversion, or None until
        calibration/spectral/'s line_matching.py output is loaded (see
        docs/project_state.md). None is the expected v1 state, not an
        error -- analyze_shot() then fits against pixel column itself via
        _PixelColumnWavelengthAxis, and the scatter falls back to a
        pixel-column x-axis, clearly labeled as such.
    camera_stream
        A CameraStream the caller owns the lifecycle of (start/stop are
        not this widget's responsibility, except for the exposure/gain
        reconfigure cycle "Run Measurement" runs when needed -- see module
        docstring). Must already be running by the time "Run Measurement"
        is clicked -- collect_n_frames() requires it.
    conversion_gain_record
        ConversionGainRecord tagging the loaded conversion-gain artifact,
        or None if no conversion-gain artifact was loaded. When supplied,
        the Acquisition Settings panel's gain_db field is drift-checked
        against it in addition to calibration_set.baseline_record.gain_db,
        same as live_view.py's LiveViewWidget.
    parent
        Standard Qt parent widget.
    '''

    back_requested = Signal()
    recalibration_requested = Signal(str)

    def __init__(
        self,
        calibration_set: CalibrationSet,
        noise_model: SensorNoiseModel,
        position_calibration: ScaleFactorPositionCalibration,
        wavelength_axis: WavelengthAxis | None,
        camera_stream: CameraStream,
        conversion_gain_record: ConversionGainRecord | None = None,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(parent)

        self._calibration_set = calibration_set
        self._noise_model = noise_model
        self._position_calibration = position_calibration
        self._wavelength_axis = wavelength_axis
        self._camera_stream = camera_stream
        self._conversion_gain_record = conversion_gain_record

        # The axis analyze_shot() actually fits against -- the real
        # wavelength_axis when one is loaded, or the pixel-column
        # fallback otherwise (see module docstring and
        # _PixelColumnWavelengthAxis). Fixed for this widget's lifetime,
        # same as wavelength_axis itself.
        self._axis_for_fit: WavelengthAxis = (
            wavelength_axis if wavelength_axis is not None else _PixelColumnWavelengthAxis()
        )

        self._current_degree = DEFAULT_DEGREE

        # None until "Run Measurement" completes at least once -- every
        # display method below treats this as "nothing to show yet"
        # rather than fabricating placeholder numbers (see
        # _refresh_measurement_display()).
        self._shot_results: list[ShotAnalysisResult] | None = None

        self._apply_pyqtgraph_theme()
        self._build_ui()
        self._refresh_measurement_display()

        # Self-consistent even if a caller constructs this widget directly
        # with an already-mismatched baseline/conversion-gain pair -- see
        # LiveViewWidget.__init__'s identical comment.
        self._settings_drifted = False
        self._recompute_settings_drift()

    # -- construction ---------------------------------------------------

    def _apply_pyqtgraph_theme(self) -> None:

        '''Global pyqtgraph color config -- must run before any PlotWidget exists.'''

        pg.setConfigOption("background", BACKGROUND_COLOR)
        pg.setConfigOption("foreground", FOREGROUND_COLOR)
        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

    def _build_ui(self) -> None:

        self.setStyleSheet(
            f"background-color: {BACKGROUND_COLOR}; color: {FOREGROUND_COLOR};"
        )

        root_layout = QHBoxLayout(self)

        left_column = QVBoxLayout()
        left_column.addWidget(self._build_run_configuration_group())
        left_column.addWidget(self._build_main_plot(), stretch=3)
        left_column.addWidget(self._build_residual_plot(), stretch=1)
        root_layout.addLayout(left_column, stretch=1)

        root_layout.addWidget(self._build_side_panel())

    def _build_run_configuration_group(self) -> QGroupBox:

        group = QGroupBox("Run Configuration")
        group.setFont(load_bundled_font(10))
        group.setStyleSheet(group_box_stylesheet())
        group.setToolTip(
            "Acquires Number of Shots frames from the camera stream, runs "
            "each through preprocessing/analysis at every fit degree, and "
            "combines the results via combine_shots(). If Acquisition "
            "Settings no longer match the camera's current exposure/gain, "
            "this first stops, reconfigures, and restarts the stream -- "
            "freezing live view for the duration."
        )
        layout = QHBoxLayout(group)

        form = QFormLayout()
        self._n_shots_spin = QSpinBox()
        self._n_shots_spin.setFont(load_bundled_font(10))
        self._n_shots_spin.setRange(MIN_N_SHOTS, MAX_N_SHOTS)
        self._n_shots_spin.setValue(DEFAULT_N_SHOTS)
        form.addRow("Number of Shots:", self._n_shots_spin)
        layout.addLayout(form)

        self._run_button = QPushButton("Run Measurement")
        self._run_button.setFont(load_bundled_font(10, bold=True))
        self._run_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: {BACKGROUND_COLOR}; "
            f"border-radius: 4px; padding: 8px; }}"
        )
        self._run_button.clicked.connect(self._on_run_clicked)
        layout.addWidget(self._run_button)

        return group

    def _build_main_plot(self) -> pg.PlotWidget:

        plot_widget = pg.PlotWidget()
        self._main_plot = plot_widget.getPlotItem()
        self._main_plot.setLabel("left", "Relative Physical Position (mm)")
        self._main_plot.setLabel("bottom", self._x_axis_label())
        self._main_plot.showGrid(x=True, y=True, alpha=0.3)
        self._style_plot_axes(self._main_plot)

        self._scatter = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 200)
        )
        self._main_plot.addItem(self._scatter)

        self._error_bars = pg.ErrorBarItem(pen=pg.mkPen(color=FOREGROUND_COLOR, width=1))
        self._main_plot.addItem(self._error_bars)

        self._fit_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=FIT_CURVE_COLOR, width=FIT_CURVE_WIDTH)
        )
        self._main_plot.addItem(self._fit_curve)

        plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return plot_widget

    def _build_residual_plot(self) -> pg.PlotWidget:

        plot_widget = pg.PlotWidget()
        self._residual_plot = plot_widget.getPlotItem()
        self._residual_plot.setLabel("left", "Residual (mm)")
        self._residual_plot.setLabel("bottom", self._x_axis_label())
        self._residual_plot.showGrid(x=True, y=True, alpha=0.3)
        self._style_plot_axes(self._residual_plot)
        self._residual_plot.setXLink(self._main_plot)
        # Residual values are already a small, readable mm figure --
        # pyqtgraph's default autoSIPrefix scale annotation ("Residual
        # (mm) x0.001"-style) clips against this plot's top edge (same
        # issue live_view.py's strip chart solved with a dedicated
        # _PowerOfTenAxisItem); simplest fix here is to just not scale.
        self._residual_plot.getAxis("left").enableAutoSIPrefix(False)

        self._residual_scatter = pg.ScatterPlotItem(
            size=6, pen=pg.mkPen(None), brush=pg.mkBrush(RESIDUAL_SCATTER_COLOR)
        )
        self._residual_plot.addItem(self._residual_scatter)

        self._residual_zero_line = pg.InfiniteLine(
            pos=0.0, angle=0, pen=pg.mkPen(color=GRID_COLOR, width=1, style=Qt.DashLine)
        )
        self._residual_plot.addItem(self._residual_zero_line)

        plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return plot_widget

    def _style_plot_axes(self, plot_item: pg.PlotItem) -> None:

        font = load_bundled_font(9)
        for axis_name in ("left", "bottom"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen(color=GRID_COLOR))
            axis.setTextPen(pg.mkPen(color=FOREGROUND_COLOR))
            axis.setTickFont(font)
            axis.label.setFont(font)

    def _build_side_panel(self) -> QWidget:

        panel = QWidget()
        panel.setFixedWidth(SIDE_PANEL_WIDTH)
        layout = QVBoxLayout(panel)

        layout.addWidget(self._build_acquisition_settings_group())
        layout.addWidget(self._build_spatial_roi_group())
        layout.addWidget(self._build_degree_selector_group())
        layout.addWidget(self._build_combined_result_group())
        layout.addStretch(1)

        self._back_button = QPushButton("Back to Live View")
        self._back_button.setFont(load_bundled_font(12, bold=True))
        self._back_button.setMinimumHeight(48)
        self._back_button.clicked.connect(self.back_requested)
        layout.addWidget(self._back_button)

        return panel

    def _build_acquisition_settings_group(self) -> QGroupBox:

        '''
        Exposure/gain display+entry, pre-filled from
        calibration_set.baseline_record. Editing either field re-evaluates
        the drift check below (against the loaded calibrations) and is
        also what "Run Measurement" compares against camera_stream's
        actual current settings to decide whether a stop/reconfigure/
        restart cycle is needed (see module docstring) -- the two checks
        are independent. Identical role to LiveViewWidget's group of the
        same name.
        '''

        group = QGroupBox("Acquisition Settings")
        group.setFont(load_bundled_font(10))
        group.setStyleSheet(group_box_stylesheet())
        group.setToolTip(
            "Pre-filled from the loaded baseline's capture settings. "
            "Drifting past tolerance from the loaded calibrations hides "
            "the Combined Result panel and scatter/fit/residual overlay "
            "(reading \"N/A\") and shows an informational recalibration "
            "message. Independently, if these values no longer match the "
            "camera's actual current exposure/gain, \"Run Measurement\" "
            "stops/reconfigures/restarts the stream to match before "
            "acquiring."
        )
        form = QFormLayout(group)

        baseline_record = self._calibration_set.baseline_record

        self._exposure_spin = QDoubleSpinBox()
        self._exposure_spin.setFont(load_bundled_font(10))
        self._exposure_spin.setRange(1.0, 1_000_000.0)
        self._exposure_spin.setDecimals(1)
        self._exposure_spin.setSuffix(" us")
        self._exposure_spin.setValue(baseline_record.exposure_us)
        self._exposure_spin.valueChanged.connect(self._on_exposure_changed)
        form.addRow("Exposure:", self._exposure_spin)

        self._gain_spin = QDoubleSpinBox()
        self._gain_spin.setFont(load_bundled_font(10))
        self._gain_spin.setRange(0.0, 48.0)
        self._gain_spin.setSingleStep(0.1)
        self._gain_spin.setSuffix(" dB")
        self._gain_spin.setValue(baseline_record.gain_db)
        self._gain_spin.valueChanged.connect(self._on_gain_changed)
        form.addRow("Gain:", self._gain_spin)

        return group

    def _build_spatial_roi_group(self) -> QGroupBox:

        self._roi_control = SpatialROIControl(
            self._position_calibration, CANONICAL_SHAPE[SPATIAL_AXIS]
        )
        self._roi_control.roi_changed.connect(self._on_roi_changed)
        return self._roi_control

    # -- acquisition settings drift detection ----------------------------

    def _on_exposure_changed(self, exposure_us: float) -> None:

        self._recompute_settings_drift()

    def _on_gain_changed(self, gain_db: float) -> None:

        self._recompute_settings_drift()

    def _recompute_settings_drift(self) -> None:

        '''
        Combined baseline/conversion-gain drift check -- identical logic
        to LiveViewWidget._recompute_settings_drift(), targeting this
        screen's own widgets. See that method's docstring for the full
        rationale (single combined state machine, popup fires once per
        drift episode).
        '''

        baseline_record = self._calibration_set.baseline_record

        exposure_drifted = exposure_has_drifted(
            self._exposure_spin.value(), baseline_record.exposure_us
        )
        baseline_gain_drifted = gain_has_drifted(
            self._gain_spin.value(), baseline_record.gain_db
        )
        conversion_gain_drifted = (
            self._conversion_gain_record is not None
            and gain_has_drifted(
                self._gain_spin.value(), self._conversion_gain_record.gain_db
            )
        )
        drifted = exposure_drifted or baseline_gain_drifted or conversion_gain_drifted

        if drifted and not self._settings_drifted:
            self._settings_drifted = True
            self._enter_drifted_state(
                baseline_drifted=exposure_drifted or baseline_gain_drifted,
                conversion_gain_drifted=conversion_gain_drifted,
            )
        elif not drifted and self._settings_drifted:
            self._settings_drifted = False
            self._exit_drifted_state()

    def _enter_drifted_state(
        self, baseline_drifted: bool, conversion_gain_drifted: bool
    ) -> None:

        '''
        Entered on a False -> True self._settings_drifted transition --
        see LiveViewWidget._enter_drifted_state()'s docstring for the full
        rationale. N/As the Combined Result group's labels and hides the
        scatter/error-bar/fit-curve/residual-scatter overlay (no heatmap
        exists on this screen to leave visible, unlike live_view.py's).

        Parameters
        ----------
        baseline_drifted
            True if exposure or gain drifted from calibration_set.
            baseline_record.
        conversion_gain_drifted
            True if gain drifted from self._conversion_gain_record (always
            False when no conversion-gain record was supplied).
        '''

        self._n_shots_label.setText("N/A")
        self._spatial_dispersion_label.setText("N/A")
        self._reduced_chi_squared_label.setText("N/A")
        self._evaluated_at_label.setText("")
        self._degree_note_label.setText("")

        self._scatter.setVisible(False)
        self._error_bars.setVisible(False)
        self._fit_curve.setVisible(False)
        self._residual_scatter.setVisible(False)

        QMessageBox.warning(
            self,
            "Recalibration Required",
            "Please recalibrate baseline and conversion gain with new settings.",
        )

        if baseline_drifted:
            self.recalibration_requested.emit("baseline")
        if conversion_gain_drifted:
            self.recalibration_requested.emit("conversion_gain")

    def _exit_drifted_state(self) -> None:

        '''
        Entered on a True -> False self._settings_drifted transition:
        restores the scatter/error-bar/fit-curve/residual-scatter
        overlay's visibility and recomputes the Combined Result panel for
        the currently-selected degree, undoing _enter_drifted_state()'s
        "N/A" placeholders.
        '''

        self._scatter.setVisible(True)
        self._error_bars.setVisible(True)
        self._fit_curve.setVisible(True)
        self._residual_scatter.setVisible(True)
        self._refresh_measurement_display()

    def _build_degree_selector_group(self) -> QGroupBox:

        group = QGroupBox("Fit Degree")
        group.setFont(load_bundled_font(10))
        group.setStyleSheet(group_box_stylesheet())
        layout = QVBoxLayout(group)

        self._degree_selector = QComboBox()
        self._degree_selector.setFont(load_bundled_font(10))
        self._degree_selector.setStyleSheet(combo_box_stylesheet())
        for degree in DEGREE_CHOICES:
            self._degree_selector.addItem(DEGREE_LABELS[degree], userData=degree)
        self._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(DEFAULT_DEGREE))
        self._degree_selector.currentIndexChanged.connect(self._on_degree_changed)
        layout.addWidget(self._degree_selector)

        self._formula_label = QLabel(fit_formula_html(DEFAULT_DEGREE, self._wavelength_axis))
        self._formula_label.setTextFormat(Qt.RichText)
        self._formula_label.setWordWrap(True)
        self._formula_label.setFont(load_bundled_font(10))
        layout.addWidget(self._formula_label)

        return group

    def _build_combined_result_group(self) -> QGroupBox:

        '''
        This screen's analogue of LiveViewWidget's "Fit Diagnostics"
        group. "Evaluate At"/"Evaluated At" are only shown for degree > 1
        (toggled in _refresh_measurement_display()) -- degree 1's spatial
        dispersion is a single number independent of wavelength, so a
        reference point doesn't apply to it. sigma_internal/sigma_external
        are deliberately not surfaced as their own rows -- same rationale
        as combine_shots()'s own docstring: sigma_zeta_combined already
        reports whichever of the two is larger, which is the one number a
        user needs.
        '''

        group = QGroupBox("Combined Result")
        group.setFont(load_bundled_font(10))
        group.setStyleSheet(group_box_stylesheet())
        form = QFormLayout(group)
        self._combined_result_form = form

        label_font = load_bundled_font(10)

        self._n_shots_label = QLabel("--")
        self._spatial_dispersion_label = QLabel("--")
        self._spatial_dispersion_label.setWordWrap(True)
        self._reduced_chi_squared_label = QLabel("--")

        self._evaluated_at_spin = QDoubleSpinBox()
        self._evaluated_at_spin.setRange(0.0, float(CANONICAL_SHAPE[SPECTRAL_AXIS] - 1))
        self._evaluated_at_spin.setDecimals(1)
        self._evaluated_at_spin.setValue(EVALUATED_AT_COLUMN)
        self._evaluated_at_spin.setSuffix(" px")
        self._evaluated_at_spin.setToolTip(
            "Pixel column each shot's spatial dispersion is evaluated at "
            "before combining across shots (degree > 1 only)."
        )
        self._evaluated_at_spin.valueChanged.connect(self._on_evaluated_at_changed)

        self._evaluated_at_label = QLabel("")
        self._evaluated_at_label.setWordWrap(True)

        self._degree_note_label = QLabel("")
        self._degree_note_label.setWordWrap(True)
        self._degree_note_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-style: italic;")

        for widget in (
            self._n_shots_label, self._spatial_dispersion_label,
            self._reduced_chi_squared_label, self._evaluated_at_spin,
            self._evaluated_at_label, self._degree_note_label,
        ):
            widget.setFont(label_font)

        form.addRow("N Shots:", self._n_shots_label)
        form.addRow("Spatial Dispersion (mm/nm):", self._spatial_dispersion_label)
        form.addRow("Reduced Chi-Squared:", self._reduced_chi_squared_label)
        form.addRow("Evaluate At:", self._evaluated_at_spin)
        form.addRow("Evaluated At:", self._evaluated_at_label)
        form.addRow("", self._degree_note_label)

        return group

    # -- run configuration -------------------------------------------------

    def _on_run_clicked(self) -> None:

        '''
        "Run Measurement" handler: reconfigures camera_stream if needed
        (see _maybe_reconfigure_camera_stream()), blocks on
        collect_n_frames(n_shots), runs every frame through
        run_preprocessing() + analyze_shot() (all DEGREE_CHOICES at once),
        then redraws the combined-result panel and scatter/fit/residual
        overlay for the currently-selected degree.

        n_shots is read from self._n_shots_spin. roi_bounds is read fresh
        from self._roi_control for each frame (matching
        LiveViewWidget-style per-call reads elsewhere in this screen),
        even though it cannot actually change mid-acquisition -- this
        call blocks the Qt event loop for its entire duration, so no user
        interaction with the ROI control can happen until it returns.

        Unlike LiveViewWidget's continuous polling loop -- where a single
        bad frame is just skipped and the next tick tries again --
        NoSignalError/SettingsMismatchError/InsufficientDataError from any
        one shot in the batch aborts the whole run: there's no "next tick"
        here to self-correct, and silently dropping just that one shot
        would understate n_shots in the combined result without saying
        so. Reported via a QMessageBox naming which shot and why (matching
        _enter_drifted_state()'s existing message-box convention), and the
        display is left exactly as it was before this click -- no partial
        _set_measurement_data()/_refresh_measurement_display() call with
        an incomplete shot_results list.

        CameraError (from _maybe_reconfigure_camera_stream()'s restart, or
        from collect_n_frames() if the camera drops mid-acquisition) and
        the RuntimeError collect_n_frames() itself documents (stream
        stopped by something else while waiting) get the same
        abort-cleanly treatment, via show_camera_error_dialog() -- the
        same routing calibration_dialogs.py already uses for every other
        real camera-touching call in this codebase.
        '''

        try:
            self._maybe_reconfigure_camera_stream()

            n_shots = self._n_shots_spin.value()
            frames = self._camera_stream.collect_n_frames(n_shots)
        except (CameraError, RuntimeError) as error:
            show_camera_error_dialog(self, str(error))
            return

        shot_results: list[ShotAnalysisResult] = []
        for frame in frames:
            try:
                processed, _saturation_result = run_preprocessing(
                    frame, self._calibration_set, roi_bounds=self._roi_control.roi_bounds_px()
                )
                shot_results.append(
                    analyze_shot(
                        processed, self._axis_for_fit, noise_model=self._noise_model,
                        degrees=DEGREE_CHOICES,
                    )
                )
            except (NoSignalError, SettingsMismatchError, InsufficientDataError) as error:
                QMessageBox.warning(
                    self,
                    "Measurement Failed",
                    f"Shot {frame.frame_id} could not be analyzed ({error}). "
                    f"Run aborted -- no results were updated.",
                )
                return

        self._set_measurement_data(shot_results)
        self._refresh_measurement_display()

    def _maybe_reconfigure_camera_stream(self) -> None:

        '''
        If the Acquisition Settings panel's exposure/gain no longer match
        what camera_stream is actually running with, stops the stream,
        applies the new values, and restarts it -- the same stop/
        reconfigure/restart cycle calibration/sensor/conversion_gain.py's
        run_conversion_gain_calibration() uses to sweep exposure, since
        CameraStream has no way to change either setting while running.
        A no-op (no interruption at all) when both values already match,
        e.g. every run after the first with an unedited panel.
        '''

        target_exposure_us = self._exposure_spin.value()
        target_gain_db = self._gain_spin.value()

        exposure_matches = math.isclose(
            target_exposure_us, self._camera_stream.exposure_us, rel_tol=1e-9, abs_tol=1e-9
        )
        gain_matches = math.isclose(
            target_gain_db, self._camera_stream.gain_db, rel_tol=1e-9, abs_tol=1e-9
        )
        if exposure_matches and gain_matches:
            return

        self._camera_stream.stop()
        self._camera_stream.exposure_us = target_exposure_us
        self._camera_stream.gain_db = target_gain_db
        self._camera_stream.start()

    # -- degree selector ----------------------------------------------------

    def _on_degree_changed(self, index: int) -> None:

        degree = self._degree_selector.itemData(index)
        if degree is None:
            return
        self._current_degree = degree
        self._refresh_measurement_display()

    def _on_evaluated_at_changed(self, column: float) -> None:

        '''
        Only affects degree > 1 (see _compute_combined_result()) -- for
        degree 1 this still updates the "Evaluated At" readout text (kept
        in sync for whenever the user switches to a degree > 1) but has no
        effect on the reported number. Re-evaluates each cached shot's
        zeta(wavelength_ref)/sigma_zeta(wavelength_ref) at the new
        reference point and recombines -- no new acquisition needed.
        '''

        self._refresh_measurement_display()

    # -- measurement data + display ---------------------------------------

    def _set_measurement_data(self, shot_results: list[ShotAnalysisResult]) -> None:

        '''
        Stashes one Run Measurement's per-shot analysis results and the
        degree-independent flattened (shot, column) arrays derived from
        them -- every raw centroid point across every shot, concatenated,
        never averaged per column (see module docstring for why). Degree-
        dependent quantities (the combined zeta, fit-curve intercept,
        residuals) are computed on demand by _refresh_measurement_display()
        instead of cached here, so switching the degree selector or
        editing the evaluate-at reference point always reflects the
        current selection.
        '''

        self._shot_results = shot_results

        columns = np.concatenate([result.centroids.columns for result in shot_results])
        x0_px = np.concatenate([result.centroids.x0 for result in shot_results])
        sigma_x0_px = np.concatenate([result.centroids.sigma_x0 for result in shot_results])

        self._measurement_columns = columns
        self._measurement_wavelength_nm = self._axis_for_fit.wavelength_nm(columns)
        self._measurement_x0_px = x0_px
        self._measurement_sigma_x0_px = sigma_x0_px

        y_values_mm, y_sigma_mm = self._convert_to_mm(x0_px, sigma_x0_px)
        self._measurement_y_values_mm = y_values_mm
        self._measurement_y_sigma_mm = y_sigma_mm
        self._measurement_x_values = self._measurement_wavelength_nm
        self._measurement_x_sigma = (
            self._wavelength_axis.sigma_wavelength_nm(columns)
            if self._wavelength_axis is not None else None
        )
        self._measurement_x_extent = (
            float(self._measurement_x_values.min()), float(self._measurement_x_values.max())
        )

    def _compute_combined_result(self, degree: int) -> CombinedSpatialDispersionResult:

        '''
        Degree 1: combine_shots() directly on each shot's fitted linear
        zeta (coefficients[1]) and its uncertainty (coefficient_sigma[1])
        -- exactly what combine_shots() was built for (see its own
        docstring's parameter description).

        Degree > 1: combine_shots() only combines the linear fit by design
        (see module docstring) -- there is no per-shot "coefficients[1]"
        that means the same thing at degree > 1. Instead, each shot's own
        zeta(wavelength_ref)/sigma_zeta(wavelength_ref) (a well-defined
        scalar at any degree, using that shot's full coefficient_
        covariance) is evaluated at the reference point, and those
        scalars are combined via the exact same combine_shots() call --
        it is generic over any (value, sigma) pairs, not specific to raw
        fit coefficients.
        '''

        if degree == 1:
            zeta_values = np.array(
                [result.fits[1].coefficients[1] for result in self._shot_results]
            )
            sigma_zeta_values = np.array(
                [result.fits[1].coefficient_sigma[1] for result in self._shot_results]
            )
        else:
            wavelength_ref_nm = np.array([self._evaluated_at_wavelength_nm()])
            zeta_values = np.array([
                float(result.fits[degree].zeta(wavelength_ref_nm)[0])
                for result in self._shot_results
            ])
            sigma_zeta_values = np.array([
                float(result.fits[degree].sigma_zeta(wavelength_ref_nm)[0])
                for result in self._shot_results
            ])

        return combine_shots(zeta_values, sigma_zeta_values)

    def _evaluated_at_wavelength_nm(self) -> float:

        '''self._evaluated_at_spin's pixel-column value, converted via self._axis_for_fit.'''

        column = np.array([self._evaluated_at_spin.value()])
        return float(self._axis_for_fit.wavelength_nm(column)[0])

    def _refresh_measurement_display(self) -> None:

        '''
        Single entry point that keeps the degree selector, the Combined
        Result panel, and the scatter/fit-curve/residual overlay
        consistent with each other for self._current_degree -- called
        after a fresh Run Measurement, a degree-selector change, or an
        evaluate-at edit. Safe to call before any measurement has been
        run (self._shot_results is None): the formula/row-visibility/
        degree-note parts still update (they don't depend on real data),
        while the numeric labels and plots stay at their "--"/empty
        initial state.
        '''

        degree = self._current_degree

        self._formula_label.setText(fit_formula_html(degree, self._wavelength_axis))

        has_reference_point = degree > 1
        self._combined_result_form.setRowVisible(self._evaluated_at_spin, has_reference_point)
        self._combined_result_form.setRowVisible(self._evaluated_at_label, has_reference_point)
        self._evaluated_at_label.setText(
            evaluated_at_text(self._evaluated_at_spin.value(), self._wavelength_axis)
            if has_reference_point else ""
        )
        self._degree_note_label.setText(DEGREE_GT_ONE_NOTE if has_reference_point else "")

        if self._shot_results is None:
            self._n_shots_label.setText("--")
            self._spatial_dispersion_label.setText("--")
            self._reduced_chi_squared_label.setText("--")
            return

        combined = self._compute_combined_result(degree)

        self._n_shots_label.setText(str(combined.n_shots))
        zeta_mm, zeta_sigma_mm = self._zeta_to_mm(combined.zeta_combined, combined.sigma_zeta_combined)
        self._spatial_dispersion_label.setText(format_value_with_uncertainty(zeta_mm, zeta_sigma_mm))
        # combine_shots() doesn't expose reduced chi-squared as its own
        # field (see combination.py) -- but it's recoverable exactly from
        # the two sigmas it does return, since weighted_scatter/(n_shots-1)
        # (the standard reduced-chi-squared definition) equals
        # (sigma_external/sigma_internal)**2 by construction:
        # weighted_scatter = sigma_external**2 * (n_shots-1) * sum(weights),
        # and sum(weights) = 1/sigma_internal**2.
        reduced_chi_squared = (combined.sigma_external / combined.sigma_internal) ** 2
        self._reduced_chi_squared_label.setText(f"{reduced_chi_squared:.3f}")

        self._recompute_fit_and_residuals(combined.zeta_combined)
        self._apply_roi_bounds(*self._roi_control.roi_bounds_mm())

    def _recompute_fit_and_residuals(self, zeta_combined: float) -> None:

        '''
        Builds the drawn fit-curve/residual arrays for the currently-
        combined zeta: a single straight line of slope zeta_combined (in
        pixel/nm units, matching analyze_shot()'s pixel-unit convention),
        anchored through the flattened (shot, column) data by a plain
        weighted-least-squares intercept -- the best-fit offset given that
        fixed, already-combined slope (see module docstring for why this
        is the only combination-consistent choice at any degree).
        '''

        weights = 1.0 / self._measurement_sigma_x0_px ** 2
        intercept_px = float(
            np.sum(
                weights * (self._measurement_x0_px - zeta_combined * self._measurement_wavelength_nm)
            ) / np.sum(weights)
        )

        fit_x = np.linspace(
            self._measurement_x_values.min(), self._measurement_x_values.max(), FIT_CURVE_N_POINTS
        )
        fit_y_px = intercept_px + zeta_combined * fit_x
        fit_y_mm, _ = self._convert_to_mm(fit_y_px, np.zeros_like(fit_y_px))

        residual_px = self._measurement_x0_px - (
            intercept_px + zeta_combined * self._measurement_wavelength_nm
        )
        residual_mm, _ = self._convert_to_mm(residual_px, np.zeros_like(residual_px))

        self._measurement_fit_x = fit_x
        self._measurement_fit_y_mm = fit_y_mm
        self._measurement_residuals_mm = residual_mm

    def _apply_roi_bounds(self, min_mm: float, max_mm: float) -> None:

        '''
        Crops the current measurement's scatter/error-bars/fit-curve/
        residuals to [min_mm, max_mm] and renders them. A view-level crop
        only -- it does not re-run preprocessing/analysis with a new ROI
        mask (that already happened once, at "Run Measurement" time, using
        whatever ROI was set then); changing the ROI control afterward
        just changes which of the already-computed points are shown and
        what range the plots are zoomed to. A no-op if no measurement has
        been run yet.
        '''

        if self._shot_results is None:
            return

        y_values = self._measurement_y_values_mm
        y_sigma = self._measurement_y_sigma_mm
        keep = (y_values >= min_mm) & (y_values <= max_mm)

        x_values = self._measurement_x_values
        self._scatter.setData(x=x_values[keep], y=y_values[keep])
        error_kwargs = dict(
            x=x_values[keep], y=y_values[keep], top=y_sigma[keep], bottom=y_sigma[keep]
        )
        if self._measurement_x_sigma is not None:
            error_kwargs["left"] = self._measurement_x_sigma[keep]
            error_kwargs["right"] = self._measurement_x_sigma[keep]
        self._error_bars.setData(**error_kwargs)

        fit_y_full = self._measurement_fit_y_mm
        keep_fit = (fit_y_full >= min_mm) & (fit_y_full <= max_mm)
        self._fit_curve.setData(
            x=self._measurement_fit_x[keep_fit], y=fit_y_full[keep_fit]
        )

        self._residual_scatter.setData(
            x=x_values[keep], y=self._measurement_residuals_mm[keep]
        )

        self._main_plot.setRange(
            xRange=self._measurement_x_extent, yRange=(min_mm, max_mm), padding=0
        )
        self._residual_plot.setRange(xRange=self._measurement_x_extent, padding=0)

    def _on_roi_changed(self, min_mm: float, max_mm: float) -> None:

        self._apply_roi_bounds(min_mm, max_mm)

    # -- pure-ish helpers (presentational only) ----------------------------

    def _x_axis_label(self) -> str:
        return wavelength_axis_label(self._wavelength_axis)

    def _convert_to_mm(
        self, x0: np.ndarray, sigma_x0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        '''
        position_calibration.convert() returns microns (see
        calibration/spatial/calibrate.py's module docstring) -- every
        on-screen y-axis call site goes through this instead of calling
        convert() directly, so the widget's "(mm)"-labeled axes always
        actually read in mm. Identical to LiveViewWidget's helper of the
        same name.
        '''

        y0, sigma_y0 = self._position_calibration.convert(x0, sigma_x0)
        return microns_to_mm(y0), microns_to_mm(sigma_y0)

    def _zeta_to_mm(self, zeta_value: float, zeta_sigma: float) -> tuple[float, float]:

        '''
        Converts a combined zeta (px/nm, combine_shots()'s native unit)
        to physical units (mm/nm) via self._convert_to_mm() -- valid for
        a slope, not just a position, because
        ScaleFactorPositionCalibration.convert() is a pure linear scale
        with no additive offset (see its own docstring). Display-only:
        callers must keep using the raw px/nm zeta_combined for
        _recompute_fit_and_residuals(), which needs it in
        analyze_shot()'s native units to match self._measurement_x0_px.
        Identical to LiveViewWidget's helper of the same name, minus the
        None-sigma case (combine_shots() always returns a real sigma).
        '''

        zeta_mm, sigma_mm = self._convert_to_mm(np.array([zeta_value]), np.array([zeta_sigma]))
        return float(zeta_mm[0]), float(sigma_mm[0])


# Functions


__all__ = [
    "ExtendedMeasurementScreen",
    "DEFAULT_N_SHOTS",
    "MIN_N_SHOTS",
    "MAX_N_SHOTS",
]
