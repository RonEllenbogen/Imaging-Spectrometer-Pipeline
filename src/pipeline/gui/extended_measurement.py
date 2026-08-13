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

The main-plot fit curve/residual subplot are never a refit of per-column-
averaged centroid positions -- this codebase already considered and
rejected that combination methodology when analysis/combination.py was
designed (see its module docstring): refitting column averages could
visually disagree with the reported combined result sitting right next to
it. At degree 1, every raw (shot, column) centroid point is plotted/
residualed against one straight line of slope zeta_combined, anchored
through the data by a plain weighted-least-squares intercept (the
best-fit offset given that fixed, already-combined slope) -- so the drawn
line's slope and the reported number can never visually contradict each
other. At degree > 1, the curve is a genuine combined polynomial instead:
compute_combined_polynomial_for_degree() combines every coefficient
(c0..c_degree) independently across shots, the same inverse-variance
weighting extended to every coefficient index rather than only the
combined zeta -- so the drawn curve/residuals depend only on
self._shot_results, not on self._evaluated_at_spin's current value (that
reference point only ever changes self._spatial_dispersion_label, the
scalar dz/dwavelength evaluated there -- an earlier version of this
screen instead redrew a fresh local tangent line through that point at
every degree, which visibly moved the whole curve on every reference-
point edit even though nothing about the underlying combined fit had
changed).

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
from pipeline.cli.calibration import DEFAULT_ARTIFACT_DIR
from pipeline.preprocessing import (
    CalibrationSet, NoSignalError, ProcessedFrame, SettingsMismatchError, run_preprocessing,
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
from pipeline.gui.roi_control import SpatialROIControl, SpectralROIControl
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
# explains the *spatial dispersion* number's combination methodology
# (per-shot scalar evaluation at the reference point, then combined) --
# not an uncertainty caveat, and not a statement about the drawn curve
# (which IS a genuine combined polynomial -- see
# compute_combined_polynomial_for_degree()). The previous "external
# uncertainty only" interim caveat (docs/project_state.md) no longer
# applies now that sigma_zeta_combined is built from a real sigma_zeta()
# call (the full covariance-propagated uncertainty, combined internal/
# external the same way as degree 1), so there is nothing left to
# disclaim about the number itself.
DEGREE_GT_ONE_NOTE = (
    "Degree > 1: spatial dispersion above is each shot's dz/dwavelength "
    "evaluated at the reference point, then combined -- the plotted "
    "curve is a separately combined polynomial, unaffected by the "
    "reference point."
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
        # _refresh_measurement_display()). stacked_image/
        # representative_frames/roi_bounds_px/column_bounds are set
        # alongside shot_results by _set_measurement_data() -- see that
        # method's docstring.
        self._shot_results: list[ShotAnalysisResult] | None = None
        self._measurement_stacked_image: np.ndarray | None = None
        self._measurement_representative_frames: dict[str, ProcessedFrame] | None = None
        self._measurement_roi_bounds_px: tuple[int, int] | None = None
        self._measurement_column_bounds: tuple[int, int] | None = None

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

        self._save_record_button = QPushButton("Save Record")
        self._save_record_button.setFont(load_bundled_font(10, bold=True))
        self._save_record_button.setToolTip(
            "Writes a complete, self-contained record of the most recent Run Measurement "
            "result (frames, centroids, fits at every degree, calibrations used, ROI, and "
            "a journal-style plot) to data/measurements/ -- a deliberately separate, "
            "explicit action from Run Measurement, so trial runs aren't all permanently saved."
        )
        # Disabled until a measurement has actually completed -- see
        # _on_run_clicked()'s end and __init__'s self._shot_results=None.
        self._save_record_button.setEnabled(False)
        self._save_record_button.clicked.connect(self._on_save_record_clicked)
        layout.addWidget(self._save_record_button)

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
        layout.addWidget(self._build_spectral_roi_group())
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

    def _build_spectral_roi_group(self) -> QGroupBox:

        '''
        Unlike self._roi_control (spatial), this control's roi_changed is
        NOT connected to any live re-render: a spectral-ROI change alters
        which columns are even analyzed (valid_columns), so -- unlike the
        spatial ROI, a pure post-hoc pixel-zeroing that never drops a
        centroid point -- there is no way to retroactively apply a change
        to an already-completed Run Measurement's self._shot_results
        without either fabricating data for columns that were never
        analyzed (widening) or misrepresenting how many columns actually
        went into the combined result (narrowing). Its current
        column_bounds() is instead read fresh in _on_run_clicked(), same
        timing as self._roi_control.roi_bounds_px() -- "changing the ROI
        control afterward" for this one specifically means "next Run
        Measurement", not "immediately", and there is deliberately no
        signal connection here to suggest otherwise.
        '''

        self._spectral_roi_control = SpectralROIControl(
            CANONICAL_SHAPE[SPECTRAL_AXIS], self._wavelength_axis
        )
        return self._spectral_roi_control

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

        Coefficients: unlike LiveViewWidget's own "Coefficients:" row (a
        single shot's full degree+1 fit.coefficients), this row shows the
        *combined* polynomial -- degree + 1 coefficients, one per line.
        At degree 1, c0/c1 are the weighted-least-squares intercept
        _recompute_fit_and_residuals() computes plus zeta_combined itself
        (the same number self._spatial_dispersion_label shows -- c1 IS
        zeta_combined only at degree 1, see _DegreeResult's docstring in
        measurement_record.py for why that stops being true above degree
        1). At degree > 1, all degree + 1 coefficients come from
        compute_combined_polynomial_for_degree() -- a genuine combined
        quadratic/cubic, not the same quantity as
        self._spatial_dispersion_label (see DEGREE_GT_ONE_NOTE). Both
        paths are recomputed by _refresh_measurement_display() on every
        degree switch, same as spatial dispersion.
        '''

        group = QGroupBox("Combined Result")
        group.setFont(load_bundled_font(10))
        group.setStyleSheet(group_box_stylesheet())
        form = QFormLayout(group)
        self._combined_result_form = form

        label_font = load_bundled_font(10)

        self._n_shots_label = QLabel("--")
        self._coefficients_label = QLabel("--")
        self._coefficients_label.setTextFormat(Qt.RichText)
        self._coefficients_label.setWordWrap(True)
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
            self._n_shots_label, self._coefficients_label, self._spatial_dispersion_label,
            self._reduced_chi_squared_label, self._evaluated_at_spin,
            self._evaluated_at_label, self._degree_note_label,
        ):
            widget.setFont(label_font)

        form.addRow("N Shots:", self._n_shots_label)
        form.addRow("Coefficients:", self._coefficients_label)
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

        n_shots is read from self._n_shots_spin. roi_bounds/column_bounds
        are read once from self._roi_control/self._spectral_roi_control
        before the acquisition loop and reused for every frame -- this
        call blocks the Qt event loop for its entire duration, so no user
        interaction with either ROI control can happen until it returns,
        making a single read equivalent to LiveViewWidget-style per-call
        reads elsewhere in this screen. Both are also stashed as
        self._measurement_roi_bounds_px/_measurement_column_bounds (see
        _set_measurement_data()) -- "Save Record" reads those captured
        values, not the live controls, since it runs later as a separate
        click and either control could have been edited in between. See
        _build_spectral_roi_group()'s docstring for why, unlike the
        spatial ROI, this is also the ONLY point at which
        self._spectral_roi_control's current value takes effect -- there
        is no live re-render on a later edit.

        Every preprocessed frame's image is folded into a running sum
        (-> a mean at the end) and checked against
        _representative_shot_indices() as it's produced, rather than
        collected into a list and reduced afterward -- deliberately, so
        this method never holds more than a small, constant number of
        full-resolution frames in memory at once (the running sum plus up
        to 3 representative frames), regardless of n_shots. n_shots is
        capped at MAX_N_SHOTS (1000) and each frame is ~18MB float64 at
        CANONICAL_SHAPE -- retaining all of them (an earlier version of
        this method did, via a plain list, for "Save Record" to later
        pick from) meant a 200-shot run alone held ~3.6GB resident for the
        rest of the widget's life, with a further multi-GB spike from
        averaging that list in one call -- the real cause of a reported
        multi-minute UI freeze on Save Record, and very likely of a
        second, unrelated-looking freeze switching the degree selector
        afterward too (an otherwise-cheap Qt operation stalling under the
        lingering memory/GC pressure, not a bug in that code path itself).

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

        roi_bounds_px = self._roi_control.roi_bounds_px()
        column_bounds = self._spectral_roi_control.column_bounds()
        representative_indices = _representative_shot_indices(len(frames))

        shot_results: list[ShotAnalysisResult] = []
        representative_frames: dict[str, ProcessedFrame] = {}
        frame_sum: np.ndarray | None = None
        for i, frame in enumerate(frames):
            try:
                processed, _saturation_result = run_preprocessing(
                    frame, self._calibration_set,
                    roi_bounds=roi_bounds_px, column_bounds=column_bounds,
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

            frame_sum = processed.image if frame_sum is None else frame_sum + processed.image
            for label, index in representative_indices.items():
                if i == index:
                    representative_frames[label] = processed

        stacked_image = frame_sum / len(frames)

        self._set_measurement_data(
            shot_results, stacked_image, representative_frames, roi_bounds_px, column_bounds,
        )
        self._refresh_measurement_display()
        self._save_record_button.setEnabled(True)

    def _on_save_record_clicked(self) -> None:

        '''
        "Save Record" handler: writes a complete record of the most
        recent Run Measurement result via
        measurement_record.save_measurement_record() -- see that
        module's docstring for exactly what gets saved. Only enabled
        once self._shot_results exists (see _on_run_clicked()'s end), so
        there is always a real measurement to save by the time this can
        be clicked.

        Imports measurement_record locally, not at module scope --
        measurement_record.py imports compute_combined_result_for_degree()/
        compute_combined_polynomial_for_degree()/compute_fit_line_and_residuals()/
        FIT_CURVE_N_POINTS back from this module, so importing it at
        module scope here would be a circular import; see
        measurement_record.py's own module docstring for the full
        reasoning. No CameraError/hardware failure mode exists here
        (nothing about saving touches the camera), but any unexpected
        exception (e.g. a disk-full/permissions error writing to
        data/measurements/) still gets a QMessageBox.warning() rather than
        propagating into the Qt event loop, matching this file's existing
        defensive-catch style for every other button handler.
        '''

        from .measurement_record import save_measurement_record

        try:
            record_dir = save_measurement_record(
                self._shot_results, self._measurement_stacked_image,
                self._measurement_representative_frames, self._calibration_set,
                self._wavelength_axis, self._position_calibration, self._axis_for_fit,
                self._measurement_roi_bounds_px, self._measurement_column_bounds,
                self._exposure_spin.value(), self._gain_spin.value(),
                artifact_dir=DEFAULT_ARTIFACT_DIR,
            )
        except OSError as error:
            QMessageBox.warning(self, "Save Record Failed", f"Could not save the record: {error}")
            return

        QMessageBox.information(self, "Record Saved", f"Saved measurement record to {record_dir}")

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

    def _set_measurement_data(
        self,
        shot_results: list[ShotAnalysisResult],
        stacked_image: np.ndarray,
        representative_frames: dict[str, ProcessedFrame],
        roi_bounds_px: tuple[int, int],
        column_bounds: tuple[int, int] | None,
    ) -> None:

        '''
        Stashes one Run Measurement's per-shot analysis results, the
        already-reduced frame data "Save Record" needs, and the ROI
        values that were actually used to produce them -- plus the
        degree-independent flattened (shot, column) arrays derived from
        shot_results, every raw centroid point across every shot
        concatenated, never averaged per column (see module docstring for
        why). Degree-dependent quantities (the combined zeta, fit-curve
        intercept, residuals) are computed on demand by
        _refresh_measurement_display() instead of cached here, so
        switching the degree selector or editing the evaluate-at reference
        point always reflects the current selection.

        stacked_image/representative_frames/roi_bounds_px/column_bounds
        exist purely so "Save Record"
        (measurement_record.save_measurement_record()) can later write an
        exact record of this specific run. stacked_image/
        representative_frames are already-reduced by _on_run_clicked()
        (a running mean plus up to 3 selected frames, computed as each
        shot is processed) rather than a full list of every frame's
        image -- see that method's docstring for why holding all of them
        here would scale badly with n_shots. roi_bounds_px/column_bounds
        are captured here rather than re-read from self._roi_control/
        self._spectral_roi_control at save time specifically because
        either control could be edited in the gap between "Run
        Measurement" and a later "Save Record" click, which would
        otherwise report the wrong ROI for this measurement (see
        _on_run_clicked()'s docstring).
        '''

        self._shot_results = shot_results
        self._measurement_stacked_image = stacked_image
        self._measurement_representative_frames = representative_frames
        self._measurement_roi_bounds_px = roi_bounds_px
        self._measurement_column_bounds = column_bounds

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

        '''See compute_combined_result_for_degree() -- this just supplies self._shot_results
        and the currently-entered evaluate-at reference point.'''

        return compute_combined_result_for_degree(
            self._shot_results, degree, self._evaluated_at_wavelength_nm(),
        )

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
            self._coefficients_label.setText("--")
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

        if degree == 1:
            self._recompute_fit_and_residuals(combined.zeta_combined)
            # c0/c1 as a pair, matching measurement_record.py's own
            # degree-1 branch exactly (see compute_fit_line_and_residuals()'s
            # docstring for why intercept+zeta_combined together define
            # the drawn line).
            self._measurement_coefficients_px = np.array(
                [self._measurement_intercept_px, combined.zeta_combined]
            )
            self._measurement_coefficient_sigma_px = np.array(
                [self._measurement_intercept_sigma_px, combined.sigma_zeta_combined]
            )
        else:
            self._recompute_combined_polynomial_fit_and_residuals(degree)

        # Same conversion regardless of coefficient order k: convert()
        # is a pure linear px->mm scale (see _zeta_to_mm()'s docstring),
        # so it's valid applied elementwise to c0 (mm) through
        # c_degree (mm/nm^degree) alike -- only the px numerator's units
        # change, never wavelength_nm, which convert() never touches.
        coefficients_mm, coefficient_sigma_mm = self._convert_to_mm(
            self._measurement_coefficients_px, self._measurement_coefficient_sigma_px,
        )
        coeff_lines = [
            f"c<sub>{k}</sub> = {format_value_with_uncertainty(float(c), float(sigma_c))}"
            for k, (c, sigma_c) in enumerate(zip(coefficients_mm, coefficient_sigma_mm))
        ]
        self._coefficients_label.setText("<br>".join(coeff_lines))

        self._apply_roi_bounds(*self._roi_control.roi_bounds_mm())

    def _recompute_fit_and_residuals(self, zeta_combined: float) -> None:

        '''
        Builds the drawn fit-curve/residual arrays for the currently-
        combined zeta, via compute_fit_line_and_residuals() (pixel units;
        see that function's docstring for the weighted-least-squares
        intercept it's anchored by), then converts the curve/residuals to
        mm for display -- the one step that function deliberately leaves
        to its caller, since it has no reason to depend on
        self._position_calibration.

        Also stashes the fit's intercept (and its uncertainty) as
        self._measurement_intercept_px/_intercept_sigma_px, for
        _refresh_measurement_display()'s Coefficients row -- c0 in the
        (c0, c1=zeta_combined) pair a straight line is defined by.
        '''

        intercept_px, intercept_sigma_px, fit_x, fit_y_px, residual_px = compute_fit_line_and_residuals(
            self._measurement_x0_px, self._measurement_sigma_x0_px, self._measurement_wavelength_nm,
            zeta_combined,
            (float(self._measurement_x_values.min()), float(self._measurement_x_values.max())),
            FIT_CURVE_N_POINTS,
        )

        self._measurement_intercept_px = intercept_px
        self._measurement_intercept_sigma_px = intercept_sigma_px

        fit_y_mm, _ = self._convert_to_mm(fit_y_px, np.zeros_like(fit_y_px))
        residual_mm, _ = self._convert_to_mm(residual_px, np.zeros_like(residual_px))

        self._measurement_fit_x = fit_x
        self._measurement_fit_y_mm = fit_y_mm
        self._measurement_residuals_mm = residual_mm

    def _recompute_combined_polynomial_fit_and_residuals(self, degree: int) -> None:

        '''
        degree > 1's counterpart to _recompute_fit_and_residuals(): builds
        the drawn fit-curve/residual arrays from a genuine combined
        polynomial (compute_combined_polynomial_for_degree()) instead of a
        single-reference-point tangent line -- every coefficient combined
        independently across shots, exactly matching
        save_measurement_record()'s own degree > 1 branch via the same
        shared free function, so the live GUI and the saved record always
        draw the identical curve for a given self._shot_results.

        Deliberately takes no reference-point/zeta argument, unlike
        _recompute_fit_and_residuals(): the combined polynomial doesn't
        depend on self._evaluated_at_spin at all, so editing it changes
        self._spatial_dispersion_label (computed separately, by
        _compute_combined_result()) but never this curve -- fixing the
        bug where an earlier version of this screen redrew a fresh local
        tangent line through the reference point on every edit, visibly
        moving the whole curve/residuals even though the underlying
        combined fit had not changed.

        Also stashes self._measurement_coefficients_px/
        _coefficient_sigma_px, for _refresh_measurement_display()'s
        Coefficients row.
        '''

        coefficients_px, coefficient_sigma_px = compute_combined_polynomial_for_degree(
            self._shot_results, degree,
        )
        self._measurement_coefficients_px = coefficients_px
        self._measurement_coefficient_sigma_px = coefficient_sigma_px

        fit_x = np.linspace(
            float(self._measurement_x_values.min()), float(self._measurement_x_values.max()),
            FIT_CURVE_N_POINTS,
        )
        fit_y_px = np.polynomial.polynomial.polyval(fit_x, coefficients_px)
        residual_px = self._measurement_x0_px - np.polynomial.polynomial.polyval(
            self._measurement_wavelength_nm, coefficients_px,
        )

        fit_y_mm, _ = self._convert_to_mm(fit_y_px, np.zeros_like(fit_y_px))
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

def _representative_shot_indices(n_shots: int) -> dict[str, int]:

    '''
    First, middle, and last shot index (by acquisition order) out of
    n_shots -- the 3 individual frames "Save Record" keeps a full-
    resolution copy of, per measurement_record.py's own module docstring
    (real, traceable primary data; not every shot, which wouldn't scale
    with n_shots). Deduplicated: at small n_shots (e.g. 2) two of the
    three candidate indices can coincide, in which case only the distinct
    ones are returned, keyed by whichever label reached that index first
    -- one frame is never saved twice under two different labels.
    '''

    candidates = [("first", 0), ("middle", n_shots // 2), ("last", n_shots - 1)]
    seen_indices: set[int] = set()
    selected: dict[str, int] = {}
    for label, index in candidates:
        if index in seen_indices:
            continue
        seen_indices.add(index)
        selected[label] = index
    return selected


def compute_combined_result_for_degree(
    shot_results: list[ShotAnalysisResult], degree: int, wavelength_ref_nm: float,
) -> CombinedSpatialDispersionResult:

    '''
    Combines shot_results' per-shot spatial dispersion at degree, exactly
    as ExtendedMeasurementScreen._compute_combined_result() does (that
    method is now a thin wrapper over this) -- pulled out as a free
    function so measurement_record.py's saved record can compute the same
    combined result for every degree, not just whichever one is currently
    selected on screen, using the exact same code path rather than
    independently re-derived logic that could silently drift from it.

    Degree 1: combine_shots() directly on each shot's fitted linear zeta
    (coefficients[1]) and its uncertainty (coefficient_sigma[1]) --
    exactly what combine_shots() was built for (see its own docstring's
    parameter description).

    Degree > 1: combine_shots() only combines the linear fit by design
    (see module docstring) -- there is no per-shot "coefficients[1]" that
    means the same thing at degree > 1. Instead, each shot's own
    zeta(wavelength_ref_nm)/sigma_zeta(wavelength_ref_nm) (a well-defined
    scalar at any degree, using that shot's full coefficient_covariance)
    is evaluated at the reference point, and those scalars are combined
    via the exact same combine_shots() call -- it is generic over any
    (value, sigma) pairs, not specific to raw fit coefficients.
    wavelength_ref_nm is ignored at degree 1 (accepted unconditionally
    anyway, so callers don't need a degree-dependent call shape).
    '''

    if degree == 1:
        zeta_values = np.array(
            [result.fits[1].coefficients[1] for result in shot_results]
        )
        sigma_zeta_values = np.array(
            [result.fits[1].coefficient_sigma[1] for result in shot_results]
        )
    else:
        wavelength_ref = np.array([wavelength_ref_nm])
        zeta_values = np.array([
            float(result.fits[degree].zeta(wavelength_ref)[0])
            for result in shot_results
        ])
        sigma_zeta_values = np.array([
            float(result.fits[degree].sigma_zeta(wavelength_ref)[0])
            for result in shot_results
        ])

    return combine_shots(zeta_values, sigma_zeta_values)


def compute_combined_polynomial_for_degree(
    shot_results: list[ShotAnalysisResult], degree: int,
) -> tuple[np.ndarray, np.ndarray]:

    '''
    Every coefficient of the degree-th polynomial fit (c0..c_degree),
    combined across shots via the exact same inverse-variance
    combine_shots() weighting compute_combined_result_for_degree() already
    uses to combine c1 alone at degree > 1 -- a real, statistically
    well-founded generalization, not an ad hoc addition: each shot's
    coefficients[k] is an independent estimate of the exact same physical
    coefficient (every shot measures the same underlying pixel ->
    wavelength dispersion relationship), so nothing about inverse-variance
    weighting is specific to k == 1. combine_shots() is called once per
    coefficient index, treating each index independently -- this ignores
    the (real) covariance between a single fit's own coefficients the same
    way this module's other marginal-uncertainty reporting already does
    elsewhere in this codebase (see calibration/spectral/calibrate.py's
    WavelengthCalibrationResult.sigma_wavelength_nm() docstring for the
    same documented approximation), not a new one introduced here.

    Gives an actual combined polynomial for degree > 1 -- the returned
    coefficients fully define a real quadratic/cubic curve
    x0(wavelength_nm) -- rather than only a single-reference-point tangent
    line, which is what ExtendedMeasurementScreen previously drew at every
    degree (moving visibly whenever the reference point changed, even
    though the underlying combined fit had not). Shared between the live
    GUI (ExtendedMeasurementScreen._refresh_measurement_display(), for
    degree > 1) and the saved record (measurement_record.py) so both draw
    the identical combined curve for a given set of shot_results, never
    independently re-derived logic that could drift between the two.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (coefficients, coefficient_sigma), each length degree + 1,
        ascending order (c0, c1, ...), in pixel units (x0_px = c0 +
        c1*wavelength_nm + c2*wavelength_nm**2 + ...).
    '''

    n_coefficients = degree + 1
    coefficients = np.empty(n_coefficients, dtype=np.float64)
    coefficient_sigma = np.empty(n_coefficients, dtype=np.float64)
    for k in range(n_coefficients):
        values = np.array([result.fits[degree].coefficients[k] for result in shot_results])
        sigma_values = np.array([result.fits[degree].coefficient_sigma[k] for result in shot_results])
        combined_k = combine_shots(values, sigma_values)
        coefficients[k] = combined_k.zeta_combined
        coefficient_sigma[k] = combined_k.sigma_zeta_combined
    return coefficients, coefficient_sigma


def compute_fit_line_and_residuals(
    x0_px: np.ndarray, sigma_x0_px: np.ndarray, wavelength_nm: np.ndarray, zeta_combined: float,
    x_range: tuple[float, float], n_points: int,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:

    '''
    The drawn fit-curve/residual arrays for a given combined zeta, in
    pixel units throughout: a single straight line of slope zeta_combined
    (matching analyze_shot()'s pixel-unit convention), anchored through
    the flattened (shot, column) data by a plain weighted-least-squares
    intercept -- the best-fit offset given that fixed, already-combined
    slope (see module docstring for why this is the only combination-
    consistent choice at any degree). Pulled out as a free function for
    the same reason as compute_combined_result_for_degree() above --
    ExtendedMeasurementScreen._recompute_fit_and_residuals() is now a thin
    wrapper over this plus its own mm-conversion (deliberately left to the
    caller: this function has no reason to depend on a
    ScaleFactorPositionCalibration).

    Parameters
    ----------
    x0_px, sigma_x0_px, wavelength_nm
        Every (shot, column) centroid's position/uncertainty (px) and
        wavelength (nm), flattened across shots -- same arrays
        ExtendedMeasurementScreen._set_measurement_data() builds.
    zeta_combined
        The already-combined slope (px/nm) this line is anchored at.
    x_range
        (min, max) wavelength/x-axis value the returned fit_x should span.
    n_points
        Number of points sampled along fit_x.

    Returns
    -------
    tuple
        (intercept_px, intercept_sigma_px, fit_x, fit_y_px, residual_px).
        intercept_sigma_px is the standard weighted-mean standard error,
        sqrt(1 / sum(weights)) -- conditional on zeta_combined as given
        (fixed), not propagating zeta_combined's own uncertainty into it.
        That mirrors the intercept's own two-stage derivation (a dependent
        second step after zeta_combined has already been combined, not a
        joint two-parameter fit), so reporting its uncertainty on that
        same conditional basis is consistent, not an extra approximation
        layered on top.
    '''

    weights = 1.0 / sigma_x0_px ** 2
    sum_weights = np.sum(weights)
    intercept_px = float(
        np.sum(weights * (x0_px - zeta_combined * wavelength_nm)) / sum_weights
    )
    intercept_sigma_px = float(np.sqrt(1.0 / sum_weights))

    fit_x = np.linspace(x_range[0], x_range[1], n_points)
    fit_y_px = intercept_px + zeta_combined * fit_x

    residual_px = x0_px - (intercept_px + zeta_combined * wavelength_nm)

    return intercept_px, intercept_sigma_px, fit_x, fit_y_px, residual_px


__all__ = [
    "ExtendedMeasurementScreen",
    "DEFAULT_N_SHOTS",
    "MIN_N_SHOTS",
    "MAX_N_SHOTS",
    "compute_combined_result_for_degree",
    "compute_combined_polynomial_for_degree",
    "compute_fit_line_and_residuals",
]
