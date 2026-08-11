'''
Extended-measurement interface: the N-shot combination workflow reached
from live_view.py's "Extended Measurement..." button. Static scatter plot
of per-shot centroid positions vs. wavelength (or pixel column, if no
wavelength calibration exists yet) with an overlaid fit line, a residual
subplot underneath it, a side info panel, and a combined-result summary --
in place of live_view.py's rolling strip chart, since this screen shows
one already-combined result rather than a live-updating trend
(docs/project_state.md #5's "Extended measurement" notes).

PHASE 1 (this file, as it stands): visual skeleton only -- layout,
styling, and placeholder/dummy data, mirroring live_view.py's own Phase 1
convention exactly. Clicking "Run Measurement" only regenerates and
redraws this widget's own placeholder scatter/fit/residual/combined-result
data; it never touches camera_stream, never reconfigures/restarts it (the
stop/reconfigure/restart cycle docs/project_state.md describes for an
exposure override is a follow-up wiring phase, same as live_view.py's
QTimer update loop), and never calls
pipeline.analysis.analyze_shot()/pipeline.analysis.combination.
combine_shots() for real. The degree selector is present and changes the
panel's *displayed* (still fake) combined-result numbers, but does not
trigger any real per-shot refit or recombination.

The Acquisition Settings side-panel section is the same kind of skeleton
as live_view.py's: exposure_us/gain_db spin boxes, pre-filled from the
loaded baseline's capture settings, feed the same drift check
(exposure_has_drifted()/gain_has_drifted(), imported from live_view.py)
against calibration_set.baseline_record (and conversion_gain_record, if
supplied). Crossing into the drifted state N/As the Combined Result group
and hides the scatter/error-bar/fit-curve/residual overlay, pops one
informational message, and emits recalibration_requested -- mirroring
LiveViewWidget's _recompute_settings_drift()/_enter_drifted_state()/
_exit_drifted_state() with this screen's own target widgets swapped in.
Does NOT reconfigure camera_stream, for the same reason live_view.py's
doesn't.
'''

# Imports

from dataclasses import dataclass

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

from pipeline.acquisition import CameraStream, CANONICAL_SHAPE, SPATIAL_AXIS, SPECTRAL_AXIS
from pipeline.analysis import SensorNoiseModel
from pipeline.analysis.interfaces import WavelengthAxis
from pipeline.calibration.sensor import ConversionGainRecord
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.preprocessing import CalibrationSet

from pipeline.gui.formatting import format_value_with_uncertainty, microns_to_mm
from pipeline.gui.live_view import (
    DEFAULT_DEGREE,
    DEGREE_CHOICES,
    DEGREE_LABELS,
    FIT_CURVE_COLOR,
    FIT_CURVE_WIDTH,
    SIDE_PANEL_WIDTH,
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


# Classes

@dataclass(frozen=True)
class _PlaceholderCombinedResult:

    '''
    Bundles one degree's worth of fake Combined Result side-panel numbers
    for the skeleton -- shaped like analysis.results.
    CombinedSpatialDispersionResult, but never constructed from a real
    combine_shots() call. Illustrative of typical real-world zeta
    magnitudes only; deliberately NOT required to be numerically
    consistent with the fake scatter/fit/residual data drawn in
    _generate_placeholder_data() (same justification live_view.py's
    _PlaceholderFit docstring already gives for its own placeholder
    numbers -- both are placeholders for different parts of the layout,
    not one coherent fake dataset).
    '''

    n_shots: int
    zeta_combined: float
    sigma_internal: float
    sigma_external: float
    sigma_zeta_combined: float
    # None for degree 1 (a real combine_shots() call combines the linear
    # zeta across shots directly -- docs/project_state.md #19/#20); a
    # caveat string for degree > 1, where the combined central value is
    # only meaningful as each shot's zeta(wavelength_ref) combined
    # afterward, not a joint fit (docs/project_state.md #5's "Extended
    # measurement" notes) -- see _build_combined_result_group().
    degree_note: str | None = None


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
        physically valid default even with no manual override).
    wavelength_axis
        Pixel -> wavelength(nm) conversion, or None until
        calibration/spectral/'s line_matching.py output is loaded (see
        docs/project_state.md). None is the expected v1 state, not an
        error -- the scatter falls back to a pixel-column x-axis, clearly
        labeled as such.
    camera_stream
        A CameraStream the caller owns the lifecycle of (start/stop are
        not this widget's responsibility). Not polled or reconfigured in
        this phase -- stored for the follow-up wiring phase's N-shot
        acquisition loop.
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

        self._current_degree = DEFAULT_DEGREE

        self._apply_pyqtgraph_theme()
        self._build_ui()
        self._populate_placeholder_data()

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
            "Skeleton only -- Run Measurement redraws this screen's own "
            "placeholder data. Does not acquire from camera_stream or "
            "call analyze_shot()/combine_shots()."
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
        calibration_set.baseline_record. SKELETON (see module docstring):
        editing either field never touches camera_stream -- it only
        drives the drift check below, warning when the entered value
        diverges from what the loaded calibrations were actually captured
        under. Identical role to LiveViewWidget's group of the same name.
        '''

        group = QGroupBox("Acquisition Settings")
        group.setFont(load_bundled_font(10))
        group.setStyleSheet(group_box_stylesheet())
        group.setToolTip(
            "Skeleton only -- does not reconfigure the camera. Pre-filled "
            "from the loaded baseline's capture settings; drifting past "
            "tolerance from the loaded calibrations hides the Combined "
            "Result panel and scatter/fit/residual overlay (reading "
            "\"N/A\") and shows an informational recalibration message."
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
        self._zeta_combined_label.setText("N/A")
        self._sigma_internal_label.setText("N/A")
        self._sigma_external_label.setText("N/A")
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
        self._update_combined_result_panel(self._current_degree)

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

        '''This screen's analogue of LiveViewWidget's "Fit Diagnostics" group.'''

        group = QGroupBox("Combined Result")
        group.setFont(load_bundled_font(10))
        group.setStyleSheet(group_box_stylesheet())
        form = QFormLayout(group)

        label_font = load_bundled_font(10)

        self._n_shots_label = QLabel("--")
        self._zeta_combined_label = QLabel("--")
        self._zeta_combined_label.setWordWrap(True)
        self._sigma_internal_label = QLabel("--")
        self._sigma_external_label = QLabel("--")
        self._degree_note_label = QLabel("")
        self._degree_note_label.setWordWrap(True)
        self._degree_note_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-style: italic;")

        for label in (
            self._n_shots_label, self._zeta_combined_label,
            self._sigma_internal_label, self._sigma_external_label, self._degree_note_label,
        ):
            label.setFont(label_font)

        form.addRow("N Shots:", self._n_shots_label)
        form.addRow("ζ Combined:", self._zeta_combined_label)
        form.addRow("σ Internal:", self._sigma_internal_label)
        form.addRow("σ External:", self._sigma_external_label)
        form.addRow("", self._degree_note_label)

        return group

    # -- placeholder data (Phase 1 only -- no real camera/analysis calls) --

    def _populate_placeholder_data(self) -> None:

        self._placeholder_combined_results = self._build_placeholder_combined_results()
        self._generate_placeholder_data()

        self._update_combined_result_panel(DEFAULT_DEGREE)

        self._apply_roi_bounds(*self._roi_control.roi_bounds_mm())

    def _generate_placeholder_data(self) -> None:

        '''
        Builds the full-range fake scatter/fit-curve/residual arrays and
        stashes them as instance attributes, without touching any
        pyqtgraph item -- _apply_roi_bounds() is what actually crops and
        renders them, so it can be re-run on its own (e.g. every time
        self._roi_control's roi_changed fires, or "Run Measurement" is
        clicked) without re-seeding the RNG unintentionally each time this
        method itself isn't called.

        Mirrors live_view.py's _generate_placeholder_data() shape, minus
        the heatmap (this screen has none -- see module docstring), plus
        residuals (observed minus the fake "true" fit-curve trend), which
        live_view.py's single-shot view has no equivalent of.
        '''

        rng = np.random.default_rng(seed=0)

        n_cols = CANONICAL_SHAPE[SPECTRAL_AXIS]
        n_rows = CANONICAL_SHAPE[SPATIAL_AXIS]

        centroid_slope_px_per_col = 0.05
        centroid_intercept_px = n_rows / 2

        def true_centroid_px(column: np.ndarray) -> np.ndarray:
            return centroid_intercept_px + centroid_slope_px_per_col * (column - n_cols / 2)

        # -- scatter + error bars ----------------------------------------
        columns = np.arange(200, 1720, 12)
        x_values = self._placeholder_x_values(columns)
        centroid_px = true_centroid_px(columns) + rng.normal(scale=3.0, size=columns.shape)
        x_sigma = (
            np.full_like(x_values, 0.4) if self._wavelength_axis is not None else None
        )

        self._placeholder_columns = columns
        self._placeholder_x_values_arr = x_values
        self._placeholder_centroid_px = centroid_px
        self._placeholder_x_sigma = x_sigma

        # -- fit-curve overlay (drawn from the same fake "true" trend, --
        # -- not from _placeholder_combined_results -- see that class's --
        # -- docstring) ----------------------------------------------------
        fit_x = np.linspace(x_values.min(), x_values.max(), 200)
        fit_columns = columns.min() + (fit_x - x_values.min()) / (
            x_values.max() - x_values.min()
        ) * (columns.max() - columns.min())
        fit_y_full, _ = self._convert_to_mm(
            true_centroid_px(fit_columns), np.zeros_like(fit_columns)
        )

        self._placeholder_fit_x = fit_x
        self._placeholder_fit_columns = fit_columns
        self._placeholder_fit_y_full = fit_y_full

        # -- residuals: observed centroid (in mm) minus the fit-curve --
        # -- trend evaluated at each observed column --------------------
        y_values_full, _ = self._convert_to_mm(
            centroid_px, np.zeros_like(centroid_px)
        )
        fit_at_columns_px = true_centroid_px(columns)
        fit_at_columns_mm, _ = self._convert_to_mm(
            fit_at_columns_px, np.zeros_like(fit_at_columns_px)
        )
        self._placeholder_residuals_full = y_values_full - fit_at_columns_mm

        self._placeholder_x_extent = (float(x_values.min()), float(x_values.max()))

        self._placeholder_rng = rng

    def _apply_roi_bounds(self, min_mm: float, max_mm: float) -> None:

        '''
        Crops the fake scatter/error-bars/fit-curve/residuals to
        [min_mm, max_mm] and renders them -- this screen's analogue of
        live_view.py's _apply_roi_bounds(), minus the heatmap masking (no
        ImageItem here -- see module docstring) and with a residual
        subplot update in place of that.
        '''

        y_values, y_sigma = self._convert_to_mm(
            self._placeholder_centroid_px,
            np.full_like(self._placeholder_centroid_px, 3.0),
        )
        keep = (y_values >= min_mm) & (y_values <= max_mm)

        x_values = self._placeholder_x_values_arr
        self._scatter.setData(x=x_values[keep], y=y_values[keep])
        error_kwargs = dict(
            x=x_values[keep], y=y_values[keep], top=y_sigma[keep], bottom=y_sigma[keep]
        )
        if self._placeholder_x_sigma is not None:
            error_kwargs["left"] = self._placeholder_x_sigma[keep]
            error_kwargs["right"] = self._placeholder_x_sigma[keep]
        self._error_bars.setData(**error_kwargs)

        fit_y_full = self._placeholder_fit_y_full
        keep_fit = (fit_y_full >= min_mm) & (fit_y_full <= max_mm)
        self._fit_curve.setData(
            x=self._placeholder_fit_x[keep_fit], y=fit_y_full[keep_fit]
        )

        self._residual_scatter.setData(
            x=x_values[keep], y=self._placeholder_residuals_full[keep]
        )

        self._main_plot.setRange(
            xRange=self._placeholder_x_extent, yRange=(min_mm, max_mm), padding=0
        )
        self._residual_plot.setRange(xRange=self._placeholder_x_extent, padding=0)

    def _on_roi_changed(self, min_mm: float, max_mm: float) -> None:

        self._apply_roi_bounds(min_mm, max_mm)

    # -- run configuration ------------------------------------------------

    def _on_run_clicked(self) -> None:

        '''
        "Run Measurement" handler. SKELETON (see module docstring): only
        regenerates and redraws this widget's own placeholder data -- does
        not acquire from self._camera_stream, and does not call
        pipeline.analysis.analyze_shot()/combine_shots(). n_shots is read
        from self._n_shots_spin only to feed the placeholder Combined
        Result panel's "N Shots" row, not to drive any real acquisition
        count.
        '''

        n_shots = self._n_shots_spin.value()
        self._placeholder_combined_results = self._build_placeholder_combined_results(
            n_shots=n_shots
        )
        self._generate_placeholder_data()
        self._update_combined_result_panel(self._current_degree)
        self._apply_roi_bounds(*self._roi_control.roi_bounds_mm())

    # -- degree selector (stub -- see module docstring) -------------------

    def _on_degree_changed(self, index: int) -> None:

        degree = self._degree_selector.itemData(index)
        if degree is None:
            return
        self._current_degree = degree
        # NOTE: this only swaps in this widget's own pre-baked placeholder
        # numbers for the newly-selected degree -- it does not re-run
        # combine_shots() or recompute anything. Real recomputation is
        # wired up in the follow-up phase.
        self._update_combined_result_panel(degree)

    def _update_combined_result_panel(self, degree: int) -> None:

        result = self._placeholder_combined_results[degree]

        self._n_shots_label.setText(str(result.n_shots))
        self._zeta_combined_label.setText(
            format_value_with_uncertainty(result.zeta_combined, result.sigma_zeta_combined)
        )
        self._sigma_internal_label.setText(f"{result.sigma_internal:.4g}")
        self._sigma_external_label.setText(f"{result.sigma_external:.4g}")

        self._formula_label.setText(fit_formula_html(degree, self._wavelength_axis))

        self._degree_note_label.setText(result.degree_note or "")

    # -- pure-ish helpers (presentational only, no camera/analysis calls) --

    def _x_axis_label(self) -> str:
        return wavelength_axis_label(self._wavelength_axis)

    def _placeholder_x_values(self, columns: np.ndarray) -> np.ndarray:
        if self._wavelength_axis is not None:
            return self._wavelength_axis.wavelength_nm(columns)
        return columns.astype(float)

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

    @staticmethod
    def _build_placeholder_combined_results(
        n_shots: int = DEFAULT_N_SHOTS,
    ) -> dict[int, _PlaceholderCombinedResult]:

        '''
        Fake Combined Result side-panel numbers, one per DEGREE_CHOICES
        value, illustrative of typical real-world zeta magnitudes once a
        real combine_shots() result exists. Only degree 1 has a real
        combine_shots()-shaped meaning (analysis/combination.py combines
        the linear zeta across shots by design); degree > 1 entries carry
        an explicit degree_note caveat -- see docs/project_state.md #5's
        "Extended measurement" notes.

        Deliberately NOT derived from (and not numerically consistent
        with) the fake scatter/fit-curve/residual data built in
        _generate_placeholder_data() -- see _PlaceholderCombinedResult's
        docstring.
        '''

        return {
            1: _PlaceholderCombinedResult(
                n_shots=n_shots,
                zeta_combined=1.6e-3,
                sigma_internal=2.1e-5,
                sigma_external=1.8e-5,
                sigma_zeta_combined=2.1e-5,
            ),
            2: _PlaceholderCombinedResult(
                n_shots=n_shots,
                zeta_combined=1.62e-3,
                sigma_internal=3.4e-5,
                sigma_external=4.0e-5,
                sigma_zeta_combined=4.0e-5,
                degree_note=(
                    "Degree > 1 combination is evaluated per-shot at a "
                    "reference wavelength/column, not a joint fit -- see "
                    "docs/project_state.md."
                ),
            ),
            3: _PlaceholderCombinedResult(
                n_shots=n_shots,
                zeta_combined=1.58e-3,
                sigma_internal=4.1e-5,
                sigma_external=5.2e-5,
                sigma_zeta_combined=5.2e-5,
                degree_note=(
                    "Degree > 1 combination is evaluated per-shot at a "
                    "reference wavelength/column, not a joint fit -- see "
                    "docs/project_state.md."
                ),
            ),
        }


# Functions


__all__ = [
    "ExtendedMeasurementScreen",
    "DEFAULT_N_SHOTS",
    "MIN_N_SHOTS",
    "MAX_N_SHOTS",
]
