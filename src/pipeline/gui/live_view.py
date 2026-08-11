'''
Live-view interface: the main screen shown once the calibration screen
hands off loaded/created calibrations. Live-updating scatter plot of
centroid position vs. wavelength (or pixel column, if no wavelength
calibration exists yet), with an overlaid fit curve, a raw-image heatmap
underneath, a side info panel, and a rolling trend chart.

PHASE 1 (this file, as it stands): visual skeleton only -- layout,
styling, and placeholder/dummy data. No real camera polling,
preprocessing, or analyze_shot() calls happen here yet; the QTimer-driven
update loop, the skip-counter state machine, and the real per-tick
computation described in the module design are a follow-up phase. The
degree selector is present and changes the panel's *displayed* (still
fake) numbers, but does not trigger any real refit.

The Acquisition Settings side-panel section (exposure_us/gain_db spin
boxes, pre-filled from the loaded baseline's capture settings) is the
same kind of skeleton: editing either field re-evaluates a combined
drift state against calibration_set.baseline_record (and
conversion_gain_record, if supplied) via _recompute_settings_drift().
Crossing from "in tolerance" to "drifted" hides the fit diagnostics
(reading "N/A") and the scatter/error-bar/fit-curve overlay (the raw
heatmap keeps displaying underneath) and pops a single informational
message; returning to "in tolerance" restores both. Does NOT reconfigure
camera_stream, though -- that's deferred to the same future "real camera
wiring" phase as the QTimer update loop above.
'''

# Imports

import math
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
    QVBoxLayout,
    QWidget,
)

from pipeline.acquisition import CameraStream, CANONICAL_SHAPE, SPATIAL_AXIS, SPECTRAL_AXIS
from pipeline.analysis import SensorNoiseModel
from pipeline.analysis.interfaces import WavelengthAxis
from pipeline.calibration.sensor import ConversionGainRecord
from pipeline.calibration.shared import EXPOSURE_MATCH_TOLERANCE_REL, GAIN_MATCH_TOLERANCE_ABS
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.preprocessing import CalibrationSet

from pipeline.gui.formatting import format_value_with_uncertainty, MICRONS_PER_MM, microns_to_mm
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

# Polynomial degree choices for the fit-curve overlay / degree selector,
# in the order they appear in the combo box.
DEGREE_CHOICES = (1, 2, 3)
DEGREE_LABELS = {1: "Linear (degree 1)", 2: "Quadratic (degree 2)", 3: "Cubic (degree 3)"}
DEFAULT_DEGREE = 1

# Rolling strip chart window, in seconds (see module docstring in the
# eventual update-loop phase -- 60s is a starting judgment call, not a
# measured requirement).
STRIP_CHART_WINDOW_SECONDS = 60.0

SIDE_PANEL_WIDTH = 300

# Spectral-column center used for the degree > 1 placeholder's "evaluated
# at" note (n_cols / 2 from _populate_placeholder_data()'s fake 1920-column
# frame) -- a fixed illustrative placeholder, like the rest of
# _PlaceholderFit, not derived from any real per-frame column range.
EVALUATED_AT_COLUMN = 960.0

# Fit-curve overlay styling -- black rather than COLOR_ACCENT, and thicker
# than the default 2px, specifically so it reads clearly against every
# region of the viridis heatmap underneath it (COLOR_ACCENT's blue was
# hard to distinguish against the heatmap's own blue/purple low end).
FIT_CURVE_COLOR = "#000000"
FIT_CURVE_WIDTH = 3

# Unicode superscript digits/minus sign, for rendering an axis's
# power-of-ten scale annotation (e.g. "x10^-3") properly instead of
# pyqtgraph's default "%g" text (e.g. "x0.001") -- see
# format_power_of_ten_superscript()/_PowerOfTenAxisItem below.
_SUPERSCRIPT_DIGITS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "-": "⁻",
}

# Classes

@dataclass(frozen=True)
class _PlaceholderFit:

    '''Bundles one degree's worth of fake side-panel numbers for the skeleton.'''

    coefficients: tuple[float, ...]
    coefficient_sigma: tuple[float, ...]
    reduced_chi_squared: float
    zeta_value: float
    zeta_sigma: float | None   # None => "uncertainty not available" note (degree > 1)
    # x-value (pixel column) the placeholder zeta_value was evaluated at --
    # only meaningful when zeta_sigma is None (degree > 1); see
    # evaluated_at_text() and the "Evaluated At:" side-panel row.
    evaluated_at_column: float | None = None


class _PowerOfTenAxisItem(pg.AxisItem):

    '''
    AxisItem whose auto-SI-prefix scale annotation is rendered as proper
    Unicode-superscript scientific notation ("x10^-3") instead of
    pyqtgraph's default plain-decimal "%g" formatting ("x0.001") -- see
    format_power_of_ten_superscript(). Used for the strip chart's left
    axis, where "Spatial Dispersion" has no physical SI unit to prefix
    (px/nm), so pyqtgraph's default unit-prefix annotation is the only
    place this scale factor is ever shown.
    '''

    def labelString(self) -> str:

        if self.labelUnits != "" or not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
            return super().labelString()

        units = f"({format_power_of_ten_superscript(1.0 / self.autoSIPrefixScale)})"
        style = ";".join(f"{k}: {self.labelStyle[k]}" for k in self.labelStyle)
        s = f"{self.labelText} {units}"
        return f"<span style='{style}'>{s}</span>"


class LiveViewWidget(QWidget):

    '''
    Main live-view widget: scatter + fit overlay + heatmap on top, a
    rolling zeta-vs-time strip chart, and a side info panel, wired around
    (not responsible for producing) an already-assembled calibration set,
    noise model, position calibration, optional wavelength axis, and a
    running camera stream.

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
        calibration/spectral/'s line_matching.py is implemented (see
        docs/project_state.md). None is the expected v1 state, not an
        error -- the scatter/heatmap fall back to a pixel-column x-axis,
        clearly labeled as such.
    camera_stream
        A CameraStream the caller owns the lifecycle of (start/stop are
        not this widget's responsibility). Not polled in this phase --
        stored for the follow-up wiring phase's QTimer loop.
    conversion_gain_record
        ConversionGainRecord (gain_db + timing/sweep metadata, no
        exposure_us -- conversion gain sweeps exposure by design) tagging
        the loaded conversion-gain artifact, or None if no conversion-gain
        artifact was loaded. When supplied, the Acquisition Settings
        panel's gain_db field is drift-checked against it in addition to
        calibration_set.baseline_record.gain_db, since the two artifacts
        can drift independently of each other.
    parent
        Standard Qt parent widget.
    '''

    recalibration_requested = Signal(str)
    extended_measurement_requested = Signal()

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
        # with an already-mismatched baseline/conversion-gain pair (e.g.
        # bypassing calibration_screen.py's own gate) -- starts in whatever
        # state the supplied records actually imply, not assumed "OK".
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
        left_column.addWidget(self._build_status_bar())
        left_column.addWidget(self._build_main_plot(), stretch=3)
        left_column.addWidget(self._build_strip_chart(), stretch=1)
        root_layout.addLayout(left_column, stretch=1)

        root_layout.addWidget(self._build_side_panel())

    def _build_status_bar(self) -> QWidget:

        # Purely visual placeholder for the connection-lost /
        # insufficient-signal states the real update loop will drive --
        # not wired to camera_stream.is_running/last_error yet.
        label = QLabel("Status: OK  (skeleton -- not connected to live camera state)")
        label.setFont(load_bundled_font(10))
        label.setStyleSheet(f"color: {ACCENT_COLOR};")
        return label

    def _build_main_plot(self) -> pg.PlotWidget:

        plot_widget = pg.PlotWidget()
        self._main_plot = plot_widget.getPlotItem()
        self._main_plot.setLabel("left", "Relative Physical Position (mm)")
        self._main_plot.setLabel("bottom", self._x_axis_label())
        self._main_plot.showGrid(x=True, y=True, alpha=0.3)
        self._style_plot_axes(self._main_plot)

        # Heatmap first (so everything else renders on top of it), then
        # scatter/error bars, then the fit curve LAST -- it traces nearly
        # the same path as the scatter points, so drawing it underneath
        # them (the original order) left it almost entirely hidden behind
        # the denser, larger scatter markers.
        self._image_item = pg.ImageItem()
        self._image_item.setColorMap(pg.colormap.get("viridis"))
        self._main_plot.addItem(self._image_item)

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

    def _build_strip_chart(self) -> pg.PlotWidget:

        plot_widget = pg.PlotWidget(
            axisItems={"left": _PowerOfTenAxisItem(orientation="left")}
        )
        self._strip_plot = plot_widget.getPlotItem()
        self._strip_plot.setLabel("left", "Spatial Dispersion")
        self._strip_plot.setLabel("bottom", "Time (s ago)")
        self._strip_plot.showGrid(x=True, y=True, alpha=0.3)
        self._style_plot_axes(self._strip_plot)

        self._strip_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=ACCENT_COLOR, width=1),
            symbol="o",
            symbolSize=4,
            symbolBrush=ACCENT_COLOR,
            symbolPen=None,
        )
        self._strip_plot.addItem(self._strip_curve)

        plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Tall enough for the rotated left-axis label to fit without
        # clipping once its power-of-ten scale annotation is appended
        # (see _PowerOfTenAxisItem) -- 140/180 (the pre-annotation cap)
        # cut "Spatial Dispersion (x10^-3)" off partway through.
        plot_widget.setMinimumHeight(260)
        plot_widget.setMaximumHeight(340)
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

        # Acquisition Settings sits above the fit-related groups: it
        # describes what the *camera* is doing (and whether that still
        # matches what the loaded calibrations were built under), which
        # is a precondition for trusting the fit diagnostics below it,
        # not a peer of them.
        layout.addWidget(self._build_acquisition_settings_group())
        layout.addWidget(self._build_spatial_roi_group())
        layout.addWidget(self._build_degree_selector_group())
        layout.addWidget(self._build_fit_diagnostics_group())
        layout.addStretch(1)

        self._extended_measurement_button = QPushButton("Extended Measurement...")
        self._extended_measurement_button.setFont(load_bundled_font(13, bold=True))
        self._extended_measurement_button.setMinimumHeight(56)
        self._extended_measurement_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_COLOR}; color: {BACKGROUND_COLOR}; "
            f"border-radius: 6px; padding: 10px; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_COLOR}; border: 2px solid {FOREGROUND_COLOR}; }}"
        )
        self._extended_measurement_button.setToolTip(
            "Opens the Extended Measurement screen (N-shot combination "
            "workflow) -- itself still a Phase 1 visual skeleton, see "
            "gui/extended_measurement.py's module docstring."
        )
        self._extended_measurement_button.clicked.connect(
            self.extended_measurement_requested
        )
        layout.addWidget(self._extended_measurement_button)

        return panel

    def _build_acquisition_settings_group(self) -> QGroupBox:

        '''
        Exposure/gain display+entry, pre-filled from
        calibration_set.baseline_record. SKELETON (see module docstring):
        editing either field never touches camera_stream -- it only
        drives the drift check below, warning when the entered value
        diverges from what the loaded calibrations were actually captured
        under.
        '''

        group = QGroupBox("Acquisition Settings")
        group.setFont(load_bundled_font(10))
        group.setStyleSheet(group_box_stylesheet())
        group.setToolTip(
            "Skeleton only -- does not reconfigure the camera. Pre-filled "
            "from the loaded baseline's capture settings; drifting past "
            "tolerance from the loaded calibrations hides the fit "
            "diagnostics and overlay (reading \"N/A\", raw heatmap still "
            "shown) and shows an informational recalibration message."
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

        # exposure_us is unused directly -- kept as the parameter so this
        # stays a valid QDoubleSpinBox.valueChanged slot signature; the
        # recompute reads the spin box's current value itself, the same
        # way _on_gain_changed does.
        self._recompute_settings_drift()

    def _on_gain_changed(self, gain_db: float) -> None:

        self._recompute_settings_drift()

    def _recompute_settings_drift(self) -> None:

        '''
        Combined baseline/conversion-gain drift check, driving a single
        self._settings_drifted state machine rather than the three
        independent per-comparison prompts a naive per-field check would
        produce. Reads self._exposure_spin/self._gain_spin's *current*
        values directly (not a parameter) so it can be called equally from
        a spin box's valueChanged slot or from __init__ before any signal
        has ever fired.

        Gain is checked against both baseline_record and (if loaded)
        conversion_gain_record independently, since the two artifacts can
        drift apart from one another -- e.g. a re-run baseline at the new
        gain would clear the baseline comparison while the conversion-gain
        artifact is still stale.

        On a False -> True transition, enters the drifted state (N/A
        diagnostics, hidden overlay, one informational popup). On a
        True -> False transition, exits it (restores real diagnostics and
        overlay visibility). No-op otherwise -- this is what makes the
        popup fire once per drift episode, not on every spin-box tick.
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
        Entered on a False -> True self._settings_drifted transition: the
        fit diagnostics can no longer be trusted against the newly-entered
        exposure/gain, so they're replaced with an explicit "N/A" rather
        than left showing stale numbers, and the scatter/error-bar/
        fit-curve overlay is hidden (the raw heatmap -- self._image_item --
        is left alone, since it's still a faithful live view of the
        detector, just not one the current fit/dispersion figures apply
        to). Pops one informational message, and emits
        recalibration_requested for whichever comparison(s) actually
        drifted, so a future recalibration-launcher still knows which
        artifact(s) need reopening.

        Parameters
        ----------
        baseline_drifted
            True if exposure or gain drifted from calibration_set.
            baseline_record.
        conversion_gain_drifted
            True if gain drifted from self._conversion_gain_record (always
            False when no conversion-gain record was supplied).
        '''

        self._chi_squared_label.setText("N/A")
        self._coefficients_label.setText("N/A")
        self._zeta_label.setText("N/A")
        self._zeta_note_label.setText("")
        self._evaluated_at_label.setText("")

        self._scatter.setVisible(False)
        self._error_bars.setVisible(False)
        self._fit_curve.setVisible(False)

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
        restores the scatter/error-bar/fit-curve overlay's visibility and
        recomputes the fit-diagnostics panel for the currently-selected
        degree, undoing _enter_drifted_state()'s "N/A" placeholders.
        '''

        self._scatter.setVisible(True)
        self._error_bars.setVisible(True)
        self._fit_curve.setVisible(True)
        self._update_fit_panel(self._current_degree)

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

        # Polynomial-form formula, kept in sync with the degree selector by
        # _on_degree_changed() -- see fit_formula_html().
        self._formula_label = QLabel(fit_formula_html(DEFAULT_DEGREE, self._wavelength_axis))
        self._formula_label.setTextFormat(Qt.RichText)
        self._formula_label.setWordWrap(True)
        self._formula_label.setFont(load_bundled_font(10))
        layout.addWidget(self._formula_label)

        return group

    def _build_fit_diagnostics_group(self) -> QGroupBox:

        group = QGroupBox("Fit Diagnostics")
        group.setFont(load_bundled_font(10))
        group.setStyleSheet(group_box_stylesheet())
        form = QFormLayout(group)

        label_font = load_bundled_font(10)

        self._chi_squared_label = QLabel("--")
        self._coefficients_label = QLabel("--")
        self._coefficients_label.setTextFormat(Qt.RichText)
        self._coefficients_label.setWordWrap(True)
        self._zeta_label = QLabel("--")
        self._zeta_label.setWordWrap(True)
        self._zeta_note_label = QLabel("")
        self._zeta_note_label.setWordWrap(True)
        self._zeta_note_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-style: italic;")
        # Degree > 1 only -- the x-value spatial dispersion's placeholder
        # value was evaluated at (see _update_fit_panel()/evaluated_at_text()).
        self._evaluated_at_label = QLabel("")
        self._evaluated_at_label.setWordWrap(True)

        for label in (
            self._chi_squared_label, self._coefficients_label,
            self._zeta_label, self._zeta_note_label, self._evaluated_at_label,
        ):
            label.setFont(label_font)

        form.addRow("Reduced Chi-Squared:", self._chi_squared_label)
        form.addRow("Coefficients:", self._coefficients_label)
        form.addRow("Spatial Dispersion:", self._zeta_label)
        form.addRow("", self._zeta_note_label)
        form.addRow("Evaluated At:", self._evaluated_at_label)

        return group

    # -- placeholder data (Phase 1 only -- no real camera/analysis calls) --

    def _populate_placeholder_data(self) -> None:

        self._placeholder_fits = self._build_placeholder_fits()
        self._generate_placeholder_data()

        self._update_strip_chart_placeholder(self._placeholder_rng)
        self._update_fit_panel(DEFAULT_DEGREE)

        self._apply_roi_bounds(*self._roi_control.roi_bounds_mm())

    def _generate_placeholder_data(self) -> None:

        '''
        Builds the full-range fake scatter/fit-curve/heatmap arrays and
        stashes them as instance attributes, without touching any
        pyqtgraph item -- _apply_roi_bounds() is what actually crops and
        renders them, so it can be re-run on its own (e.g. every time
        self._roi_control's roi_changed fires) without re-seeding the RNG
        and drawing a new random dataset each time.
        '''

        rng = np.random.default_rng(seed=0)

        n_rows, n_cols = CANONICAL_SHAPE[SPATIAL_AXIS], CANONICAL_SHAPE[SPECTRAL_AXIS]
        self._placeholder_n_rows = n_rows
        self._placeholder_n_cols = n_cols

        # A single fake "true" beam-centroid trend (in raw spatial pixels,
        # as a function of spectral column), shared by the heatmap and the
        # scatter/fit-curve so the picture is at least internally
        # consistent -- NOT derived from any real analysis.
        centroid_slope_px_per_col = 0.05
        centroid_intercept_px = n_rows / 2

        def true_centroid_px(column: np.ndarray) -> np.ndarray:
            return centroid_intercept_px + centroid_slope_px_per_col * (column - n_cols / 2)

        # -- scatter + error bars --------------------------------------
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
        # -- not from _placeholder_fits -- see docstring above) ---------
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

        # -- fake raw-preprocessed-frame heatmap, standing in for -------
        # -- ProcessedFrame.image, using the same beam-centroid trend ---
        row_idx = np.arange(n_rows)[:, None]
        col_idx = np.arange(n_cols)[None, :]
        beam_center = true_centroid_px(col_idx)
        image = 200 * np.exp(-((row_idx - beam_center) ** 2) / (2 * 60**2))
        image = image + rng.normal(scale=5, size=image.shape)
        image = np.clip(image, 0, None).astype(np.float32)

        self._placeholder_image_full = image

        # An image must already be assigned before setRect() below --
        # pyqtgraph's ImageItem.setRect() scales by self.width()/
        # self.height(), which silently fall back to 1.0 if no image has
        # ever been set yet, producing a wildly wrong transform (the
        # visible plot then shows only the image's extreme top-left
        # corner, stretched to fill the whole view -- a uniform flat
        # color, not a heatmap). _apply_roi_bounds() below calls
        # setImage() again with the actual ROI-cropped array; this call
        # only exists to make width()/height() valid in time for setRect().
        self._image_item.setImage(image)

        # setRect()'s extent depends only on the (fixed) frame shape and
        # wavelength axis, never on the current ROI, so it's computed
        # once here rather than on every _apply_roi_bounds() call.
        x0, x1 = self._heatmap_x_extent(first_column=0, last_column=n_cols - 1)
        y0, y1 = self._convert_to_mm(
            np.array([0.0, float(n_rows)]), np.array([0.0, 0.0])
        )[0]
        self._image_item.setRect(
            min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
        )
        self._placeholder_x_extent = (min(x0, x1), max(x0, x1))

        # Kept for _update_strip_chart_placeholder() to continue the same
        # random stream _populate_placeholder_data() threads it through,
        # rather than resetting to a fresh seed=0 generator and changing
        # the strip chart's random values.
        self._placeholder_rng = rng

    def _apply_roi_bounds(self, min_mm: float, max_mm: float) -> None:

        '''
        Crops the fake scatter/error-bars/fit-curve/heatmap to
        [min_mm, max_mm] and renders them -- the real system's analogue of
        preprocessing/steps/roi.py's apply_roi() zeroing rows outside the
        spatial ROI (no valid centroid there), so out-of-window scatter/
        fit points are dropped entirely rather than merely clipped from
        view. Callable both at startup (_populate_placeholder_data()) and
        on every self._roi_control.roi_changed signal (_on_roi_changed()).

        Sets BOTH axis ranges via one setRange() call, even though the ROI
        only ever changes the y-range -- pinning x explicitly (rather than
        leaving it on pyqtgraph's autorange) works around a real pyqtgraph
        quirk: ScatterPlotItem's default pxMode markers size themselves in
        screen pixels, and converting that to a data-space bounding rect
        for autorange requires a pixel<->data transform that doesn't exist
        yet before the widget's first paint. Fixing y alone left x on that
        not-yet-valid autorange, which computed a wildly wrong x extent
        (observed: roughly [-1.9e5, 3.8e6] instead of [0, 1919]) that
        squeezed all the real scatter/fit-curve data into an invisible
        sliver. x's own extent never depends on the ROI, so re-pinning it
        to the same value every call is harmless.
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

        row_min_px, row_max_px = self._roi_control.roi_bounds_px()
        masked_image = self._placeholder_image_full.copy()
        masked_image[:row_min_px, :] = 0
        masked_image[row_max_px:, :] = 0
        self._image_item.setImage(masked_image)

        self._main_plot.setRange(
            xRange=self._placeholder_x_extent, yRange=(min_mm, max_mm), padding=0
        )

    def _on_roi_changed(self, min_mm: float, max_mm: float) -> None:

        self._apply_roi_bounds(min_mm, max_mm)

    def _update_strip_chart_placeholder(self, rng: np.random.Generator) -> None:

        n_points = 60
        t = np.linspace(-STRIP_CHART_WINDOW_SECONDS, 0, n_points)
        zeta_center = self._placeholder_fits[DEFAULT_DEGREE].zeta_value
        trend = zeta_center + 0.05 * zeta_center * np.sin(t / 12.0)
        noisy = trend + rng.normal(scale=0.02 * abs(zeta_center) + 1e-6, size=t.shape)
        self._strip_curve.setData(x=t, y=noisy)

    # -- degree selector (stub -- see module docstring) -----------------

    def _on_degree_changed(self, index: int) -> None:

        degree = self._degree_selector.itemData(index)
        if degree is None:
            return
        self._current_degree = degree
        # NOTE: this only swaps in this widget's own pre-baked placeholder
        # numbers for the newly-selected degree -- it does not re-run
        # analyze_shot() or refit anything. Real recomputation is wired up
        # in the follow-up phase.
        self._update_fit_panel(degree)

    def _update_fit_panel(self, degree: int) -> None:

        fit = self._placeholder_fits[degree]

        self._chi_squared_label.setText(f"{fit.reduced_chi_squared:.3f}")

        coeff_lines = [
            f"c<sub>{i}</sub> = {format_value_with_uncertainty(c, s)}"
            for i, (c, s) in enumerate(zip(fit.coefficients, fit.coefficient_sigma))
        ]
        self._coefficients_label.setText("<br>".join(coeff_lines))

        self._formula_label.setText(fit_formula_html(degree, self._wavelength_axis))

        if fit.zeta_sigma is not None:
            self._zeta_label.setText(format_value_with_uncertainty(fit.zeta_value, fit.zeta_sigma))
            self._zeta_note_label.setText("")
            self._evaluated_at_label.setText("")
        else:
            self._zeta_label.setText(f"{fit.zeta_value:.4g} (no uncertainty)")
            self._evaluated_at_label.setText(
                evaluated_at_text(fit.evaluated_at_column, self._wavelength_axis)
            )
            self._zeta_note_label.setText(
                "Uncertainty not available for degree > 1 in live view "
                "(no internal covariance-based estimate exists yet)."
            )

    # -- pure-ish helpers (presentational only, no camera/analysis calls) --

    def _x_axis_label(self) -> str:
        return wavelength_axis_label(self._wavelength_axis)

    def _placeholder_x_values(self, columns: np.ndarray) -> np.ndarray:
        if self._wavelength_axis is not None:
            return self._wavelength_axis.wavelength_nm(columns)
        return columns.astype(float)

    def _heatmap_x_extent(self, first_column: int, last_column: int) -> tuple[float, float]:
        return heatmap_x_extent(self._wavelength_axis, first_column, last_column)

    def _convert_to_mm(
        self, x0: np.ndarray, sigma_x0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        '''
        position_calibration.convert() returns microns (see
        calibration/spatial/calibrate.py's module docstring) -- every
        on-screen y-axis/heatmap call site goes through this instead of
        calling convert() directly, so the widget's "(mm)"-labeled axis
        always actually reads in mm.
        '''

        y0, sigma_y0 = self._position_calibration.convert(x0, sigma_x0)
        return microns_to_mm(y0), microns_to_mm(sigma_y0)

    @staticmethod
    def _build_placeholder_fits() -> dict[int, _PlaceholderFit]:

        '''
        Fake side-panel numbers, illustrative of typical real-world
        zeta magnitudes (px/nm-scale dispersion) once a real
        SpatialDispersionFitResult exists. Deliberately NOT derived from
        (and not numerically consistent with) the fake scatter/fit-curve/
        heatmap drawn in _populate_placeholder_data(), which instead uses
        its own simple "true" pixel-space trend -- both are placeholders
        for different parts of the layout, not a single coherent fake
        dataset.
        '''

        return {
            1: _PlaceholderFit(
                coefficients=(0.02, 1.6e-3),
                coefficient_sigma=(0.01, 2.1e-5),
                reduced_chi_squared=1.04,
                zeta_value=1.6e-3,
                zeta_sigma=2.1e-5,
            ),
            2: _PlaceholderFit(
                coefficients=(0.01, 1.55e-3, 3.0e-7),
                coefficient_sigma=(0.01, 3.0e-5, 1.0e-7),
                reduced_chi_squared=0.98,
                zeta_value=1.62e-3,
                zeta_sigma=None,
                evaluated_at_column=EVALUATED_AT_COLUMN,
            ),
            3: _PlaceholderFit(
                coefficients=(0.01, 1.5e-3, 2.8e-7, -4.0e-10),
                coefficient_sigma=(0.01, 4.0e-5, 1.5e-7, 6.0e-10),
                reduced_chi_squared=0.97,
                zeta_value=1.58e-3,
                zeta_sigma=None,
                evaluated_at_column=EVALUATED_AT_COLUMN,
            ),
        }


# Functions

def format_power_of_ten_superscript(multiplier: float) -> str:

    '''
    Formats a power-of-ten multiplier (e.g. 0.001) as "x10^-3" using
    real Unicode superscript characters, for _PowerOfTenAxisItem's axis
    scale annotation. Pure/non-Qt, unlike _PowerOfTenAxisItem itself, so
    it's directly unit-testable.

    Parameters
    ----------
    multiplier
        A strictly positive, finite power of ten (e.g.
        1.0 / AxisItem.autoSIPrefixScale). Anything that isn't a power
        of ten (within floating-point tolerance) raises -- silently
        rendering a non-power-of-ten value this way would be wrong, not
        just imprecise.

    Returns
    -------
    str
        e.g. "×10⁻³" for multiplier=0.001, "×10³" for multiplier=1000.0.

    Raises
    ------
    ValueError
        If multiplier is not finite/positive, or not a power of ten.
    '''

    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(f"multiplier must be finite and positive, got {multiplier!r}")

    exponent = round(math.log10(multiplier))
    if not math.isclose(multiplier, 10.0 ** exponent, rel_tol=1e-9):
        raise ValueError(f"{multiplier!r} is not a power of ten")

    sign = "-" if exponent < 0 else ""
    digits = str(abs(exponent))
    superscript_exponent = "".join(_SUPERSCRIPT_DIGITS[c] for c in sign + digits)
    return f"×10{superscript_exponent}"


def wavelength_axis_label(wavelength_axis: WavelengthAxis | None) -> str:

    '''
    X-axis label for the main plot: real wavelength when a
    WavelengthAxis is supplied, an explicit "not yet available" fallback
    label when it's None (the expected v1 state, not an error -- see
    calibration/spectral/line_matching.py's module docstring).
    '''

    if wavelength_axis is not None:
        return "Wavelength (nm)"
    return "Pixel Column (wavelength calibration not yet available)"


def heatmap_x_extent(
    wavelength_axis: WavelengthAxis | None, first_column: int, last_column: int
) -> tuple[float, float]:

    '''
    X-extent, in plot units, to stretch the heatmap ImageItem across via
    setRect() -- a one-time-per-frame linear axis transform, not
    per-pixel interpolation (see this module's design notes). Falls back
    to the raw 1:1 pixel-column extent when no wavelength calibration
    exists yet.

    Parameters
    ----------
    wavelength_axis
        Supplies wavelength_nm(pixel), or None for the pixel-column
        fallback.
    first_column, last_column
        Pixel-column indices of the image's first and last spectral
        columns.

    Returns
    -------
    tuple[float, float]
        (x_start, x_end) in plot units.
    '''

    if wavelength_axis is None:
        return float(first_column), float(last_column)
    endpoints = wavelength_axis.wavelength_nm(np.array([first_column, last_column]))
    return float(endpoints[0]), float(endpoints[1])


def evaluated_at_text(column: float, wavelength_axis: WavelengthAxis | None) -> str:

    '''
    Value shown in the degree > 1 side-panel's "Evaluated At:" row --
    the x-value the placeholder spatial-dispersion figure was evaluated
    at, converted to wavelength when a WavelengthAxis exists (matching
    this module's existing pixel/wavelength fallback convention), else
    left as a raw pixel-column index.

    Parameters
    ----------
    column
        Pixel-column index the value was evaluated at.
    wavelength_axis
        Supplies wavelength_nm(pixel), or None for the pixel-column
        fallback.

    Returns
    -------
    str
        e.g. "532.4 nm" or "Pixel column 960".
    '''

    if wavelength_axis is not None:
        wavelength_nm = float(wavelength_axis.wavelength_nm(np.array([column]))[0])
        return f"{wavelength_nm:.1f} nm"
    return f"Pixel column {round(column):d}"


def fit_formula_html(degree: int, wavelength_axis: WavelengthAxis | None) -> str:

    '''
    Rich-text (HTML) polynomial-form formula for the given fit degree,
    e.g. "x<sub>0</sub>(λ) = c<sub>0</sub> + c<sub>1</sub>λ" for
    degree 1, with real sub/superscripts for QLabel's built-in rich-text
    support (see LiveViewWidget's degree-selector group, which keeps this
    in sync with the currently selected degree). Uses "column" in place
    of "λ" when no WavelengthAxis exists yet, matching this module's
    existing pixel/wavelength fallback convention.

    Parameters
    ----------
    degree
        Polynomial degree (1, 2, or 3 -- see DEGREE_CHOICES).
    wavelength_axis
        Supplies wavelength_nm(pixel), or None for the pixel-column
        fallback.

    Returns
    -------
    str
        HTML fragment suitable for a QLabel with rich text enabled.
    '''

    variable = "λ" if wavelength_axis is not None else "column"
    # A single-letter variable (lambda) reads fine multiplied directly
    # against its coefficient ("c1λ"); a word-length fallback
    # ("column") needs a visible multiplication dot or it reads as one
    # run-together word ("c1column").
    separator = "" if len(variable) == 1 else "·"

    terms = ["c<sub>0</sub>"]
    for power in range(1, degree + 1):
        power_part = variable if power == 1 else f"{variable}<sup>{power}</sup>"
        terms.append(f"c<sub>{power}</sub>{separator}{power_part}")

    return f"x<sub>0</sub>({variable}) = " + " + ".join(terms)


def exposure_has_drifted(current_exposure_us: float, baseline_exposure_us: float) -> bool:

    '''
    True if current_exposure_us differs from baseline_exposure_us by more
    than EXPOSURE_MATCH_TOLERANCE_REL -- the exact relative-difference
    formula calibration/shared/metadata.py's check_settings_match() uses,
    reused here rather than approximated differently, so the GUI's drift
    warning and preprocessing's own hard settings-mismatch check agree on
    what counts as "the same" exposure.

    Parameters
    ----------
    current_exposure_us
        The value currently entered in the Acquisition Settings panel.
    baseline_exposure_us
        calibration_set.baseline_record.exposure_us -- what the loaded
        baseline was actually captured under.

    Returns
    -------
    bool
    '''

    exposure_diff_rel = abs(current_exposure_us - baseline_exposure_us) / baseline_exposure_us
    return exposure_diff_rel > EXPOSURE_MATCH_TOLERANCE_REL


def gain_has_drifted(current_gain_db: float, reference_gain_db: float) -> bool:

    '''
    True if current_gain_db differs from reference_gain_db by more than
    GAIN_MATCH_TOLERANCE_ABS -- the exact absolute-difference formula
    calibration/shared/metadata.py's check_settings_match() uses. Generic
    over which artifact's gain_db is being compared against (baseline's or
    conversion_gain_record's), since both are checked with the same
    tolerance and the same math.

    Parameters
    ----------
    current_gain_db
        The value currently entered in the Acquisition Settings panel.
    reference_gain_db
        The captured gain_db to compare against (baseline_record's or
        conversion_gain_record's).

    Returns
    -------
    bool
    '''

    gain_diff_abs = abs(current_gain_db - reference_gain_db)
    return gain_diff_abs > GAIN_MATCH_TOLERANCE_ABS


__all__ = [
    "LiveViewWidget",
    "wavelength_axis_label",
    "heatmap_x_extent",
    "evaluated_at_text",
    "fit_formula_html",
    "format_power_of_ten_superscript",
    "exposure_has_drifted",
    "gain_has_drifted",
    "DEGREE_CHOICES",
    "DEGREE_LABELS",
    "DEFAULT_DEGREE",
    "STRIP_CHART_WINDOW_SECONDS",
    "MICRONS_PER_MM",
    "EVALUATED_AT_COLUMN",
    "FIT_CURVE_COLOR",
    "FIT_CURVE_WIDTH",
]
