'''
Live-view interface: the main screen shown once the calibration screen
hands off loaded/created calibrations. Live-updating scatter plot of
centroid position vs. wavelength (or pixel column, if no wavelength
calibration exists yet), with an overlaid fit curve, a raw-image heatmap
underneath, a side info panel, and a rolling trend chart.

A QTimer (see __init__'s self._update_timer, interval
DEFAULT_UPDATE_INTERVAL_MS -- ~5Hz, the target refresh rate noted in
docs/project_state.md) drives the real per-tick computation:
self._on_timer_tick() pulls the newest frame off camera_stream (a no-op
skip if none has arrived yet), runs it through run_preprocessing() with
self._roi_control's current bounds, then analyze_shot() at the currently
selected degree, and pushes the result into the scatter/error-bar/
fit-curve/heatmap/strip-chart/side-panel widgets (_display_shot_result())
-- replacing the construction-time placeholder feed
(_populate_placeholder_data()) that still seeds the widget's very first
paint, before any real frame has arrived. The timer is started
unconditionally at the end of __init__ (not deferred to first show()):
this widget's whole existing __init__ already finishes every other piece
of setup synchronously (see _populate_placeholder_data()/
_recompute_settings_drift() below), a caller only ever constructs it once
its calibration_set/camera_stream are genuinely ready, and a tick before
the stream has produced a first frame is a harmless no-op -- so there's
no meaningful window where deferring to show() would help.

A frame that preprocesses fine but yields too few valid spectral columns
for the selected fit degree (InsufficientDataError) is skipped rather
than shown as an error -- see docs/project_state.md's "Skip-frame
handling". After MAX_CONSECUTIVE_SKIPS (~10) such skips in a row, the
display switches to an explicit "insufficient signal" state (fit overlay
hidden, diagnostics read "N/A") rather than silently freezing on the last
good frame; the raw heatmap still updates on a skip, since preprocessing
itself succeeded -- only the fit failed for lack of columns.

camera_stream.last_error becoming non-None mid-session (a fatal
CameraError -- e.g. the camera physically disconnects -- see
CameraStream's own docstring) is checked explicitly, every tick, before
ever calling get_latest_frame() -- that call would otherwise keep
returning the same stale FrameData forever (CameraStream never clears it
once its background thread has died), silently re-analyzing and
redrawing one frozen frame indefinitely with nothing but a small
status-label text change to indicate it. last_error, not is_running alone
-- is_running is also False for the harmless, common "never started yet"
case, which must keep falling through to the ordinary no-frame-yet
handling unchanged. Detected, this instead pops a message box and gives
the fit overlay the same hidden/"N/A" treatment as the insufficient-
signal state (see _enter_camera_disconnected_state()) -- takes priority
over the settings-drift check below, since a dead camera
makes drift a moot question.

The Acquisition Settings side-panel section (exposure_us/gain_db spin
boxes, pre-filled from the loaded baseline's capture settings) still does
NOT reconfigure camera_stream -- editing either field only re-evaluates a
combined drift state against calibration_set.baseline_record (and
conversion_gain_record, if supplied) via _recompute_settings_drift().
Crossing from "in tolerance" to "drifted" hides the fit diagnostics
(reading "N/A") and the scatter/error-bar/fit-curve overlay (the raw
heatmap keeps displaying underneath) and pops a single informational
message; returning to "in tolerance" restores both. While drifted,
_on_timer_tick() returns immediately without running any real
preprocessing/analysis -- the loaded calibrations are untrusted against
the entered settings, so there's nothing valid to compute -- rather than
fighting the drift UI's own hide/restore logic.

Degree selection (the combo box) does NOT trigger a synchronous refit:
_on_degree_changed() only updates self._current_degree, and the next
timer tick (which reads it fresh) supplies the real fit/curve/diagnostics
for the new degree -- at most one tick interval later, close enough given
the ~5Hz cadence that a dedicated refit path isn't worth the complexity.
Before any real tick has ever landed, it also swaps in this widget's own
placeholder numbers for the newly picked degree immediately, via
_update_fit_panel() -- but only then; see that method's own docstring
(and _on_roi_changed()'s/_exit_drifted_state()'s, which guard the same
way) for why it must never repaint from placeholder data once real data
already exists on screen.

Real per-tick fits close the "degree > 1 has no internal zeta
uncertainty" gap noted in docs/project_state.md: SpatialDispersionFitResult
.sigma_zeta() (exact coefficient-covariance propagation) is called for
every degree, not just degree 1, so the "uncertainty not available" caveat
only ever appears on the construction-time placeholder numbers now, never
on a real result.
'''

# Imports

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
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
from pipeline.analysis import (
    analyze_shot,
    InsufficientDataError,
    SensorNoiseModel,
    ShotAnalysisResult,
    SpatialDispersionFitResult,
)
from pipeline.analysis.interfaces import WavelengthAxis
from pipeline.calibration.sensor import ConversionGainRecord
from pipeline.calibration.shared import EXPOSURE_MATCH_TOLERANCE_REL, GAIN_MATCH_TOLERANCE_ABS
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.preprocessing import (
    CalibrationSet,
    NoSignalError,
    ProcessedFrame,
    run_preprocessing,
    SettingsMismatchError,
)

from pipeline.gui.formatting import (
    coefficient_unit,
    format_value_with_uncertainty,
    MICRONS_PER_MM,
    microns_to_mm,
    microns_to_nm,
)
from pipeline.gui.roi_control import SpatialROIControl, SpectralROIControl
from pipeline.gui.theme import (
    COLOR_ACCENT as ACCENT_COLOR,
    COLOR_BACKGROUND as BACKGROUND_COLOR,
    COLOR_ERROR as ERROR_COLOR,
    COLOR_TEXT_PRIMARY as FOREGROUND_COLOR,
    COLOR_PLOT_GRID as GRID_COLOR,
    combo_box_stylesheet,
    group_box_stylesheet,
    load_bundled_font,
)


# Constants

logger = logging.getLogger(__name__)

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

# QTimer tick interval driving the real update loop -- ~5Hz, the target
# refresh rate from docs/project_state.md's live-view design notes
# (analyze_shot() profiles at ~11.7ms/call, well under this budget; the
# plotting redraw, not the science pipeline, is the real constraint).
# Overridable per-instance via LiveViewWidget's update_interval_ms
# constructor parameter, e.g. to shrink it for fast, deterministic tests.
DEFAULT_UPDATE_INTERVAL_MS = 200

# Number of consecutive InsufficientDataError skips (too few valid
# spectral columns to fit the selected degree) before the display gives
# up on the last good frame and switches to an explicit "insufficient
# signal" state instead. An unverified starting constant, same treatment
# as SNR_THRESHOLD/SIGMA_THRESHOLD elsewhere -- see docs/project_state.md's
# "Skip-frame handling".
MAX_CONSECUTIVE_SKIPS = 10

# Placeholder sigma (in pixel-column units) fed to TotalLeastSquaresFit's
# scipy.odr backend by _PixelColumnWavelengthAxis when no real wavelength
# calibration is loaded -- scipy.odr requires a strictly positive input
# standard deviation on both axes, but a bare pixel-column index has no
# real uncertainty of its own (see the WavelengthAxis fallback convention
# used throughout this module); never displayed anywhere, only fed to the
# fitter to keep it numerically valid.
FALLBACK_PIXEL_COLUMN_SIGMA = 0.5

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

# Centroid scatter color -- deliberately neither FOREGROUND_COLOR (the
# error bars' color, so a semi-transparent near-white marker used to
# blend into the near-white error-bar lines through it) nor
# FIT_CURVE_COLOR (black would blend the markers into the fit curve
# drawn on top of them instead). A warm orange reads clearly against
# viridis's cool blue/green/purple range the same way FIT_CURVE_COLOR's
# black does, without competing with either neighboring layer.
CENTROID_SCATTER_COLOR = "#ff9d5c"

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
    (nm/nm, a composite unit -- see _zeta_to_nm()), so pyqtgraph's default
    unit-prefix annotation is the only place this scale factor is ever
    shown.
    '''

    def labelString(self) -> str:

        if self.labelUnits != "" or not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
            return super().labelString()

        units = f"({format_power_of_ten_superscript(1.0 / self.autoSIPrefixScale)})"
        style = ";".join(f"{k}: {self.labelStyle[k]}" for k in self.labelStyle)
        s = f"{self.labelText} {units}"
        return f"<span style='{style}'>{s}</span>"


class _PixelColumnWavelengthAxis:

    '''
    Trivial WavelengthAxis stand-in used internally by the real timer
    loop's analyze_shot() calls when no real spectral calibration has
    been loaded (self._wavelength_axis is None) -- analyze_shot()
    requires a WavelengthAxis to fit against, but every on-screen element
    already falls back to a plain pixel-column axis in that case (see
    wavelength_axis_label()/heatmap_x_extent() above), so this makes the
    fit run in that same fallback domain: wavelength_nm() is the identity
    map on pixel-column index. See FALLBACK_PIXEL_COLUMN_SIGMA for why
    sigma_wavelength_nm() is a small positive constant rather than 0.
    '''

    def wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:
        return np.asarray(pixel, dtype=float)

    def sigma_wavelength_nm(self, pixel: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(pixel, dtype=float), FALLBACK_PIXEL_COLUMN_SIGMA)


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
        not this widget's responsibility). Polled every self._update_timer
        tick via get_latest_frame() -- see module docstring.
    conversion_gain_record
        ConversionGainRecord (gain_db + timing/sweep metadata, no
        exposure_us -- conversion gain sweeps exposure by design) tagging
        the loaded conversion-gain artifact, or None if no conversion-gain
        artifact was loaded. When supplied, the Acquisition Settings
        panel's gain_db field is drift-checked against it in addition to
        calibration_set.baseline_record.gain_db, since the two artifacts
        can drift independently of each other.
    update_interval_ms
        Real update loop's QTimer interval, in milliseconds -- defaults to
        DEFAULT_UPDATE_INTERVAL_MS (~5Hz). Exposed as a constructor
        parameter (rather than only the class-level default) so tests can
        shrink it for fast, deterministic execution without monkeypatching.
    parent
        Standard Qt parent widget.
    '''

    recalibration_requested = Signal(str)
    extended_measurement_requested = Signal()
    back_to_calibration_requested = Signal()

    def __init__(
        self,
        calibration_set: CalibrationSet,
        noise_model: SensorNoiseModel,
        position_calibration: ScaleFactorPositionCalibration,
        wavelength_axis: WavelengthAxis | None,
        camera_stream: CameraStream,
        conversion_gain_record: ConversionGainRecord | None = None,
        update_interval_ms: int = DEFAULT_UPDATE_INTERVAL_MS,
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

        # Real-loop state -- see module docstring and _on_timer_tick().
        self._consecutive_skips = 0
        self._insufficient_signal = False
        self._camera_disconnected = False
        self._strip_chart_history: list[tuple[float, float]] = []
        self._pixel_column_wavelength_axis = _PixelColumnWavelengthAxis()

        # Flips permanently False->True the first time _display_shot_result()
        # draws a real analyze_shot() result -- see _on_roi_changed()'s own
        # docstring for why this matters.
        self._displayed_real_data = False

        self._apply_pyqtgraph_theme()
        self._build_ui()
        self._populate_placeholder_data()

        # Self-consistent even if a caller constructs this widget directly
        # with an already-mismatched baseline/conversion-gain pair (e.g.
        # bypassing calibration_screen.py's own gate) -- starts in whatever
        # state the supplied records actually imply, not assumed "OK".
        self._settings_drifted = False
        self._recompute_settings_drift()

        self._update_status_label()

        # See module docstring for why this starts unconditionally here
        # rather than being deferred to first show().
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(update_interval_ms)
        self._update_timer.timeout.connect(self._on_timer_tick)
        self._update_timer.start()

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

        # Real text/color are set by _update_status_label() (called once
        # at the end of __init__, then every timer tick) -- this initial
        # text is only ever visible for the instant between widget
        # construction and that first call.
        self._status_label = QLabel("Status: OK")
        self._status_label.setFont(load_bundled_font(10))
        self._status_label.setStyleSheet(f"color: {ACCENT_COLOR};")
        return self._status_label

    def _build_main_plot(self) -> pg.PlotWidget:

        plot_widget = pg.PlotWidget()
        self._main_plot = plot_widget.getPlotItem()
        self._main_plot.setLabel("left", "Relative Physical Position (mm)")
        self._main_plot.setLabel("bottom", self._x_axis_label())
        self._main_plot.showGrid(x=True, y=True, alpha=0.3)
        self._style_plot_axes(self._main_plot)

        # Heatmap first (so everything else renders on top of it), then
        # error bars, then the scatter -- pyqtgraph paints items in add
        # order, and the error bar's vertical line runs straight through
        # each point's centre, so adding it before the scatter (rather
        # than after) keeps it from painting over and hiding the point.
        # Fit curve LAST -- it traces nearly the same path as the scatter
        # points, so drawing it underneath them (the original order) left
        # it almost entirely hidden behind the denser, larger scatter
        # markers. setZValue() below pins this same stacking order
        # explicitly rather than relying on add order alone.
        self._image_item = pg.ImageItem()
        self._image_item.setColorMap(pg.colormap.get("viridis"))
        self._main_plot.addItem(self._image_item)

        self._error_bars = pg.ErrorBarItem(pen=pg.mkPen(color=FOREGROUND_COLOR, width=1))
        self._error_bars.setZValue(0)
        self._main_plot.addItem(self._error_bars)

        # Fully opaque CENTROID_SCATTER_COLOR, not a near-white brush --
        # a semi-transparent white point over the (also near-white)
        # FOREGROUND_COLOR error-bar lines blended into the same shade,
        # making the two indistinguishable regardless of paint order.
        self._scatter = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(None), brush=pg.mkBrush(CENTROID_SCATTER_COLOR)
        )
        self._scatter.setZValue(10)
        self._main_plot.addItem(self._scatter)

        self._fit_curve = pg.PlotDataItem(
            pen=pg.mkPen(color=FIT_CURVE_COLOR, width=FIT_CURVE_WIDTH)
        )
        self._fit_curve.setZValue(20)
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
        layout.addWidget(self._build_spectral_roi_group())
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
            "workflow)."
        )
        self._extended_measurement_button.clicked.connect(
            self.extended_measurement_requested
        )
        layout.addWidget(self._extended_measurement_button)

        self._back_to_calibration_button = QPushButton("Back to Calibration")
        self._back_to_calibration_button.setFont(load_bundled_font(12, bold=True))
        self._back_to_calibration_button.setMinimumHeight(48)
        self._back_to_calibration_button.setToolTip(
            "Returns to the calibration screen. Stops this screen's camera "
            "stream first, freeing it for a new baseline/flat-field/"
            "conversion-gain/spectral capture."
        )
        self._back_to_calibration_button.clicked.connect(
            self.back_to_calibration_requested
        )
        layout.addWidget(self._back_to_calibration_button)

        return panel

    def _build_acquisition_settings_group(self) -> QGroupBox:

        '''
        Exposure/gain display+entry, pre-filled from
        calibration_set.baseline_record. Editing either field never
        touches camera_stream -- unlike ExtendedMeasurementScreen's own
        Acquisition Settings panel, which does reconfigure the camera on
        Run, live view's continuous polling loop has no natural point to
        pause for a mid-stream reconfiguration, so this deliberately stays
        read-only-to-the-camera and only drives the drift check below,
        warning when the entered value diverges from what the loaded
        calibrations were actually captured under.
        '''

        group = QGroupBox("Acquisition Settings")
        group.setFont(load_bundled_font(10))
        group.setStyleSheet(group_box_stylesheet())
        group.setToolTip(
            "Does not reconfigure the camera -- entering a different "
            "exposure/gain here only checks it against the loaded "
            "calibrations. Pre-filled from the loaded baseline's capture "
            "settings; drifting past tolerance from the loaded "
            "calibrations hides the fit diagnostics and overlay (reading "
            "\"N/A\", raw heatmap still shown) and shows an informational "
            "recalibration message."
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

        self._spectral_roi_control = SpectralROIControl(
            CANONICAL_SHAPE[SPECTRAL_AXIS], self._wavelength_axis
        )
        self._spectral_roi_control.roi_changed.connect(self._on_spectral_roi_changed)
        return self._spectral_roi_control

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
        restores the scatter/error-bar/fit-curve overlay's visibility.
        Before any real tick has landed (self._displayed_real_data still
        False), also recomputes the fit-diagnostics panel from
        _placeholder_fits via _update_fit_panel(), undoing
        _enter_drifted_state()'s "N/A" placeholders with this widget's own
        pre-baked numbers -- there's nothing real yet to show instead.
        Once real data has been displayed at least once, _update_fit_panel()
        must NOT be called here -- same reasoning as _on_roi_changed()'s
        and _on_degree_changed()'s own docstrings: it would replace
        whatever real result was on screen before the drift episode with
        stale placeholder numbers (including degree > 1's placeholder-only
        "uncertainty not available" note, which isn't true of real data)
        for however long until the next real tick lands. _on_timer_tick()
        only skips real work while self._settings_drifted is True, so the
        very next tick after this method runs will call
        _update_fit_panel_from_result() with genuinely current numbers
        anyway -- there's no gap to fill here once real data exists.
        '''

        self._scatter.setVisible(True)
        self._error_bars.setVisible(True)
        self._fit_curve.setVisible(True)
        if not self._displayed_real_data:
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
        form.addRow("Spatial Dispersion (nm/nm):", self._zeta_label)
        form.addRow("", self._zeta_note_label)
        form.addRow("Evaluated At:", self._evaluated_at_label)

        return group

    # -- placeholder data (Phase 1 only -- no real camera/analysis calls) --

    def _populate_placeholder_data(self) -> None:

        self._placeholder_fits = self._build_placeholder_fits()
        self._generate_placeholder_data()

        self._update_strip_chart_placeholder(self._placeholder_rng)
        self._update_fit_panel(DEFAULT_DEGREE)

        self._apply_roi_bounds()

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
        x_values = self._x_values_for_columns(columns)
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
        # Data-independent (only the frame shape and wavelength axis feed
        # into it) -- computed once here and reused by both
        # _apply_roi_bounds() (placeholder rendering) and
        # _display_shot_result() (real rendering) below.
        self._main_plot_x_extent = (min(x0, x1), max(x0, x1))

        # Kept for _update_strip_chart_placeholder() to continue the same
        # random stream _populate_placeholder_data() threads it through,
        # rather than resetting to a fresh seed=0 generator and changing
        # the strip chart's random values.
        self._placeholder_rng = rng

    def _current_x_extent(self) -> tuple[float, float]:

        '''
        The main plot's x-extent for the CURRENTLY selected spectral ROI
        window (self._spectral_roi_control.column_window(), which is
        always the full frame at the default), as opposed to
        self._main_plot_x_extent, the fixed full-frame extent used only
        for self._image_item.setRect() (the heatmap's data-space mapping,
        which must stay full regardless of the viewport's current zoom --
        see _generate_placeholder_data()). Every setRange() call in this
        widget uses this method instead of the fixed extent, so narrowing
        self._spectral_roi_control actually zooms the plot into the
        selected window, mirroring self._roi_control's y-axis zoom.
        '''

        column_min, column_max = self._spectral_roi_control.column_window()
        x0, x1 = self._heatmap_x_extent(first_column=column_min, last_column=column_max - 1)
        return (min(x0, x1), max(x0, x1))

    def _apply_roi_bounds(self) -> None:

        '''
        Crops the fake scatter/error-bars/fit-curve/heatmap to
        self._roi_control's and self._spectral_roi_control's current
        bounds and renders them -- the real system's analogue of
        preprocessing/steps/roi.py's apply_roi() zeroing rows outside the
        spatial ROI, and preprocessing/steps/spectral_roi.py's
        apply_spectral_roi() overriding which columns are valid, combined:
        an out-of-window scatter/fit point on EITHER axis is dropped
        entirely rather than merely clipped from view. Callable both at
        startup (_populate_placeholder_data()) and, ONLY before the first
        real tick has landed, from either ROI control's roi_changed signal
        (_on_roi_changed()/_on_spectral_roi_changed()) -- see those
        methods' own docstrings for why they must stop calling this once
        self._displayed_real_data is True (this always repaints from
        self._placeholder_*, which would overwrite real on-screen data
        with stale fake data on every ROI edit otherwise).

        The heatmap is masked by the spatial ROI only, not the spectral
        one: apply_spectral_roi() never zeroes pixels (see its own
        docstring -- it only overrides which columns count for analysis,
        the same contract as the automatic SNR gate it replaces), so
        neither does this placeholder counterpart, for consistency with
        what the real per-tick heatmap will actually show once real data
        arrives.
        '''

        min_mm, max_mm = self._roi_control.roi_bounds_mm()
        column_min, column_max = self._spectral_roi_control.column_window()

        y_values, y_sigma = self._convert_to_mm(
            self._placeholder_centroid_px,
            np.full_like(self._placeholder_centroid_px, 3.0),
        )
        keep = (
            (y_values >= min_mm) & (y_values <= max_mm)
            & (self._placeholder_columns >= column_min) & (self._placeholder_columns < column_max)
        )

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
        keep_fit = (
            (fit_y_full >= min_mm) & (fit_y_full <= max_mm)
            & (self._placeholder_fit_columns >= column_min) & (self._placeholder_fit_columns < column_max)
        )
        self._fit_curve.setData(
            x=self._placeholder_fit_x[keep_fit], y=fit_y_full[keep_fit]
        )

        row_min_px, row_max_px = self._roi_control.roi_bounds_px()
        masked_image = self._placeholder_image_full.copy()
        masked_image[:row_min_px, :] = 0
        masked_image[row_max_px:, :] = 0
        self._image_item.setImage(masked_image)

        self._main_plot.setRange(
            xRange=self._current_x_extent(), yRange=(min_mm, max_mm), padding=0
        )

    def _on_roi_changed(self, min_mm: float, max_mm: float) -> None:

        '''
        Handler for self._roi_control.roi_changed. Before any real tick
        has ever landed (self._displayed_real_data still False -- e.g.
        the camera hasn't produced a frame yet), re-renders the
        placeholder data cropped to the current bounds via
        _apply_roi_bounds(), exactly as before. Once real data has been
        displayed at least once, _apply_roi_bounds() must NOT be called
        here: it unconditionally repaints the scatter/error-bars/
        fit-curve/heatmap from self._placeholder_* -- stale, fake data
        generated once at construction -- which would overwrite whatever
        real result is currently on screen with it, every single time the
        ROI control changes, until the next timer tick redraws real data
        over it again. Only the plot's y-range needs to change here in
        that case; the heatmap/scatter/fit-curve themselves will reflect
        the new ROI on the next real tick, once run_preprocessing() has
        actually re-masked a frame against it.
        '''

        if self._displayed_real_data:
            self._main_plot.setRange(
                xRange=self._current_x_extent(), yRange=(min_mm, max_mm), padding=0
            )
        else:
            self._apply_roi_bounds()

    def _on_spectral_roi_changed(self, column_min: int, column_max: int) -> None:

        '''
        Handler for self._spectral_roi_control.roi_changed -- the spectral
        counterpart to _on_roi_changed() above, same before/after-real-data
        split and same reasoning (see that method's docstring). The only
        difference is which axis of the plot's view range reacts: here
        it's x (via _current_x_extent(), which reads the just-changed
        bounds), there it's y.
        '''

        if self._displayed_real_data:
            self._main_plot.setRange(
                xRange=self._current_x_extent(), yRange=self._roi_control.roi_bounds_mm(), padding=0
            )
        else:
            self._apply_roi_bounds()

    def _update_strip_chart_placeholder(self, rng: np.random.Generator) -> None:

        n_points = 60
        t = np.linspace(-STRIP_CHART_WINDOW_SECONDS, 0, n_points)
        zeta_center = self._placeholder_fits[DEFAULT_DEGREE].zeta_value
        trend = zeta_center + 0.05 * zeta_center * np.sin(t / 12.0)
        noisy = trend + rng.normal(scale=0.02 * abs(zeta_center) + 1e-6, size=t.shape)
        self._strip_curve.setData(x=t, y=noisy)

    # -- live update loop (QTimer-driven -- see module docstring) --------

    def _update_status_label(self) -> None:

        '''
        Reflects self._camera_stream's real is_running/last_error state.
        Called once at the end of __init__ (so the very first paint
        already shows reality, not the construction-time default text)
        and again on every timer tick thereafter. last_error takes
        priority over is_running since a stream that died from a fatal
        CameraError is also, necessarily, no longer running -- reporting
        the error is more useful than just "stopped".
        '''

        last_error = self._camera_stream.last_error
        if last_error is not None:
            self._status_label.setText(f"Status: Camera error -- {last_error}")
            self._status_label.setStyleSheet(f"color: {ERROR_COLOR};")
        elif not self._camera_stream.is_running:
            self._status_label.setText("Status: Camera stopped")
            self._status_label.setStyleSheet(f"color: {ERROR_COLOR};")
        else:
            self._status_label.setText("Status: OK")
            self._status_label.setStyleSheet(f"color: {ACCENT_COLOR};")

    def _on_timer_tick(self) -> None:

        '''
        self._update_timer's timeout slot -- see module docstring for the
        full per-tick flow and the drift/skip-counter interactions.
        Deliberately swallows every expected failure mode itself (a dead
        camera stream, missing frame, bad-frame/settings-mismatch
        preprocessing errors, insufficient-column fit errors) rather than
        letting any of them escape as an uncaught exception out of a Qt
        slot -- plus a broad catch-all around _display_shot_result()
        specifically, as a last-resort safety net for whatever *isn't*
        one of those expected modes (see that call site's own comment for
        why).
        '''

        self._update_status_label()

        if self._camera_stream.last_error is not None:
            # last_error, not is_running alone -- is_running is ALSO False
            # for a stream that was simply never started (a normal,
            # common state, e.g. right at construction, or in several of
            # this widget's own tests), which must keep falling through
            # to the ordinary "no frame yet" handling below, unchanged.
            # last_error is the same signal _update_status_label() already
            # uses to distinguish a genuine fatal CameraError (background
            # thread died -- see CameraStream's own docstring) from that
            # harmless case, and from a clean stop() (also is_running=
            # False, but last_error stays None). Without this distinction,
            # get_latest_frame() would keep returning the same stale
            # FrameData forever after a real fatal error (CameraStream
            # never clears it on thread death), so it's checked BEFORE
            # ever calling get_latest_frame(), not discovered by its
            # absence. Takes priority over the settings-drift check below:
            # a dead camera is the more fundamental problem, and "are the
            # entered settings still trustworthy" doesn't matter if
            # there's no live stream to apply them to.
            if not self._camera_disconnected:
                self._enter_camera_disconnected_state()
            return
        if self._camera_disconnected:
            self._exit_camera_disconnected_state()

        if self._settings_drifted:
            # Untrusted calibrations against the entered settings -- see
            # module docstring for why this skips real work entirely
            # rather than fighting _enter_drifted_state()/_exit_drifted_
            # state()'s own hide/restore of the same overlay.
            return

        frame = self._camera_stream.get_latest_frame()
        if frame is None:
            # Nothing grabbed yet (stream not started, or hasn't produced
            # its first frame) -- leave whatever's on screen alone rather
            # than treating "no new data this tick" as an error.
            return

        try:
            processed, _saturation_result = run_preprocessing(
                frame, self._calibration_set,
                roi_bounds=self._roi_control.roi_bounds_px(),
                column_bounds=self._spectral_roi_control.column_bounds(),
            )
        except (NoSignalError, SettingsMismatchError):
            # A genuinely signal-free raw frame, or the frame's actual
            # exposure_us/gain_db no longer matching calibration_set.
            # baseline_record -- distinct failure modes from "not enough
            # valid columns to fit" below (and not currently surfaced
            # anywhere else per-tick), so just skip this frame rather than
            # let either propagate out of a Qt slot.
            return

        axis = (
            self._wavelength_axis
            if self._wavelength_axis is not None
            else self._pixel_column_wavelength_axis
        )

        try:
            result = analyze_shot(
                processed, axis, noise_model=self._noise_model,
                degrees=(self._current_degree,),
            )
        except InsufficientDataError:
            # Preprocessing succeeded (so the raw view is still genuinely
            # current) but too few columns cleared the signal threshold to
            # fit the selected degree -- update the heatmap alone, count
            # the skip, and only give up on the overlay after
            # MAX_CONSECUTIVE_SKIPS in a row (see module docstring).
            self._image_item.setImage(processed.image)
            self._consecutive_skips += 1
            if self._consecutive_skips >= MAX_CONSECUTIVE_SKIPS and not self._insufficient_signal:
                self._enter_insufficient_signal_state()
            return

        try:
            self._display_shot_result(processed, result)
        except Exception:
            # Last-resort safety net, not the primary handling for any
            # *expected* failure mode (those are the specific except
            # clauses above) -- this method's own docstring promises every
            # expected failure is swallowed here rather than escaping a
            # Qt slot, and a broken display update for one tick is exactly
            # as recoverable as a bad frame or an under-subscribed fit, so
            # it gets the same skip/counter treatment. Logged (not
            # silently dropped) so a genuine bug is still visible to
            # whoever's running this session, even though the GUI itself
            # keeps running. Concretely: this is what would have caught
            # the exact-degree+1-columns crash before analysis/
            # dispersion_fitting.py's InsufficientDataError threshold was
            # corrected to exclude that degenerate case -- kept as
            # defense-in-depth against whatever the next unforeseen edge
            # case turns out to be, not a substitute for fixing this one
            # at its source.
            logger.exception("live view: error displaying frame %d's analysis result", processed.frame_id)
            self._consecutive_skips += 1
            if self._consecutive_skips >= MAX_CONSECUTIVE_SKIPS and not self._insufficient_signal:
                self._enter_insufficient_signal_state()
            return

        self._consecutive_skips = 0
        if self._insufficient_signal:
            self._exit_insufficient_signal_state()

    def _display_shot_result(self, processed: ProcessedFrame, result: ShotAnalysisResult) -> None:

        '''
        Pushes one successful analyze_shot() result into the scatter/
        error-bar/fit-curve/heatmap/strip-chart/side-panel widgets --
        the real-data counterpart to _apply_roi_bounds()'s placeholder
        rendering. fit.coefficients stay in the raw pixel/wavelength
        units analyze_shot() itself reports (position_calibration is
        deliberately not passed to analyze_shot() -- see
        calibration/spatial/'s scale-factor-only design); the
        scatter/error-bar/fit-curve y values (physical position) go
        through self._convert_to_mm(), to match the main plot's "Relative
        Physical Position (mm)" y-axis -- that graph axis stays in mm
        regardless of the nm-based convention below. zeta specifically
        (the one coefficient given its own side-panel/strip-chart
        display, both quoted values rather than graph axes) goes through
        self._zeta_to_nm() at its own two call sites below and in
        _update_fit_panel_from_result() -- valid because
        ScaleFactorPositionCalibration.convert() is a pure linear scale,
        so converting a slope this way is exactly as correct as
        converting a position.
        '''

        self._displayed_real_data = True

        fit = result.fits[self._current_degree]
        columns = result.centroids.columns
        x_values = self._x_values_for_columns(columns)

        y_values, y_sigma = self._convert_to_mm(result.centroids.x0, result.centroids.sigma_x0)
        self._scatter.setData(x=x_values, y=y_values)

        error_kwargs = dict(x=x_values, y=y_values, top=y_sigma, bottom=y_sigma)
        if self._wavelength_axis is not None:
            x_sigma = self._wavelength_axis.sigma_wavelength_nm(columns)
            error_kwargs["left"] = x_sigma
            error_kwargs["right"] = x_sigma
        self._error_bars.setData(**error_kwargs)

        fit_x = np.linspace(x_values.min(), x_values.max(), 200)
        fit_y_px = np.polynomial.polynomial.polyval(fit_x, fit.coefficients)
        fit_y_mm, _ = self._convert_to_mm(fit_y_px, np.zeros_like(fit_y_px))
        self._fit_curve.setData(x=fit_x, y=fit_y_mm)

        self._image_item.setImage(processed.image)
        self._main_plot.setRange(
            xRange=self._current_x_extent(), yRange=self._roi_control.roi_bounds_mm(), padding=0
        )

        # "Central wavelength of the currently-valid columns" -- see
        # docs/project_state.md's degree > 1 spec.
        eval_x = float(np.median(x_values))
        self._update_fit_panel_from_result(fit, eval_x)
        zeta_nm, _ = self._zeta_to_nm(float(fit.zeta(np.array([eval_x]))[0]))
        self._append_strip_chart_point(zeta_nm)

    def _update_fit_panel_from_result(
        self, fit: SpatialDispersionFitResult, eval_x: float
    ) -> None:

        '''
        Real-data counterpart to _update_fit_panel() (which reads
        self._placeholder_fits): the same label layout, but sourced from
        a real SpatialDispersionFitResult and evaluated at eval_x via
        fit.zeta()/fit.sigma_zeta() -- the latter uses the fit's full
        coefficient covariance (see SpatialDispersionFitResult.sigma_zeta's
        docstring), so degree > 1 gets a real internal uncertainty here,
        closing the gap _update_fit_panel()'s placeholder path still
        deliberately leaves open (see module docstring).
        '''

        self._chi_squared_label.setText(f"{fit.reduced_chi_squared:.3f}")

        coefficients_nm, coefficient_sigma_nm = self._convert_to_nm(
            np.asarray(fit.coefficients, dtype=np.float64), np.asarray(fit.coefficient_sigma, dtype=np.float64)
        )
        coeff_lines = [
            f"c<sub>{i}</sub> ({coefficient_unit(i)}) = {format_value_with_uncertainty(float(c), float(s))}"
            for i, (c, s) in enumerate(zip(coefficients_nm, coefficient_sigma_nm))
        ]
        self._coefficients_label.setText("<br>".join(coeff_lines))

        self._formula_label.setText(fit_formula_html(fit.degree, self._wavelength_axis))

        eval_point = np.array([eval_x])
        zeta_value_px = float(fit.zeta(eval_point)[0])
        zeta_sigma_px = float(fit.sigma_zeta(eval_point)[0])
        zeta_value, zeta_sigma = self._zeta_to_nm(zeta_value_px, zeta_sigma_px)
        self._zeta_label.setText(format_value_with_uncertainty(zeta_value, zeta_sigma))
        self._zeta_note_label.setText("")

        self._evaluated_at_label.setText(
            evaluated_at_text(eval_x, self._wavelength_axis) if fit.degree > 1 else ""
        )

    def _append_strip_chart_point(self, zeta_value: float) -> None:

        '''
        Real-data counterpart to _update_strip_chart_placeholder(): appends
        one (now, zeta_value) sample to a rolling history trimmed to the
        last STRIP_CHART_WINDOW_SECONDS, then redraws the strip curve
        against "seconds ago" (0 = this tick, negative = further in the
        past), matching the placeholder version's x convention.
        '''

        now = time.monotonic()
        self._strip_chart_history.append((now, zeta_value))
        cutoff = now - STRIP_CHART_WINDOW_SECONDS
        self._strip_chart_history = [
            (t, z) for t, z in self._strip_chart_history if t >= cutoff
        ]

        t = np.array([sample_t - now for sample_t, _ in self._strip_chart_history])
        z = np.array([sample_z for _, sample_z in self._strip_chart_history])
        self._strip_curve.setData(x=t, y=z)

    def _enter_insufficient_signal_state(self) -> None:

        '''
        Entered once self._consecutive_skips reaches MAX_CONSECUTIVE_SKIPS
        InsufficientDataError skips in a row: hides the scatter/error-bar/
        fit-curve overlay and replaces the side-panel diagnostics with an
        explicit "N/A", the same visual treatment _enter_drifted_state()
        gives a settings-drift episode -- but without popping a message
        box or emitting recalibration_requested, since low signal isn't
        something a recalibration would fix. The raw heatmap is left
        alone; it's still being updated every skip (see _on_timer_tick()).
        '''

        self._insufficient_signal = True
        self._chi_squared_label.setText("N/A")
        self._coefficients_label.setText("N/A")
        self._zeta_label.setText("N/A")
        self._zeta_note_label.setText("")
        self._evaluated_at_label.setText("")
        self._scatter.setVisible(False)
        self._error_bars.setVisible(False)
        self._fit_curve.setVisible(False)

    def _exit_insufficient_signal_state(self) -> None:

        '''
        Entered on the first tick to succeed after an insufficient-signal
        episode -- restores overlay visibility. Diagnostics/scatter/fit
        data itself is populated right after by the same tick's
        _display_shot_result() call, not here.
        '''

        self._insufficient_signal = False
        self._scatter.setVisible(True)
        self._error_bars.setVisible(True)
        self._fit_curve.setVisible(True)

    def _enter_camera_disconnected_state(self) -> None:

        '''
        Entered the first tick after self._camera_stream.last_error
        becomes non-None -- the background acquisition thread has exited
        from a fatal CameraError (see CameraStream's own docstring: "any
        other CameraError is fatal immediately"). Deliberately keyed on
        last_error, not is_running alone: is_running is also False for
        the ordinary "never started yet" case, which must NOT trigger
        this (see _on_timer_tick()'s own comment -- an earlier version of
        this check used is_running alone and wrongly fired on every
        widget built around a not-yet-started stream, in tests and
        potentially in real use before the first frame arrives). Without
        this, get_latest_frame() would keep returning the same stale
        FrameData forever (CameraStream never clears it on thread death),
        and this tick loop would just keep re-analyzing and redrawing
        that one frame indefinitely -- indistinguishable from a genuinely
        live but momentarily static feed, except for the small status-
        label text _update_status_label() already sets, easy to miss.

        Pops a message box (like _enter_drifted_state()'s) so the
        disconnect is impossible to miss, and hides the fit overlay +
        "N/A"s the diagnostics -- the same visual treatment
        _enter_insufficient_signal_state() gives low signal. The raw
        heatmap is deliberately left showing its last frame, as a
        reference of what the beam looked like right before the
        disconnect, not blanked -- same reasoning as the insufficient-
        signal state leaving it alone.

        Nothing in this codebase currently restarts a dead CameraStream
        from inside LiveViewWidget itself -- the real recovery path is
        the "Back to Calibration" button, which tears this whole widget
        down and MainWindow builds a fresh one around a newly-started
        stream (see app.py's _on_back_to_calibration_requested()) -- so
        this message names that path explicitly rather than implying a
        fix is possible from here.
        '''

        self._camera_disconnected = True
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
            "Camera Disconnected",
            f"The camera stream has stopped responding: {self._camera_stream.last_error}\n\n"
            f"Reconnect the camera, then use \"Back to Calibration\" to rebuild the connection.",
        )

    def _exit_camera_disconnected_state(self) -> None:

        '''
        Defensive symmetry with _enter_camera_disconnected_state(),
        mirroring _exit_insufficient_signal_state()'s same role. Nothing
        currently makes self._camera_stream.last_error go back to None on
        an already-disconnected stream within one LiveViewWidget's
        lifetime (a fresh start() would clear it, but nothing calls
        start() again from inside this widget -- see that method's own
        docstring for the real recovery path), so this realistically
        never runs today -- kept so this doesn't become a silent trap if
        that ever changes.
        '''

        self._camera_disconnected = False
        self._scatter.setVisible(True)
        self._error_bars.setVisible(True)
        self._fit_curve.setVisible(True)

    # -- degree selector (stub -- see module docstring) -----------------

    def _on_degree_changed(self, index: int) -> None:

        '''
        Before any real tick has landed (self._displayed_real_data still
        False), swaps in this widget's own pre-baked placeholder numbers
        for the newly-selected degree via _update_fit_panel() -- there's
        nothing real on screen yet to preserve. Once real data has been
        displayed at least once, this must NOT call _update_fit_panel():
        that method always shows _placeholder_fits[degree]'s stale,
        construction-time-only content, including a "Uncertainty not
        available for degree > 1" note that real data never actually has
        (see _update_fit_panel_from_result(), which always clears that
        note -- real sigma_zeta() is available at every degree). Calling
        it here would replace genuine on-screen results with that stale
        placeholder/note for however long until the next real tick lands
        -- exactly the same class of bug _on_roi_changed() had (see that
        method's own docstring). Once real data exists, self._current_degree
        alone is enough: the next tick reads it fresh and calls
        _update_fit_panel_from_result() with the real fit for the new
        degree, curve included.
        '''

        degree = self._degree_selector.itemData(index)
        if degree is None:
            return
        self._current_degree = degree
        if not self._displayed_real_data:
            self._update_fit_panel(degree)

    def _update_fit_panel(self, degree: int) -> None:

        fit = self._placeholder_fits[degree]

        self._chi_squared_label.setText(f"{fit.reduced_chi_squared:.3f}")

        coefficients_nm, coefficient_sigma_nm = self._convert_to_nm(
            np.asarray(fit.coefficients, dtype=np.float64), np.asarray(fit.coefficient_sigma, dtype=np.float64)
        )
        coeff_lines = [
            f"c<sub>{i}</sub> ({coefficient_unit(i)}) = {format_value_with_uncertainty(float(c), float(s))}"
            for i, (c, s) in enumerate(zip(coefficients_nm, coefficient_sigma_nm))
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

    def _x_values_for_columns(self, columns: np.ndarray) -> np.ndarray:
        # Wavelength (nm) per column if a real axis is loaded, else the raw
        # pixel-column index -- shared by the placeholder feed and real
        # per-tick rendering (_display_shot_result()) alike.
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

    def _convert_to_nm(
        self, x0: np.ndarray, sigma_x0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        '''
        position_calibration.convert() returns microns -- this is the
        nm-based counterpart to self._convert_to_mm(), used for every
        *quoted* spatial-dispersion/coefficient value (side-panel labels,
        strip chart) rather than a plotted graph axis, which stays in mm
        via self._convert_to_mm() regardless (see module docstring).
        Applying this elementwise to a whole coefficients array (c0
        through c_degree) is valid even though each c_k has different
        units (nm, nm/nm, nm/nm^2, ...): convert() is a pure linear px->
        physical scale with no additive offset, so only the numerator's
        units change for every k -- the wavelength_nm denominator implicit
        in c_k's units is never touched by this conversion.
        '''

        y0, sigma_y0 = self._position_calibration.convert(x0, sigma_x0)
        return microns_to_nm(y0), microns_to_nm(sigma_y0)

    def _zeta_to_nm(
        self, zeta_value: float, zeta_sigma: float | None = None
    ) -> tuple[float, float | None]:

        '''
        Converts a fitted zeta (analyze_shot()'s native px/nm) to physical
        units (nm/nm) via self._convert_to_nm() -- this codebase's
        quoted-value convention for spatial dispersion (see
        formatting.coefficient_unit()'s docstring), distinct from the
        mm-based self._convert_to_mm() used for plotted graph axes. Valid
        for a slope/derivative, not just a position, because
        ScaleFactorPositionCalibration.convert() is a pure linear scale
        with no additive offset (see its own docstring) -- the same
        combined_factor that turns a pixel position into a physical one
        turns a px/nm slope into an nm/nm one.

        zeta_sigma
            None (degree > 1's placeholder "uncertainty not available"
            case) passes through as None; a real sigma converts alongside
            zeta_value.
        '''

        zeta_nm, sigma_nm = self._convert_to_nm(
            np.array([zeta_value]), np.array([zeta_sigma if zeta_sigma is not None else 0.0])
        )
        return float(zeta_nm[0]), (float(sigma_nm[0]) if zeta_sigma is not None else None)

    def _build_placeholder_fits(self) -> dict[int, _PlaceholderFit]:

        '''
        Fake side-panel numbers, illustrative of typical real-world
        zeta magnitudes (px/nm-scale dispersion, converted to nm/nm via
        self._zeta_to_nm() -- matching the real-data path's units) once a
        real SpatialDispersionFitResult exists. Deliberately NOT derived
        from (and not numerically consistent with) the fake scatter/fit-
        curve/heatmap drawn in _populate_placeholder_data(), which instead
        uses its own simple "true" pixel-space trend -- both are
        placeholders for different parts of the layout, not a single
        coherent fake dataset. coefficients/coefficient_sigma stay in raw
        px/nm^k units, unconverted, same scoping as the real-data path
        (see _update_fit_panel_from_result()) -- _update_fit_panel()
        converts them to nm at display time via self._convert_to_nm().
        '''

        zeta_1, zeta_sigma_1 = self._zeta_to_nm(1.6e-3, 2.1e-5)
        zeta_2, _ = self._zeta_to_nm(1.62e-3)
        zeta_3, _ = self._zeta_to_nm(1.58e-3)

        return {
            1: _PlaceholderFit(
                coefficients=(0.02, 1.6e-3),
                coefficient_sigma=(0.01, 2.1e-5),
                reduced_chi_squared=1.04,
                zeta_value=zeta_1,
                zeta_sigma=zeta_sigma_1,
            ),
            2: _PlaceholderFit(
                coefficients=(0.01, 1.55e-3, 3.0e-7),
                coefficient_sigma=(0.01, 3.0e-5, 1.0e-7),
                reduced_chi_squared=0.98,
                zeta_value=zeta_2,
                zeta_sigma=None,
                evaluated_at_column=EVALUATED_AT_COLUMN,
            ),
            3: _PlaceholderFit(
                coefficients=(0.01, 1.5e-3, 2.8e-7, -4.0e-10),
                coefficient_sigma=(0.01, 4.0e-5, 1.5e-7, 6.0e-10),
                reduced_chi_squared=0.97,
                zeta_value=zeta_3,
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
    "DEFAULT_UPDATE_INTERVAL_MS",
    "MAX_CONSECUTIVE_SKIPS",
]
