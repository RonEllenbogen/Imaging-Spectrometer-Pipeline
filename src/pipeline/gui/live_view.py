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
same kind of skeleton: editing either field runs the real drift-detection
comparison against calibration_set.baseline_record (and
conversion_gain_record, if supplied) and can pop a "recalibrate?" prompt,
but does NOT reconfigure camera_stream -- that's deferred to the same
future "real camera wiring" phase as the QTimer update loop above.
'''

# Imports

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
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

from pipeline.acquisition import CameraStream
from pipeline.analysis import SensorNoiseModel
from pipeline.analysis.interfaces import WavelengthAxis
from pipeline.calibration.sensor import ConversionGainRecord
from pipeline.calibration.shared import EXPOSURE_MATCH_TOLERANCE_REL, GAIN_MATCH_TOLERANCE_ABS
from pipeline.calibration.spatial import ScaleFactorPositionCalibration
from pipeline.preprocessing import CalibrationSet

from pipeline.gui.theme import (
    COLOR_ACCENT as ACCENT_COLOR,
    COLOR_BACKGROUND as BACKGROUND_COLOR,
    COLOR_TEXT_PRIMARY as FOREGROUND_COLOR,
    COLOR_PLOT_GRID as GRID_COLOR,
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

# Classes

@dataclass(frozen=True)
class _PlaceholderFit:

    '''Bundles one degree's worth of fake side-panel numbers for the skeleton.'''

    coefficients: tuple[float, ...]
    coefficient_sigma: tuple[float, ...]
    reduced_chi_squared: float
    zeta_value: float
    zeta_sigma: float | None   # None => "uncertainty not available" note (degree > 1)


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
        self._main_plot.setLabel("left", "Physical position (mm)")
        self._main_plot.setLabel("bottom", self._x_axis_label())
        self._main_plot.showGrid(x=True, y=True, alpha=0.3)
        self._style_plot_axes(self._main_plot)

        # Heatmap first, so the scatter/fit curve render on top of it.
        self._image_item = pg.ImageItem()
        self._main_plot.addItem(self._image_item)

        self._fit_curve = pg.PlotDataItem(pen=pg.mkPen(color=ACCENT_COLOR, width=2))
        self._main_plot.addItem(self._fit_curve)

        self._scatter = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 200)
        )
        self._main_plot.addItem(self._scatter)

        self._error_bars = pg.ErrorBarItem(pen=pg.mkPen(color=FOREGROUND_COLOR, width=1))
        self._main_plot.addItem(self._error_bars)

        plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return plot_widget

    def _build_strip_chart(self) -> pg.PlotWidget:

        plot_widget = pg.PlotWidget()
        self._strip_plot = plot_widget.getPlotItem()
        self._strip_plot.setLabel("left", "zeta")
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
        plot_widget.setMinimumHeight(140)
        plot_widget.setMaximumHeight(180)
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
        layout.addWidget(self._build_degree_selector_group())
        layout.addWidget(self._build_fit_diagnostics_group())
        layout.addStretch(1)

        self._extended_measurement_button = QPushButton("Extended Measurement...")
        self._extended_measurement_button.setFont(load_bundled_font(10))
        self._extended_measurement_button.setToolTip(
            "Not yet implemented -- placeholder for a future extended-"
            "measurement / N-shot combination workflow."
        )
        # Deliberately left unconnected: building the feature itself is a
        # non-goal of this phase (see module docstring).
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
        group.setToolTip(
            "Skeleton only -- does not reconfigure the camera. Pre-filled "
            "from the loaded baseline's capture settings; drifting past "
            "tolerance from the loaded calibrations prompts a "
            "recalibration check."
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

    # -- acquisition settings drift detection ----------------------------

    def _on_exposure_changed(self, exposure_us: float) -> None:

        baseline_exposure_us = self._calibration_set.baseline_record.exposure_us
        if not exposure_has_drifted(exposure_us, baseline_exposure_us):
            return

        self._prompt_recalibration(
            artifact_type="baseline",
            message=(
                f"Exposure has changed from {baseline_exposure_us:.1f} us "
                f"(baseline) to {exposure_us:.1f} us. Recalibrate baseline?"
            ),
        )

    def _on_gain_changed(self, gain_db: float) -> None:

        # Gain is checked against baseline and (if loaded) conversion-gain
        # independently, since the two artifacts can drift apart from one
        # another -- e.g. a re-run baseline at the new gain would clear the
        # first warning while the conversion-gain artifact is still stale.
        baseline_gain_db = self._calibration_set.baseline_record.gain_db
        if gain_has_drifted(gain_db, baseline_gain_db):
            self._prompt_recalibration(
                artifact_type="baseline",
                message=(
                    f"Gain has changed from {baseline_gain_db:.1f} dB "
                    f"(baseline) to {gain_db:.1f} dB. Recalibrate baseline?"
                ),
            )

        if self._conversion_gain_record is not None:
            conversion_gain_db = self._conversion_gain_record.gain_db
            if gain_has_drifted(gain_db, conversion_gain_db):
                self._prompt_recalibration(
                    artifact_type="conversion_gain",
                    message=(
                        f"Gain has changed from {conversion_gain_db:.1f} dB "
                        f"(conversion-gain calibration) to {gain_db:.1f} dB. "
                        f"Recalibrate conversion gain?"
                    ),
                )

    def _prompt_recalibration(self, artifact_type: str, message: str) -> None:

        '''
        Asks the user whether to recalibrate a specific drifted artifact.
        Emits recalibration_requested(artifact_type) on "Yes" -- nothing
        in this codebase yet connects to that signal (see its own
        docstring in the class header), the same not-yet-wired treatment
        as _extended_measurement_button below.

        Parameters
        ----------
        artifact_type
            Which artifact drifted -- "baseline" or "conversion_gain".
        message
            Full prompt text, naming the setting, its captured value, and
            the newly-entered value.
        '''

        response = QMessageBox.question(
            self,
            "Recalibration Recommended",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response == QMessageBox.Yes:
            self.recalibration_requested.emit(artifact_type)

    def _build_degree_selector_group(self) -> QGroupBox:

        group = QGroupBox("Fit degree")
        group.setFont(load_bundled_font(10))
        layout = QVBoxLayout(group)

        self._degree_selector = QComboBox()
        self._degree_selector.setFont(load_bundled_font(10))
        for degree in DEGREE_CHOICES:
            self._degree_selector.addItem(DEGREE_LABELS[degree], userData=degree)
        self._degree_selector.setCurrentIndex(DEGREE_CHOICES.index(DEFAULT_DEGREE))
        self._degree_selector.currentIndexChanged.connect(self._on_degree_changed)
        layout.addWidget(self._degree_selector)

        return group

    def _build_fit_diagnostics_group(self) -> QGroupBox:

        group = QGroupBox("Fit diagnostics")
        group.setFont(load_bundled_font(10))
        form = QFormLayout(group)

        label_font = load_bundled_font(10)

        self._chi_squared_label = QLabel("--")
        self._coefficients_label = QLabel("--")
        self._coefficients_label.setWordWrap(True)
        self._zeta_label = QLabel("--")
        self._zeta_note_label = QLabel("")
        self._zeta_note_label.setWordWrap(True)
        self._zeta_note_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-style: italic;")

        for label in (
            self._chi_squared_label, self._coefficients_label,
            self._zeta_label, self._zeta_note_label,
        ):
            label.setFont(label_font)

        form.addRow("Reduced chi-squared:", self._chi_squared_label)
        form.addRow("Coefficients:", self._coefficients_label)
        form.addRow("Zeta:", self._zeta_label)
        form.addRow("", self._zeta_note_label)

        return group

    # -- placeholder data (Phase 1 only -- no real camera/analysis calls) --

    def _populate_placeholder_data(self) -> None:

        self._placeholder_fits = self._build_placeholder_fits()

        rng = np.random.default_rng(seed=0)

        n_rows, n_cols = 1200, 1920
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
        y_values, y_sigma = self._position_calibration.convert(
            centroid_px, np.full_like(centroid_px, 3.0)
        )
        x_sigma = (
            np.full_like(x_values, 0.4) if self._wavelength_axis is not None else None
        )

        self._scatter.setData(x=x_values, y=y_values)
        error_kwargs = dict(x=x_values, y=y_values, top=y_sigma, bottom=y_sigma)
        if x_sigma is not None:
            error_kwargs["left"] = x_sigma
            error_kwargs["right"] = x_sigma
        self._error_bars.setData(**error_kwargs)

        # -- fit-curve overlay (drawn from the same fake "true" trend, --
        # -- not from _placeholder_fits -- see docstring above) ---------
        fit_x = np.linspace(x_values.min(), x_values.max(), 200)
        fit_columns = columns.min() + (fit_x - x_values.min()) / (
            x_values.max() - x_values.min()
        ) * (columns.max() - columns.min())
        fit_y, _ = self._position_calibration.convert(
            true_centroid_px(fit_columns), np.zeros_like(fit_columns)
        )
        self._fit_curve.setData(x=fit_x, y=fit_y)

        # -- fake raw-preprocessed-frame heatmap, standing in for -------
        # -- ProcessedFrame.image, using the same beam-centroid trend ---
        row_idx = np.arange(n_rows)[:, None]
        col_idx = np.arange(n_cols)[None, :]
        beam_center = true_centroid_px(col_idx)
        image = 200 * np.exp(-((row_idx - beam_center) ** 2) / (2 * 60**2))
        image = image + rng.normal(scale=5, size=image.shape)
        image = np.clip(image, 0, None).astype(np.float32)

        self._image_item.setImage(image)
        x0, x1 = self._heatmap_x_extent(first_column=0, last_column=n_cols - 1)
        y0, y1 = self._position_calibration.convert(
            np.array([0.0, float(n_rows)]), np.array([0.0, 0.0])
        )[0]
        self._image_item.setRect(
            min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)
        )

        self._update_strip_chart_placeholder(rng)
        self._update_fit_panel(DEFAULT_DEGREE)

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
            f"c{i} = {c:.4g} +/- {s:.2g}"
            for i, (c, s) in enumerate(zip(fit.coefficients, fit.coefficient_sigma))
        ]
        self._coefficients_label.setText("\n".join(coeff_lines))

        if fit.zeta_sigma is not None:
            self._zeta_label.setText(f"{fit.zeta_value:.4g} +/- {fit.zeta_sigma:.2g}")
            self._zeta_note_label.setText("")
        else:
            self._zeta_label.setText(f"{fit.zeta_value:.4g} (no uncertainty)")
            self._zeta_note_label.setText(
                "Uncertainty not available for degree > 1 in live view "
                "(evaluated at this frame's median x-value; no internal "
                "covariance-based estimate exists yet)."
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
            ),
            3: _PlaceholderFit(
                coefficients=(0.01, 1.5e-3, 2.8e-7, -4.0e-10),
                coefficient_sigma=(0.01, 4.0e-5, 1.5e-7, 6.0e-10),
                reduced_chi_squared=0.97,
                zeta_value=1.58e-3,
                zeta_sigma=None,
            ),
        }


# Functions

def wavelength_axis_label(wavelength_axis: WavelengthAxis | None) -> str:

    '''
    X-axis label for the main plot: real wavelength when a
    WavelengthAxis is supplied, an explicit "not yet available" fallback
    label when it's None (the expected v1 state, not an error -- see
    calibration/spectral/line_matching.py's module docstring).
    '''

    if wavelength_axis is not None:
        return "Wavelength (nm)"
    return "Pixel column (wavelength calibration not yet available)"


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
    "exposure_has_drifted",
    "gain_has_drifted",
    "DEGREE_CHOICES",
    "DEGREE_LABELS",
    "DEFAULT_DEGREE",
    "STRIP_CHART_WINDOW_SECONDS",
]
