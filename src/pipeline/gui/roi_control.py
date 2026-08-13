'''
Reusable min/max crop controls: SpatialROIControl for the spatial axis,
SpectralROIControl for the spectral axis. Both are deliberately
self-contained -- zero dependency on live_view.py -- so either can be
embedded both in live_view.py's live-view screen and an Extended
Measurement dialog, without either pulling in the other.

SpatialROIControl's bounds are entered and displayed in mm, matching
live_view.py's main-plot y-axis units (see gui/formatting.py's
microns_to_mm()/mm_to_microns()), even though the underlying
ScaleFactorPositionCalibration works in microns -- a simple, always-
invertible linear scale factor.

SpectralROIControl's bounds are entered and displayed in raw pixel-column
units instead of wavelength (nm), even when a wavelength calibration is
loaded: unlike the spatial scale factor, analysis.interfaces.WavelengthAxis
is a general degree-N polynomial pixel->wavelength_nm fit with no inverse
method, and may not exist at all yet (wavelength_axis is None until a
spectral calibration is loaded) -- pixel-column bounds are always
well-defined regardless. A read-only wavelength hint is shown alongside
the bounds when a WavelengthAxis is available, purely for the user's
reference (see wavelength_nm(), the one direction that's always defined).
'''

# Imports

import math

import numpy as np
from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..analysis.interfaces import WavelengthAxis
from ..calibration.spatial.calibrate import PIXEL_PITCH_UM, ScaleFactorPositionCalibration
from .formatting import microns_to_mm, mm_to_microns
from .theme import COLOR_BACKGROUND, COLOR_ERROR, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, load_bundled_font

# Constants

# Classes


class SpatialROIControl(QGroupBox):

    '''
    Reusable min/max spatial-axis crop control, in the same physical
    units (mm) as live_view.py's main-plot y-axis. Self-contained --
    zero dependency on live_view.py -- so a future Extended Measurement
    dialog can embed it directly.
    '''

    roi_changed = Signal(float, float)  # (min_mm, max_mm) -- emitted only on a successfully-applied (valid) change

    def __init__(
        self,
        position_calibration: ScaleFactorPositionCalibration,
        n_rows: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Spatial ROI", parent)

        self._position_calibration = position_calibration
        self._n_rows = n_rows

        extent_um, _ = position_calibration.convert(float(n_rows), 0.0)
        self._extent_mm = microns_to_mm(extent_um)
        self._last_valid_bounds: tuple[float, float] = (0.0, self._extent_mm)

        mm_per_px = microns_to_mm(PIXEL_PITCH_UM * position_calibration.scale_factor)
        decimals = max(3, math.ceil(-math.log10(mm_per_px)) + 1)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setRange(0.0, self._extent_mm)
        self._min_spin.setDecimals(decimals)
        self._min_spin.setSingleStep(mm_per_px)
        self._min_spin.setSuffix(" mm")
        self._min_spin.setValue(0.0)
        form.addRow("Min:", self._min_spin)

        self._max_spin = QDoubleSpinBox()
        self._max_spin.setRange(0.0, self._extent_mm)
        self._max_spin.setDecimals(decimals)
        self._max_spin.setSingleStep(mm_per_px)
        self._max_spin.setSuffix(" mm")
        self._max_spin.setValue(self._extent_mm)
        form.addRow("Max:", self._max_spin)

        layout.addLayout(form)

        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(f"color: {COLOR_ERROR};")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        self._reset_button = QPushButton("Reset to Full Range")
        self._reset_button.clicked.connect(self._on_reset_clicked)
        layout.addWidget(self._reset_button)

        self._min_spin.valueChanged.connect(self._on_bound_changed)
        self._max_spin.valueChanged.connect(self._on_bound_changed)

        self.setFont(load_bundled_font(10))
        self._min_spin.setFont(load_bundled_font(10))
        self._max_spin.setFont(load_bundled_font(10))
        self._error_label.setFont(load_bundled_font(10))
        self._reset_button.setFont(load_bundled_font(10))
        self.setStyleSheet(
            f"QGroupBox {{ background-color: {COLOR_BACKGROUND}; color: {COLOR_TEXT_PRIMARY}; }}"
        )

    def _on_bound_changed(self, _value: float) -> None:

        min_mm, max_mm = self._min_spin.value(), self._max_spin.value()
        if min_mm >= max_mm:
            self._error_label.setText("Min must be less than max.")
            self._error_label.setVisible(True)
            self._revert_to_last_valid()
            return
        self._error_label.setVisible(False)
        self._last_valid_bounds = (min_mm, max_mm)
        self.roi_changed.emit(min_mm, max_mm)

    def _revert_to_last_valid(self) -> None:

        min_mm, max_mm = self._last_valid_bounds
        with QSignalBlocker(self._min_spin), QSignalBlocker(self._max_spin):
            self._min_spin.setValue(min_mm)
            self._max_spin.setValue(max_mm)

    def _on_reset_clicked(self) -> None:

        with QSignalBlocker(self._min_spin), QSignalBlocker(self._max_spin):
            self._min_spin.setValue(0.0)
            self._max_spin.setValue(self._extent_mm)
        self._error_label.setVisible(False)
        self._last_valid_bounds = (0.0, self._extent_mm)
        self.roi_changed.emit(0.0, self._extent_mm)

    def roi_bounds_mm(self) -> tuple[float, float]:

        return self._min_spin.value(), self._max_spin.value()

    def roi_bounds_px(self) -> tuple[int, int]:

        '''
        NOTE: round() on both ends can under-cover the requested mm
        window by up to ~1 row (e.g. min rounds up, max rounds down) --
        harmless today (nothing pixel-exact consumes this yet), but
        reconsider floor()/ceil() if this ever feeds
        preprocessing/steps/roi.py's apply_roi() for real.
        '''

        min_mm, max_mm = self.roi_bounds_mm()
        min_px = round(float(self._position_calibration.to_pixels(mm_to_microns(min_mm))))
        max_px = round(float(self._position_calibration.to_pixels(mm_to_microns(max_mm))))
        min_px = max(0, min(self._n_rows, min_px))
        max_px = max(0, min(self._n_rows, max_px))
        return min_px, max_px


class SpectralROIControl(QGroupBox):

    '''
    Reusable min/max spectral-axis (column) crop control, in raw pixel-
    column units -- see module docstring for why not nm. Self-contained --
    zero dependency on live_view.py -- so a future Extended Measurement
    dialog can embed it directly, mirroring SpatialROIControl's own shape
    (min/max spin boxes, "Reset to Full Range", min < max validation) per
    docs/project_state.md's "GUI live view: manual ROI entry -- spectral
    axis" item.

    Unlike SpatialROIControl's roi_bounds_px() (always the current
    window, even at full range -- harmless there since applying the full
    spatial ROI is a physical no-op), this control distinguishes
    column_window() (always current, for display/viewport purposes) from
    column_bounds() (None at the full-range default). The distinction
    matters because preprocessing.steps.spectral_roi.apply_spectral_roi()
    OVERRIDES the automatic per-column SNR gate rather than just zeroing
    pixels -- passing the full range as an explicit override would force
    every column valid regardless of actual signal, silently disabling
    the SNR gate instead of leaving it in place unchanged.
    '''

    roi_changed = Signal(int, int)  # (column_min, column_max) -- emitted only on a successfully-applied (valid) change

    def __init__(
        self,
        n_columns: int,
        wavelength_axis: WavelengthAxis | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Spectral ROI", parent)

        self._n_columns = n_columns
        self._wavelength_axis = wavelength_axis
        self._last_valid_bounds: tuple[int, int] = (0, n_columns)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._min_spin = QSpinBox()
        self._min_spin.setRange(0, n_columns)
        self._min_spin.setValue(0)
        form.addRow("Min (column):", self._min_spin)

        self._max_spin = QSpinBox()
        self._max_spin.setRange(0, n_columns)
        self._max_spin.setValue(n_columns)
        form.addRow("Max (column):", self._max_spin)

        layout.addLayout(form)

        self._wavelength_hint_label = QLabel("")
        self._wavelength_hint_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        self._wavelength_hint_label.setVisible(wavelength_axis is not None)
        layout.addWidget(self._wavelength_hint_label)

        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(f"color: {COLOR_ERROR};")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        self._reset_button = QPushButton("Reset to Full Range")
        self._reset_button.clicked.connect(self._on_reset_clicked)
        layout.addWidget(self._reset_button)

        self._min_spin.valueChanged.connect(self._on_bound_changed)
        self._max_spin.valueChanged.connect(self._on_bound_changed)

        self.setFont(load_bundled_font(10))
        self._min_spin.setFont(load_bundled_font(10))
        self._max_spin.setFont(load_bundled_font(10))
        self._wavelength_hint_label.setFont(load_bundled_font(9))
        self._error_label.setFont(load_bundled_font(10))
        self._reset_button.setFont(load_bundled_font(10))
        self.setStyleSheet(
            f"QGroupBox {{ background-color: {COLOR_BACKGROUND}; color: {COLOR_TEXT_PRIMARY}; }}"
        )

        self._update_wavelength_hint()

    def _on_bound_changed(self, _value: int) -> None:

        column_min, column_max = self._min_spin.value(), self._max_spin.value()
        if column_min >= column_max:
            self._error_label.setText("Min must be less than max.")
            self._error_label.setVisible(True)
            self._revert_to_last_valid()
            return
        self._error_label.setVisible(False)
        self._last_valid_bounds = (column_min, column_max)
        self._update_wavelength_hint()
        self.roi_changed.emit(column_min, column_max)

    def _revert_to_last_valid(self) -> None:

        column_min, column_max = self._last_valid_bounds
        with QSignalBlocker(self._min_spin), QSignalBlocker(self._max_spin):
            self._min_spin.setValue(column_min)
            self._max_spin.setValue(column_max)

    def _on_reset_clicked(self) -> None:

        with QSignalBlocker(self._min_spin), QSignalBlocker(self._max_spin):
            self._min_spin.setValue(0)
            self._max_spin.setValue(self._n_columns)
        self._error_label.setVisible(False)
        self._last_valid_bounds = (0, self._n_columns)
        self._update_wavelength_hint()
        self.roi_changed.emit(0, self._n_columns)

    def _update_wavelength_hint(self) -> None:

        if self._wavelength_axis is None:
            return
        column_min, column_max = self.column_window()
        endpoints = self._wavelength_axis.wavelength_nm(
            np.array([column_min, column_max - 1], dtype=float)
        )
        low, high = sorted((float(endpoints[0]), float(endpoints[1])))
        self._wavelength_hint_label.setText(f"≈ {low:.1f} – {high:.1f} nm")

    def column_window(self) -> tuple[int, int]:

        '''
        Current (column_min, column_max) as entered, always -- including
        at the full-range default. For viewport/display purposes (e.g.
        zooming the main plot's x-axis to match); see class docstring for
        why this differs from column_bounds() below.
        '''

        return self._min_spin.value(), self._max_spin.value()

    def column_bounds(self) -> tuple[int, int] | None:

        '''
        Current (column_min, column_max), or None if at the full-range
        default -- directly usable as run_preprocessing()'s column_bounds
        argument. See class docstring for why full range must map to
        None rather than the literal (0, n_columns) tuple.
        '''

        column_min, column_max = self.column_window()
        if (column_min, column_max) == (0, self._n_columns):
            return None
        return column_min, column_max


# Functions


__all__ = ["SpatialROIControl", "SpectralROIControl"]
