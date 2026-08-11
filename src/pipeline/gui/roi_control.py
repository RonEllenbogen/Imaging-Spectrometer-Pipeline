'''
Reusable min/max spatial-axis crop control. Deliberately self-contained
-- zero dependency on live_view.py -- so it can be embedded both in
live_view.py's live-view screen and, later, an Extended Measurement
dialog, without either pulling in the other. Bounds are entered and
displayed in mm, matching live_view.py's main-plot y-axis units (see
gui/formatting.py's microns_to_mm()/mm_to_microns()), even though the
underlying ScaleFactorPositionCalibration works in microns.
'''

# Imports

import math

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..calibration.spatial.calibrate import PIXEL_PITCH_UM, ScaleFactorPositionCalibration
from .formatting import microns_to_mm, mm_to_microns
from .theme import COLOR_BACKGROUND, COLOR_ERROR, COLOR_TEXT_PRIMARY, load_bundled_font

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


# Functions


__all__ = ["SpatialROIControl"]
