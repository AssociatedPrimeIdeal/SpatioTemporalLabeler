from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QSlider, QWidget


class FloatSliderSpin(QWidget):
    """A continuous-value slider paired with an exact numeric editor."""

    valueChanged = Signal(float)

    def __init__(self, decimals: int = 5, parent: Any = None) -> None:
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 1.0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setTracking(True)
        self.slider.setMinimumWidth(130)
        self.slider.valueChanged.connect(self._slider_changed)
        layout.addWidget(self.slider, 1)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setMaximumWidth(105)
        self.spin.valueChanged.connect(self._spin_changed)
        layout.addWidget(self.spin)
        self.set_range(0.0, 1.0)

    def set_range(self, minimum: float, maximum: float) -> None:
        if maximum <= minimum:
            maximum = minimum + 1e-9
        current = self.value()
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        step = max((self._maximum - self._minimum) / 1000.0, 1e-9)
        self.spin.blockSignals(True)
        self.spin.setRange(self._minimum, self._maximum)
        self.spin.setSingleStep(step)
        self.spin.blockSignals(False)
        self.set_value(min(max(current, self._minimum), self._maximum))

    def set_value(self, value: float) -> None:
        value = min(max(float(value), self._minimum), self._maximum)
        fraction = (value - self._minimum) / (self._maximum - self._minimum)
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(int(round(fraction * self.slider.maximum())))
        self.spin.setValue(value)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def value(self) -> float:
        return float(self.spin.value())

    def _slider_changed(self, position: int) -> None:
        fraction = position / self.slider.maximum()
        value = self._minimum + fraction * (self._maximum - self._minimum)
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.valueChanged.emit(value)

    def _spin_changed(self, value: float) -> None:
        fraction = (float(value) - self._minimum) / (self._maximum - self._minimum)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(fraction * self.slider.maximum())))
        self.slider.blockSignals(False)
        self.valueChanged.emit(float(value))
