from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from .value_slider import FloatSliderSpin


class WindowLevelPanel(QWidget):
    changed = Signal(float, float)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        self.level_control = FloatSliderSpin(decimals=3)
        self.level = self.level_control.spin
        self.level_label = QLabel()
        form.addRow(self.level_label, self.level_control)
        self.width_control = FloatSliderSpin(decimals=3)
        self.width = self.width_control.spin
        self.width_label = QLabel()
        form.addRow(self.width_label, self.width_control)
        self.level_control.valueChanged.connect(self._emit_changed)
        self.width_control.valueChanged.connect(self._emit_changed)
        layout.addLayout(form)
        layout.addStretch()
        self.set_language("en")

    def set_image_range(self, low: float, high: float) -> None:
        span = max(abs(high - low), 1e-6)
        self.level_control.set_range(low, high)
        self.width_control.set_range(max(span / 1000.0, 1e-9), span * 2.0)

    def set_values(self, level: float, width: float) -> None:
        self.level_control.set_value(level)
        self.width_control.set_value(width)

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.level_label.setText("窗位" if chinese else "Window level")
        self.width_label.setText("窗宽" if chinese else "Window width")

    def _emit_changed(self, _value: float) -> None:
        self.changed.emit(self.level_control.value(), self.width_control.value())
