from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .value_slider import FloatSliderSpin


METHOD_NAMES = {
    "manual": "Manual",
    "otsu": "Otsu",
    "triangle": "Triangle",
    "li": "Li",
    "yen": "Yen",
    "isodata": "Isodata",
    "multi_otsu": "Multi-Otsu",
    "kittler": "Kittler",
    "local_gaussian": "Local Gaussian",
    "sauvola": "Sauvola",
    "phansalkar": "Phansalkar",
    "hysteresis": "Hysteresis",
}


class ThresholdPanel(QWidget):
    changed = Signal()
    methodChanged = Signal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._image_minimum = 0.0
        self._image_maximum = 1.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.enabled = QCheckBox()
        self.enabled.toggled.connect(self.changed)
        layout.addWidget(self.enabled)
        self.preview = QCheckBox()
        self.preview.setChecked(True)
        self.preview.toggled.connect(self.changed)
        layout.addWidget(self.preview)
        form = QFormLayout()
        self.method = QComboBox()
        for key, name in METHOD_NAMES.items():
            self.method.addItem(name, key)
        self.method.currentIndexChanged.connect(self._method_changed)
        self.method_label = QLabel()
        form.addRow(self.method_label, self.method)

        self.lower_control = FloatSliderSpin(decimals=2)
        self.lower_control.set_range(0.0, 100.0)
        self.lower_control.spin.setSuffix(" %")
        self.lower_control.valueChanged.connect(self._lower_changed)
        self.lower = self.lower_control.spin
        self.lower_slider = self.lower_control.slider
        self.lower_label = QLabel()
        form.addRow(self.lower_label, self.lower_control)

        self.upper_control = FloatSliderSpin(decimals=2)
        self.upper_control.set_range(0.0, 100.0)
        self.upper_control.set_value(100.0)
        self.upper_control.spin.setSuffix(" %")
        self.upper_control.valueChanged.connect(self._upper_changed)
        self.upper = self.upper_control.spin
        self.upper_slider = self.upper_control.slider
        self.upper_label = QLabel()
        form.addRow(self.upper_label, self.upper_control)

        self.radius = QSpinBox()
        self.radius.setRange(1, 64)
        self.radius.setValue(7)
        self.radius.valueChanged.connect(self.changed)
        self.radius_label = QLabel()
        form.addRow(self.radius_label, self.radius)
        layout.addLayout(form)
        layout.addStretch()
        self.set_language("en")

    def set_image_range(self, low: float, high: float) -> None:
        self._image_minimum = float(min(low, high))
        self._image_maximum = float(max(low, high))
        self.set_percent_bounds(0.0, 100.0)

    def set_bounds(self, lower: float, upper: float) -> None:
        lower, upper = sorted((float(lower), float(upper)))
        span = self._image_maximum - self._image_minimum
        if span <= 0.0:
            self.set_percent_bounds(0.0, 100.0)
            return
        self.set_percent_bounds(
            100.0 * (lower - self._image_minimum) / span,
            100.0 * (upper - self._image_minimum) / span,
        )

    def set_percent_bounds(self, lower: float, upper: float) -> None:
        lower, upper = sorted((float(lower), float(upper)))
        self.lower_control.set_value(float(np.clip(lower, 0.0, 100.0)))
        self.upper_control.set_value(float(np.clip(upper, 0.0, 100.0)))
        self.changed.emit()

    @property
    def intensity_bounds(self) -> tuple[float, float]:
        span = self._image_maximum - self._image_minimum
        lower = self._image_minimum + self.lower_control.value() * span / 100.0
        upper = self._image_minimum + self.upper_control.value() * span / 100.0
        return lower, upper

    def _lower_changed(self, value: float) -> None:
        if value > self.upper_control.value():
            self.upper_control.set_value(value)
        self.changed.emit()

    def _upper_changed(self, value: float) -> None:
        if value < self.lower_control.value():
            self.lower_control.set_value(value)
        self.changed.emit()

    def _method_changed(self, _index: int) -> None:
        self.methodChanged.emit(self.method_key)

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.enabled.setText("启用阈值蒙版" if chinese else "Enable threshold mask")
        self.preview.setText("实时预览生效范围" if chinese else "Live selection preview")
        self.method_label.setText("方法" if chinese else "Method")
        self.lower_label.setText("下限" if chinese else "Lower")
        self.upper_label.setText("上限" if chinese else "Upper")
        self.radius_label.setText("局部半径" if chinese else "Local radius")

    @property
    def method_key(self) -> str:
        return str(self.method.currentData())
