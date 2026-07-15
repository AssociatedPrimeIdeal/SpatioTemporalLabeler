from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout, QWidget

from .value_slider import FloatSliderSpin


class RegionGrowPanel(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        self.scope = QComboBox()
        self.scope.addItem("2D slice", "2d")
        self.scope.addItem("3D frame", "3d")
        self.scope_label = QLabel()
        form.addRow(self.scope_label, self.scope)
        self.tolerance_control = FloatSliderSpin(decimals=4)
        self.tolerance = self.tolerance_control.spin
        self.tolerance_label = QLabel()
        form.addRow(self.tolerance_label, self.tolerance_control)
        layout.addLayout(form)
        layout.addStretch()
        self.set_language("en")

    def set_image_range(self, low: float, high: float) -> None:
        span = max(abs(high - low), 1e-6)
        self.tolerance_control.set_range(0.0, span)
        self.tolerance_control.set_value(span * 0.05)

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.scope_label.setText("范围" if chinese else "Scope")
        self.tolerance_label.setText("容差" if chinese else "Tolerance")
        self.scope.setItemText(0, "二维切片" if chinese else "2D slice")
        self.scope.setItemText(1, "三维帧" if chinese else "3D frame")
