from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .value_slider import FloatSliderSpin


class RegionGrowPanel(QWidget):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        self.tolerance_control = FloatSliderSpin(
            decimals=4, orientation=Qt.Orientation.Vertical
        )
        self.tolerance = self.tolerance_control.spin
        self.tolerance_label = QLabel()
        form.addRow(self.tolerance_label, self.tolerance_control)
        self.spatial_range = QDoubleSpinBox()
        self.spatial_range.setRange(0.0, 500.0)
        self.spatial_range.setDecimals(2)
        self.spatial_range.setSingleStep(1.0)
        self.spatial_range.setValue(12.0)
        self.spatial_range.setSuffix(" mm")
        self.spatial_range_label = QLabel()
        form.addRow(self.spatial_range_label, self.spatial_range)
        self.frames_each_side = QSpinBox()
        self.frames_each_side.setRange(0, 100)
        self.frames_each_side.setValue(1)
        self.frames_each_side_label = QLabel()
        form.addRow(self.frames_each_side_label, self.frames_each_side)
        self.replace_other_labels = QCheckBox()
        self.replace_other_labels.setChecked(False)
        form.addRow(self.replace_other_labels)
        layout.addLayout(form)
        layout.addStretch()
        self.set_language("en")

    def set_image_range(self, low: float, high: float) -> None:
        span = max(abs(high - low), 1e-6)
        self.tolerance_control.set_range(0.0, span)
        self.tolerance_control.set_value(span * 0.05)

    def set_frame_count(self, frame_count: int) -> None:
        self.frames_each_side.setMaximum(max(0, int(frame_count) - 1))

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.tolerance_label.setText(
            "强度容差" if chinese else "Intensity tolerance"
        )
        self.spatial_range_label.setText("空间范围" if chinese else "Spatial range")
        self.frames_each_side_label.setText("单侧帧数" if chinese else "Frames each side")
        self.replace_other_labels.setText(
            "替换其他标签" if chinese else "Replace other labels"
        )
