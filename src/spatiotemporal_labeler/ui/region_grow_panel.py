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
        self.max_displacement = QDoubleSpinBox()
        self.max_displacement.setRange(0.0, 100.0)
        self.max_displacement.setDecimals(2)
        self.max_displacement.setSingleStep(0.5)
        self.max_displacement.setValue(4.8)
        self.max_displacement.setSuffix(" mm")
        self.max_displacement_label = QLabel()
        form.addRow(self.max_displacement_label, self.max_displacement)
        self.all_frames = QCheckBox()
        self.all_frames.setChecked(True)
        form.addRow(self.all_frames)
        self.frames_each_side = QSpinBox()
        self.frames_each_side.setRange(0, 100)
        self.frames_each_side.setValue(1)
        self.frames_each_side.setEnabled(False)
        self.frames_each_side_label = QLabel()
        form.addRow(self.frames_each_side_label, self.frames_each_side)
        self.all_frames.toggled.connect(
            lambda checked: self.frames_each_side.setEnabled(not checked)
        )
        self.replace_other_labels = QCheckBox()
        self.replace_other_labels.setChecked(False)
        form.addRow(self.replace_other_labels)
        layout.addLayout(form, 1)
        self.set_language("en")

    def set_image_range(self, low: float, high: float) -> None:
        self.tolerance_control.set_range(0.0, 1.0)
        self.tolerance_control.set_value(0.05)

    def set_frame_count(self, frame_count: int) -> None:
        self.frames_each_side.setMaximum(max(0, int(frame_count) - 1))

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.tolerance_label.setText(
            "帧间强度容差" if chinese else "Inter-frame intensity tolerance"
        )
        self.max_displacement_label.setText(
            "最大帧间位移" if chinese else "Maximum inter-frame shift"
        )
        self.all_frames.setText("所有时间帧" if chinese else "All frames")
        self.frames_each_side_label.setText("单侧帧数" if chinese else "Frames each side")
        self.replace_other_labels.setText(
            "替换其他标签" if chinese else "Replace other labels"
        )
