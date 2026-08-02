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
            decimals=4, orientation=Qt.Orientation.Horizontal
        )
        self.tolerance = self.tolerance_control.spin
        self.tolerance_label = QLabel()
        form.addRow(self.tolerance_label, self.tolerance_control)
        # 最大物理位移与强度容差均提供滑条和精确数值输入。
        self.max_displacement_control = FloatSliderSpin(
            decimals=2, orientation=Qt.Orientation.Horizontal
        )
        self.max_displacement_control.set_range(0.0, 100.0)
        self.max_displacement = self.max_displacement_control.spin
        self.max_displacement_control.set_value(4.8)
        self.max_displacement.setSuffix(" mm")
        self.max_displacement_label = QLabel()
        form.addRow(self.max_displacement_label, self.max_displacement_control)
        # 运动连续性使用物理位移，避免与原始强度容差混合单位。
        self.motion_smoothness = QDoubleSpinBox()
        self.motion_smoothness.setRange(0.0, 10.0)
        self.motion_smoothness.setDecimals(2)
        self.motion_smoothness.setSingleStep(0.1)
        self.motion_smoothness.setValue(1.0)
        self.motion_smoothness_label = QLabel()
        form.addRow(self.motion_smoothness_label, self.motion_smoothness)
        self.max_motion_change = QDoubleSpinBox()
        self.max_motion_change.setRange(0.0, 100.0)
        self.max_motion_change.setDecimals(2)
        self.max_motion_change.setSingleStep(0.5)
        self.max_motion_change.setValue(2.4)
        self.max_motion_change.setSuffix(" mm")
        self.max_motion_change_label = QLabel()
        form.addRow(self.max_motion_change_label, self.max_motion_change)
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
        # 容差直接使用原始图像强度单位，并以完整数据范围的 5% 作为默认值。
        span = max(abs(float(high) - float(low)), 1e-9)
        self.tolerance_control.set_range(0.0, span)
        self.tolerance_control.set_value(span * 0.05)

    def set_frame_count(self, frame_count: int) -> None:
        self.frames_each_side.setMaximum(max(0, int(frame_count) - 1))

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.tolerance_label.setText(
            "帧间原始强度容差" if chinese else "Inter-frame raw intensity tolerance"
        )
        self.max_displacement_label.setText(
            "最大帧间位移" if chinese else "Maximum inter-frame shift"
        )
        self.motion_smoothness_label.setText(
            "运动平滑权重" if chinese else "Motion smoothness"
        )
        self.max_motion_change_label.setText(
            "最大运动变化" if chinese else "Maximum motion change"
        )
        self.all_frames.setText("所有时间帧" if chinese else "All frames")
        self.frames_each_side_label.setText("单侧帧数" if chinese else "Frames each side")
        self.replace_other_labels.setText(
            "替换其他标签" if chinese else "Replace other labels"
        )
