from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class InterpolationPanel(QWidget):
    applyRequested = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()

        self.start_label = QLabel()
        self.start_frame = QSpinBox()
        self.start_frame.setRange(1, 1)
        form.addRow(self.start_label, self.start_frame)

        self.end_label = QLabel()
        self.end_frame = QSpinBox()
        self.end_frame.setRange(1, 1)
        form.addRow(self.end_label, self.end_frame)

        self.labels_label = QLabel()
        self.labels_scope = QComboBox()
        self.labels_scope.addItem("", "selected")
        self.labels_scope.addItem("", "all")
        form.addRow(self.labels_label, self.labels_scope)

        self.wrap_time = QCheckBox()
        # Label sequences in this application are typically complete periodic
        # cycles, so the cyclic path is the safe default. Users can still turn
        # it off for a sequence that represents only a non-periodic segment.
        self.wrap_time.setChecked(True)
        form.addRow(self.wrap_time)

        self.phase_hint = QLabel()
        self.phase_hint.setWordWrap(True)
        layout.addWidget(self.phase_hint)

        layout.addLayout(form)
        self.apply_button = QPushButton()
        self.apply_button.clicked.connect(self.applyRequested)
        layout.addWidget(self.apply_button)
        layout.addStretch()
        self.set_language("en")

    def set_detected_phases(self, phases: tuple[int, int, int] | None) -> None:
        self._phase_markers = phases
        self._update_phase_hint()

    def _update_phase_hint(self) -> None:
        phases = getattr(self, "_phase_markers", None)
        if phases is None:
            self.phase_hint.setText(
                "No cardiac phase estimate" if self.wrap_time.text().startswith("Wrap")
                else "未检测到心动周期"
            )
            return
        start, peak, end = (int(value) + 1 for value in phases)
        chinese = self.wrap_time.text().startswith("跨")
        self.phase_hint.setText(
            (
                f"检测相位：收缩开始 {start}，峰值 {peak}，舒张开始 {end}"
                if chinese
                else f"Detected phases: systole start {start}, peak {peak}, diastole start {end}"
            )
        )

    def set_frame_count(self, count: int, current_frame: int = 0) -> None:
        count = max(1, int(count))
        current = max(0, min(int(current_frame), count - 1))
        start = min(current, max(0, count - 3))
        end = min(count - 1, start + 2)
        self.start_frame.setRange(1, count)
        self.end_frame.setRange(1, count)
        self.start_frame.setValue(start + 1)
        self.end_frame.setValue(end + 1)
        self.apply_button.setEnabled(count >= 3)

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.start_label.setText("起始关键帧" if chinese else "Start keyframe")
        self.end_label.setText("结束关键帧" if chinese else "End keyframe")
        self.labels_label.setText("作用标签" if chinese else "Labels")
        self.labels_scope.setItemText(0, "选中标签" if chinese else "Selected label")
        self.labels_scope.setItemText(1, "所有标签" if chinese else "All labels")
        self.wrap_time.setText(
            "跨越时间轴首尾（循环插值）"
            if chinese
            else "Wrap across the time-axis ends (cyclic interpolation)"
        )
        self._update_phase_hint()
        self.apply_button.setText("插值中间帧" if chinese else "Interpolate intermediate frames")
