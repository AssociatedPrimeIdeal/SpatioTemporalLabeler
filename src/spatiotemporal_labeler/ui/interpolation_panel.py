from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .frame_labels import FrameLabel


class InterpolationPanel(QWidget):
    applyRequested = Signal()
    addCurrentRequested = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._frame_count = 1
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

        # The original panel exposed both start/end fields and two separate
        # frame lists.  Keep the fields as a compatibility fallback for
        # scripted callers, but present one unambiguous interpolation-frame
        # list to users.  The list combines cardiac phase markers, timeline
        # flags, and frames added with ``Add current frame``.
        for widget in (
            self.start_label,
            self.start_frame,
            self.end_label,
            self.end_frame,
        ):
            widget.setVisible(False)
        layout.addLayout(form)
        self.keyframe_label = QLabel()
        layout.addWidget(self.keyframe_label)
        self.keyframe_list = QListWidget()
        self.keyframe_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
        )
        self.keyframe_list.setAlternatingRowColors(True)
        self.keyframe_list.setMaximumHeight(110)
        layout.addWidget(self.keyframe_list)
        # Kept as an alias for API compatibility with older integrations.
        self.frame_labels_label = self.keyframe_label
        self.frame_labels_list = self.keyframe_list
        keyframe_buttons = QHBoxLayout()
        self.add_keyframe_button = QPushButton()
        self.add_keyframe_button.clicked.connect(self.addCurrentRequested)
        keyframe_buttons.addWidget(self.add_keyframe_button)
        self.remove_keyframe_button = QPushButton()
        self.remove_keyframe_button.clicked.connect(self._remove_selected_keyframe)
        keyframe_buttons.addWidget(self.remove_keyframe_button)
        self.clear_keyframes_button = QPushButton()
        self.clear_keyframes_button.clicked.connect(self._clear_keyframes)
        keyframe_buttons.addWidget(self.clear_keyframes_button)
        layout.addLayout(keyframe_buttons)

        self.phase_hint = QLabel()
        self.phase_hint.setWordWrap(True)
        layout.addWidget(self.phase_hint)

        self.apply_button = QPushButton()
        self.apply_button.clicked.connect(self.applyRequested)
        layout.addWidget(self.apply_button)
        layout.addStretch()
        self._manual_keyframes: set[int] = set()
        self._timeline_labels: dict[int, FrameLabel] = {}
        self.set_language("en")

    def set_detected_phases(self, phases: tuple[int, int, int] | None) -> None:
        self._phase_markers = phases
        self._update_phase_hint()

    def add_keyframe(self, frame_index: int) -> None:
        frame = int(frame_index)
        if not 0 <= frame < self._frame_count:
            return
        self._manual_keyframes.add(frame)
        self._rebuild_keyframe_list(select_frame=frame)
        for row in range(self.keyframe_list.count()):
            item = self.keyframe_list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == frame:
                item.setCheckState(Qt.CheckState.Checked)
                break

    def keyframe_values(self) -> tuple[int, ...]:
        return tuple(
            int(self.keyframe_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self.keyframe_list.count())
            if self.keyframe_list.item(row).checkState() == Qt.CheckState.Checked
        )

    def set_frame_labels(self, labels: tuple[FrameLabel, ...] | list[FrameLabel]) -> None:
        """Populate the single interpolation-frame list.

        Cardiac markers and user timeline flags are checked by default.  The
        checked state is retained when the active image is refreshed, while
        manually added frames remain in the list even though they have no
        timeline label.
        """
        existing_states = {
            int(self.keyframe_list.item(row).data(Qt.ItemDataRole.UserRole)):
            self.keyframe_list.item(row).checkState() == Qt.CheckState.Checked
            for row in range(self.keyframe_list.count())
        }
        by_frame = {
            int(label.frame): label
            for label in labels
            if 0 <= int(label.frame) < self._frame_count
        }
        self._timeline_labels = by_frame
        self._rebuild_keyframe_list(existing_states=existing_states)

    def _rebuild_keyframe_list(
        self,
        *,
        select_frame: int | None = None,
        existing_states: dict[int, bool] | None = None,
    ) -> None:
        existing_states = existing_states or {
            int(self.keyframe_list.item(row).data(Qt.ItemDataRole.UserRole)):
            self.keyframe_list.item(row).checkState() == Qt.CheckState.Checked
            for row in range(self.keyframe_list.count())
        }
        by_frame = self._timeline_labels
        frames = sorted(set(by_frame) | self._manual_keyframes)
        self.keyframe_list.clear()
        for frame in frames:
            label = by_frame.get(frame)
            text = (
                f"{label.name}  ({frame + 1})"
                if label is not None
                else f"Frame {frame + 1}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, frame)
            item.setData(
                Qt.ItemDataRole.UserRole + 1,
                label.phase_key if label else None,
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if existing_states.get(frame, True)
                else Qt.CheckState.Unchecked
            )
            self.keyframe_list.addItem(item)
        if select_frame is not None:
            for row in range(self.keyframe_list.count()):
                if (
                    int(self.keyframe_list.item(row).data(Qt.ItemDataRole.UserRole))
                    == select_frame
                ):
                    self.keyframe_list.setCurrentRow(row)
                    break

    def selected_frame_label_frames(self) -> tuple[int, ...]:
        return self.keyframe_values()

    def _remove_selected_keyframe(self) -> None:
        row = self.keyframe_list.currentRow()
        if row >= 0:
            item = self.keyframe_list.item(row)
            frame = int(item.data(Qt.ItemDataRole.UserRole))
            self._manual_keyframes.discard(frame)
            # Timeline labels remain available, but are no longer selected.
            if item.data(Qt.ItemDataRole.UserRole + 1) is not None:
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                self.keyframe_list.takeItem(row)

    def _clear_keyframes(self) -> None:
        self._manual_keyframes.clear()
        self.keyframe_list.clear()

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
        self._frame_count = count
        self._manual_keyframes.clear()
        self._timeline_labels = {}
        self.keyframe_list.clear()
        self.apply_button.setEnabled(count >= 3)

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.start_label.setText("起始关键帧" if chinese else "Start keyframe")
        self.end_label.setText("结束关键帧" if chinese else "End keyframe")
        self.labels_label.setText("作用标签" if chinese else "Labels")
        self.keyframe_label.setText(
            "插值帧（勾选时间轴标签或加入当前帧）"
            if chinese
            else "Interpolation frames (checked timeline labels or added frames)"
        )
        self.labels_scope.setItemText(0, "选中标签" if chinese else "Selected label")
        self.labels_scope.setItemText(1, "所有标签" if chinese else "All labels")
        self.add_keyframe_button.setText("加入当前帧" if chinese else "Add current frame")
        self.remove_keyframe_button.setText("删除" if chinese else "Remove")
        self.clear_keyframes_button.setText("清空" if chinese else "Clear")
        self.wrap_time.setText(
            "跨越时间轴首尾（循环插值）"
            if chinese
            else "Wrap across the time-axis ends (cyclic interpolation)"
        )
        self._update_phase_hint()
        self.apply_button.setText("插值中间帧" if chinese else "Interpolate intermediate frames")
