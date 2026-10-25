from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QMessageBox,
    QSlider,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .render_settings import RenderSettings
from .orientation_settings import ANATOMICAL_VECTORS, AXIS_DIRECTIONS, VIEW_AXES, OrientationSettings


DEFAULT_SHORTCUTS = {
    "brush": "B",
    "eraser": "E",
    "lasso": "S",
    "contour": "L",
    "picker": "I",
    "grow": "G",
    "all_frames_hold": "CapsLock",
    "threshold_bypass": "Q",
    "all_frames_bypass": "W",
    "all_frames_current_threshold": "",
    "hide_labels_hold": "H",
    "reset_view": "R",
    "temporal_stretch": "V",
    "previous_time": "Left",
    "next_time": "Right",
    "copy_frame": "Ctrl+C",
    "paste_frame": "Ctrl+V",
    "copy_previous_frame": "Ctrl+Shift+Left",
    "copy_next_frame": "Ctrl+Shift+Right",
    "frame_label": "K",
}

SHORTCUT_LABELS = {
    "en": {
        "brush": "Brush",
        "eraser": "Eraser",
        "lasso": "Scissors lasso",
        "contour": "Contour fill",
        "picker": "Hold to pick label",
        "grow": "Temporal propagation",
        "all_frames_hold": "All frames (threshold)",
        "threshold_bypass": "Ignore threshold",
        "all_frames_bypass": "All frames, ignore threshold",
        "all_frames_current_threshold": "All frames (current-frame threshold)",
        "hide_labels_hold": "Hold to hide 2D labels",
        "reset_view": "Reset 2D view",
        "temporal_stretch": "Cycle temporal time stretch",
        "previous_time": "Previous time frame",
        "next_time": "Next time frame",
        "copy_frame": "Copy current frame labels",
        "paste_frame": "Paste labels into current frame",
        "copy_previous_frame": "Copy previous frame to current",
        "copy_next_frame": "Copy next frame to current",
        "frame_label": "Add a timeline frame label",
    },
    "zh_CN": {
        "brush": "画笔",
        "eraser": "橡皮擦",
        "lasso": "剪刀套索",
        "contour": "闭合线填充",
        "picker": "按住标签取色",
        "grow": "时间传播",
        "all_frames_hold": "全帧（受阈值限制）",
        "threshold_bypass": "忽略阈值",
        "all_frames_bypass": "全帧且忽略阈值",
        "all_frames_current_threshold": "全帧（当前帧阈值）",
        "hide_labels_hold": "按住隐藏二维标签",
        "reset_view": "恢复二维视图",
        "temporal_stretch": "切换时空视图时间轴拉伸",
        "previous_time": "上一时间帧",
        "next_time": "下一时间帧",
        "copy_frame": "复制当前帧标签",
        "paste_frame": "将标签粘贴到当前帧",
        "copy_previous_frame": "复制上一帧到当前帧",
        "copy_next_frame": "复制下一帧到当前帧",
        "frame_label": "添加时间轴帧标签",
    },
}


class LabeledSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(
        self,
        minimum: int,
        maximum: int,
        value: int,
        suffix: str = "",
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.suffix = suffix
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.setTracking(True)
        layout.addWidget(self.slider, 1)
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setMinimumWidth(48)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._value_changed)
        self._value_changed(value)

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int) -> None:
        self.slider.setValue(value)

    def _value_changed(self, value: int) -> None:
        self.value_label.setText(f"{value}{self.suffix}")
        self.valueChanged.emit(value)


class SettingsDialog(QDialog):
    renderSettingsChanged = Signal(object)
    orientationSettingsChanged = Signal(object)

    def __init__(
        self,
        language: str,
        shortcuts: dict[str, str],
        parent: Any = None,
        render_settings: RenderSettings | None = None,
        orientation_settings: OrientationSettings | None = None,
    ) -> None:
        super().__init__(parent)
        chinese = False
        render_settings = render_settings or RenderSettings()
        orientation_settings = orientation_settings or OrientationSettings()
        self.setWindowTitle("设置" if chinese else "Settings")
        self.resize(540, 600)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        general_page = QWidget()
        form = QFormLayout(general_page)

        labels = SHORTCUT_LABELS["en"]
        self.shortcut_edits: dict[str, QKeySequenceEdit] = {}
        for key in DEFAULT_SHORTCUTS:
            editor = QKeySequenceEdit(QKeySequence(shortcuts.get(key, DEFAULT_SHORTCUTS[key])))
            editor.setClearButtonEnabled(True)
            form.addRow(labels[key], editor)
            self.shortcut_edits[key] = editor
        general_scroll = QScrollArea()
        general_scroll.setWidgetResizable(True)
        general_scroll.setWidget(general_page)
        self.tabs.addTab(general_scroll, "常规" if chinese else "General")
        self._build_orientation_page(chinese, orientation_settings)

        render_page = QWidget()
        render_form = QFormLayout(render_page)
        self.render_style = QComboBox()
        style_names = (
            (("clinical", "临床"), ("matte", "哑光"), ("glossy", "高光"))
            if chinese
            else (("clinical", "Clinical"), ("matte", "Matte"), ("glossy", "Glossy"))
        )
        for key, name in style_names:
            self.render_style.addItem(name, key)
        self.render_style.setCurrentIndex(
            max(0, self.render_style.findData(render_settings.style))
        )
        render_form.addRow("渲染风格" if chinese else "Rendering style", self.render_style)

        self.render_lighting = LabeledSlider(40, 160, render_settings.lighting, "%")
        render_form.addRow("光照" if chinese else "Lighting", self.render_lighting)
        self.render_smoothing = LabeledSlider(0, 16, render_settings.smoothing)
        render_form.addRow(
            "表面平滑" if chinese else "Surface smoothing", self.render_smoothing
        )

        self.render_detail = QComboBox()
        detail_names = (
            (("performance", "性能"), ("balanced", "平衡"), ("fine", "精细"))
            if chinese
            else (
                ("performance", "Performance"),
                ("balanced", "Balanced"),
                ("fine", "Fine"),
            )
        )
        for key, name in detail_names:
            self.render_detail.addItem(name, key)
        self.render_detail.setCurrentIndex(
            max(0, self.render_detail.findData(render_settings.detail))
        )
        render_form.addRow("细节等级" if chinese else "Detail level", self.render_detail)
        self.tabs.addTab(render_page, "3D 渲染" if chinese else "3D Rendering")

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(120)
        self._render_timer.timeout.connect(
            lambda: self.renderSettingsChanged.emit(self.render_values())
        )
        self.render_style.currentIndexChanged.connect(self._render_control_changed)
        self.render_lighting.valueChanged.connect(self._render_control_changed)
        self.render_smoothing.valueChanged.connect(self._render_control_changed)
        self.render_detail.currentIndexChanged.connect(self._render_control_changed)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        if chinese:
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
            buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
            buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText(
                "恢复默认"
            )
        layout.addWidget(buttons)

    def _build_orientation_page(
        self, chinese: bool, settings: OrientationSettings
    ) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        views = QGroupBox("2D 视图（含预览）" if chinese else "2D views and previews")
        grid = QGridLayout(views)
        grid.addWidget(QLabel("视图" if chinese else "View"), 0, 0)
        grid.addWidget(QLabel("屏幕左 → 右" if chinese else "Screen left → right"), 0, 1)
        grid.addWidget(QLabel("屏幕下 → 上" if chinese else "Screen bottom → top"), 0, 2)
        self.orientation_combos: dict[str, tuple[QComboBox, QComboBox]] = {}
        for row, (plane, axes) in enumerate(VIEW_AXES.items(), 1):
            grid.addWidget(QLabel(plane), row, 0)
            combos = []
            for column, axis in enumerate(axes, 1):
                combo = QComboBox()
                directions = AXIS_DIRECTIONS[axis]
                if chinese and axis == "T":
                    directions = ("较早", "较晚")
                combo.addItem(" → ".join(directions), False)
                combo.addItem(" → ".join(directions[::-1]), True)
                grid.addWidget(combo, row, column)
                combos.append(combo)
            self.orientation_combos[plane] = tuple(combos)
        layout.addWidget(views)

        spatial = QGroupBox("3D 视图" if chinese else "3D view")
        form = QFormLayout(spatial)
        direction_names = (
            {"A": "前方 (A)", "P": "后方 (P)", "R": "右侧 (R)", "L": "左侧 (L)",
             "H": "头侧 (H)", "F": "足侧 (F)"}
            if chinese else
            {"A": "Anterior (A)", "P": "Posterior (P)", "R": "Right (R)", "L": "Left (L)",
             "H": "Head (H)", "F": "Feet (F)"}
        )
        self.camera_view = QComboBox()
        self.camera_up = QComboBox()
        for direction, name in direction_names.items():
            self.camera_view.addItem(name, direction)
            self.camera_up.addItem(name, direction)
        form.addRow("默认从哪侧观察" if chinese else "Default view from", self.camera_view)
        form.addRow("屏幕上方" if chinese else "Screen up", self.camera_up)
        arrows = QWidget()
        arrow_layout = QHBoxLayout(arrows)
        arrow_layout.setContentsMargins(0, 0, 0, 0)
        self.axis_direction_combos: dict[str, QComboBox] = {}
        for axis in "XYZ":
            combo = QComboBox()
            for direction in AXIS_DIRECTIONS[axis]:
                combo.addItem(f"{axis}: {direction}", direction)
            self.axis_direction_combos[axis] = combo
            arrow_layout.addWidget(combo)
        form.addRow("方向轴箭头" if chinese else "Axis arrows", arrows)
        layout.addWidget(spatial)
        layout.addStretch()
        self.tabs.addTab(page, "视图方向" if chinese else "View Orientation")
        self._set_orientation_values(settings)
        controls = [self.camera_view, self.camera_up, *self.axis_direction_combos.values()]
        for combos in self.orientation_combos.values():
            controls.extend(combos)
        for combo in controls:
            combo.currentIndexChanged.connect(self._orientation_control_changed)

    def _update_camera_up_choices(self) -> None:
        direction = ANATOMICAL_VECTORS[str(self.camera_view.currentData())]
        blocked = self.camera_up.blockSignals(True)
        for index in range(self.camera_up.count()):
            up = ANATOMICAL_VECTORS[str(self.camera_up.itemData(index))]
            perpendicular = not any(a * b for a, b in zip(direction, up))
            self.camera_up.model().item(index).setEnabled(perpendicular)
        if not self.camera_up.model().item(self.camera_up.currentIndex()).isEnabled():
            fallback = "A" if self.camera_view.currentData() in {"H", "F"} else "H"
            self.camera_up.setCurrentIndex(self.camera_up.findData(fallback))
        self.camera_up.blockSignals(blocked)

    def _set_orientation_values(self, settings: OrientationSettings) -> None:
        controls = {self.camera_view: settings.camera_view, self.camera_up: settings.camera_up}
        for plane, combos in self.orientation_combos.items():
            controls.update(zip(combos, settings.flips(plane)))
        controls.update(
            (self.axis_direction_combos[axis], direction)
            for axis, direction in zip("XYZ", settings.axis_directions)
        )
        for combo, value in controls.items():
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(value))
            combo.blockSignals(blocked)
        self._update_camera_up_choices()

    def _orientation_control_changed(self, _value: object = None) -> None:
        self._update_camera_up_choices()
        self.orientationSettingsChanged.emit(self.orientation_values())

    def orientation_values(self) -> OrientationSettings:
        values = {}
        for plane, (horizontal, vertical) in self.orientation_combos.items():
            values[f"{plane}/horizontal"] = horizontal.currentData()
            values[f"{plane}/vertical"] = vertical.currentData()
        values["3d/view"] = self.camera_view.currentData()
        values["3d/up"] = self.camera_up.currentData()
        for axis, combo in self.axis_direction_combos.items():
            values[f"3d/{axis}"] = combo.currentData()
        return OrientationSettings.normalized(values)

    def _restore_defaults(self) -> None:
        for key, value in DEFAULT_SHORTCUTS.items():
            self.shortcut_edits[key].setKeySequence(QKeySequence(value))
        self._set_orientation_values(OrientationSettings())
        self.orientationSettingsChanged.emit(self.orientation_values())
        defaults = RenderSettings()
        self.render_style.setCurrentIndex(self.render_style.findData(defaults.style))
        self.render_lighting.set_value(defaults.lighting)
        self.render_smoothing.set_value(defaults.smoothing)
        self.render_detail.setCurrentIndex(self.render_detail.findData(defaults.detail))
        self._render_timer.start()

    def _render_control_changed(self, _value: object = None) -> None:
        self._render_timer.start()

    def done(self, result: int) -> None:
        self._render_timer.stop()
        super().done(result)

    def _validate_and_accept(self) -> None:
        sequences = [editor.keySequence() for editor in self.shortcut_edits.values()]
        values = [
            sequence.toString(QKeySequence.SequenceFormat.PortableText)
            for sequence in sequences
        ]
        assigned = [value for value in values if value]
        invalid_hold_key = any(
            self.shortcut_edits[key].keySequence().count() > 1
            for key in (
                "picker",
                "all_frames_hold",
                "threshold_bypass",
                "all_frames_bypass",
                "all_frames_current_threshold",
                "hide_labels_hold",
                "reset_view",
                "temporal_stretch",
                "previous_time",
                "next_time",
            )
        )
        if len(assigned) != len(set(assigned)) or invalid_hold_key:
            QMessageBox.warning(
                self,
                "Shortcut conflict",
                "Each action must use a different single-chord shortcut.",
            )
            return
        self.accept()

    def values(self) -> tuple[str, dict[str, str]]:
        return "en", {
            key: editor.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            for key, editor in self.shortcut_edits.items()
        }

    def render_values(self) -> RenderSettings:
        return RenderSettings.normalized(
            {
                "style": self.render_style.currentData(),
                "lighting": self.render_lighting.value(),
                "smoothing": self.render_smoothing.value(),
                "detail": self.render_detail.currentData(),
            }
        )
