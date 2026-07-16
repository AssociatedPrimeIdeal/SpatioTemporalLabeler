from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QMessageBox,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spatiotemporal_labeler.i18n import LANGUAGES

from .render_settings import RenderSettings


DEFAULT_SHORTCUTS = {
    "brush": "B",
    "eraser": "E",
    "lasso": "S",
    "contour": "L",
    "picker": "I",
    "grow": "G",
    "all_frames_hold": "CapsLock",
    "threshold_bypass": "Q",
    "reset_view": "R",
    "previous_time": "Left",
    "next_time": "Right",
}

SHORTCUT_LABELS = {
    "en": {
        "brush": "Brush",
        "eraser": "Eraser",
        "lasso": "Scissors lasso",
        "contour": "Contour fill",
        "picker": "Hold to pick label",
        "grow": "Seed region grow",
        "all_frames_hold": "Hold for all frames",
        "threshold_bypass": "Hold to bypass threshold",
        "reset_view": "Reset 2D view",
        "previous_time": "Previous time frame",
        "next_time": "Next time frame",
    },
    "zh_CN": {
        "brush": "画笔",
        "eraser": "橡皮擦",
        "lasso": "剪刀套索",
        "contour": "闭合线填充",
        "picker": "按住标签取色",
        "grow": "种子区域生长",
        "all_frames_hold": "按住启用所有时间帧",
        "threshold_bypass": "按住绕过阈值蒙版",
        "reset_view": "恢复二维视图",
        "previous_time": "上一时间帧",
        "next_time": "下一时间帧",
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

    def __init__(
        self,
        language: str,
        shortcuts: dict[str, str],
        parent: Any = None,
        render_settings: RenderSettings | None = None,
    ) -> None:
        super().__init__(parent)
        chinese = language == "zh_CN"
        render_settings = render_settings or RenderSettings()
        self.setWindowTitle("设置" if chinese else "Settings")
        self.resize(540, 600)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        general_page = QWidget()
        form = QFormLayout(general_page)
        self.language_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self.language_combo.addItem(name, code)
        self.language_combo.setCurrentIndex(
            max(0, self.language_combo.findData(language))
        )
        form.addRow("语言" if chinese else "Language", self.language_combo)
        form.addRow(QLabel("快捷键" if chinese else "Keyboard shortcuts"))
        labels = SHORTCUT_LABELS["zh_CN" if chinese else "en"]
        self.shortcut_edits: dict[str, QKeySequenceEdit] = {}
        for key in DEFAULT_SHORTCUTS:
            editor = QKeySequenceEdit(QKeySequence(shortcuts.get(key, DEFAULT_SHORTCUTS[key])))
            editor.setClearButtonEnabled(True)
            form.addRow(labels[key], editor)
            self.shortcut_edits[key] = editor
        self.tabs.addTab(general_page, "常规" if chinese else "General")

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

    def _restore_defaults(self) -> None:
        for key, value in DEFAULT_SHORTCUTS.items():
            self.shortcut_edits[key].setKeySequence(QKeySequence(value))
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
                "reset_view",
                "previous_time",
                "next_time",
            )
        )
        if len(assigned) != len(set(assigned)) or invalid_hold_key:
            QMessageBox.warning(
                self,
                "快捷键冲突" if self.language_combo.currentData() == "zh_CN" else "Shortcut conflict",
                "每个操作需要使用不同的单组快捷键。"
                if self.language_combo.currentData() == "zh_CN"
                else "Each action must use a different single-chord shortcut.",
            )
            return
        self.accept()

    def values(self) -> tuple[str, dict[str, str]]:
        return str(self.language_combo.currentData()), {
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
