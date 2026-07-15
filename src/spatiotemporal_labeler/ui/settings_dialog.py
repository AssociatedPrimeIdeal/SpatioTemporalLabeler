from __future__ import annotations

from typing import Any

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QKeySequenceEdit,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from spatiotemporal_labeler.i18n import LANGUAGES


DEFAULT_SHORTCUTS = {
    "brush": "B",
    "eraser": "E",
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


class SettingsDialog(QDialog):
    def __init__(
        self, language: str, shortcuts: dict[str, str], parent: Any = None
    ) -> None:
        super().__init__(parent)
        chinese = language == "zh_CN"
        self.setWindowTitle("设置" if chinese else "Settings")
        self.resize(520, 560)
        layout = QVBoxLayout(self)
        form = QFormLayout()
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
        layout.addLayout(form)
        layout.addStretch()
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
        layout.addWidget(buttons)

    def _restore_defaults(self) -> None:
        for key, value in DEFAULT_SHORTCUTS.items():
            self.shortcut_edits[key].setKeySequence(QKeySequence(value))

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
