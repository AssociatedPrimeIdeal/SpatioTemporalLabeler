from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class ImportSelectionDialog(QDialog):
    """Choose which files to load and how each file is interpreted."""

    def __init__(
        self,
        paths: list[Path],
        inferred_masks: set[Path],
        language: str = "en",
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.paths = paths
        chinese = language == "zh_CN"
        self.setWindowTitle("导入文件" if chinese else "Import files")
        self.resize(720, min(560, 150 + 42 * len(paths)))
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "勾选需要加载的文件，并指定其类型。"
                if chinese
                else "Select files to load and classify each as an image or label sequence."
            )
        )
        self.table = QTableWidget(len(paths), 3)
        self.table.setHorizontalHeaderLabels(
            ["加载" if chinese else "Load", "文件" if chinese else "File", "类型" if chinese else "Type"]
        )
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._checks: list[QCheckBox] = []
        self._types: list[QComboBox] = []
        for row, path in enumerate(paths):
            check = QCheckBox()
            check.setChecked(True)
            check.setStyleSheet("margin-left: 12px")
            self.table.setCellWidget(row, 0, check)
            item = QTableWidgetItem(str(path))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(str(path))
            self.table.setItem(row, 1, item)
            kind = QComboBox()
            kind.addItem("图像" if chinese else "Image", False)
            kind.addItem("标签序列" if chinese else "Label sequence", True)
            kind.setCurrentIndex(1 if path in inferred_masks else 0)
            self.table.setCellWidget(row, 2, kind)
            self._checks.append(check)
            self._types.append(kind)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selections(self) -> list[tuple[Path, bool]]:
        return [
            (path, bool(kind.currentData()))
            for path, check, kind in zip(self.paths, self._checks, self._types)
            if check.isChecked()
        ]
