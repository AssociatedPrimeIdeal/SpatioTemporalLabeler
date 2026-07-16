from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout, QWidget

from spatiotemporal_labeler.model import LabelDefinition


class LassoPanel(QWidget):
    """Options for immediate lasso erase and label replacement gestures."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()

        self.operation_label = QLabel()
        self.operation = QComboBox()
        self.operation.addItem("", "erase")
        self.operation.addItem("", "replace")
        self.operation.currentIndexChanged.connect(self._operation_changed)
        form.addRow(self.operation_label, self.operation)

        self.source_label = QLabel()
        self.source = QComboBox()
        self.source.addItem("", "selected")
        self.source.addItem("", "all")
        form.addRow(self.source_label, self.source)

        self.target_label = QLabel()
        self.target = QComboBox()
        form.addRow(self.target_label, self.target)

        layout.addLayout(form)
        layout.addStretch()
        self.set_language("en")
        self._operation_changed()

    def set_labels(
        self, definitions: dict[int, LabelDefinition], active_value: int
    ) -> None:
        previous = self.target.currentData()
        self.target.blockSignals(True)
        self.target.clear()
        for definition in sorted(definitions.values(), key=lambda item: item.value):
            swatch = QPixmap(14, 14)
            swatch.fill(QColor(*definition.color))
            self.target.addItem(
                QIcon(swatch),
                f"{definition.value}   {definition.name}",
                definition.value,
            )
        target = previous if previous in definitions else None
        if target is None:
            target = next((value for value in definitions if value != active_value), active_value)
        index = self.target.findData(target)
        self.target.setCurrentIndex(max(0, index))
        self.target.blockSignals(False)

    def source_value(self, active_value: int) -> int | None:
        return None if self.source.currentData() == "all" else int(active_value)

    @property
    def target_value(self) -> int:
        value = self.target.currentData()
        return int(value) if value is not None else 0

    @property
    def operation_key(self) -> str:
        return str(self.operation.currentData())

    def _operation_changed(self, _index: int = 0) -> None:
        replacing = self.operation_key == "replace"
        self.target_label.setVisible(replacing)
        self.target.setVisible(replacing)

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.operation_label.setText("操作" if chinese else "Operation")
        self.operation.setItemText(0, "擦除" if chinese else "Erase")
        self.operation.setItemText(1, "替换标签" if chinese else "Replace label")
        self.source_label.setText("来源标签" if chinese else "Source label")
        self.source.setItemText(0, "当前选中标签" if chinese else "Selected label")
        self.source.setItemText(1, "所有标签" if chinese else "All labels")
        self.target_label.setText("目标标签" if chinese else "Target label")
