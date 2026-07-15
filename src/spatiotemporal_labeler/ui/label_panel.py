from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spatiotemporal_labeler.i18n import translate
from spatiotemporal_labeler.model import LabelDefinition


class LabelPanel(QWidget):
    labelSelected = Signal(int)
    visibilityChanged = Signal(int, bool)
    addRequested = Signal()
    deleteRequested = Signal(int)
    renameRequested = Signal(int)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.language = "en"
        self.setObjectName("labelPanel")
        self.setMinimumWidth(170)
        self.setMaximumWidth(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.title = QLabel()
        self.title.setObjectName("sectionTitle")
        layout.addWidget(self.title)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.currentItemChanged.connect(self._selection_changed)
        self.list_widget.itemChanged.connect(self._item_changed)
        self.list_widget.itemDoubleClicked.connect(self._rename_requested)
        layout.addWidget(self.list_widget, 1)
        controls = QHBoxLayout()
        self.add_button = QToolButton()
        self.add_button.setIcon(
            QIcon.fromTheme(
                "list-add", self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder)
            )
        )
        self.add_button.setIconSize(QSize(16, 16))
        self.add_button.clicked.connect(self.addRequested)
        controls.addWidget(self.add_button)
        self.delete_button = QToolButton()
        self.delete_button.setIcon(
            QIcon.fromTheme("edit-delete", self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        )
        self.delete_button.setIconSize(QSize(16, 16))
        self.delete_button.clicked.connect(self._delete_requested)
        controls.addWidget(self.delete_button)
        controls.addStretch()
        layout.addLayout(controls)
        self.set_language("en")

    def set_language(self, language: str) -> None:
        self.language = language
        self.title.setText(translate(language, "labels"))
        self.add_button.setToolTip(translate(language, "add_label"))
        self.delete_button.setToolTip(translate(language, "delete_label"))

    def set_labels(
        self, definitions: dict[int, LabelDefinition], selected_value: int | None = None
    ) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        selected_item = None
        for definition in sorted(definitions.values(), key=lambda item: item.value):
            item = QListWidgetItem(f"{definition.value}   {definition.name}")
            item.setData(Qt.ItemDataRole.UserRole, definition.value)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(
                Qt.CheckState.Checked if definition.visible else Qt.CheckState.Unchecked
            )
            swatch = QPixmap(14, 14)
            swatch.fill(QColor(*definition.color))
            item.setIcon(QIcon(swatch))
            self.list_widget.addItem(item)
            if definition.value == selected_value:
                selected_item = item
        if selected_item is None and self.list_widget.count():
            selected_item = self.list_widget.item(0)
        self.list_widget.setCurrentItem(selected_item)
        self.list_widget.blockSignals(False)
        if selected_item is not None:
            self.labelSelected.emit(int(selected_item.data(Qt.ItemDataRole.UserRole)))

    def _selection_changed(self, current: QListWidgetItem | None, _previous: Any) -> None:
        if current is not None:
            self.labelSelected.emit(int(current.data(Qt.ItemDataRole.UserRole)))

    def select_label(self, value: int) -> bool:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == value:
                self.list_widget.setCurrentItem(item)
                return True
        return False

    def _item_changed(self, item: QListWidgetItem) -> None:
        value = int(item.data(Qt.ItemDataRole.UserRole))
        self.visibilityChanged.emit(value, item.checkState() == Qt.CheckState.Checked)

    def _delete_requested(self) -> None:
        item = self.list_widget.currentItem()
        if item is not None:
            self.deleteRequested.emit(int(item.data(Qt.ItemDataRole.UserRole)))

    def _rename_requested(self, item: QListWidgetItem) -> None:
        self.renameRequested.emit(int(item.data(Qt.ItemDataRole.UserRole)))
