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
    QMenu,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from spatiotemporal_labeler.i18n import translate
from spatiotemporal_labeler.model import LabelDefinition


THRESHOLD_MASK_ITEM = "threshold_mask"
OPACITY_ROLE = int(Qt.ItemDataRole.UserRole) + 1
THRESHOLD_MASK_COLOR = (35, 210, 190)


class LabelPanel(QWidget):
    labelSelected = Signal(int)
    visibilityChanged = Signal(int, bool)
    addRequested = Signal()
    deleteRequested = Signal(int)
    renameRequested = Signal(int)
    colorRequested = Signal(int)
    opacityChanged = Signal(int, float)
    thresholdVisibilityChanged = Signal(bool)
    thresholdDeleteRequested = Signal()
    thresholdOpacityChanged = Signal(float)
    globalOpacityChanged = Signal(float)

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
        self.list_widget.itemDoubleClicked.connect(self._color_requested)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_label_menu)
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

        self.global_opacity_label = QLabel()
        layout.addWidget(self.global_opacity_label)
        opacity_row = QHBoxLayout()
        self.global_opacity = QSlider(Qt.Orientation.Horizontal)
        self.global_opacity.setRange(0, 100)
        self.global_opacity.setValue(100)
        self.global_opacity.valueChanged.connect(
            lambda value: self.globalOpacityChanged.emit(value / 100.0)
        )
        opacity_row.addWidget(self.global_opacity, 1)
        self.global_opacity_value = QLabel("100%")
        self.global_opacity.setMinimumWidth(70)
        self.global_opacity.valueChanged.connect(
            lambda value: self.global_opacity_value.setText(f"{value}%")
        )
        opacity_row.addWidget(self.global_opacity_value)
        layout.addLayout(opacity_row)
        self.set_language("en")

    def set_language(self, language: str) -> None:
        self.language = language
        self.title.setText(translate(language, "labels"))
        self.add_button.setToolTip(translate(language, "add_label"))
        self.delete_button.setToolTip(translate(language, "delete_label"))
        self.global_opacity_label.setText(
            "全部标签透明度" if language == "zh_CN" else "All labels opacity"
        )

    def set_labels(
        self,
        definitions: dict[int, LabelDefinition],
        selected_value: int | None = None,
        *,
        threshold_present: bool = False,
        threshold_visible: bool = True,
        threshold_opacity: float = 1.0,
    ) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        selected_item = None
        for definition in sorted(definitions.values(), key=lambda item: item.value):
            item = QListWidgetItem(f"{definition.value}   {definition.name}")
            item.setData(Qt.ItemDataRole.UserRole, definition.value)
            item.setData(OPACITY_ROLE, float(definition.opacity))
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
        if threshold_present:
            item = QListWidgetItem(
                "阈值蒙版" if self.language == "zh_CN" else "Threshold mask"
            )
            item.setData(Qt.ItemDataRole.UserRole, THRESHOLD_MASK_ITEM)
            item.setData(OPACITY_ROLE, float(threshold_opacity))
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(
                Qt.CheckState.Checked if threshold_visible else Qt.CheckState.Unchecked
            )
            swatch = QPixmap(14, 14)
            swatch.fill(QColor(*THRESHOLD_MASK_COLOR))
            item.setIcon(QIcon(swatch))
            self.list_widget.addItem(item)
        if selected_item is None and self.list_widget.count():
            selected_item = self.list_widget.item(0)
        self.list_widget.setCurrentItem(selected_item)
        self.list_widget.blockSignals(False)
        if selected_item is not None:
            self.labelSelected.emit(int(selected_item.data(Qt.ItemDataRole.UserRole)))

    def _selection_changed(self, current: QListWidgetItem | None, _previous: Any) -> None:
        if current is not None:
            value = current.data(Qt.ItemDataRole.UserRole)
            if value != THRESHOLD_MASK_ITEM:
                self.labelSelected.emit(int(value))

    def select_label(self, value: int) -> bool:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            item_value = item.data(Qt.ItemDataRole.UserRole)
            if item_value != THRESHOLD_MASK_ITEM and int(item_value) == value:
                self.list_widget.setCurrentItem(item)
                return True
        return False

    def _item_changed(self, item: QListWidgetItem) -> None:
        value = item.data(Qt.ItemDataRole.UserRole)
        visible = item.checkState() == Qt.CheckState.Checked
        if value == THRESHOLD_MASK_ITEM:
            self.thresholdVisibilityChanged.emit(visible)
        else:
            self.visibilityChanged.emit(int(value), visible)

    def _delete_requested(self) -> None:
        item = self.list_widget.currentItem()
        if item is not None:
            value = item.data(Qt.ItemDataRole.UserRole)
            if value == THRESHOLD_MASK_ITEM:
                self.thresholdDeleteRequested.emit()
            else:
                self.deleteRequested.emit(int(value))

    def _color_requested(self, item: QListWidgetItem) -> None:
        value = item.data(Qt.ItemDataRole.UserRole)
        if value != THRESHOLD_MASK_ITEM:
            self.colorRequested.emit(int(value))

    def set_global_opacity(self, opacity: float) -> None:
        self.global_opacity.blockSignals(True)
        self.global_opacity.setValue(int(round(100.0 * float(opacity))))
        self.global_opacity.blockSignals(False)
        self.global_opacity_value.setText(f"{self.global_opacity.value()}%")

    def _show_label_menu(self, position: Any) -> None:
        item = self.list_widget.itemAt(position)
        if item is None:
            return
        value = item.data(Qt.ItemDataRole.UserRole)
        opacity = float(item.data(OPACITY_ROLE) or 0.0)
        menu = QMenu(self)
        if value != THRESHOLD_MASK_ITEM:
            rename_action = menu.addAction(translate(self.language, "rename_label"))
            rename_action.triggered.connect(
                lambda: self.renameRequested.emit(int(value))
            )
            menu.addSeparator()
        action = QWidgetAction(menu)
        control = QWidget(menu)
        layout = QVBoxLayout(control)
        layout.setContentsMargins(10, 7, 10, 7)
        label = QLabel(control)
        slider = QSlider(Qt.Orientation.Horizontal, control)
        slider.setRange(0, 100)
        slider.setValue(int(round(opacity * 100.0)))
        slider.setMinimumWidth(150)

        def opacity_changed(percent: int) -> None:
            item.setData(OPACITY_ROLE, percent / 100.0)
            label.setText(
                ("透明度" if self.language == "zh_CN" else "Opacity")
                + f"  {percent}%"
            )
            if value == THRESHOLD_MASK_ITEM:
                self.thresholdOpacityChanged.emit(percent / 100.0)
            else:
                self.opacityChanged.emit(int(value), percent / 100.0)

        label.setText(
            ("透明度" if self.language == "zh_CN" else "Opacity")
            + f"  {slider.value()}%"
        )
        slider.valueChanged.connect(opacity_changed)
        layout.addWidget(label)
        layout.addWidget(slider)
        action.setDefaultWidget(control)
        menu.addAction(action)
        menu.exec(self.list_widget.viewport().mapToGlobal(position))
