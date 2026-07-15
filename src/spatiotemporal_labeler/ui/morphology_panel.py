from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MorphologyPanel(QWidget):
    applyRequested = Signal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()

        self.operation_label = QLabel()
        self.operation = QComboBox()
        self.operation.addItem("", "remove_small_components")
        self.operation.addItem("", "fill_holes")
        self.operation.addItem("", "opening")
        self.operation.addItem("", "closing")
        self.operation.currentIndexChanged.connect(self._update_parameter_visibility)
        form.addRow(self.operation_label, self.operation)

        self.labels_label = QLabel()
        self.labels_scope = QComboBox()
        self.labels_scope.addItem("", "selected")
        self.labels_scope.addItem("", "all")
        form.addRow(self.labels_label, self.labels_scope)

        self.frames_label = QLabel()
        self.frames_scope = QComboBox()
        self.frames_scope.addItem("", "current")
        self.frames_scope.addItem("", "all")
        form.addRow(self.frames_label, self.frames_scope)

        self.minimum_volume_label = QLabel()
        self.minimum_volume = QDoubleSpinBox()
        self.minimum_volume.setRange(0.001, 1_000_000_000.0)
        self.minimum_volume.setDecimals(3)
        self.minimum_volume.setValue(100.0)
        self.minimum_volume.setSuffix(" mm\u00b3")
        form.addRow(self.minimum_volume_label, self.minimum_volume)

        self.connectivity_label = QLabel()
        self.connectivity = QComboBox()
        self.connectivity.addItem("6", 1)
        self.connectivity.addItem("18", 2)
        self.connectivity.addItem("26", 3)
        form.addRow(self.connectivity_label, self.connectivity)

        self.radius_label = QLabel()
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.1, 100.0)
        self.radius.setDecimals(2)
        self.radius.setValue(1.0)
        self.radius.setSingleStep(0.5)
        self.radius.setSuffix(" mm")
        form.addRow(self.radius_label, self.radius)

        layout.addLayout(form)
        self.apply_button = QPushButton()
        self.apply_button.clicked.connect(self.applyRequested)
        layout.addWidget(self.apply_button)
        layout.addStretch()
        self.set_language("en")
        self._update_parameter_visibility()

    def _update_parameter_visibility(self, _index: int = 0) -> None:
        operation = str(self.operation.currentData())
        removes_components = operation == "remove_small_components"
        uses_connectivity = operation in {"remove_small_components", "fill_holes"}
        uses_radius = operation in {"opening", "closing"}
        self.minimum_volume_label.setVisible(removes_components)
        self.minimum_volume.setVisible(removes_components)
        self.connectivity_label.setVisible(uses_connectivity)
        self.connectivity.setVisible(uses_connectivity)
        self.radius_label.setVisible(uses_radius)
        self.radius.setVisible(uses_radius)

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.operation_label.setText("操作" if chinese else "Operation")
        self.labels_label.setText("作用标签" if chinese else "Labels")
        self.frames_label.setText("作用帧" if chinese else "Frames")
        self.minimum_volume_label.setText(
            "最小连通域体积" if chinese else "Minimum component volume"
        )
        self.connectivity_label.setText("邻接方式" if chinese else "Connectivity")
        self.radius_label.setText("物理半径" if chinese else "Physical radius")
        operation_names = (
            ("去除小连通域", "Remove small components"),
            ("填充孔洞", "Fill holes"),
            ("开运算", "Opening"),
            ("闭运算", "Closing"),
        )
        for index, (zh_text, en_text) in enumerate(operation_names):
            self.operation.setItemText(index, zh_text if chinese else en_text)
        self.labels_scope.setItemText(0, "选中标签" if chinese else "Selected label")
        self.labels_scope.setItemText(1, "所有标签" if chinese else "All labels")
        self.frames_scope.setItemText(0, "当前帧" if chinese else "Current frame")
        self.frames_scope.setItemText(1, "所有帧" if chinese else "All frames")
        self.apply_button.setText("应用" if chinese else "Apply")
