from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spatiotemporal_labeler.io import Sequence4D


PREVIEW_PLANES = ("X-Y", "X-Z", "Y-Z", "X-T", "Y-T", "Z-T")


class ThumbnailPlot(pg.PlotWidget):
    activated = Signal(int)

    def __init__(self, index: int, parent: Any = None) -> None:
        super().__init__(parent=parent, background="#101719")
        self.index = index
        self._geometry_signature: tuple[int, int, str] | None = None
        self.item = pg.ImageItem()
        self.addItem(self.item)
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setMouseEnabled(x=False, y=False)
        self.setMinimumSize(130, 90)
        self.setFixedHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_image(
        self,
        image: np.ndarray,
        levels: tuple[float, float],
        plane: str,
    ) -> None:
        self.item.setImage(np.asarray(image).T, autoLevels=False, levels=levels)
        invert_x = plane == "Y-Z"
        signature = (*image.shape, plane)
        view_box = self.getViewBox()
        view_box.invertX(invert_x)
        view_box.setAspectLocked(True)
        if signature != self._geometry_signature:
            self._geometry_signature = signature
            self.autoRange(padding=0.02)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
            return
        factor = 0.82 if event.angleDelta().y() > 0 else 1.0 / 0.82
        scene_position = self.mapToScene(event.position().toPoint())
        center = self.getViewBox().mapSceneToView(scene_position)
        self.getViewBox().scaleBy((factor, factor), center=center)
        event.accept()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.index)
            event.accept()
            return
        super().mousePressEvent(event)


class ImagePreviewStrip(QFrame):
    imageActivated = Signal(int)
    collapsedChanged = Signal(bool)
    planeChanged = Signal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("imagePreviewStrip")
        self._collapsed = False
        self._language = "en"
        self._plots: dict[int, ThumbnailPlot] = {}
        self._cards: dict[int, QWidget] = {}
        self.layout_root = QVBoxLayout(self)
        self.layout_root.setContentsMargins(5, 5, 5, 5)
        self.layout_root.setSpacing(5)
        header = QHBoxLayout()
        self.title = QLabel("Other images")
        header.addWidget(self.title)
        header.addStretch()
        self.plane_selector = QComboBox()
        self.plane_selector.addItems(PREVIEW_PLANES)
        self.plane_selector.currentTextChanged.connect(self.planeChanged.emit)
        header.addWidget(self.plane_selector)
        self.collapse_button = QToolButton()
        self.collapse_button.clicked.connect(self.toggle_collapsed)
        header.addWidget(self.collapse_button)
        self.layout_root.addLayout(header)
        self.content = QWidget()
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(5)
        self.items_layout = QVBoxLayout()
        self.items_layout.setSpacing(5)
        content_layout.addLayout(self.items_layout)
        content_layout.addStretch()
        self.layout_root.addWidget(self.content, 1)
        self.set_collapsed(False, emit=False)

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @property
    def plane(self) -> str:
        return self.plane_selector.currentText()

    def set_plane(self, plane: str, emit: bool = True) -> None:
        if plane not in PREVIEW_PLANES or plane == self.plane:
            return
        signals_were_blocked = self.plane_selector.blockSignals(not emit)
        self.plane_selector.setCurrentText(plane)
        self.plane_selector.blockSignals(signals_were_blocked)

    def set_language(self, language: str) -> None:
        self._language = language
        self.title.setText("其他图像" if language == "zh_CN" else "Other images")
        self.plane_selector.setToolTip("预览平面" if language == "zh_CN" else "Preview plane")
        self._update_collapse_button()

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool, emit: bool = True) -> None:
        collapsed = bool(collapsed)
        changed = collapsed != self._collapsed
        self._collapsed = collapsed
        self.title.setVisible(not collapsed)
        self.plane_selector.setVisible(not collapsed)
        self.content.setVisible(not collapsed)
        if collapsed:
            self.setMinimumWidth(32)
            self.setMaximumWidth(32)
            self.layout_root.setContentsMargins(2, 5, 2, 5)
        else:
            self.setMinimumWidth(155)
            self.setMaximumWidth(210)
            self.layout_root.setContentsMargins(5, 5, 5, 5)
        self._update_collapse_button()
        if changed and emit:
            self.collapsedChanged.emit(collapsed)

    def _update_collapse_button(self) -> None:
        icon = (
            QStyle.StandardPixmap.SP_ArrowRight
            if self._collapsed
            else QStyle.StandardPixmap.SP_ArrowLeft
        )
        self.collapse_button.setIcon(self.style().standardIcon(icon))
        chinese = self._language == "zh_CN"
        if self._collapsed:
            self.collapse_button.setToolTip("展开其他图像" if chinese else "Expand other images")
        else:
            self.collapse_button.setToolTip("收起其他图像" if chinese else "Collapse other images")

    def rebuild(self, images: list[Sequence4D], active_index: int) -> None:
        while self.items_layout.count():
            item = self.items_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._plots.clear()
        self._cards.clear()
        for index, sequence in enumerate(images):
            if index == active_index:
                continue
            card = QWidget()
            card.setMaximumHeight(135)
            card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(0, 0, 0, 0)
            card_layout.setSpacing(2)
            name = QLabel(sequence.display_name)
            name.setToolTip(sequence.display_name)
            card_layout.addWidget(name)
            plot = ThumbnailPlot(index)
            plot.activated.connect(self.imageActivated)
            card_layout.addWidget(plot)
            self.items_layout.addWidget(card)
            self._plots[index] = plot
            self._cards[index] = card
        self.setVisible(bool(self._plots))

    def update_images(
        self,
        images: list[Sequence4D],
        cursor: tuple[int, int, int, int],
    ) -> None:
        if self._collapsed:
            return
        for index, plot in self._plots.items():
            sequence = images[index]
            image = self._extract_preview(sequence, cursor, self.plane)
            finite = image[np.isfinite(image)]
            if finite.size:
                low, high = np.percentile(finite, (1.0, 99.0))
                levels = (float(low), float(high if high != low else low + 1.0))
            else:
                levels = (0.0, 1.0)
            plot.set_image(image, levels, self.plane)

    @staticmethod
    def _extract_preview(
        sequence: Sequence4D,
        cursor: tuple[int, int, int, int],
        plane: str,
    ) -> np.ndarray:
        x, y, z, t = (
            int(np.clip(cursor[axis], 0, sequence.data.shape[axis] - 1))
            for axis in range(4)
        )
        if plane == "X-Y":
            return sequence.data[:, :, z, t]
        if plane == "X-Z":
            return sequence.data[:, y, :, t]
        if plane == "Y-Z":
            return sequence.data[x, :, :, t]
        if plane == "X-T":
            return sequence.data[:, y, z, :]
        if plane == "Y-T":
            return sequence.data[x, :, z, :]
        return sequence.data[x, y, :, :]
