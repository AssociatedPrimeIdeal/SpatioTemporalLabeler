from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QTimer, Qt, Signal
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
SPATIAL_PLANES = {"X-Y": (0, 1), "X-Z": (0, 2), "Y-Z": (1, 2)}
TEMPORAL_PLANES = {"X-T": 0, "Y-T": 1, "Z-T": 2}


class ThumbnailImageItem(pg.ImageItem):
    windowLevelRequested = Signal(float, float)

    def mouseDragEvent(self, event: Any) -> None:  # noqa: N802 - pyqtgraph API
        if event.button() != Qt.MouseButton.MiddleButton:
            super().mouseDragEvent(event)
            return
        if not event.isStart() and not event.isFinish():
            current = event.scenePos() if hasattr(event, "scenePos") else event.pos()
            previous = (
                event.lastScenePos() if hasattr(event, "lastScenePos") else event.lastPos()
            )
            delta = current - previous
            self.windowLevelRequested.emit(float(delta.x()), float(delta.y()))
        event.accept()


class ThumbnailPlot(pg.PlotWidget):
    activated = Signal(int)
    levelsChanged = Signal(int, float, float)

    def __init__(
        self,
        index: int,
        initial_levels: tuple[float, float] | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent=parent, background="#101719")
        self.index = index
        self._geometry_signature: tuple[object, ...] | None = None
        self._default_levels: tuple[float, float] | None = None
        self._levels = initial_levels
        self._value_range = (0.0, 1.0)
        self.item = ThumbnailImageItem()
        self.item.windowLevelRequested.connect(self._adjust_window_level)
        self.addItem(self.item)
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setMouseEnabled(x=False, y=False)
        self.setMinimumSize(130, 90)
        self.setFixedHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.level_feedback = QLabel(self.viewport())
        self.level_feedback.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.level_feedback.setStyleSheet(
            "color: #f4f8f8; background: rgba(8, 18, 20, 205); "
            "border: 1px solid #668084; border-radius: 2px; padding: 2px 4px;"
        )
        self.level_feedback.hide()
        self._feedback_timer = QTimer(self)
        self._feedback_timer.setSingleShot(True)
        self._feedback_timer.setInterval(900)
        self._feedback_timer.timeout.connect(self.level_feedback.hide)

    def set_image(
        self,
        image: np.ndarray,
        default_levels: tuple[float, float],
        value_range: tuple[float, float],
        plane: str,
        spacing: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self._default_levels = default_levels
        self._value_range = (
            min(value_range[0], default_levels[0]),
            max(value_range[1], default_levels[1]),
        )
        if self._levels is None:
            self._levels = default_levels
        self.item.setImage(np.asarray(image).T, autoLevels=False, levels=self._levels)
        horizontal_spacing, vertical_spacing = map(float, spacing)
        width, height = image.shape
        self.item.setRect(
            QRectF(
                -0.5 * horizontal_spacing,
                -0.5 * vertical_spacing,
                width * horizontal_spacing,
                height * vertical_spacing,
            )
        )
        # Main spatial views are RAS-oriented and invert X.  The temporal
        # view does the same only for X-T, so previews must use the same map.
        invert_x = plane in SPATIAL_PLANES or plane == "X-T"
        signature = (
            *image.shape,
            plane,
            horizontal_spacing,
            vertical_spacing,
        )
        view_box = self.getViewBox()
        view_box.invertX(invert_x)
        view_box.setAspectLocked(plane in SPATIAL_PLANES)
        if signature != self._geometry_signature:
            self._geometry_signature = signature
            self.autoRange(padding=0.02)

    @property
    def levels(self) -> tuple[float, float] | None:
        return self._levels

    def _adjust_window_level(self, delta_width: float, delta_level: float) -> None:
        if self._levels is None:
            return
        data_low, data_high = self._value_range
        span = max(float(data_high - data_low), 1e-9)
        low, high = self._levels
        center = (low + high) * 0.5 - float(delta_level) * span / 200.0
        width = high - low + float(delta_width) * span / 200.0
        center = float(np.clip(center, data_low, data_high))
        width = float(np.clip(width, max(span / 1000.0, 1e-9), span * 2.0))
        self._levels = (center - width * 0.5, center + width * 0.5)
        self.item.setLevels(self._levels)
        self.levelsChanged.emit(self.index, *self._levels)
        self._show_level_feedback()

    def reset_display(self) -> bool:
        if self._default_levels is None:
            return False
        self._levels = self._default_levels
        self.item.setLevels(self._levels)
        self.autoRange(padding=0.02)
        self.levelsChanged.emit(self.index, *self._levels)
        self._show_level_feedback()
        return True

    def _show_level_feedback(self) -> None:
        if self._levels is None:
            return
        low, high = self._levels
        self.level_feedback.setText(
            f"WL {(low + high) * 0.5:.5g}   WW {high - low:.5g}"
        )
        self.level_feedback.adjustSize()
        self._position_level_feedback()
        self.level_feedback.show()
        self.level_feedback.raise_()
        self._feedback_timer.start()

    def _position_level_feedback(self) -> None:
        viewport = self.viewport()
        self.level_feedback.move(
            6,
            max(6, viewport.height() - self.level_feedback.height() - 6),
        )

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if "level_feedback" in self.__dict__:
            self._position_level_feedback()

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
            return
        factor = 0.82 if event.angleDelta().y() > 0 else 1.0 / 0.82
        scene_position = self.mapToScene(event.position().toPoint())
        center = self.getViewBox().mapSceneToView(scene_position)
        self.getViewBox().scaleBy((factor, factor), center=center)
        event.accept()

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.index)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ImagePreviewStrip(QFrame):
    imageActivated = Signal(int)
    imageLevelsChanged = Signal(int, float, float)
    collapsedChanged = Signal(bool)
    planeChanged = Signal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("imagePreviewStrip")
        self._collapsed = False
        self._language = "en"
        self._plots: dict[int, ThumbnailPlot] = {}
        self._cards: dict[int, QWidget] = {}
        self._value_ranges: dict[int, tuple[float, float]] = {}
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
            self.setMaximumWidth(420)
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

    def rebuild(
        self,
        images: list[Sequence4D],
        active_index: int,
        levels_by_image: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        while self.items_layout.count():
            item = self.items_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._plots.clear()
        self._cards.clear()
        self._value_ranges = {
            index: self._sequence_value_range(sequence)
            for index, sequence in enumerate(images)
            if index != active_index
        }
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
            initial_levels = (
                levels_by_image.get(id(sequence)) if levels_by_image is not None else None
            )
            plot = ThumbnailPlot(index, initial_levels=initial_levels)
            plot.activated.connect(self.imageActivated)
            plot.levelsChanged.connect(self.imageLevelsChanged.emit)
            card_layout.addWidget(plot)
            self.items_layout.addWidget(card)
            self._plots[index] = plot
            self._cards[index] = card
        self.setVisible(bool(self._plots))

    def reset_hovered_preview(self, watched: object | None = None) -> bool:
        for plot in self._plots.values():
            watched_in_plot = watched is plot or watched is plot.viewport()
            if isinstance(watched, QWidget):
                watched_in_plot = watched_in_plot or plot.isAncestorOf(watched)
            if watched_in_plot or plot.underMouse() or plot.viewport().underMouse():
                return plot.reset_display()
        return False

    def update_images(
        self,
        images: list[Sequence4D],
        cursor: tuple[int, int, int, int],
        reference_image: Sequence4D | None = None,
    ) -> None:
        if self._collapsed:
            return
        for index, plot in self._plots.items():
            sequence = images[index]
            mapped_cursor = self._map_cursor(sequence, cursor, reference_image)
            image = self._extract_preview(sequence, mapped_cursor, self.plane)
            finite = image[np.isfinite(image)]
            if finite.size:
                low, high = np.percentile(finite, (1.0, 99.0))
                levels = (float(low), float(high if high != low else low + 1.0))
            else:
                levels = (0.0, 1.0)
            plot.set_image(
                image,
                levels,
                self._value_ranges[index],
                self.plane,
                self._plane_spacing(sequence, self.plane),
            )

    @staticmethod
    def _plane_spacing(sequence: Sequence4D, plane: str) -> tuple[float, float]:
        if plane in SPATIAL_PLANES:
            horizontal, vertical = SPATIAL_PLANES[plane]
            return sequence.spacing_xyz[horizontal], sequence.spacing_xyz[vertical]
        return sequence.spacing_xyz[TEMPORAL_PLANES[plane]], 1.0

    @staticmethod
    def _map_cursor(
        sequence: Sequence4D,
        cursor: tuple[int, int, int, int],
        reference_image: Sequence4D | None,
    ) -> tuple[int, int, int, int]:
        if reference_image is None or sequence is reference_image:
            return tuple(int(value) for value in cursor)
        reference_index = np.asarray(cursor[:3], dtype=float)
        reference_transform = reference_image.transform
        world = np.asarray(reference_transform.origin_ras, dtype=float) + (
            reference_transform.direction_ras
            @ (reference_index * np.asarray(reference_image.spacing_xyz, dtype=float))
        )
        target_transform = sequence.transform
        basis = target_transform.direction_ras @ np.diag(sequence.spacing_xyz)
        try:
            target_index = np.linalg.solve(
                basis, world - np.asarray(target_transform.origin_ras, dtype=float)
            )
            spatial = tuple(int(value) for value in np.rint(target_index))
        except np.linalg.LinAlgError:
            spatial = tuple(int(value) for value in cursor[:3])
        return (*spatial, int(cursor[3]))

    @staticmethod
    def _sequence_value_range(sequence: Sequence4D) -> tuple[float, float]:
        sample = sequence.data.flat[:: max(1, sequence.data.size // 500_000)]
        finite = sample[np.isfinite(sample)]
        if not finite.size:
            return (0.0, 1.0)
        low, high = float(finite.min()), float(finite.max())
        return (low, high if high != low else low + 1.0)

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
