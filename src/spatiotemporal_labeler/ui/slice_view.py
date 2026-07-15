from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsRectItem

from spatiotemporal_labeler.model import LabelDefinition, default_label

pg.setConfigOption("imageAxisOrder", "row-major")


PLANE_DIRECTIONS = {
    "X-Y": ("L - R", "P - A"),
    "X-Z": ("L - R", "F - H"),
    "Y-Z": ("P - A", "F - H"),
}
TEMPORAL_DIRECTIONS = {"X-T": "L - R", "Y-T": "P - A", "Z-T": "F - H"}
AXIS_COLORS = {"X": "#ff5f5f", "Y": "#58d178", "Z": "#579dff", "T": "#f1cc4b"}


class FootprintOverlay:
    """Exact plot-space preview of the active brush, eraser, or contour pixel."""

    def __init__(self, view: pg.PlotWidget) -> None:
        self.view = view
        self.diameter_mm = 6.0
        self.shape = "round"
        self.tool = "brush"
        self.display_spacing = (1.0, 1.0)
        self.edit_spacing = (1.0, 1.0)
        self.position = (0, 0)
        self.hovered = False
        self.temporary_erase = False
        self.ellipse = QGraphicsEllipseItem()
        self.rectangle = QGraphicsRectItem()
        for item in (self.ellipse, self.rectangle):
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setZValue(60)
            item.hide()
            view.addItem(item)

    def configure(self, diameter_mm: float, shape: str, tool: str) -> None:
        self.diameter_mm = diameter_mm
        self.shape = shape
        self.tool = tool
        self._refresh()

    def set_spacing(
        self,
        display_spacing: tuple[float, float],
        edit_spacing: tuple[float, float],
    ) -> None:
        self.display_spacing = display_spacing
        self.edit_spacing = edit_spacing
        self._refresh()

    def show_at(self, h: int, v: int, visible: bool = True) -> None:
        self.position = (h, v)
        self.hovered = visible
        self._refresh()

    def set_temporary_erase(self, enabled: bool) -> None:
        self.temporary_erase = enabled
        self._refresh()

    def _refresh(self) -> None:
        self.ellipse.hide()
        self.rectangle.hide()
        if not self.hovered:
            return
        h, v = self.position
        center_h = h * self.display_spacing[0]
        center_v = v * self.display_spacing[1]
        effective_tool = "eraser" if self.temporary_erase else self.tool
        color = {
            "brush": "#39d5c5",
            "eraser": "#ff6b64",
            "contour": "#ffd166",
            "picker": "#bda7ff",
            "grow": "#69dc93",
        }[effective_tool]
        pen = pg.mkPen(color, width=1.7)
        brush = pg.mkBrush((*pg.mkColor(color).getRgb()[:3], 28))
        if effective_tool in {"contour", "picker", "grow"}:
            width, height = self.display_spacing
            self.rectangle.setRect(
                QRectF(center_h - width / 2.0, center_v - height / 2.0, width, height)
            )
            self.rectangle.setPen(pen)
            self.rectangle.setBrush(brush)
            self.rectangle.show()
            return
        radius_h = (
            self.diameter_mm / 2.0 / self.edit_spacing[0] * self.display_spacing[0]
        )
        radius_v = (
            self.diameter_mm / 2.0 / self.edit_spacing[1] * self.display_spacing[1]
        )
        bounds = QRectF(
            center_h - radius_h,
            center_v - radius_v,
            radius_h * 2.0,
            radius_v * 2.0,
        )
        item = self.ellipse if self.shape == "round" else self.rectangle
        item.setRect(bounds)
        item.setPen(pen)
        item.setBrush(brush)
        item.show()


def label_overlay(
    mask: np.ndarray,
    definitions: dict[int, LabelDefinition] | None = None,
    alpha: int = 124,
) -> np.ndarray:
    labels = np.asarray(mask)
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    if definitions is None:
        definitions = {
            int(value): default_label(int(value)) for value in np.unique(labels) if int(value) > 0
        }
    for definition in definitions.values():
        if not definition.visible:
            continue
        selected = labels == definition.value
        rgba[..., :3][selected] = definition.color
        rgba[..., 3][selected] = alpha
    return rgba


def threshold_overlay(selection: np.ndarray) -> np.ndarray:
    selected = np.asarray(selection, dtype=bool)
    rgba = np.zeros((*selected.shape, 4), dtype=np.uint8)
    rgba[..., :3][selected] = (35, 210, 190)
    rgba[..., 3][selected] = 72
    return rgba


class EditableImageItem(pg.ImageItem):
    strokeStarted = Signal(int, int, bool)
    strokeMoved = Signal(int, int, bool)
    strokeFinished = Signal(int, int, bool)
    navigateRequested = Signal(int, int)
    doubleClicked = Signal(int, int)
    hoverChanged = Signal(int, int, bool)
    panRequested = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptHoverEvents(True)
        self._single_click: tuple[int, int] | None = None
        self._single_click_timer = QTimer()
        self._single_click_timer.setSingleShot(True)
        application = QApplication.instance()
        interval = application.doubleClickInterval() if application is not None else 250
        self._single_click_timer.setInterval(interval)
        self._single_click_timer.timeout.connect(self._emit_single_click)

    def _voxel(self, position: QPointF) -> tuple[int, int]:
        return int(np.floor(position.x())), int(np.floor(position.y()))

    def mouseDragEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        pan_requested = event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        )
        if pan_requested:
            if not event.isStart() and not event.isFinish():
                delta = event.pos() - event.lastPos()
                self.panRequested.emit(float(delta.x()), float(delta.y()))
            event.accept()
            return
        if event.button() not in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        }:
            event.ignore()
            return
        h, v = self._voxel(event.pos())
        temporary_erase = event.button() == Qt.MouseButton.RightButton
        if event.isStart():
            self.strokeStarted.emit(h, v, temporary_erase)
        elif event.isFinish():
            self.strokeFinished.emit(h, v, temporary_erase)
        else:
            self.strokeMoved.emit(h, v, temporary_erase)
        event.accept()

    def mouseClickEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        h, v = self._voxel(event.pos())
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.navigateRequested.emit(h, v)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and event.double():
            self._single_click_timer.stop()
            self._single_click = None
            self.doubleClicked.emit(h, v)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._single_click = (h, v)
            self._single_click_timer.start()
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.strokeStarted.emit(h, v, True)
            self.strokeFinished.emit(h, v, True)
            event.accept()
            return
        event.ignore()

    def _emit_single_click(self) -> None:
        if self._single_click is None:
            return
        h, v = self._single_click
        self._single_click = None
        self.strokeStarted.emit(h, v, False)
        self.strokeFinished.emit(h, v, False)

    def hoverEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if event.isExit():
            self.hoverChanged.emit(0, 0, False)
            return
        h, v = self._voxel(event.pos())
        self.hoverChanged.emit(h, v, True)
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.navigateRequested.emit(h, v)


class SliceView(pg.PlotWidget):
    strokeStarted = Signal(str, int, int, bool)
    strokeMoved = Signal(str, int, int, bool)
    strokeFinished = Signal(str, int, int, bool)
    navigateRequested = Signal(str, int, int)
    viewDoubleClicked = Signal(str)
    sliceStepRequested = Signal(str, int)
    brushSizeStepRequested = Signal(int)
    hoverMoved = Signal(str, int, int, bool)

    def __init__(self, plane: str, parent: Any = None) -> None:
        super().__init__(parent=parent, background="#101719")
        self.plane = plane
        self.spacing = (1.0, 1.0)
        self._data_rect: QRectF | None = None
        self._geometry_signature: tuple[float, ...] | None = None
        self.image_item = EditableImageItem()
        self.threshold_item = pg.ImageItem()
        self.mask_item = pg.ImageItem()
        self.contour_item = pg.PlotDataItem(
            pen=pg.mkPen("#ffe082", width=1.4),
            symbol="s",
            symbolSize=3,
            symbolPen=None,
            symbolBrush=pg.mkBrush("#ffe082"),
        )
        h_axis, v_axis, _ = {
            "X-Y": ("X", "Y", "Z"),
            "X-Z": ("X", "Z", "Y"),
            "Y-Z": ("Y", "Z", "X"),
        }[plane]
        self.crosshair_vertical = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(AXIS_COLORS[h_axis], width=1.3)
        )
        self.crosshair_horizontal = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(AXIS_COLORS[v_axis], width=1.3)
        )
        self.crosshair_vertical.setZValue(30)
        self.crosshair_horizontal.setZValue(30)
        self.threshold_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.mask_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.contour_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.threshold_item.setZValue(5)
        self.mask_item.setZValue(10)
        self.contour_item.setZValue(20)
        self.addItem(self.image_item)
        self.addItem(self.threshold_item)
        self.addItem(self.mask_item)
        self.addItem(self.contour_item)
        self.addItem(self.crosshair_vertical)
        self.addItem(self.crosshair_horizontal)
        self.footprint = FootprintOverlay(self)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setMouseEnabled(x=False, y=False)
        self.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.getPlotItem().setTitle(plane, color="#dce8e9", size="10pt")
        bottom, left = PLANE_DIRECTIONS[plane]
        self.getPlotItem().setLabel("bottom", bottom, color="#8ea4a8")
        self.getPlotItem().setLabel("left", left, color="#8ea4a8")
        self.getViewBox().invertY(False)
        self.image_item.strokeStarted.connect(self._stroke_started)
        self.image_item.strokeMoved.connect(self._stroke_moved)
        self.image_item.strokeFinished.connect(self._stroke_finished)
        self.image_item.navigateRequested.connect(
            lambda h, v: self.navigateRequested.emit(self.plane, h, v)
        )
        self.image_item.doubleClicked.connect(
            lambda _h, _v: self.viewDoubleClicked.emit(self.plane)
        )
        self.image_item.hoverChanged.connect(self._hover_changed)
        self.image_item.panRequested.connect(self._pan)

    def _hover_changed(self, h: int, v: int, visible: bool) -> None:
        self.footprint.show_at(h, v, visible)
        self.hoverMoved.emit(self.plane, h, v, visible)

    def _stroke_started(self, h: int, v: int, temporary_erase: bool) -> None:
        self.footprint.set_temporary_erase(temporary_erase)
        self.footprint.show_at(h, v)
        self.strokeStarted.emit(self.plane, h, v, temporary_erase)

    def _stroke_moved(self, h: int, v: int, temporary_erase: bool) -> None:
        self.footprint.show_at(h, v)
        self.strokeMoved.emit(self.plane, h, v, temporary_erase)

    def _stroke_finished(self, h: int, v: int, temporary_erase: bool) -> None:
        self.footprint.show_at(h, v)
        self.strokeFinished.emit(self.plane, h, v, temporary_erase)
        self.footprint.set_temporary_erase(False)

    def set_editing_footprint(self, diameter_mm: float, shape: str, tool: str) -> None:
        self.footprint.configure(diameter_mm, shape, tool)

    def set_slice(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        spacing: tuple[float, float],
        levels: tuple[float, float],
        fixed_axis: str,
        fixed_index: int,
        labels: dict[int, LabelDefinition] | None = None,
        cursor: tuple[int, int] | None = None,
        threshold: np.ndarray | None = None,
    ) -> None:
        self.spacing = spacing
        self.footprint.set_spacing(spacing, spacing)
        height, width = image.shape[1], image.shape[0]
        rect = QRectF(
            -0.5 * spacing[0],
            -0.5 * spacing[1],
            width * spacing[0],
            height * spacing[1],
        )
        self.image_item.setImage(np.asarray(image).T, autoLevels=False, levels=levels)
        self.image_item.setRect(rect)
        if threshold is None:
            self.threshold_item.clear()
        else:
            self.threshold_item.setImage(
                threshold_overlay(threshold).transpose(1, 0, 2), autoLevels=False
            )
            self.threshold_item.setRect(rect)
        if mask is None:
            self.set_mask_overlay(None, labels)
        else:
            self.set_mask_overlay(mask, labels, rect)
        self.getPlotItem().setTitle(
            f"{self.plane}  |  {fixed_axis} {fixed_index + 1}",
            color="#dce8e9",
            size="10pt",
        )
        signature = (float(width), float(height), *map(float, spacing))
        self._data_rect = rect
        if signature != self._geometry_signature:
            self._geometry_signature = signature
            self.reset_view()
        self.getViewBox().setAspectLocked(True)
        if cursor is not None:
            self.crosshair_vertical.setValue(cursor[0] * spacing[0])
            self.crosshair_horizontal.setValue(cursor[1] * spacing[1])

    def set_mask_overlay(
        self,
        mask: np.ndarray | None,
        labels: dict[int, LabelDefinition] | None = None,
        rect: QRectF | None = None,
    ) -> None:
        if mask is None:
            self.mask_item.clear()
            return
        self.mask_item.setImage(
            label_overlay(mask, labels).transpose(1, 0, 2), autoLevels=False
        )
        target_rect = rect if rect is not None else self._data_rect
        if target_rect is not None:
            self.mask_item.setRect(target_rect)

    def set_contour(self, points: list[tuple[int, int]]) -> None:
        if not points:
            self.contour_item.setData([], [])
            return
        x = [point[0] * self.spacing[0] for point in points]
        y = [point[1] * self.spacing[1] for point in points]
        self.contour_item.setData(x, y, connect="finite")

    def _pan(self, delta_h: float, delta_v: float) -> None:
        self.getViewBox().translateBy(
            x=-delta_h * self.spacing[0], y=-delta_v * self.spacing[1]
        )

    def reset_view(self) -> None:
        if self._data_rect is not None:
            self.setRange(self._data_rect, padding=0.02)
            self.getViewBox().setAspectLocked(True)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        delta = 1 if event.angleDelta().y() > 0 else -1
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 0.82 if delta > 0 else 1.0 / 0.82
            scene_position = self.mapToScene(event.position().toPoint())
            center = self.getViewBox().mapSceneToView(scene_position)
            self.getViewBox().scaleBy((factor, factor), center=center)
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.brushSizeStepRequested.emit(delta)
        else:
            self.sliceStepRequested.emit(self.plane, delta)
        event.accept()


class TemporalView(pg.PlotWidget):
    strokeStarted = Signal(str, int, int, bool)
    strokeMoved = Signal(str, int, int, bool)
    strokeFinished = Signal(str, int, int, bool)
    cursorRequested = Signal(str, int, int)
    viewDoubleClicked = Signal(str)
    brushSizeStepRequested = Signal(int)
    hoverMoved = Signal(str, int, int, bool)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent=parent, background="#101719")
        self.mode = "X-T"
        self.spacing = 1.0
        self._data_rect: QRectF | None = None
        self._geometry_signature: tuple[float, ...] | None = None
        self.image_item = EditableImageItem()
        self.threshold_item = pg.ImageItem()
        self.mask_item = pg.ImageItem()
        self.contour_item = pg.PlotDataItem(
            pen=pg.mkPen("#ffe082", width=1.4),
            symbol="s",
            symbolSize=3,
            symbolPen=None,
            symbolBrush=pg.mkBrush("#ffe082"),
        )
        self.threshold_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.mask_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.contour_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.threshold_item.setZValue(5)
        self.mask_item.setZValue(10)
        self.contour_item.setZValue(20)
        self.time_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(AXIS_COLORS["T"], width=1.3))
        self.spatial_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(AXIS_COLORS["X"], width=1.3)
        )
        self.time_line.setZValue(30)
        self.spatial_line.setZValue(30)
        self.addItem(self.image_item)
        self.addItem(self.threshold_item)
        self.addItem(self.mask_item)
        self.addItem(self.contour_item)
        self.addItem(self.time_line)
        self.addItem(self.spatial_line)
        self.footprint = FootprintOverlay(self)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setMouseEnabled(x=False, y=False)
        self.image_item.strokeStarted.connect(self._stroke_started)
        self.image_item.strokeMoved.connect(self._stroke_moved)
        self.image_item.strokeFinished.connect(self._stroke_finished)
        self.image_item.navigateRequested.connect(
            lambda h, v: self.cursorRequested.emit(self.mode, h, v)
        )
        self.image_item.doubleClicked.connect(
            lambda _h, _v: self.viewDoubleClicked.emit(self.mode)
        )
        self.image_item.hoverChanged.connect(self._hover_changed)
        self.image_item.panRequested.connect(self._pan)

    def _hover_changed(self, h: int, v: int, visible: bool) -> None:
        self.footprint.show_at(h, v, visible)
        self.hoverMoved.emit(self.mode, h, v, visible)

    def _stroke_started(self, h: int, v: int, temporary_erase: bool) -> None:
        self.footprint.set_temporary_erase(temporary_erase)
        self.footprint.show_at(h, v)
        self.strokeStarted.emit(self.mode, h, v, temporary_erase)

    def _stroke_moved(self, h: int, v: int, temporary_erase: bool) -> None:
        self.footprint.show_at(h, v)
        self.strokeMoved.emit(self.mode, h, v, temporary_erase)

    def _stroke_finished(self, h: int, v: int, temporary_erase: bool) -> None:
        self.footprint.show_at(h, v)
        self.strokeFinished.emit(self.mode, h, v, temporary_erase)
        self.footprint.set_temporary_erase(False)

    def set_editing_footprint(self, diameter_mm: float, shape: str, tool: str) -> None:
        self.footprint.configure(diameter_mm, shape, tool)

    def set_sequence_slice(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        mode: str,
        spacing: float,
        levels: tuple[float, float],
        time_index: int,
        fixed_text: str,
        labels: dict[int, LabelDefinition] | None = None,
        cursor: tuple[int, int] | None = None,
        threshold: np.ndarray | None = None,
    ) -> None:
        self.mode = mode
        self.spacing = spacing
        self.footprint.set_spacing((spacing, 1.0), (spacing, spacing))
        width, height = image.shape
        rect = QRectF(-0.5 * spacing, -0.5, width * spacing, float(height))
        self.image_item.setImage(image.T, autoLevels=False, levels=levels)
        self.image_item.setRect(rect)
        if threshold is None:
            self.threshold_item.clear()
        else:
            self.threshold_item.setImage(
                threshold_overlay(threshold).transpose(1, 0, 2), autoLevels=False
            )
            self.threshold_item.setRect(rect)
        if mask is None:
            self.set_mask_overlay(None, labels)
        else:
            self.set_mask_overlay(mask, labels, rect)
        self.time_line.setValue(time_index)
        if cursor is not None:
            self.spatial_line.setValue(cursor[0] * spacing)
            self.time_line.setValue(cursor[1])
        self.getPlotItem().setTitle(
            f"{mode}  |  {fixed_text}", color="#dce8e9", size="10pt"
        )
        self.getPlotItem().setLabel("bottom", TEMPORAL_DIRECTIONS[mode], color="#8ea4a8")
        self.getPlotItem().setLabel("left", "Time", color="#8ea4a8")
        axis = mode[0]
        self.spatial_line.setPen(pg.mkPen(AXIS_COLORS[axis], width=1.3))
        signature = (float(width), float(height), float(spacing))
        self._data_rect = rect
        if signature != self._geometry_signature:
            self._geometry_signature = signature
            self.reset_view()

    def set_mask_overlay(
        self,
        mask: np.ndarray | None,
        labels: dict[int, LabelDefinition] | None = None,
        rect: QRectF | None = None,
    ) -> None:
        if mask is None:
            self.mask_item.clear()
            return
        self.mask_item.setImage(
            label_overlay(mask, labels).transpose(1, 0, 2), autoLevels=False
        )
        target_rect = rect if rect is not None else self._data_rect
        if target_rect is not None:
            self.mask_item.setRect(target_rect)

    def set_contour(self, points: list[tuple[int, int]]) -> None:
        if not points:
            self.contour_item.setData([], [])
            return
        x = [point[0] * self.spacing for point in points]
        y = [point[1] for point in points]
        self.contour_item.setData(x, y, connect="finite")

    def _pan(self, delta_h: float, delta_v: float) -> None:
        self.getViewBox().translateBy(x=-delta_h * self.spacing, y=-delta_v)

    def reset_view(self) -> None:
        if self._data_rect is not None:
            self.setRange(self._data_rect, padding=0.02)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        delta = 1 if event.angleDelta().y() > 0 else -1
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 0.82 if delta > 0 else 1.0 / 0.82
            scene_position = self.mapToScene(event.position().toPoint())
            center = self.getViewBox().mapSceneToView(scene_position)
            self.getViewBox().scaleBy((factor, factor), center=center)
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.brushSizeStepRequested.emit(delta)
        event.accept()
