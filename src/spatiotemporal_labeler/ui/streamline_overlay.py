"""One passive graphics item for a cached, speed-colored slice projection."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPicture, QPolygonF
from PySide6.QtWidgets import QGraphicsObject

from spatiotemporal_labeler.tools.streamlines import speed_rgb


class StreamlineOverlay(QGraphicsObject):
    def __init__(self, plot) -> None:
        super().__init__()
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(17)
        self._picture = QPicture()
        self._bounds = QRectF()
        self._signature = None
        # Picture compilation is display work only. Keep recent slice pictures
        # so back-and-forth navigation only swaps an already painted object.
        self._pictures = OrderedDict()
        plot.addItem(self)

    def boundingRect(self):  # noqa: N802 - Qt API
        return self._bounds

    def paint(self, painter, *_args):
        painter.drawPicture(0, 0, self._picture)

    def clear(self) -> None:
        self.prepareGeometryChange()
        self._picture = QPicture()
        self._bounds = QRectF()
        self._signature = None
        self._pictures.clear()
        self.update()

    def set_segments(self, rows, *, key=None, width=1.5, opacity=0.9,
                     speed_max=100.0, arrows=True, min_speed=0.0) -> None:
        if rows is None:
            if self._signature is not None:
                self.clear()
            return
        signature = (key, width, opacity, speed_max, arrows, min_speed)
        if signature == self._signature:
            return
        self._signature = signature
        self.prepareGeometryChange()
        if signature in self._pictures:
            self._picture, self._bounds = self._pictures.pop(signature)
            self._pictures[signature] = (self._picture, self._bounds)
            self.update()
            return
        picture = QPicture()
        painter = QPainter(picture)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rows = np.asarray(rows)
        rows = rows[rows[:, 4] >= min_speed]
        if len(rows):
            bins = np.clip((rows[:, 4] / max(speed_max, 1e-6) * 15).astype(int), 0, 15)
            colors = speed_rgb(np.arange(16) / 15.0)
            for index, rgb in enumerate(colors):
                selected = rows[bins == index]
                if not len(selected):
                    continue
                color = QColor(*map(int, rgb))
                color.setAlphaF(float(opacity))
                pen = pg.mkPen(color, width=width)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawLines([QLineF(*map(float, row[:4])) for row in selected])
                if arrows:
                    painter.setBrush(color)
                    # Sparse arrowheads follow the actual local velocity,
                    # including the reversed upstream half of each trajectory.
                    for row in selected[::24]:
                        start, end = row[:2], row[2:4]
                        vector = end - start
                        length = float(np.linalg.norm(vector))
                        if length < 1e-6:
                            continue
                        unit = vector / length
                        center = (start + end) * 0.5
                        size = min(length * 0.65, 1.5)
                        side = np.array([-unit[1], unit[0]])
                        painter.drawPolygon(QPolygonF([
                            QPointF(*(center + size * unit)),
                            QPointF(*(center - size * unit + size * 0.6 * side)),
                            QPointF(*(center - size * unit - size * 0.6 * side)),
                        ]))
            minimum = np.minimum(rows[:, :2], rows[:, 2:4]).min(axis=0)
            maximum = np.maximum(rows[:, :2], rows[:, 2:4]).max(axis=0)
            self._bounds = QRectF(float(minimum[0]) - 2, float(minimum[1]) - 2,
                                  float(maximum[0] - minimum[0]) + 4,
                                  float(maximum[1] - minimum[1]) + 4)
        else:
            self._bounds = QRectF()
        painter.end()
        self._picture = picture
        self._pictures[signature] = (picture, self._bounds)
        while len(self._pictures) > 48:
            self._pictures.popitem(last=False)
        self.update()
