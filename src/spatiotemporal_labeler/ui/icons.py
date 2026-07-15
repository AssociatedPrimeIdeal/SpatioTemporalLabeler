from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF


def tool_icon(name: str, color: str) -> QIcon:
    """Create crisp toolbar icons without an external icon-theme dependency."""
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(color), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if name == "brush":
            painter.drawLine(QPointF(6, 18), QPointF(16, 8))
            painter.drawEllipse(QRectF(15, 4, 5, 5))
            painter.drawArc(QRectF(3, 15, 7, 6), 210 * 16, 220 * 16)
        elif name == "eraser":
            painter.save()
            painter.translate(12, 12)
            painter.rotate(-38)
            painter.drawRoundedRect(QRectF(-7, -4, 14, 8), 2, 2)
            painter.drawLine(QPointF(1, -4), QPointF(1, 4))
            painter.restore()
            painter.drawLine(QPointF(5, 20), QPointF(18, 20))
        elif name == "contour":
            vertices = [
                QPointF(5, 17),
                QPointF(8, 6),
                QPointF(18, 8),
                QPointF(19, 17),
            ]
            painter.drawPolyline(QPolygonF([*vertices, vertices[0]]))
            for point in vertices:
                painter.drawRect(QRectF(point.x() - 1.2, point.y() - 1.2, 2.4, 2.4))
        elif name == "picker":
            painter.drawEllipse(QRectF(4, 4, 16, 16))
            painter.drawEllipse(QRectF(9, 9, 6, 6))
            painter.drawLine(QPointF(12, 2), QPointF(12, 7))
            painter.drawLine(QPointF(12, 17), QPointF(12, 22))
            painter.drawLine(QPointF(2, 12), QPointF(7, 12))
            painter.drawLine(QPointF(17, 12), QPointF(22, 12))
        elif name == "grow":
            painter.drawEllipse(QRectF(9, 9, 6, 6))
            painter.drawArc(QRectF(5, 5, 14, 14), 15 * 16, 240 * 16)
            painter.drawArc(QRectF(2.5, 2.5, 19, 19), 195 * 16, 210 * 16)
        elif name == "threshold":
            painter.drawRect(QRectF(4, 5, 16, 14))
            painter.drawLine(QPointF(7, 15), QPointF(17, 15))
            painter.drawLine(QPointF(7, 10), QPointF(17, 10))
            painter.drawEllipse(QRectF(8, 8, 4, 4))
            painter.drawEllipse(QRectF(14, 13, 4, 4))
        elif name == "window":
            painter.drawRect(QRectF(4, 5, 16, 14))
            painter.drawLine(QPointF(8, 8), QPointF(8, 16))
            painter.drawLine(QPointF(16, 8), QPointF(16, 16))
            painter.drawEllipse(QRectF(6.5, 10.5, 3, 3))
            painter.drawEllipse(QRectF(14.5, 8.5, 3, 3))
        elif name == "import":
            painter.drawRoundedRect(QRectF(3.5, 4.5, 12.5, 15), 1.5, 1.5)
            painter.drawLine(QPointF(18.5, 4), QPointF(18.5, 15))
            painter.drawLine(QPointF(14.5, 11), QPointF(18.5, 15))
            painter.drawLine(QPointF(22.5, 11), QPointF(18.5, 15))
        elif name == "morphology":
            painter.drawEllipse(QRectF(3.5, 7.5, 9, 9))
            painter.drawEllipse(QRectF(11.5, 4.5, 9, 9))
            painter.drawLine(QPointF(7, 19), QPointF(17, 19))
            painter.drawLine(QPointF(12, 16), QPointF(12, 22))
        elif name == "interpolate":
            painter.drawRoundedRect(QRectF(3.5, 4.5, 6, 6), 1, 1)
            painter.drawRoundedRect(QRectF(14.5, 13.5, 6, 6), 1, 1)
            painter.drawEllipse(QRectF(10.5, 9.5, 3, 3))
            painter.drawLine(QPointF(9.5, 9.5), QPointF(11, 10.5))
            painter.drawLine(QPointF(13, 12.5), QPointF(14.5, 14))
        elif name == "image":
            painter.drawRoundedRect(QRectF(3.5, 4.5, 17, 15), 1.5, 1.5)
            painter.drawEllipse(QRectF(7, 8, 2.5, 2.5))
            painter.drawPolyline(
                QPolygonF([QPointF(5.5, 17), QPointF(10, 12.5), QPointF(13, 15), QPointF(16, 11), QPointF(19, 16)])
            )
        elif name == "labels":
            painter.drawRoundedRect(QRectF(4, 5, 13, 13), 2, 2)
            painter.drawRoundedRect(QRectF(8, 8, 12, 12), 2, 2)
            painter.drawLine(QPointF(12, 12), QPointF(16, 12))
            painter.drawLine(QPointF(12, 16), QPointF(17, 16))
        elif name == "new":
            painter.drawRoundedRect(QRectF(4, 4, 12, 16), 1.5, 1.5)
            painter.drawLine(QPointF(12, 14), QPointF(21, 14))
            painter.drawLine(QPointF(16.5, 9.5), QPointF(16.5, 18.5))
        elif name in {"save", "save_as"}:
            painter.drawRoundedRect(QRectF(4, 3.5, 16, 17), 1.5, 1.5)
            painter.drawRect(QRectF(7, 4, 9, 6))
            painter.drawRoundedRect(QRectF(7, 14, 10, 6), 1, 1)
            if name == "save_as":
                painter.drawLine(QPointF(14.5, 17.5), QPointF(21, 11))
                painter.drawLine(QPointF(18.5, 10.5), QPointF(21.5, 13.5))
        elif name in {"undo", "redo"}:
            if name == "undo":
                points = [QPointF(10, 6), QPointF(5, 11), QPointF(10, 16)]
                painter.drawPolyline(QPolygonF(points))
                painter.drawArc(QRectF(6, 7, 14, 11), -70 * 16, 220 * 16)
            else:
                points = [QPointF(14, 6), QPointF(19, 11), QPointF(14, 16)]
                painter.drawPolyline(QPolygonF(points))
                painter.drawArc(QRectF(4, 7, 14, 11), 30 * 16, 220 * 16)
    finally:
        painter.end()
    return QIcon(pixmap)
