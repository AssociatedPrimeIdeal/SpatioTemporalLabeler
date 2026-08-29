from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from PySide6.QtCore import QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget


# Keep these colours in one place so the timeline and the X-T cardiac guides
# always agree.  The values are intentionally high-contrast on the light UI.
PHASE_LABEL_COLORS = {
    "systole_start": "#45d6c8",
    "peak": "#ffb347",
    "diastole_start": "#b58cff",
}
FRAME_LABEL_DEFAULT_COLOR = "#4f8cc9"


@dataclass
class FrameLabel:
    frame: int
    name: str
    color: str
    phase_key: str | None = None


class FrameLabelTimeline(QWidget):
    """A compact clickable timeline of named frame markers."""

    labelClicked = Signal(int)
    labelContextRequested = Signal(int, QPoint)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._frame_count = 1
        self._labels: tuple[FrameLabel, ...] = ()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(30)
        self.setMaximumHeight(42)
        self.setMouseTracking(True)

    @property
    def labels(self) -> tuple[FrameLabel, ...]:
        return self._labels

    def sizeHint(self) -> QSize:
        return QSize(300, 32)

    def set_frame_count(self, count: int) -> None:
        self._frame_count = max(1, int(count))
        self.update()

    def set_labels(self, labels: Iterable[FrameLabel]) -> None:
        self._labels = tuple(
            sorted(
                (
                    FrameLabel(
                        int(label.frame),
                        str(label.name),
                        str(label.color),
                        label.phase_key,
                    )
                    for label in labels
                    if 0 <= int(label.frame) < self._frame_count
                ),
                key=lambda label: (label.frame, label.name),
            )
        )
        self.update()

    def _frame_x(self, frame: int) -> float:
        if self._frame_count <= 1:
            return max(0.0, (self.width() - 1) / 2.0)
        return float(frame) * max(0.0, self.width() - 1) / float(self._frame_count - 1)

    def _label_at(self, x: float, y: float) -> FrameLabel | None:
        if not self._labels:
            return None
        nearest = min(self._labels, key=lambda label: abs(self._frame_x(label.frame) - x))
        # A marker remains easy to hit even when neighbouring frames are dense,
        # while clicks far away continue to pass through the timeline.
        tolerance = 75.0 if y >= 12.0 else 12.0
        return nearest if abs(self._frame_x(nearest.frame) - x) <= tolerance else None

    def paintEvent(self, _event: object) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor("#9eafb2"))
        painter.drawLine(0, 8, max(0, self.width() - 1), 8)
        for label in self._labels:
            x = self._frame_x(label.frame)
            color = QColor(label.color)
            if not color.isValid():
                color = QColor("#53666a")
            painter.setPen(color)
            painter.setBrush(color)
            painter.drawLine(round(x), 11, round(x), 2)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x, 1),
                        QPointF(x - 4.5, 7),
                        QPointF(x + 4.5, 7),
                    ]
                )
            )
            painter.drawText(
                int(round(x - 75)),
                13,
                150,
                max(1, self.height() - 13),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                label.name,
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        label = self._label_at(float(event.position().x()), float(event.position().y()))
        if label is None:
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.labelClicked.emit(int(label.frame))
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.labelContextRequested.emit(int(label.frame), event.globalPosition().toPoint())
            event.accept()
            return
        event.ignore()
