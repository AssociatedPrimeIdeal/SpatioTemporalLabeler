import os

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from spatiotemporal_labeler.ui.icons import tool_icon
from spatiotemporal_labeler.ui.slice_view import EditableImageItem


class ClickEvent:
    def __init__(self, button, modifiers=Qt.KeyboardModifier.NoModifier, double=False):
        self._button = button
        self._modifiers = modifiers
        self._double = double
        self.accepted = False

    def pos(self):
        return QPointF(4.2, 7.8)

    def button(self):
        return self._button

    def modifiers(self):
        return self._modifiers

    def double(self):
        return self._double

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


class HoverEvent:
    def __init__(self, modifiers=Qt.KeyboardModifier.NoModifier):
        self._modifiers = modifiers

    def isExit(self):
        return False

    def pos(self):
        return QPointF(4.2, 7.8)

    def modifiers(self):
        return self._modifiers


class DragEvent:
    def __init__(self, button, modifiers=Qt.KeyboardModifier.NoModifier):
        self._button = button
        self._modifiers = modifiers
        self.accepted = False

    def button(self):
        return self._button

    def modifiers(self):
        return self._modifiers

    def isStart(self):
        return False

    def isFinish(self):
        return False

    def pos(self):
        return QPointF(8.0, 11.0)

    def lastPos(self):
        return QPointF(5.0, 7.0)

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def ensure_application():
    return QApplication.instance() or QApplication([])


def test_all_toolbar_icons_render_without_error():
    application = ensure_application()

    for name in (
        "image",
        "labels",
        "new",
        "save",
        "save_as",
        "undo",
        "redo",
        "brush",
        "eraser",
        "contour",
        "threshold",
        "window",
    ):
        assert not tool_icon(name, "#147b86").isNull()

    assert application is not None


def test_shift_click_locates_without_starting_a_stroke():
    ensure_application()
    item = EditableImageItem()
    located = []
    strokes = []
    item.navigateRequested.connect(lambda h, v: located.append((h, v)))
    item.strokeStarted.connect(lambda h, v, erase: strokes.append((h, v, erase)))
    event = ClickEvent(Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)

    item.mouseClickEvent(event)

    assert event.accepted
    assert located == [(4, 7)]
    assert strokes == []


def test_right_click_is_a_temporary_eraser_stroke():
    ensure_application()
    item = EditableImageItem()
    started = []
    finished = []
    item.strokeStarted.connect(lambda h, v, erase: started.append((h, v, erase)))
    item.strokeFinished.connect(lambda h, v, erase: finished.append((h, v, erase)))
    event = ClickEvent(Qt.MouseButton.RightButton)

    item.mouseClickEvent(event)

    assert event.accepted
    assert started == [(4, 7, True)]
    assert finished == [(4, 7, True)]


def test_shift_hover_locates_without_a_mouse_button():
    ensure_application()
    item = EditableImageItem()
    located = []
    item.navigateRequested.connect(lambda h, v: located.append((h, v)))

    item.hoverEvent(HoverEvent(Qt.KeyboardModifier.ShiftModifier))

    assert located == [(4, 7)]


def test_double_click_does_not_emit_a_single_click_stroke():
    ensure_application()
    item = EditableImageItem()
    strokes = []
    double_clicks = []
    item.strokeStarted.connect(lambda h, v, erase: strokes.append((h, v, erase)))
    item.doubleClicked.connect(lambda h, v: double_clicks.append((h, v)))
    event = ClickEvent(Qt.MouseButton.LeftButton, double=True)

    item.mouseClickEvent(event)

    assert event.accepted
    assert strokes == []
    assert double_clicks == [(4, 7)]


def test_middle_drag_requests_window_level_without_panning_or_stroking():
    ensure_application()
    item = EditableImageItem()
    adjustments = []
    pans = []
    strokes = []
    item.windowLevelRequested.connect(lambda width, level: adjustments.append((width, level)))
    item.panRequested.connect(lambda h, v: pans.append((h, v)))
    item.strokeMoved.connect(lambda h, v, erase: strokes.append((h, v, erase)))
    event = DragEvent(Qt.MouseButton.MiddleButton)

    item.mouseDragEvent(event)

    assert event.accepted
    assert adjustments == [(3.0, 4.0)]
    assert pans == []
    assert strokes == []


def test_shift_left_drag_still_requests_pan():
    ensure_application()
    item = EditableImageItem()
    pans = []
    adjustments = []
    item.panRequested.connect(lambda h, v: pans.append((h, v)))
    item.windowLevelRequested.connect(lambda width, level: adjustments.append((width, level)))
    event = DragEvent(Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)

    item.mouseDragEvent(event)

    assert event.accepted
    assert pans == [(3.0, 4.0)]
    assert adjustments == []
