import os

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtWidgets import QApplication

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.ui.image_strip import ImagePreviewStrip


class WheelEvent:
    def __init__(self, modifiers):
        self._modifiers = modifiers
        self.accepted = False

    def modifiers(self):
        return self._modifiers

    def angleDelta(self):
        return QPoint(0, 120)

    def position(self):
        return QPointF(65.0, 55.0)

    def accept(self):
        self.accepted = True


class DragEvent:
    def __init__(self, delta):
        self._delta = QPointF(*delta)
        self.accepted = False

    def button(self):
        return Qt.MouseButton.MiddleButton

    def isStart(self):
        return False

    def isFinish(self):
        return False

    def scenePos(self):
        return self._delta

    def lastScenePos(self):
        return QPointF()

    def accept(self):
        self.accepted = True


class DoubleClickEvent:
    def __init__(self):
        self.accepted = False

    def button(self):
        return Qt.MouseButton.LeftButton

    def accept(self):
        self.accepted = True


def make_sequence(name):
    transform = AxisTransform(
        original_axis_for_canonical=(0, 1, 2, 3),
        flipped_canonical_axes=(False, False, False),
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_ras=(0.0, 0.0, 0.0),
        direction_ras=np.eye(3),
    )
    data = np.arange(4 * 5 * 3 * 6, dtype=np.float32).reshape((4, 5, 3, 6))
    sequence = Sequence4D(data, {}, transform)
    sequence.path = name
    return sequence


def test_other_image_strip_collapses_instead_of_closing(tmp_path):
    QApplication.instance() or QApplication([])
    images = [make_sequence(tmp_path / "first.nrrd"), make_sequence(tmp_path / "second.nrrd")]
    strip = ImagePreviewStrip()
    states = []
    strip.collapsedChanged.connect(states.append)
    strip.rebuild(images, active_index=0)
    strip.update_images(images, (0, 0, 0, 0))

    assert len(strip._plots) == 1
    assert not strip.collapsed
    assert strip.content.isVisibleTo(strip)

    strip.set_collapsed(True)

    assert strip.collapsed
    assert strip.maximumWidth() == 32
    assert strip.content.isHidden()
    assert states == [True]

    strip.set_collapsed(False)

    assert strip.maximumWidth() == 420


def test_other_image_preview_plane_is_selectable(tmp_path):
    QApplication.instance() or QApplication([])
    images = [make_sequence(tmp_path / "first.nrrd"), make_sequence(tmp_path / "second.nrrd")]
    strip = ImagePreviewStrip()
    changed = []
    strip.planeChanged.connect(changed.append)
    strip.rebuild(images, active_index=0)
    cursor = (1, 2, 1, 4)

    strip.set_plane("X-Z")
    strip.update_images(images, cursor)

    assert strip.plane == "X-Z"
    assert changed == ["X-Z"]
    assert np.array_equal(strip._plots[1].item.image, images[1].data[:, 2, :, 4].T)
    assert strip._plots[1].getViewBox().state["xInverted"]
    assert strip._plots[1].getViewBox().state["aspectLocked"]

    strip.set_plane("Y-T")
    strip.update_images(images, cursor)

    assert np.array_equal(strip._plots[1].item.image, images[1].data[1, :, 1, :].T)
    assert not strip._plots[1].getViewBox().state["xInverted"]
    assert not strip._plots[1].getViewBox().state["aspectLocked"]


def test_other_image_preview_uses_physical_location_spacing_and_orientation(tmp_path):
    QApplication.instance() or QApplication([])
    images = [make_sequence(tmp_path / "first.nrrd"), make_sequence(tmp_path / "second.nrrd")]
    images[0].transform = AxisTransform(
        original_axis_for_canonical=(0, 1, 2, 3),
        flipped_canonical_axes=(False, False, False),
        spacing_xyz=(2.0, 3.0, 4.0),
        origin_ras=(0.0, 0.0, 0.0),
        direction_ras=np.eye(3),
    )
    images[1].transform = AxisTransform(
        original_axis_for_canonical=(0, 1, 2, 3),
        flipped_canonical_axes=(False, False, False),
        spacing_xyz=(2.0, 3.0, 4.0),
        origin_ras=(0.0, 0.0, 4.0),
        direction_ras=np.eye(3),
    )
    strip = ImagePreviewStrip()
    strip.rebuild(images, active_index=0)

    strip.update_images(images, (1, 2, 2, 4), reference_image=images[0])

    plot = strip._plots[1]
    assert np.array_equal(plot.item.image, images[1].data[:, :, 1, 4].T)
    assert plot.getViewBox().state["xInverted"]
    assert np.isclose(plot.item.transform().m11(), 2.0)
    assert np.isclose(plot.item.transform().m22(), 3.0)


def test_other_image_preview_ctrl_wheel_zoom_survives_image_refresh(tmp_path):
    QApplication.instance() or QApplication([])
    images = [make_sequence(tmp_path / "first.nrrd"), make_sequence(tmp_path / "second.nrrd")]
    strip = ImagePreviewStrip()
    strip.rebuild(images, active_index=0)
    strip.update_images(images, (0, 0, 0, 0))
    plot = strip._plots[1]
    initial_width = np.ptp(plot.getViewBox().viewRange()[0])
    event = WheelEvent(Qt.KeyboardModifier.ControlModifier)

    plot.wheelEvent(event)
    zoomed_width = np.ptp(plot.getViewBox().viewRange()[0])
    strip.update_images(images, (1, 1, 1, 1))

    assert event.accepted
    assert zoomed_width < initial_width
    assert np.isclose(np.ptp(plot.getViewBox().viewRange()[0]), zoomed_width)

    strip.set_plane("Y-Z")
    strip.update_images(images, (1, 1, 1, 1))
    assert plot.getViewBox().state["xInverted"]


def test_other_image_preview_middle_drag_adjusts_its_own_window(tmp_path):
    QApplication.instance() or QApplication([])
    images = [
        make_sequence(tmp_path / "first.nrrd"),
        make_sequence(tmp_path / "second.nrrd"),
        make_sequence(tmp_path / "third.nrrd"),
    ]
    strip = ImagePreviewStrip()
    strip.rebuild(images, active_index=0)
    strip.update_images(images, (0, 0, 0, 0))
    plot = strip._plots[1]
    untouched_levels = strip._plots[2].levels
    initial_low, initial_high = plot.levels
    initial_center = (initial_low + initial_high) * 0.5
    initial_width = initial_high - initial_low
    event = DragEvent((20.0, -10.0))
    changes = []
    strip.imageLevelsChanged.connect(
        lambda index, low, high: changes.append((index, low, high))
    )

    plot.item.mouseDragEvent(event)

    adjusted_levels = plot.levels
    adjusted_low, adjusted_high = adjusted_levels
    assert event.accepted
    assert adjusted_high - adjusted_low > initial_width
    assert (adjusted_low + adjusted_high) * 0.5 > initial_center
    assert strip._plots[2].levels == untouched_levels
    assert tuple(plot.item.levels) == adjusted_levels
    assert changes == [(1, *adjusted_levels)]
    assert not plot.level_feedback.isHidden()
    assert "WL" in plot.level_feedback.text()
    assert "WW" in plot.level_feedback.text()

    strip.update_images(images, (1, 1, 1, 1))

    assert plot.levels == adjusted_levels
    assert tuple(plot.item.levels) == adjusted_levels


def test_other_image_preview_reset_restores_auto_levels_and_zoom(tmp_path, monkeypatch):
    QApplication.instance() or QApplication([])
    images = [make_sequence(tmp_path / "first.nrrd"), make_sequence(tmp_path / "second.nrrd")]
    strip = ImagePreviewStrip()
    strip.rebuild(images, active_index=0)
    strip.update_images(images, (0, 0, 0, 0))
    plot = strip._plots[1]
    initial_levels = plot.levels
    initial_width = np.ptp(plot.getViewBox().viewRange()[0])
    plot.item.mouseDragEvent(DragEvent((20.0, -10.0)))
    plot.wheelEvent(WheelEvent(Qt.KeyboardModifier.ControlModifier))
    monkeypatch.setattr(plot, "underMouse", lambda: True)

    assert strip.reset_hovered_preview()

    assert plot.levels == initial_levels
    assert tuple(plot.item.levels) == initial_levels
    assert np.isclose(np.ptp(plot.getViewBox().viewRange()[0]), initial_width)


def test_other_image_preview_activates_on_double_click_and_restores_saved_levels(tmp_path):
    QApplication.instance() or QApplication([])
    images = [make_sequence(tmp_path / "first.nrrd"), make_sequence(tmp_path / "second.nrrd")]
    saved_levels = (40.0, 80.0)
    strip = ImagePreviewStrip()
    activated = []
    strip.imageActivated.connect(activated.append)
    strip.rebuild(images, active_index=0, levels_by_image={id(images[1]): saved_levels})
    strip.update_images(images, (0, 0, 0, 0))
    event = DoubleClickEvent()

    strip._plots[1].mouseDoubleClickEvent(event)

    assert event.accepted
    assert activated == [1]
    assert strip._plots[1].levels == saved_levels
