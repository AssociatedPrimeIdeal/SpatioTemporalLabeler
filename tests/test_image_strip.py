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

    strip.set_plane("Y-T")
    strip.update_images(images, cursor)

    assert np.array_equal(strip._plots[1].item.image, images[1].data[1, :, 1, :].T)


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
