import os

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import numpy as np
from PySide6.QtWidgets import QApplication

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.ui.image_strip import ImagePreviewStrip


def make_sequence(name):
    transform = AxisTransform(
        original_axis_for_canonical=(0, 1, 2, 3),
        flipped_canonical_axes=(False, False, False),
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_ras=(0.0, 0.0, 0.0),
        direction_ras=np.eye(3),
    )
    sequence = Sequence4D(np.ones((4, 5, 2, 1), dtype=np.float32), {}, transform)
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
