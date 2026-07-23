import numpy as np

from spatiotemporal_labeler.i18n import translate
from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.model import default_label, labels_from_sequence, store_labels


def make_sequence() -> Sequence4D:
    transform = AxisTransform(
        original_axis_for_canonical=(0, 1, 2, 3),
        flipped_canonical_axes=(False, False, False),
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_ras=(0.0, 0.0, 0.0),
        direction_ras=np.eye(3),
    )
    data = np.zeros((3, 3, 3, 2), dtype=np.uint8)
    data[1, 1, 1, 0] = 2
    return Sequence4D(data, {}, transform)


def test_label_definitions_round_trip_through_header():
    sequence = make_sequence()
    definitions = labels_from_sequence(sequence)
    definitions[2].name = "Aorta"
    definitions[2].color = (10, 20, 30)
    definitions[2].opacity = 0.35
    store_labels(sequence, definitions)

    restored = labels_from_sequence(sequence)

    assert restored[2].name == "Aorta"
    assert restored[2].color == (10, 20, 30)
    assert restored[2].opacity == 0.35


def test_english_is_available_as_default_interface_language():
    assert translate("en", "display_image") == "Display image"
    assert translate("zh_CN", "display_image") == "显示图像"


def test_default_label_colors_do_not_repeat_in_uint8_range():
    colors = [default_label(value).color for value in range(1, 256)]

    assert len(set(colors)) == len(colors)
