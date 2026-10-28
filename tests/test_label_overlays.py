import os
os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import numpy as np
import pytest

from spatiotemporal_labeler.model import LabelDefinition, default_label
from spatiotemporal_labeler.ui.label_overlay import LabelOverlayBuffer


def reference_overlay(mask, definitions, opacity=1.0):
    result = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for definition in definitions.values():
        if definition.visible:
            selected = mask == definition.value
            result[..., :3][selected] = definition.color
            result[..., 3][selected] = round(
                124 * np.clip(definition.opacity, 0, 1) * np.clip(opacity, 0, 1)
            )
    return result


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint32, np.int64, np.float32])
@pytest.mark.parametrize("opacity", [0.0, 0.4, 1.0, 2.0])
def test_lookup_matches_original_colors_visibility_and_opacity(dtype, opacity):
    mask = np.array([[0, 1, 2, 3], [7, 2, 1, 0]], dtype=dtype)
    definitions = {value: default_label(value) for value in (1, 2, 3)}
    definitions[1].opacity = 0.35
    definitions[2].visible = False
    definitions[3].opacity = 2.0
    np.testing.assert_array_equal(
        LabelOverlayBuffer().render(mask, definitions, global_opacity=opacity),
        reference_overlay(mask, definitions, opacity),
    )


def test_signed_sparse_and_boolean_values_do_not_alias_lookup_slots():
    cases = [
        np.array([[-32768, -1, 0, 1, 32767]], dtype=np.int16),
        np.array([[0, 1, 2**40]], dtype=np.uint64),
        np.array([[False, True]], dtype=bool),
    ]
    definitions = {
        value: LabelDefinition(value, str(value), (21, 45, 90))
        for value in (-32768, 1, 2**40)
    }
    for mask in cases:
        np.testing.assert_array_equal(
            LabelOverlayBuffer().render(mask, definitions),
            reference_overlay(mask, definitions),
        )


def test_local_updates_reuse_the_buffer_and_style_changes_recolor_every_pixel():
    mask = np.ones((9, 11), dtype=np.uint8)
    definitions = {value: default_label(value) for value in (1, 2)}
    buffer = LabelOverlayBuffer()
    rgba = buffer.render(mask, definitions)
    original = rgba.copy()
    region = (slice(2, 4), slice(5, 8))
    mask[region] = 2
    assert buffer.render(mask, definitions, dirty_bounds=region) is rgba
    np.testing.assert_array_equal(rgba, reference_overlay(mask, definitions))
    np.testing.assert_array_equal(rgba[:2], original[:2])
    definitions[1].color = (200, 10, 40)
    buffer.render(mask, definitions, dirty_bounds=region)
    np.testing.assert_array_equal(rgba, reference_overlay(mask, definitions))
    definitions[2].visible = False
    buffer.render(mask, definitions, dirty_bounds=region)
    np.testing.assert_array_equal(rgba, reference_overlay(mask, definitions))
    buffer.clear()
    assert buffer.rgba is None


def test_switching_same_shape_sources_invalidates_partial_updates():
    definitions = {value: default_label(value) for value in (1, 2)}
    first = np.ones((5, 6), dtype=np.uint8)
    second = np.full_like(first, 2)
    buffer = LabelOverlayBuffer()
    rgba = buffer.render(first, definitions)
    assert buffer.render(second, definitions, dirty_bounds=(slice(0, 1), slice(0, 1))) is rgba
    np.testing.assert_array_equal(rgba, reference_overlay(second, definitions))
