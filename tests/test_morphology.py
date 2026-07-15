from __future__ import annotations

import numpy as np

from spatiotemporal_labeler.tools import (
    apply_label_morphology,
    remove_small_components,
)


def test_remove_small_components_supports_6_and_26_connectivity():
    selection = np.zeros((4, 4, 4), dtype=bool)
    selection[1, 1, 1] = True
    selection[2, 2, 2] = True

    face_connected = remove_small_components(selection, 2.0, connectivity=1)
    corner_connected = remove_small_components(selection, 2.0, connectivity=3)

    assert not np.any(face_connected)
    assert np.array_equal(corner_connected, selection)


def test_remove_small_components_handles_singleton_spatial_axis():
    frame = np.zeros((7, 7, 1), dtype=np.uint8)
    frame[1:3, 1:3, 0] = 1
    frame[5, 5, 0] = 1

    result = apply_label_morphology(
        frame, "remove_small_components", [1], minimum_volume_mm3=2.0
    )

    assert np.all(result[1:3, 1:3, 0] == 1)
    assert result[5, 5, 0] == 0


def test_fill_holes_never_overwrites_a_neighboring_label():
    frame = np.zeros((7, 7, 7), dtype=np.uint8)
    frame[1:6, 1:6, 1:6] = 1
    frame[2:5, 2:5, 2:5] = 0
    frame[3, 3, 3] = 2

    result = apply_label_morphology(frame, "fill_holes", [1])

    assert result[2, 2, 2] == 1
    assert result[3, 3, 3] == 2
    assert np.count_nonzero(result == 2) == 1


def test_all_label_morphology_processes_each_label_independently():
    frame = np.zeros((8, 8, 1), dtype=np.uint8)
    frame[1:3, 1:3, 0] = 1
    frame[5:7, 5:7, 0] = 2
    frame[0, 7, 0] = 1
    frame[7, 0, 0] = 2

    result = apply_label_morphology(
        frame, "remove_small_components", [1, 2], minimum_volume_mm3=2.0
    )

    assert np.count_nonzero(result == 1) == 4
    assert np.count_nonzero(result == 2) == 4


def test_opening_removes_spurs_and_closing_fills_small_gaps():
    frame = np.zeros((9, 9, 1), dtype=np.uint8)
    frame[2:7, 2:7, 0] = 1
    frame[1, 1, 0] = 1

    opened = apply_label_morphology(frame, "opening", [1])
    assert opened[1, 1, 0] == 0
    assert opened[4, 4, 0] == 1

    frame[4, 4, 0] = 0
    closed = apply_label_morphology(frame, "closing", [1])
    assert closed[4, 4, 0] == 1


def test_component_threshold_uses_physical_volume_for_anisotropic_data():
    selection = np.zeros((5, 5, 2), dtype=bool)
    selection[1, 1, 0] = True
    selection[3, 3, :] = True

    result = remove_small_components(
        selection,
        minimum_volume_mm3=7.5,
        spacing_xyz=(1.0, 1.0, 5.0),
    )

    assert not result[1, 1, 0]
    assert np.all(result[3, 3, :])


def test_physical_radius_does_not_cross_a_thick_slice_gap():
    frame = np.zeros((7, 7, 3), dtype=np.uint8)
    frame[2:5, 2:5, 0] = 1
    frame[2:5, 2:5, 2] = 1

    result = apply_label_morphology(
        frame,
        "closing",
        [1],
        spacing_xyz=(1.0, 1.0, 5.0),
        radius_mm=1.5,
    )

    assert not np.any(result[:, :, 1])
