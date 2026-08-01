import numpy as np

from spatiotemporal_labeler.tools import (
    apply_disk,
    apply_square,
    fill_polygon,
    polygon_selection,
    raster_line,
    transform_selected_labels,
)


def test_disk_uses_physical_spacing():
    plane = np.zeros((21, 21), dtype=np.uint8)
    apply_disk(plane, (10, 10), radius_mm=4.0, spacing=(1.0, 2.0), value=3)

    assert plane[6, 10] == 3
    assert plane[10, 8] == 3
    assert plane[10, 7] == 0


def test_closed_polygon_fills_inside_only():
    plane = np.zeros((12, 12), dtype=np.uint8)
    fill_polygon(plane, [(2, 2), (9, 2), (9, 9), (2, 9)], value=5)

    assert plane[5, 5] == 5
    assert plane[1, 5] == 0
    assert plane[10, 5] == 0


def test_closed_polygon_preserves_rasterized_boundary_pixels():
    plane = np.zeros((12, 12), dtype=np.uint8)
    boundary = raster_line((2, 2), (9, 5))
    points = [*boundary, *raster_line((9, 5), (3, 9))[1:], *raster_line((3, 9), (2, 2))[1:]]

    fill_polygon(plane, points, value=6)

    assert all(plane[h, v] == 6 for h, v in points)


def test_square_uses_physical_diameter():
    plane = np.zeros((15, 15), dtype=np.uint8)
    apply_square(plane, (7, 7), radius_mm=4.0, spacing=(1.0, 2.0), value=4)

    assert plane[3, 5] == 4
    assert plane[11, 9] == 4
    assert plane[2, 7] == 0
    assert plane[7, 10] == 0


def test_brush_operations_only_write_inside_the_allowed_selection():
    allowed = np.zeros((15, 15), dtype=bool)
    allowed[7, 7] = True

    disk = np.zeros(allowed.shape, dtype=np.uint8)
    apply_disk(
        disk,
        (7, 7),
        radius_mm=4.0,
        spacing=(1.0, 1.0),
        value=3,
        allowed=allowed,
    )
    square = np.zeros(allowed.shape, dtype=np.uint8)
    apply_square(
        square,
        (7, 7),
        radius_mm=4.0,
        spacing=(1.0, 1.0),
        value=4,
        allowed=allowed,
    )

    assert np.count_nonzero(disk) == 1
    assert disk[7, 7] == 3
    assert np.count_nonzero(square) == 1
    assert square[7, 7] == 4


def test_raster_line_contains_each_pixel_without_gaps():
    points = raster_line((1, 1), (7, 4))

    assert points[0] == (1, 1)
    assert points[-1] == (7, 4)
    assert all(
        max(abs(next_h - h), abs(next_v - v)) == 1
        for (h, v), (next_h, next_v) in zip(points, points[1:])
    )


def test_lasso_selection_is_implicitly_closed_and_transforms_only_source_label():
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[2:10, 2:10] = 1
    labels[5, 5] = 2
    selection = polygon_selection((12, 12), [(3, 3), (8, 3), (8, 8), (3, 8)])

    changed = transform_selected_labels(labels, selection, 3, source_value=1)

    assert changed > 0
    assert labels[4, 4] == 3
    assert labels[5, 5] == 2
    assert labels[2, 2] == 1
