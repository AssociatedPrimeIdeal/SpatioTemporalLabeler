import numpy as np

from spatiotemporal_labeler.tools import apply_disk, apply_square, fill_polygon, raster_line


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


def test_raster_line_contains_each_pixel_without_gaps():
    points = raster_line((1, 1), (7, 4))

    assert points[0] == (1, 1)
    assert points[-1] == (7, 4)
    assert all(
        max(abs(next_h - h), abs(next_v - v)) == 1
        for (h, v), (next_h, next_v) in zip(points, points[1:])
    )
