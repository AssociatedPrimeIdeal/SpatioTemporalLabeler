from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def apply_disk(
    plane: NDArray[np.integer],
    center: tuple[int, int],
    radius_mm: float,
    spacing: tuple[float, float],
    value: int,
    allowed: NDArray[np.bool_] | None = None,
) -> tuple[slice, slice] | None:
    """Paint a physically circular disk into a 2D H,V plane."""
    h_center, v_center = center
    h_radius = max(1, int(np.ceil(radius_mm / spacing[0])))
    v_radius = max(1, int(np.ceil(radius_mm / spacing[1])))
    h0, h1 = max(0, h_center - h_radius), min(plane.shape[0], h_center + h_radius + 1)
    v0, v1 = max(0, v_center - v_radius), min(plane.shape[1], v_center + v_radius + 1)
    if h0 >= h1 or v0 >= v1:
        return None
    h_grid, v_grid = np.ogrid[h0:h1, v0:v1]
    disk = ((h_grid - h_center) * spacing[0]) ** 2 + ((v_grid - v_center) * spacing[1]) ** 2
    region = plane[h0:h1, v0:v1]
    footprint = disk <= radius_mm**2
    if allowed is not None:
        footprint &= np.asarray(allowed[h0:h1, v0:v1], dtype=bool)
    region[footprint] = value
    return slice(h0, h1), slice(v0, v1)


def apply_square(
    plane: NDArray[np.integer],
    center: tuple[int, int],
    radius_mm: float,
    spacing: tuple[float, float],
    value: int,
    allowed: NDArray[np.bool_] | None = None,
) -> tuple[slice, slice] | None:
    """Paint a physically square, axis-aligned footprint into a 2D H,V plane."""
    h_center, v_center = center
    h_radius = max(0, int(np.floor(radius_mm / spacing[0])))
    v_radius = max(0, int(np.floor(radius_mm / spacing[1])))
    h0, h1 = max(0, h_center - h_radius), min(plane.shape[0], h_center + h_radius + 1)
    v0, v1 = max(0, v_center - v_radius), min(plane.shape[1], v_center + v_radius + 1)
    if h0 >= h1 or v0 >= v1:
        return None
    region = plane[h0:h1, v0:v1]
    if allowed is None:
        region[...] = value
    else:
        region[np.asarray(allowed[h0:h1, v0:v1], dtype=bool)] = value
    return slice(h0, h1), slice(v0, v1)


def raster_line(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    """Return every voxel center crossed by a Bresenham line, including both ends."""
    h0, v0 = start
    h1, v1 = end
    delta_h = abs(h1 - h0)
    step_h = 1 if h0 < h1 else -1
    delta_v = -abs(v1 - v0)
    step_v = 1 if v0 < v1 else -1
    error = delta_h + delta_v
    points: list[tuple[int, int]] = []
    while True:
        points.append((h0, v0))
        if h0 == h1 and v0 == v1:
            return points
        doubled = 2 * error
        if doubled >= delta_v:
            error += delta_v
            h0 += step_h
        if doubled <= delta_h:
            error += delta_h
            v0 += step_v


def fill_polygon(
    plane: NDArray[np.integer], points: Sequence[tuple[int, int]], value: int
) -> tuple[slice, slice] | None:
    """Fill a closed polygon in a 2D H,V plane using scanline intersections."""
    if len(points) < 3:
        return None
    polygon = np.asarray(points, dtype=float)
    h_min = max(0, int(np.floor(polygon[:, 0].min())))
    h_max = min(plane.shape[0] - 1, int(np.ceil(polygon[:, 0].max())))
    v_min = max(0, int(np.floor(polygon[:, 1].min())))
    v_max = min(plane.shape[1] - 1, int(np.ceil(polygon[:, 1].max())))
    if h_min > h_max or v_min > v_max:
        return None

    closed = np.vstack([polygon, polygon[0]])
    for v in range(v_min, v_max + 1):
        scan_v = v + 0.5
        intersections: list[float] = []
        for start, end in zip(closed[:-1], closed[1:]):
            if (start[1] <= scan_v < end[1]) or (end[1] <= scan_v < start[1]):
                ratio = (scan_v - start[1]) / (end[1] - start[1])
                intersections.append(float(start[0] + ratio * (end[0] - start[0])))
        intersections.sort()
        for left, right in zip(intersections[::2], intersections[1::2]):
            start_h = max(h_min, int(np.ceil(left - 0.5)))
            end_h = min(h_max, int(np.floor(right - 0.5)))
            if start_h <= end_h:
                plane[start_h : end_h + 1, v] = value
    for h, v in np.asarray(points, dtype=int):
        if 0 <= h < plane.shape[0] and 0 <= v < plane.shape[1]:
            plane[h, v] = value
    return slice(h_min, h_max + 1), slice(v_min, v_max + 1)


def polygon_selection(
    shape: tuple[int, int], points: Sequence[tuple[int, int]]
) -> NDArray[np.bool_]:
    """Return the rasterized interior and boundary of an implicitly closed polygon."""
    selection = np.zeros(shape, dtype=np.uint8)
    fill_polygon(selection, points, 1)
    return selection.astype(bool, copy=False)


def transform_selected_labels(
    labels: NDArray[np.integer],
    selection: NDArray[np.bool_],
    target_value: int,
    source_value: int | None = None,
) -> int:
    """Replace selected label voxels, optionally filtering by one source value."""
    data = np.asarray(labels)
    selected = np.asarray(selection, dtype=bool)
    if data.shape != selected.shape:
        raise ValueError(
            f"Label selection shape {selected.shape} does not match data {data.shape}"
        )
    eligible = selected & (data != 0 if source_value is None else data == source_value)
    changed = eligible & (data != int(target_value))
    data[changed] = int(target_value)
    return int(np.count_nonzero(changed))
