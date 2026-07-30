from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from skimage.feature import match_template


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


def find_similar_patch_center(
    reference: NDArray[np.number],
    target: NDArray[np.number],
    center: tuple[int, int],
    patch_radius: tuple[int, int],
    search_radius: tuple[int, int],
    minimum_similarity: float = -1.0,
) -> tuple[int, int] | None:
    """Return the best normalized-correlation match that clears the threshold."""
    reference_data = np.asarray(reference, dtype=np.float64)
    target_data = np.asarray(target, dtype=np.float64)
    if reference_data.ndim != 2 or target_data.ndim != 2:
        raise ValueError("Patch matching requires two-dimensional image planes")
    if reference_data.shape != target_data.shape:
        raise ValueError("Reference and target image planes must have the same shape")

    h_center, v_center = (int(center[0]), int(center[1]))
    if not (
        0 <= h_center < reference_data.shape[0]
        and 0 <= v_center < reference_data.shape[1]
    ):
        raise ValueError("Patch center must be inside the image plane")
    h_patch, v_patch = (max(0, int(value)) for value in patch_radius)
    h_search, v_search = (max(0, int(value)) for value in search_radius)

    def edge_padded_region(
        data: NDArray[np.float64],
        h_start: int,
        h_stop: int,
        v_start: int,
        v_stop: int,
    ) -> NDArray[np.float64]:
        clipped_h0 = max(0, h_start)
        clipped_h1 = min(data.shape[0], h_stop)
        clipped_v0 = max(0, v_start)
        clipped_v1 = min(data.shape[1], v_stop)
        region = data[clipped_h0:clipped_h1, clipped_v0:clipped_v1]
        return np.pad(
            region,
            (
                (max(0, -h_start), max(0, h_stop - data.shape[0])),
                (max(0, -v_start), max(0, v_stop - data.shape[1])),
            ),
            mode="edge",
        )

    reference_patch = edge_padded_region(
        reference_data,
        h_center - h_patch,
        h_center + h_patch + 1,
        v_center - v_patch,
        v_center + v_patch + 1,
    )

    h0 = max(0, h_center - h_search)
    h1 = min(target_data.shape[0] - 1, h_center + h_search)
    v0 = max(0, v_center - v_search)
    v1 = min(target_data.shape[1] - 1, v_center + v_search)
    search_region = edge_padded_region(
        target_data,
        h0 - h_patch,
        h1 + h_patch + 1,
        v0 - v_patch,
        v1 + v_patch + 1,
    )

    finite_reference = np.isfinite(reference_patch)
    minimum_count = max(1, int(np.ceil(reference_patch.size / 2.0)))
    if np.count_nonzero(finite_reference) < minimum_count:
        return None
    reference_fill = float(np.mean(reference_patch[finite_reference]))
    reference_patch = np.where(finite_reference, reference_patch, reference_fill)
    finite_search = np.isfinite(search_region)
    if not np.any(finite_search):
        return None
    search_fill = float(np.mean(search_region[finite_search]))
    search_region = np.where(finite_search, search_region, search_fill)
    score = 1.0 - match_template(search_region, reference_patch, pad_input=False)
    score[~np.isfinite(score)] = np.inf

    candidate_h, candidate_v = np.indices(score.shape)
    candidate_h += h0
    candidate_v += v0
    distance = (candidate_h - h_center) ** 2 + (candidate_v - v_center) ** 2
    order = np.lexsort(
        (
            candidate_v.ravel(),
            candidate_h.ravel(),
            distance.ravel(),
            score.ravel(),
        )
    )
    if not order.size or not np.isfinite(score.ravel()[order[0]]):
        return None
    best = int(order[0])
    similarity = 1.0 - float(score.ravel()[best])
    if similarity + 1e-12 < float(minimum_similarity):
        return None
    return int(candidate_h.ravel()[best]), int(candidate_v.ravel()[best])


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
