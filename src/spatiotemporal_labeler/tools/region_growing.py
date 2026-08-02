"""Stroke-seeded temporal label propagation for canonical ``[X, Y, Z, T]`` arrays.

The tool deliberately has no within-frame neighborhood traversal.  A stroke is
the complete source patch in its frame; every accepted voxel can only nominate
one intensity-compatible voxel in the immediately adjacent frame.  A failed
frame ends that temporal direction, so propagation cannot jump over a phase
whose image values no longer support the edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RegionGrowConfig:
    """Parameters for one stroke-seeded temporal propagation operation."""

    tolerance: float
    max_displacement_mm: float = 4.8
    temporal_radius: int | None = None
    replace_other_labels: bool = False
    max_changed_voxels: int = 500_000

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("Tolerance must be a finite non-negative value")
        if (
            not np.isfinite(self.max_displacement_mm)
            or self.max_displacement_mm < 0
        ):
            raise ValueError("Maximum displacement must be a finite non-negative value")
        if self.temporal_radius is not None and self.temporal_radius < 0:
            raise ValueError("Temporal radius must be non-negative or None")
        if self.max_changed_voxels < 1:
            raise ValueError("Maximum changed voxels must be positive")


@dataclass(frozen=True)
class RegionGrowResult:
    """The complete result of a pure temporal propagation operation.

    ``accepted_voxels`` is in canonical ``[X, Y, Z, T]`` order.  It contains
    the stroke seeds and the voxel-wise matches in each reached neighboring
    frame, so assigning all returned coordinates to the active label is valid.
    ``seed_median`` is reported in the source image's raw intensity space.
    """

    accepted_voxels: NDArray[np.intp]
    added_voxel_count: int
    replaced_voxel_count: int
    source_label_counts: dict[int, int]
    seed_median: float | None
    roi_slices: tuple[slice, slice, slice, slice]
    aborted: bool = False
    abort_reason: str | None = None

    @property
    def changed_voxel_count(self) -> int:
        return self.added_voxel_count + self.replaced_voxel_count


_SAFETY_ABORT_REASON = "maximum changed voxel safety limit exceeded"
_DISPLACEMENT_EPSILON = np.finfo(np.float32).eps


def _empty_result(
    roi_slices: tuple[slice, slice, slice, slice],
    seed_median: float | None = None,
    *,
    aborted: bool = False,
    abort_reason: str | None = None,
) -> RegionGrowResult:
    return RegionGrowResult(
        accepted_voxels=np.empty((0, 4), dtype=np.intp),
        added_voxel_count=0,
        replaced_voxel_count=0,
        source_label_counts={},
        seed_median=seed_median,
        roi_slices=roi_slices,
        aborted=aborted,
        abort_reason=abort_reason,
    )


def _as_seed_array(
    seeds: Iterable[tuple[int, int, int, int]] | NDArray[np.integer],
) -> NDArray[np.intp]:
    seed_array = np.asarray(list(seeds) if not isinstance(seeds, np.ndarray) else seeds)
    if seed_array.size == 0:
        return np.empty((0, 4), dtype=np.intp)
    if seed_array.ndim != 2 or seed_array.shape[1] != 4:
        raise ValueError("Stroke seeds must have shape (N, 4) in [X, Y, Z, T] order")
    if not np.issubdtype(seed_array.dtype, np.integer):
        raise ValueError("Stroke seed coordinates must be integers")
    return np.asarray(seed_array, dtype=np.intp)


def _temporal_offsets(
    spacing: NDArray[np.float64], max_displacement_mm: float
) -> NDArray[np.intp]:
    """Return physically valid displacement offsets in deterministic order."""

    limits = np.ceil(max_displacement_mm / spacing).astype(np.intp)
    offsets: list[tuple[float, int, int, int]] = []
    for dx in range(-int(limits[0]), int(limits[0]) + 1):
        for dy in range(-int(limits[1]), int(limits[1]) + 1):
            for dz in range(-int(limits[2]), int(limits[2]) + 1):
                distance_squared = (
                    (dx * spacing[0]) ** 2
                    + (dy * spacing[1]) ** 2
                    + (dz * spacing[2]) ** 2
                )
                if distance_squared <= max_displacement_mm**2 + _DISPLACEMENT_EPSILON:
                    offsets.append((float(distance_squared), dx, dy, dz))
    offsets.sort()
    return np.asarray([offset[1:] for offset in offsets], dtype=np.intp)


def grow_region_4d(
    image: NDArray[np.number],
    labels: NDArray[np.integer],
    stroke_seeds: Iterable[tuple[int, int, int, int]] | NDArray[np.integer],
    *,
    active_label: int,
    spacing_xyz: tuple[float, float, float],
    t0: int,
    config: RegionGrowConfig,
    threshold_mask: NDArray[np.bool_] | None = None,
) -> RegionGrowResult:
    """Propagate a stroke patch through adjacent frames without spatial growth.

    The starting frame accepts only valid stroke seeds.  For each neighboring
    frame, every source voxel independently selects its best finite candidate
    inside the physical displacement window, provided its raw intensity changes
    by no more than ``config.tolerance``.  A target can be selected by
    multiple sources but is accepted once; this permits the patch to shrink or
    deform without ever increasing through same-frame flood filling.
    """

    image_data = np.asarray(image)
    label_data = np.asarray(labels)
    if image_data.ndim != 4:
        raise ValueError("Image data must use canonical [X, Y, Z, T] ordering")
    if label_data.shape != image_data.shape:
        raise ValueError(
            "Label data must have the same [X, Y, Z, T] shape as image data"
        )
    if not np.issubdtype(label_data.dtype, np.integer):
        raise ValueError("Label data must have an integer dtype")
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    if (
        spacing.shape != (3,)
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0)
    ):
        raise ValueError("Spatial spacing must contain three positive finite values")
    if not 0 <= int(t0) < image_data.shape[3]:
        raise ValueError("t0 must be inside the image sequence")
    threshold = (
        None if threshold_mask is None else np.asarray(threshold_mask, dtype=bool)
    )
    if threshold is not None and threshold.shape != image_data.shape:
        raise ValueError("Threshold mask must match the image shape")

    seeds = _as_seed_array(stroke_seeds)
    shape = np.asarray(image_data.shape, dtype=np.intp)
    in_bounds = np.all((seeds >= 0) & (seeds < shape), axis=1)
    stroke_seeds = np.unique(seeds[in_bounds & (seeds[:, 3] == int(t0))], axis=0)
    if not stroke_seeds.size:
        return _empty_result(tuple(slice(0, 0) for _ in range(4)))

    radius = config.temporal_radius
    time_start = 0 if radius is None else max(0, int(t0) - int(radius))
    time_stop = image_data.shape[3] if radius is None else min(
        image_data.shape[3], int(t0) + int(radius) + 1
    )
    roi_slices = (
        slice(0, image_data.shape[0]),
        slice(0, image_data.shape[1]),
        slice(0, image_data.shape[2]),
        slice(time_start, time_stop),
    )
    image_roi = image_data[roi_slices]
    labels_roi = label_data[roi_slices]
    threshold_roi = None if threshold is None else threshold[roi_slices]
    local_seeds = stroke_seeds.copy()
    local_seeds[:, 3] -= time_start
    stroke_time = int(t0) - time_start
    active_label_value = int(active_label)

    # 保留原始强度，避免 P1/P99 截断改变用户设置的容差含义。
    raw_image = np.asarray(image_roi)

    def allowed_coordinate(coordinate: tuple[int, int, int], time_index: int) -> bool:
        coordinate_4d = coordinate + (time_index,)
        if threshold_roi is not None and not bool(threshold_roi[coordinate_4d]):
            return False
        label = int(labels_roi[coordinate_4d])
        return (
            label == 0
            or label == active_label_value
            or config.replace_other_labels
        ) and bool(np.isfinite(raw_image[coordinate + (time_index,)]))

    valid_seed_mask = np.asarray(
        [
            allowed_coordinate(tuple(int(value) for value in seed[:3]), stroke_time)
            for seed in local_seeds
        ],
        dtype=bool,
    )
    valid_seeds = local_seeds[valid_seed_mask]
    if not valid_seeds.size:
        return _empty_result(roi_slices)
    seed_values = raw_image[
        tuple(valid_seeds[:, axis] for axis in range(4))
    ]
    seed_median = float(np.median(seed_values.astype(np.float64, copy=False)))
    offsets = _temporal_offsets(spacing, config.max_displacement_mm)
    spatial_shape = image_roi.shape[:3]
    changed_count = 0
    safety_exceeded = False

    def record_change(coordinate: tuple[int, int, int], time_index: int) -> bool:
        nonlocal changed_count, safety_exceeded
        if int(labels_roi[coordinate + (time_index,)]) == active_label_value:
            return True
        changed_count += 1
        if changed_count > config.max_changed_voxels:
            safety_exceeded = True
            return False
        return True

    source = np.zeros(spatial_shape, dtype=bool)
    for seed in valid_seeds:
        coordinate = tuple(int(value) for value in seed[:3])
        if source[coordinate]:
            continue
        if not record_change(coordinate, stroke_time):
            return _empty_result(
                roi_slices,
                seed_median,
                aborted=True,
                abort_reason=_SAFETY_ABORT_REASON,
            )
        source[coordinate] = True

    accepted_by_time: dict[int, NDArray[np.bool_]] = {stroke_time: source}

    def best_match(
        parent: tuple[int, int, int], source_time: int, target_time: int
    ) -> tuple[int, int, int] | None:
        source_value = float(raw_image[parent + (source_time,)])
        best_key: tuple[float, int, int, int] | None = None
        best_coordinate: tuple[int, int, int] | None = None
        for offset in offsets:
            coordinate = tuple(int(parent[axis] + offset[axis]) for axis in range(3))
            if any(
                coordinate[axis] < 0 or coordinate[axis] >= spatial_shape[axis]
                for axis in range(3)
            ) or not allowed_coordinate(coordinate, target_time):
                continue
            difference = abs(
                float(raw_image[coordinate + (target_time,)]) - source_value
            )
            if difference > config.tolerance:
                continue
            key = (difference, *(abs(int(value)) for value in offset))
            if best_key is None or key < best_key:
                best_key = key
                best_coordinate = coordinate
        return best_coordinate

    # Each direction has an independent consecutive chain.  It cannot revive
    # after an empty frame, which prevents unsupported hard filling.
    max_steps = image_roi.shape[3] if radius is None else int(radius)
    for direction in (1, -1):
        source_time = stroke_time
        source_result = source
        for _step in range(max_steps):
            target_time = source_time + direction
            if target_time < 0 or target_time >= image_roi.shape[3]:
                break
            target = np.zeros(spatial_shape, dtype=bool)
            for parent_array in np.argwhere(source_result):
                parent = tuple(int(value) for value in parent_array)
                candidate = best_match(parent, source_time, target_time)
                if candidate is None or target[candidate]:
                    continue
                if not record_change(candidate, target_time):
                    return _empty_result(
                        roi_slices,
                        seed_median,
                        aborted=True,
                        abort_reason=_SAFETY_ABORT_REASON,
                    )
                target[candidate] = True
            if not np.any(target):
                break
            accepted_by_time[target_time] = target
            source_time = target_time
            source_result = target

    local_coordinates: list[NDArray[np.intp]] = []
    for local_time, accepted in accepted_by_time.items():
        spatial_coordinates = np.argwhere(accepted).astype(np.intp, copy=False)
        time_column = np.full(
            (spatial_coordinates.shape[0], 1), local_time, dtype=np.intp
        )
        local_coordinates.append(
            np.concatenate((spatial_coordinates, time_column), axis=1)
        )
    accepted_local = np.concatenate(local_coordinates, axis=0)
    accepted_coordinates = accepted_local.copy()
    accepted_coordinates[:, 3] += time_start
    original_labels = labels_roi[tuple(accepted_local[:, axis] for axis in range(4))]
    changed_labels = original_labels[original_labels != active_label_value]
    added = int(np.count_nonzero(changed_labels == 0))
    replaced_labels = changed_labels[changed_labels != 0]
    values, counts = np.unique(replaced_labels, return_counts=True)
    return RegionGrowResult(
        accepted_voxels=accepted_coordinates,
        added_voxel_count=added,
        replaced_voxel_count=int(replaced_labels.size),
        source_label_counts={
            int(value): int(count) for value, count in zip(values, counts)
        },
        seed_median=seed_median,
        roi_slices=roi_slices,
    )
