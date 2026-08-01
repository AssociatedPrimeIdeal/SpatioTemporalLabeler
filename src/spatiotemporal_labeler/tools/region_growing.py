"""Pure, local stroke-seeded region growing for canonical ``[X, Y, Z, T]`` arrays.

The grow is deliberately ordered in time.  It completes a spatial 3D grow in
the stroke frame, then independently propagates that result one frame at a
time forward and backward.  This prevents a conventional 4D flood fill from
jumping to an unrelated structure in a later frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RegionGrowConfig:
    """Parameters for one local, stroke-seeded 3D+t grow operation."""

    tolerance: float
    spatial_margin_mm: float
    temporal_radius: int
    replace_other_labels: bool = False
    max_changed_voxels: int = 500_000

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("Tolerance must be a finite non-negative value")
        if not np.isfinite(self.spatial_margin_mm) or self.spatial_margin_mm < 0:
            raise ValueError("Spatial margin must be a finite non-negative value")
        if self.temporal_radius < 0:
            raise ValueError("Temporal radius must be non-negative")
        if self.max_changed_voxels < 1:
            raise ValueError("Maximum changed voxels must be positive")


@dataclass(frozen=True)
class RegionGrowResult:
    """The complete result of a pure grow operation.

    ``accepted_voxels`` uses canonical ``[X, Y, Z, T]`` coordinates. It includes
    already-labelled active voxels used as connectivity, so assigning every
    returned coordinate to the active label is always valid.
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


_SPATIAL_FACE_NEIGHBORS = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)
_SAFETY_ABORT_REASON = "maximum changed voxel safety limit exceeded"
_TEMPORAL_SUPPORT_RADIUS = 1
_TEMPORAL_TOLERANCE_MULTIPLIER = 1.5


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


def _priority(difference: float, tolerance: float) -> float:
    """Return a finite ordering value while making zero tolerance exact."""

    if tolerance == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / tolerance


def _support_metrics(
    source_accepted: NDArray[np.bool_],
    source_image: NDArray[np.number],
    target_image: NDArray[np.number],
    radius: int,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Return the target support dilation and best finite source difference.

    The 3x3x3 (or configured) local search is vectorized over each offset.  It
    avoids allocating a full 4D neighborhood tensor, which keeps release-time
    latency bounded by the local ROI rather than the full sequence.
    """

    spatial_shape = source_accepted.shape
    support = np.zeros(spatial_shape, dtype=bool)
    temporal_difference = np.full(spatial_shape, np.inf, dtype=np.float64)
    source_finite = np.isfinite(source_image)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                source_slices: list[slice] = []
                target_slices: list[slice] = []
                for axis, offset in enumerate((dx, dy, dz)):
                    size = spatial_shape[axis]
                    source_start = max(0, -offset)
                    source_stop = min(size, size - offset)
                    target_start = max(0, offset)
                    target_stop = min(size, size + offset)
                    source_slices.append(slice(source_start, source_stop))
                    target_slices.append(slice(target_start, target_stop))
                source_index = tuple(source_slices)
                target_index = tuple(target_slices)
                supported = source_accepted[source_index] & source_finite[source_index]
                if not np.any(supported):
                    continue
                support_view = support[target_index]
                support_view |= supported
                difference = np.abs(
                    np.asarray(target_image[target_index], dtype=np.float64)
                    - np.asarray(source_image[source_index], dtype=np.float64)
                )
                difference[~supported] = np.inf
                temporal_view = temporal_difference[target_index]
                np.minimum(temporal_view, difference, out=temporal_view)
    return support, temporal_difference


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
    """Compute a local sequential 3D+t region grow without mutating inputs.

    Valid stroke seeds are forced into the stroke-frame result.  The stroke
    frame grows with six spatial neighbors.  Each later frame is independently
    grown from the immediately preceding accepted frame in that direction, and
    every target candidate must remain in its Chebyshev support dilation.
    """

    image_data = np.asarray(image)
    label_data = np.asarray(labels)
    if image_data.ndim != 4:
        raise ValueError("Image data must use canonical [X, Y, Z, T] ordering")
    if label_data.shape != image_data.shape:
        raise ValueError("Label data must have the same [X, Y, Z, T] shape as image data")
    if not np.issubdtype(label_data.dtype, np.integer):
        raise ValueError("Label data must have an integer dtype")
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Spatial spacing must contain three positive finite values")
    if not 0 <= int(t0) < image_data.shape[3]:
        raise ValueError("t0 must be inside the image sequence")
    threshold = None if threshold_mask is None else np.asarray(threshold_mask, dtype=bool)
    if threshold is not None and threshold.shape != image_data.shape:
        raise ValueError("Threshold mask must match the image shape")

    seeds = _as_seed_array(stroke_seeds)
    shape = np.asarray(image_data.shape, dtype=np.intp)
    in_bounds = np.all((seeds >= 0) & (seeds < shape), axis=1)
    stroke_seeds_at_t0 = seeds[in_bounds & (seeds[:, 3] == int(t0))]
    if not stroke_seeds_at_t0.size:
        empty_roi = tuple(slice(0, 0) for _ in range(4))
        return _empty_result(empty_roi)

    margin_voxels = np.ceil(config.spatial_margin_mm / spacing).astype(np.intp)
    spatial_min = np.maximum(0, stroke_seeds_at_t0[:, :3].min(axis=0) - margin_voxels)
    spatial_max = np.minimum(
        shape[:3], stroke_seeds_at_t0[:, :3].max(axis=0) + margin_voxels + 1
    )
    time_start = max(0, int(t0) - int(config.temporal_radius))
    time_stop = min(image_data.shape[3], int(t0) + int(config.temporal_radius) + 1)
    roi_slices = (
        slice(int(spatial_min[0]), int(spatial_max[0])),
        slice(int(spatial_min[1]), int(spatial_max[1])),
        slice(int(spatial_min[2]), int(spatial_max[2])),
        slice(time_start, time_stop),
    )

    image_roi = image_data[roi_slices]
    labels_roi = label_data[roi_slices]
    threshold_roi = None if threshold is None else threshold[roi_slices]
    offsets = np.asarray(
        (spatial_min[0], spatial_min[1], spatial_min[2], time_start), dtype=np.intp
    )
    local_seeds = np.unique(stroke_seeds_at_t0 - offsets, axis=0)
    stroke_time = int(t0) - time_start
    active_label_value = int(active_label)

    def seed_is_valid(seed: NDArray[np.intp]) -> bool:
        coordinate = tuple(int(value) for value in seed)
        if not np.isfinite(image_roi[coordinate]):
            return False
        if threshold_roi is not None and not bool(threshold_roi[coordinate]):
            return False
        label = int(labels_roi[coordinate])
        return label in (0, active_label_value) or config.replace_other_labels

    valid_seed_mask = np.asarray([seed_is_valid(seed) for seed in local_seeds], dtype=bool)
    valid_seeds = local_seeds[valid_seed_mask]
    if not valid_seeds.size:
        return _empty_result(roi_slices)
    seed_values = image_roi[
        tuple(valid_seeds[:, axis] for axis in range(4))
    ].astype(np.float64, copy=False)
    seed_median = float(np.median(seed_values))
    spatial_shape = image_roi.shape[:3]
    temporal_tolerance = config.tolerance * _TEMPORAL_TOLERANCE_MULTIPLIER
    changed_count = 0
    safety_exceeded = False

    def record_change(label: int) -> bool:
        """Record a proposed mutation and preserve atomic failure semantics."""

        nonlocal changed_count, safety_exceeded
        if label == active_label_value:
            return True
        changed_count += 1
        if changed_count > config.max_changed_voxels:
            safety_exceeded = True
            return False
        return True

    def allowed_label(label: int) -> bool:
        return label == 0 or label == active_label_value or config.replace_other_labels

    def grow_stroke_frame() -> NDArray[np.bool_] | None:
        """Complete the local six-connected grow in the frame carrying the stroke."""

        accepted = np.zeros(spatial_shape, dtype=bool)
        examined = np.zeros(spatial_shape, dtype=bool)
        heap: list[tuple[float, int, int, int]] = []
        for seed in valid_seeds:
            coordinate = tuple(int(value) for value in seed[:3])
            if accepted[coordinate]:
                continue
            label = int(labels_roi[coordinate + (stroke_time,)])
            if not record_change(label):
                return None
            accepted[coordinate] = True
            examined[coordinate] = True
            heapq.heappush(heap, (0.0, *coordinate))

        while heap:
            _cost, x, y, z = heapq.heappop(heap)
            for dx, dy, dz in _SPATIAL_FACE_NEIGHBORS:
                neighbor = (x + dx, y + dy, z + dz)
                if any(
                    coordinate < 0 or coordinate >= spatial_shape[axis]
                    for axis, coordinate in enumerate(neighbor)
                ) or examined[neighbor]:
                    continue
                examined[neighbor] = True
                coordinate_4d = neighbor + (stroke_time,)
                if threshold_roi is not None and not bool(threshold_roi[coordinate_4d]):
                    continue
                label = int(labels_roi[coordinate_4d])
                if not allowed_label(label):
                    continue
                # Existing active-label voxels are connectivity, not new intensity
                # candidates.  They remain constrained by ROI and threshold.
                if label == active_label_value:
                    accepted[neighbor] = True
                    heapq.heappush(heap, (0.0, *neighbor))
                    continue
                intensity = float(image_roi[coordinate_4d])
                if not np.isfinite(intensity):
                    continue
                difference = abs(intensity - seed_median)
                if difference > config.tolerance:
                    continue
                if not record_change(label):
                    return None
                accepted[neighbor] = True
                heapq.heappush(
                    heap, (_priority(difference, config.tolerance), *neighbor)
                )
        return accepted

    def grow_supported_frame(
        target_time: int, source_accepted: NDArray[np.bool_]
    ) -> NDArray[np.bool_] | None:
        """Grow one frame constrained by the immediately adjacent source frame."""

        support, temporal_difference = _support_metrics(
            source_accepted,
            image_roi[..., target_time - 1]
            if target_time > stroke_time
            else image_roi[..., target_time + 1],
            image_roi[..., target_time],
            _TEMPORAL_SUPPORT_RADIUS,
        )
        if not np.any(support):
            return np.zeros(spatial_shape, dtype=bool)
        accepted = np.zeros(spatial_shape, dtype=bool)
        examined = np.zeros(spatial_shape, dtype=bool)
        heap: list[tuple[float, int, int, int]] = []

        def consider(coordinate: tuple[int, int, int]) -> None:
            if examined[coordinate] or safety_exceeded:
                return
            examined[coordinate] = True
            if not support[coordinate]:
                return
            coordinate_4d = coordinate + (target_time,)
            if threshold_roi is not None and not bool(threshold_roi[coordinate_4d]):
                return
            label = int(labels_roi[coordinate_4d])
            if not allowed_label(label):
                return
            intensity = float(image_roi[coordinate_4d])
            time_difference = float(temporal_difference[coordinate])
            if not np.isfinite(intensity) or not np.isfinite(time_difference):
                return
            # Unlike the stroke frame, every target-frame voxel, including an
            # existing active label, must satisfy both intensity conditions.
            intensity_difference = abs(intensity - seed_median)
            if (
                intensity_difference > config.tolerance
                or time_difference > temporal_tolerance
            ):
                return
            priority = max(
                _priority(intensity_difference, config.tolerance),
                _priority(time_difference, temporal_tolerance),
            )
            if not record_change(label):
                return
            accepted[coordinate] = True
            heapq.heappush(heap, (priority, *coordinate))

        # All temporally-supported accepted source locations are the initial
        # roots.  Subsequent expansion is six-connected and remains in support.
        for coordinate_array in np.argwhere(support):
            consider(tuple(int(value) for value in coordinate_array))
            if safety_exceeded:
                return None
        while heap:
            _cost, x, y, z = heapq.heappop(heap)
            for dx, dy, dz in _SPATIAL_FACE_NEIGHBORS:
                neighbor = (x + dx, y + dy, z + dz)
                if all(
                    0 <= coordinate < spatial_shape[axis]
                    for axis, coordinate in enumerate(neighbor)
                ):
                    consider(neighbor)
                    if safety_exceeded:
                        return None
        return accepted

    accepted_by_time: dict[int, NDArray[np.bool_]] = {}
    stroke_result = grow_stroke_frame()
    if stroke_result is None or safety_exceeded:
        return _empty_result(
            roi_slices, seed_median, aborted=True, abort_reason=_SAFETY_ABORT_REASON
        )
    accepted_by_time[stroke_time] = stroke_result

    # Each direction only uses the result from its immediate predecessor.  An
    # empty target terminates that direction rather than allowing a time jump.
    for direction in (1, -1):
        source_time = stroke_time
        for _step in range(config.temporal_radius):
            target_time = source_time + direction
            if target_time < 0 or target_time >= image_roi.shape[3]:
                break
            source_result = accepted_by_time[source_time]
            target_result = grow_supported_frame(target_time, source_result)
            if target_result is None or safety_exceeded:
                return _empty_result(
                    roi_slices,
                    seed_median,
                    aborted=True,
                    abort_reason=_SAFETY_ABORT_REASON,
                )
            if not np.any(target_result):
                break
            accepted_by_time[target_time] = target_result
            source_time = target_time

    local_coordinates: list[NDArray[np.intp]] = []
    for local_time, accepted in accepted_by_time.items():
        spatial_coordinates = np.argwhere(accepted).astype(np.intp, copy=False)
        if not spatial_coordinates.size:
            continue
        time_column = np.full((spatial_coordinates.shape[0], 1), local_time, dtype=np.intp)
        local_coordinates.append(np.concatenate((spatial_coordinates, time_column), axis=1))
    if not local_coordinates:
        return _empty_result(roi_slices, seed_median)
    accepted_local_coordinates = np.concatenate(local_coordinates, axis=0)
    accepted_coordinates = accepted_local_coordinates + offsets
    original_labels = labels_roi[
        tuple(accepted_local_coordinates[:, axis] for axis in range(4))
    ]
    changed_labels = original_labels[original_labels != active_label_value]
    added = int(np.count_nonzero(changed_labels == 0))
    replaced_labels = changed_labels[changed_labels != 0]
    sources, counts = np.unique(replaced_labels, return_counts=True)
    source_label_counts = {
        int(source): int(count) for source, count in zip(sources, counts)
    }
    return RegionGrowResult(
        accepted_voxels=accepted_coordinates,
        added_voxel_count=added,
        replaced_voxel_count=int(replaced_labels.size),
        source_label_counts=source_label_counts,
        seed_median=seed_median,
        roi_slices=roi_slices,
    )
