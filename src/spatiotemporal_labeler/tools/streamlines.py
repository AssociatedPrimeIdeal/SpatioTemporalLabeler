"""Instantaneous, physical-space streamlines and cached orthogonal slab views.

The scalar sequence loader reorients *sampling locations*, not vector values.
The Segdata exporters use HF/AP/RL components, so their positive directions
are F/P/L in RAS. Component directions must therefore be specified separately.
The batched midpoint integration and segment/slab clipping follow the approach
in autoflow's streamlines and slice_streamlines modules, without a dependency
on autoflow or PyVista. These are instantaneous streamlines, not pathlines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import map_coordinates


PLANES = {"X-Y": (0, 1, 2), "X-Z": (0, 2, 1), "Y-Z": (1, 2, 0)}
DIRECTIONS = {
    "R": (1, 0, 0), "L": (-1, 0, 0),
    "A": (0, 1, 0), "P": (0, -1, 0),
    "H": (0, 0, 1), "F": (0, 0, -1),
}


class StreamlineCancelled(Exception):
    pass


@dataclass(frozen=True)
class StreamlineConfig:
    directions: tuple[str, str, str] = ("F", "P", "L")
    seed_count: int = 300
    step_mm: float = 1.0
    max_steps: int = 160
    max_length_mm: float = 200.0
    min_speed: float = 5.0  # after input-unit calibration; native values may remain unscaled
    source_scale: float = 1.0
    component_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def validate(self) -> None:
        if len(self.directions) != 3 or any(d not in DIRECTIONS for d in self.directions):
            raise ValueError("Choose three anatomical component directions.")
        basis = np.array([DIRECTIONS[d] for d in self.directions], dtype=float).T
        if abs(np.linalg.det(basis)) < 0.5:
            raise ValueError("Components must cover each of R/L, A/P, and H/F once.")
        if not 1 <= self.seed_count <= 2000 or not 1 <= self.max_steps <= 1000:
            raise ValueError("Seed count or maximum steps is outside the supported range.")
        positive = (self.step_mm, self.max_length_mm, self.source_scale)
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Step, length, and source scale must be positive and finite.")
        if not np.isfinite(self.min_speed) or self.min_speed < 0:
            raise ValueError("Minimum speed must be nonnegative and finite.")
        if (len(self.component_scales) != 3
                or any(not np.isfinite(v) or v <= 0 for v in self.component_scales)):
            raise ValueError("Set a positive VENC for each phase component.")


@dataclass
class SliceSegments:
    """Packed [x0, y0, x1, y1, speed] rows, indexed without recomputation."""

    values: np.ndarray
    offsets: np.ndarray

    def at(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self.offsets) - 1:
            return np.empty((0, 5), dtype=np.float32)
        return self.values[self.offsets[index]:self.offsets[index + 1]]


@dataclass
class StreamlineFrame:
    world_segments: np.ndarray
    speeds: np.ndarray
    slices: dict[str, SliceSegments]

    @property
    def nbytes(self) -> int:
        return self.world_segments.nbytes + self.speeds.nbytes + sum(
            value.values.nbytes + value.offsets.nbytes for value in self.slices.values()
        )


def _check_cancel(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise StreamlineCancelled()


def validate_sources(sources) -> int:
    if len(sources) != 3 or any(source is None for source in sources):
        raise ValueError("Select an image sequence for each velocity component.")
    reference = sources[0]
    count = max(source.frame_count for source in sources)
    for source in sources:
        if not reference.spatially_compatible_with(source):
            raise ValueError("Velocity components must have matching voxel grids.")
        if source.frame_count not in (1, count):
            raise ValueError("Component frame counts must match; a 3D component may broadcast.")
    return count


def trace_streamlines(
    components, spacing, direction, config: StreamlineConfig,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed line segments in grid-local mm and calibrated magnitudes.

    Each trace uses a frozen 3D velocity frame, both upstream and downstream.
    Integrating in local physical coordinates with the inverse direction matrix
    also supports oblique grids. No component signs follow display flips.
    """
    config.validate()
    spacing = np.asarray(spacing, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Spacing must contain three positive finite values.")
    if direction.shape != (3, 3) or not np.all(np.isfinite(direction)):
        raise ValueError("Direction must be a finite 3 by 3 matrix.")
    arrays = [np.asarray(component, dtype=np.float32) for component in components]
    if len(arrays) != 3 or arrays[0].ndim != 3 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Components must be three matching 3D arrays.")
    _check_cancel(cancelled)
    basis = np.asarray([DIRECTIONS[d] for d in config.directions], dtype=float).T
    to_local = np.linalg.solve(direction, basis) * (config.source_scale * np.asarray(config.component_scales))
    field = np.einsum("...c,dc->...d", np.stack(arrays, axis=-1), to_local).astype(np.float32)
    finite = np.all(np.isfinite(field), axis=-1)
    field[~finite] = 0
    world_field = np.einsum("...c,dc->...d", field, direction)
    speed = np.linalg.norm(world_field, axis=-1)
    threshold = max(float(config.min_speed), 1e-6)
    occupied = np.flatnonzero(finite & (speed >= threshold))
    empty = (np.empty((0, 2, 3), np.float32), np.empty(0, np.float32))
    if not len(occupied):
        return empty
    rng = np.random.default_rng(0)
    picks = occupied[rng.choice(len(occupied), min(config.seed_count, len(occupied)), replace=False)]
    seeds = np.column_stack(np.unravel_index(picks, field.shape[:3])) * spacing
    del world_field, speed, occupied
    finite_bytes = finite.astype(np.uint8)

    def sample(points):
        coordinates = (points / spacing).T
        vectors = np.column_stack([
            map_coordinates(field[..., c], coordinates, order=1, mode="constant", cval=0.0,
                            prefilter=False) for c in range(3)
        ])
        magnitudes = np.linalg.norm(vectors @ direction.T, axis=1)
        inside = map_coordinates(finite_bytes, coordinates, order=0, mode="constant", cval=0)
        valid = (inside > 0) & np.isfinite(magnitudes) & (magnitudes >= threshold)
        return vectors / np.maximum(magnitudes[:, None], 1e-12), magnitudes, valid

    segments, speeds = [], []
    # The length cap is per entire seed trajectory, shared by both directions.
    step = min(config.step_mm, config.max_length_mm / 2.0)
    steps = min(config.max_steps, int(np.ceil(config.max_length_mm / (2.0 * step))))
    for sign in (-1.0, 1.0):
        points = seeds.copy()
        alive = np.arange(len(seeds))
        for index in range(steps):
            _check_cancel(cancelled)
            if not len(alive):
                break
            selected = points[alive]
            unit, start_speed, valid = sample(selected)
            distance = min(step, config.max_length_mm / 2.0 - index * step)
            middle_unit, _, middle_valid = sample(selected + sign * 0.5 * distance * unit)
            following = selected + sign * distance * middle_unit
            _, end_speed, end_valid = sample(following)
            valid &= middle_valid & end_valid
            if index > 8:
                # Stop closed loops after they return near their seed.
                valid &= np.linalg.norm((following - seeds[alive]) @ direction.T, axis=1) > 0.75 * step
            before, after = selected[valid], following[valid]
            if len(before):
                segments.append(np.stack((before, after) if sign > 0 else (after, before), axis=1))
                speeds.append((start_speed[valid] + end_speed[valid]) * 0.5)
            alive = alive[valid]
            points[alive] = after
    if not segments:
        return empty
    return np.concatenate(segments).astype(np.float32), np.concatenate(speeds).astype(np.float32)


def project_all_slices(
    segments: np.ndarray, speeds: np.ndarray, shape, spacing, origin, direction,
    slab_mm: float = 0.0, cancelled: Callable[[], bool] = lambda: False,
) -> StreamlineFrame:
    """Clip every segment to every slice's physical slab, retaining crossings.

    A zero thickness selects one voxel along each plane's fixed axis. Endpoints
    outside both faces still contribute when the segment crosses the slab.
    """
    if not np.isfinite(slab_mm) or slab_mm < 0:
        raise ValueError("Slab thickness must be nonnegative and finite.")
    spacing = np.asarray(spacing, dtype=float)
    segments = np.asarray(segments, dtype=np.float32).reshape(-1, 2, 3)
    speeds = np.asarray(speeds, dtype=np.float32).reshape(-1)
    if len(segments) != len(speeds):
        raise ValueError("Every segment must have one speed.")
    result = {}
    for plane, (h, v, fixed) in PLANES.items():
        a, b = segments[:, 0], segments[:, 1]
        da, db = a[:, fixed], b[:, fixed]
        delta = db - da
        parallel = np.abs(delta) < 1e-10
        safe_delta = np.where(parallel, 1.0, delta)
        half = (slab_mm if slab_mm > 0 else spacing[fixed]) * 0.5
        lower, upper = np.minimum(da, db) - half, np.maximum(da, db) + half
        values, offsets = [], [0]
        for index in range(shape[fixed]):
            _check_cancel(cancelled)
            center = index * spacing[fixed]
            candidates = np.flatnonzero((lower <= center) & (center <= upper))
            first = (center - half - da[candidates]) / safe_delta[candidates]
            second = (center + half - da[candidates]) / safe_delta[candidates]
            lo = np.where(parallel[candidates], 0.0, np.maximum(0.0, np.minimum(first, second)))
            hi = np.where(parallel[candidates], 1.0, np.minimum(1.0, np.maximum(first, second)))
            start = a[candidates] + lo[:, None] * (b[candidates] - a[candidates])
            end = a[candidates] + hi[:, None] * (b[candidates] - a[candidates])
            keep = (hi >= lo) & (np.linalg.norm(end[:, [h, v]] - start[:, [h, v]], axis=1) > 1e-7)
            rows = np.column_stack((start[keep][:, [h, v]], end[keep][:, [h, v]], speeds[candidates[keep]]))
            values.append(rows.astype(np.float32))
            offsets.append(offsets[-1] + len(rows))
        result[plane] = SliceSegments(np.concatenate(values), np.asarray(offsets, dtype=np.int64))
    world = segments @ np.asarray(direction, dtype=float).T + np.asarray(origin, dtype=float)
    return StreamlineFrame(world.astype(np.float32), speeds, result)


def speed_rgb(fractions: np.ndarray) -> np.ndarray:
    """Shared blue/cyan/yellow/red speed scale for 2D and VTK."""
    stops = np.array([[35, 65, 210], [0, 195, 235], [60, 220, 110],
                      [250, 225, 40], [235, 45, 30]], dtype=float)
    x = np.clip(np.asarray(fractions), 0, 1) * (len(stops) - 1)
    lo = np.minimum(x.astype(int), len(stops) - 2)
    return np.rint(stops[lo] + (x - lo)[..., None] * (stops[lo + 1] - stops[lo])).astype(np.uint8)
