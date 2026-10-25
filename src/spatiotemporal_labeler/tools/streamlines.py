"""Instantaneous, physical-space streamlines and cached orthogonal slab views.

The scalar sequence loader reorients *sampling locations*, not vector values.
The Segdata exporters use HF/AP/RL components, so their positive directions
are F/P/L in RAS. Component directions must therefore be specified separately.
The batched midpoint integration and segment/slab clipping follow the approach
in autoflow's streamlines and slice_streamlines modules, without a dependency
on autoflow or PyVista. These are instantaneous streamlines, not pathlines.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    seed_min_speed: float = 0.0
    seed_distribution: str = "grid"
    grid_spacing_mm: float = 3.0
    grid_units: str = "voxel"
    grid_spacing_voxels: float = 3.0
    limit_grid_seeds: bool = False
    rng_seed: int = 0
    integration_direction: str = "both"
    integrator: str = "midpoint"
    min_length_mm: float = 0.0
    max_turn_degrees: float = 180.0
    constrain_to_mask: bool = False

    def validate(self) -> None:
        if len(self.directions) != 3 or any(d not in DIRECTIONS for d in self.directions):
            raise ValueError("Choose three anatomical component directions.")
        basis = np.array([DIRECTIONS[d] for d in self.directions], dtype=float).T
        if abs(np.linalg.det(basis)) < 0.5:
            raise ValueError("Components must cover each of R/L, A/P, and H/F once.")
        if self.seed_count < 1 or self.max_steps < 1:
            raise ValueError("Seed count and maximum steps must be positive.")
        positive = (self.step_mm, self.max_length_mm, self.source_scale)
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Step, length, and source scale must be positive and finite.")
        if not np.isfinite(self.min_speed) or self.min_speed < 0:
            raise ValueError("Minimum speed must be nonnegative and finite.")
        if (len(self.component_scales) != 3
                or any(not np.isfinite(v) or v <= 0 for v in self.component_scales)):
            raise ValueError("Set a positive VENC for each phase component.")
        if (not np.isfinite(self.seed_min_speed) or self.seed_min_speed < 0
                or not np.isfinite(self.grid_spacing_mm) or self.grid_spacing_mm <= 0):
            raise ValueError("Seed threshold and grid spacing are invalid.")
        if self.seed_distribution not in {"random", "grid"} or self.rng_seed < 0:
            raise ValueError("Seed distribution or random seed is invalid.")
        if (self.grid_units not in {"voxel", "mm"} or not np.isfinite(self.grid_spacing_voxels)
                or self.grid_spacing_voxels <= 0):
            raise ValueError("Grid interval must be positive and use voxels or mm.")
        if self.integration_direction not in {"both", "forward", "backward"}:
            raise ValueError("Choose forward, backward, or bidirectional integration.")
        if self.integrator not in {"midpoint", "rk4"}:
            raise ValueError("Choose midpoint or RK4 integration.")
        if (not np.isfinite(self.min_length_mm) or self.min_length_mm < 0
                or self.min_length_mm > self.max_length_mm
                or not np.isfinite(self.max_turn_degrees)
                or not 0 < self.max_turn_degrees <= 180):
            raise ValueError("Minimum length or maximum turn angle is invalid.")


@dataclass(frozen=True)
class PlanarConfig:
    enabled: bool = False
    seed_count: int = 40  # maximum seeds per individual spatial slice

    def validate(self):
        if self.seed_count < 1:
            raise ValueError("Planar seed count must be positive.")


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
    planar_slices: dict[str, SliceSegments] | None = None

    @property
    def nbytes(self) -> int:
        return self.world_segments.nbytes + self.speeds.nbytes + sum(
            value.values.nbytes + value.offsets.nbytes
            for group in (self.slices, self.planar_slices or {}) for value in group.values()
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


def _world_field(components, config):
    arrays = [np.asarray(component, dtype=np.float32) for component in components]
    if len(arrays) != 3 or arrays[0].ndim != 3 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Components must be three matching 3D arrays.")
    basis = np.asarray([DIRECTIONS[d] for d in config.directions], dtype=float).T
    basis *= config.source_scale * np.asarray(config.component_scales)
    return np.einsum("...c,dc->...d", np.stack(arrays, axis=-1), basis).astype(np.float32)


def _integrate_field(field, spacing, basis, config, seed_mask, cancelled):
    """Integrate 2D or 3D grid-local vectors with a physical world-space metric."""
    _check_cancel(cancelled)
    ndim = field.shape[-1]
    spacing = np.asarray(spacing, dtype=float)
    basis = np.asarray(basis, dtype=float)
    field = np.asarray(field, dtype=np.float32)
    finite = np.all(np.isfinite(field), axis=-1)
    field = np.where(finite[..., None], field, 0)
    if seed_mask is not None:
        seed_mask = np.asarray(seed_mask, dtype=bool)
        if seed_mask.shape != field.shape[:-1]:
            raise ValueError("Seed mask must match the component voxel grid.")
    speed = np.linalg.norm(np.einsum("...c,dc->...d", field, basis), axis=-1)
    threshold = max(float(config.min_speed), 1e-8)
    eligible = finite & (speed >= max(threshold, config.seed_min_speed))
    if seed_mask is not None:
        eligible &= seed_mask
    occupied = np.flatnonzero(eligible)
    empty = (np.empty((0, 2, ndim), np.float32), np.empty(0, np.float32))
    if not len(occupied):
        return empty
    rng = np.random.default_rng(config.rng_seed)
    if config.seed_distribution == "grid":
        coordinates = np.column_stack(np.unravel_index(occupied, field.shape[:-1]))
        grid_interval = (spacing * config.grid_spacing_voxels if config.grid_units == "voxel"
                         else np.full(ndim, config.grid_spacing_mm))
        # Match autoflow's regular, half-cell-offset physical lattice. Unlike
        # selecting one voxel per occupied bin, this retains the exact pitch.
        axes = []
        for axis in range(ndim):
            first = max(0, int(np.ceil(coordinates[:, axis].min() * spacing[axis] / grid_interval[axis] - 0.5)))
            last = int(np.ceil((coordinates[:, axis].max() + 1) * spacing[axis] / grid_interval[axis] - 0.5))
            axes.append((np.arange(first, last) + 0.5) * grid_interval[axis])
        sizes = tuple(len(axis) for axis in axes)
        count = int(np.prod(sizes, dtype=np.int64))
        if count > 20000000:
            raise ValueError("Grid exceeds 20 million candidates; increase the grid interval.")
        parts = []
        for start in range(0, count, 131072):
            _check_cancel(cancelled)
            indices = np.unravel_index(np.arange(start, min(start + 131072, count)), sizes)
            points = np.column_stack([axes[axis][indices[axis]] for axis in range(ndim)])
            voxels = np.floor(points / spacing + 0.5).astype(np.int64)
            valid = np.all((voxels >= 0) & (voxels < np.asarray(field.shape[:-1])), axis=1)
            rows = np.flatnonzero(valid)
            valid[rows] &= eligible[tuple(voxels[rows].T)]
            parts.append(points[valid])
        seeds = np.concatenate(parts) if parts else np.empty((0, ndim))
        if config.limit_grid_seeds and len(seeds) > config.seed_count:
            seeds = seeds[np.linspace(0, len(seeds) - 1, config.seed_count, dtype=int)]
    else:
        picks = occupied[rng.choice(len(occupied), min(config.seed_count, len(occupied)), replace=False)]
        seeds = np.column_stack(np.unravel_index(picks, field.shape[:-1])) * spacing
    del speed, occupied, eligible
    support = finite.copy()
    if config.constrain_to_mask and seed_mask is not None:
        support &= seed_mask
    support = support.astype(np.uint8)

    def sample(points):
        coordinates = (points / spacing).T
        vectors = np.column_stack([
            map_coordinates(field[..., c], coordinates, order=1, mode="constant", cval=0.0,
                            prefilter=False) for c in range(ndim)
        ])
        magnitudes = np.linalg.norm(vectors @ basis.T, axis=1)
        inside = map_coordinates(support, coordinates, order=0, mode="constant", cval=0)
        valid = (inside > 0) & np.isfinite(magnitudes) & (magnitudes >= threshold)
        return vectors / np.maximum(magnitudes[:, None], 1e-12), magnitudes, valid

    signs = {"both": (-1.0, 1.0), "forward": (1.0,), "backward": (-1.0,)}[config.integration_direction]
    length_budget = config.max_length_mm / len(signs)
    step = min(config.step_mm, length_budget)
    steps = min(config.max_steps, int(np.ceil(length_budget / step)))
    turn_limit = np.cos(np.deg2rad(config.max_turn_degrees))
    output_segments, output_speeds = [], []
    # Large seed requests run in bounded batches, with no silent density cap.
    for batch_start in range(0, len(seeds), 512):
        _check_cancel(cancelled)
        batch = seeds[batch_start:batch_start + 512]
        _, seed_speeds, seed_valid = sample(batch)
        batch = batch[seed_valid & (seed_speeds >= max(threshold, config.seed_min_speed))]
        if not len(batch):
            continue
        segments, speeds, owners = [], [], []
        for sign in signs:
            points = batch.copy()
            alive = np.arange(len(batch))
            for index in range(steps):
                _check_cancel(cancelled)
                if not len(alive):
                    break
                selected = points[alive]
                k1, start_speed, valid = sample(selected)
                distance = min(step, length_budget - index * step)
                k2, _, valid2 = sample(selected + sign * 0.5 * distance * k1)
                valid &= valid2
                if config.integrator == "rk4":
                    k3, _, valid3 = sample(selected + sign * 0.5 * distance * k2)
                    k4, _, valid4 = sample(selected + sign * distance * k3)
                    tangent = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                    valid &= valid3 & valid4
                else:
                    tangent = k2
                following = selected + sign * distance * tangent
                end_unit, end_speed, end_valid = sample(following)
                valid &= end_valid
                if config.max_turn_degrees < 180:
                    alignment = np.einsum("ij,ij->i", k1 @ basis.T, end_unit @ basis.T)
                    valid &= alignment >= turn_limit
                if index > 8:
                    valid &= np.linalg.norm((following - batch[alive]) @ basis.T, axis=1) > 0.75 * step
                before, after = selected[valid], following[valid]
                if len(before):
                    segments.append(np.stack((before, after) if sign > 0 else (after, before), axis=1))
                    speeds.append((start_speed[valid] + end_speed[valid]) * 0.5)
                    owners.append(alive[valid].copy())
                alive = alive[valid]
                points[alive] = after
        if segments:
            batch_segments = np.concatenate(segments).astype(np.float32)
            batch_speeds = np.concatenate(speeds).astype(np.float32)
            if config.min_length_mm > 0:
                owners = np.concatenate(owners)
                lengths = np.linalg.norm((batch_segments[:, 1] - batch_segments[:, 0]) @ basis.T, axis=1)
                totals = np.bincount(owners, weights=lengths, minlength=len(batch))
                keep = totals[owners] >= config.min_length_mm
                batch_segments, batch_speeds = batch_segments[keep], batch_speeds[keep]
            output_segments.append(batch_segments)
            output_speeds.append(batch_speeds)
    if not output_segments:
        return empty
    return np.concatenate(output_segments), np.concatenate(output_speeds)


def trace_streamlines(
    components, spacing, direction, config: StreamlineConfig,
    cancelled: Callable[[], bool] = lambda: False, *, seed_mask=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace a frozen 3D field; mask restricts seeds and optionally trajectories."""
    config.validate()
    spacing = np.asarray(spacing, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Spacing must contain three positive finite values.")
    if direction.shape != (3, 3) or not np.all(np.isfinite(direction)):
        raise ValueError("Direction must be a finite 3 by 3 matrix.")
    field = _world_field(components, config)
    field = np.einsum("...c,dc->...d", field, np.linalg.inv(direction)).astype(np.float32)
    return _integrate_field(field, spacing, direction, config, seed_mask, cancelled)


def trace_planar_slices(
    components, spacing, direction, config: StreamlineConfig, planar: PlanarConfig,
    cancelled: Callable[[], bool] = lambda: False, *, seed_mask=None,
) -> dict[str, SliceSegments]:
    """Cache independent in-plane streamlines on every XY/XZ/YZ slice.

    Project the physical vector onto the slice's tangent basis, discarding its
    normal component before seeding, termination, integration, and coloring.
    A large through-plane speed cannot create spurious in-plane streamlines.
    """
    config.validate()
    planar.validate()
    world = _world_field(components, config)
    shape = world.shape[:3]
    spacing = np.asarray(spacing, dtype=float)
    direction = np.asarray(direction, dtype=float)
    planar_config = replace(config, seed_count=planar.seed_count)
    result = {}
    for plane, (h, v, fixed) in PLANES.items():
        basis = direction[:, [h, v]]
        projector = np.linalg.pinv(basis)
        values, offsets = [], [0]
        for index in range(shape[fixed]):
            _check_cancel(cancelled)
            field = np.einsum("...c,dc->...d", np.take(world, index, axis=fixed), projector).astype(np.float32)
            mask = None if seed_mask is None else np.take(seed_mask, index, axis=fixed)
            segments, speeds = _integrate_field(field, spacing[[h, v]], basis,
                                                planar_config, mask, cancelled)
            rows = np.column_stack((segments.reshape(-1, 4), speeds)).astype(np.float32)
            values.append(rows)
            offsets.append(offsets[-1] + len(rows))
        result[plane] = SliceSegments(np.concatenate(values), np.asarray(offsets, dtype=np.int64))
    return result


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
