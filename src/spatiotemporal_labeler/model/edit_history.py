from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spatiotemporal_labeler.io import Sequence4D


@dataclass
class EditCommand:
    """Sparse voxel changes produced by one atomic editing gesture."""

    mask: Sequence4D
    flat_indices: NDArray[np.intp]
    before_values: NDArray
    after_values: NDArray
    focus_frame: int

    def undo(self) -> None:
        self.mask.data.flat[self.flat_indices] = self.before_values
        self.mask.dirty = True

    def redo(self) -> None:
        self.mask.data.flat[self.flat_indices] = self.after_values
        self.mask.dirty = True

    def changed_label_values(self) -> set[int]:
        """Return positive label values affected on either side of this edit."""
        if not self.before_values.size and not self.after_values.size:
            return set()
        values = np.union1d(np.unique(self.before_values), np.unique(self.after_values))
        return {int(value) for value in values if int(value) > 0}


def _contiguous_frame_slice(frames: tuple[int, ...]) -> slice | None:
    if not frames:
        return None
    start = int(frames[0])
    if frames == tuple(range(start, start + len(frames))):
        return slice(start, start + len(frames))
    return None


def _spatial_slices(
    mask: Sequence4D, bounds: tuple[slice, slice, slice] | None,
) -> tuple[slice, slice, slice]:
    selected = bounds or tuple(slice(0, length) for length in mask.shape_xyz)
    if len(selected) != 3:
        raise ValueError("Edit bounds must contain one slice for each spatial axis")
    normalized = []
    for bound, length in zip(selected, mask.shape_xyz):
        start, stop, step = bound.indices(length)
        if step != 1:
            raise ValueError("Edit bounds must use a step of 1")
        normalized.append(slice(start, max(start, stop)))
    return tuple(normalized)


def capture_frames(
    mask: Sequence4D,
    frames: tuple[int, ...],
    spatial_bounds: tuple[slice, slice, slice] | None = None,
) -> NDArray:
    """Copy selected frames without advanced indexing when they are contiguous."""
    if not frames or any(frame < 0 or frame >= mask.frame_count for frame in frames):
        raise ValueError("Edit frames must be inside the label sequence")
    frame_slice = _contiguous_frame_slice(frames)
    bounds = _spatial_slices(mask, spatial_bounds)
    if frame_slice is not None:
        return mask.data[bounds + (frame_slice,)].copy()
    return mask.data[bounds + (np.asarray(frames, dtype=np.intp),)].copy()


def restore_frames(
    mask: Sequence4D,
    frames: tuple[int, ...],
    snapshot: NDArray,
    spatial_bounds: tuple[slice, slice, slice] | None = None,
) -> None:
    """Restore a snapshot captured by :func:`capture_frames`."""
    bounds = _spatial_slices(mask, spatial_bounds)
    expected_shape = (*(bound.stop - bound.start for bound in bounds), len(frames))
    if snapshot.shape != expected_shape:
        raise ValueError(f"Edit snapshot shape {snapshot.shape} does not match {expected_shape}")
    frame_slice = _contiguous_frame_slice(frames)
    if frame_slice is not None:
        mask.data[bounds + (frame_slice,)] = snapshot
        return
    for local_frame, frame in enumerate(frames):
        mask.data[bounds + (frame,)] = snapshot[..., local_frame]


def build_edit_command(
    mask: Sequence4D,
    frames: tuple[int, ...],
    before: NDArray,
    focus_frame: int,
    spatial_bounds: tuple[slice, slice, slice] | None = None,
    snapshot_bounds: tuple[slice, slice, slice] | None = None,
) -> EditCommand | None:
    """Build a sparse command by comparing captured frames with their current values."""
    frame_indices = np.asarray(frames, dtype=np.intp)
    captured_bounds = _spatial_slices(mask, snapshot_bounds)
    expected_shape = (*(bound.stop - bound.start for bound in captured_bounds), len(frames))
    if before.shape != expected_shape:
        raise ValueError(f"Edit snapshot shape {before.shape} does not match {expected_shape}")

    spatial_slices = _spatial_slices(mask, spatial_bounds or snapshot_bounds)
    if any(
        bound.start < captured.start or bound.stop > captured.stop
        for bound, captured in zip(spatial_slices, captured_bounds)
    ):
        raise ValueError("Edit bounds must be inside the captured snapshot")
    local_slices = tuple(
        slice(bound.start - captured.start, bound.stop - captured.start)
        for bound, captured in zip(spatial_slices, captured_bounds)
    )
    offsets = [bound.start for bound in spatial_slices]
    before_region = before[local_slices + (slice(None),)]
    frame_slice = _contiguous_frame_slice(frames)
    frame_selection: slice | NDArray[np.intp] = (
        frame_slice if frame_slice is not None else frame_indices
    )
    after_region = mask.data[spatial_slices + (frame_selection,)]
    changed = before_region != after_region
    if not np.any(changed):
        return None

    x, y, z, local_time = np.nonzero(changed)
    x += offsets[0]
    y += offsets[1]
    z += offsets[2]
    time = frame_indices[local_time]
    flat_indices = np.ravel_multi_index((x, y, z, time), mask.data.shape).astype(
        np.intp, copy=False
    )
    return EditCommand(
        mask=mask,
        flat_indices=flat_indices,
        before_values=before_region[changed].copy(),
        after_values=after_region[changed].copy(),
        focus_frame=focus_frame,
    )
