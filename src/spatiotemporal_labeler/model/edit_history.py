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


def build_edit_command(
    mask: Sequence4D,
    frames: tuple[int, ...],
    before: NDArray,
    focus_frame: int,
) -> EditCommand | None:
    """Build a sparse command by comparing captured frames with their current values."""
    frame_indices = np.asarray(frames, dtype=np.intp)
    expected_shape = (*mask.shape_xyz, len(frames))
    if before.shape != expected_shape:
        raise ValueError(f"Edit snapshot shape {before.shape} does not match {expected_shape}")

    after = mask.data[..., frame_indices]
    changed = before != after
    if not np.any(changed):
        return None

    x, y, z, local_time = np.nonzero(changed)
    time = frame_indices[local_time]
    flat_indices = np.ravel_multi_index((x, y, z, time), mask.data.shape).astype(
        np.intp, copy=False
    )
    return EditCommand(
        mask=mask,
        flat_indices=flat_indices,
        before_values=before[changed].copy(),
        after_values=after[changed].copy(),
        focus_frame=focus_frame,
    )
