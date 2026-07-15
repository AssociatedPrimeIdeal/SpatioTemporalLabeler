from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


def _signed_distance(
    selection: NDArray,
    spacing_xyz: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Return a boundary-aware signed distance field in millimetres."""
    binary = np.asarray(selection, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Label interpolation expects 3D keyframes, got {binary.shape}")
    spacing = np.asarray(spacing_xyz, dtype=float)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Spacing must contain three positive values")

    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    crop = tuple(slice(1, -1) for _ in range(3))
    inside = ndimage.distance_transform_edt(padded, sampling=spacing)[crop]
    outside = ndimage.distance_transform_edt(~padded, sampling=spacing)[crop]
    return np.asarray(inside - outside, dtype=np.float64)


def interpolate_label_frames(
    data: NDArray,
    start_frame: int,
    end_frame: int,
    label_values: Iterable[int],
    *,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Interpolate multilabel keyframes with physical signed distance fields.

    The returned array contains only frames strictly between the two keyframes.
    Labels outside ``label_values`` are preserved and block interpolated additions.
    """
    source = np.asarray(data)
    if source.ndim != 4:
        raise ValueError(f"Label interpolation expects 3D+t data, got {source.shape}")
    start, end = int(start_frame), int(end_frame)
    if not 0 <= start < end < source.shape[3]:
        raise ValueError("Keyframes must be ordered and inside the label sequence")
    if end - start < 2:
        raise ValueError("At least one frame is required between the keyframes")

    requested = tuple(sorted({int(value) for value in label_values if int(value) > 0}))
    fields: list[tuple[int, NDArray[np.float64], NDArray[np.float64]]] = []
    missing: list[int] = []
    for value in requested:
        start_selection = source[..., start] == value
        end_selection = source[..., end] == value
        if not np.any(start_selection) and not np.any(end_selection):
            continue
        if not np.any(start_selection) or not np.any(end_selection):
            missing.append(value)
            continue
        fields.append(
            (
                value,
                _signed_distance(start_selection, spacing_xyz),
                _signed_distance(end_selection, spacing_xyz),
            )
        )
    if missing:
        values = ", ".join(str(value) for value in missing)
        raise ValueError(f"Labels must exist in both keyframes: {values}")
    if not fields:
        raise ValueError("No selected labels exist in both keyframes")

    result = source[..., start + 1 : end].copy()
    selected_values = np.asarray([field[0] for field in fields], dtype=source.dtype)
    result[np.isin(result, selected_values)] = 0
    span = float(end - start)
    for offset, frame in enumerate(range(start + 1, end)):
        alpha = (frame - start) / span
        scores = np.stack(
            [(1.0 - alpha) * first + alpha * last for _, first, last in fields],
            axis=0,
        )
        winning_index = np.argmax(scores, axis=0)
        winning_score = np.take_along_axis(scores, winning_index[None, ...], axis=0)[0]
        available = result[..., offset] == 0
        for index, value in enumerate(selected_values):
            result[..., offset][available & (winning_index == index) & (winning_score >= 0)] = value
    return result
