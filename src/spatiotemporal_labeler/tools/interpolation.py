from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


def keyframe_intermediate_frames(
    keyframes: Iterable[int], frame_count: int, *, wrap: bool = True
) -> tuple[int, ...]:
    """Return all non-keyframe indices covered by consecutive keyframe spans."""
    count = int(frame_count)
    if count < 1:
        raise ValueError("The label sequence must contain at least one frame")
    frames = tuple(int(value) for value in keyframes)
    if len(frames) < 2:
        raise ValueError("At least two label keyframes are required")
    if len(set(frames)) != len(frames):
        raise ValueError("Label keyframes must be unique")
    if any(value < 0 or value >= count for value in frames):
        raise ValueError("Label keyframes must be inside the sequence")
    if frames != tuple(sorted(frames)):
        raise ValueError("Label keyframes must be in ascending time order")

    paths: list[tuple[int, ...]] = []
    for start, end in zip(frames, frames[1:]):
        paths.append(tuple(range(start + 1, end)))
    if wrap:
        span = (frames[0] - frames[-1]) % count
        paths.append(tuple((frames[-1] + offset) % count for offset in range(1, span)))
    return tuple(frame for path in paths for frame in path)


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
    wrap: bool = False,
    progress: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Interpolate multilabel keyframes with physical signed distance fields.

    The returned array contains only frames strictly between the two keyframes,
    in forward time order. With ``wrap=True``, that order follows the cyclic
    path across the last frame back to frame zero. Labels outside
    ``label_values`` are preserved and block interpolated additions.
    """
    source = np.asarray(data)
    if source.ndim != 4:
        raise ValueError(f"Label interpolation expects 3D+t data, got {source.shape}")
    start, end = int(start_frame), int(end_frame)
    frame_count = source.shape[3]
    if not 0 <= start < frame_count or not 0 <= end < frame_count:
        raise ValueError("Keyframes must be inside the label sequence")
    if wrap:
        span = (end - start) % frame_count
        if span == 0:
            raise ValueError("Wrapped keyframes must be different")
        frames = tuple((start + offset) % frame_count for offset in range(1, span))
    else:
        if not start < end:
            raise ValueError("Keyframes must be ordered and inside the label sequence")
        span = end - start
        frames = tuple(range(start + 1, end))
    if span < 2:
        raise ValueError("At least one frame is required between the keyframes")

    requested = tuple(sorted({int(value) for value in label_values if int(value) > 0}))
    fields: list[tuple[int, NDArray[np.float64], NDArray[np.float64]]] = []
    missing: list[int] = []
    for value in requested:
        if progress is not None and not progress():
            raise RuntimeError("Label interpolation cancelled")
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

    # Keep the path order explicit.  The wrapped path is normally non-contiguous,
    # so avoid advanced frame indexing and copy each frame with a basic slice.
    result = np.empty((*source.shape[:3], len(frames)), dtype=source.dtype)
    for offset, frame in enumerate(frames):
        result[..., offset] = source[..., frame]
    selected_values = np.asarray([field[0] for field in fields], dtype=source.dtype)
    result[np.isin(result, selected_values)] = 0
    span_float = float(span)
    for offset, frame in enumerate(frames):
        if progress is not None and not progress():
            raise RuntimeError("Label interpolation cancelled")
        alpha = float(offset + 1) / span_float
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


def interpolate_label_keyframes(
    data: NDArray,
    keyframes: Iterable[int],
    label_values: Iterable[int],
    *,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    wrap: bool = True,
    progress: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Interpolate labels between any number of user-selected keyframes.

    Keyframes are kept unchanged. Consecutive spans are interpolated in
    ascending time order; with ``wrap=True`` the final keyframe connects back
    to the first so every non-keyframe in a periodic sequence is covered.
    """
    source = np.asarray(data)
    if source.ndim != 4:
        raise ValueError(f"Label interpolation expects 3D+t data, got {source.shape}")
    frames = tuple(int(value) for value in keyframes)
    intermediate = keyframe_intermediate_frames(frames, source.shape[3], wrap=wrap)
    if not intermediate:
        raise ValueError("At least one frame is required between the keyframes")

    result = source.copy()
    segments = list(zip(frames, frames[1:]))
    if wrap:
        segments.append((frames[-1], frames[0]))
    for start, end in segments:
        if wrap or start < end:
            span = (end - start) % source.shape[3] if wrap else end - start
            if span < 2:
                continue
            segment = interpolate_label_frames(
                source,
                start,
                end,
                label_values,
                spacing_xyz=spacing_xyz,
                wrap=wrap,
                progress=progress,
            )
            path = (
                tuple((start + offset) % source.shape[3] for offset in range(1, span))
                if wrap
                else tuple(range(start + 1, end))
            )
            for offset, frame in enumerate(path):
                result[..., frame] = segment[..., offset]
    return result
