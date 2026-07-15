from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage import filters


GLOBAL_METHODS = (
    "otsu",
    "triangle",
    "li",
    "yen",
    "isodata",
    "multi_otsu",
    "kittler",
)
LOCAL_METHODS = ("local_gaussian", "sauvola", "phansalkar", "hysteresis")
THRESHOLD_METHODS = ("manual", *GLOBAL_METHODS, *LOCAL_METHODS)


def _finite_values(data: NDArray) -> NDArray[np.float64]:
    values = np.asarray(data, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("The image contains no finite intensity values")
    if values.size > 2_000_000:
        values = values[:: int(np.ceil(values.size / 2_000_000))]
    return values


def kittler_threshold(data: NDArray, bins: int = 256) -> float:
    """Return the Kittler-Illingworth minimum-error threshold."""
    values = _finite_values(data)
    low, high = float(values.min()), float(values.max())
    if low == high:
        return low
    histogram, edges = np.histogram(values, bins=bins, range=(low, high))
    centers = (edges[:-1] + edges[1:]) * 0.5
    probability = histogram.astype(np.float64)
    probability /= probability.sum()
    weight_a = np.cumsum(probability)
    weight_b = 1.0 - weight_a
    moment = np.cumsum(probability * centers)
    second_moment = np.cumsum(probability * centers**2)
    total_moment = moment[-1]
    total_second = second_moment[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_a = moment / weight_a
        mean_b = (total_moment - moment) / weight_b
        variance_a = second_moment / weight_a - mean_a**2
        variance_b = (total_second - second_moment) / weight_b - mean_b**2
        sigma_a = np.sqrt(np.maximum(variance_a, 0.0))
        sigma_b = np.sqrt(np.maximum(variance_b, 0.0))
        objective = (
            1.0
            + 2.0 * (weight_a * np.log(sigma_a) + weight_b * np.log(sigma_b))
            - 2.0
            * (weight_a * np.log(weight_a) + weight_b * np.log(weight_b))
        )
    valid = (
        (weight_a > 0.0)
        & (weight_b > 0.0)
        & (sigma_a > 0.0)
        & (sigma_b > 0.0)
        & np.isfinite(objective)
    )
    if not np.any(valid):
        return float(filters.threshold_otsu(values))
    candidates = np.flatnonzero(valid)
    return float(centers[candidates[np.argmin(objective[valid])]])


def automatic_thresholds(data: NDArray, method: str) -> tuple[float, float]:
    """Calculate global lower/upper bounds for an automatic threshold method."""
    values = _finite_values(data)
    low, high = float(values.min()), float(values.max())
    if low == high:
        return low, high
    functions = {
        "otsu": filters.threshold_otsu,
        "triangle": filters.threshold_triangle,
        "li": filters.threshold_li,
        "yen": filters.threshold_yen,
        "isodata": filters.threshold_isodata,
    }
    if method in functions:
        return float(functions[method](values)), high
    if method == "multi_otsu":
        classes = min(3, int(np.unique(values).size))
        if classes < 2:
            return low, high
        thresholds = filters.threshold_multiotsu(values, classes=classes)
        return float(thresholds[-1]), high
    if method == "kittler":
        return kittler_threshold(values), high
    if method == "hysteresis":
        threshold = float(filters.threshold_otsu(values))
        return low + 0.8 * (threshold - low), threshold
    if method in {"local_gaussian", "sauvola", "phansalkar"}:
        return low, high
    raise ValueError(f"Unknown automatic threshold method: {method}")


def _odd_window(radius: int, shape: Iterable[int]) -> tuple[int, ...]:
    requested = max(3, int(radius) * 2 + 1)
    result = []
    for size in shape:
        maximum = int(size) if int(size) % 2 else int(size) - 1
        result.append(max(1, min(requested, maximum)))
    return tuple(result)


def _phansalkar_threshold(frame: NDArray, radius: int) -> NDArray[np.float64]:
    values = np.asarray(frame, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.full(values.shape, np.inf, dtype=np.float64)
    low, high = float(finite.min()), float(finite.max())
    scale = high - low
    normalized = (values - low) / scale if scale > 0 else np.zeros_like(values)
    size = _odd_window(radius, values.shape)
    mean = ndimage.uniform_filter(normalized, size=size, mode="nearest")
    mean_square = ndimage.uniform_filter(normalized**2, size=size, mode="nearest")
    deviation = np.sqrt(np.maximum(mean_square - mean**2, 0.0))
    threshold = mean * (
        1.0
        + 2.0 * np.exp(-10.0 * mean)
        + 0.25 * (deviation / 0.5 - 1.0)
    )
    return threshold * scale + low


def build_threshold_mask(
    data: NDArray,
    method: str,
    lower: float,
    upper: float,
    radius: int = 7,
) -> NDArray[np.bool_]:
    """Build an intensity selection with the same shape as a 3D+t image."""
    values = np.asarray(data)
    if values.ndim not in (2, 3, 4):
        raise ValueError(f"Thresholding expects 2D, 3D, or 4D data, got {values.shape}")
    if lower > upper:
        lower, upper = upper, lower
    finite = np.isfinite(values)
    if method in ("manual", *GLOBAL_METHODS):
        return finite & (values >= lower) & (values <= upper)

    result = np.zeros(values.shape, dtype=bool)
    frames = range(values.shape[-1]) if values.ndim == 4 else (None,)
    for frame_index in frames:
        frame = values[..., frame_index] if frame_index is not None else values
        if method == "local_gaussian":
            local = ndimage.gaussian_filter(
                np.asarray(frame, dtype=np.float64),
                sigma=max(0.5, float(radius) / 3.0),
                mode="nearest",
            )
            selected = frame >= local
        elif method == "sauvola":
            local = filters.threshold_sauvola(
                np.asarray(frame, dtype=np.float64),
                window_size=_odd_window(radius, frame.shape),
            )
            selected = frame >= local
        elif method == "phansalkar":
            selected = frame >= _phansalkar_threshold(frame, radius)
        elif method == "hysteresis":
            selected = filters.apply_hysteresis_threshold(frame, lower, upper)
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        selected = np.asarray(selected) & np.isfinite(frame)
        if method != "hysteresis":
            selected &= frame >= lower
            selected &= frame <= upper
        if frame_index is None:
            result[...] = selected
        else:
            result[..., frame_index] = selected
    return result


def connected_seed_region(candidate: NDArray, seed: tuple[int, ...]) -> NDArray[np.bool_]:
    """Return the face-connected candidate component containing ``seed``."""
    allowed = np.asarray(candidate, dtype=bool)
    if len(seed) != allowed.ndim or any(
        coordinate < 0 or coordinate >= allowed.shape[axis]
        for axis, coordinate in enumerate(seed)
    ):
        raise ValueError("Seed is outside the candidate array")
    if not allowed[seed]:
        return np.zeros_like(allowed)
    structure = ndimage.generate_binary_structure(allowed.ndim, 1)
    components, _ = ndimage.label(allowed, structure=structure)
    component = int(components[seed])
    return components == component
