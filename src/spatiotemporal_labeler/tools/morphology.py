from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


MORPHOLOGY_OPERATIONS = (
    "remove_small_components",
    "fill_holes",
    "opening",
    "closing",
)


def _connectivity_structure(
    shape: tuple[int, ...], connectivity: int
) -> NDArray[np.bool_]:
    """Build a face/edge/corner structure without crossing singleton axes."""
    ndim = len(shape)
    rank = int(np.clip(connectivity, 1, ndim))
    structure_shape = (3,) * ndim
    center = np.ones(ndim, dtype=np.intp)
    coordinates = np.indices(structure_shape, dtype=np.intp)
    offsets = coordinates - center.reshape((ndim,) + (1,) * ndim)
    allowed = np.count_nonzero(offsets, axis=0) <= rank
    for axis, size in enumerate(shape):
        if size == 1:
            allowed &= offsets[axis] == 0
    return allowed


def remove_small_components(
    selection: NDArray,
    minimum_volume_mm3: float,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    connectivity: int = 1,
) -> NDArray[np.bool_]:
    """Remove connected components smaller than a physical volume threshold."""
    binary = np.asarray(selection, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Connected components expect a 3D frame, got {binary.shape}")
    spacing = np.asarray(spacing_xyz, dtype=float)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Spacing must contain three positive values")
    voxel_volume_mm3 = float(np.prod(spacing))
    threshold = max(0.0, float(minimum_volume_mm3))
    structure = _connectivity_structure(binary.shape, connectivity)
    components, count = ndimage.label(binary, structure=structure)
    if count == 0:
        return np.zeros_like(binary)
    volumes_mm3 = np.bincount(components.ravel()) * voxel_volume_mm3
    keep = volumes_mm3 >= threshold
    keep[0] = False
    return keep[components]


def physical_ball(
    shape: tuple[int, ...],
    spacing_xyz: tuple[float, float, float],
    radius_mm: float,
) -> NDArray[np.bool_]:
    """Return a ball sampled in physical space, restricted along singleton axes."""
    ndim = len(shape)
    spacing = np.asarray(spacing_xyz[:ndim], dtype=float)
    radius = float(radius_mm)
    if ndim not in (2, 3):
        raise ValueError(f"Physical morphology expects 2D or 3D data, got {shape}")
    if spacing.size != ndim or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Spacing must contain {ndim} positive values")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Morphology radius must be positive")

    extents = [
        0 if size == 1 else max(1, int(np.ceil(radius / axis_spacing)))
        for size, axis_spacing in zip(shape, spacing)
    ]
    coordinates = np.ogrid[
        tuple(slice(-extent, extent + 1) for extent in extents)
    ]
    squared_distance = sum(
        (coordinate * axis_spacing) ** 2
        for coordinate, axis_spacing in zip(coordinates, spacing)
    )
    return np.asarray(squared_distance <= radius**2 + 1e-12, dtype=bool)


def apply_label_morphology(
    frame: NDArray,
    operation: str,
    label_values: Iterable[int],
    *,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    minimum_volume_mm3: float = 100.0,
    connectivity: int = 1,
    radius_mm: float = 1.0,
) -> np.ndarray:
    """Apply one binary operation per label while preserving neighboring labels.

    Removed voxels become background. New voxels are written only where the input
    frame was background, so one label can never overwrite another label.
    """
    source = np.asarray(frame)
    if source.ndim != 3:
        raise ValueError(f"Label morphology expects one 3D frame, got {source.shape}")
    if operation not in MORPHOLOGY_OPERATIONS:
        raise ValueError(f"Unknown morphology operation: {operation}")

    values = tuple(sorted({int(value) for value in label_values if int(value) > 0}))
    structure = _connectivity_structure(source.shape, connectivity)
    physical_structure = (
        physical_ball(source.shape, spacing_xyz, radius_mm)
        if operation in {"opening", "closing"}
        else None
    )
    transformed: dict[int, NDArray[np.bool_]] = {}
    for value in values:
        binary = source == value
        if operation == "remove_small_components":
            result = remove_small_components(
                binary, minimum_volume_mm3, spacing_xyz, connectivity
            )
        elif operation == "fill_holes":
            result = ndimage.binary_fill_holes(binary, structure=structure)
        elif operation == "opening":
            result = ndimage.binary_opening(binary, structure=physical_structure)
        else:
            result = ndimage.binary_closing(binary, structure=physical_structure)
        transformed[value] = np.asarray(result, dtype=bool)

    output = source.copy()
    for value, result in transformed.items():
        output[(source == value) & ~result] = 0
    for value, result in transformed.items():
        additions = result & (source == 0) & (output == 0)
        output[additions] = value
    return output
