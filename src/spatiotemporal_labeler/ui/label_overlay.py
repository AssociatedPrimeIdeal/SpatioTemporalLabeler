"""Reusable label-color buffers, with bounded lookup tables for small dtypes."""

from __future__ import annotations

import numpy as np

from spatiotemporal_labeler.model import LabelDefinition, default_label


class LabelOverlayBuffer:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.rgba: np.ndarray | None = None
        self._signature: tuple | None = None
        self._source_signature: tuple | None = None
        self._palette: np.ndarray | None = None
        self._colors: tuple = ()

    def render(
        self,
        mask: np.ndarray,
        definitions: dict[int, LabelDefinition] | None = None,
        *,
        alpha: int = 124,
        global_opacity: float = 1.0,
        dirty_bounds: tuple[slice, slice] | None = None,
    ) -> np.ndarray:
        labels = np.asarray(mask)
        if definitions is None:
            definitions = {
                int(value): default_label(int(value))
                for value in np.unique(labels) if int(value) > 0
            }
        opacity = np.clip(global_opacity, 0.0, 1.0)
        colors = tuple(
            (int(definition.value), tuple(definition.color),
             int(round(alpha * np.clip(definition.opacity, 0.0, 1.0) * opacity)))
            for definition in definitions.values() if definition.visible
        )
        signature = (labels.dtype.str, colors)
        source_signature = (labels.__array_interface__["data"][0], labels.strides)
        changed_source = source_signature != self._source_signature
        self._source_signature = source_signature
        changed_style = signature != self._signature
        if changed_style:
            self._signature = signature
            self._colors = colors
            self._palette = None
            # uint8/int16 are the common label formats. Larger/sparse label
            # IDs use comparisons instead of allocating an unbounded LUT.
            if labels.dtype.kind in "bui" and labels.dtype.itemsize <= 2:
                size = 2 if labels.dtype.kind == "b" else 1 << (8 * labels.dtype.itemsize)
                self._palette = np.zeros((size, 4), dtype=np.uint8)
                limits = (0, 1) if labels.dtype.kind == "b" else (
                    np.iinfo(labels.dtype).min, np.iinfo(labels.dtype).max
                )
                for value, color, effective_alpha in colors:
                    if limits[0] <= value <= limits[1]:
                        self._palette[value % size] = (*color, effective_alpha)
        if self.rgba is None or self.rgba.shape != (*labels.shape, 4):
            self.rgba = np.empty((*labels.shape, 4), dtype=np.uint8)
            dirty_bounds = None
        if changed_style or changed_source:
            dirty_bounds = None
        region = (...,) if dirty_bounds is None else dirty_bounds
        source = labels[region]
        target = self.rgba[region]
        if self._palette is not None:
            # Wrap signed small integers to their dtype's unsigned LUT slot.
            np.take(self._palette, source, axis=0, out=target, mode="wrap")
        else:
            target.fill(0)
            for value, color, effective_alpha in self._colors:
                target[source == value] = (*color, effective_alpha)
        return self.rgba
