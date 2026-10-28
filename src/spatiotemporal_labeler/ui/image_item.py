from typing import Any

import numpy as np
import pyqtgraph as pg

from .label_overlay import LabelOverlayBuffer


class DisplayImageItem(pg.ImageItem):
    """Release display buffers on clear and give Qt valid singleton-row strides."""

    def setImage(self, image: Any = None, *args: Any, **kwargs: Any) -> None:  # noqa: N802
        if image is not None and 1 in image.shape[:2]:
            # A transposed singleton axis can be C-contiguous while its row
            # stride is smaller than the row width. Qt rejects that buffer;
            # PySide can then retain the source sequence even after closing it.
            image = np.array(image, copy=True, order="C")
        super().setImage(image, *args, **kwargs)

    def clear(self) -> None:
        super().clear()
        # pyqtgraph's clear() only drops image. QImage can still own an array
        # view, and the processing buffers can retain the last displayed slice.
        self.qimage = None
        self._processingBuffer = None
        self._displayBuffer = None
        self._imageNanLocations = None
        self._imageHasNans = None


class LabelImageItem(DisplayImageItem):
    """Keep one RGBA buffer and recolor only edited pixels when possible."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.overlay = LabelOverlayBuffer()
        super().__init__(*args, **kwargs)

    def set_labels(self, mask, labels=None, *, global_opacity=1.0, dirty_bounds=None) -> None:
        rgba = self.overlay.render(
            mask, labels, global_opacity=global_opacity, dirty_bounds=dirty_bounds,
        )
        self.setImage(rgba.transpose(1, 0, 2), autoLevels=False)

    def clear(self) -> None:
        super().clear()
        self.overlay.clear()
