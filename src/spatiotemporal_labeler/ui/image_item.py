from typing import Any

import numpy as np
import pyqtgraph as pg


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
