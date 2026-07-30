from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def strongest_signal_frame(data: NDArray[np.number]) -> int:
    """Return the earliest frame with the largest finite absolute-signal sum."""
    sequence = np.asarray(data)
    if sequence.ndim != 4 or sequence.shape[3] < 1:
        raise ValueError("Signal-frame selection requires nonempty 4D image data")

    strengths = np.zeros(sequence.shape[3], dtype=np.float64)
    for frame in range(sequence.shape[3]):
        values = sequence[..., frame]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        dtype = np.complex128 if np.iscomplexobj(values) else np.float64
        finite_values = np.asarray(values[finite], dtype=dtype)
        strengths[frame] = np.abs(finite_values).sum(dtype=np.float64)
    return int(np.argmax(strengths))
