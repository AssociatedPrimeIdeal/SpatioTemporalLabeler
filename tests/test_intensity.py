import numpy as np
import pytest

from spatiotemporal_labeler.tools import strongest_signal_frame


def test_strongest_signal_frame_uses_absolute_finite_signal_and_earliest_tie():
    data = np.zeros((2, 2, 1, 4), dtype=np.float32)
    data[..., 0] = 2.0
    data[..., 1] = -3.0
    data[..., 2] = 3.0
    data[0, 0, 0, 2] = np.nan
    data[..., 3] = np.inf

    assert strongest_signal_frame(data) == 1
    assert strongest_signal_frame(np.zeros_like(data)) == 0


def test_strongest_signal_frame_rejects_non_4d_or_empty_time_data():
    with pytest.raises(ValueError, match="nonempty 4D"):
        strongest_signal_frame(np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="nonempty 4D"):
        strongest_signal_frame(np.zeros((2, 2, 2, 0), dtype=np.float32))
