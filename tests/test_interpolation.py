from __future__ import annotations

import numpy as np
import pytest

from spatiotemporal_labeler.tools import interpolate_label_frames


def test_interpolates_a_label_between_two_keyframes():
    data = np.zeros((9, 9, 1, 3), dtype=np.uint8)
    data[1:4, 3:6, 0, 0] = 1
    data[3:6, 3:6, 0, 2] = 1

    result = interpolate_label_frames(data, 0, 2, [1])

    assert result.shape == (9, 9, 1, 1)
    assert np.all(result[2:5, 3:6, 0, 0] == 1)
    assert np.count_nonzero(result[..., 0] == 1) == 9


def test_interpolation_preserves_unselected_labels_as_barriers():
    data = np.zeros((9, 9, 1, 3), dtype=np.uint8)
    data[2:7, 2:7, 0, 0] = 1
    data[2:7, 2:7, 0, 2] = 1
    data[4, 4, 0, 1] = 2

    result = interpolate_label_frames(data, 0, 2, [1])

    assert result[4, 4, 0, 0] == 2
    assert np.count_nonzero(result[..., 0] == 2) == 1


def test_interpolation_requires_each_label_in_both_keyframes():
    data = np.zeros((5, 5, 1, 3), dtype=np.uint8)
    data[1:3, 1:3, 0, 0] = 1

    with pytest.raises(ValueError, match="both keyframes"):
        interpolate_label_frames(data, 0, 2, [1])
