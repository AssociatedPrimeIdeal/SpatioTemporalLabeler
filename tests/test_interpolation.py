from __future__ import annotations

import numpy as np
import pytest

from spatiotemporal_labeler.tools import (
    interpolate_label_frames,
    interpolate_label_keyframes,
    keyframe_intermediate_frames,
)


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


def test_cyclic_interpolation_follows_frames_across_the_time_axis_end():
    data = np.zeros((9, 9, 1, 5), dtype=np.uint8)
    data[1:4, 3:6, 0, 3] = 1
    data[3:6, 3:6, 0, 1] = 1

    result = interpolate_label_frames(data, 3, 1, [1], wrap=True)

    assert result.shape == (9, 9, 1, 2)
    assert np.count_nonzero(result[..., 0] == 1) > 0  # frame 4
    assert np.count_nonzero(result[..., 1] == 1) > 0  # frame 0
    with pytest.raises(ValueError, match="different"):
        interpolate_label_frames(data, 2, 2, [1], wrap=True)


def test_multiple_keyframes_fill_each_span_and_the_cyclic_tail():
    data = np.zeros((9, 9, 1, 6), dtype=np.uint8)
    for frame in (0, 2, 4):
        data[3:5, 3:6, 0, frame] = 1
    result = interpolate_label_keyframes(data, [0, 2, 4], [1], wrap=True)

    assert keyframe_intermediate_frames([0, 2, 4], 6, wrap=True) == (1, 3, 5)
    for frame in (0, 2, 4):
        assert np.array_equal(result[..., frame], data[..., frame])
    for frame in (1, 3, 5):
        assert np.count_nonzero(result[..., frame] == 1) > 0


def test_multiple_keyframes_require_ascending_unique_indices():
    data = np.zeros((5, 5, 1, 4), dtype=np.uint8)
    data[2, 2, 0, (0, 2)] = 1
    with pytest.raises(ValueError, match="ascending"):
        interpolate_label_keyframes(data, [2, 0], [1])
    with pytest.raises(ValueError, match="unique"):
        keyframe_intermediate_frames([0, 0], 4)
