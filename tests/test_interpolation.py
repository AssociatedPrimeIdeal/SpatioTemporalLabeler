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


def test_interpolation_reports_progress_for_labels_and_frames():
    data = np.zeros((5, 5, 1, 4), dtype=np.uint8)
    data[1:3, 1:3, 0, 0] = 1
    data[2:4, 2:4, 0, 3] = 1
    calls = []
    interpolate_label_frames(
        data, 0, 3, [1], progress=lambda: calls.append(True) or True
    )
    assert len(calls) == 3  # one SDF preparation and two intermediate frames


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


@pytest.mark.parametrize("wrap, expected_calls", [(False, 6), (True, 6)])
def test_shared_keyframe_distance_fields_are_computed_once(monkeypatch, wrap, expected_calls):
    from spatiotemporal_labeler.tools import interpolation

    data = np.zeros((12, 11, 3, 9), dtype=np.uint8)
    for frame in (0, 3, 6):
        data[2:5, 2:5, 1, frame] = 1
        data[7:10, 6:9, 1, frame] = 2
    calls = []
    original = interpolation._signed_distance

    def tracked(selection, spacing):
        calls.append(1)
        return original(selection, spacing)

    monkeypatch.setattr(interpolation, "_signed_distance", tracked)
    result = interpolate_label_keyframes(data, [0, 3, 6], iter([1, 2]), wrap=wrap)
    assert len(calls) == expected_calls
    assert np.array_equal(result[..., (0, 3, 6)], data[..., (0, 3, 6)])
    assert np.any(result[..., 4] == 2)  # An iterator must cover every span.


def test_streamed_label_scores_preserve_first_label_ties_and_barriers(monkeypatch):
    from spatiotemporal_labeler.tools import interpolation

    data = np.zeros((4, 4, 1, 3), dtype=np.uint16)
    data[0, 0, 0, (0, 2)] = 1
    data[1, 1, 0, (0, 2)] = 2
    data[2, 2, 0, 1] = 9
    monkeypatch.setattr(interpolation, "_signed_distance", lambda *args: np.ones((4, 4, 1)))
    result = interpolate_label_frames(data, 0, 2, [2, 1])
    assert result[2, 2, 0, 0] == 9
    assert np.all(result[..., 0][data[..., 1] != 9] == 1)


def test_interpolation_can_cancel_after_reusing_an_endpoint():
    data = np.zeros((7, 7, 1, 7), dtype=np.uint8)
    data[2:5, 2:5, 0, (0, 3, 6)] = 1
    calls = []

    def progress():
        calls.append(1)
        return len(calls) < 4

    before = data.copy()
    with pytest.raises(RuntimeError, match="cancelled"):
        interpolate_label_keyframes(data, [0, 3, 6], [1], progress=progress)
    np.testing.assert_array_equal(data, before)
