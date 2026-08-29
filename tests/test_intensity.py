import numpy as np
import pytest

from spatiotemporal_labeler.tools import detect_cardiac_phases, strongest_signal_frame


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


def test_detect_cardiac_phases_finds_peak_and_circular_systolic_boundaries():
    signal = [1.0, 1.0, 1.0, 2.0, 5.0, 8.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    data = np.zeros((4, 4, 1, len(signal)), dtype=np.float32)
    for frame, value in enumerate(signal):
        data[..., frame] = value

    phases = detect_cardiac_phases(data)

    assert phases.peak == 5
    assert phases.systole_start != phases.peak
    assert phases.diastole_start != phases.peak
    assert 0 <= phases.systole_start < data.shape[3]
    assert 0 <= phases.diastole_start < data.shape[3]
    assert 0.0 < phases.confidence <= 1.0


def test_detect_cardiac_phases_rejects_a_flat_signal():
    data = np.ones((3, 3, 1, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="no detectable cardiac peak"):
        detect_cardiac_phases(data)
