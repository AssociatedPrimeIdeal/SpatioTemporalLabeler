from dataclasses import replace

import numpy as np
import pytest

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.tools.streamlines import (
    StreamlineCancelled, StreamlineConfig, project_all_slices,
    trace_streamlines, validate_sources,
)


def sequence(data, direction=None, origin=(0, 0, 0), spacing=(1, 1, 1)):
    transform = AxisTransform((0, 1, 2, 3), (False, False, False), spacing, origin,
                              np.eye(3) if direction is None else direction)
    return Sequence4D(np.asarray(data), {}, transform)


def test_segdata_components_are_hf_ap_rl_independently_of_sampling_orientation():
    components = [np.full((9, 10, 11), v, np.float32) for v in (30, 20, 10)]
    angle = 0.3
    direction = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    config = StreamlineConfig(seed_count=20, max_steps=4, step_mm=0.3)
    segments, speeds = trace_streamlines(components, (2, 3, 4), direction, config)
    assert len(segments)
    delta = (segments[:, 1] - segments[:, 0]) @ direction.T
    expected = np.array([-10, -20, -30]) / np.linalg.norm([10, 20, 30])
    np.testing.assert_allclose(delta / np.linalg.norm(delta, axis=1)[:, None],
                               np.broadcast_to(expected, delta.shape), atol=1e-5)
    np.testing.assert_allclose(speeds, np.linalg.norm([10, 20, 30]), rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(delta, axis=1), 0.3, atol=3e-6)


def test_manual_ras_mapping_units_and_length_limit():
    components = [np.full((20, 4, 4), v) for v in (0.5, 0, 0)]
    config = StreamlineConfig(directions=("R", "A", "H"), seed_count=1,
                              step_mm=0.4, max_steps=100, max_length_mm=3.0,
                              min_speed=1, source_scale=100)
    segments, speeds = trace_streamlines(components, (2, 1, 1), np.eye(3), config)
    delta = segments[:, 1] - segments[:, 0]
    assert np.all(delta[:, 0] > 0)
    np.testing.assert_allclose(delta[:, 1:], 0)
    assert np.linalg.norm(delta, axis=1).sum() <= 3.00001
    np.testing.assert_allclose(speeds, 50)


def test_phase_calibration_uses_each_components_venc_before_vector_mapping():
    components = [np.full((10, 10, 10), np.pi / 2)] * 3
    config = StreamlineConfig(seed_count=5, max_steps=2, step_mm=0.2,
                              component_scales=tuple(v / np.pi for v in (100, 200, 300)))
    segments, speeds = trace_streamlines(components, (1, 1, 1), np.eye(3), config)
    expected = np.array([-150, -100, -50])
    delta = segments[:, 1] - segments[:, 0]
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(delta / np.linalg.norm(delta, axis=1)[:, None],
                               np.broadcast_to(expected, delta.shape), atol=1e-5)
    np.testing.assert_allclose(speeds, np.linalg.norm([50, 100, 150]), rtol=1e-6)


def test_constant_zero_or_nonfinite_field_is_empty_and_source_is_unchanged():
    source = np.zeros((5, 4, 3), np.float32)
    source[2, 2, 2] = np.nan
    original = source.copy()
    segments, speeds = trace_streamlines([source] * 3, (1, 1, 1), np.eye(3), StreamlineConfig())
    assert segments.shape == (0, 2, 3) and speeds.shape == (0,)
    np.testing.assert_equal(source, original)
    frame = project_all_slices(segments, speeds, source.shape, (1, 1, 1), (0, 0, 0), np.eye(3))
    assert all(len(slices.values) == 0 for slices in frame.slices.values())


def test_cancellation_interrupts_integrator_and_projection():
    config = StreamlineConfig()
    with pytest.raises(StreamlineCancelled):
        trace_streamlines([np.ones((3, 3, 3))] * 3, (1, 1, 1), np.eye(3), config, lambda: True)
    with pytest.raises(StreamlineCancelled):
        project_all_slices(np.zeros((0, 2, 3)), [], (3, 3, 3), (1, 1, 1),
                           (0, 0, 0), np.eye(3), cancelled=lambda: True)


def test_slab_clips_crossings_keeps_direction_and_preserves_separate_segments():
    segments = np.array([[[0, 0, -2], [4, 0, 2]], [[9, 1, 2], [5, 1, -2]],
                         [[3, 3, -2], [3, 3, 2]]], np.float32)
    frame = project_all_slices(segments, [10, 20, 30], (10, 4, 3), (1, 1, 1),
                               (20, 30, 40), np.eye(3), slab_mm=2)
    rows = frame.slices["X-Y"].at(0)
    np.testing.assert_allclose(rows, [[1, 0, 3, 0, 10], [8, 1, 6, 1, 20]])
    np.testing.assert_allclose(frame.world_segments, segments + [20, 30, 40])
    assert np.shares_memory(rows, frame.slices["X-Y"].values)
    np.testing.assert_array_equal(rows, frame.slices["X-Y"].at(0))


def test_auto_slab_uses_each_fixed_axis_spacing_and_has_an_index_for_every_slice():
    segments = np.array([[[0, 0, 0], [5, 0, 0]]], np.float32)
    frame = project_all_slices(segments, [7], (6, 4, 3), (1, 2, 3), (0, 0, 0), np.eye(3))
    assert len(frame.slices["X-Y"].offsets) == 4
    assert len(frame.slices["X-Z"].offsets) == 5
    assert len(frame.slices["Y-Z"].offsets) == 7
    assert len(frame.slices["X-Y"].at(0)) == 1
    assert len(frame.slices["X-Y"].at(1)) == 0


def test_sources_check_complete_grid_and_allow_only_single_frame_broadcast():
    a = sequence(np.zeros((4, 3, 2, 2)))
    b = sequence(np.zeros((4, 3, 2, 1)))
    assert validate_sources((a, b, a)) == 2
    with pytest.raises(ValueError, match="voxel grids"):
        validate_sources((a, b, sequence(b.data, origin=(1, 0, 0))))
    with pytest.raises(ValueError, match="frame counts"):
        validate_sources((a, b, sequence(np.zeros((4, 3, 2, 3)))))
    with pytest.raises(ValueError, match="each velocity"):
        validate_sources((a, None, b))


@pytest.mark.parametrize("config", [
    StreamlineConfig(directions=("R", "L", "H")),
    StreamlineConfig(step_mm=0), StreamlineConfig(min_speed=float("nan")),
    StreamlineConfig(max_steps=1001), StreamlineConfig(seed_count=0),
])
def test_invalid_parameters_fail_before_computation(config):
    with pytest.raises(ValueError):
        config.validate()
