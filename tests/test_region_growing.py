import numpy as np

from spatiotemporal_labeler.tools import RegionGrowConfig, grow_region_4d


def propagate(image, labels, seeds, *, t0=0, **config_values):
    spacing_xyz = config_values.pop("spacing_xyz", (1.0, 1.0, 1.0))
    config = RegionGrowConfig(
        tolerance=config_values.pop("tolerance", 0.05),
        max_displacement_mm=config_values.pop("max_displacement_mm", 0.0),
        temporal_radius=config_values.pop("temporal_radius", None),
        **config_values,
    )
    return grow_region_4d(
        image,
        labels,
        seeds,
        active_label=1,
        spacing_xyz=spacing_xyz,
        t0=t0,
        config=config,
    )


def accepted_set(result):
    return {tuple(int(value) for value in coordinate) for coordinate in result.accepted_voxels}


def test_propagation_is_pure_and_does_not_expand_in_the_stroke_frame():
    image = np.ones((5, 1, 1, 3), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    original_image = image.copy()
    original_labels = labels.copy()

    result = propagate(image, labels, [(2, 0, 0, 1)], t0=1, tolerance=0.0)

    assert accepted_set(result) == {(2, 0, 0, frame) for frame in range(3)}
    assert result.added_voxel_count == 3
    assert result.replaced_voxel_count == 0
    assert np.array_equal(image, original_image)
    assert np.array_equal(labels, original_labels)


def test_each_source_voxel_can_deform_independently_within_the_displacement_window():
    image = np.zeros((6, 1, 1, 2), dtype=np.float32)
    image[1, :, :, 0] = 0.4
    image[2, :, :, 0] = 0.8
    image[2, :, :, 1] = 0.4
    image[4, :, :, 1] = 0.8
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = propagate(
        image,
        labels,
        [(1, 0, 0, 0), (2, 0, 0, 0)],
        tolerance=0.01,
        max_displacement_mm=2.0,
    )

    assert accepted_set(result) == {
        (1, 0, 0, 0),
        (2, 0, 0, 0),
        (2, 0, 0, 1),
        (4, 0, 0, 1),
    }


def test_zero_displacement_requires_the_same_spatial_coordinate():
    image = np.zeros((4, 1, 1, 2), dtype=np.float32)
    image[1, :, :, 0] = 0.8
    image[2, :, :, 1] = 0.8
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = propagate(
        image,
        labels,
        [(1, 0, 0, 0)],
        tolerance=0.01,
        max_displacement_mm=0.0,
    )

    assert accepted_set(result) == {(1, 0, 0, 0)}


def test_large_interframe_intensity_change_stops_that_direction_without_skipping():
    image = np.zeros((1, 1, 1, 4), dtype=np.float32)
    image[..., 0] = 0.2
    image[..., 1] = 0.21
    image[..., 2] = 0.8
    image[..., 3] = 0.21
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = propagate(
        image,
        labels,
        [(0, 0, 0, 0)],
        tolerance=0.05,
        max_displacement_mm=0.0,
    )

    assert accepted_set(result) == {(0, 0, 0, 0), (0, 0, 0, 1)}


def test_interframe_tolerance_uses_raw_intensity_units_without_percentile_scaling():
    # 远端高信号建立大动态范围，确保结果不受 P1/P99 归一化影响。
    image = np.zeros((10, 10, 1, 2), dtype=np.float32)
    image[:2, :, :, :] = 1_000.0
    image[5, 5, 0, 0] = 100.0
    image[5, 5, 0, 1] = 101.0
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = propagate(
        image,
        labels,
        [(5, 5, 0, 0)],
        tolerance=0.5,
        max_displacement_mm=0.0,
    )

    assert accepted_set(result) == {(5, 5, 0, 0)}


def test_motion_smoothness_rejects_a_tightly_adjacent_vessel_with_a_discontinuous_path():
    # 相邻血管在第三帧有同强度候选；连续位移应保留原血管的运动方向。
    image = np.zeros((3, 1, 1, 3), dtype=np.float32)
    image[0, 0, 0, 0] = 100.0
    image[1, 0, 0, 1] = 100.0
    image[1, 0, 0, 2] = 100.0
    image[2, 0, 0, 2] = 100.0
    labels = np.zeros(image.shape, dtype=np.uint8)

    unsmoothed = propagate(
        image,
        labels,
        [(0, 0, 0, 0)],
        tolerance=0.0,
        max_displacement_mm=1.0,
        motion_smoothness=0.0,
    )
    smoothed = propagate(
        image,
        labels,
        [(0, 0, 0, 0)],
        tolerance=0.0,
        max_displacement_mm=1.0,
        motion_smoothness=1.0,
        max_motion_change_mm=0.1,
    )

    assert (1, 0, 0, 2) in accepted_set(unsmoothed)
    assert (2, 0, 0, 2) not in accepted_set(unsmoothed)
    assert (1, 0, 0, 2) not in accepted_set(smoothed)
    assert (2, 0, 0, 2) in accepted_set(smoothed)


def test_forward_and_backward_chains_stop_independently():
    image = np.zeros((1, 1, 1, 5), dtype=np.float32)
    image[..., 1:4] = 0.3
    image[..., 0] = 0.9
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = propagate(
        image,
        labels,
        [(0, 0, 0, 2)],
        t0=2,
        tolerance=0.05,
        max_displacement_mm=0.0,
    )

    assert accepted_set(result) == {
        (0, 0, 0, 1),
        (0, 0, 0, 2),
        (0, 0, 0, 3),
    }


def test_default_temporal_range_attempts_all_frames_and_explicit_range_limits_it():
    image = np.ones((1, 1, 1, 5), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)

    all_frames = propagate(image, labels, [(0, 0, 0, 2)], t0=2, tolerance=0.0)
    limited = propagate(
        image,
        labels,
        [(0, 0, 0, 2)],
        t0=2,
        tolerance=0.0,
        temporal_radius=1,
    )

    assert {coordinate[3] for coordinate in accepted_set(all_frames)} == set(range(5))
    assert {coordinate[3] for coordinate in accepted_set(limited)} == {1, 2, 3}


def test_other_labels_are_barriers_unless_replacement_is_enabled():
    image = np.ones((1, 1, 1, 2), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[..., 1] = 2

    blocked = propagate(image, labels, [(0, 0, 0, 0)], tolerance=0.0)
    replacing = propagate(
        image,
        labels,
        [(0, 0, 0, 0)],
        tolerance=0.0,
        replace_other_labels=True,
    )

    assert accepted_set(blocked) == {(0, 0, 0, 0)}
    assert accepted_set(replacing) == {(0, 0, 0, 0), (0, 0, 0, 1)}
    assert replacing.replaced_voxel_count == 1
    assert replacing.source_label_counts == {2: 1}


def test_threshold_selection_applies_to_seeds_and_temporal_targets():
    image = np.ones((1, 1, 1, 3), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    threshold = np.array([True, False, True]).reshape(1, 1, 1, 3)

    constrained = grow_region_4d(
        image,
        labels,
        [(0, 0, 0, 0)],
        active_label=1,
        spacing_xyz=(1.0, 1.0, 1.0),
        t0=0,
        config=RegionGrowConfig(0.0, 0.0),
        threshold_mask=threshold,
    )

    assert accepted_set(constrained) == {(0, 0, 0, 0)}


def test_nonfinite_values_are_not_valid_seeds_or_temporal_targets():
    image = np.ones((1, 1, 1, 3), dtype=np.float32)
    image[..., 1] = np.nan
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = propagate(image, labels, [(0, 0, 0, 0)], tolerance=0.0)
    invalid_only = propagate(image, labels, [(0, 0, 0, 1)], t0=1)

    assert accepted_set(result) == {(0, 0, 0, 0)}
    assert invalid_only.accepted_voxels.size == 0
    assert invalid_only.seed_median is None


def test_physical_displacement_respects_anisotropic_spacing():
    image = np.zeros((2, 2, 1, 2), dtype=np.float32)
    image[0, 0, 0, 0] = 0.7
    image[0, 1, 0, 1] = 0.7
    image[1, 0, 0, 1] = 0.7
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = propagate(
        image,
        labels,
        [(0, 0, 0, 0)],
        tolerance=0.01,
        max_displacement_mm=2.0,
        spacing_xyz=(3.0, 1.0, 1.0),
    )

    assert accepted_set(result) == {(0, 0, 0, 0), (0, 1, 0, 1)}


def test_safety_limit_aborts_without_a_partial_result_or_input_mutation():
    image = np.ones((2, 1, 1, 2), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    original = labels.copy()

    result = propagate(
        image,
        labels,
        [(0, 0, 0, 0), (1, 0, 0, 0)],
        tolerance=0.0,
        max_changed_voxels=2,
    )

    assert result.aborted
    assert result.abort_reason == "maximum changed voxel safety limit exceeded"
    assert result.accepted_voxels.size == 0
    assert result.changed_voxel_count == 0
    assert np.array_equal(labels, original)
