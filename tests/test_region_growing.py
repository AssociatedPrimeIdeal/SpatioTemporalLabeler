import numpy as np

from spatiotemporal_labeler.tools import RegionGrowConfig, grow_region_4d


def grow(image, labels, seeds, **config_values):
    config = RegionGrowConfig(
        tolerance=config_values.pop("tolerance", 0.0),
        spatial_margin_mm=config_values.pop("spatial_margin_mm", 20.0),
        temporal_radius=config_values.pop("temporal_radius", 0),
        **config_values,
    )
    return grow_region_4d(
        image,
        labels,
        seeds,
        active_label=1,
        spacing_xyz=(1.0, 1.0, 1.0),
        t0=0,
        config=config,
    )


def accepted_set(result):
    return {tuple(int(value) for value in coordinate) for coordinate in result.accepted_voxels}


def grow_at(image, labels, seeds, *, t0, **config_values):
    config = RegionGrowConfig(
        tolerance=config_values.pop("tolerance", 0.0),
        spatial_margin_mm=config_values.pop("spatial_margin_mm", 20.0),
        temporal_radius=config_values.pop("temporal_radius", 1),
        **config_values,
    )
    return grow_region_4d(
        image,
        labels,
        seeds,
        active_label=1,
        spacing_xyz=(1.0, 1.0, 1.0),
        t0=t0,
        config=config,
    )


def test_grow_is_pure_and_reports_added_voxels():
    image = np.ones((5, 1, 1, 1), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    original_image = image.copy()
    original_labels = labels.copy()

    result = grow(image, labels, [(2, 0, 0, 0)])

    assert accepted_set(result) == {(x, 0, 0, 0) for x in range(5)}
    assert result.added_voxel_count == 5
    assert result.replaced_voxel_count == 0
    assert result.source_label_counts == {}
    assert np.array_equal(image, original_image)
    assert np.array_equal(labels, original_labels)


def test_other_labels_are_barriers_without_replacement():
    image = np.ones((7, 1, 1, 1), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[4, 0, 0, 0] = 2

    result = grow(image, labels, [(1, 0, 0, 0)])

    assert accepted_set(result) == {(x, 0, 0, 0) for x in range(4)}
    assert result.added_voxel_count == 4
    assert result.replaced_voxel_count == 0


def test_replacement_overwrites_only_reached_other_labels_and_reports_sources():
    image = np.ones((7, 1, 1, 1), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[3, 0, 0, 0] = 2
    labels[6, 0, 0, 0] = 3

    result = grow(
        image,
        labels,
        [(1, 0, 0, 0)],
        replace_other_labels=True,
    )

    assert accepted_set(result) == {(x, 0, 0, 0) for x in range(7)}
    assert result.added_voxel_count == 5
    assert result.replaced_voxel_count == 2
    assert result.source_label_counts == {2: 1, 3: 1}


def test_existing_active_label_traverses_even_when_its_intensity_is_outside_tolerance():
    image = np.ones((5, 1, 1, 1), dtype=np.float32)
    image[2, 0, 0, 0] = 100.0
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[2, 0, 0, 0] = 1

    result = grow(image, labels, [(0, 0, 0, 0)], tolerance=0.0)

    assert accepted_set(result) == {(x, 0, 0, 0) for x in range(5)}
    assert result.added_voxel_count == 4


def test_temporal_range_clips_at_sequence_edges_without_wrapping():
    image = np.ones((1, 1, 1, 4), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    config = RegionGrowConfig(0.0, 0.0, temporal_radius=1)

    result = grow_region_4d(
        image,
        labels,
        [(0, 0, 0, 0)],
        active_label=1,
        spacing_xyz=(1.0, 1.0, 1.0),
        t0=0,
        config=config,
    )

    assert accepted_set(result) == {(0, 0, 0, 0), (0, 0, 0, 1)}
    assert result.roi_slices[3] == slice(0, 2)


def test_temporal_grow_stops_after_an_intermediate_frame_has_no_result():
    image = np.ones((1, 1, 1, 4), dtype=np.float32)
    image[..., 2] = 10.0
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = grow_at(
        image,
        labels,
        [(0, 0, 0, 0)],
        t0=0,
        temporal_radius=3,
    )

    assert accepted_set(result) == {(0, 0, 0, 0), (0, 0, 0, 1)}
    assert (0, 0, 0, 3) not in accepted_set(result)


def test_temporal_grow_requires_immediate_local_support_for_every_voxel():
    image = np.zeros((6, 1, 1, 2), dtype=np.float32)
    image[1, 0, 0, 0] = 1.0
    image[2, 0, 0, 1] = 1.0  # One voxel displacement is supported.
    image[4, 0, 0, 1] = 1.0  # Same intensity but too far from the source result.
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = grow_at(
        image,
        labels,
        [(1, 0, 0, 0)],
        t0=0,
        spatial_margin_mm=20.0,
    )

    assert accepted_set(result) == {(1, 0, 0, 0), (2, 0, 0, 1)}


def test_stroke_frame_finishes_3d_growth_before_temporal_propagation():
    image = np.zeros((4, 1, 1, 2), dtype=np.float32)
    image[:3, 0, 0, 0] = 1.0
    image[3, 0, 0, 1] = 1.0
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = grow_at(
        image,
        labels,
        [(0, 0, 0, 0)],
        t0=0,
        spatial_margin_mm=20.0,
    )

    # Frame 1 at x=3 is supported only after the stroke-frame 3D path reaches
    # x=2; it is not directly adjacent to the user seed.
    assert accepted_set(result) == {
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (2, 0, 0, 0),
        (3, 0, 0, 1),
    }


def test_temporal_grow_requires_both_stroke_and_temporal_intensity_similarity():
    image = np.zeros((3, 1, 1, 2), dtype=np.float32)
    image[1, 0, 0, 0] = 1.0
    image[2, 0, 0, 0] = 100.0
    image[1, 0, 0, 1] = 1.25  # dT passes, but dI exceeds tolerance.
    image[2, 0, 0, 1] = -1.0  # dI passes, but dT exceeds 1.5 * tolerance.
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = grow_at(
        image,
        labels,
        [(0, 0, 0, 0)],
        t0=0,
        spatial_margin_mm=20.0,
        tolerance=1.0,
    )

    assert accepted_set(result) == {(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)}


def test_nonfinite_target_intensity_terminates_that_temporal_direction():
    image = np.ones((1, 1, 1, 3), dtype=np.float32)
    image[..., 1] = np.nan
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = grow_at(
        image,
        labels,
        [(0, 0, 0, 0)],
        t0=0,
        temporal_radius=2,
    )

    assert accepted_set(result) == {(0, 0, 0, 0)}


def test_forward_and_backward_propagation_are_independent():
    image = np.ones((1, 1, 1, 3), dtype=np.float32)
    image[..., 2] = 10.0
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = grow_at(
        image,
        labels,
        [(0, 0, 0, 1)],
        t0=1,
        temporal_radius=1,
    )

    assert accepted_set(result) == {(0, 0, 0, 0), (0, 0, 0, 1)}


def test_temporal_other_labels_are_barriers_unless_replacement_is_enabled():
    image = np.ones((1, 1, 1, 2), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[..., 1] = 2

    blocked = grow_at(image, labels, [(0, 0, 0, 0)], t0=0)
    replacing = grow_at(
        image,
        labels,
        [(0, 0, 0, 0)],
        t0=0,
        replace_other_labels=True,
    )

    assert accepted_set(blocked) == {(0, 0, 0, 0)}
    assert accepted_set(replacing) == {(0, 0, 0, 0), (0, 0, 0, 1)}
    assert replacing.replaced_voxel_count == 1
    assert replacing.source_label_counts == {2: 1}


def test_existing_active_labels_in_target_frames_still_need_both_intensity_checks():
    image = np.ones((1, 1, 1, 2), dtype=np.float32)
    image[..., 1] = 10.0
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[..., 1] = 1

    result = grow_at(image, labels, [(0, 0, 0, 0)], t0=0)

    assert accepted_set(result) == {(0, 0, 0, 0)}


def test_all_valid_stroke_seeds_define_the_fixed_median_and_are_forced():
    image = np.array([2.0, 6.0, 10.0], dtype=np.float32).reshape(3, 1, 1, 1)
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = grow(
        image,
        labels,
        [(0, 0, 0, 0), (2, 0, 0, 0)],
        tolerance=0.0,
    )

    assert result.seed_median == 6.0
    assert accepted_set(result) == {
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (2, 0, 0, 0),
    }


def test_threshold_selection_blocks_growth_and_absence_models_bypass():
    image = np.ones((4, 1, 1, 1), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    threshold = np.array([True, True, False, True]).reshape(4, 1, 1, 1)
    config = RegionGrowConfig(0.0, 20.0, 0)

    constrained = grow_region_4d(
        image,
        labels,
        [(0, 0, 0, 0)],
        active_label=1,
        spacing_xyz=(1.0, 1.0, 1.0),
        t0=0,
        config=config,
        threshold_mask=threshold,
    )
    bypassed = grow_region_4d(
        image,
        labels,
        [(0, 0, 0, 0)],
        active_label=1,
        spacing_xyz=(1.0, 1.0, 1.0),
        t0=0,
        config=config,
    )

    assert accepted_set(constrained) == {(0, 0, 0, 0), (1, 0, 0, 0)}
    assert accepted_set(bypassed) == {(x, 0, 0, 0) for x in range(4)}


def test_nonfinite_values_are_not_valid_seeds_or_growth_candidates():
    image = np.array([1.0, np.nan, 1.0, np.inf, 1.0], dtype=np.float32).reshape(
        5, 1, 1, 1
    )
    labels = np.zeros(image.shape, dtype=np.uint8)

    result = grow(image, labels, [(0, 0, 0, 0), (1, 0, 0, 0)])
    invalid_only = grow(image, labels, [(1, 0, 0, 0)])

    assert accepted_set(result) == {(0, 0, 0, 0)}
    assert invalid_only.accepted_voxels.size == 0
    assert invalid_only.seed_median is None


def test_anisotropic_spacing_converts_the_physical_roi_margin_per_axis():
    image = np.ones((9, 9, 9, 1), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    config = RegionGrowConfig(0.0, spatial_margin_mm=2.0, temporal_radius=0)

    result = grow_region_4d(
        image,
        labels,
        [(4, 4, 4, 0)],
        active_label=1,
        spacing_xyz=(1.0, 3.0, 0.5),
        t0=0,
        config=config,
    )

    assert result.roi_slices == (slice(2, 7), slice(3, 6), slice(0, 9), slice(0, 1))


def test_safety_limit_aborts_without_a_partial_result_or_input_mutation():
    image = np.ones((5, 1, 1, 1), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    original = labels.copy()

    result = grow(
        image,
        labels,
        [(2, 0, 0, 0)],
        max_changed_voxels=2,
    )

    assert result.aborted
    assert result.abort_reason == "maximum changed voxel safety limit exceeded"
    assert result.accepted_voxels.size == 0
    assert result.changed_voxel_count == 0
    assert np.array_equal(labels, original)
