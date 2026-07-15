import numpy as np

from spatiotemporal_labeler.tools import (
    automatic_thresholds,
    build_threshold_mask,
    connected_seed_region,
    kittler_threshold,
)


def test_manual_threshold_is_inclusive_and_finite():
    data = np.asarray([0.0, 1.0, 2.0, 3.0, np.nan])

    selected = build_threshold_mask(data.reshape(1, 5), "manual", 1.0, 2.0)

    assert selected.tolist() == [[False, True, True, False, False]]


def test_global_automatic_methods_separate_a_bimodal_image():
    data = np.concatenate([np.zeros(1000), np.full(1000, 10.0)])

    for method in ("otsu", "triangle", "li", "yen", "isodata", "multi_otsu", "kittler"):
        lower, upper = automatic_thresholds(data, method)
        assert 0.0 <= lower <= 10.0
        assert upper == 10.0

    assert 0.0 <= kittler_threshold(data) <= 10.0


def test_local_methods_and_hysteresis_preserve_shape():
    data = np.zeros((9, 9, 3, 2), dtype=np.float32)
    data[3:7, 3:7, 1, :] = 5.0

    for method in ("local_gaussian", "sauvola", "phansalkar", "hysteresis"):
        selected = build_threshold_mask(data, method, 1.0, 4.0 if method == "hysteresis" else 5.0, radius=2)
        assert selected.shape == data.shape
        assert selected.dtype == np.bool_


def test_connected_seed_region_does_not_cross_a_barrier():
    candidate = np.ones((7, 7), dtype=bool)
    candidate[3, :] = False

    selected = connected_seed_region(candidate, (1, 2))

    assert selected[1, 2]
    assert not np.any(selected[4:, :])
