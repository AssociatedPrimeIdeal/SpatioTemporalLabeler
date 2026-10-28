from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest

from spatiotemporal_labeler.tools.streamlines import (
    PlanarConfig, StreamlineCancelled, StreamlineConfig, _trace_planar_batch,
    initialize_planar_worker, trace_planar_slices,
)


@pytest.fixture(scope="module")
def worker_pool():
    context = get_context("spawn")
    cancel = context.Event()
    with ProcessPoolExecutor(max_workers=2, mp_context=context,
                             initializer=initialize_planar_worker, initargs=(cancel,)) as pool:
        yield pool, cancel


@pytest.mark.parametrize("integrator", ["midpoint", "rk4"])
@pytest.mark.parametrize("distribution", ["grid", "random"])
def test_process_batches_match_serial_slices_bit_for_bit(worker_pool, integrator, distribution):
    pool, _ = worker_pool
    rng = np.random.default_rng(51)
    shape = (9, 10, 11)
    components = [rng.normal(8, 1, shape).astype(np.float32) for _ in range(3)]
    direction = np.array([[1, 0.2, 0], [0, 1, 0.1], [0, 0, 1]], dtype=float)
    selected = rng.random(shape) > 0.2
    selected[:, :, :2] = False
    config = StreamlineConfig(seed_count=5, seed_distribution=distribution,
                              limit_grid_seeds=True, max_steps=5, integrator=integrator,
                              constrain_to_mask=True, rng_seed=6)
    planar = PlanarConfig(True, 4)
    arguments = (components, (1.3, 2.1, 0.8), direction, config, planar)
    serial = trace_planar_slices(*arguments, seed_mask=selected)
    parallel = trace_planar_slices(*arguments, seed_mask=selected, executor=pool, max_pending=2)
    for plane in serial:
        np.testing.assert_array_equal(parallel[plane].values, serial[plane].values)
        np.testing.assert_array_equal(parallel[plane].offsets, serial[plane].offsets)


def test_worker_event_cancels_without_returning_partial_geometry(worker_pool):
    pool, cancel = worker_pool
    cancel.set()
    try:
        future = pool.submit(_trace_planar_batch, np.ones((2, 8, 9, 3), np.float32),
                             None, np.ones(2), np.eye(3)[:, :2], StreamlineConfig())
        with pytest.raises(StreamlineCancelled):
            future.result(timeout=15)
    finally:
        cancel.clear()


def test_parent_can_cancel_before_submitting_slice_batches(worker_pool):
    pool, _ = worker_pool
    with pytest.raises(StreamlineCancelled):
        trace_planar_slices([np.ones((8, 9, 10), np.float32)] * 3,
                            (1, 1, 1), np.eye(3), StreamlineConfig(), PlanarConfig(True),
                            cancelled=lambda: True, executor=pool)
