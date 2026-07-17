import numpy as np

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.model import (
    build_edit_command,
    capture_frames,
    restore_frames,
)


def make_mask() -> Sequence4D:
    transform = AxisTransform(
        original_axis_for_canonical=(0, 1, 2, 3),
        flipped_canonical_axes=(False, False, False),
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_ras=(0.0, 0.0, 0.0),
        direction_ras=np.eye(3),
    )
    return Sequence4D(np.zeros((5, 6, 7, 3), dtype=np.uint8), {}, transform)


def test_all_frame_edit_is_one_sparse_undo_command():
    mask = make_mask()
    frames = (0, 1, 2)
    before = mask.data[..., list(frames)].copy()
    mask.data[2, 3, 4, :] = 7

    command = build_edit_command(mask, frames, before, focus_frame=1)

    assert command is not None
    assert command.flat_indices.size == 3
    assert command.focus_frame == 1
    command.undo()
    assert np.all(mask.data[2, 3, 4, :] == 0)
    command.redo()
    assert np.all(mask.data[2, 3, 4, :] == 7)


def test_noop_edit_does_not_create_history():
    mask = make_mask()
    frames = (1,)
    before = mask.data[..., list(frames)].copy()

    assert build_edit_command(mask, frames, before, focus_frame=1) is None


def test_bounded_edit_builds_global_sparse_indices():
    mask = make_mask()
    frames = (0, 2)
    before = mask.data[..., list(frames)].copy()
    mask.data[3, 4, 5, 2] = 9

    command = build_edit_command(
        mask,
        frames,
        before,
        focus_frame=2,
        spatial_bounds=(slice(2, 4), slice(3, 5), slice(4, 6)),
    )

    assert command is not None
    assert command.flat_indices.tolist() == [
        np.ravel_multi_index((3, 4, 5, 2), mask.data.shape)
    ]
    command.undo()
    assert mask.data[3, 4, 5, 2] == 0
    command.redo()
    assert mask.data[3, 4, 5, 2] == 9


def test_capture_and_restore_contiguous_and_sparse_frames():
    mask = make_mask()
    mask.data[..., 0] = 1
    mask.data[..., 1] = 2
    mask.data[..., 2] = 3

    contiguous = capture_frames(mask, (0, 1, 2))
    sparse = capture_frames(mask, (0, 2))
    mask.data[...] = 9
    restore_frames(mask, (0, 1, 2), contiguous)

    assert np.array_equal(mask.data, contiguous)
    mask.data[..., 0] = 7
    mask.data[..., 2] = 7
    restore_frames(mask, (0, 2), sparse)
    assert np.all(mask.data[..., 0] == 1)
    assert np.all(mask.data[..., 1] == 2)
    assert np.all(mask.data[..., 2] == 3)


def test_edit_command_reports_changed_positive_labels():
    mask = make_mask()
    frames = (0,)
    before = capture_frames(mask, frames)
    mask.data[1, 1, 1, 0] = 2
    mask.data[2, 2, 2, 0] = 7

    command = build_edit_command(mask, frames, before, focus_frame=0)

    assert command is not None
    assert command.changed_label_values() == {2, 7}
