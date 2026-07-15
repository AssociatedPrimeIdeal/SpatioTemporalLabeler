import numpy as np

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.model import build_edit_command


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
