from __future__ import annotations

from pathlib import Path

import nrrd
import nibabel as nib
import numpy as np
import pytest

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.model import default_label, labels_from_sequence, store_labels


SAMPLE_DIR = Path(
    "/nas-data2/ryy/CMR4DFlow2026/Segdata/data4seg/Aorta/"
    "Aorta_Center003_GE_15T_Voyager_Exam31216-12135015-C17-0006/4D"
)
REPOSITORY_SAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sample-data"


def sequence_header(time_first: bool = False) -> dict:
    spatial = [
        np.asarray([0.0, 0.0, -2.4]),
        np.asarray([0.0, -1.7, 0.0]),
        np.asarray([-1.2, 0.0, 0.0]),
    ]
    time = np.asarray([np.nan, np.nan, np.nan])
    directions = [time, *spatial] if time_first else [*spatial, time]
    kinds = ["list", "domain", "domain", "domain"] if time_first else [
        "domain", "domain", "domain", "list"
    ]
    return {
        "space": "right-anterior-superior",
        "space directions": np.asarray(directions),
        "space origin": np.asarray([20.0, 30.0, 40.0]),
        "kinds": kinds,
        "encoding": "gzip",
    }


@pytest.mark.parametrize(
    ("shape", "time_first", "expected_order"),
    [
        ((4, 5, 6, 2), False, (2, 1, 0, 3)),
        ((2, 4, 5, 6), True, (3, 2, 1, 0)),
    ],
)
def test_axis_transform_is_reversible(shape, time_first, expected_order):
    source = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    transform = AxisTransform.from_header(sequence_header(time_first), source.shape)
    canonical = transform.to_canonical(source)

    assert transform.original_axis_for_canonical == expected_order
    assert transform.flipped_canonical_axes == (True, True, True)
    assert np.array_equal(transform.to_original(canonical), source)


@pytest.mark.parametrize("time_first", [False, True])
def test_save_preserves_original_axis_semantics(tmp_path, time_first):
    shape = (2, 4, 5, 6) if time_first else (4, 5, 6, 2)
    source_data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    source_path = tmp_path / "source.seg.seq.nrrd"
    saved_path = tmp_path / "saved.seg.seq.nrrd"
    nrrd.write(str(source_path), source_data, sequence_header(time_first), index_order="F")

    sequence = Sequence4D.load(source_path)
    sequence.data[0, 0, 0, 0] = 77
    expected_original = sequence.transform.to_original(sequence.data).copy()
    sequence.save(saved_path)
    saved_data, saved_header = nrrd.read(str(saved_path), index_order="F")

    assert np.array_equal(saved_data, expected_original)
    assert tuple(saved_header["kinds"]) == tuple(sequence_header(time_first)["kinds"])
    assert np.allclose(
        saved_header["space directions"],
        sequence_header(time_first)["space directions"],
        equal_nan=True,
    )


def test_grid_compatibility_includes_physical_geometry():
    data = np.zeros((4, 5, 6, 2), dtype=np.uint8)
    transform = AxisTransform.from_header(sequence_header(), data.shape)
    first = Sequence4D(data.copy(), sequence_header(), transform)
    shifted_header = sequence_header()
    shifted_header["space origin"] = np.asarray([21.0, 30.0, 40.0])
    shifted = Sequence4D(
        data.copy(), shifted_header, AxisTransform.from_header(shifted_header, data.shape)
    )

    assert not first.compatible_with(shifted)


def test_3d_nrrd_is_normalized_with_one_time_frame_and_saved_as_3d(tmp_path):
    source = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    header = {
        "space": "right-anterior-superior",
        "space directions": np.diag([1.0, 2.0, 3.0]),
        "space origin": np.zeros(3),
        "kinds": ["domain", "domain", "domain"],
        "spacings": [1.0, 2.0, 3.0],
        "thicknesses": [1.0, 2.0, 3.0],
        "axis mins": [0.0, 0.0, 0.0],
        "axis maxs": [3.0, 8.0, 15.0],
        "labels": ["x", "y", "z"],
        "units": ["mm", "mm", "mm"],
    }
    source_path = tmp_path / "source.nrrd"
    saved_path = tmp_path / "saved.nrrd"
    nrrd.write(str(source_path), source, header, index_order="F")

    sequence = Sequence4D.load(source_path)
    sequence.save(saved_path)
    saved, _ = nrrd.read(str(saved_path), index_order="F")

    assert sequence.data.shape == (4, 5, 6, 1)
    assert saved.shape == source.shape
    assert np.array_equal(saved, source)


def test_3d_nrrd_can_map_to_all_4d_frames_and_save_as_4d(tmp_path):
    source = np.zeros((4, 5, 6), dtype=np.uint8)
    source[1:3, 2:4, 3:5] = 2
    header = {
        "space": "right-anterior-superior",
        "space directions": np.diag([1.0, 2.0, 3.0]),
        "space origin": np.zeros(3),
        "kinds": ["domain", "domain", "domain"],
        "spacings": [1.0, 2.0, 3.0],
        "thicknesses": [1.0, 2.0, 3.0],
        "axis mins": [0.0, 0.0, 0.0],
        "axis maxs": [3.0, 8.0, 15.0],
        "labels": ["x", "y", "z"],
        "units": ["mm", "mm", "mm"],
    }
    source_path = tmp_path / "source.seg.nrrd"
    saved_path = tmp_path / "mapped.seg.seq.nrrd"
    nrrd.write(str(source_path), source, header, index_order="F")

    mapped = Sequence4D.load(source_path).map_single_frame_to(4)
    mapped.save(saved_path)
    saved, saved_header = nrrd.read(str(saved_path), index_order="F")

    assert mapped.path == saved_path
    assert mapped.data.shape == (4, 5, 6, 4)
    assert np.all(mapped.data == mapped.data[..., [0]])
    assert saved.shape == (4, 5, 6, 4)
    assert tuple(saved_header["kinds"]) == ("domain", "domain", "domain", "list")
    for field in ("spacings", "thicknesses", "axis mins", "axis maxs", "labels", "units"):
        assert len(saved_header[field]) == 4


def test_3d_label_can_map_to_one_selected_frame():
    data = np.zeros((4, 5, 6, 1), dtype=np.uint8)
    data[1:3, 2:4, 3:5, 0] = 2
    transform = AxisTransform.from_header(
        {
            "space directions": np.diag([1.0, 1.0, 1.0]),
            "kinds": ["domain", "domain", "domain"],
        },
        data.shape[:3],
    )
    source = Sequence4D(data, {}, transform)

    mapped = source.map_single_frame_to(5, target_frame=3)

    assert not np.any(mapped.data[..., :3])
    assert np.array_equal(mapped.data[..., 3], data[..., 0])
    assert not np.any(mapped.data[..., 4])
    assert mapped.path is None
    assert mapped.dirty


def test_3d_nifti_can_map_to_4d_and_round_trip(tmp_path):
    source = np.zeros((4, 5, 6), dtype=np.uint8)
    source[1:3, 2:4, 3:5] = 4
    affine = np.asarray(
        [[-2.0, 0.0, 0.0, 20.0], [0.0, 3.0, 0.0, -12.0], [0.0, 0.0, 4.0, 7.0], [0.0, 0.0, 0.0, 1.0]]
    )
    source_path = tmp_path / "source.seg.nii.gz"
    saved_path = tmp_path / "mapped.seg.nii.gz"
    nib.save(nib.Nifti1Image(source, affine), source_path)

    mapped = Sequence4D.load(source_path).map_single_frame_to(3, target_frame=1)
    mapped.save(saved_path)
    saved = nib.load(saved_path)

    saved_data = np.asanyarray(saved.dataobj)
    assert saved_data.shape == (4, 5, 6, 3)
    assert not np.any(saved_data[..., 0])
    assert np.array_equal(saved_data[..., 1], source)
    assert not np.any(saved_data[..., 2])
    assert np.allclose(saved.affine, affine)


@pytest.mark.parametrize("shape", [(4, 5, 6), (4, 5, 6, 2)])
def test_nifti_3d_and_4d_round_trip_preserves_layout(tmp_path, shape):
    source = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    affine = np.asarray(
        [[-2.0, 0.0, 0.0, 20.0], [0.0, 3.0, 0.0, -12.0], [0.0, 0.0, 4.0, 7.0], [0.0, 0.0, 0.0, 1.0]]
    )
    source_path = tmp_path / "source.nii.gz"
    saved_path = tmp_path / "saved.nii.gz"
    nib.save(nib.Nifti1Image(source, affine), source_path)

    sequence = Sequence4D.load(source_path)
    sequence.save(saved_path)
    saved = nib.load(saved_path)

    assert sequence.data.shape == (*shape[:3], shape[3] if len(shape) == 4 else 1)
    assert np.array_equal(np.asanyarray(saved.dataobj), source)
    assert np.allclose(saved.affine, affine)


def test_nifti_round_trip_preserves_label_definitions_in_an_extension(tmp_path):
    source = np.zeros((4, 5, 6), dtype=np.uint8)
    source[1:3, 2:4, 2:5] = 3
    source_path = tmp_path / "source.nii.gz"
    saved_path = tmp_path / "saved.nii.gz"
    nib.save(nib.Nifti1Image(source, np.eye(4)), source_path)
    sequence = Sequence4D.load(source_path)
    definition = default_label(3)
    definition.name = "Vessel"
    definition.color = (12, 34, 56)
    store_labels(sequence, {3: definition})

    sequence.save(saved_path)
    reloaded = Sequence4D.load(saved_path)
    labels = labels_from_sequence(reloaded)

    assert labels[3].name == "Vessel"
    assert labels[3].color == (12, 34, 56)


@pytest.mark.parametrize(
    ("name", "expected_dtype"),
    [("pcmra.seq.nrrd", np.dtype("float32")), ("seg.seq.nrrd", np.dtype("uint8"))],
)
def test_repository_sample_layout_is_reversible(name, expected_dtype):
    path = REPOSITORY_SAMPLE_DIR / name
    original, _ = nrrd.read(str(path), index_order="F")
    sequence = Sequence4D.load(path)

    assert original.dtype == expected_dtype
    assert sequence.data.shape == (74, 112, 110, 18)
    assert np.array_equal(sequence.transform.to_original(sequence.data), original)


@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="site-specific sample data is unavailable")
@pytest.mark.parametrize("name", ["seg.seq.nrrd", "edited_mask.seq.nrrd"])
def test_real_sample_layout_round_trip(name):
    path = SAMPLE_DIR / name
    original, _ = nrrd.read(str(path), index_order="F")
    sequence = Sequence4D.load(path)

    assert sequence.data.shape == (74, 112, 110, 18)
    assert np.array_equal(sequence.transform.to_original(sequence.data), original)
