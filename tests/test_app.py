from __future__ import annotations

from spatiotemporal_labeler.app import application_icon_path, build_parser, startup_files


def test_source_application_icon_is_available():
    icon = application_icon_path()

    assert icon is not None
    assert icon.name == "app-icon.png"
    assert icon.parent.name == "assets"


def test_sequence_directory_discovers_all_images_and_masks(tmp_path):
    names = [
        "edited_mask.seq.nrrd",
        "flow_x.seq.nrrd",
        "flow_y.seq.nrrd",
        "flow_z.seq.nrrd",
        "mag.seq.nrrd",
        "pcmra.seq.nrrd",
        "seg.seq.nrrd",
        "ignored.nrrd",
        "anatomy.nii.gz",
    ]
    for name in names:
        (tmp_path / name).touch()

    arguments = build_parser().parse_args([str(tmp_path)])
    images, masks = startup_files(arguments)

    assert [path.name for path in images] == [
        "anatomy.nii.gz",
        "flow_x.seq.nrrd",
        "flow_y.seq.nrrd",
        "flow_z.seq.nrrd",
        "ignored.nrrd",
        "mag.seq.nrrd",
        "pcmra.seq.nrrd",
    ]
    assert [path.name for path in masks] == [
        "edited_mask.seq.nrrd",
        "seg.seq.nrrd",
    ]


def test_startup_files_deduplicates_explicit_and_discovered_paths(tmp_path):
    image = tmp_path / "mag.seq.nrrd"
    mask = tmp_path / "seg.seq.nrrd"
    image.touch()
    mask.touch()

    arguments = build_parser().parse_args(
        ["--image", str(image), "--mask", str(mask), str(tmp_path)]
    )
    images, masks = startup_files(arguments)

    assert images == [image]
    assert masks == [mask]
