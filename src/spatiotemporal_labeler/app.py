from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .io import is_supported_image_path
from .ui import MainWindow


MASK_NAME_TOKENS = ("seg", "mask", "label")


def application_icon_path() -> Path | None:
    candidates = [Path(__file__).resolve().parent / "assets" / "app-icon.png"]
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "app-icon.png")
    return next((path for path in candidates if path.is_file()), None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive 3D+t medical image segmentation editor")
    parser.add_argument("paths", nargs="*", type=Path, help="NIfTI/NRRD files or a directory")
    parser.add_argument("--image", action="append", default=[], type=Path, help="3D/4D image")
    parser.add_argument("--mask", action="append", default=[], type=Path, help="3D/4D label map")
    return parser


def is_mask_path(path: Path) -> bool:
    return any(token in path.name.lower() for token in MASK_NAME_TOKENS)


def preferred_last(paths: list[Path], preferred_name: str) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: (path.name.lower() == preferred_name, path.name.lower()),
    )


def startup_files(arguments: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    images = list(arguments.image)
    masks = list(arguments.mask)
    for path in arguments.paths:
        if path.is_dir():
            sequence_files = sorted(
                (
                    candidate
                    for candidate in path.iterdir()
                    if candidate.is_file()
                    and is_supported_image_path(candidate)
                ),
                key=lambda candidate: candidate.name.lower(),
            )
            directory_images = [
                candidate for candidate in sequence_files if not is_mask_path(candidate)
            ]
            directory_masks = [candidate for candidate in sequence_files if is_mask_path(candidate)]
            images.extend(preferred_last(directory_images, "pcmra.seq.nrrd"))
            masks.extend(preferred_last(directory_masks, "seg.seq.nrrd"))
        elif is_mask_path(path):
            masks.append(path)
        else:
            images.append(path)
    return list(dict.fromkeys(images)), list(dict.fromkeys(masks))


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    application = QApplication(sys.argv[:1])
    application.setApplicationName("SpatioTemporal Labeler")
    application.setOrganizationName("Hulab")
    icon_path = application_icon_path()
    if icon_path is not None:
        application.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    images, masks = startup_files(arguments)
    for path in images:
        window._load_with_feedback(path, is_mask=False)
    for path in masks:
        window._load_with_feedback(path, is_mask=True)
    window.show()
    return application.exec()
