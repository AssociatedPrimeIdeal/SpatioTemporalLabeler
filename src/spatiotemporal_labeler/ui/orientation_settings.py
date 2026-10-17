from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


VIEW_AXES = {
    "X-Y": ("X", "Y"),
    "X-Z": ("X", "Z"),
    "Y-Z": ("Y", "Z"),
    "X-T": ("X", "T"),
    "Y-T": ("Y", "T"),
    "Z-T": ("Z", "T"),
}
AXIS_DIRECTIONS = {
    "X": ("L", "R"),
    "Y": ("P", "A"),
    "Z": ("F", "H"),
    "T": ("Earlier", "Later"),
}
ANATOMICAL_VECTORS = {
    "R": (1.0, 0.0, 0.0),
    "L": (-1.0, 0.0, 0.0),
    "A": (0.0, 1.0, 0.0),
    "P": (0.0, -1.0, 0.0),
    "H": (0.0, 0.0, 1.0),
    "F": (0.0, 0.0, -1.0),
}


def _boolean(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return default


@dataclass(frozen=True)
class OrientationSettings:
    """Display preferences only; sequence coordinates remain canonical RAS."""

    horizontal_flips: tuple[bool, ...] = (False, False, True, False, False, False)
    vertical_flips: tuple[bool, ...] = (False, False, False, False, False, False)
    camera_view: str = "A"
    camera_up: str = "H"
    axis_directions: tuple[str, str, str] = ("R", "A", "H")

    def flips(self, plane: str) -> tuple[bool, bool]:
        index = tuple(VIEW_AXES).index(plane)
        return self.horizontal_flips[index], self.vertical_flips[index]

    def axis_label(self, plane: str, coordinate: int) -> str:
        axis = VIEW_AXES[plane][coordinate]
        if axis == "T":
            return "Time"
        directions = AXIS_DIRECTIONS[axis]
        if self.flips(plane)[coordinate]:
            directions = directions[::-1]
        return " - ".join(directions)

    def as_dict(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for plane in VIEW_AXES:
            horizontal, vertical = self.flips(plane)
            values[f"{plane}/horizontal"] = horizontal
            values[f"{plane}/vertical"] = vertical
        values["3d/view"] = self.camera_view
        values["3d/up"] = self.camera_up
        for axis, direction in zip("XYZ", self.axis_directions):
            values[f"3d/{axis}"] = direction
        return values

    @classmethod
    def normalized(cls, values: Mapping[str, object] | None = None) -> OrientationSettings:
        source = values or {}
        defaults = cls()
        horizontal = []
        vertical = []
        for plane in VIEW_AXES:
            flip_h, flip_v = defaults.flips(plane)
            horizontal.append(_boolean(source.get(f"{plane}/horizontal"), flip_h))
            vertical.append(_boolean(source.get(f"{plane}/vertical"), flip_v))
        view = str(source.get("3d/view", defaults.camera_view))
        if view not in ANATOMICAL_VECTORS:
            view = defaults.camera_view
        up = str(source.get("3d/up", defaults.camera_up))
        if up not in ANATOMICAL_VECTORS or any(
            a * b for a, b in zip(ANATOMICAL_VECTORS[view], ANATOMICAL_VECTORS[up])
        ):
            up = "A" if view in {"H", "F"} else "H"
        arrows = []
        for axis, default in zip("XYZ", defaults.axis_directions):
            direction = str(source.get(f"3d/{axis}", default))
            arrows.append(direction if direction in AXIS_DIRECTIONS[axis] else default)
        return cls(tuple(horizontal), tuple(vertical), view, up, tuple(arrows))
