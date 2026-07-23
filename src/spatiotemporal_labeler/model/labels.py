from __future__ import annotations

import colorsys
import json
from dataclasses import asdict, dataclass

import numpy as np

from spatiotemporal_labeler.io import Sequence4D


LABEL_HEADER_KEY = "SpatioTemporalLabelerLabels"


@dataclass
class LabelDefinition:
    value: int
    name: str
    color: tuple[int, int, int]
    visible: bool = True
    opacity: float = 1.0


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result


def _default_color(value: int) -> tuple[int, int, int]:
    # Independent low-discrepancy dimensions spread neighboring labels across
    # hue while also varying saturation and brightness.
    hue = _radical_inverse(value, 2)
    saturation = 0.62 + 0.30 * _radical_inverse(value, 3)
    brightness = 0.74 + 0.24 * _radical_inverse(value, 5)
    return tuple(
        int(round(channel * 255.0))
        for channel in colorsys.hsv_to_rgb(hue, saturation, brightness)
    )


def default_label(value: int) -> LabelDefinition:
    return LabelDefinition(value, f"Label {value}", _default_color(value))


def labels_from_sequence(sequence: Sequence4D) -> dict[int, LabelDefinition]:
    definitions: dict[int, LabelDefinition] = {}
    raw = sequence.header.get(LABEL_HEADER_KEY)
    if raw:
        try:
            for item in json.loads(str(raw)):
                value = int(item["value"])
                color = tuple(int(channel) for channel in item["color"])
                opacity = float(np.clip(float(item.get("opacity", 1.0)), 0.0, 1.0))
                definitions[value] = LabelDefinition(
                    value, str(item["name"]), color, opacity=opacity
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            definitions = {}
    for value in np.unique(sequence.data):
        numeric = int(value)
        if numeric > 0 and numeric not in definitions:
            definitions[numeric] = default_label(numeric)
    if not definitions:
        definitions[1] = default_label(1)
    return dict(sorted(definitions.items()))


def store_labels(sequence: Sequence4D, definitions: dict[int, LabelDefinition]) -> None:
    records = []
    for definition in sorted(definitions.values(), key=lambda item: item.value):
        record = asdict(definition)
        record.pop("visible", None)
        records.append(record)
    sequence.header[LABEL_HEADER_KEY] = json.dumps(records, separators=(",", ":"))
