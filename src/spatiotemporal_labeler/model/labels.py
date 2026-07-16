from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from spatiotemporal_labeler.io import Sequence4D


LABEL_HEADER_KEY = "SpatioTemporalLabelerLabels"
DEFAULT_COLORS = (
    (0, 188, 212),
    (239, 83, 80),
    (255, 193, 7),
    (102, 187, 106),
    (171, 71, 188),
    (255, 112, 67),
    (66, 165, 245),
    (38, 166, 154),
    (236, 64, 122),
    (212, 225, 87),
    (126, 87, 194),
    (255, 167, 38),
)


@dataclass
class LabelDefinition:
    value: int
    name: str
    color: tuple[int, int, int]
    visible: bool = True
    opacity: float = 1.0


def default_label(value: int) -> LabelDefinition:
    return LabelDefinition(value, f"Label {value}", DEFAULT_COLORS[(value - 1) % len(DEFAULT_COLORS)])


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
