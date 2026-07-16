from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


RENDER_STYLES = ("clinical", "matte", "glossy")
DETAIL_LEVELS = ("performance", "balanced", "fine")
DETAIL_REDUCTION = {
    "performance": 0.55,
    "balanced": 0.35,
    "fine": 0.10,
}


@dataclass(frozen=True)
class RenderSettings:
    style: str = "clinical"
    lighting: int = 100
    smoothing: int = 8
    detail: str = "balanced"

    @classmethod
    def normalized(cls, values: Mapping[str, object] | None = None) -> RenderSettings:
        source = values or {}
        style = str(source.get("style", cls.style))
        detail = str(source.get("detail", cls.detail))
        try:
            lighting = int(source.get("lighting", cls.lighting))
        except (TypeError, ValueError):
            lighting = cls.lighting
        try:
            smoothing = int(source.get("smoothing", cls.smoothing))
        except (TypeError, ValueError):
            smoothing = cls.smoothing
        return cls(
            style=style if style in RENDER_STYLES else cls.style,
            lighting=max(40, min(160, lighting)),
            smoothing=max(0, min(16, smoothing)),
            detail=detail if detail in DETAIL_LEVELS else cls.detail,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "style": self.style,
            "lighting": self.lighting,
            "smoothing": self.smoothing,
            "detail": self.detail,
        }
