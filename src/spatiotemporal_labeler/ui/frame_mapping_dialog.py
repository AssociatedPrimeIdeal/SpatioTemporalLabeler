from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class FrameMappingDialog(QDialog):
    """Choose how one 3D label frame maps onto a 4D image sequence."""

    def __init__(
        self,
        frame_count: int,
        language: str = "en",
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        chinese = language == "zh_CN"
        self.setWindowTitle("映射 3D 标签" if chinese else "Map 3D labels")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "当前图像包含多个时间帧。请选择如何使用这个 3D 标签。"
                if chinese
                else "The active image has multiple time frames. Choose how to use this 3D label sequence."
            )
        )
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("复制到所有时间帧" if chinese else "Copy to all time frames", "all")
        self.mode.addItem("仅放入指定帧" if chinese else "Place in one frame", "single")
        form.addRow("方式" if chinese else "Mapping", self.mode)
        self.frame = QSpinBox()
        self.frame.setRange(1, max(1, int(frame_count)))
        form.addRow("目标帧" if chinese else "Target frame", self.frame)
        layout.addLayout(form)
        self.mode.currentIndexChanged.connect(self._sync_enabled)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._sync_enabled()

    def _sync_enabled(self, _index: int = 0) -> None:
        self.frame.setEnabled(self.mode.currentData() == "single")

    def target_frame(self) -> int | None:
        return self.frame.value() - 1 if self.mode.currentData() == "single" else None
