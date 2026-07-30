from __future__ import annotations

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SnapBrushPanel(QWidget):
    """Controls for adjacent-frame image-patch matching."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        form = QFormLayout()

        self.frame_radius_label = QLabel()
        self.frame_radius = QSpinBox()
        self.frame_radius.setRange(1, 20)
        self.frame_radius.setValue(1)
        form.addRow(self.frame_radius_label, self.frame_radius)

        self.patch_radius_label = QLabel()
        self.patch_radius = QDoubleSpinBox()
        self.patch_radius.setRange(0.5, 40.0)
        self.patch_radius.setDecimals(1)
        self.patch_radius.setSingleStep(0.5)
        self.patch_radius.setValue(3.0)
        self.patch_radius.setSuffix(" mm")
        form.addRow(self.patch_radius_label, self.patch_radius)

        self.search_radius_label = QLabel()
        self.search_radius = QDoubleSpinBox()
        self.search_radius.setRange(0.0, 100.0)
        self.search_radius.setDecimals(1)
        self.search_radius.setSingleStep(1.0)
        self.search_radius.setValue(10.0)
        self.search_radius.setSuffix(" mm")
        form.addRow(self.search_radius_label, self.search_radius)

        self.minimum_similarity_label = QLabel()
        self.minimum_similarity = QDoubleSpinBox()
        self.minimum_similarity.setRange(0.0, 100.0)
        self.minimum_similarity.setDecimals(0)
        self.minimum_similarity.setSingleStep(5.0)
        self.minimum_similarity.setValue(50.0)
        self.minimum_similarity.setSuffix(" %")
        form.addRow(self.minimum_similarity_label, self.minimum_similarity)

        layout.addLayout(form)
        layout.addStretch()
        self.set_language("en")

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.frame_radius_label.setText("单侧邻帧数" if chinese else "Frames each side")
        self.patch_radius_label.setText("模板半径" if chinese else "Patch radius")
        self.search_radius_label.setText("搜索半径" if chinese else "Search radius")
        self.minimum_similarity_label.setText(
            "最低相似度" if chinese else "Minimum similarity"
        )
