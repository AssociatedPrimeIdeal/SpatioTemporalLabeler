from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
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

        self.all_frames = QCheckBox()
        self.all_frames.setChecked(True)
        self.all_frames.toggled.connect(self._update_frame_radius_enabled)
        form.addRow(self.all_frames)

        self.frame_radius_label = QLabel()
        self.frame_radius = QSpinBox()
        self.frame_radius.setRange(1, 20)
        self.frame_radius.setValue(1)
        form.addRow(self.frame_radius_label, self.frame_radius)

        self.patch_radius_label = QLabel()
        self.patch_radius = QSpinBox()
        self.patch_radius.setRange(1, 40)
        self.patch_radius.setValue(3)
        self.patch_radius.setSuffix(" px")
        form.addRow(self.patch_radius_label, self.patch_radius)

        self.search_radius_label = QLabel()
        self.search_radius = QSpinBox()
        self.search_radius.setRange(0, 100)
        self.search_radius.setValue(5)
        self.search_radius.setSuffix(" px")
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
        self._update_frame_radius_enabled()

    def _update_frame_radius_enabled(self, _checked: bool = False) -> None:
        enabled = not self.all_frames.isChecked()
        self.frame_radius_label.setEnabled(enabled)
        self.frame_radius.setEnabled(enabled)

    def set_language(self, language: str) -> None:
        chinese = language == "zh_CN"
        self.all_frames.setText("所有时间帧" if chinese else "All temporal frames")
        self.frame_radius_label.setText("单侧邻帧数" if chinese else "Frames each side")
        self.patch_radius_label.setText("模板半径" if chinese else "Patch radius")
        self.search_radius_label.setText("最大位移" if chinese else "Maximum displacement")
        self.minimum_similarity_label.setText(
            "最低相似度" if chinese else "Minimum similarity"
        )
