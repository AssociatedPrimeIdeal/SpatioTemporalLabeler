from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QVBoxLayout, QWidget,
)

from spatiotemporal_labeler.i18n import translate
from spatiotemporal_labeler.tools.streamlines import StreamlineConfig
from .value_slider import FloatSliderSpin


class StreamlinePanel(QWidget):
    computationChanged = Signal()
    displayChanged = Signal()
    recalculateRequested = Signal()
    clearRequested = Signal()

    def __init__(self, settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.language = "en"
        self._sources = []
        self._progress = (0, 0, "")
        layout = QVBoxLayout(self)
        self.enabled = QCheckBox()
        layout.addWidget(self.enabled)
        self.explanation = QLabel()
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)
        self.form = QFormLayout()
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.addLayout(self.form)
        self.sources, self.directions = [], []
        self.labels = {}
        for index, (name, direction) in enumerate(zip(("flow_x", "flow_y", "flow_z"), ("F", "P", "L"))):
            row = QHBoxLayout()
            source = QComboBox()
            source.setMinimumContentsLength(8)
            source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            source.addItem("—", None)
            axis = QComboBox()
            for value in ("R", "L", "A", "P", "H", "F"):
                axis.addItem(value, value)
            value = str(settings.value(f"streamlines/direction{index}", direction))
            axis.setCurrentIndex(max(0, axis.findData(value)))
            row.addWidget(source, 1)
            row.addWidget(axis)
            self.form.addRow(name, row)
            self.sources.append(source)
            self.directions.append(axis)
            source.currentIndexChanged.connect(self._computation_changed)
            axis.currentIndexChanged.connect(self._computation_changed)
        self.units = QComboBox()
        self.units.addItems(["Source values", "cm/s", "m/s", "Phase (rad) + VENC"])
        self.units.setCurrentIndex(int(settings.value("streamlines/units", 0)))
        self._add_row("stream_units", self.units)
        self.venc = []
        venc_row = QHBoxLayout()
        for index in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(0, 3000)
            spin.setDecimals(2)
            spin.setSpecialValueText("VENC")
            spin.setValue(float(settings.value(f"streamlines/venc{index}", 0)))
            spin.setKeyboardTracking(False)
            venc_row.addWidget(spin)
            self.venc.append(spin)
        self._add_row("stream_venc", venc_row)
        self.seed_count = QSpinBox()
        self.seed_count.setRange(1, 2000)
        self.seed_count.setValue(int(settings.value("streamlines/seeds", 300)))
        self._add_row("stream_seeds", self.seed_count)
        self.max_steps = QSpinBox()
        self.max_steps.setRange(1, 1000)
        self.max_steps.setValue(int(settings.value("streamlines/steps", 160)))
        self._add_row("stream_steps", self.max_steps)
        self.step = self._slider("stream_step", "step", 0.1, 10, 1, 2)
        self.length = self._slider("stream_length", "length", 1, 1000, 200, 1)
        self.min_speed = self._slider("stream_min_speed", "min_speed", 0, 200, 0.1, 3)
        self.slab = self._slider("stream_slab", "slab", 0, 30, 0, 2)
        self.slab.spin.setSpecialValueText("Auto")
        self.show_2d = QCheckBox()
        self.show_3d = QCheckBox()
        self.arrows = QCheckBox()
        for widget, key in ((self.show_2d, "2d"), (self.show_3d, "3d"), (self.arrows, "arrows")):
            value = settings.value(f"streamlines/{key}", True)
            widget.setChecked(str(value).lower() not in {"false", "0"})
            layout.addWidget(widget)
            widget.toggled.connect(self._display_changed)
        self.width = self._slider("stream_width", "width", 0.5, 5, 1.5, 1)
        self.opacity = self._slider("stream_opacity", "opacity", 0, 1, 0.9, 2)
        self.speed_max = self._slider("stream_speed_max", "speed_max", 0.01, 300, 1, 2)
        self.scale_hint = QLabel()
        self.scale_hint.setWordWrap(True)
        self.scale_hint.setStyleSheet("color: #57cae8")
        layout.addWidget(self.scale_hint)
        buttons = QHBoxLayout()
        self.recalculate = QPushButton()
        self.clear = QPushButton()
        buttons.addWidget(self.recalculate)
        buttons.addWidget(self.clear)
        layout.addLayout(buttons)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)
        self.enabled.toggled.connect(self._computation_changed)
        self.units.currentIndexChanged.connect(self._units_changed)
        for spin in self.venc:
            spin.valueChanged.connect(self._computation_changed)
        self.seed_count.valueChanged.connect(self._computation_changed)
        self.max_steps.valueChanged.connect(self._computation_changed)
        for widget in (self.step, self.length, self.min_speed, self.slab):
            widget.valueChanged.connect(self._computation_changed)
        for widget in (self.width, self.opacity, self.speed_max):
            widget.valueChanged.connect(self._display_changed)
        self.recalculate.clicked.connect(self.recalculateRequested)
        self.clear.clicked.connect(self.clearRequested)
        self._update_unit_controls()
        self.set_language("en")

    def _add_row(self, key, widget):
        label = QLabel()
        label.setWordWrap(True)
        label.setMinimumWidth(125)
        label.setMaximumWidth(140)
        self.labels[key] = label
        self.form.addRow(label, widget)

    def _slider(self, text, key, minimum, maximum, default, decimals):
        slider = FloatSliderSpin(decimals=decimals)
        slider.slider.setMinimumWidth(70)
        slider.spin.setMinimumWidth(78)
        slider.spin.setMaximumWidth(94)
        slider.set_range(minimum, maximum)
        slider.set_value(float(self.settings.value(f"streamlines/{key}", default)))
        self._add_row(text, slider)
        return slider

    def _save_settings(self):
        for index, direction in enumerate(self.directions):
            self.settings.setValue(f"streamlines/direction{index}", direction.currentData())
            self.settings.setValue(f"streamlines/venc{index}", self.venc[index].value())
        for key, value in {
            "units": self.units.currentIndex(), "seeds": self.seed_count.value(),
            "steps": self.max_steps.value(), "step": self.step.value(),
            "length": self.length.value(), "min_speed": self.min_speed.value(),
            "slab": self.slab.value(), "2d": self.show_2d.isChecked(),
            "3d": self.show_3d.isChecked(), "arrows": self.arrows.isChecked(),
            "width": self.width.value(), "opacity": self.opacity.value(),
            "speed_max": self.speed_max.value(),
        }.items():
            self.settings.setValue(f"streamlines/{key}", value)

    def _computation_changed(self, *_args):
        self._save_settings()
        self.computationChanged.emit()

    def _display_changed(self, *_args):
        self._save_settings()
        self._update_scale_hint()
        self.displayChanged.emit()

    def set_sources(self, sources):
        self._sources = list(sources)
        previous = tuple(combo.currentData() for combo in self.sources)
        for index, combo in enumerate(self.sources):
            selected = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("—", None)
            default = None
            for sequence in sources:
                combo.addItem(sequence.display_name, id(sequence))
                if sequence.display_name.lower().split(".")[0] == ("flow_x", "flow_y", "flow_z")[index]:
                    default = id(sequence)
            target = selected if combo.findData(selected) > 0 else default
            combo.setCurrentIndex(max(0, combo.findData(target)))
            combo.blockSignals(False)
        if previous != tuple(combo.currentData() for combo in self.sources):
            self.computationChanged.emit()

    def selected_sources(self):
        by_id = {id(source): source for source in self._sources}
        return tuple(by_id.get(combo.currentData()) for combo in self.sources)

    def config(self):
        return StreamlineConfig(
            directions=tuple(combo.currentData() for combo in self.directions),
            seed_count=self.seed_count.value(), step_mm=self.step.value(),
            max_steps=self.max_steps.value(), max_length_mm=self.length.value(),
            min_speed=self.min_speed.value(), source_scale=100.0 if self.units.currentIndex() == 2 else 1.0,
            component_scales=(tuple(spin.value() / np.pi for spin in self.venc)
                              if self.units.currentIndex() == 3 else (1.0, 1.0, 1.0)),
        )

    def display_style(self):
        return dict(width=self.width.value(), opacity=self.opacity.value(),
                    speed_max=self.speed_max.value(), arrows=self.arrows.isChecked())

    def set_progress(self, done, total, message=""):
        self._progress = (done, total, message)
        if message:
            self.status.setText(message)
        else:
            key = "stream_ready" if total and done == total else "stream_progress"
            self.status.setText(translate(self.language, key, done=done, total=total))

    def _update_scale_hint(self):
        self.scale_hint.setText(translate(self.language, "stream_scale", value=f"{self.speed_max.value():g}",
                                          unit=self.speed_unit()))

    def speed_unit(self):
        return translate(self.language, "stream_source_units") if self.units.currentIndex() == 0 else "cm/s"

    def _update_unit_controls(self):
        native = self.units.currentIndex() == 0
        self.min_speed.set_range(0, 10 if native else 300)
        self.speed_max.set_range(0.01 if native else 1, 10 if native else 1000)
        for spin in self.venc:
            spin.setEnabled(self.units.currentIndex() == 3)
        self.form.setRowVisible(self.labels["stream_venc"], self.units.currentIndex() == 3)
        for key in ("stream_min_speed", "stream_speed_max"):
            self.labels[key].setText(translate(self.language, key, unit=self.speed_unit()))

    def _units_changed(self, *_args):
        self._update_unit_controls()
        self.min_speed.set_value(0.1 if self.units.currentIndex() == 0 else 5)
        self.speed_max.set_value(1 if self.units.currentIndex() == 0 else 100)
        self._update_scale_hint()
        self._computation_changed()

    def set_language(self, language):
        self.language = language
        for key, label in self.labels.items():
            label.setText(translate(language, key, unit=self.speed_unit()))
        for widget, key in (
            (self.enabled, "stream_enabled"), (self.explanation, "stream_explanation"),
            (self.show_2d, "stream_2d"), (self.show_3d, "stream_3d"),
            (self.arrows, "stream_arrows"), (self.recalculate, "stream_recalculate"),
            (self.clear, "stream_clear"),
        ):
            widget.setText(translate(language, key))
        for combo in self.directions:
            for index in range(combo.count()):
                direction = combo.itemData(index)
                combo.setItemText(index, translate(language, "stream_dir_" + direction))
        self.units.setItemText(0, translate(language, "stream_native"))
        self.units.setItemText(3, translate(language, "stream_phase"))
        self._update_scale_hint()
        self.set_progress(*self._progress)
