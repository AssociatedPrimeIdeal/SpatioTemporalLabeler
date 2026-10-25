from __future__ import annotations

import re
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from spatiotemporal_labeler.i18n import translate
from spatiotemporal_labeler.tools.streamlines import PlanarConfig, StreamlineConfig
from .value_slider import FloatSliderSpin


class StreamlinePanel(QWidget):
    computationChanged = Signal()
    displayChanged = Signal()
    modeChanged = Signal()
    recalculateRequested = Signal()
    clearRequested = Signal()

    def __init__(self, settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.language = "en"
        self._sources, self._masks = [], []
        self._progress = (0, 0, "")
        self.labels, self._pages, self._rows, self._tips = {}, [], {}, {}
        self._options, self._saved = {}, {}
        layout = QVBoxLayout(self)
        self.enabled = QCheckBox()
        layout.addWidget(self.enabled)
        self.explanation = QLabel()
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.form = self._page("stream_tab_inputs")
        self.sources, self.directions = [], []
        for index, direction in enumerate(("F", "P", "L")):
            row = QHBoxLayout()
            source = QComboBox()
            source.setMinimumContentsLength(8)
            source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            source.addItem("—", None)
            axis = QComboBox()
            for value in ("R", "L", "A", "P", "H", "F"):
                axis.addItem(value, value)
            axis.setCurrentIndex(max(0, axis.findData(str(settings.value(f"streamlines/direction{index}", direction)))))
            row.addWidget(source, 1)
            row.addWidget(axis)
            self._row(f"stream_component{index}", row)
            self._tips[source] = "stream_help_inputs"
            self._tips[axis] = "stream_help_directions"
            self.sources.append(source)
            self.directions.append(axis)
        self.units = QComboBox()
        self.units.addItems(["Source values", "cm/s", "m/s", "Phase + VENC"])
        self.units.setCurrentIndex(int(settings.value("streamlines/units", 0)))
        self._row("stream_units", self.units, "stream_help_units")
        self.venc = []
        venc_row = QHBoxLayout()
        for index in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(0, 10000)
            spin.setDecimals(2)
            spin.setSpecialValueText("VENC")
            spin.setValue(float(settings.value(f"streamlines/venc{index}", 0)))
            spin.setKeyboardTracking(False)
            venc_row.addWidget(spin)
            self._tips[spin] = "stream_help_venc"
            self.venc.append(spin)
        self._row("stream_venc", venc_row)

        self._page("stream_tab_seeds")
        self.seed_mask = QComboBox()
        self.seed_mask.addItem("", None)
        self._row("stream_seed_mask", self.seed_mask, "stream_help_mask")
        self.mask_labels = QLineEdit(str(settings.value("streamlines/mask_labels", "")))
        self._row("stream_mask_labels", self.mask_labels, "stream_help_mask_labels")
        self.constrain_mask = self._check("stream_constrain_mask", "constrain_mask", False, "stream_help_constrain_mask")
        self.seed_count = self._integer("stream_seeds", "seeds", 1, 1000000, 300, "stream_help_seeds")
        self.distribution = self._choice("stream_distribution", "distribution", "grid", (
            ("random", "stream_random"), ("grid", "stream_grid")), "stream_help_distribution")
        self.limit_grid = self._check("stream_limit_grid", "limit_grid", False, "stream_help_limit_grid")
        self.grid_units = self._choice("stream_grid_units", "grid_units", "voxel", (
            ("voxel", "stream_grid_voxel"), ("mm", "stream_grid_mm")), "stream_help_grid_units")
        self.grid_spacing = self._slider("stream_grid_spacing", "grid_interval", 0.1, 50, 3, 2, "stream_help_grid_spacing")
        self.rng_seed = self._integer("stream_rng", "rng", 0, 2147483647, 0, "stream_help_rng")
        self.seed_min_speed = self._slider("stream_seed_min_speed", "seed_min_speed", 0, 300, 0, 3, "stream_help_seed_min_speed")

        self._page("stream_tab_trace")
        self.integration_direction = self._choice("stream_integration_direction", "integration_direction", "both", (
            ("both", "stream_both"), ("forward", "stream_forward"), ("backward", "stream_backward")), "stream_help_integration_direction")
        self.integrator = self._choice("stream_integrator", "integrator", "midpoint", (
            ("midpoint", "stream_midpoint"), ("rk4", "stream_rk4")), "stream_help_integrator")
        self.step = self._slider("stream_step", "step", 0.01, 10, 1, 2, "stream_help_step")
        self.max_steps = self._integer("stream_steps", "steps", 1, 1000000, 160, "stream_help_steps")
        self.length = self._slider("stream_length", "length", 0.1, 2000, 200, 1, "stream_help_length")
        self.min_length = self._slider("stream_min_length", "min_length", 0, 500, 0, 1, "stream_help_min_length")
        self.min_speed = self._slider("stream_min_speed", "min_speed", 0, 300, 0.1, 3, "stream_help_min_speed")
        self.max_turn = self._slider("stream_max_turn", "max_turn", 1, 180, 180, 1, "stream_help_max_turn")

        self._page("stream_tab_2d")
        self.show_2d = self._check("stream_2d", "2d", True, "stream_help_2d")
        self.mode_2d = self._choice("stream_mode_2d", "mode_2d", "projection", (
            ("planar", "stream_mode_planar"), ("projection", "stream_mode_projection")), "stream_help_mode_2d")
        self.mode_hint = QLabel()
        self.mode_hint.setWordWrap(True)
        self._current_form.addRow(self.mode_hint)
        self.planar_seeds = self._integer("stream_planar_seeds", "planar_seeds", 1, 1000000, 40, "stream_help_planar_seeds")
        self.slab = self._slider("stream_slab", "slab", 0, 100, 0, 2, "stream_help_slab")
        self.slab.spin.setSpecialValueText("Auto")

        self._page("stream_tab_style")
        self.show_3d = self._check("stream_3d", "3d", True, "stream_help_3d")
        self.arrows = self._check("stream_arrows", "arrows", True, "stream_help_arrows")
        self.width = self._slider("stream_width", "width", 0.5, 8, 1.5, 1, "stream_help_width")
        self.opacity = self._slider("stream_opacity", "opacity", 0, 1, 0.9, 2, "stream_help_opacity")
        self.display_min_speed = self._slider("stream_display_min_speed", "display_min_speed", 0, 300, 0, 3, "stream_help_display_min_speed")
        self.speed_max = self._slider("stream_speed_max", "speed_max", 0.01, 1000, 1, 2, "stream_help_speed_max")
        self.scale_hint = QLabel()
        self.scale_hint.setWordWrap(True)
        self._current_form.addRow(self.scale_hint)

        self.cache_group = QGroupBox()
        cache_layout = QVBoxLayout(self.cache_group)
        buttons = QHBoxLayout()
        self.recalculate, self.clear = QPushButton(), QPushButton()
        buttons.addWidget(self.recalculate)
        buttons.addWidget(self.clear)
        cache_layout.addLayout(buttons)
        self.status = QLabel()
        self.status.setWordWrap(True)
        cache_layout.addWidget(self.status)
        layout.addWidget(self.cache_group)

        self.enabled.toggled.connect(self._computation_changed)
        self.units.currentIndexChanged.connect(self._units_changed)
        for widget in self.sources + self.directions:
            widget.currentIndexChanged.connect(self._computation_changed)
        for widget in self.venc:
            widget.valueChanged.connect(self._computation_changed)
        self.seed_mask.currentIndexChanged.connect(self._mask_changed)
        self.mask_labels.editingFinished.connect(self._computation_changed)
        display_widgets = {self.show_2d, self.show_3d, self.arrows, self.width, self.opacity,
                           self.display_min_speed, self.speed_max}
        for widget in self._saved.values():
            if widget is self.mode_2d:
                continue
            signal = (widget.toggled if isinstance(widget, QCheckBox) else
                      widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged)
            signal.connect(self._display_changed if widget in display_widgets else self._computation_changed)
        self.distribution.currentIndexChanged.connect(self._update_mode_controls)
        self.limit_grid.toggled.connect(self._update_mode_controls)
        self.grid_units.currentIndexChanged.connect(self._update_mode_controls)
        self.mode_2d.currentIndexChanged.connect(self._update_mode_controls)
        self.mode_2d.currentIndexChanged.connect(self._mode_changed)
        self.recalculate.clicked.connect(self.recalculateRequested)
        self.clear.clicked.connect(self.clearRequested)
        self._update_unit_controls()
        self._update_mode_controls()
        self.set_language("en")

    def _page(self, key):
        page = QWidget()
        form = QFormLayout(page)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.tabs.addTab(page, "")
        self._pages.append(key)
        self._current_form = form
        return form

    def _row(self, key, widget, tip=None):
        label = QLabel()
        label.setWordWrap(True)
        label.setMinimumWidth(115)
        label.setMaximumWidth(140)
        self.labels[key] = label
        self._current_form.addRow(label, widget)
        self._rows[key] = (self._current_form, widget)
        if tip:
            self._tips[widget] = tip
            self._tips[label] = tip

    def _slider(self, text, key, minimum, maximum, default, decimals, tip):
        slider = FloatSliderSpin(decimals=decimals)
        slider.slider.setMinimumWidth(65)
        slider.spin.setMinimumWidth(75)
        slider.spin.setMaximumWidth(94)
        slider.set_range(minimum, maximum)
        slider.set_value(float(self.settings.value(f"streamlines/{key}", default)))
        self._row(text, slider, tip)
        self._saved[key] = slider
        return slider

    def _integer(self, text, key, minimum, maximum, default, tip):
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(int(self.settings.value(f"streamlines/{key}", default)))
        spin.setKeyboardTracking(False)
        spin.setAccelerated(True)
        self._row(text, spin, tip)
        self._saved[key] = spin
        return spin

    def _choice(self, text, key, default, options, tip):
        combo = QComboBox()
        for value, _ in options:
            combo.addItem(value, value)
        combo.setCurrentIndex(max(0, combo.findData(str(self.settings.value(f"streamlines/{key}", default)))))
        self._options[combo] = options
        self._row(text, combo, tip)
        self._saved[key] = combo
        return combo

    def _check(self, text, key, default, tip):
        check = QCheckBox()
        check.setChecked(str(self.settings.value(f"streamlines/{key}", default)).lower() not in {"false", "0"})
        self._row(text, check, tip)
        self._saved[key] = check
        return check

    def _save_settings(self):
        for index, direction in enumerate(self.directions):
            self.settings.setValue(f"streamlines/direction{index}", direction.currentData())
            self.settings.setValue(f"streamlines/venc{index}", self.venc[index].value())
        self.settings.setValue("streamlines/units", self.units.currentIndex())
        self.settings.setValue("streamlines/mask_labels", self.mask_labels.text())
        for key, widget in self._saved.items():
            value = (widget.isChecked() if isinstance(widget, QCheckBox) else
                     widget.currentData() if isinstance(widget, QComboBox) else widget.value())
            self.settings.setValue(f"streamlines/{key}", value)

    def _computation_changed(self, *_args):
        self._save_settings()
        self.computationChanged.emit()

    def _display_changed(self, *_args):
        self._save_settings()
        self._update_scale_hint()
        self.displayChanged.emit()

    def _mode_changed(self, *_args):
        self._save_settings()
        self.modeChanged.emit()

    def set_sources(self, sources):
        self._sources = list(sources)
        previous = tuple(combo.currentData() for combo in self.sources)
        for index, combo in enumerate(self.sources):
            selected = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("—", None)
            default = None
            for source in sources:
                combo.addItem(source.display_name, id(source))
                if source.display_name.lower().split(".")[0] == ("flow_x", "flow_y", "flow_z")[index]:
                    default = id(source)
            target = selected if combo.findData(selected) > 0 else default
            combo.setCurrentIndex(max(0, combo.findData(target)))
            combo.blockSignals(False)
        if previous != tuple(combo.currentData() for combo in self.sources):
            self.computationChanged.emit()

    def selected_sources(self):
        by_id = {id(source): source for source in self._sources}
        return tuple(by_id.get(combo.currentData()) for combo in self.sources)

    def set_masks(self, masks):
        self._masks = list(masks)
        selected = self.seed_mask.currentData()
        self.seed_mask.blockSignals(True)
        self.seed_mask.clear()
        self.seed_mask.addItem(translate(self.language, "stream_no_mask"), None)
        for mask in masks:
            self.seed_mask.addItem(mask.display_name, id(mask))
        self.seed_mask.setCurrentIndex(max(0, self.seed_mask.findData(selected)))
        self.seed_mask.blockSignals(False)
        self._update_mode_controls()
        if selected != self.seed_mask.currentData():
            self.computationChanged.emit()

    def selected_mask(self):
        return next((mask for mask in self._masks if id(mask) == self.seed_mask.currentData()), None)

    def selected_mask_labels(self):
        text = self.mask_labels.text().strip()
        if not text or self.selected_mask() is None:
            return ()
        try:
            values = tuple(sorted({int(value) for value in re.split(r"[,;\s]+", text)}))
            if any(value <= 0 for value in values):
                raise ValueError()
            return values
        except ValueError:
            raise ValueError(translate(self.language, "stream_mask_labels_invalid")) from None

    def _mask_changed(self, *_args):
        self._update_mode_controls()
        self._computation_changed()

    def config(self):
        return StreamlineConfig(
            directions=tuple(combo.currentData() for combo in self.directions),
            seed_count=self.seed_count.value(), step_mm=self.step.value(),
            max_steps=self.max_steps.value(), max_length_mm=self.length.value(),
            min_speed=self.min_speed.value(), source_scale=100.0 if self.units.currentIndex() == 2 else 1.0,
            component_scales=(tuple(spin.value() / np.pi for spin in self.venc)
                              if self.units.currentIndex() == 3 else (1.0, 1.0, 1.0)),
            seed_min_speed=self.seed_min_speed.value(), seed_distribution=self.distribution.currentData(),
            grid_spacing_mm=self.grid_spacing.value(), grid_units=self.grid_units.currentData(),
            grid_spacing_voxels=self.grid_spacing.value(), rng_seed=self.rng_seed.value(),
            limit_grid_seeds=self.limit_grid.isChecked(),
            integration_direction=self.integration_direction.currentData(), integrator=self.integrator.currentData(),
            min_length_mm=self.min_length.value(), max_turn_degrees=self.max_turn.value(),
            constrain_to_mask=self.constrain_mask.isChecked(),
        )

    def planar_config(self):
        return PlanarConfig(self.mode_2d.currentData() == "planar", self.planar_seeds.value())

    def display_style(self):
        return dict(width=self.width.value(), opacity=self.opacity.value(), speed_max=self.speed_max.value(),
                    min_speed=self.display_min_speed.value(), arrows=self.arrows.isChecked())

    def set_progress(self, done, total, message=""):
        self._progress = (done, total, message)
        key = "stream_ready" if total and done == total else "stream_progress" if total else "stream_idle"
        self.status.setText(message or translate(self.language, key, done=done, total=total))

    def speed_unit(self):
        return translate(self.language, "stream_source_units") if self.units.currentIndex() == 0 else "cm/s"

    def _update_scale_hint(self):
        self.scale_hint.setText(translate(self.language, "stream_scale", value=f"{self.speed_max.value():g}", unit=self.speed_unit()))

    def _update_unit_controls(self):
        native = self.units.currentIndex() == 0
        for widget in (self.min_speed, self.seed_min_speed, self.display_min_speed):
            widget.set_range(0, 1000)
        self.speed_max.set_range(0.01 if native else 1, 1000)
        self.form.setRowVisible(self.labels["stream_venc"], self.units.currentIndex() == 3)
        for key in ("stream_min_speed", "stream_seed_min_speed", "stream_display_min_speed", "stream_speed_max"):
            self.labels[key].setText(translate(self.language, key, unit=self.speed_unit()))

    def _units_changed(self, *_args):
        self._update_unit_controls()
        self.min_speed.set_value(0.1 if self.units.currentIndex() == 0 else 5)
        self.seed_min_speed.set_value(0)
        self.display_min_speed.set_value(0)
        self.speed_max.set_value(1 if self.units.currentIndex() == 0 else 100)
        self._update_scale_hint()
        self._computation_changed()

    def _update_mode_controls(self, *_args):
        planar = self.mode_2d.currentData() == "planar"
        self.planar_seeds.setEnabled(planar)
        self.slab.setEnabled(not planar)
        self.grid_spacing.setEnabled(self.distribution.currentData() == "grid")
        self.grid_units.setEnabled(self.distribution.currentData() == "grid")
        self.limit_grid.setEnabled(self.distribution.currentData() == "grid")
        self.seed_count.setEnabled(self.distribution.currentData() == "random" or self.limit_grid.isChecked())
        self.planar_seeds.setEnabled(planar and (self.distribution.currentData() == "random" or self.limit_grid.isChecked()))
        self.rng_seed.setEnabled(self.distribution.currentData() == "random")
        self.labels["stream_grid_spacing"].setText(translate(
            self.language, "stream_grid_interval_voxel" if self.grid_units.currentData() == "voxel"
            else "stream_grid_interval_mm"))
        has_mask = self.selected_mask() is not None
        self.mask_labels.setEnabled(has_mask)
        self.constrain_mask.setEnabled(has_mask)
        self.mode_hint.setText(translate(self.language, "stream_planar_hint" if planar else "stream_projection_hint"))

    def set_language(self, language):
        self.language = language
        for index, key in enumerate(self._pages):
            self.tabs.setTabText(index, translate(language, key))
        for key, label in self.labels.items():
            label.setText(translate(language, key, unit=self.speed_unit()))
        for widget, key in ((self.enabled, "stream_enabled"), (self.explanation, "stream_explanation"),
                            (self.recalculate, "stream_recalculate"), (self.clear, "stream_clear")):
            widget.setText(translate(language, key))
        self.cache_group.setTitle(translate(language, "stream_cache"))
        for combo, options in self._options.items():
            for index, (_, key) in enumerate(options):
                combo.setItemText(index, translate(language, key))
        for combo in self.directions:
            for index in range(combo.count()):
                combo.setItemText(index, translate(language, "stream_dir_" + combo.itemData(index)))
        for widget, key in self._tips.items():
            widget.setToolTip(translate(language, key))
        self.units.setItemText(0, translate(language, "stream_native"))
        self.units.setItemText(3, translate(language, "stream_phase"))
        self.seed_mask.setItemText(0, translate(language, "stream_no_mask"))
        self.mask_labels.setPlaceholderText(translate(language, "stream_mask_labels_placeholder"))
        self._update_mode_controls()
        self._update_scale_hint()
        self.set_progress(*self._progress)
