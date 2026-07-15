from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSettings, QSize, Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QKeyEvent,
    QKeySequence,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spatiotemporal_labeler.i18n import LANGUAGES, translate
from spatiotemporal_labeler.io import Sequence4D, is_supported_image_path
from spatiotemporal_labeler.model import (
    EditCommand,
    LabelDefinition,
    build_edit_command,
    default_label,
    labels_from_sequence,
    store_labels,
)
from spatiotemporal_labeler.tools import (
    apply_disk,
    apply_label_morphology,
    apply_square,
    automatic_thresholds,
    build_threshold_mask,
    connected_seed_region,
    fill_polygon,
    interpolate_label_frames,
    raster_line,
)

from .icons import tool_icon
from .frame_mapping_dialog import FrameMappingDialog
from .image_strip import ImagePreviewStrip
from .import_dialog import ImportSelectionDialog
from .interpolation_panel import InterpolationPanel
from .label_panel import LabelPanel
from .morphology_panel import MorphologyPanel
from .region_grow_panel import RegionGrowPanel
from .settings_dialog import DEFAULT_SHORTCUTS, SettingsDialog
from .slice_view import SliceView, TemporalView
from .threshold_panel import ThresholdPanel
from .viewer_3d import Mask3DViewer
from .window_level_panel import WindowLevelPanel


PLANE_AXES = {"X-Y": (0, 1, 2), "X-Z": (0, 2, 1), "Y-Z": (1, 2, 0)}
TEMPORAL_AXES = {"X-T": 0, "Y-T": 1, "Z-T": 2}
AXIS_NAMES = ("X", "Y", "Z", "T")
MEDICAL_IMAGE_FILTER = "Medical images (*.nrrd *.nii *.nii.gz)"
MASK_NAME_TOKENS = ("seg", "mask", "label")


@dataclass
class PendingContour:
    mask: Sequence4D
    plane: str
    context: tuple[int, ...]
    frames: tuple[int, ...]
    before: np.ndarray
    focus_frame: int
    points: list[tuple[int, int]]
    label_value: int
    ignore_threshold: bool = False


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings()
        self.language = str(self.settings.value("language", "en"))
        self.shortcuts = {
            key: str(self.settings.value(f"shortcuts/{key}", value))
            for key, value in DEFAULT_SHORTCUTS.items()
        }
        self.resize(1680, 960)
        self.setMinimumSize(1120, 700)
        self.setAcceptDrops(True)
        self.images: list[Sequence4D] = []
        self.masks: list[Sequence4D] = []
        self.cursor = [0, 0, 0, 0]
        self.tool = "brush"
        self.active_label_value = 1
        self._levels = (0.0, 1.0)
        self._image_value_range = (0.0, 1.0)
        self._threshold_cache: np.ndarray | None = None
        self._threshold_signature: tuple[object, ...] | None = None
        self._all_frames_held = False
        self._picker_held = False
        self._threshold_bypass_held = False
        self._stroke_ignore_threshold = False
        self._maximized_plane: str | None = None
        self._mask_labels: dict[int, dict[int, LabelDefinition]] = {}
        self._stroke_before: np.ndarray | None = None
        self._stroke_mask: Sequence4D | None = None
        self._stroke_frame = 0
        self._stroke_frames: tuple[int, ...] = ()
        self._stroke_context: tuple[int, ...] = ()
        self._stroke_tool = "brush"
        self._tool_before_grow = "brush"
        self._contour: list[tuple[int, int]] = []
        self._pending_contour: PendingContour | None = None
        self._undo_stack: list[EditCommand] = []
        self._redo_stack: list[EditCommand] = []
        self._build_ui()
        self._build_actions()
        self._apply_style()
        self._set_language(self.language)
        self._apply_shortcuts()
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)
        self._update_enabled_state()

    def _tr(self, key: str, **values: object) -> str:
        return translate(self.language, key, **values)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(7)

        source_bar = QHBoxLayout()
        self.image_source_label = QLabel()
        source_bar.addWidget(self.image_source_label)
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(230)
        self.image_combo.currentIndexChanged.connect(self._active_image_changed)
        source_bar.addWidget(self.image_combo, 1)
        source_bar.addSpacing(12)
        self.mask_source_label = QLabel()
        source_bar.addWidget(self.mask_source_label)
        self.mask_combo = QComboBox()
        self.mask_combo.setMinimumWidth(230)
        self.mask_combo.currentIndexChanged.connect(self._active_mask_changed)
        source_bar.addWidget(self.mask_combo, 1)
        root_layout.addLayout(source_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.label_panel = LabelPanel()
        self.label_panel.labelSelected.connect(self._label_selected)
        self.label_panel.visibilityChanged.connect(self._label_visibility_changed)
        self.label_panel.addRequested.connect(self._add_label)
        self.label_panel.deleteRequested.connect(self._delete_label)
        self.label_panel.renameRequested.connect(self._rename_label)
        splitter.addWidget(self.label_panel)

        slices = QWidget()
        slices_layout = QHBoxLayout(slices)
        slices_layout.setContentsMargins(0, 0, 0, 0)
        slices_layout.setSpacing(5)
        self.view_grid = QWidget()
        grid = QGridLayout(self.view_grid)
        self.view_grid_layout = grid
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(5)
        self.slice_views = {plane: SliceView(plane) for plane in ("X-Y", "X-Z", "Y-Z")}
        grid.addWidget(self.slice_views["X-Y"], 0, 0)
        grid.addWidget(self.slice_views["X-Z"], 0, 1)
        grid.addWidget(self.slice_views["Y-Z"], 1, 0)

        self.temporal_panel = QWidget()
        self.temporal_panel.setObjectName("temporalPanel")
        temporal_layout = QVBoxLayout(self.temporal_panel)
        temporal_layout.setContentsMargins(0, 0, 0, 0)
        temporal_layout.setSpacing(0)
        temporal_header = QHBoxLayout()
        temporal_header.setContentsMargins(7, 4, 7, 4)
        self.temporal_header_label = QLabel()
        temporal_header.addWidget(self.temporal_header_label)
        temporal_header.addStretch()
        self.temporal_mode = QComboBox()
        self.temporal_mode.addItems(["X-T", "Y-T", "Z-T"])
        self.temporal_mode.currentIndexChanged.connect(self._temporal_mode_changed)
        temporal_header.addWidget(self.temporal_mode)
        temporal_layout.addLayout(temporal_header)
        self.temporal_view = TemporalView()
        self.temporal_view.strokeStarted.connect(self._stroke_started)
        self.temporal_view.strokeMoved.connect(self._stroke_moved)
        self.temporal_view.strokeFinished.connect(self._stroke_finished)
        self.temporal_view.cursorRequested.connect(self._temporal_cursor_requested)
        self.temporal_view.viewDoubleClicked.connect(self._view_double_clicked)
        self.temporal_view.brushSizeStepRequested.connect(self._adjust_brush_size)
        self.temporal_view.hoverMoved.connect(self._view_hovered)
        temporal_layout.addWidget(self.temporal_view, 1)
        grid.addWidget(self.temporal_panel, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        slices_layout.addWidget(self.view_grid, 1)
        self.image_previews = ImagePreviewStrip()
        self.image_previews.imageActivated.connect(self.image_combo.setCurrentIndex)
        self.image_previews.collapsedChanged.connect(self._image_previews_collapsed)
        self.image_previews.hide()
        slices_layout.addWidget(self.image_previews)
        splitter.addWidget(slices)

        render_panel = QWidget()
        render_layout = QVBoxLayout(render_panel)
        render_layout.setContentsMargins(0, 0, 0, 0)
        self.render_title = QLabel()
        self.render_title.setObjectName("viewerTitle")
        render_layout.addWidget(self.render_title)
        self.viewer_3d = Mask3DViewer()
        render_layout.addWidget(self.viewer_3d, 1)
        splitter.addWidget(render_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([205, 980, 485])
        root_layout.addWidget(splitter, 1)

        navigation = QHBoxLayout()
        self.slider_axis_label = QLabel()
        navigation.addWidget(self.slider_axis_label)
        self.slider_axis = QComboBox()
        self.slider_axis.addItem("T Time", 3)
        self.slider_axis.addItem("X", 0)
        self.slider_axis.addItem("Y", 1)
        self.slider_axis.addItem("Z", 2)
        self.slider_axis.currentIndexChanged.connect(self._slider_axis_changed)
        navigation.addWidget(self.slider_axis)
        self.axis_slider = QSlider(Qt.Orientation.Horizontal)
        self.axis_slider.setRange(0, 0)
        self.axis_slider.setTracking(True)
        self.axis_slider.valueChanged.connect(self._slider_value_changed)
        navigation.addWidget(self.axis_slider, 1)
        self.slider_value_label = QLabel("T 0 / 0")
        self.slider_value_label.setMinimumWidth(100)
        navigation.addWidget(self.slider_value_label)
        self.all_frames_toggle = QCheckBox()
        self.all_frames_toggle.toggled.connect(self._all_frames_toggled)
        navigation.addWidget(self.all_frames_toggle)
        root_layout.addLayout(navigation)

        for view in self.slice_views.values():
            view.strokeStarted.connect(self._stroke_started)
            view.strokeMoved.connect(self._stroke_moved)
            view.strokeFinished.connect(self._stroke_finished)
            view.navigateRequested.connect(self._navigate_in_plane)
            view.viewDoubleClicked.connect(self._view_double_clicked)
            view.sliceStepRequested.connect(self._step_slice)
            view.brushSizeStepRequested.connect(self._adjust_brush_size)
            view.hoverMoved.connect(self._view_hovered)
        self.setCentralWidget(root)

        self.threshold_dock = QDockWidget(self)
        self.threshold_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.threshold_panel = ThresholdPanel()
        self.threshold_panel.changed.connect(self._threshold_changed)
        self.threshold_panel.enabled.toggled.connect(self._threshold_enabled_changed)
        self.threshold_panel.methodChanged.connect(self._threshold_method_changed)
        self.threshold_dock.setWidget(self.threshold_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.threshold_dock)
        self.threshold_dock.hide()

        self.grow_dock = QDockWidget(self)
        self.grow_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.grow_panel = RegionGrowPanel()
        self.grow_dock.setWidget(self.grow_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.grow_dock)
        self.grow_dock.hide()

        self.morphology_dock = QDockWidget(self)
        self.morphology_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.morphology_panel = MorphologyPanel()
        self.morphology_panel.applyRequested.connect(self._apply_morphology)
        self.morphology_dock.setWidget(self.morphology_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.morphology_dock)
        self.morphology_dock.hide()

        self.interpolation_dock = QDockWidget(self)
        self.interpolation_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.interpolation_panel = InterpolationPanel()
        self.interpolation_panel.applyRequested.connect(self._apply_interpolation)
        self.interpolation_dock.setWidget(self.interpolation_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.interpolation_dock)
        self.interpolation_dock.hide()

        self.display_dock = QDockWidget(self)
        self.display_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.window_level_panel = WindowLevelPanel()
        self.window_level_panel.changed.connect(self._window_level_changed)
        self.display_dock.setWidget(self.window_level_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.display_dock)
        self.display_dock.hide()
        self.tabifyDockWidget(self.threshold_dock, self.grow_dock)
        self.tabifyDockWidget(self.grow_dock, self.morphology_dock)
        self.tabifyDockWidget(self.morphology_dock, self.interpolation_dock)
        self.tabifyDockWidget(self.interpolation_dock, self.display_dock)

    def _build_actions(self) -> None:
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(self.toolbar)
        self.import_action = QAction(tool_icon("import", "#147b86"), "", self)
        self.import_action.triggered.connect(self.open_import_dialog)
        self.toolbar.addAction(self.import_action)
        self.new_mask_action = QAction(tool_icon("new", "#147b86"), "", self)
        self.new_mask_action.triggered.connect(self.new_mask)
        self.toolbar.addAction(self.new_mask_action)
        self.save_action = QAction(
            tool_icon("save", "#147b86"), "", self
        )
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save_mask)
        self.toolbar.addAction(self.save_action)
        self.save_as_action = QAction(tool_icon("save_as", "#147b86"), "", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.save_mask(save_as=True))
        self.toolbar.addAction(self.save_as_action)
        self.toolbar.addSeparator()

        self.undo_action = QAction(
            tool_icon("undo", "#53666a"), "", self
        )
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)
        self.toolbar.addAction(self.undo_action)
        self.redo_action = QAction(
            tool_icon("redo", "#53666a"), "", self
        )
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.redo_action.triggered.connect(self.redo)
        self.toolbar.addAction(self.redo_action)
        self.toolbar.addSeparator()

        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)
        self.tool_actions: dict[str, QAction] = {}
        for key in ("brush", "eraser", "contour", "picker", "grow"):
            action = QAction("", self, checkable=True)
            icon_color = {
                "brush": "#087f8c",
                "eraser": "#d95252",
                "contour": "#b77a00",
                "picker": "#7259b8",
                "grow": "#25834f",
            }[key]
            action.setIcon(tool_icon(key, icon_color))
            action.triggered.connect(
                lambda checked, name=key: self._tool_action_triggered(name, checked)
            )
            tool_group.addAction(action)
            self.toolbar.addAction(action)
            self.tool_actions[key] = action
        self.tool_actions["brush"].setChecked(True)

        self.toolbar.addSeparator()
        self.threshold_mask_action = QAction(
            tool_icon("threshold", "#168894"), "", self, checkable=True
        )
        self.threshold_mask_action.toggled.connect(self._threshold_action_toggled)
        self.threshold_button = QToolButton()
        self.threshold_button.setDefaultAction(self.threshold_mask_action)
        self.threshold_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolbar.addWidget(self.threshold_button)
        self.morphology_action = self.morphology_dock.toggleViewAction()
        self.morphology_action.setIcon(tool_icon("morphology", "#8a6418"))
        self.morphology_button = QToolButton()
        self.morphology_button.setDefaultAction(self.morphology_action)
        self.morphology_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolbar.addWidget(self.morphology_button)
        self.interpolation_action = self.interpolation_dock.toggleViewAction()
        self.interpolation_action.setIcon(tool_icon("interpolate", "#3f6f9e"))
        self.interpolation_button = QToolButton()
        self.interpolation_button.setDefaultAction(self.interpolation_action)
        self.interpolation_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolbar.addWidget(self.interpolation_button)
        self.display_action = self.display_dock.toggleViewAction()
        self.display_action.setIcon(tool_icon("window", "#53666a"))
        self.display_button = QToolButton()
        self.display_button.setDefaultAction(self.display_action)
        self.display_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolbar.addWidget(self.display_button)
        self.toolbar.addSeparator()
        self.diameter_label = QLabel()
        self.toolbar.addWidget(self.diameter_label)
        self.brush_diameter = QDoubleSpinBox()
        self.brush_diameter.setRange(1.0, 80.0)
        self.brush_diameter.setValue(6.0)
        self.brush_diameter.setSuffix(" mm")
        self.brush_diameter.setSingleStep(1.0)
        self.brush_diameter.valueChanged.connect(self._brush_settings_changed)
        self.toolbar.addWidget(self.brush_diameter)
        self.shape_label = QLabel()
        self.toolbar.addWidget(self.shape_label)
        self.brush_shape = QComboBox()
        self.brush_shape.addItem("", "round")
        self.brush_shape.addItem("", "square")
        self.brush_shape.currentIndexChanged.connect(self._brush_settings_changed)
        self.toolbar.addWidget(self.brush_shape)
        self.file_menu = self.menuBar().addMenu("")
        self.file_menu.addActions(
            [
                self.import_action,
                self.new_mask_action,
                self.save_action,
                self.save_as_action,
            ]
        )
        self.edit_menu = self.menuBar().addMenu("")
        self.edit_menu.addActions([self.undo_action, self.redo_action, *self.tool_actions.values()])
        self.view_menu = self.menuBar().addMenu("")
        self.image_previews_action = QAction("", self, checkable=True)
        self.image_previews_action.setChecked(True)
        self.image_previews_action.toggled.connect(self._toggle_image_previews)
        self.view_menu.addAction(self.image_previews_action)
        self.view_menu.addAction(self.threshold_mask_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.threshold_dock.toggleViewAction())
        self.view_menu.addAction(self.grow_dock.toggleViewAction())
        self.view_menu.addAction(self.morphology_action)
        self.view_menu.addAction(self.interpolation_action)
        self.view_menu.addAction(self.display_action)
        self.language_menu = self.view_menu.addMenu("")
        language_group = QActionGroup(self)
        language_group.setExclusive(True)
        self.language_actions: dict[str, QAction] = {}
        for code, name in LANGUAGES.items():
            action = QAction(name, self, checkable=True)
            action.triggered.connect(
                lambda checked, language=code: self._set_language(language) if checked else None
            )
            language_group.addAction(action)
            self.language_menu.addAction(action)
            self.language_actions[code] = action
        self.settings_action = QAction("", self)
        self.settings_action.triggered.connect(self._open_settings)
        self.edit_menu.addSeparator()
        self.edit_menu.addAction(self.settings_action)

        decrease = QAction(self)
        decrease.setShortcut(QKeySequence("["))
        decrease.triggered.connect(lambda: self.brush_diameter.stepDown())
        self.addAction(decrease)
        increase = QAction(self)
        increase.setShortcut(QKeySequence("]"))
        increase.triggered.connect(lambda: self.brush_diameter.stepUp())
        self.addAction(increase)
        self.cancel_contour_action = QAction(self)
        self.cancel_contour_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        self.cancel_contour_action.triggered.connect(self._cancel_pending_contour)
        self.addAction(self.cancel_contour_action)
        self._brush_settings_changed()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #e8eeee;
                color: #172326;
                font-family: "Noto Sans", "Segoe UI";
                font-size: 10pt;
            }
            QMenuBar, QMenu, QToolBar, QStatusBar { background: #f8fafa; }
            QMenuBar { border-bottom: 1px solid #c7d0d2; }
            QMenuBar::item:selected, QMenu::item:selected { background: #d7eaec; }
            QToolBar { border-bottom: 1px solid #b8c5c8; spacing: 4px; padding: 6px; }
            QToolBar::separator { background: #cbd4d6; width: 1px; margin: 4px 6px; }
            QToolButton { padding: 5px 7px; border: 1px solid transparent; border-radius: 3px; }
            QToolButton:hover { background: #e2ecee; border-color: #aec3c7; }
            QToolButton:pressed { background: #cfdee0; }
            QToolButton:checked { background: #c3e4e6; border-color: #16808a; color: #083f45; }
            QToolButton:disabled { color: #98a4a6; }
            QComboBox, QDoubleSpinBox {
                background: #ffffff; border: 1px solid #a9b9bc; border-radius: 3px;
                min-height: 22px; padding: 3px 7px;
            }
            QComboBox:focus, QDoubleSpinBox:focus { border-color: #168894; }
            QSlider::groove:horizontal { height: 6px; background: #b8c4c6; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #168894; border-radius: 3px; }
            QSlider::handle:horizontal {
                width: 17px; margin: -6px 0; background: #ffffff;
                border: 2px solid #087d89; border-radius: 8px;
            }
            QCheckBox { spacing: 7px; font-weight: 600; }
            QCheckBox::indicator { width: 17px; height: 17px; }
            QLabel#viewerTitle, QLabel#sectionTitle {
                background: #17282c; color: #e8f0f1; padding: 7px 10px; font-weight: 700;
            }
            QWidget#labelPanel { background: #f5f8f8; border: 1px solid #bdc9cb; }
            QListWidget { background: #ffffff; border: 1px solid #c1cdcf; outline: 0; }
            QListWidget::item { min-height: 29px; padding: 3px 5px; }
            QListWidget::item:hover { background: #eef5f5; }
            QListWidget::item:selected { background: #c9e5e7; color: #10282c; }
            QWidget#temporalPanel { background: #d7e0e1; }
            QSplitter::handle { background: #9eafb2; width: 3px; }
            QStatusBar { border-top: 1px solid #c6d0d2; }
            """
        )

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if hasattr(self, "toolbar"):
            style = (
                Qt.ToolButtonStyle.ToolButtonIconOnly
                if event.size().width() < 1300
                else Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            self.toolbar.setToolButtonStyle(style)

    def _set_language(self, language: str) -> None:
        self.language = language if language in LANGUAGES else "en"
        self.settings.setValue("language", self.language)
        if hasattr(self, "language_actions"):
            self.language_actions[self.language].setChecked(True)
        self._retranslate_ui()

    def _retranslate_ui(self) -> None:
        self.setWindowTitle(self._tr("window_title"))
        self.image_source_label.setText(self._tr("display_image"))
        self.mask_source_label.setText(self._tr("edit_mask"))
        self.temporal_header_label.setText(self._tr("temporal_view"))
        self.render_title.setText(self._tr("render_3d"))
        self.slider_axis_label.setText(self._tr("slider_axis"))
        self.slider_axis.setItemText(0, f"T {self._tr('time')}")
        self.toolbar.setWindowTitle(self._tr("toolbar"))
        self.import_action.setText(self._tr("import"))
        self.import_action.setToolTip(self._tr("import_tip"))
        self.new_mask_action.setText(self._tr("new_mask"))
        self.save_action.setText(self._tr("save"))
        self.save_as_action.setText(self._tr("save_as"))
        self.undo_action.setText(self._tr("undo"))
        self.redo_action.setText(self._tr("redo"))
        for name, action in self.tool_actions.items():
            action.setText(self._tr(name))
        self.diameter_label.setText(self._tr("diameter"))
        self.brush_diameter.setToolTip(self._tr("brush_size_tip"))
        self.shape_label.setText(self._tr("shape"))
        self.brush_shape.setItemText(0, self._tr("round_shape"))
        self.brush_shape.setItemText(1, self._tr("square_shape"))
        self.all_frames_toggle.setText(self._tr("all_frames"))
        self.all_frames_toggle.setToolTip(self._tr("all_frames_tip"))
        self.threshold_mask_action.setText(self._tr("threshold_mask"))
        self.settings_action.setText(self._tr("settings"))
        self.image_previews_action.setText(self._tr("image_previews"))
        self.threshold_dock.setWindowTitle(self._tr("threshold_dock"))
        self.grow_dock.setWindowTitle(self._tr("grow_dock"))
        self.morphology_dock.setWindowTitle(self._tr("morphology_dock"))
        self.morphology_action.setText(self._tr("morphology"))
        self.interpolation_dock.setWindowTitle(self._tr("interpolation_dock"))
        self.interpolation_action.setText(self._tr("interpolation"))
        self.display_dock.setWindowTitle(self._tr("display_dock"))
        self.display_action.setText(self._tr("display_dock"))
        self.file_menu.setTitle(self._tr("file"))
        self.edit_menu.setTitle(self._tr("edit"))
        self.view_menu.setTitle(self._tr("view"))
        self.language_menu.setTitle(self._tr("language"))
        self.label_panel.set_language(self.language)
        self.threshold_panel.set_language(self.language)
        self.grow_panel.set_language(self.language)
        self.morphology_panel.set_language(self.language)
        self.interpolation_panel.set_language(self.language)
        self.window_level_panel.set_language(self.language)
        self.image_previews.set_language(self.language)
        for view in [*self.slice_views.values(), self.temporal_view]:
            view.setToolTip(self._tr("locate_tip"))
        if self.active_image is None:
            self.statusBar().showMessage(self._tr("load_start"))

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.language, self.shortcuts, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        language, shortcuts = dialog.values()
        self.shortcuts = shortcuts
        for key, value in shortcuts.items():
            self.settings.setValue(f"shortcuts/{key}", value)
        self._apply_shortcuts()
        self._set_language(language)

    def _apply_shortcuts(self) -> None:
        if not hasattr(self, "tool_actions"):
            return
        for key in ("brush", "eraser", "contour", "grow"):
            self.tool_actions[key].setShortcut(QKeySequence(self.shortcuts.get(key, "")))
        self.tool_actions["picker"].setShortcut(QKeySequence())

    def _event_matches(self, event: QKeyEvent, shortcut_key: str) -> bool:
        sequence = QKeySequence(self.shortcuts.get(shortcut_key, ""))
        return sequence.count() == 1 and event.keyCombination() == sequence[0]

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if event.type() in {
            QEvent.Type.ApplicationDeactivate,
            QEvent.Type.WindowDeactivate,
        }:
            self._all_frames_held = False
            self._threshold_bypass_held = False
            self._set_picker_held(False)
            return super().eventFilter(watched, event)
        if event.type() not in {QEvent.Type.KeyPress, QEvent.Type.KeyRelease}:
            return super().eventFilter(watched, event)
        if not self.isActiveWindow():
            return super().eventFilter(watched, event)
        key_event = event
        if not isinstance(key_event, QKeyEvent) or key_event.isAutoRepeat():
            return super().eventFilter(watched, event)
        editing_text = isinstance(watched, (QLineEdit, QAbstractSpinBox, QComboBox))
        pressed = event.type() == QEvent.Type.KeyPress
        if self._event_matches(key_event, "all_frames_hold"):
            self._all_frames_held = pressed
            return True
        if pressed and self._event_matches(key_event, "picker") and not editing_text:
            self._set_picker_held(pressed)
            return True
        picker_sequence = QKeySequence(self.shortcuts.get("picker", ""))
        if (
            not pressed
            and self._picker_held
            and picker_sequence.count() == 1
            and key_event.key() == picker_sequence[0].key()
        ):
            self._set_picker_held(False)
            return True
        if self._event_matches(key_event, "threshold_bypass") and not editing_text:
            self._threshold_bypass_held = pressed
            return True
        if pressed and not editing_text:
            if self._event_matches(key_event, "previous_time"):
                self._step_time(-1)
                return True
            if self._event_matches(key_event, "next_time"):
                self._step_time(1)
                return True
            if self._event_matches(key_event, "reset_view"):
                self._reset_2d_views()
                return True
        return super().eventFilter(watched, event)

    def _step_time(self, delta: int) -> None:
        image = self.active_image
        if image is None:
            return
        previous = self.cursor[3]
        self.cursor[3] = int(np.clip(previous + delta, 0, image.frame_count - 1))
        if self.cursor[3] == previous:
            return
        self.refresh_views()
        self._refresh_3d()

    def _sync_window_controls(self) -> None:
        low, high = self._levels
        width = max(float(high - low), 1e-9)
        center = float((low + high) * 0.5)
        self.window_level_panel.set_values(center, width)

    def _window_level_changed(self, center: float, width: float) -> None:
        width = max(float(width), 1e-9)
        self._levels = (center - width * 0.5, center + width * 0.5)
        self.refresh_views()

    def _invalidate_threshold(self) -> None:
        self._threshold_cache = None
        self._threshold_signature = None

    def _threshold_changed(self) -> None:
        self._cancel_pending_contour(silent=True)
        self._invalidate_threshold()
        self.refresh_views()

    def _threshold_action_toggled(self, enabled: bool) -> None:
        if self.threshold_panel.enabled.isChecked() != enabled:
            self.threshold_panel.enabled.setChecked(enabled)
        if enabled:
            self.threshold_panel.preview.setChecked(True)
            self.threshold_dock.show()
            self.threshold_dock.raise_()

    def _threshold_enabled_changed(self, enabled: bool) -> None:
        if self.threshold_mask_action.isChecked() != enabled:
            self.threshold_mask_action.blockSignals(True)
            self.threshold_mask_action.setChecked(enabled)
            self.threshold_mask_action.blockSignals(False)
        if enabled:
            self.threshold_panel.preview.setChecked(True)

    def _threshold_method_changed(self, method: str) -> None:
        image = self.active_image
        if image is None:
            return
        if method == "manual":
            self._threshold_changed()
            return
        try:
            lower, upper = automatic_thresholds(
                image.data[..., self.cursor[3]], method
            )
            self.threshold_panel.set_bounds(lower, upper)
        except Exception as error:
            QMessageBox.critical(self, self._tr("threshold_failed"), str(error))

    def _threshold_selection(self) -> np.ndarray | None:
        image = self.active_image
        if image is None or not self.threshold_panel.enabled.isChecked():
            return None
        signature = (
            id(image),
            self.threshold_panel.method_key,
            float(self.threshold_panel.lower.value()),
            float(self.threshold_panel.upper.value()),
            int(self.threshold_panel.radius.value()),
        )
        if self._threshold_cache is None or signature != self._threshold_signature:
            self._threshold_cache = build_threshold_mask(
                image.data,
                self.threshold_panel.method_key,
                float(self.threshold_panel.lower.value()),
                float(self.threshold_panel.upper.value()),
                int(self.threshold_panel.radius.value()),
            )
            self._threshold_signature = signature
        return self._threshold_cache

    def _toggle_image_previews(self, enabled: bool) -> None:
        self.image_previews.set_collapsed(not enabled)
        self._rebuild_image_previews()

    def _image_previews_collapsed(self, collapsed: bool) -> None:
        self.image_previews_action.blockSignals(True)
        self.image_previews_action.setChecked(not collapsed)
        self.image_previews_action.blockSignals(False)
        if not collapsed:
            self.image_previews.update_images(self.images, tuple(self.cursor))

    def _rebuild_image_previews(self) -> None:
        if not hasattr(self, "image_previews_action"):
            return
        if len(self.images) <= 1:
            self.image_previews.hide()
            return
        self.image_previews.rebuild(self.images, self.image_combo.currentIndex())
        self.image_previews.set_collapsed(not self.image_previews_action.isChecked(), emit=False)
        self.image_previews.update_images(self.images, tuple(self.cursor))

    def _view_double_clicked(self, plane: str) -> None:
        if self._pending_contour is not None:
            self._confirm_contour(plane)
            return
        widgets: dict[str, QWidget] = {
            **self.slice_views,
            "X-T": self.temporal_panel,
            "Y-T": self.temporal_panel,
            "Z-T": self.temporal_panel,
        }
        if self._maximized_plane == plane:
            self._maximized_plane = None
            self._restore_view_grid_layout()
            self._rebuild_image_previews()
            return
        self._maximized_plane = plane
        selected = widgets[plane]
        for widget in [*self.slice_views.values(), self.temporal_panel]:
            self.view_grid_layout.removeWidget(widget)
            widget.setVisible(widget is selected)
        self.view_grid_layout.addWidget(selected, 0, 0, 2, 2)
        selected.show()
        self.image_previews.hide()

    def _restore_view_grid_layout(self) -> None:
        placements = (
            (self.slice_views["X-Y"], 0, 0),
            (self.slice_views["X-Z"], 0, 1),
            (self.slice_views["Y-Z"], 1, 0),
            (self.temporal_panel, 1, 1),
        )
        for widget, _row, _column in placements:
            self.view_grid_layout.removeWidget(widget)
        for widget, row, column in placements:
            self.view_grid_layout.addWidget(widget, row, column)
            widget.show()

    def _reset_2d_views(self) -> None:
        if self._maximized_plane in TEMPORAL_AXES:
            self.temporal_view.reset_view()
        elif self._maximized_plane in self.slice_views:
            self.slice_views[self._maximized_plane].reset_view()
        else:
            for view in [*self.slice_views.values(), self.temporal_view]:
                view.reset_view()

    @property
    def active_image(self) -> Sequence4D | None:
        index = self.image_combo.currentIndex()
        return self.images[index] if 0 <= index < len(self.images) else None

    @property
    def selected_mask(self) -> Sequence4D | None:
        index = self.mask_combo.currentIndex()
        return self.masks[index] if 0 <= index < len(self.masks) else None

    @property
    def active_mask(self) -> Sequence4D | None:
        mask = self.selected_mask
        image = self.active_image
        return mask if mask is not None and image is not None and image.compatible_with(mask) else None

    @property
    def active_labels(self) -> dict[int, LabelDefinition]:
        mask = self.active_mask
        return self._labels_for(mask) if mask is not None else {}

    def _labels_for(self, mask: Sequence4D) -> dict[int, LabelDefinition]:
        key = id(mask)
        if key not in self._mask_labels:
            self._mask_labels[key] = labels_from_sequence(mask)
        return self._mask_labels[key]

    def _sync_label_panel(self) -> None:
        mask = self.selected_mask
        definitions = self._labels_for(mask) if mask is not None else {}
        self.label_panel.set_labels(definitions, self.active_label_value)

    def open_import_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self, self._tr("import_title"), "", MEDICAL_IMAGE_FILTER
        )
        paths = [Path(path).expanduser().resolve() for path in selected]
        if paths:
            self._import_paths(paths, prompt=True)

    def _import_paths(self, paths: list[Path], prompt: bool) -> None:
        paths = list(dict.fromkeys(paths))
        inferred = {
            path
            for path in paths
            if any(token in path.name.lower() for token in MASK_NAME_TOKENS)
        }
        if prompt or len(paths) > 1:
            dialog = ImportSelectionDialog(paths, inferred, self.language, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            selections = dialog.selections()
        else:
            selections = [(paths[0], paths[0] in inferred)]
        for path, is_mask in selections:
            self._load_with_feedback(path, is_mask)

    def open_image_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, self._tr("load_image_title"), "", MEDICAL_IMAGE_FILTER
        )
        for path in paths:
            self._load_with_feedback(path, is_mask=False)

    def open_mask_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, self._tr("load_mask_title"), "", MEDICAL_IMAGE_FILTER
        )
        for path in paths:
            self._load_with_feedback(path, is_mask=True)

    def _load_with_feedback(self, path: str | Path, is_mask: bool) -> None:
        try:
            self.load_mask(path) if is_mask else self.load_image(path)
        except Exception as error:
            QMessageBox.critical(self, self._tr("load_failed"), f"{path}\n\n{error}")

    def load_image(self, path: str | Path) -> Sequence4D:
        sequence = Sequence4D.load(path)
        self.images.append(sequence)
        self.image_combo.addItem(sequence.display_name)
        self.image_combo.setCurrentIndex(len(self.images) - 1)
        self._rebuild_image_previews()
        self.statusBar().showMessage(
            self._tr(
                "loaded_image",
                name=sequence.display_name,
                shape=sequence.data.shape,
                spacing=sequence.spacing_xyz,
            )
        )
        return sequence

    def load_mask(
        self,
        path: str | Path,
        frame_mapping: str | int | None = "prompt",
    ) -> Sequence4D | None:
        sequence = Sequence4D.load(path)
        image = self.active_image
        if not np.issubdtype(sequence.data.dtype, np.integer):
            raise ValueError(self._tr("mask_integer", dtype=sequence.data.dtype))
        if (
            image is not None
            and image.frame_count > 1
            and sequence.frame_count == 1
            and sequence.transform.original_ndim == 3
            and image.spatially_compatible_with(sequence)
        ):
            mapping = frame_mapping
            if mapping == "prompt":
                dialog = FrameMappingDialog(image.frame_count, self.language, self)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return None
                mapping = dialog.target_frame()
            if mapping is not None and not isinstance(mapping, int):
                raise ValueError(f"Unknown frame mapping: {mapping}")
            sequence = sequence.map_single_frame_to(image.frame_count, mapping)
        self.masks.append(sequence)
        self._mask_labels[id(sequence)] = labels_from_sequence(sequence)
        self.mask_combo.addItem(sequence.display_name)
        self.mask_combo.setCurrentIndex(len(self.masks) - 1)
        if image is not None and not image.compatible_with(sequence):
            self.statusBar().showMessage(self._tr("mask_mismatch", name=sequence.display_name))
        else:
            self.statusBar().showMessage(self._tr("loaded_mask", name=sequence.display_name))
        return sequence

    def new_mask(self) -> None:
        image = self.active_image
        if image is None:
            QMessageBox.information(
                self, self._tr("new_mask_title"), self._tr("no_image_new_mask")
            )
            return
        mask = Sequence4D.blank_mask_from(image)
        self.masks.append(mask)
        self._mask_labels[id(mask)] = {1: default_label(1)}
        self.mask_combo.addItem(mask.display_name)
        self.mask_combo.setCurrentIndex(len(self.masks) - 1)
        self._update_mask_combo_text()

    def save_mask(self, save_as: bool = False) -> bool:
        mask = self.active_mask
        return self._save_specific_mask(mask, save_as) if mask is not None else False

    def _save_specific_mask(self, mask: Sequence4D, save_as: bool = False) -> bool:
        destination = mask.path
        if save_as or destination is None:
            default_name = "seg.nii.gz" if mask.source_format == "nifti" else "seg.seq.nrrd"
            initial = str(destination or Path.cwd() / default_name)
            selected, _ = QFileDialog.getSaveFileName(
                self,
                self._tr("save_mask_title"),
                initial,
                (
                    "NIfTI image (*.nii.gz *.nii)"
                    if mask.source_format == "nifti"
                    else "NRRD sequence (*.seq.nrrd *.nrrd)"
                ),
            )
            if not selected:
                return False
            destination = Path(selected)
        try:
            store_labels(mask, self._labels_for(mask))
            mask.save(destination)
        except Exception as error:
            QMessageBox.critical(self, self._tr("save_failed"), str(error))
            return False
        self._update_mask_combo_text()
        self.statusBar().showMessage(self._tr("saved", path=destination), 5000)
        return True

    def _active_image_changed(self, _index: int) -> None:
        self._cancel_pending_contour(silent=True)
        image = self.active_image
        if image is None:
            self.refresh_views()
            return
        self.cursor = [max(0, size // 2) for size in image.data.shape]
        sample = image.data.ravel()[:: max(1, image.data.size // 500_000)]
        finite = sample[np.isfinite(sample)]
        if finite.size:
            low, high = np.percentile(finite, (1.0, 99.0))
            self._levels = (float(low), float(high if high != low else low + 1.0))
            data_low, data_high = float(finite.min()), float(finite.max())
            self._image_value_range = (data_low, data_high)
            self.threshold_panel.set_image_range(data_low, data_high)
            self.grow_panel.set_image_range(data_low, data_high)
            self.window_level_panel.set_image_range(data_low, data_high)
            self._sync_window_controls()
            if self.threshold_panel.method_key != "manual":
                self._threshold_method_changed(self.threshold_panel.method_key)
        self._invalidate_threshold()
        if self.active_mask is None:
            match = next(
                (index for index, mask in enumerate(self.masks) if image.compatible_with(mask)),
                None,
            )
            if match is not None:
                self.mask_combo.setCurrentIndex(match)
        self._slider_axis_changed()
        self._rebuild_image_previews()
        self._sync_label_panel()
        self.refresh_views(update_3d=True)
        self._update_enabled_state()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        if any(
            url.isLocalFile()
            and (Path(url.toLocalFile()).is_dir() or is_supported_image_path(url.toLocalFile()))
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API
        paths: list[Path] = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile()).expanduser().resolve()
            if path.is_dir():
                paths.extend(
                    sorted(
                        (
                            candidate
                            for candidate in path.iterdir()
                            if candidate.is_file() and is_supported_image_path(candidate)
                        ),
                        key=lambda candidate: candidate.name.lower(),
                    )
                )
            elif is_supported_image_path(path):
                paths.append(path)
        paths = list(dict.fromkeys(paths))
        if not paths:
            event.ignore()
            return
        self._import_paths(paths, prompt=len(paths) > 1)
        event.acceptProposedAction()

    def _active_mask_changed(self, _index: int) -> None:
        self._cancel_pending_contour(silent=True)
        self._sync_label_panel()
        self._sync_interpolation_panel()
        self.refresh_views(update_3d=True)
        self._update_enabled_state()

    def _sync_interpolation_panel(self) -> None:
        mask = self.active_mask
        self.interpolation_panel.set_frame_count(
            mask.frame_count if mask is not None else 1,
            self.cursor[3],
        )

    def _label_selected(self, value: int) -> None:
        self.active_label_value = value
        self._update_enabled_state()

    def _label_visibility_changed(self, value: int, visible: bool) -> None:
        definition = self.active_labels.get(value)
        if definition is None:
            return
        definition.visible = visible
        self.refresh_views()
        self._refresh_3d()

    def _add_label(self) -> None:
        mask = self.active_mask
        if mask is None:
            return
        definitions = self._labels_for(mask)
        suggested = max(definitions, default=0) + 1
        maximum = min(int(np.iinfo(mask.data.dtype).max), 2_147_483_647)
        value, accepted = QInputDialog.getInt(
            self, self._tr("add_label"), self._tr("label_id"), suggested, 1, maximum
        )
        if not accepted:
            return
        if value in definitions:
            self.active_label_value = value
            self._sync_label_panel()
            return
        default_name = self._tr("label_default", value=value)
        name, accepted = QInputDialog.getText(
            self, self._tr("add_label"), self._tr("label_name"), text=default_name
        )
        if not accepted:
            return
        definition = default_label(value)
        definition.name = name.strip() or default_name
        definitions[value] = definition
        mask.dirty = True
        self.active_label_value = value
        self._sync_label_panel()
        self._update_mask_combo_text()
        self._update_enabled_state()

    def _delete_label(self, value: int) -> None:
        mask = self.active_mask
        if mask is None or value not in self.active_labels:
            return
        choice = QMessageBox.question(
            self,
            self._tr("delete_label_title"),
            self._tr("delete_label_confirm", value=value),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        indices = np.flatnonzero(mask.data == value).astype(np.intp, copy=False)
        if indices.size:
            before = mask.data.flat[indices].copy()
            after = np.zeros(indices.size, dtype=mask.data.dtype)
            mask.data.flat[indices] = after
            self._undo_stack.append(EditCommand(mask, indices, before, after, self.cursor[3]))
            self._undo_stack = self._undo_stack[-30:]
            self._redo_stack.clear()
        self.active_labels.pop(value, None)
        mask.dirty = True
        self.active_label_value = next(iter(self.active_labels), 1)
        self._sync_label_panel()
        self._update_mask_combo_text()
        self.refresh_views(update_3d=True)
        self._update_enabled_state()

    def _rename_label(self, value: int) -> None:
        mask = self.active_mask
        definition = self.active_labels.get(value)
        if mask is None or definition is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            self._tr("rename_label"),
            self._tr("label_name"),
            text=definition.name,
        )
        if accepted and name.strip():
            definition.name = name.strip()
            mask.dirty = True
            self._sync_label_panel()
            self._update_mask_combo_text()

    def _extract_spatial(self, sequence: Sequence4D, plane: str) -> np.ndarray:
        return self._extract_spatial_data(sequence.data, plane)

    def _extract_spatial_data(self, data: np.ndarray, plane: str) -> np.ndarray:
        x, y, z, t = self.cursor
        if plane == "X-Y":
            return data[:, :, z, t]
        if plane == "X-Z":
            return data[:, y, :, t]
        return data[x, :, :, t]

    def refresh_views(self, update_3d: bool = False) -> None:
        image = self.active_image
        if image is None:
            self.viewer_3d.set_mask(None)
            self.viewer_3d.set_cursor(None)
            return
        self._clip_cursor()
        mask = self.active_mask
        labels = self.active_labels
        threshold = self._threshold_selection()
        show_threshold = threshold is not None and self.threshold_panel.preview.isChecked()
        for plane, view in self.slice_views.items():
            h_axis, v_axis, fixed_axis = PLANE_AXES[plane]
            view.set_slice(
                self._extract_spatial(image, plane),
                self._extract_spatial(mask, plane) if mask is not None else None,
                (image.spacing_xyz[h_axis], image.spacing_xyz[v_axis]),
                self._levels,
                AXIS_NAMES[fixed_axis],
                self.cursor[fixed_axis],
                labels,
                (self.cursor[h_axis], self.cursor[v_axis]),
                self._extract_spatial_data(threshold, plane) if show_threshold else None,
            )
        mode = self.temporal_mode.currentText()
        x, y, z, t = self.cursor
        if mode == "X-T":
            image_temporal = image.data[:, y, z, :]
            mask_temporal = mask.data[:, y, z, :] if mask is not None else None
            threshold_temporal = threshold[:, y, z, :] if show_threshold else None
            spacing, fixed = image.spacing_xyz[0], f"Y {y + 1}, Z {z + 1}"
        elif mode == "Y-T":
            image_temporal = image.data[x, :, z, :]
            mask_temporal = mask.data[x, :, z, :] if mask is not None else None
            threshold_temporal = threshold[x, :, z, :] if show_threshold else None
            spacing, fixed = image.spacing_xyz[1], f"X {x + 1}, Z {z + 1}"
        else:
            image_temporal = image.data[x, y, :, :]
            mask_temporal = mask.data[x, y, :, :] if mask is not None else None
            threshold_temporal = threshold[x, y, :, :] if show_threshold else None
            spacing, fixed = image.spacing_xyz[2], f"X {x + 1}, Y {y + 1}"
        self.temporal_view.set_sequence_slice(
            image_temporal,
            mask_temporal,
            mode,
            spacing,
            self._levels,
            t,
            fixed,
            labels,
            (self.cursor[TEMPORAL_AXES[mode]], t),
            threshold_temporal,
        )
        self._sync_slider()
        self.viewer_3d.set_cursor(
            tuple(self.cursor[:3]),
            spacing=image.spacing_xyz,
            origin=image.transform.origin_ras,
            direction=image.transform.direction_ras,
        )
        if update_3d:
            self._refresh_3d(immediate=True)
        if self.image_previews.isVisible() and self._maximized_plane is None:
            self.image_previews.update_images(self.images, tuple(self.cursor))

    def _refresh_3d(self, immediate: bool = False) -> None:
        mask = self.active_mask
        if mask is None:
            self.viewer_3d.set_mask(None)
            return
        self.viewer_3d.set_mask(
            mask.data[:, :, :, self.cursor[3]],
            labels=self.active_labels,
            spacing=mask.spacing_xyz,
            origin=mask.transform.origin_ras,
            direction=mask.transform.direction_ras,
            immediate=immediate,
        )

    def _stroke_started(
        self, plane: str, h: int, v: int, temporary_erase: bool = False
    ) -> None:
        mask = self.active_mask
        effective_tool = "eraser" if temporary_erase else self.tool
        if self._picker_held and not temporary_erase:
            self._pick_label(plane, h, v, show_background=False)
            return
        if not temporary_erase and effective_tool == "picker":
            self._pick_label(plane, h, v)
            return
        if not temporary_erase and effective_tool == "grow":
            self._grow_from_seed(plane, h, v)
            return
        if (
            mask is None
            or not self.active_labels
            or not self._valid_plane_point(plane, h, v)
            or (effective_tool == "contour" and self._pending_contour is not None)
        ):
            return
        if temporary_erase and self._pending_contour is not None:
            self._cancel_pending_contour(silent=True)
        self._stroke_mask = mask
        self._stroke_tool = effective_tool
        self._stroke_ignore_threshold = self._threshold_bypass_held
        self._stroke_frame = self.cursor[3]
        self._stroke_context = self._plane_context(plane)
        if plane in TEMPORAL_AXES:
            self._stroke_frames = tuple(range(mask.frame_count))
        else:
            self._stroke_frames = (
                tuple(range(mask.frame_count))
                if self.all_frames_toggle.isChecked() or self._all_frames_held
                else (self._stroke_frame,)
            )
        self._stroke_before = mask.data[..., list(self._stroke_frames)].copy()
        self._contour = [(h, v)]
        if self._stroke_tool == "contour":
            self._view_for_plane(plane).set_contour(self._contour)
        else:
            self._paint_to(plane, h, v)

    def _stroke_moved(
        self, plane: str, h: int, v: int, _temporary_erase: bool = False
    ) -> None:
        if self._stroke_before is None or not self._valid_plane_point(plane, h, v):
            return
        if self._stroke_tool == "contour":
            if not self._contour or self._contour[-1] != (h, v):
                self._contour.extend(raster_line(self._contour[-1], (h, v))[1:])
                self._view_for_plane(plane).set_contour(self._contour)
        else:
            self._paint_to(plane, h, v)

    def _stroke_finished(
        self, plane: str, h: int, v: int, _temporary_erase: bool = False
    ) -> None:
        mask, before = self._stroke_mask, self._stroke_before
        if mask is None or before is None:
            return
        if self._stroke_tool == "contour":
            if self._valid_plane_point(plane, h, v) and self._contour[-1] != (h, v):
                self._contour.extend(raster_line(self._contour[-1], (h, v))[1:])
            self._pending_contour = PendingContour(
                mask,
                plane,
                self._stroke_context,
                self._stroke_frames,
                before,
                self._stroke_frame,
                list(self._contour),
                self.active_label_value,
                self._stroke_ignore_threshold,
            )
            self._clear_stroke(keep_contour=True)
            self.statusBar().showMessage(self._tr("contour_pending"), 6000)
            return
        if self._valid_plane_point(plane, h, v):
            self._paint_to(plane, h, v)
        self._commit_stroke(mask, before)

    def _paint_to(self, plane: str, h: int, v: int) -> None:
        mask, image = self._stroke_mask, self.active_image
        if mask is None or image is None:
            return
        value = 0 if self._stroke_tool == "eraser" else self.active_label_value
        start_h, start_v = self._contour[-1] if self._contour else (h, v)
        points = raster_line((start_h, start_v), (h, v))
        spacing = self._editing_spacing(plane, image)
        if plane in TEMPORAL_AXES:
            planes = [self._mutable_plane(mask, plane, None, self._stroke_context)]
        else:
            planes = [
                self._mutable_plane(mask, plane, frame, self._stroke_context)
                for frame in self._stroke_frames
            ]
        threshold = None if self._stroke_ignore_threshold else self._threshold_selection()
        if threshold is None:
            allowed_planes: list[np.ndarray | None] = [None] * len(planes)
        elif plane in TEMPORAL_AXES:
            allowed_planes = [
                self._array_plane(threshold, plane, None, self._stroke_context)
            ]
        else:
            allowed_planes = [
                self._array_plane(threshold, plane, frame, self._stroke_context)
                for frame in self._stroke_frames
            ]
        for plane_data, allowed in zip(planes, allowed_planes):
            original = plane_data.copy() if allowed is not None else None
            for point in points:
                operation = apply_disk if self.brush_shape.currentData() == "round" else apply_square
                operation(
                    plane_data,
                    point,
                    self.brush_diameter.value() / 2.0,
                    spacing,
                    value,
                )
            if allowed is not None and original is not None:
                plane_data[~allowed] = original[~allowed]
        if not self._contour or self._contour[-1] != (h, v):
            self._contour.append((h, v))
        self._refresh_stroke_overlays()

    def _refresh_stroke_overlays(self) -> None:
        mask = self._stroke_mask
        if mask is None:
            return
        labels = self._labels_for(mask)
        visible_planes = (*PLANE_AXES, self.temporal_mode.currentText())
        for plane in visible_planes:
            frame = None if plane in TEMPORAL_AXES else self._stroke_frame
            mask_plane = self._array_plane(
                mask.data, plane, frame, self._plane_context(plane)
            )
            self._view_for_plane(plane).set_mask_overlay(mask_plane, labels)

    def _confirm_contour(self, plane: str) -> None:
        pending = self._pending_contour
        if pending is None:
            return
        if pending.plane != plane:
            self.statusBar().showMessage(self._tr("contour_wrong_view"), 4000)
            return
        value = pending.label_value
        if plane in TEMPORAL_AXES:
            planes = [self._mutable_plane(pending.mask, plane, None, pending.context)]
        else:
            planes = [
                self._mutable_plane(pending.mask, plane, frame, pending.context)
                for frame in pending.frames
            ]
        threshold = None if pending.ignore_threshold else self._threshold_selection()
        if threshold is None:
            allowed_planes: list[np.ndarray | None] = [None] * len(planes)
        elif plane in TEMPORAL_AXES:
            allowed_planes = [self._array_plane(threshold, plane, None, pending.context)]
        else:
            allowed_planes = [
                self._array_plane(threshold, plane, frame, pending.context)
                for frame in pending.frames
            ]
        for plane_data, allowed in zip(planes, allowed_planes):
            original = plane_data.copy() if allowed is not None else None
            fill_polygon(plane_data, pending.points, value)
            if allowed is not None and original is not None:
                plane_data[~allowed] = original[~allowed]
        command = build_edit_command(
            pending.mask, pending.frames, pending.before, pending.focus_frame
        )
        if command is not None:
            pending.mask.dirty = True
            self._undo_stack.append(command)
            self._undo_stack = self._undo_stack[-30:]
            self._redo_stack.clear()
        self._view_for_plane(plane).set_contour([])
        self._pending_contour = None
        self._update_mask_combo_text()
        self.refresh_views()
        self._refresh_3d()
        self._update_enabled_state()

    def _commit_stroke(self, mask: Sequence4D, before: np.ndarray) -> None:
        command = build_edit_command(mask, self._stroke_frames, before, self._stroke_frame)
        if command is not None:
            mask.dirty = True
            self._undo_stack.append(command)
            self._undo_stack = self._undo_stack[-30:]
            self._redo_stack.clear()
        self._clear_stroke()
        self._update_mask_combo_text()
        self.refresh_views()
        self._refresh_3d()
        self._update_enabled_state()

    def _clear_stroke(self, keep_contour: bool = False) -> None:
        self._stroke_before = None
        self._stroke_mask = None
        self._stroke_frames = ()
        self._stroke_context = ()
        self._stroke_tool = self.tool
        self._stroke_ignore_threshold = False
        if not keep_contour:
            self._contour = []

    def _cancel_pending_contour(self, silent: bool = False) -> None:
        if self._pending_contour is None:
            return
        self._view_for_plane(self._pending_contour.plane).set_contour([])
        self._pending_contour = None
        self._contour = []
        if not silent:
            self.statusBar().showMessage(self._tr("contour_cancelled"), 3000)

    def _plane_context(self, plane: str) -> tuple[int, ...]:
        x, y, z = self.cursor[:3]
        return {
            "X-Y": (z,),
            "X-Z": (y,),
            "Y-Z": (x,),
            "X-T": (y, z),
            "Y-T": (x, z),
            "Z-T": (x, y),
        }[plane]

    def _mutable_plane(
        self,
        mask: Sequence4D,
        plane: str,
        frame: int | None,
        context: tuple[int, ...],
    ) -> np.ndarray:
        return self._array_plane(mask.data, plane, frame, context)

    def _array_plane(
        self,
        data: np.ndarray,
        plane: str,
        frame: int | None,
        context: tuple[int, ...],
    ) -> np.ndarray:
        if plane == "X-Y":
            return data[:, :, context[0], frame]
        if plane == "X-Z":
            return data[:, context[0], :, frame]
        if plane == "Y-Z":
            return data[context[0], :, :, frame]
        if plane == "X-T":
            return data[:, context[0], context[1], :]
        if plane == "Y-T":
            return data[context[0], :, context[1], :]
        return data[context[0], context[1], :, :]

    def _editing_spacing(
        self, plane: str, image: Sequence4D
    ) -> tuple[float, float]:
        if plane in PLANE_AXES:
            h_axis, v_axis, _ = PLANE_AXES[plane]
            return image.spacing_xyz[h_axis], image.spacing_xyz[v_axis]
        spacing = image.spacing_xyz[TEMPORAL_AXES[plane]]
        return spacing, spacing

    def _view_for_plane(self, plane: str) -> SliceView | TemporalView:
        return self.temporal_view if plane in TEMPORAL_AXES else self.slice_views[plane]

    def _valid_plane_point(self, plane: str, h: int, v: int) -> bool:
        image = self.active_image
        if image is None:
            return False
        if plane in PLANE_AXES:
            h_axis, v_axis, _ = PLANE_AXES[plane]
            return 0 <= h < image.data.shape[h_axis] and 0 <= v < image.data.shape[v_axis]
        axis = TEMPORAL_AXES[plane]
        return 0 <= h < image.data.shape[axis] and 0 <= v < image.frame_count

    def _voxel_for_plane(self, plane: str, h: int, v: int) -> tuple[int, int, int, int]:
        x, y, z, t = self.cursor
        return {
            "X-Y": (h, v, z, t),
            "X-Z": (h, y, v, t),
            "Y-Z": (x, h, v, t),
            "X-T": (h, y, z, v),
            "Y-T": (x, h, z, v),
            "Z-T": (x, y, h, v),
        }[plane]

    def _view_hovered(self, plane: str, h: int, v: int, visible: bool) -> None:
        if self._picker_held and visible:
            self._pick_label(plane, h, v, show_background=False)

    def _pick_label(
        self, plane: str, h: int, v: int, show_background: bool = True
    ) -> None:
        mask = self.active_mask
        if mask is None or not self._valid_plane_point(plane, h, v):
            return
        value = int(mask.data[self._voxel_for_plane(plane, h, v)])
        if value <= 0:
            if show_background:
                self.statusBar().showMessage(self._tr("picked_background"), 2500)
            return
        if value == self.active_label_value:
            return
        if value not in self.active_labels:
            self.active_labels[value] = default_label(value)
            self._sync_label_panel()
        self.active_label_value = value
        self.label_panel.select_label(value)

    def _grow_from_seed(self, plane: str, h: int, v: int) -> None:
        image, mask = self.active_image, self.active_mask
        if image is None or mask is None or not self._valid_plane_point(plane, h, v):
            return
        if plane in TEMPORAL_AXES:
            self.statusBar().showMessage(self._tr("grow_temporal"), 3000)
            return
        voxel = self._voxel_for_plane(plane, h, v)
        x, y, z, frame = voxel
        threshold = self._threshold_selection()
        if threshold is not None and not bool(threshold[voxel]):
            self.statusBar().showMessage(self._tr("grow_outside_mask"), 3000)
            return
        before = mask.data[..., [frame]].copy()
        tolerance = float(self.grow_panel.tolerance.value())
        seed_value = float(image.data[voxel])
        if not np.isfinite(seed_value):
            return
        scope = str(self.grow_panel.scope.currentData())
        if scope == "3d":
            intensities = np.asarray(image.data[..., frame], dtype=np.float64)
            labels = mask.data[..., frame]
            candidate = np.isfinite(intensities) & (np.abs(intensities - seed_value) <= tolerance)
            candidate &= (labels == 0) | (labels == self.active_label_value)
            if threshold is not None:
                candidate &= threshold[..., frame]
            region = connected_seed_region(candidate, (x, y, z))
            labels[region] = self.active_label_value
        else:
            context = self._plane_context(plane)
            intensities = np.asarray(
                self._array_plane(image.data, plane, frame, context), dtype=np.float64
            )
            labels = self._mutable_plane(mask, plane, frame, context)
            candidate = np.isfinite(intensities) & (np.abs(intensities - seed_value) <= tolerance)
            candidate &= (labels == 0) | (labels == self.active_label_value)
            if threshold is not None:
                candidate &= self._array_plane(threshold, plane, frame, context)
            region = connected_seed_region(candidate, (h, v))
            labels[region] = self.active_label_value
        command = build_edit_command(mask, (frame,), before, frame)
        if command is not None:
            mask.dirty = True
            self._undo_stack.append(command)
            self._undo_stack = self._undo_stack[-30:]
            self._redo_stack.clear()
        self._update_mask_combo_text()
        self.refresh_views()
        self._refresh_3d()
        self._update_enabled_state()

    def _apply_morphology(self) -> None:
        mask = self.active_mask
        if mask is None or not self.active_labels:
            return
        self._cancel_pending_contour(silent=True)
        frames = (
            tuple(range(mask.frame_count))
            if self.morphology_panel.frames_scope.currentData() == "all"
            else (self.cursor[3],)
        )
        label_values = (
            tuple(sorted(self.active_labels))
            if self.morphology_panel.labels_scope.currentData() == "all"
            else (self.active_label_value,)
        )
        before = mask.data[..., list(frames)].copy()
        operation = str(self.morphology_panel.operation.currentData())
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for frame in frames:
                mask.data[..., frame] = apply_label_morphology(
                    mask.data[..., frame],
                    operation,
                    label_values,
                    spacing_xyz=mask.spacing_xyz,
                    minimum_volume_mm3=self.morphology_panel.minimum_volume.value(),
                    connectivity=int(self.morphology_panel.connectivity.currentData()),
                    radius_mm=self.morphology_panel.radius.value(),
                )
        except Exception as error:
            mask.data[..., list(frames)] = before
            QMessageBox.critical(self, self._tr("morphology_failed"), str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()

        command = build_edit_command(mask, frames, before, self.cursor[3])
        if command is None:
            self.statusBar().showMessage(self._tr("morphology_no_changes"), 3500)
            return
        mask.dirty = True
        self._undo_stack.append(command)
        self._undo_stack = self._undo_stack[-30:]
        self._redo_stack.clear()
        self._update_mask_combo_text()
        self.refresh_views(update_3d=True)
        self._update_enabled_state()
        self.statusBar().showMessage(
            self._tr("morphology_applied", count=command.flat_indices.size), 5000
        )

    def _apply_interpolation(self) -> None:
        mask = self.active_mask
        if mask is None or not self.active_labels:
            return
        self._cancel_pending_contour(silent=True)
        start = self.interpolation_panel.start_frame.value() - 1
        end = self.interpolation_panel.end_frame.value() - 1
        frames = tuple(range(start + 1, end))
        if not frames:
            QMessageBox.information(
                self,
                self._tr("interpolation_dock"),
                self._tr("interpolation_frame_error"),
            )
            return
        label_values = (
            tuple(sorted(self.active_labels))
            if self.interpolation_panel.labels_scope.currentData() == "all"
            else (self.active_label_value,)
        )
        before = mask.data[..., list(frames)].copy()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            mask.data[..., start + 1 : end] = interpolate_label_frames(
                mask.data,
                start,
                end,
                label_values,
                spacing_xyz=mask.spacing_xyz,
            )
        except Exception as error:
            mask.data[..., list(frames)] = before
            QMessageBox.critical(self, self._tr("interpolation_failed"), str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()

        command = build_edit_command(mask, frames, before, self.cursor[3])
        if command is None:
            self.statusBar().showMessage(self._tr("interpolation_no_changes"), 3500)
            return
        mask.dirty = True
        self._undo_stack.append(command)
        self._undo_stack = self._undo_stack[-30:]
        self._redo_stack.clear()
        self._update_mask_combo_text()
        self.refresh_views(update_3d=True)
        self._update_enabled_state()
        self.statusBar().showMessage(
            self._tr("interpolation_applied", count=command.flat_indices.size), 5000
        )

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._cancel_pending_contour(silent=True)
        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)
        self._show_command_mask(command)

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._cancel_pending_contour(silent=True)
        command = self._redo_stack.pop()
        command.redo()
        self._undo_stack.append(command)
        self._show_command_mask(command)

    def _show_command_mask(self, command: EditCommand) -> None:
        if self.active_image is None or not self.active_image.compatible_with(command.mask):
            matching_image = next(
                (
                    index
                    for index, image in enumerate(self.images)
                    if image.compatible_with(command.mask)
                ),
                None,
            )
            if matching_image is not None:
                self.image_combo.setCurrentIndex(matching_image)
        mask_index = next(
            index for index, mask in enumerate(self.masks) if mask is command.mask
        )
        self.mask_combo.setCurrentIndex(mask_index)
        self.cursor[3] = command.focus_frame
        for value in np.unique(command.mask.data.flat[command.flat_indices]):
            numeric = int(value)
            if numeric > 0 and numeric not in self._labels_for(command.mask):
                self._labels_for(command.mask)[numeric] = default_label(numeric)
        self._sync_label_panel()
        self._update_mask_combo_text()
        self.refresh_views(update_3d=True)
        self._update_enabled_state()

    def _navigate_in_plane(self, plane: str, h: int, v: int) -> None:
        if not self._valid_plane_point(plane, h, v):
            return
        h_axis, v_axis, _ = PLANE_AXES[plane]
        self.cursor[h_axis], self.cursor[v_axis] = h, v
        self.refresh_views()

    def _step_slice(self, plane: str, delta: int) -> None:
        image = self.active_image
        if image is None:
            return
        fixed_axis = PLANE_AXES[plane][2]
        self.cursor[fixed_axis] = int(
            np.clip(self.cursor[fixed_axis] + delta, 0, image.data.shape[fixed_axis] - 1)
        )
        self.refresh_views()

    def _temporal_cursor_requested(self, mode: str, axis_index: int, time_index: int) -> None:
        image = self.active_image
        if image is None:
            return
        axis = TEMPORAL_AXES[mode]
        self.cursor[axis] = int(np.clip(axis_index, 0, image.data.shape[axis] - 1))
        self.cursor[3] = int(np.clip(time_index, 0, image.frame_count - 1))
        self.refresh_views()
        self._refresh_3d()

    def _temporal_mode_changed(self, _index: int) -> None:
        if self._pending_contour is not None and self._pending_contour.plane in TEMPORAL_AXES:
            self._cancel_pending_contour(silent=True)
        if self._maximized_plane in TEMPORAL_AXES:
            self._maximized_plane = self.temporal_mode.currentText()
        self.refresh_views()

    def _slider_axis_changed(self, _index: int = 0) -> None:
        image = self.active_image
        axis = int(self.slider_axis.currentData())
        maximum = image.data.shape[axis] - 1 if image is not None else 0
        self.axis_slider.blockSignals(True)
        self.axis_slider.setRange(0, maximum)
        self.axis_slider.setValue(self.cursor[axis] if image is not None else 0)
        self.axis_slider.blockSignals(False)
        self._sync_slider()

    def _slider_value_changed(self, value: int) -> None:
        axis = int(self.slider_axis.currentData())
        old_time = self.cursor[3]
        self.cursor[axis] = value
        self.refresh_views()
        if axis == 3 and value != old_time:
            self._refresh_3d(immediate=False)

    def _sync_slider(self) -> None:
        image = self.active_image
        axis = int(self.slider_axis.currentData())
        if image is None:
            self.slider_value_label.setText(f"{AXIS_NAMES[axis]} 0 / 0")
            return
        self.axis_slider.blockSignals(True)
        self.axis_slider.setValue(self.cursor[axis])
        self.axis_slider.blockSignals(False)
        self.slider_value_label.setText(
            f"{AXIS_NAMES[axis]} {self.cursor[axis] + 1} / {image.data.shape[axis]}"
        )

    def _clip_cursor(self) -> None:
        image = self.active_image
        if image is not None:
            for axis, size in enumerate(image.data.shape):
                self.cursor[axis] = int(np.clip(self.cursor[axis], 0, size - 1))

    def _tool_action_triggered(self, name: str, checked: bool) -> None:
        if name == "grow" and self.tool == "grow":
            fallback = self._tool_before_grow
            if fallback not in self.tool_actions or fallback == "grow":
                fallback = "brush"
            self.tool_actions[fallback].setChecked(True)
            self._set_tool(fallback)
            return
        if checked:
            self._set_tool(name)

    def _set_tool(self, name: str) -> None:
        previous = self.tool
        if self.tool != name:
            self._cancel_pending_contour(silent=True)
        if name == "grow" and previous != "grow":
            self._tool_before_grow = previous
        self.tool = name
        if name == "grow":
            self.grow_dock.show()
            self.grow_dock.raise_()
        elif previous == "grow":
            self.grow_dock.hide()
        self._brush_settings_changed()
        self.statusBar().showMessage(self._tr(f"tool_{name}"), 3500)

    def _set_picker_held(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._picker_held == enabled:
            return
        self._picker_held = enabled
        self._brush_settings_changed()

    def _brush_settings_changed(self, _value: object = None) -> None:
        if not hasattr(self, "brush_shape"):
            return
        shape = str(self.brush_shape.currentData())
        diameter = float(self.brush_diameter.value())
        effective_tool = "picker" if self._picker_held else self.tool
        for view in [*self.slice_views.values(), self.temporal_view]:
            view.set_editing_footprint(diameter, shape, effective_tool)

    def _adjust_brush_size(self, steps: int) -> None:
        self.brush_diameter.setValue(
            float(
                np.clip(
                    self.brush_diameter.value() + steps * self.brush_diameter.singleStep(),
                    self.brush_diameter.minimum(),
                    self.brush_diameter.maximum(),
                )
            )
        )

    def _all_frames_toggled(self, enabled: bool) -> None:
        self.statusBar().showMessage(
            self._tr("all_frames_on" if enabled else "all_frames_off"), 4000
        )

    def _update_mask_combo_text(self) -> None:
        for index, mask in enumerate(self.masks):
            self.mask_combo.setItemText(index, f"{mask.display_name}{' *' if mask.dirty else ''}")

    def _update_enabled_state(self) -> None:
        has_image = self.active_image is not None
        has_mask = self.active_mask is not None
        has_label = bool(self.active_labels)
        self.new_mask_action.setEnabled(has_image)
        self.save_action.setEnabled(has_mask)
        self.save_as_action.setEnabled(has_mask)
        self.undo_action.setEnabled(bool(self._undo_stack))
        self.redo_action.setEnabled(bool(self._redo_stack))
        self.label_panel.setEnabled(has_mask)
        self.all_frames_toggle.setEnabled(has_mask)
        self.threshold_panel.setEnabled(has_image)
        self.threshold_mask_action.setEnabled(has_image)
        self.grow_panel.setEnabled(has_image and has_mask)
        self.morphology_panel.setEnabled(has_mask and has_label)
        self.morphology_action.setEnabled(has_mask and has_label)
        self.interpolation_panel.setEnabled(has_mask and has_label and self.active_mask.frame_count >= 3)
        self.interpolation_action.setEnabled(has_mask and has_label and self.active_mask.frame_count >= 3)
        self.window_level_panel.setEnabled(has_image)
        self.display_action.setEnabled(has_image)
        for action in self.tool_actions.values():
            action.setEnabled(has_mask and has_label)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        dirty = [mask for mask in self.masks if mask.dirty]
        if not dirty:
            event.accept()
            return
        choice = QMessageBox.warning(
            self,
            self._tr("unsaved_title"),
            self._tr("unsaved_message", count=len(dirty)),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if choice == QMessageBox.StandardButton.Save:
            for mask in dirty:
                if not self._save_specific_mask(mask):
                    event.ignore()
                    return
        event.accept()
