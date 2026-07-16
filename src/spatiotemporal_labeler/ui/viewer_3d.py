from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent, QResizeEvent, QShowEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget
from skimage.draw import polygon as rasterize_polygon
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - registers the OpenGL render backend
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import VTK_INT, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import (
    vtkDecimatePro,
    vtkFlyingEdges3D,
    vtkPolyDataNormals,
    vtkWindowedSincPolyDataFilter,
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkFiltersGeneral import vtkContourTriangulator
from vtkmodules.vtkImagingCore import vtkImageConstantPad, vtkImageThreshold
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActor2D,
    vtkLightKit,
    vtkPolyDataMapper,
    vtkPolyDataMapper2D,
    vtkRenderer,
)

from spatiotemporal_labeler.model import LabelDefinition, default_label

from .render_settings import DETAIL_REDUCTION, RenderSettings


STYLE_PRESETS = {
    "clinical": {
        "ambient": 0.18,
        "diffuse": 0.78,
        "specular": 0.34,
        "specular_power": 36.0,
        "key_to_fill": 2.8,
        "key_to_head": 3.5,
        "background": (0.018, 0.026, 0.029),
        "background_2": (0.065, 0.078, 0.082),
    },
    "matte": {
        "ambient": 0.28,
        "diffuse": 0.74,
        "specular": 0.08,
        "specular_power": 12.0,
        "key_to_fill": 1.5,
        "key_to_head": 2.0,
        "background": (0.045, 0.048, 0.052),
        "background_2": (0.090, 0.095, 0.100),
    },
    "glossy": {
        "ambient": 0.12,
        "diffuse": 0.72,
        "specular": 0.72,
        "specular_power": 68.0,
        "key_to_fill": 4.2,
        "key_to_head": 5.0,
        "background": (0.012, 0.017, 0.024),
        "background_2": (0.045, 0.058, 0.075),
    },
}


@dataclass
class SegmentPipeline:
    threshold: Any
    pad: Any
    surface: Any
    smoother: Any
    decimator: Any
    normals: Any
    mapper: Any
    actor: Any


class Mask3DViewer(QWidget):
    """Smoothed multilabel rendering with a persistent spatial-cursor marker."""

    lassoStarted = Signal()
    lassoFinished = Signal(object)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        self.vtk_widget.installEventFilter(self)
        self._lasso_enabled = False
        self._lasso_active = False
        self._alt_rotate_active = False
        self._lasso_points: list[QPointF] = []

        render_window = self.vtk_widget.GetRenderWindow()
        render_window.SetMultiSamples(8)
        render_window.SetNumberOfLayers(2)
        self.interaction_style = vtkInteractorStyleTrackballCamera()
        self.interaction_style.UseTimersOff()
        render_window.GetInteractor().SetInteractorStyle(self.interaction_style)

        self.renderer = vtkRenderer()
        self.renderer.SetLayer(0)
        self.renderer.SetBackground(0.025, 0.040, 0.044)
        self.renderer.SetBackground2(0.075, 0.100, 0.105)
        self.renderer.GradientBackgroundOn()
        # VTK FXAA can blank a renderer under forwarded X11/OpenGL. MSAA on the
        # render window provides compatible antialiasing on Linux and Windows.
        self.renderer.UseFXAAOff()
        render_window.AddRenderer(self.renderer)

        self.cursor_renderer = vtkRenderer()
        self.cursor_renderer.SetLayer(1)
        self.cursor_renderer.SetPreserveColorBuffer(True)
        self.cursor_renderer.SetPreserveDepthBuffer(False)
        self.cursor_renderer.SetInteractive(False)
        self.cursor_renderer.SetActiveCamera(self.renderer.GetActiveCamera())
        render_window.AddRenderer(self.cursor_renderer)
        self.interaction_style.AddObserver(
            "InteractionEvent", self._reset_camera_clipping_range
        )

        self.light_kit = vtkLightKit()
        self.light_kit.SetKeyLightIntensity(0.82)
        self.light_kit.SetKeyToFillRatio(2.8)
        self.light_kit.SetKeyToHeadRatio(3.5)
        self.light_kit.AddLightsToRenderer(self.renderer)

        self.image = vtkImageData()
        self.image.SetDimensions(1, 1, 1)
        self.image.AllocateScalars(VTK_INT, 1)
        self.image.GetPointData().GetScalars().SetTuple1(0, 0)

        self.segment_pipelines: dict[int, SegmentPipeline] = {}
        self.render_settings = RenderSettings()
        self._apply_scene_settings()

        self.cursor_source = vtkSphereSource()
        self.cursor_source.SetThetaResolution(28)
        self.cursor_source.SetPhiResolution(20)
        self.cursor_mapper = vtkPolyDataMapper()
        self.cursor_mapper.SetInputConnection(self.cursor_source.GetOutputPort())
        self.cursor_actor = vtkActor()
        self.cursor_actor.SetMapper(self.cursor_mapper)
        self.cursor_actor.SetVisibility(False)
        self.cursor_actor.GetProperty().SetColor(1.0, 0.78, 0.18)
        self.cursor_actor.GetProperty().LightingOff()
        self.cursor_renderer.AddActor(self.cursor_actor)
        self._build_lasso_actors()

        axes = vtkAxesActor()
        axes.SetXAxisLabelText("R")
        axes.SetYAxisLabelText("A")
        axes.SetZAxisLabelText("H")
        axes.SetShaftTypeToCylinder()
        axes.SetNormalizedShaftLength(0.72, 0.72, 0.72)
        axes.SetNormalizedTipLength(0.28, 0.28, 0.28)
        axes.SetCylinderRadius(0.035)
        axes.SetConeRadius(0.18)
        for caption in (
            axes.GetXAxisCaptionActor2D(),
            axes.GetYAxisCaptionActor2D(),
            axes.GetZAxisCaptionActor2D(),
        ):
            caption.GetCaptionTextProperty().SetColor(0.90, 0.95, 0.95)
            caption.GetCaptionTextProperty().BoldOn()
            caption.GetCaptionTextProperty().SetFontSize(18)
        self.orientation = vtkOrientationMarkerWidget()
        self.orientation.SetOrientationMarker(axes)
        self.orientation.SetInteractor(render_window.GetInteractor())
        self.orientation.SetViewport(0.0, 0.0, 0.20, 0.20)

        self._pending: tuple[
            np.ndarray,
            tuple[float, ...],
            tuple[float, ...],
            np.ndarray,
            tuple[LabelDefinition, ...],
            float,
        ] | None = None
        self._has_camera = False
        self._initialized = False
        self._cursor_state: tuple[float, float, float, float] | None = None
        self._last_cursor_render_time = 0.0
        self._cursor_throttle_ms = 50
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.timeout.connect(self._render_cursor)
        self._last_apply_time = 0.0
        self._last_apply_duration_ms = 0.0
        self._throttle_ms = 75
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self._throttle_ms)
        self._timer.timeout.connect(self._apply_pending)

    def _build_lasso_actors(self) -> None:
        color = (1.0, 0.78, 0.34)
        self._lasso_fill_data = vtkPolyData()
        self._lasso_fill_triangulator = vtkContourTriangulator()
        self._lasso_fill_triangulator.SetInputData(self._lasso_fill_data)
        fill_mapper = vtkPolyDataMapper2D()
        fill_mapper.SetInputConnection(self._lasso_fill_triangulator.GetOutputPort())
        self._lasso_fill_actor = vtkActor2D()
        self._lasso_fill_actor.SetMapper(fill_mapper)
        self._lasso_fill_actor.GetProperty().SetColor(*color)
        self._lasso_fill_actor.GetProperty().SetOpacity(0.20)

        self._lasso_path_data = vtkPolyData()
        path_mapper = vtkPolyDataMapper2D()
        path_mapper.SetInputData(self._lasso_path_data)
        self._lasso_path_actor = vtkActor2D()
        self._lasso_path_actor.SetMapper(path_mapper)
        self._lasso_path_actor.GetProperty().SetColor(*color)
        self._lasso_path_actor.GetProperty().SetLineWidth(2.0)

        self._lasso_closure_data = vtkPolyData()
        closure_mapper = vtkPolyDataMapper2D()
        closure_mapper.SetInputData(self._lasso_closure_data)
        self._lasso_closure_actor = vtkActor2D()
        self._lasso_closure_actor.SetMapper(closure_mapper)
        self._lasso_closure_actor.GetProperty().SetColor(*color)
        self._lasso_closure_actor.GetProperty().SetLineWidth(1.6)

        for actor in (
            self._lasso_fill_actor,
            self._lasso_path_actor,
            self._lasso_closure_actor,
        ):
            actor.SetVisibility(False)
            self.cursor_renderer.AddViewProp(actor)

    @staticmethod
    def _set_polyline_data(
        data: vtkPolyData,
        points: list[tuple[float, float]],
        *,
        closed: bool = False,
    ) -> None:
        vtk_points = vtkPoints()
        for x, y in points:
            vtk_points.InsertNextPoint(float(x), float(y), 0.0)
        lines = vtkCellArray()
        if len(points) >= 2:
            indices = list(range(len(points)))
            if closed:
                indices.append(0)
            lines.InsertNextCell(len(indices))
            for index in indices:
                lines.InsertCellPoint(index)
        data.SetPoints(vtk_points)
        data.SetLines(lines)
        data.Modified()

    @staticmethod
    def _set_dashed_line_data(
        data: vtkPolyData,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        start_array = np.asarray(start, dtype=float)
        delta = np.asarray(end, dtype=float) - start_array
        length = float(np.linalg.norm(delta))
        vtk_points = vtkPoints()
        lines = vtkCellArray()
        if length > 0.0:
            direction = delta / length
            offset = 0.0
            while offset < length:
                segment_end = min(offset + 7.0, length)
                first = start_array + direction * offset
                last = start_array + direction * segment_end
                first_index = vtk_points.InsertNextPoint(*first, 0.0)
                last_index = vtk_points.InsertNextPoint(*last, 0.0)
                lines.InsertNextCell(2)
                lines.InsertCellPoint(first_index)
                lines.InsertCellPoint(last_index)
                offset += 12.0
        data.SetPoints(vtk_points)
        data.SetLines(lines)
        data.Modified()

    def set_lasso_points(self, points: list[QPointF]) -> None:
        self._lasso_points = list(points)
        render_width, render_height = self.vtk_widget.GetRenderWindow().GetSize()
        widget_width, widget_height = self.vtk_widget.width(), self.vtk_widget.height()
        if min(render_width, render_height, widget_width, widget_height) <= 0:
            return
        scale_x = render_width / float(widget_width)
        scale_y = render_height / float(widget_height)
        display_points = [
            (point.x() * scale_x, render_height - 1.0 - point.y() * scale_y)
            for point in points
        ]
        self._set_polyline_data(self._lasso_path_data, display_points)
        self._lasso_path_actor.SetVisibility(bool(display_points))
        self._set_polyline_data(
            self._lasso_fill_data, display_points, closed=True
        )
        self._lasso_fill_actor.SetVisibility(len(display_points) >= 3)
        if len(display_points) >= 2:
            self._set_dashed_line_data(
                self._lasso_closure_data,
                display_points[-1],
                display_points[0],
            )
            self._lasso_closure_actor.SetVisibility(True)
        else:
            self._lasso_closure_actor.SetVisibility(False)
        if self._initialized:
            self.vtk_widget.GetRenderWindow().Render()

    def set_lasso_enabled(self, enabled: bool) -> None:
        self._lasso_enabled = bool(enabled)
        self.vtk_widget.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        if not enabled:
            self.clear_lasso()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._lasso_points:
            QTimer.singleShot(0, lambda: self.set_lasso_points(self._lasso_points))

    def clear_lasso(self) -> None:
        was_visible = any(
            actor.GetVisibility()
            for actor in (
                self._lasso_fill_actor,
                self._lasso_path_actor,
                self._lasso_closure_actor,
            )
        )
        self._lasso_active = False
        self._lasso_points = []
        for actor in (
            self._lasso_fill_actor,
            self._lasso_path_actor,
            self._lasso_closure_actor,
        ):
            actor.SetVisibility(False)
        if self._initialized and was_visible:
            self.vtk_widget.GetRenderWindow().Render()

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if watched is not self.vtk_widget:
            return super().eventFilter(watched, event)
        if not isinstance(event, QMouseEvent):
            return super().eventFilter(watched, event)
        left_pressed = bool(event.buttons() & Qt.MouseButton.LeftButton)
        left_event = event.button() == Qt.MouseButton.LeftButton
        if self._alt_rotate_active:
            if event.type() == QEvent.Type.MouseButtonRelease and left_event:
                self._alt_rotate_active = False
            return super().eventFilter(watched, event)
        if (
            event.type() == QEvent.Type.MouseButtonPress
            and left_event
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self._alt_rotate_active = True
            return super().eventFilter(watched, event)
        if (
            event.type() == QEvent.Type.MouseButtonPress
            and left_event
            and self._lasso_enabled
        ):
            self._lasso_active = True
            self._lasso_points = [event.position()]
            self.set_lasso_points(self._lasso_points)
            self.lassoStarted.emit()
            return True
        if event.type() == QEvent.Type.MouseMove and self._lasso_active:
            point = event.position()
            if not self._lasso_points or (
                point - self._lasso_points[-1]
            ).manhattanLength() >= 1.0:
                self._lasso_points.append(point)
                self.set_lasso_points(self._lasso_points)
            return True
        if (
            event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and self._lasso_active
        ):
            point = event.position()
            if not self._lasso_points or point != self._lasso_points[-1]:
                self._lasso_points.append(point)
            self._lasso_active = False
            self.set_lasso_points(self._lasso_points)
            self.lassoFinished.emit(
                [(float(item.x()), float(item.y())) for item in self._lasso_points]
            )
            self.clear_lasso()
            return True
        if left_event or left_pressed:
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def lasso_voxel_mask(
        self,
        shape: tuple[int, int, int],
        points: list[tuple[float, float]],
        *,
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float],
        direction: np.ndarray,
    ) -> np.ndarray:
        """Project voxel centers and select those inside a screen-space lasso."""
        result = np.zeros(shape, dtype=bool)
        if len(points) < 3 or any(size <= 0 for size in shape):
            return result
        render_width, render_height = self.vtk_widget.GetRenderWindow().GetSize()
        widget_width, widget_height = self.vtk_widget.width(), self.vtk_widget.height()
        if min(render_width, render_height, widget_width, widget_height) <= 0:
            return result

        scale_x = render_width / float(widget_width)
        scale_y = render_height / float(widget_height)
        polygon_x = np.asarray([point[0] * scale_x for point in points], dtype=float)
        polygon_y = np.asarray([point[1] * scale_y for point in points], dtype=float)
        screen_selection = np.zeros((render_height, render_width), dtype=bool)
        rows, columns = rasterize_polygon(
            polygon_y, polygon_x, shape=screen_selection.shape
        )
        screen_selection[rows, columns] = True

        self.renderer.ComputeAspect()
        aspect_pair = self.renderer.GetAspect()
        aspect = float(aspect_pair[0]) / max(float(aspect_pair[1]), 1e-12)
        vtk_matrix = self.renderer.GetActiveCamera().GetCompositeProjectionTransformMatrix(
            aspect, 0.0, 1.0
        )
        projection = np.asarray(
            [
                [vtk_matrix.GetElement(row, column) for column in range(4)]
                for row in range(4)
            ],
            dtype=float,
        )
        spacing_array = np.asarray(spacing, dtype=float)
        origin_array = np.asarray(origin, dtype=float)
        direction_array = np.asarray(direction, dtype=float)
        viewport = self.renderer.GetViewport()
        viewport_x0, viewport_y0, viewport_x1, viewport_y1 = map(float, viewport)
        grid_x, grid_y = np.meshgrid(
            np.arange(shape[0], dtype=float),
            np.arange(shape[1], dtype=float),
            indexing="ij",
        )
        for z_index in range(shape[2]):
            voxels = np.vstack(
                (
                    grid_x.ravel(),
                    grid_y.ravel(),
                    np.full(grid_x.size, float(z_index)),
                )
            )
            world = origin_array[:, None] + direction_array @ (
                voxels * spacing_array[:, None]
            )
            homogeneous = np.vstack((world, np.ones(world.shape[1], dtype=float)))
            clip = projection @ homogeneous
            w_component = clip[3]
            valid = (
                np.isfinite(w_component)
                & (w_component > 1e-12)
                & np.all(np.isfinite(clip[:2]), axis=0)
            )
            normalized = np.zeros((2, clip.shape[1]), dtype=float)
            normalized[:, valid] = clip[:2, valid] / w_component[valid]
            display_x = (
                viewport_x0
                + (normalized[0] + 1.0) * 0.5 * (viewport_x1 - viewport_x0)
            ) * render_width
            display_y_bottom = (
                viewport_y0
                + (normalized[1] + 1.0) * 0.5 * (viewport_y1 - viewport_y0)
            ) * render_height
            display_y = render_height - 1.0 - display_y_bottom
            columns = np.rint(display_x).astype(np.int64)
            rows = np.rint(display_y).astype(np.int64)
            valid &= (
                (columns >= 0)
                & (columns < render_width)
                & (rows >= 0)
                & (rows < render_height)
            )
            selected = np.zeros(valid.shape, dtype=bool)
            selected[valid] = screen_selection[rows[valid], columns[valid]]
            result[:, :, z_index] = selected.reshape(shape[:2])
        return result

    def set_render_settings(
        self, settings: RenderSettings, render: bool = True
    ) -> None:
        settings = RenderSettings.normalized(settings.as_dict())
        geometry_changed = (
            settings.smoothing != self.render_settings.smoothing
            or settings.detail != self.render_settings.detail
        )
        self.render_settings = settings
        self._apply_scene_settings()
        for pipeline in self.segment_pipelines.values():
            self._configure_pipeline(pipeline)
            if geometry_changed:
                pipeline.smoother.Modified()
                pipeline.decimator.Modified()
        if self._initialized and render:
            self.vtk_widget.GetRenderWindow().Render()
            self._last_cursor_render_time = monotonic()

    def _apply_scene_settings(self) -> None:
        preset = STYLE_PRESETS[self.render_settings.style]
        self.renderer.SetBackground(*preset["background"])
        self.renderer.SetBackground2(*preset["background_2"])
        self.light_kit.SetKeyLightIntensity(
            0.82 * float(self.render_settings.lighting) / 100.0
        )
        self.light_kit.SetKeyToFillRatio(preset["key_to_fill"])
        self.light_kit.SetKeyToHeadRatio(preset["key_to_head"])
        self.light_kit.Update()

    def _configure_pipeline(self, pipeline: SegmentPipeline) -> None:
        pipeline.smoother.SetNumberOfIterations(max(1, self.render_settings.smoothing))
        pipeline.decimator.SetInputConnection(
            pipeline.smoother.GetOutputPort()
            if self.render_settings.smoothing > 0
            else pipeline.surface.GetOutputPort()
        )
        pipeline.decimator.SetTargetReduction(
            DETAIL_REDUCTION[self.render_settings.detail]
        )
        preset = STYLE_PRESETS[self.render_settings.style]
        material = pipeline.actor.GetProperty()
        material.SetInterpolationToPhong()
        material.SetAmbient(preset["ambient"])
        material.SetDiffuse(preset["diffuse"])
        material.SetSpecular(preset["specular"])
        material.SetSpecularPower(preset["specular_power"])

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        if not self._initialized:
            self.vtk_widget.Initialize()
            self.orientation.SetEnabled(1)
            self.orientation.InteractiveOff()
            self._initialized = True
        QTimer.singleShot(0, self._render_visible_scene)

    def _render_visible_scene(self) -> None:
        visible = [
            pipeline
            for pipeline in self.segment_pipelines.values()
            if pipeline.actor.GetVisibility()
        ]
        if visible:
            for pipeline in visible:
                pipeline.normals.Update()
            self._frame_camera()
            self._has_camera = True
        self.vtk_widget.GetRenderWindow().Render()
        self._cursor_timer.stop()
        self._last_cursor_render_time = monotonic()

    def _frame_camera(self) -> None:
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Zoom(1.30)
        self._reset_camera_clipping_range()

    def _reset_camera_clipping_range(self, *_args: Any) -> None:
        bounds = np.asarray(self.renderer.ComputeVisiblePropBounds(), dtype=float)
        if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
            return
        lower = bounds[[0, 2, 4]]
        upper = bounds[[1, 3, 5]]
        if np.any(upper < lower):
            return
        diagonal = float(np.linalg.norm(upper - lower))
        padding = max(diagonal * 0.05, 1e-3)
        expanded = (
            bounds
            + np.asarray((-padding, padding, -padding, padding, -padding, padding))
        )
        self.renderer.ResetCameraClippingRange(tuple(map(float, expanded)))

    def _schedule_cursor_render(self) -> None:
        if self._cursor_timer.isActive():
            return
        elapsed_ms = (monotonic() - self._last_cursor_render_time) * 1000.0
        delay = 0 if self._last_cursor_render_time == 0.0 else max(
            0, int(np.ceil(self._cursor_throttle_ms - elapsed_ms))
        )
        self._cursor_timer.start(delay)

    def _render_cursor(self) -> None:
        if not self._initialized:
            return
        self.vtk_widget.GetRenderWindow().Render()
        self._last_cursor_render_time = monotonic()

    def set_cursor(
        self,
        cursor: tuple[int, int, int] | None,
        spacing: tuple[float, float, float] | None = None,
        origin: tuple[float, float, float] | None = None,
        direction: np.ndarray | None = None,
    ) -> None:
        changed = False
        if cursor is None:
            changed = self._cursor_state is not None or bool(self.cursor_actor.GetVisibility())
            self._cursor_state = None
            self.cursor_actor.SetVisibility(False)
        else:
            spacing_array = np.asarray(spacing or (1.0, 1.0, 1.0), dtype=float)
            origin_array = np.asarray(origin or (0.0, 0.0, 0.0), dtype=float)
            direction_array = np.asarray(direction if direction is not None else np.eye(3))
            world = origin_array + direction_array @ (
                np.asarray(cursor, dtype=float) * spacing_array
            )
            radius = max(0.9, float(np.max(spacing_array)) * 1.15)
            state = (*map(float, world), radius)
            changed = self._cursor_state is None or not np.allclose(
                state, self._cursor_state, atol=1e-6
            )
            self._cursor_state = state
            if changed:
                self.cursor_source.SetCenter(*world)
                self.cursor_source.SetRadius(radius)
                self.cursor_source.Modified()
            self.cursor_actor.SetVisibility(True)
        if self._initialized and changed:
            self._schedule_cursor_render()

    def set_mask(
        self,
        frame: np.ndarray | None,
        spacing: tuple[float, float, float] | None = None,
        origin: tuple[float, float, float] | None = None,
        direction: np.ndarray | None = None,
        immediate: bool = False,
        labels: dict[int, LabelDefinition] | None = None,
        global_opacity: float = 1.0,
    ) -> None:
        if frame is None:
            self._timer.stop()
            self._pending = None
            for pipeline in self.segment_pipelines.values():
                pipeline.actor.SetVisibility(False)
            if self._initialized:
                self.vtk_widget.GetRenderWindow().Render()
                self._cursor_timer.stop()
                self._last_cursor_render_time = monotonic()
            return
        if labels is None:
            labels = {
                int(value): default_label(int(value))
                for value in np.unique(frame)
                if int(value) > 0
            }
        definitions = tuple(
            definition
            for definition in labels.values()
            if definition.value > 0 and definition.visible
        )
        visible_values = {definition.value for definition in definitions}
        for value, pipeline in self.segment_pipelines.items():
            pipeline.actor.SetVisibility(value in visible_values)
        self._pending = (
            np.asarray(frame),
            tuple(spacing or (1.0, 1.0, 1.0)),
            tuple(origin or (0.0, 0.0, 0.0)),
            np.asarray(direction if direction is not None else np.eye(3), dtype=float),
            definitions,
            float(np.clip(global_opacity, 0.0, 1.0)),
        )
        if immediate:
            self._timer.stop()
            self._apply_pending()
            return

        elapsed_ms = (monotonic() - self._last_apply_time) * 1000.0
        if not self._timer.isActive():
            cooldown_ms = max(
                float(self._throttle_ms), self._last_apply_duration_ms * 0.75
            )
            delay = 0 if self._last_apply_time == 0.0 else max(
                1, int(np.ceil(cooldown_ms - elapsed_ms))
            )
            self._timer.start(delay)

    def _apply_pending(self) -> None:
        if self._pending is None:
            return
        started = monotonic()
        frame, spacing, origin, direction, definitions, global_opacity = self._pending
        self._pending = None
        dimensions_changed = tuple(self.image.GetDimensions()) != tuple(frame.shape)
        self.image.SetDimensions(*frame.shape)
        self.image.SetSpacing(*spacing)
        self.image.SetOrigin(*origin)
        matrix = self.image.GetDirectionMatrix()
        for row in range(3):
            for column in range(3):
                matrix.SetElement(row, column, float(direction[row, column]))
        frame_int = np.asarray(frame, dtype=np.int32)
        scalars = numpy_to_vtk(frame_int.ravel(order="F"), deep=True, array_type=VTK_INT)
        self.image.GetPointData().SetScalars(scalars)
        self.image.Modified()

        visible_values = {definition.value for definition in definitions}
        for definition in definitions:
            pipeline = self._segment_pipeline(definition)
            red, green, blue = (channel / 255.0 for channel in definition.color)
            pipeline.actor.GetProperty().SetColor(red, green, blue)
            pipeline.actor.GetProperty().SetOpacity(
                float(np.clip(definition.opacity * global_opacity, 0.0, 1.0))
            )
            pipeline.actor.SetVisibility(True)
            pipeline.threshold.Modified()
            pipeline.pad.SetOutputWholeExtent(
                -1,
                frame.shape[0],
                -1,
                frame.shape[1],
                -1,
                frame.shape[2],
            )
            pipeline.pad.Modified()
        for value, pipeline in self.segment_pipelines.items():
            if value not in visible_values:
                pipeline.actor.SetVisibility(False)
        if dimensions_changed or not self._has_camera:
            for definition in definitions:
                self.segment_pipelines[definition.value].normals.Update()
            self._frame_camera()
            self._has_camera = True
        else:
            self._reset_camera_clipping_range()
        if self._initialized:
            self.vtk_widget.GetRenderWindow().Render()
            self._cursor_timer.stop()
            self._last_cursor_render_time = monotonic()
        completed = monotonic()
        self._last_apply_duration_ms = (completed - started) * 1000.0
        self._last_apply_time = completed

    def set_label_opacities(
        self,
        labels: dict[int, LabelDefinition],
        global_opacity: float,
    ) -> None:
        global_opacity = float(np.clip(global_opacity, 0.0, 1.0))
        for value, pipeline in self.segment_pipelines.items():
            definition = labels.get(value)
            if definition is None:
                continue
            pipeline.actor.GetProperty().SetOpacity(
                float(np.clip(definition.opacity * global_opacity, 0.0, 1.0))
            )
        if self._initialized:
            self.vtk_widget.GetRenderWindow().Render()

    def _segment_pipeline(self, definition: LabelDefinition) -> SegmentPipeline:
        existing = self.segment_pipelines.get(definition.value)
        if existing is not None:
            return existing
        threshold = vtkImageThreshold()
        threshold.SetInputData(self.image)
        threshold.ThresholdBetween(definition.value, definition.value)
        threshold.SetInValue(1)
        threshold.SetOutValue(0)
        threshold.SetOutputScalarTypeToUnsignedChar()

        pad = vtkImageConstantPad()
        pad.SetInputConnection(threshold.GetOutputPort())
        pad.SetConstant(0)
        dimensions = self.image.GetDimensions()
        pad.SetOutputWholeExtent(
            -1, dimensions[0], -1, dimensions[1], -1, dimensions[2]
        )

        surface = vtkFlyingEdges3D()
        surface.SetInputConnection(pad.GetOutputPort())
        surface.SetValue(0, 0.5)
        surface.ComputeNormalsOff()
        surface.ComputeGradientsOff()

        smoother = vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(surface.GetOutputPort())
        smoother.SetNumberOfIterations(self.render_settings.smoothing)
        smoother.SetPassBand(0.10)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()

        decimator = vtkDecimatePro()
        decimator.SetInputConnection(smoother.GetOutputPort())
        decimator.SetTargetReduction(DETAIL_REDUCTION[self.render_settings.detail])
        decimator.PreserveTopologyOn()
        decimator.SplittingOff()

        normals = vtkPolyDataNormals()
        normals.SetInputConnection(decimator.GetOutputPort())
        normals.SetFeatureAngle(80)
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.SetVisibility(False)
        red, green, blue = (channel / 255.0 for channel in definition.color)
        actor.GetProperty().SetColor(red, green, blue)
        actor.GetProperty().EdgeVisibilityOff()
        self.renderer.AddActor(actor)
        pipeline = SegmentPipeline(
            threshold, pad, surface, smoother, decimator, normals, mapper, actor
        )
        self._configure_pipeline(pipeline)
        self.segment_pipelines[definition.value] = pipeline
        return pipeline
