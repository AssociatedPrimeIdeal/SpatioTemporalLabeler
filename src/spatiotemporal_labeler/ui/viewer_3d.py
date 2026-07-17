from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

import numpy as np
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRunnable,
    QThreadPool,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent, QMouseEvent, QResizeEvent, QShowEvent
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


@dataclass(frozen=True)
class SurfaceState:
    frame: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: np.ndarray
    cache_key: object


@dataclass(frozen=True)
class SurfaceRequest:
    generation: int
    state: SurfaceState
    values: frozenset[int]
    settings: RenderSettings


def _label_surface(
    frame: np.ndarray,
    value: int,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    direction: np.ndarray,
    settings: RenderSettings,
) -> vtkPolyData:
    selected = np.asarray(frame) == int(value)
    occupied = [
        np.flatnonzero(np.any(selected, axis=tuple(other for other in range(3) if other != axis)))
        for axis in range(3)
    ]
    if any(not indices.size for indices in occupied):
        return vtkPolyData()
    lower = np.asarray([indices[0] for indices in occupied], dtype=np.intp)
    upper = np.asarray([indices[-1] + 1 for indices in occupied], dtype=np.intp)
    crop_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    binary = np.asarray(selected[crop_slices], dtype=np.uint8)

    image = vtkImageData()
    image.SetDimensions(*binary.shape)
    image.SetSpacing(*spacing)
    spacing_array = np.asarray(spacing, dtype=float)
    direction_array = np.asarray(direction, dtype=float)
    crop_origin = np.asarray(origin, dtype=float) + direction_array @ (lower * spacing_array)
    image.SetOrigin(*crop_origin)
    matrix = image.GetDirectionMatrix()
    for row in range(3):
        for column in range(3):
            matrix.SetElement(row, column, float(direction_array[row, column]))
    scalars = numpy_to_vtk(binary.ravel(order="F"), deep=True)
    image.GetPointData().SetScalars(scalars)

    pad = vtkImageConstantPad()
    pad.SetInputData(image)
    pad.SetConstant(0)
    pad.SetOutputWholeExtent(
        -1,
        binary.shape[0],
        -1,
        binary.shape[1],
        -1,
        binary.shape[2],
    )
    surface = vtkFlyingEdges3D()
    surface.SetInputConnection(pad.GetOutputPort())
    surface.SetValue(0, 0.5)
    surface.ComputeNormalsOff()
    surface.ComputeGradientsOff()

    geometry_output = surface.GetOutputPort()
    if settings.smoothing > 0:
        smoother = vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(geometry_output)
        smoother.SetNumberOfIterations(settings.smoothing)
        smoother.SetPassBand(0.10)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        geometry_output = smoother.GetOutputPort()

    decimator = vtkDecimatePro()
    decimator.SetInputConnection(geometry_output)
    decimator.SetTargetReduction(DETAIL_REDUCTION[settings.detail])
    decimator.PreserveTopologyOn()
    decimator.SplittingOff()
    normals = vtkPolyDataNormals()
    normals.SetInputConnection(decimator.GetOutputPort())
    normals.SetFeatureAngle(80)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    output = vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    return output


def _build_surface_meshes(
    state: SurfaceState,
    values: frozenset[int],
    settings: RenderSettings,
) -> dict[int, vtkPolyData]:
    return {
        value: _label_surface(
            state.frame,
            value,
            state.spacing,
            state.origin,
            state.direction,
            settings,
        )
        for value in sorted(values)
    }


class SurfaceTaskSignals(QObject):
    finished = Signal(int, object, float)
    failed = Signal(int, str)


class SurfaceTask(QRunnable):
    def __init__(self, request: SurfaceRequest) -> None:
        super().__init__()
        self.request = request
        self.signals = SurfaceTaskSignals()

    @Slot()
    def run(self) -> None:
        started = monotonic()
        try:
            meshes = _build_surface_meshes(
                self.request.state,
                self.request.values,
                self.request.settings,
            )
        except Exception as error:
            self.signals.failed.emit(self.request.generation, str(error))
            return
        self.signals.finished.emit(
            self.request.generation,
            meshes,
            (monotonic() - started) * 1000.0,
        )


class Mask3DViewer(QWidget):
    """Smoothed multilabel rendering with a persistent spatial-cursor marker."""

    lassoStarted = Signal()
    lassoFinished = Signal(object)
    renderFailed = Signal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        self.vtk_widget.installEventFilter(self)
        self._lasso_requested = False
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
        self._rendering_enabled = True
        self._display_definitions: dict[int, LabelDefinition] = {}
        self._global_opacity = 1.0
        self._surface_state: SurfaceState | None = None
        self._rendered_cache_key: object | None = None
        self._surface_generation = 0
        self._active_request: SurfaceRequest | None = None
        self._tasks: dict[int, SurfaceTask] = {}
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)
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

        self._pending: SurfaceRequest | None = None
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
        self._lasso_requested = bool(enabled)
        self._lasso_enabled = self._lasso_requested and self._rendering_enabled
        self.vtk_widget.setCursor(
            Qt.CursorShape.CrossCursor
            if self._lasso_enabled
            else Qt.CursorShape.ArrowCursor
        )
        if not self._lasso_enabled:
            self.clear_lasso()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._lasso_points:
            QTimer.singleShot(0, lambda: self.set_lasso_points(self._lasso_points))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._surface_generation += 1
        self._rendering_enabled = False
        self._timer.stop()
        self._cursor_timer.stop()
        self._pending = None
        self._thread_pool.clear()
        self._thread_pool.waitForDone()
        self._active_request = None
        self._tasks.clear()
        super().closeEvent(event)

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
        if geometry_changed and self._surface_state is not None and self._rendering_enabled:
            self._queue_surface_request(
                self._surface_state,
                self._visible_values(),
                immediate=False,
            )
        elif self._initialized and render and self._rendering_enabled:
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

    def _visible_values(self) -> frozenset[int]:
        return frozenset(
            value
            for value, definition in self._display_definitions.items()
            if value > 0 and definition.visible
        )

    def _apply_actor_styles(self) -> None:
        visible_values = self._visible_values() if self._rendering_enabled else frozenset()
        preset = STYLE_PRESETS[self.render_settings.style]
        for value, pipeline in self.segment_pipelines.items():
            definition = self._display_definitions.get(value)
            pipeline.actor.SetVisibility(definition is not None and value in visible_values)
            if definition is None:
                continue
            red, green, blue = (channel / 255.0 for channel in definition.color)
            material = pipeline.actor.GetProperty()
            material.SetColor(red, green, blue)
            material.SetOpacity(
                float(np.clip(definition.opacity * self._global_opacity, 0.0, 1.0))
            )
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
        if visible and not self._has_camera:
            self._frame_camera()
            self._has_camera = True
        if self._rendering_enabled:
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
        if not self._rendering_enabled or self._cursor_timer.isActive():
            return
        elapsed_ms = (monotonic() - self._last_cursor_render_time) * 1000.0
        delay = 0 if self._last_cursor_render_time == 0.0 else max(
            0, int(np.ceil(self._cursor_throttle_ms - elapsed_ms))
        )
        self._cursor_timer.start(delay)

    def _render_cursor(self) -> None:
        if not self._initialized or not self._rendering_enabled:
            return
        self.vtk_widget.GetRenderWindow().Render()
        self._last_cursor_render_time = monotonic()

    @property
    def rendering_enabled(self) -> bool:
        return self._rendering_enabled

    def set_rendering_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._rendering_enabled:
            return
        self._rendering_enabled = enabled
        self._surface_generation += 1
        self._timer.stop()
        self._pending = None
        self._rendered_cache_key = None
        self._lasso_enabled = self._lasso_requested and enabled
        self.vtk_widget.setCursor(
            Qt.CursorShape.CrossCursor
            if self._lasso_enabled
            else Qt.CursorShape.ArrowCursor
        )
        if not enabled:
            self.clear_lasso()
            for pipeline in self.segment_pipelines.values():
                pipeline.actor.SetVisibility(False)
                pipeline.mapper.SetInputData(vtkPolyData())
            self.cursor_actor.SetVisibility(False)
        else:
            self.cursor_actor.SetVisibility(self._cursor_state is not None)
        if self._initialized:
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
            self.cursor_actor.SetVisibility(self._rendering_enabled)
        if self._initialized and changed and self._rendering_enabled:
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
        dirty_values: set[int] | frozenset[int] | None = None,
        cache_key: object | None = None,
    ) -> None:
        if frame is None:
            self._surface_generation += 1
            self._timer.stop()
            self._pending = None
            self._surface_state = None
            self._rendered_cache_key = None
            self._display_definitions.clear()
            for pipeline in self.segment_pipelines.values():
                pipeline.actor.SetVisibility(False)
            if self._initialized and self._rendering_enabled:
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
        self._display_definitions = {
            int(value): LabelDefinition(
                definition.value,
                definition.name,
                tuple(definition.color),
                definition.visible,
                definition.opacity,
            )
            for value, definition in labels.items()
            if int(value) > 0
        }
        self._global_opacity = float(np.clip(global_opacity, 0.0, 1.0))
        self._apply_actor_styles()
        state = SurfaceState(
            np.asarray(frame).copy() if self._rendering_enabled else np.asarray(frame),
            tuple(map(float, spacing or (1.0, 1.0, 1.0))),
            tuple(map(float, origin or (0.0, 0.0, 0.0))),
            np.asarray(direction if direction is not None else np.eye(3), dtype=float).copy(),
            cache_key if cache_key is not None else ("array", id(frame)),
        )
        self._surface_state = state
        if not self._rendering_enabled:
            return
        visible_values = self._visible_values()
        requested_values = (
            visible_values
            if dirty_values is None or state.cache_key != self._rendered_cache_key
            else frozenset(int(value) for value in dirty_values if int(value) in visible_values)
        )
        self._queue_surface_request(state, requested_values, immediate)

    def _queue_surface_request(
        self,
        state: SurfaceState,
        values: frozenset[int],
        immediate: bool,
    ) -> None:
        if not self._rendering_enabled:
            return
        combined_values = set(values)
        for outstanding in (self._active_request, self._pending):
            if outstanding is not None and outstanding.state.cache_key == state.cache_key:
                combined_values.update(outstanding.values)
        combined_values.intersection_update(self._visible_values())
        self._surface_generation += 1
        request = SurfaceRequest(
            self._surface_generation,
            state,
            frozenset(combined_values),
            self.render_settings,
        )
        self._pending = request
        if not request.values:
            self._pending = None
            self._rendered_cache_key = state.cache_key
            self._apply_actor_styles()
            if self._initialized:
                self.vtk_widget.GetRenderWindow().Render()
            return
        if immediate:
            self._timer.stop()
            self._pending = None
            started = monotonic()
            try:
                meshes = _build_surface_meshes(state, request.values, request.settings)
            except Exception as error:
                self.renderFailed.emit(str(error))
                return
            self._apply_surface_result(
                request,
                meshes,
                (monotonic() - started) * 1000.0,
            )
            return
        if self._active_request is not None:
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
        if (
            self._pending is None
            or self._active_request is not None
            or not self._rendering_enabled
        ):
            return
        request = self._pending
        self._pending = None
        task = SurfaceTask(request)
        task.signals.finished.connect(self._surface_task_finished)
        task.signals.failed.connect(self._surface_task_failed)
        self._active_request = request
        self._tasks[request.generation] = task
        self._thread_pool.start(task)

    @Slot(int, object, float)
    def _surface_task_finished(
        self,
        generation: int,
        meshes: object,
        worker_duration_ms: float,
    ) -> None:
        task = self._tasks.pop(generation, None)
        request = task.request if task is not None else None
        if self._active_request is not None and self._active_request.generation == generation:
            self._active_request = None
        if (
            request is not None
            and generation == self._surface_generation
            and self._rendering_enabled
            and isinstance(meshes, dict)
        ):
            self._apply_surface_result(request, meshes, worker_duration_ms)
        if self._pending is not None and self._rendering_enabled:
            self._timer.start(0)

    @Slot(int, str)
    def _surface_task_failed(self, generation: int, message: str) -> None:
        self._tasks.pop(generation, None)
        if self._active_request is not None and self._active_request.generation == generation:
            self._active_request = None
        if generation == self._surface_generation:
            self.renderFailed.emit(message)
        if self._pending is not None and self._rendering_enabled:
            self._timer.start(0)

    def _apply_surface_result(
        self,
        request: SurfaceRequest,
        meshes: dict[int, vtkPolyData],
        worker_duration_ms: float,
    ) -> None:
        if request.generation != self._surface_generation or not self._rendering_enabled:
            return
        started = monotonic()
        for value, mesh in meshes.items():
            definition = self._display_definitions.get(value)
            if definition is None:
                continue
            pipeline = self._segment_pipeline(definition)
            pipeline.mapper.SetInputData(mesh)
            pipeline.mapper.Modified()
        self._apply_actor_styles()
        visible = [
            pipeline
            for pipeline in self.segment_pipelines.values()
            if pipeline.actor.GetVisibility()
        ]
        if visible and (
            not self._has_camera or request.state.cache_key != self._rendered_cache_key
        ):
            self._frame_camera()
            self._has_camera = True
        else:
            self._reset_camera_clipping_range()
        if self._initialized:
            self.vtk_widget.GetRenderWindow().Render()
            self._cursor_timer.stop()
            self._last_cursor_render_time = monotonic()
        completed = monotonic()
        self._rendered_cache_key = request.state.cache_key
        self._last_apply_duration_ms = worker_duration_ms + (completed - started) * 1000.0
        self._last_apply_time = completed

    def set_label_opacities(
        self,
        labels: dict[int, LabelDefinition],
        global_opacity: float,
    ) -> None:
        self._display_definitions = {
            int(value): LabelDefinition(
                definition.value,
                definition.name,
                tuple(definition.color),
                definition.visible,
                definition.opacity,
            )
            for value, definition in labels.items()
            if int(value) > 0
        }
        self._global_opacity = float(np.clip(global_opacity, 0.0, 1.0))
        self._apply_actor_styles()
        if self._initialized and self._rendering_enabled:
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
