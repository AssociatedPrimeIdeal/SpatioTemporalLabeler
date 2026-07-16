from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - registers the OpenGL render backend
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import VTK_INT
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import (
    vtkDecimatePro,
    vtkFlyingEdges3D,
    vtkPolyDataNormals,
    vtkWindowedSincPolyDataFilter,
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkImagingCore import vtkImageConstantPad, vtkImageThreshold
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkLightKit,
    vtkPolyDataMapper,
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

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

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
        frame, spacing, origin, direction, definitions = self._pending
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
