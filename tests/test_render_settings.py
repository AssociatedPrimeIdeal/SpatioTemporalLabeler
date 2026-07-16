import os

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import pytest
from PySide6.QtWidgets import QApplication

from spatiotemporal_labeler.model import default_label
from spatiotemporal_labeler.ui.render_settings import RenderSettings
from spatiotemporal_labeler.ui.settings_dialog import DEFAULT_SHORTCUTS, SettingsDialog
from spatiotemporal_labeler.ui.viewer_3d import Mask3DViewer, STYLE_PRESETS


def ensure_application():
    return QApplication.instance() or QApplication([])


def test_render_settings_normalize_persisted_values():
    settings = RenderSettings.normalized(
        {
            "style": "unknown",
            "lighting": 500,
            "smoothing": -4,
            "detail": "invalid",
        }
    )

    assert settings == RenderSettings(lighting=160, smoothing=0)


def test_settings_dialog_exposes_simple_3d_rendering_controls():
    ensure_application()
    selected = RenderSettings(style="glossy", lighting=125, smoothing=5, detail="fine")
    dialog = SettingsDialog(
        "en", DEFAULT_SHORTCUTS, render_settings=selected
    )

    assert dialog.tabs.count() == 2
    assert dialog.tabs.tabText(1) == "3D Rendering"
    assert dialog.render_values() == selected

    dialog._restore_defaults()

    assert dialog.render_values() == RenderSettings()
    dialog._render_timer.start()
    dialog.reject()
    assert not dialog._render_timer.isActive()
    dialog.deleteLater()


def test_viewer_applies_material_lighting_and_geometry_settings():
    ensure_application()
    viewer = Mask3DViewer()
    pipeline = viewer._segment_pipeline(default_label(1))
    settings = RenderSettings(style="glossy", lighting=140, smoothing=3, detail="fine")

    viewer.set_render_settings(RenderSettings(smoothing=0), render=False)
    assert (
        pipeline.decimator.GetInputConnection(0, 0).GetProducer().GetClassName()
        == "vtkFlyingEdges3D"
    )

    viewer.set_render_settings(settings, render=False)

    preset = STYLE_PRESETS["glossy"]
    material = pipeline.actor.GetProperty()
    assert pipeline.smoother.GetNumberOfIterations() == 3
    assert (
        pipeline.decimator.GetInputConnection(0, 0).GetProducer().GetClassName()
        == "vtkWindowedSincPolyDataFilter"
    )
    assert pipeline.decimator.GetTargetReduction() == pytest.approx(0.10)
    assert material.GetAmbient() == pytest.approx(preset["ambient"])
    assert material.GetDiffuse() == pytest.approx(preset["diffuse"])
    assert material.GetSpecular() == pytest.approx(preset["specular"])
    assert material.GetSpecularPower() == pytest.approx(preset["specular_power"])
    assert viewer.light_kit.GetKeyLightIntensity() == pytest.approx(0.82 * 1.40)
    viewer.close()
    viewer.deleteLater()
