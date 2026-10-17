import os

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtWidgets import QApplication, QDialog

from spatiotemporal_labeler.ui import main_window as main_window_module
from spatiotemporal_labeler.ui.image_strip import ThumbnailPlot
from spatiotemporal_labeler.ui.orientation_settings import (
    ANATOMICAL_VECTORS,
    VIEW_AXES,
    OrientationSettings,
)
from spatiotemporal_labeler.ui.settings_dialog import DEFAULT_SHORTCUTS, SettingsDialog
from spatiotemporal_labeler.ui.slice_view import SliceView, TemporalView
from spatiotemporal_labeler.ui.viewer_3d import Mask3DViewer


def ensure_application():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    return application


def test_persisted_orientation_rejects_invalid_values_and_parallel_camera_up():
    selected = OrientationSettings.normalized(
        {"X-Y/horizontal": "true", "X-T/vertical": "1", "3d/view": "F",
         "3d/up": "H", "3d/X": "L", "3d/Y": "R", "3d/Z": "F"}
    )

    assert selected.flips("X-Y") == (True, False)
    assert selected.flips("X-T") == (False, True)
    assert selected.camera_up == "A"
    assert selected.axis_directions == ("L", "A", "F")
    assert OrientationSettings.normalized(selected.as_dict()) == selected
    assert OrientationSettings.normalized({"3d/view": "invalid", "3d/up": None}) == OrientationSettings()


@pytest.mark.parametrize("language", ["en", "zh_CN"])
def test_orientation_dialog_previews_valid_camera_choices_and_restores_defaults(language):
    ensure_application()
    selected = OrientationSettings.normalized({"X-Z/horizontal": True, "3d/X": "L"})
    dialog = SettingsDialog(language, DEFAULT_SHORTCUTS, orientation_settings=selected)
    changes = []
    dialog.orientationSettingsChanged.connect(changes.append)

    assert dialog.orientation_values() == selected
    assert dialog.tabs.tabText(1) == ("视图方向" if language == "zh_CN" else "View Orientation")
    dialog.camera_view.setCurrentIndex(dialog.camera_view.findData("H"))
    assert changes[-1].camera_view == "H"
    assert changes[-1].camera_up == "A"
    for direction in ("H", "F"):
        index = dialog.camera_up.findData(direction)
        assert not dialog.camera_up.model().item(index).isEnabled()
    dialog._restore_defaults()
    assert dialog.orientation_values() == OrientationSettings()
    assert changes[-1] == OrientationSettings()
    dialog.reject()
    dialog.deleteLater()


@pytest.mark.parametrize("plane", VIEW_AXES)
@pytest.mark.parametrize("horizontal,vertical", [(False, False), (True, False), (False, True), (True, True)])
def test_flipped_views_keep_screen_pixels_labels_and_edit_coordinates_aligned(plane, horizontal, vertical):
    application = ensure_application()
    settings = OrientationSettings.normalized(
        {f"{plane}/horizontal": horizontal, f"{plane}/vertical": vertical}
    )
    image = np.arange(40, dtype=np.float32).reshape(8, 5)
    original = image.copy()
    mask = np.zeros((8, 5), dtype=np.uint8)
    mask[1, 2] = 1
    spatial = plane in {"X-Y", "X-Z", "Y-Z"}
    view = SliceView(plane) if spatial else TemporalView()
    view.set_orientation_settings(settings)
    if spatial:
        view.set_slice(image, mask, (2.0, 3.0), (0.0, 39.0), "Z", 0, cursor=(1, 2))
    else:
        view.set_sequence_slice(image, mask, plane, 2.0, (0.0, 39.0), 2, "", cursor=(1, 2))
    preview = ThumbnailPlot(0)
    preview.set_orientation_settings(settings)
    preview.set_image(image, (0.0, 39.0), (0.0, 39.0), plane, (2.0, 3.0))
    view.resize(400, 300)
    view.show()
    preview.show()
    application.processEvents()

    for item in (view.image_item, preview.item):
        first = item.mapToScene(QPointF(0.5, 0.5))
        last_h = item.mapToScene(QPointF(7.5, 0.5))
        last_v = item.mapToScene(QPointF(0.5, 4.5))
        assert (first.x() > last_h.x()) == horizontal
        assert (first.y() < last_v.y()) == vertical
    selected = view.image_item.mapToScene(QPointF(1.5, 2.5))
    assert view.mask_item.mapToScene(QPointF(1.5, 2.5)) == selected
    assert view.mask_item.image[2, 1, 3] > 0
    requested = []
    view.strokeStarted.connect(lambda *args: requested.append(args))

    class EraseClick:
        def pos(self):
            return view.image_item.mapFromScene(selected)

        def button(self):
            return Qt.MouseButton.RightButton

        def modifiers(self):
            return Qt.KeyboardModifier.NoModifier

        def accept(self):
            pass

    view.image_item.mouseClickEvent(EraseClick())
    assert requested == [(plane, 1, 2, True)]
    assert np.array_equal(image, original)
    if spatial:
        vertical_tip = view._locator_handles[0][0]._polygon().at(0)
        horizontal_tip = view._locator_handles[1][0]._polygon().at(0)
        assert (vertical_tip.y() > 0) == vertical
        assert (horizontal_tip.x() < 0) == horizontal
    for widget in (view, preview):
        widget.close()
        widget.deleteLater()


def test_3d_preferences_change_camera_and_real_arrow_directions_without_moving_data():
    ensure_application()
    viewer = Mask3DViewer()
    camera = viewer.renderer.GetActiveCamera()
    camera.SetFocalPoint(10, 20, 30)
    camera.SetPosition(10, 120, 30)
    viewer.set_cursor((2, 3, 4), (1.0, 2.0, 3.0))
    cursor = viewer.cursor_actor.GetPosition()
    selected = OrientationSettings(camera_view="P", camera_up="H", axis_directions=("L", "P", "F"))
    viewer.set_orientation_settings(selected, render=False)

    assert np.allclose(camera.GetDirectionOfProjection(), (0, 1, 0))
    assert np.allclose(camera.GetViewUp(), (0, 0, 1))
    assert np.allclose(camera.GetFocalPoint(), (10, 20, 30))
    assert viewer.cursor_actor.GetPosition() == cursor
    axes = viewer.orientation_axes
    assert (axes.GetXAxisLabelText(), axes.GetYAxisLabelText(), axes.GetZAxisLabelText()) == ("L", "P", "F")
    matrix = axes.GetUserMatrix()
    for index, direction in enumerate(selected.axis_directions):
        endpoint = [0.0, 0.0, 0.0, 1.0]
        endpoint[index] = 1.0
        assert np.allclose(matrix.MultiplyPoint(endpoint)[:3], ANATOMICAL_VECTORS[direction])
    camera.Azimuth(35)
    rotated = camera.GetPosition()
    changed_2d = selected.as_dict()
    changed_2d["X-Y/horizontal"] = True
    viewer.set_orientation_settings(OrientationSettings.normalized(changed_2d), render=False)
    assert camera.GetPosition() == rotated
    viewer.close()
    viewer.deleteLater()


@pytest.mark.parametrize("accepted", [False, True])
def test_settings_preview_cancel_and_persistence_restore_views_and_rotated_camera(tmp_path, monkeypatch, accepted):
    ensure_application()
    filename = str(tmp_path / "orientation.ini")
    monkeypatch.setattr(main_window_module, "QSettings", lambda: QSettings(filename, QSettings.Format.IniFormat))
    window = main_window_module.MainWindow()
    camera = window.viewer_3d.renderer.GetActiveCamera()
    camera.Azimuth(27)
    original_position = camera.GetPosition()
    original_up = camera.GetViewUp()
    selected = OrientationSettings.normalized(
        {"X-Y/horizontal": True, "X-Z/vertical": True, "Y-T/vertical": True,
         "3d/view": "P", "3d/up": "R", "3d/X": "L", "3d/Z": "F"}
    )

    class PreviewDialog(SettingsDialog):
        def exec(self):
            self._set_orientation_values(selected)
            self.orientationSettingsChanged.emit(self.orientation_values())
            assert window.slice_views["X-Y"].getAxis("bottom").labelText == "R - L"
            assert window.slice_views["X-Z"].getViewBox().state["yInverted"]
            return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected

    monkeypatch.setattr(main_window_module, "SettingsDialog", PreviewDialog)
    window._open_settings()
    expected = selected if accepted else OrientationSettings()
    assert window.orientation_settings == expected
    assert window.image_previews.orientation_settings == expected
    if not accepted:
        assert camera.GetPosition() == original_position
        assert camera.GetViewUp() == original_up
    window.settings.sync()
    reopened = main_window_module.MainWindow()
    assert reopened.orientation_settings == expected
    assert reopened.slice_views["X-Y"].getViewBox().state["xInverted"] == accepted
    assert reopened.viewer_3d.orientation_settings == expected
    for widget in (window, reopened):
        widget.close()
        widget.deleteLater()
