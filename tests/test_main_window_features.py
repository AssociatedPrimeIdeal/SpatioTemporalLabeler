import os

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.model import default_label
from spatiotemporal_labeler.ui import MainWindow
from spatiotemporal_labeler.ui.viewer_3d import Mask3DViewer


def ensure_application():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    application.setOrganizationName("SpatioTemporalLabelerTests")
    application.setApplicationName("SpatioTemporalLabelerTests")
    return application


def make_sequence(data):
    transform = AxisTransform(
        original_axis_for_canonical=(0, 1, 2, 3),
        flipped_canonical_axes=(False, False, False),
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_ras=(0.0, 0.0, 0.0),
        direction_ras=np.eye(3),
    )
    return Sequence4D(np.asarray(data), {}, transform)


def make_window(image_data, mask_data):
    ensure_application()
    window = MainWindow()
    image = make_sequence(image_data)
    mask = make_sequence(mask_data)
    window.images.append(image)
    window.image_combo.addItem("image")
    window.masks.append(mask)
    window._mask_labels[id(mask)] = {1: default_label(1), 2: default_label(2)}
    window.mask_combo.addItem("mask")
    return window, image, mask


def finish_window(window, mask):
    mask.dirty = False
    window.close()
    window.deleteLater()


def test_threshold_constrains_brush_and_bypass_ignores_it():
    image_data = np.zeros((7, 7, 1, 2), dtype=np.float32)
    image_data[3:, :, :, :] = 10.0
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.cursor = [3, 3, 0, 0]
    window.brush_diameter.setValue(1.0)
    window.threshold_panel.set_bounds(5.0, 10.0)
    window.threshold_panel.enabled.setChecked(True)

    window._stroke_started("X-Y", 3, 3)
    window._stroke_finished("X-Y", 3, 3)

    assert np.all(mask.data[image_data < 5.0] == 0)
    assert mask.data[3, 3, 0, 0] == 1

    window._threshold_bypass_held = True
    window._stroke_started("X-Y", 1, 3)
    window._stroke_finished("X-Y", 1, 3)

    assert mask.data[1, 3, 0, 0] == 1
    finish_window(window, mask)


def test_held_all_frames_picker_and_region_barrier():
    image_data = np.ones((7, 7, 1, 2), dtype=np.float32)
    mask_data = np.zeros(image_data.shape, dtype=np.uint8)
    mask_data[3, :, 0, 0] = 2
    window, _image, mask = make_window(image_data, mask_data)
    window.cursor = [1, 2, 0, 0]
    window.brush_diameter.setValue(1.0)
    window._all_frames_held = True

    window._stroke_started("X-Y", 1, 2)
    window._stroke_finished("X-Y", 1, 2)

    assert np.all(mask.data[1, 2, 0, :] == 1)

    window._pick_label("X-Y", 3, 2)
    assert window.active_label_value == 2
    window.active_label_value = 1
    window.grow_panel.tolerance.setValue(0.0)
    window.grow_panel.scope.setCurrentIndex(0)
    window._grow_from_seed("X-Y", 1, 1)

    assert np.all(mask.data[:3, :, 0, 0] == 1)
    assert np.all(mask.data[3, :, 0, 0] == 2)
    assert np.all(mask.data[4:, :, 0, 0] == 0)
    finish_window(window, mask)


def test_picker_shortcut_tracks_hover_without_replacing_brush():
    image_data = np.ones((7, 7, 1, 1), dtype=np.float32)
    mask_data = np.zeros(image_data.shape, dtype=np.uint8)
    mask_data[4, 3, 0, 0] = 2
    window, _image, mask = make_window(image_data, mask_data)
    window.shortcuts["picker"] = "I"
    window._apply_shortcuts()
    window.isActiveWindow = lambda: True

    pressed = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_I,
        Qt.KeyboardModifier.NoModifier,
    )
    released = QKeyEvent(
        QEvent.Type.KeyRelease,
        Qt.Key.Key_I,
        Qt.KeyboardModifier.NoModifier,
    )

    assert window.tool_actions["picker"].shortcut().isEmpty()
    assert window.eventFilter(window, pressed)
    window._view_hovered("X-Y", 4, 3, True)
    assert window.active_label_value == 2
    assert window.tool == "brush"
    assert window.tool_actions["brush"].isChecked()
    window._stroke_started("X-Y", 2, 2)
    window._stroke_finished("X-Y", 2, 2)
    assert mask.data[2, 2, 0, 0] == 0
    assert window.eventFilter(window, released)
    assert not window._picker_held
    assert window.tool == "brush"
    finish_window(window, mask)


def test_seed_grow_action_closes_panel_and_restores_previous_tool():
    image_data = np.ones((7, 7, 1, 1), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window._update_enabled_state()

    window.tool_actions["grow"].trigger()
    assert window.tool == "grow"
    assert not window.grow_dock.isHidden()

    window.tool_actions["grow"].trigger()
    assert window.tool == "brush"
    assert window.tool_actions["brush"].isChecked()
    assert window.grow_dock.isHidden()
    finish_window(window, mask)


def test_morphology_respects_selected_label_and_current_frame_and_undoes():
    image_data = np.ones((7, 7, 1, 2), dtype=np.float32)
    mask_data = np.zeros(image_data.shape, dtype=np.uint8)
    mask_data[1:3, 1:3, 0, :] = 1
    mask_data[5, 5, 0, :] = 1
    mask_data[6, 0, 0, :] = 2
    window, _image, mask = make_window(image_data, mask_data)
    window.cursor[3] = 0
    window.active_label_value = 1
    window.morphology_panel.minimum_volume.setValue(2.0)
    original = mask.data.copy()

    window._apply_morphology()

    assert mask.data[5, 5, 0, 0] == 0
    assert mask.data[5, 5, 0, 1] == 1
    assert np.all(mask.data[6, 0, 0, :] == 2)
    assert len(window._undo_stack) == 1

    window.undo()
    assert np.array_equal(mask.data, original)

    window.morphology_panel.labels_scope.setCurrentIndex(1)
    window.morphology_panel.frames_scope.setCurrentIndex(1)
    window._apply_morphology()
    assert np.all(mask.data[5, 5, 0, :] == 0)
    assert np.all(mask.data[6, 0, 0, :] == 0)
    assert np.all(mask.data[1:3, 1:3, 0, :] == 1)

    window.undo()
    assert np.array_equal(mask.data, original)
    finish_window(window, mask)


def test_toolbar_uses_one_import_action_for_images_and_labels():
    image_data = np.ones((3, 3, 1, 1), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )

    assert window.import_action in window.file_menu.actions()
    assert not hasattr(window, "open_image_action")
    assert not hasattr(window, "open_mask_action")
    finish_window(window, mask)


def test_keyframe_interpolation_is_one_undoable_edit():
    image_data = np.ones((9, 9, 1, 3), dtype=np.float32)
    mask_data = np.zeros(image_data.shape, dtype=np.uint8)
    mask_data[1:4, 3:6, 0, 0] = 1
    mask_data[3:6, 3:6, 0, 2] = 1
    window, _image, mask = make_window(image_data, mask_data)
    original = mask.data.copy()
    window.interpolation_panel.start_frame.setValue(1)
    window.interpolation_panel.end_frame.setValue(3)

    window._apply_interpolation()

    assert np.count_nonzero(mask.data[..., 1]) == 9
    assert len(window._undo_stack) == 1
    window.undo()
    assert np.array_equal(mask.data, original)
    finish_window(window, mask)


def test_loading_3d_labels_can_copy_all_frames_or_target_one(monkeypatch):
    ensure_application()
    window = MainWindow()
    image = make_sequence(np.ones((5, 6, 1, 4), dtype=np.float32))
    window.images.append(image)
    window.image_combo.addItem("image")
    source_data = np.zeros((5, 6, 1, 1), dtype=np.uint8)
    source_data[2:4, 2:5, 0, 0] = 1
    source = make_sequence(source_data)
    source.transform = AxisTransform(
        original_axis_for_canonical=(0, 1, 2, 3),
        flipped_canonical_axes=(False, False, False),
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_ras=(0.0, 0.0, 0.0),
        direction_ras=np.eye(3),
        original_ndim=3,
    )
    monkeypatch.setattr(Sequence4D, "load", lambda _path: source)

    copied = window.load_mask("source.seg.nrrd", frame_mapping=None)
    assert copied is not None
    assert copied.data.shape == image.data.shape
    assert np.all(copied.data == copied.data[..., [0]])

    targeted = window.load_mask("source.seg.nrrd", frame_mapping=2)
    assert targeted is not None
    assert not np.any(targeted.data[..., :2])
    assert np.array_equal(targeted.data[..., 2], source.data[..., 0])
    assert not np.any(targeted.data[..., 3])
    copied.dirty = False
    finish_window(window, targeted)


def test_threshold_and_window_sliders_update_live_and_are_independent():
    image_data = np.zeros((7, 7, 1, 1), dtype=np.float32)
    image_data[3:, :, :, :] = 10.0
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.cursor = [3, 3, 0, 0]
    assert window.threshold_panel.parent() is not window.grow_panel.parent()
    window.threshold_panel.enabled.setChecked(True)
    window.threshold_panel.set_bounds(0.0, 10.0)
    before = int(window._threshold_selection().sum())

    window.threshold_panel.lower_slider.setValue(500)

    after = int(window._threshold_selection().sum())
    preview = window.slice_views["X-Y"].threshold_item.image
    assert after < before
    assert preview is not None
    assert np.any(preview[..., 3] > 0)

    old_levels = window._levels
    window.window_level_panel.level_control.slider.setValue(750)

    assert window._levels != old_levels
    finish_window(window, mask)


def test_selecting_threshold_method_calculates_without_a_button():
    image_data = np.zeros((8, 8, 2, 1), dtype=np.float32)
    image_data[4:, :, :, :] = 10.0
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    panel = window.threshold_panel

    panel.method.setCurrentIndex(panel.method.findData("otsu"))

    assert not hasattr(panel, "auto_button")
    assert 0.0 < panel.lower.value() < 10.0
    assert panel.upper.value() == 10.0
    finish_window(window, mask)


def test_double_click_maximizes_across_the_full_two_by_two_panel():
    ensure_application()
    image_data = np.zeros((7, 7, 2, 1), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.view_grid.resize(800, 600)
    window.view_grid_layout.setGeometry(window.view_grid.rect())
    target = window.slice_views["X-Y"]
    normal_width = target.width()

    window._view_double_clicked("X-Y")
    window.view_grid_layout.setGeometry(window.view_grid.rect())

    assert target.width() > normal_width * 1.7
    assert target.width() >= window.view_grid.width() - 4
    assert target.height() >= window.view_grid.height() - 4
    assert window.slice_views["X-Z"].isHidden()

    window._view_double_clicked("X-Y")
    window.view_grid_layout.setGeometry(window.view_grid.rect())

    assert not window.slice_views["X-Z"].isHidden()
    assert not window.temporal_panel.isHidden()
    finish_window(window, mask)


def test_3d_viewer_builds_one_binary_surface_actor_per_label():
    ensure_application()
    viewer = Mask3DViewer()
    frame = np.zeros((8, 8, 8), dtype=np.uint8)
    frame[1:4, 1:4, 1:4] = 1
    frame[5:7, 5:7, 5:7] = 2
    labels = {1: default_label(1), 2: default_label(2)}

    viewer.set_mask(frame, labels=labels, immediate=True)

    assert set(viewer.segment_pipelines) == {1, 2}
    first = viewer.segment_pipelines[1]
    second = viewer.segment_pipelines[2]
    assert first.actor is not second.actor
    assert first.mapper.GetScalarVisibility() == 0
    assert first.decimator.GetTargetReduction() > 0.0
    viewer.close()
    viewer.deleteLater()


def test_3d_viewer_uses_non_inertial_trackball_interaction():
    ensure_application()
    viewer = Mask3DViewer()

    style = viewer.vtk_widget.GetRenderWindow().GetInteractor().GetInteractorStyle()

    assert style.GetClassName() == "vtkInteractorStyleTrackballCamera"
    assert style.GetUseTimers() == 0
    viewer.close()
    viewer.deleteLater()


def test_brush_updates_all_2d_overlays_and_defers_3d_until_release():
    image_data = np.zeros((12, 12, 2, 3), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.cursor = [4, 4, 0, 0]
    full_refreshes = []
    surface_refreshes = []
    overlay_refreshes = {plane: [] for plane in ("X-Y", "X-Z", "Y-Z", "X-T")}
    window.refresh_views = lambda *args, **kwargs: full_refreshes.append(None)
    window._refresh_3d = lambda *args, **kwargs: surface_refreshes.append(None)
    for plane, view in window.slice_views.items():
        view.set_mask_overlay = (
            lambda data, labels, plane=plane: overlay_refreshes[plane].append(
                np.asarray(data).copy()
            )
        )
    window.temporal_view.set_mask_overlay = (
        lambda data, labels: overlay_refreshes["X-T"].append(np.asarray(data).copy())
    )

    window._stroke_started("X-Y", 4, 4)
    window._stroke_moved("X-Y", 8, 8)

    assert {plane: len(refreshes) for plane, refreshes in overlay_refreshes.items()} == {
        plane: 2 for plane in overlay_refreshes
    }
    assert overlay_refreshes["X-Y"][-1][8, 8] == 1
    assert overlay_refreshes["X-Z"][-1][8, 0] == 1
    assert overlay_refreshes["Y-Z"][-1][8, 0] == 1
    assert overlay_refreshes["X-T"][-1][8, 0] == 1
    assert full_refreshes == []
    assert surface_refreshes == []

    window._stroke_finished("X-Y", 8, 8)

    assert {plane: len(refreshes) for plane, refreshes in overlay_refreshes.items()} == {
        plane: 3 for plane in overlay_refreshes
    }
    assert len(full_refreshes) == 1
    assert len(surface_refreshes) == 1
    finish_window(window, mask)
