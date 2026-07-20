import os

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.model import default_label
from spatiotemporal_labeler.ui import MainWindow
from spatiotemporal_labeler.ui.label_panel import THRESHOLD_MASK_ITEM
from spatiotemporal_labeler.ui.viewer_3d import Mask3DViewer
from spatiotemporal_labeler.ui import viewer_3d as viewer_3d_module


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


def test_add_label_suggests_the_first_unused_positive_value(monkeypatch):
    image_data = np.ones((3, 3, 1, 1), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window._mask_labels[id(mask)] = {
        value: default_label(value) for value in (1, 2, 3, 4, 7, 8)
    }
    suggested_values = []

    def get_int(_parent, _title, _label, value, _minimum, _maximum):
        suggested_values.append(value)
        return value, True

    monkeypatch.setattr(
        "spatiotemporal_labeler.ui.main_window.QInputDialog.getInt", get_int
    )
    monkeypatch.setattr(
        "spatiotemporal_labeler.ui.main_window.QInputDialog.getText",
        lambda *_args, **_kwargs: ("Label 5", True),
    )

    window._add_label()

    assert suggested_values == [5]
    assert 5 in window.active_labels
    assert window.active_label_value == 5
    finish_window(window, mask)


def test_threshold_constrains_brush_and_bypass_ignores_it():
    image_data = np.zeros((7, 7, 1, 2), dtype=np.float32)
    image_data[3:, :, :, :] = 10.0
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.cursor = [3, 3, 0, 0]
    window.brush_diameter.setValue(1.0)
    window.threshold_panel.set_bounds(5.0, 10.0)
    window._apply_threshold_mask()

    window._stroke_started("X-Y", 3, 3)
    window._stroke_finished("X-Y", 3, 3)

    assert np.all(mask.data[image_data < 5.0] == 0)
    assert mask.data[3, 3, 0, 0] == 1

    window._threshold_bypass_held = True
    window._stroke_started("X-Y", 1, 3)
    window._stroke_finished("X-Y", 1, 3)

    assert mask.data[1, 3, 0, 0] == 1
    finish_window(window, mask)


def test_threshold_apply_creates_one_toggleable_replaceable_special_label():
    image_data = np.zeros((7, 7, 1, 2), dtype=np.float32)
    image_data[3:, :, :, :] = 10.0
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.threshold_panel.set_bounds(5.0, 10.0)

    window._apply_threshold_mask()

    first = window._applied_threshold_mask.copy()
    threshold_items = [
        window.label_panel.list_widget.item(row)
        for row in range(window.label_panel.list_widget.count())
        if window.label_panel.list_widget.item(row).data(Qt.ItemDataRole.UserRole)
        == THRESHOLD_MASK_ITEM
    ]
    assert len(threshold_items) == 1
    assert window._active_threshold_selection() is not None

    window.threshold_panel.set_bounds(0.0, 0.0)
    window._apply_threshold_mask()

    assert not np.array_equal(window._applied_threshold_mask, first)
    assert sum(
        window.label_panel.list_widget.item(row).data(Qt.ItemDataRole.UserRole)
        == THRESHOLD_MASK_ITEM
        for row in range(window.label_panel.list_widget.count())
    ) == 1
    threshold_item = next(
        window.label_panel.list_widget.item(row)
        for row in range(window.label_panel.list_widget.count())
        if window.label_panel.list_widget.item(row).data(Qt.ItemDataRole.UserRole)
        == THRESHOLD_MASK_ITEM
    )
    threshold_item.setCheckState(Qt.CheckState.Unchecked)
    assert window._active_threshold_selection() is None
    window.label_panel.list_widget.setCurrentItem(threshold_item)
    window.label_panel._delete_requested()
    assert window._applied_threshold_mask is None
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


def test_2d_lasso_erase_respects_all_time_frames_and_is_one_undo():
    image_data = np.ones((9, 9, 1, 3), dtype=np.float32)
    mask_data = np.ones(image_data.shape, dtype=np.uint8)
    window, _image, mask = make_window(image_data, mask_data)
    original = mask.data.copy()
    window._set_tool("lasso")
    window.all_frames_toggle.setChecked(True)

    window._stroke_started("X-Y", 2, 2)
    window._stroke_moved("X-Y", 6, 2)
    window._stroke_moved("X-Y", 6, 6)
    window._stroke_finished("X-Y", 2, 6)

    assert np.all(mask.data[4, 4, 0, :] == 0)
    assert np.all(mask.data[1, 1, 0, :] == 1)
    assert window.slice_views["X-Y"].lasso_overlay.path.xData is None
    assert len(window.slice_views["X-Y"].lasso_overlay.fill.polygon()) == 0
    assert len(window._undo_stack) == 1
    window.undo()
    assert np.array_equal(mask.data, original)
    finish_window(window, mask)


def test_3d_lasso_uses_the_same_projected_region_in_all_time_frames():
    image_data = np.ones((7, 7, 2, 3), dtype=np.float32)
    mask_data = np.ones(image_data.shape, dtype=np.uint8)
    window, _image, mask = make_window(image_data, mask_data)
    selection = np.zeros(mask.data.shape[:3], dtype=bool)
    selection[2:5, 2:5, :] = True
    window.viewer_3d.lasso_voxel_mask = lambda *args, **kwargs: selection
    cleared = []
    window.viewer_3d.clear_lasso = lambda: cleared.append(None)
    window._set_tool("lasso")
    cleared.clear()
    window.all_frames_toggle.setChecked(True)

    window._lasso_3d_started()
    window._lasso_3d_finished([(1.0, 1.0), (5.0, 1.0), (5.0, 5.0)])

    assert np.all(mask.data[3, 3, :, :] == 0)
    assert np.all(mask.data[0, 0, :, :] == 1)
    assert cleared
    assert len(window._undo_stack) == 1
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


def test_hover_status_shows_voxel_ras_intensity_and_label():
    image_data = np.zeros((5, 6, 7, 3), dtype=np.float32)
    mask_data = np.zeros(image_data.shape, dtype=np.uint8)
    image_data[1, 2, 3, 1] = 123.456
    mask_data[1, 2, 3, 1] = 2
    window, image, mask = make_window(image_data, mask_data)
    image.transform = AxisTransform(
        original_axis_for_canonical=(0, 1, 2, 3),
        flipped_canonical_axes=(False, False, False),
        spacing_xyz=(2.0, 3.0, 4.0),
        origin_ras=(10.0, 20.0, 30.0),
        direction_ras=np.asarray(
            (
                (0.0, -1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
    )
    mask.transform = image.transform
    window._set_language("en")
    window.cursor = [0, 0, 3, 1]

    window._view_hovered("X-Y", 1, 2, True)

    assert window.cursor_status.text() == (
        "X 2  Y 3  Z 4  T 2 | RAS (4, 22, 42) mm | "
        "Intensity 123.46 | Label 2"
    )

    image.data[4, 2, 3, 2] = 7.5
    mask.data[4, 2, 3, 2] = 1
    window.cursor = [0, 2, 3, 1]
    window._view_hovered("X-T", 4, 2, True)
    assert window.cursor_status.text() == (
        "X 5  Y 3  Z 4  T 3 | RAS (4, 28, 42) mm | "
        "Intensity 7.5 | Label 1"
    )

    mask.data[1, 2, 3, 1] = 1
    window._view_hovered("X-Y", 1, 2, True)
    window.refresh_views()
    assert window.cursor_status.text().endswith("Label 1")

    window._view_hovered("X-Y", 0, 0, False)
    assert window.cursor_status.text() == ""
    finish_window(window, mask)


def test_2d_views_do_not_show_hover_operation_tooltips():
    image_data = np.ones((5, 6, 2, 1), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )

    assert all(
        view.toolTip() == ""
        for view in [*window.slice_views.values(), window.temporal_view]
    )
    finish_window(window, mask)


def test_hide_labels_hold_shortcut_hides_and_restores_every_2d_overlay():
    image_data = np.ones((5, 6, 2, 3), dtype=np.float32)
    mask_data = np.ones(image_data.shape, dtype=np.uint8)
    window, _image, mask = make_window(image_data, mask_data)
    window.shortcuts["hide_labels_hold"] = "H"
    window.isActiveWindow = lambda: True
    window._applied_threshold_mask = np.ones(image_data.shape, dtype=bool)
    window._applied_threshold_image = window.active_image
    window.refresh_views()
    views = [*window.slice_views.values(), window.temporal_view]
    pressed = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_H,
        Qt.KeyboardModifier.NoModifier,
    )
    released = QKeyEvent(
        QEvent.Type.KeyRelease,
        Qt.Key.Key_H,
        Qt.KeyboardModifier.NoModifier,
    )

    assert window.eventFilter(window, pressed)
    assert window._labels_hidden_held
    assert all(not view.mask_item.isVisible() for view in views)
    assert all(not view.applied_threshold_item.isVisible() for view in views)
    assert all(view.threshold_item.isVisible() for view in views)

    window.refresh_views()
    assert all(not view.mask_item.isVisible() for view in views)
    assert all(not view.applied_threshold_item.isVisible() for view in views)

    assert window.eventFilter(window, released)
    assert not window._labels_hidden_held
    assert all(view.mask_item.isVisible() for view in views)
    assert all(view.applied_threshold_item.isVisible() for view in views)
    finish_window(window, mask)


def test_enter_confirms_a_pending_contour():
    image_data = np.ones((7, 7, 1, 1), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.isActiveWindow = lambda: True
    window._pending_contour = type("Pending", (), {"plane": "X-Y"})()
    confirmed = []
    window._confirm_contour = lambda plane: confirmed.append(plane)
    pressed = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Return,
        Qt.KeyboardModifier.NoModifier,
    )

    assert window.eventFilter(window, pressed)
    assert confirmed == ["X-Y"]
    window._pending_contour = None
    finish_window(window, mask)


def test_up_and_down_step_the_last_clicked_spatial_slice():
    image_data = np.zeros((9, 8, 7, 2), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.isActiveWindow = lambda: True
    window.cursor = [4, 3, 2, 1]
    window.refresh_views = lambda: None
    up = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Up,
        Qt.KeyboardModifier.NoModifier,
    )
    down = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Down,
        Qt.KeyboardModifier.NoModifier,
    )

    window.slice_views["X-Z"].viewActivated.emit("X-Z")
    assert window.eventFilter(window, up)
    assert window.cursor == [4, 4, 2, 1]

    window.slice_views["Y-Z"].viewActivated.emit("Y-Z")
    assert window.eventFilter(window, down)
    assert window.cursor == [3, 4, 2, 1]
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
    assert window.close_all_action in window.file_menu.actions()
    assert window.close_all_action.shortcut().toString() == "Ctrl+W"
    assert not hasattr(window, "open_image_action")
    assert not hasattr(window, "open_mask_action")
    finish_window(window, mask)


def test_close_all_files_clears_the_complete_workspace(monkeypatch):
    image_data = np.ones((5, 4, 3, 2), dtype=np.float32)
    window, image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.refresh_views(update_3d=True)
    mask.dirty = True
    window._applied_threshold_mask = np.ones(image_data.shape, dtype=bool)
    window._applied_threshold_image = image
    window._undo_stack.append(object())
    window._redo_stack.append(object())
    window._last_clicked_slice_plane = "X-Z"
    monkeypatch.setattr(
        "spatiotemporal_labeler.ui.main_window.QMessageBox.warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Discard,
    )

    assert window.close_all_files()

    assert window.images == []
    assert window.masks == []
    assert window.image_combo.count() == 0
    assert window.mask_combo.count() == 0
    assert window._mask_labels == {}
    assert window._image_levels == {}
    assert window._applied_threshold_mask is None
    assert window._applied_threshold_image is None
    assert window._undo_stack == []
    assert window._redo_stack == []
    assert window.cursor == [0, 0, 0, 0]
    assert window._last_clicked_slice_plane is None
    assert not window.close_all_action.isEnabled()
    assert window.axis_slider.maximum() == 0
    assert window.label_panel.list_widget.count() == 0
    assert all(view.image_item.image is None for view in window.slice_views.values())
    assert window.temporal_view.image_item.image is None
    assert window.viewer_3d._surface_state is None
    assert window.viewer_3d._cursor_state is None

    mask.dirty = False
    window.images.append(image)
    window.image_combo.addItem("reloaded image")
    window.masks.append(mask)
    window._mask_labels[id(mask)] = {1: default_label(1)}
    window.mask_combo.addItem("reloaded labels")
    window.refresh_views()
    assert all(view.image_item.image is not None for view in window.slice_views.values())
    assert window.temporal_view.image_item.image is not None
    assert window.close_all_action.isEnabled()
    finish_window(window, mask)


def test_close_all_files_cancel_preserves_unsaved_workspace(monkeypatch):
    image_data = np.ones((3, 3, 1, 1), dtype=np.float32)
    window, image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    mask.dirty = True
    monkeypatch.setattr(
        "spatiotemporal_labeler.ui.main_window.QMessageBox.warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Cancel,
    )

    assert not window.close_all_files()
    assert window.images == [image]
    assert window.masks == [mask]
    assert window.image_combo.count() == 1
    assert window.mask_combo.count() == 1
    mask.dirty = False
    finish_window(window, mask)


def test_close_all_files_saves_dirty_masks_before_reset(monkeypatch):
    image_data = np.ones((3, 3, 1, 1), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    mask.dirty = True
    saved = []
    monkeypatch.setattr(
        "spatiotemporal_labeler.ui.main_window.QMessageBox.warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Save,
    )
    window._save_specific_mask = lambda candidate: saved.append(candidate) or True

    assert window.close_all_files()
    assert saved == [mask]
    assert window.images == []
    assert window.masks == []
    mask.dirty = False
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
    window.threshold_dock.show()
    window.threshold_panel.set_bounds(0.0, 10.0)
    before = int(window._threshold_selection().sum())

    window.threshold_panel.lower_slider.setValue(500)

    after = int(window._threshold_selection().sum())
    preview = window.slice_views["X-Y"].threshold_item.image
    assert after < before
    assert preview is not None
    assert np.any(preview[..., 3] > 0)

    full_refreshes = []
    window.refresh_views = lambda *args, **kwargs: full_refreshes.append(None)
    old_levels = window._levels
    window.window_level_panel.level_control.slider.setValue(750)

    assert window._levels != old_levels
    assert full_refreshes == []
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
    assert 0.0 < panel.lower.value() < 100.0
    assert panel.upper.value() == 100.0
    lower, upper = panel.intensity_bounds
    assert 0.0 < lower < 10.0
    assert upper == 10.0
    finish_window(window, mask)


def test_threshold_percentages_preserve_tiny_intensity_ranges():
    image_data = np.linspace(1.23456789, 1.23456799, 16, dtype=np.float64).reshape(
        (4, 4, 1, 1)
    )
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    panel = window.threshold_panel

    panel.lower_slider.setValue(375)
    lower, upper = panel.intensity_bounds

    expected = image_data.min() + 0.375 * (image_data.max() - image_data.min())
    assert panel.lower.value() == 37.5
    assert panel.upper.value() == 100.0
    assert np.isclose(lower, expected, rtol=0.0, atol=1e-14)
    assert upper == image_data.max()
    finish_window(window, mask)


def test_local_threshold_preview_only_computes_the_current_frame():
    image_data = np.linspace(0.0, 10.0, 7 * 7 * 3 * 4, dtype=np.float32).reshape(
        (7, 7, 3, 4)
    )
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    panel = window.threshold_panel
    window.threshold_dock.show()
    panel.method.blockSignals(True)
    panel.method.setCurrentIndex(panel.method.findData("local_gaussian"))
    panel.method.blockSignals(False)
    window.cursor[3] = 2
    window._invalidate_threshold()
    window._threshold_selection = lambda: (_ for _ in ()).throw(
        AssertionError("live preview requested the full 4D threshold")
    )

    window.refresh_views()

    assert set(window._threshold_frame_cache) == {2}
    assert window.slice_views["X-Y"].threshold_item.image is not None
    finish_window(window, mask)


def test_spatial_thresholded_brush_does_not_request_all_frames():
    image_data = np.ones((7, 7, 2, 3), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.threshold_panel.set_percent_bounds(0.0, 100.0)
    window.cursor = [3, 3, 0, 0]
    window._apply_threshold_mask()
    window._threshold_selection = lambda: (_ for _ in ()).throw(
        AssertionError("spatial brush recomputed an applied threshold mask")
    )

    window._stroke_started("X-Y", 3, 3)
    window._stroke_finished("X-Y", 3, 3)

    assert mask.data[3, 3, 0, 0] == 1
    assert mask.data[3, 3, 0, 0] == 1
    finish_window(window, mask)


def test_middle_drag_window_level_direction_and_preview_plane_follow_shift():
    image_data = np.linspace(0.0, 100.0, 5 * 6 * 7 * 8, dtype=np.float32).reshape(
        (5, 6, 7, 8)
    )
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window._levels = (25.0, 75.0)
    window._image_value_range = (0.0, 100.0)
    window.window_level_panel.set_image_range(0.0, 100.0)

    window._adjust_window_level(20.0, -10.0)

    assert window._levels == (25.0, 85.0)
    assert window.window_level_panel.level.value() == 55.0
    assert window.window_level_panel.width.value() == 60.0

    window._navigate_in_plane("X-Z", 2, 3)
    assert window.image_previews.plane == "X-Z"
    window._temporal_cursor_requested("Y-T", 4, 6)
    assert window.image_previews.plane == "Y-T"
    finish_window(window, mask)


def test_window_levels_follow_each_image_between_main_view_and_preview():
    first_data = np.linspace(0.0, 100.0, 5 * 6 * 7 * 2, dtype=np.float32).reshape(
        (5, 6, 7, 2)
    )
    window, first_image, mask = make_window(
        first_data, np.zeros(first_data.shape, dtype=np.uint8)
    )
    second_image = make_sequence(first_data + 1000.0)
    window.images.append(second_image)
    window.image_combo.addItem("second image")
    window._rebuild_image_previews()
    window._window_level_changed(45.0, 30.0)
    first_levels = (30.0, 60.0)
    second_plot = window.image_previews._plots[1]

    second_plot._adjust_window_level(20.0, -10.0)
    preview_levels = second_plot.levels

    assert window._image_levels[id(first_image)] == first_levels
    assert window._image_levels[id(second_image)] == preview_levels
    assert window.preview_splitter.indexOf(window.view_grid) == 0
    assert window.preview_splitter.indexOf(window.image_previews) == 1

    second_plot.activated.emit(1)

    assert window.active_image is second_image
    assert window._levels == preview_levels
    window._window_level_changed(1100.0, 50.0)
    window.image_combo.setCurrentIndex(0)
    assert window._levels == first_levels
    window.image_combo.setCurrentIndex(1)
    assert window._levels == (1075.0, 1125.0)
    finish_window(window, mask)


def test_reset_shortcut_prefers_the_hovered_image_preview(monkeypatch):
    image_data = np.ones((5, 6, 2, 1), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.shortcuts["reset_view"] = "R"
    window.isActiveWindow = lambda: True
    main_resets = []
    monkeypatch.setattr(
        window.image_previews,
        "reset_hovered_preview",
        lambda _watched=None: True,
    )
    monkeypatch.setattr(window, "_reset_2d_views", lambda: main_resets.append(None))
    pressed = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_R,
        Qt.KeyboardModifier.NoModifier,
    )

    assert window.eventFilter(window, pressed)
    assert main_resets == []
    finish_window(window, mask)


def test_time_navigation_queues_all_visible_labels():
    image_data = np.zeros((5, 5, 2, 3), dtype=np.float32)
    mask_data = np.zeros(image_data.shape, dtype=np.uint8)
    mask_data[1:3, 1:3, :, :] = 1
    mask_data[3:5, 3:5, :, :] = 2
    window, _image, mask = make_window(image_data, mask_data)
    calls = []
    window.viewer_3d.set_mask = lambda frame, **kwargs: calls.append(
        (np.asarray(frame), kwargs)
    )
    window.cursor[3] = 1

    window._refresh_navigation_3d()

    assert len(calls) == 1
    assert set(calls[0][1]["labels"]) == {1, 2}
    assert calls[0][1]["immediate"] is False
    finish_window(window, mask)


def test_3d_rendering_toggle_defaults_on_and_stops_surface_requests():
    image_data = np.zeros((5, 5, 2, 2), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    refreshes = []
    window._refresh_3d = lambda *args, **kwargs: refreshes.append((args, kwargs))

    assert window.render_toggle.isChecked()
    assert window.viewer_3d.rendering_enabled

    window.render_toggle.setChecked(False)
    assert not window.viewer_3d.rendering_enabled
    assert refreshes == []

    window.render_toggle.setChecked(True)
    assert window.viewer_3d.rendering_enabled
    assert len(refreshes) == 1

    window._set_language("zh_CN")
    assert window.render_toggle.text() == "显示"
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
    labels[1].opacity = 0.4

    viewer.set_mask(frame, labels=labels, immediate=True, global_opacity=0.5)

    assert set(viewer.segment_pipelines) == {1, 2}
    first = viewer.segment_pipelines[1]
    second = viewer.segment_pipelines[2]
    assert first.actor is not second.actor
    assert first.mapper.GetScalarVisibility() == 0
    assert first.decimator.GetTargetReduction() > 0.0
    assert np.isclose(first.actor.GetProperty().GetOpacity(), 0.2)
    viewer.close()
    viewer.deleteLater()


def test_3d_viewer_rebuilds_only_dirty_labels(monkeypatch):
    ensure_application()
    viewer = Mask3DViewer()
    frame = np.zeros((8, 8, 8), dtype=np.uint8)
    frame[1:4, 1:4, 1:4] = 1
    frame[5:7, 5:7, 5:7] = 2
    labels = {1: default_label(1), 2: default_label(2)}
    requested = []
    build_meshes = viewer_3d_module._build_surface_meshes

    def record_request(state, values, settings):
        requested.append(set(values))
        return build_meshes(state, values, settings)

    monkeypatch.setattr(viewer_3d_module, "_build_surface_meshes", record_request)
    viewer.set_mask(frame, labels=labels, immediate=True, cache_key=("mask", 0))
    frame[2, 2, 2] = 0
    viewer.set_mask(
        frame,
        labels=labels,
        immediate=True,
        dirty_values={1},
        cache_key=("mask", 0),
    )

    assert requested == [{1, 2}, {1}]
    assert viewer.segment_pipelines[2].actor.GetVisibility()
    viewer.close()
    viewer.deleteLater()


def test_3d_viewer_discards_stale_background_results():
    ensure_application()
    viewer = Mask3DViewer()
    frame = np.zeros((6, 6, 6), dtype=np.uint8)
    frame[1:5, 1:5, 1:5] = 1
    labels = {1: default_label(1)}
    try:
        viewer.set_mask(frame, labels=labels, cache_key=("mask", 0))
        viewer._timer.stop()
        stale_request = viewer._pending
        assert stale_request is not None

        viewer.set_mask(frame, labels=labels, cache_key=("mask", 1))
        viewer._timer.stop()
        current_request = viewer._pending
        assert current_request is not None

        viewer._apply_surface_result(stale_request, {}, 1.0)
        assert viewer._rendered_cache_key is None

        viewer._apply_surface_result(current_request, {}, 1.0)
        assert viewer._rendered_cache_key == ("mask", 1)
    finally:
        viewer.close()
        viewer.deleteLater()


def test_3d_viewer_uses_non_inertial_trackball_interaction():
    ensure_application()
    viewer = Mask3DViewer()

    style = viewer.vtk_widget.GetRenderWindow().GetInteractor().GetInteractorStyle()

    assert style.GetClassName() == "vtkInteractorStyleTrackballCamera"
    assert style.GetUseTimers() == 0
    assert viewer.cursor_renderer.GetInteractive() == 0
    assert style.HasObserver("InteractionEvent")
    viewer.close()
    viewer.deleteLater()


def test_3d_viewer_starts_in_coronal_right_to_left_orientation():
    ensure_application()
    viewer = Mask3DViewer()
    camera = viewer.renderer.GetActiveCamera()
    view_direction = np.asarray(camera.GetDirectionOfProjection())
    view_up = np.asarray(camera.GetViewUp())
    screen_right = np.cross(view_direction, view_up)

    assert np.allclose(view_direction, (0.0, -1.0, 0.0))
    assert np.allclose(view_up, (0.0, 0.0, 1.0))
    assert np.allclose(screen_right, (-1.0, 0.0, 0.0))
    viewer.close()
    viewer.deleteLater()


def test_3d_left_drag_is_reserved_for_lasso_and_alt_left_reaches_rotation():
    ensure_application()
    viewer = Mask3DViewer()

    def mouse_event(
        event_type: QEvent.Type,
        button: Qt.MouseButton,
        buttons: Qt.MouseButton,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> QMouseEvent:
        return QMouseEvent(
            event_type,
            QPointF(8.0, 8.0),
            QPointF(8.0, 8.0),
            QPointF(8.0, 8.0),
            button,
            buttons,
            modifiers,
        )

    plain_press = mouse_event(
        QEvent.Type.MouseButtonPress,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    assert viewer.eventFilter(viewer.vtk_widget, plain_press)

    alt_press = mouse_event(
        QEvent.Type.MouseButtonPress,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.AltModifier,
    )
    assert not viewer.eventFilter(viewer.vtk_widget, alt_press)
    assert viewer._alt_rotate_active
    alt_release = mouse_event(
        QEvent.Type.MouseButtonRelease,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.AltModifier,
    )
    assert not viewer.eventFilter(viewer.vtk_widget, alt_release)
    assert not viewer._alt_rotate_active

    started = []
    finished = []
    viewer.set_lasso_enabled(True)
    viewer.lassoStarted.connect(lambda: started.append(None))
    viewer.lassoFinished.connect(lambda points: finished.append(points))
    assert viewer.eventFilter(viewer.vtk_widget, plain_press)
    plain_release = mouse_event(
        QEvent.Type.MouseButtonRelease,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
    )
    assert viewer.eventFilter(viewer.vtk_widget, plain_release)
    assert started and finished
    assert viewer._lasso_points == []
    viewer.close()
    viewer.deleteLater()


def test_3d_rotation_keeps_visible_surface_inside_the_clipping_range():
    ensure_application()
    viewer = Mask3DViewer()
    frame = np.zeros((20, 24, 28), dtype=np.uint8)
    frame[2:18, 3:21, 4:25] = 1
    viewer.set_mask(
        frame,
        spacing=(1.5, 2.0, 2.5),
        labels={1: default_label(1)},
        immediate=True,
    )
    camera = viewer.renderer.GetActiveCamera()
    camera.Azimuth(137.0)
    camera.Elevation(41.0)
    camera.OrthogonalizeViewUp()

    viewer.interaction_style.InvokeEvent("InteractionEvent")

    bounds = viewer.renderer.ComputeVisiblePropBounds()
    corners = np.asarray(
        [
            (x, y, z)
            for x in bounds[:2]
            for y in bounds[2:4]
            for z in bounds[4:6]
        ]
    )
    camera_position = np.asarray(camera.GetPosition())
    view_direction = np.asarray(camera.GetDirectionOfProjection())
    depths = (corners - camera_position) @ view_direction
    near, far = camera.GetClippingRange()
    assert near < float(np.min(depths))
    assert far > float(np.max(depths))
    viewer.close()
    viewer.deleteLater()


def test_locator_drag_updates_the_shared_spatial_cursor():
    image_data = np.zeros((9, 8, 7, 2), dtype=np.float32)
    window, _image, mask = make_window(
        image_data, np.zeros(image_data.shape, dtype=np.uint8)
    )
    window.cursor = [1, 2, 3, 0]

    window.slice_views["X-Y"].locatorDragged.emit("X", 6)

    assert window.cursor == [6, 2, 3, 0]
    finish_window(window, mask)


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
            lambda data, labels, *args, plane=plane, **kwargs: overlay_refreshes[plane].append(
                np.asarray(data).copy()
            )
        )
    window.temporal_view.set_mask_overlay = (
        lambda data, labels, *args, **kwargs: overlay_refreshes["X-T"].append(
            np.asarray(data).copy()
        )
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
    assert window._stroke_bounds == (slice(1, 12), slice(1, 12), slice(0, 1))
    assert full_refreshes == []
    assert surface_refreshes == []

    window._stroke_finished("X-Y", 8, 8)

    assert {plane: len(refreshes) for plane, refreshes in overlay_refreshes.items()} == {
        plane: 3 for plane in overlay_refreshes
    }
    assert full_refreshes == []
    assert len(surface_refreshes) == 1
    finish_window(window, mask)
