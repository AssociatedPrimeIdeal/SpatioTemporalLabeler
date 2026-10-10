import os

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.model import default_label
from spatiotemporal_labeler.ui import MainWindow
from spatiotemporal_labeler.ui.label_panel import THRESHOLD_MASK_ITEM


def make_sequence(data):
    transform = AxisTransform(
        original_axis_for_canonical=(0, 1, 2, 3),
        flipped_canonical_axes=(False, False, False),
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_ras=(0.0, 0.0, 0.0),
        direction_ras=np.eye(3),
    )
    return Sequence4D(np.asarray(data), {}, transform)


@pytest.fixture
def current_threshold_window():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    application.setOrganizationName("SpatioTemporalLabelerTests")
    application.setApplicationName("SpatioTemporalLabelerTests")
    windows = []

    def create(mask_data):
        window = MainWindow()
        windows.append(window)
        image = make_sequence(np.ones(mask_data.shape, dtype=np.float32))
        mask = make_sequence(mask_data)
        window.images.append(image)
        window.image_combo.addItem("image")
        window.masks.append(mask)
        window._mask_labels[id(mask)] = {1: default_label(1), 2: default_label(2)}
        window.mask_combo.addItem("mask")
        threshold = np.zeros(mask_data.shape, dtype=bool)
        # Other frames deliberately disagree with the source frame's selection.
        threshold[..., 0] = True
        window._applied_threshold_mask = threshold
        window._applied_threshold_image = image
        window._sync_label_panel()
        window.all_frames_current_threshold_toggle.setChecked(True)
        return window, threshold, mask

    yield create
    for window in windows:
        for mask in window.masks:
            mask.dirty = False
        window.close()
        window.deleteLater()
    application.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("plane,axes", [("X-Y", (0, 1)), ("X-Z", (0, 2)), ("Y-Z", (1, 2))])
@pytest.mark.parametrize("shape", ["round", "square"])
def test_brush_broadcasts_the_source_footprint_and_is_undoable(current_threshold_window, plane, axes, shape):
    mask_data = np.zeros((7, 7, 7, 3), dtype=np.uint8)
    center = (3, 3, 3)
    neighbor = list(center)
    neighbor[axes[0]] += 1
    neighbor = tuple(neighbor)
    # The source is already labeled: broadcasting its delta would miss these.
    mask_data[center + (1,)] = 1
    mask_data[neighbor + (1,)] = 1
    original = mask_data.copy()
    window, threshold, mask = current_threshold_window(mask_data)
    threshold[center + (1,)] = True
    threshold[neighbor + (1,)] = True
    threshold_before = threshold.copy()
    window.cursor = [3, 3, 3, 1]
    window.brush_diameter.setValue(3.0)
    window.brush_shape.setCurrentIndex(window.brush_shape.findData(shape))

    window._stroke_started(plane, 3, 3)
    window._stroke_finished(plane, 3, 3)

    expected = original.copy()
    expected[center + (slice(None),)] = 1
    expected[neighbor + (slice(None),)] = 1
    assert np.array_equal(mask.data, expected)
    assert np.array_equal(threshold, threshold_before)
    assert window._applied_threshold_visible
    assert len(window._undo_stack) == 1
    window.undo()
    assert np.array_equal(mask.data, original)
    window.redo()
    assert np.array_equal(mask.data, expected)


@pytest.mark.parametrize("temporary_erase", [False, True])
def test_eraser_only_removes_the_active_label(current_threshold_window, temporary_erase):
    mask_data = np.ones((7, 7, 1, 3), dtype=np.uint8)
    mask_data[3, 3, 0, 1] = 0
    mask_data[4, 3, 0, :] = 2
    original = mask_data.copy()
    window, threshold, mask = current_threshold_window(mask_data)
    threshold[3:5, 3, 0, 1] = True
    window.cursor = [3, 3, 0, 1]
    window.brush_diameter.setValue(3.0)
    if not temporary_erase:
        window._set_tool("eraser")

    window._stroke_started("X-Y", 3, 3, temporary_erase)
    window._stroke_finished("X-Y", 3, 3, temporary_erase)

    expected = original.copy()
    expected[3, 3, 0, :] = 0
    assert np.array_equal(mask.data, expected)
    assert len(window._undo_stack) == 1
    window.undo()
    assert np.array_equal(mask.data, original)


@pytest.mark.parametrize("tool", ["contour", "lasso"])
def test_polygon_keeps_its_source_frame_and_mode(current_threshold_window, tool):
    mask_data = np.full((7, 7, 1, 3), int(tool == "lasso"), dtype=np.uint8)
    original = mask_data.copy()
    window, threshold, mask = current_threshold_window(mask_data)
    threshold[3, 3, 0, 1] = True
    window.cursor = [3, 3, 0, 1]
    window._set_tool(tool)

    window._stroke_started("X-Y", 2, 2)
    window._stroke_moved("X-Y", 5, 2)
    window._stroke_moved("X-Y", 5, 5)
    window.cursor[3] = 2
    window.all_frames_bypass_toggle.setChecked(True)
    window._stroke_finished("X-Y", 2, 5)
    if tool == "contour":
        assert window._pending_contour.threshold_frame == 1
        window._confirm_contour("X-Y")

    expected = original.copy()
    expected[3, 3, 0, :] = int(tool == "contour")
    assert np.array_equal(mask.data, expected)
    assert len(window._undo_stack) == 1
    window.undo()
    assert np.array_equal(mask.data, original)
    window.redo()
    assert np.array_equal(mask.data, expected)


def test_3d_lasso_keeps_its_source_frame(current_threshold_window):
    mask_data = np.ones((5, 5, 3, 3), dtype=np.uint8)
    original = mask_data.copy()
    window, threshold, mask = current_threshold_window(mask_data)
    threshold[2, 2, 1, 1] = True
    selection = np.zeros(mask_data.shape[:3], dtype=bool)
    selection[1:4, 1:4, :] = True
    window.viewer_3d.lasso_voxel_mask = lambda *args, **kwargs: selection
    window.cursor = [2, 2, 1, 1]
    window._set_tool("lasso")

    window._lasso_3d_started()
    window.cursor[3] = 2
    window.all_frames_bypass_toggle.setChecked(True)
    window._lasso_3d_finished([(1.0, 1.0), (4.0, 1.0), (4.0, 4.0)])

    expected = original.copy()
    expected[2, 2, 1, :] = 0
    assert np.array_equal(mask.data, expected)
    assert len(window._undo_stack) == 1
    assert window._lasso_3d_threshold_frame is None
    window.undo()
    assert np.array_equal(mask.data, original)
    window.redo()
    assert np.array_equal(mask.data, expected)


@pytest.mark.parametrize("inactive_reason", ["missing", "unchecked", "incompatible"])
def test_inactive_threshold_leaves_all_frames_unrestricted(current_threshold_window, inactive_reason):
    window, _threshold, mask = current_threshold_window(
        np.zeros((5, 5, 1, 3), dtype=np.uint8)
    )
    if inactive_reason == "missing":
        window._delete_threshold_mask()
    elif inactive_reason == "unchecked":
        threshold_item = next(
            window.label_panel.list_widget.item(index)
            for index in range(window.label_panel.list_widget.count())
            if window.label_panel.list_widget.item(index).data(Qt.ItemDataRole.UserRole)
            == THRESHOLD_MASK_ITEM
        )
        threshold_item.setCheckState(Qt.CheckState.Unchecked)
    else:
        window._applied_threshold_image = make_sequence(
            np.ones((3, 3, 1, 3), dtype=np.float32)
        )
    window.cursor = [2, 2, 0, 1]
    window.brush_diameter.setValue(1.0)

    window._stroke_started("X-Y", 2, 2)
    window._stroke_finished("X-Y", 2, 2)

    assert np.all(mask.data[2, 2, 0, :] == 1)
    assert np.count_nonzero(mask.data) == 3


def test_each_stroke_selects_its_own_source_frame(current_threshold_window):
    window, threshold, mask = current_threshold_window(
        np.zeros((5, 5, 1, 3), dtype=np.uint8)
    )
    threshold[1, 2, 0, 1] = True
    threshold[3, 2, 0, 2] = True
    window.cursor = [2, 2, 0, 1]
    window.brush_diameter.setValue(1.0)
    window._stroke_started("X-Y", 1, 2)
    # Time navigation while dragging must not change the source threshold.
    window.cursor[3] = 2
    window._stroke_moved("X-Y", 3, 2)
    window._stroke_finished("X-Y", 3, 2)
    assert np.all(mask.data[1, 2, 0, :] == 1)
    assert not np.any(mask.data[3, 2, 0, :])

    window._stroke_started("X-Y", 3, 2)
    window._stroke_finished("X-Y", 3, 2)
    assert np.all(mask.data[3, 2, 0, :] == 1)
    assert len(window._undo_stack) == 2


def test_optional_shortcut_toggles_and_temporarily_overrides_a_mode(current_threshold_window):
    window, threshold, mask = current_threshold_window(
        np.zeros((5, 5, 1, 3), dtype=np.uint8)
    )
    threshold[2, 2, 0, 1] = True
    window.cursor = [2, 2, 0, 1]
    window.brush_diameter.setValue(1.0)
    window.shortcuts["all_frames_current_threshold"] = "D"
    window.isActiveWindow = lambda: True
    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_D, Qt.KeyboardModifier.NoModifier)
    release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_D, Qt.KeyboardModifier.NoModifier)
    window.all_frames_bypass_toggle.setChecked(True)
    assert not window.all_frames_current_threshold_toggle.isChecked()
    assert window.eventFilter(window, press)
    assert window.eventFilter(window, release)
    assert window.all_frames_current_threshold_toggle.isChecked()
    assert not window.all_frames_bypass_toggle.isChecked()

    window.all_frames_bypass_toggle.setChecked(True)
    assert window.eventFilter(window, press)
    window._stroke_started("X-Y", 2, 2)
    assert window.eventFilter(window, release)
    window._stroke_moved("X-Y", 3, 2)
    window._stroke_finished("X-Y", 3, 2)
    assert np.all(mask.data[2, 2, 0, :] == 1)
    assert not np.any(mask.data[3, 2, 0, :])
    assert window.all_frames_bypass_toggle.isChecked()
    assert not window.all_frames_current_threshold_toggle.isChecked()


def test_held_current_frame_bypass_overrides_broadcast(current_threshold_window):
    window, _threshold, mask = current_threshold_window(
        np.zeros((5, 5, 1, 3), dtype=np.uint8)
    )
    window.cursor = [2, 2, 0, 1]
    window.brush_diameter.setValue(1.0)
    window._handle_edit_mode_shortcut("threshold_bypass", True)
    window._stroke_started("X-Y", 2, 2)
    window._handle_edit_mode_shortcut("threshold_bypass", False)
    window._stroke_finished("X-Y", 2, 2)

    assert mask.data[2, 2, 0, 1] == 1
    assert not np.any(mask.data[2, 2, 0, (0, 2)])
    assert window.all_frames_current_threshold_toggle.isChecked()


def test_temporal_view_keeps_per_frame_thresholds(current_threshold_window):
    window, threshold, mask = current_threshold_window(
        np.zeros((5, 1, 1, 3), dtype=np.uint8)
    )
    threshold[..., 0] = False
    threshold[2, 0, 0, 1] = True
    window.cursor = [2, 0, 0, 1]
    window.brush_diameter.setValue(1.0)

    window._stroke_started("X-T", 2, 0)
    window._stroke_finished("X-T", 2, 2)

    assert mask.data[2, 0, 0, 1] == 1
    assert np.count_nonzero(mask.data) == 1


def test_temporal_propagation_keeps_per_frame_thresholds(current_threshold_window):
    window, threshold, mask = current_threshold_window(
        np.zeros((5, 5, 1, 3), dtype=np.uint8)
    )
    threshold[..., 0] = False
    threshold[2, 2, 0, 1] = True
    window.cursor = [2, 2, 0, 1]
    window.brush_diameter.setValue(1.0)
    window._set_tool("grow")

    window._stroke_started("X-Y", 2, 2)
    window._stroke_finished("X-Y", 2, 2)

    assert mask.data[2, 2, 0, 1] == 1
    assert np.count_nonzero(mask.data) == 1
