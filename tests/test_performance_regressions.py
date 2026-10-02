import gc
import os
import weakref

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.model import default_label
from spatiotemporal_labeler.ui import MainWindow
from spatiotemporal_labeler.ui import main_window as main_window_module
from spatiotemporal_labeler.ui.image_item import DisplayImageItem
from spatiotemporal_labeler.ui.viewer_3d import Mask3DViewer, _build_surface_meshes


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("SpatioTemporalLabelerTests")
    app.setApplicationName("SpatioTemporalLabelerTests")
    return app


@pytest.fixture
def viewer(application):
    widget = Mask3DViewer()
    yield widget
    widget.close()
    widget.deleteLater()
    application.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def window(application):
    widget = MainWindow()
    yield widget
    for mask in widget.masks:
        mask.dirty = False
    widget.close()
    widget.deleteLater()
    application.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def sequence(data):
    transform = AxisTransform(
        (0, 1, 2, 3), (False, False, False),
        (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3),
    )
    return Sequence4D(data, {}, transform)


def navigation_request(viewer, frame_number, scene="mask"):
    frame = np.zeros((8, 8, 8), dtype=np.uint8)
    frame[frame_number:frame_number + 2, 2:6, 2:6] = 1
    viewer.set_mask(
        frame, labels={1: default_label(1)},
        cache_key=(scene, frame_number), scene_key=scene, interactive=True,
    )


def finish_task(viewer, task):
    request = task.request
    meshes = _build_surface_meshes(request.state, request.values, request.settings)
    viewer._surface_task_finished(request.generation, meshes, 100.0)


def test_continuous_navigation_displays_completed_frames_before_input_stops(viewer, monkeypatch):
    tasks = []
    monkeypatch.setattr(viewer._thread_pool, "start", tasks.append)
    navigation_request(viewer, 1)
    navigation_request(viewer, 2)
    finish_task(viewer, tasks[0])

    assert viewer._rendered_cache_key == ("mask", 1)
    viewer._timer.stop()
    viewer._apply_pending()
    navigation_request(viewer, 3)
    finish_task(viewer, tasks[1])
    assert viewer._rendered_cache_key == ("mask", 2)

    viewer._apply_surface_result(tasks[0].request, {}, 100.0)
    assert viewer._rendered_cache_key == ("mask", 2)
    viewer._timer.stop()
    viewer._apply_pending()
    finish_task(viewer, tasks[2])
    assert viewer._rendered_cache_key == ("mask", 3)


def test_navigation_result_cannot_restore_a_closed_or_replaced_scene(viewer, monkeypatch):
    tasks = []
    monkeypatch.setattr(viewer._thread_pool, "start", tasks.append)
    navigation_request(viewer, 1)
    viewer.set_mask(None)
    navigation_request(viewer, 2, scene="next-case")
    finish_task(viewer, tasks[0])
    assert viewer._rendered_cache_key is None
    assert not viewer.segment_pipelines


def test_navigation_result_cannot_overwrite_a_subsequent_edit(viewer, monkeypatch):
    tasks = []
    monkeypatch.setattr(viewer._thread_pool, "start", tasks.append)
    navigation_request(viewer, 1)
    edited = np.zeros((8, 8, 8), dtype=np.uint8)
    viewer.set_mask(
        edited, labels={1: default_label(1)},
        cache_key=("mask", 1), scene_key="mask", dirty_values={1},
    )
    finish_task(viewer, tasks[0])
    assert viewer._rendered_cache_key is None
    assert not viewer.segment_pipelines


def test_close_releases_meshes_and_actors(viewer):
    frame = np.ones((8, 8, 8), dtype=np.uint8)
    viewer.set_mask(frame, immediate=True)
    pipeline = viewer.segment_pipelines[1]
    assert pipeline.mapper.GetInput().GetNumberOfPoints() > 0

    viewer.set_mask(None)

    assert pipeline.mapper.GetInput().GetNumberOfPoints() == 0
    assert not viewer.segment_pipelines
    assert viewer.renderer.GetActors().GetNumberOfItems() == 0


def test_single_frame_temporal_rendering_and_close_release_source_data(window):
    window.render_toggle.setChecked(False)
    for _ in range(3):
        image = sequence(np.zeros((8, 8, 8, 1), dtype=np.uint8))
        source = weakref.ref(image.data)
        window.images.append(image)
        window.image_combo.addItem("single frame")
        window.temporal_mode.setCurrentText("Z-T")
        window.refresh_views()
        window.temporal_view.image_item.render()
        assert not window.temporal_view.image_item.qimage.isNull()
        del image

        assert window.close_all_files()
        gc.collect()
        assert source() is None
        assert window.temporal_view.image_item.qimage is None


def test_clearing_a_rendered_image_releases_the_sequence_owned_by_qimage(application):
    item = DisplayImageItem(axisOrder="row-major")
    data = np.zeros((4, 8, 16), dtype=np.uint8)
    source = weakref.ref(data)
    item.setImage(data[0], autoLevels=False, levels=(0, 255))
    item.render()
    assert np.shares_memory(item.qimage.data, data)
    del data

    item.clear()
    gc.collect()
    assert source() is None


def test_rebuilding_previews_releases_old_images_before_deferred_deletion(window):
    image = sequence(np.zeros((4, 5, 3, 2), dtype=np.float32))
    source = weakref.ref(image.data)
    strip = window.image_previews
    strip.rebuild([image], active_index=-1)
    strip.update_images([image], (1, 2, 1, 0))
    old_preview = strip._plots[0]
    assert old_preview.item.image is not None
    del image

    strip.rebuild([], active_index=-1)
    gc.collect()
    assert source() is None
    assert old_preview.item.image is None


def test_threshold_constrained_3d_scissors_stream_frames_and_preserve_undo(window, monkeypatch):
    image = sequence(np.zeros((4, 4, 4, 3), dtype=np.float32))
    mask = sequence(np.ones(image.data.shape, dtype=np.uint8))
    window.images.append(image)
    window.image_combo.addItem("image")
    window.masks.append(mask)
    window._mask_labels[id(mask)] = {1: default_label(1)}
    window.mask_combo.addItem("mask")
    window._applied_threshold_image = image
    window._applied_threshold_mask = np.zeros(image.data.shape, dtype=bool)
    window._applied_threshold_mask[1, 1, 1, :] = True
    window.all_frames_toggle.setChecked(True)
    window._set_tool("lasso")
    monkeypatch.setattr(window.viewer_3d, "lasso_voxel_mask", lambda shape, *a, **kw: np.ones(shape, bool))
    threshold_frames = []
    transformed_frames = []
    threshold_frame = window._active_threshold_frame
    transform = main_window_module.transform_selected_labels

    def record_threshold(frame):
        threshold_frames.append(frame)
        return threshold_frame(frame)

    def record_transform(*args):
        assert len(threshold_frames) == len(transformed_frames) + 1
        transformed_frames.append(True)
        return transform(*args)

    monkeypatch.setattr(window, "_active_threshold_frame", record_threshold)
    monkeypatch.setattr(main_window_module, "transform_selected_labels", record_transform)
    window._lasso_3d_started()
    window._lasso_3d_finished([(0, 0), (2, 0), (2, 2)])
    assert threshold_frames == [0, 1, 2]
    assert np.count_nonzero(mask.data == 0) == 3
    assert len(window._undo_stack) == 1
    window.undo()
    assert np.all(mask.data == 1)


@pytest.mark.parametrize("plane", ["X-Y", "X-Z", "Y-Z", "X-T", "Y-T", "Z-T"])
def test_brush_snapshots_only_the_editing_plane_and_preserves_undo(window, plane):
    image = sequence(np.ones((9, 8, 7, 4), dtype=np.float32))
    mask = sequence(np.zeros(image.data.shape, dtype=np.uint8))
    window.images.append(image)
    window.image_combo.addItem("image")
    window.masks.append(mask)
    window._mask_labels[id(mask)] = {1: default_label(1)}
    window.mask_combo.addItem("mask")
    window._set_tool("brush")
    window.brush_diameter.setValue(1.0)
    window.all_frames_toggle.setChecked(True)
    window.cursor = [4, 4, 3, 1]
    if plane.endswith("-T"):
        window.temporal_mode.setCurrentText(plane)
    window._stroke_started(plane, 2, 2)
    assert window._stroke_before.nbytes <= 9 * 8 * 4
    window._stroke_finished(plane, 3, 2)
    painted = mask.data.copy()
    assert np.any(painted)
    assert len(window._undo_stack) == 1
    window.undo()
    assert not np.any(mask.data)
    window.redo()
    assert np.array_equal(mask.data, painted)


def test_loading_images_rebuilds_previews_once_per_activation(window, monkeypatch):
    calls = []
    rebuild = window._rebuild_image_previews

    def record_rebuild():
        calls.append(True)
        rebuild()

    monkeypatch.setattr(window, "_rebuild_image_previews", record_rebuild)
    monkeypatch.setattr(
        Sequence4D, "load", lambda _path: sequence(np.zeros((4, 5, 3, 2), dtype=np.float32)),
    )
    for index in range(3):
        window.load_image(f"image-{index}.nrrd")
    assert len(calls) == 3
    assert len(window.image_previews._plots) == 2
