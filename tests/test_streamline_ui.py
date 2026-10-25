import os
os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from spatiotemporal_labeler.io import AxisTransform, Sequence4D
from spatiotemporal_labeler.tools.streamlines import StreamlineConfig, project_all_slices
from spatiotemporal_labeler.ui import MainWindow
from spatiotemporal_labeler.ui import streamline_jobs as jobs
from spatiotemporal_labeler.ui.streamline_jobs import StreamlineController, load_frame, save_frame, source_digest
from spatiotemporal_labeler.ui.streamline_panel import StreamlinePanel
from spatiotemporal_labeler.ui.viewer_3d import Mask3DViewer


@pytest.fixture
def application():
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    app.setOrganizationName("SpatioTemporalLabelerTests")
    app.setApplicationName("SpatioTemporalLabelerTests")
    return app


def sequence(value, name="image", frames=2, origin=(0, 0, 0)):
    transform = AxisTransform((0, 1, 2, 3), (False, False, False),
                              (1, 1, 1), origin, np.eye(3))
    return Sequence4D(np.full((8, 7, 6, frames), value, np.float32), {}, transform,
                      path=Path(name + ".seq.nrrd"))


def wait_for(app, predicate, timeout=15):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Background streamline operation did not finish")


def test_content_hash_uses_loaded_samples_not_filename_and_handles_strides():
    a, b = sequence(3), sequence(4)
    assert source_digest(a, lambda: False) != source_digest(b, lambda: False)
    b.data = a.data[..., ::-1]
    assert source_digest(a, lambda: False) == source_digest(b, lambda: False)


def test_disk_round_trip_preserves_every_slice_and_detects_broken_index(tmp_path):
    segments = np.array([[[1, 2, 1], [6, 2, 3]]], np.float32)
    frame = project_all_slices(segments, [50], (8, 7, 6), (1, 1, 1), (10, 20, 30), np.eye(3))
    path = tmp_path / "frame.npz"
    save_frame(path, frame)
    loaded = load_frame(path, (8, 7, 6))
    np.testing.assert_array_equal(loaded.world_segments, frame.world_segments)
    for plane in frame.slices:
        np.testing.assert_array_equal(loaded.slices[plane].values, frame.slices[plane].values)
        np.testing.assert_array_equal(loaded.slices[plane].offsets, frame.slices[plane].offsets)
    with pytest.raises(ValueError):
        load_frame(path, (9, 7, 6))


def test_all_frames_precomputed_current_first_and_disk_reused_for_slab_change(application, tmp_path, monkeypatch):
    sources = (sequence(10), sequence(5), sequence(20))
    controller = StreamlineController(cache_dir=tmp_path)
    config = StreamlineConfig(seed_count=8, max_steps=10)
    arrivals = []
    controller.framesChanged.connect(lambda: arrivals.append(tuple(controller.frames)))
    original = jobs.trace_streamlines
    traces = []
    def traced(*args, **kwargs):
        traces.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(jobs, "trace_streamlines", traced)
    try:
        controller.request(sources, config, 0, current_frame=1)
        wait_for(application, lambda: len(controller.frames) == 2)
        assert arrivals[1][0] == 1
        assert len(traces) == 2
        cached = controller.frames[0].world_segments.copy()
        controller.clear()
        controller.request(sources, config, 0)
        wait_for(application, lambda: len(controller.frames) == 2)
        assert len(traces) == 2
        np.testing.assert_array_equal(controller.frames[0].world_segments, cached)
        controller.request(sources, config, 4)
        wait_for(application, lambda: len(controller.frames) == 2)
        assert len(traces) == 2  # Changing a slab only reprojects cached traces.
        # Corrupt a saved projection; its intact trace still avoids integration.
        for path in tmp_path.glob("view_*.npz"):
            path.write_bytes(b"not a numpy archive")
        controller.clear()
        controller.request(sources, config, 4)
        wait_for(application, lambda: len(controller.frames) == 2)
        assert len(traces) == 2
        controller.request(sources, replace(config, step_mm=0.5), 4)
        wait_for(application, lambda: len(controller.frames) == 2)
        assert len(traces) == 4
    finally:
        controller.shutdown()


def test_stale_generation_never_updates_current_frames(application, tmp_path):
    controller = StreamlineController(cache_dir=tmp_path)
    old_generation = controller._generation
    controller.clear()
    fake = project_all_slices(np.empty((0, 2, 3)), [], (2, 2, 2), (1, 1, 1), (0, 0, 0), np.eye(3))
    controller._queue.put((old_generation, "frame", (0, fake)))
    controller._queue.put((old_generation, "error", "stale error"))
    controller._poll()
    assert controller.frames == {} and controller.reference is None
    controller.shutdown()


def test_panel_selects_named_defaults_and_preserves_manual_inputs(application, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    panel = StreamlinePanel(settings)
    sources = [sequence(1, name) for name in ("flow_z", "flow_x", "flow_y", "custom")]
    panel.set_sources(sources)
    assert panel.selected_sources() == (sources[1], sources[2], sources[0])
    assert panel.config().directions == ("F", "P", "L")
    assert panel.units.currentIndex() == 0
    assert panel.speed_unit() == "source units"
    panel.sources[0].setCurrentIndex(panel.sources[0].findData(id(sources[3])))
    panel.set_sources(sources + [sequence(7, "extra")])
    assert panel.selected_sources()[0] is sources[3]
    panel.set_language("zh_CN")
    assert panel.enabled.text() == "启用流线叠加"
    panel.units.setCurrentIndex(3)
    with pytest.raises(ValueError, match="positive VENC"):
        panel.config().validate()
    for spin, venc in zip(panel.venc, (100, 150, 200)):
        spin.setValue(venc)
    panel.config().validate()
    np.testing.assert_allclose(panel.config().component_scales, np.array([100, 150, 200]) / np.pi)
    panel.deleteLater()


def test_vtk_geometry_is_ras_and_visibility_does_not_depend_on_labels(application):
    viewer = Mask3DViewer()
    frame = project_all_slices(np.array([[[1, 2, 1], [4, 2, 1]]]), [20], (8, 7, 6),
                               (1, 1, 1), (10, 20, 30), np.eye(3))
    viewer.set_streamlines(frame, key="frame", scene_key="source", render=False)
    actor = viewer._streamline_actor
    np.testing.assert_allclose(actor.GetMapper().GetInput().GetPoint(0), [11, 22, 31])
    viewer.set_rendering_enabled(False)
    assert not actor.GetVisibility()
    viewer.set_rendering_enabled(True)
    assert actor.GetVisibility()
    camera = viewer.capture_camera()
    viewer.set_mask(None)
    viewer.set_streamlines(frame, key="frame", scene_key="source", render=False)
    assert viewer.renderer.GetActiveCamera().GetPosition() == camera.GetPosition()
    viewer.close()
    assert viewer._streamline_actor is None
    viewer.deleteLater()


def test_navigation_and_display_controls_never_request_integration(application, tmp_path, monkeypatch):
    window = MainWindow()
    window.streamlines.cache_dir = tmp_path
    sources = [sequence(15, name) for name in ("flow_x", "flow_y", "flow_z")]
    background = sequence(100, "pcmra")
    window.images.extend(sources + [background])
    for source in window.images:
        window.image_combo.addItem(source.display_name)
    window.image_combo.setCurrentIndex(3)
    panel = window.streamline_panel
    panel.set_sources(window.images)
    panel.seed_count.setValue(8)
    panel.max_steps.setValue(10)
    panel.enabled.setChecked(True)
    window._compute_streamlines()
    try:
        wait_for(application, lambda: len(window.streamlines.frames) == 2)
        def no_integration(*_args, **_kwargs):
            raise AssertionError("Navigation requested streamline computation")
        monkeypatch.setattr(window.streamlines, "request", no_integration)
        original_geometry = window.viewer_3d._streamline_actor.GetMapper().GetInput()
        for z in range(background.shape_xyz[2]):
            window.cursor[2] = z
            window.refresh_views()
        assert window.viewer_3d._streamline_actor.GetMapper().GetInput() is original_geometry
        assert window.slice_views["X-Y"].streamline_overlay.acceptedMouseButtons() == Qt.MouseButton.NoButton
        panel.width.set_value(2.5)
        panel._display_changed()
        window.image_combo.setCurrentIndex(0)
        window.cursor[3] = 1
        window.refresh_views()
        assert window.viewer_3d._streamline_actor is not None
        incompatible = sequence(1, "other-grid", origin=(10, 0, 0))
        window.images.append(incompatible)
        window.image_combo.addItem(incompatible.display_name)
        window.image_combo.setCurrentIndex(4)
        assert window.viewer_3d._streamline_actor is None
        assert not window.slice_views["X-Y"].streamline_overlay.isVisible()
        assert window.close_all_files()
        assert window.streamlines.frames == {}
        assert panel.selected_sources() == (None, None, None)
    finally:
        window.close()
        window.deleteLater()
