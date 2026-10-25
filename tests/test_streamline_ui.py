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
from spatiotemporal_labeler.tools.streamlines import PlanarConfig, StreamlineConfig, project_all_slices
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


def test_planar_cache_and_mutable_mask_snapshot(application, tmp_path, monkeypatch):
    sources = (sequence(10), sequence(0), sequence(0))
    mask = sequence(0, "seed-mask", frames=1)
    mask.data[2:6, 2:5, 1:4] = 2
    config = StreamlineConfig(directions=("R", "A", "H"), seed_count=3,
                              seed_distribution="random", max_steps=4)
    controller = StreamlineController(cache_dir=tmp_path)
    calls = []
    original = jobs.trace_planar_slices
    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(jobs, "trace_planar_slices", tracked)
    try:
        controller.request(sources, config, 0, seed_mask=mask, mask_labels=(2,), planar=PlanarConfig(True, 2))
        mask.data.fill(0)  # A worker must continue using its own immutable snapshot.
        wait_for(application, lambda: len(controller.frames) == 2)
        assert len(calls) == 2
        assert len(controller.frames[0].planar_slices["X-Y"].values) > 0
        assert len(controller.frames[0].speeds) > 0
        mask.data[2:6, 2:5, 1:4] = 2
        controller.clear()
        controller.request(sources, config, 0, seed_mask=mask, mask_labels=(2,), planar=PlanarConfig(True, 2))
        wait_for(application, lambda: len(controller.frames) == 2)
        assert len(calls) == 2
        mask.data.fill(0)
        controller.request(sources, config, 0, seed_mask=mask, mask_labels=(2,),
                           mask_revision=1, planar=PlanarConfig(True, 2))
        wait_for(application, lambda: len(controller.frames) == 2)
        assert len(controller.frames[0].speeds) == 0
        assert len(controller.frames[0].planar_slices["X-Y"].values) == 0
    finally:
        controller.shutdown()


def test_panel_selects_named_defaults_and_preserves_manual_inputs(application, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    panel = StreamlinePanel(settings)
    sources = [sequence(1, name) for name in ("flow_z", "flow_x", "flow_y", "custom")]
    panel.set_sources(sources)
    assert panel.selected_sources() == (sources[1], sources[2], sources[0])
    assert panel.config().directions == ("F", "P", "L")
    assert panel.config().seed_distribution == "grid"
    assert panel.config().grid_spacing_voxels == 3
    assert not panel.config().limit_grid_seeds
    assert panel.seed_count.maximum() == 1000000
    assert panel.tabs.count() == 5
    assert panel.units.currentIndex() == 0
    assert panel.speed_unit() == "source units"
    panel.sources[0].setCurrentIndex(panel.sources[0].findData(id(sources[3])))
    panel.set_sources(sources + [sequence(7, "extra")])
    assert panel.selected_sources()[0] is sources[3]
    panel.set_language("zh_CN")
    assert panel.enabled.text() == "Enable streamline overlays"
    panel.units.setCurrentIndex(3)
    with pytest.raises(ValueError, match="positive VENC"):
        panel.config().validate()
    for spin, venc in zip(panel.venc, (100, 150, 200)):
        spin.setValue(venc)
    panel.config().validate()
    np.testing.assert_allclose(panel.config().component_scales, np.array([100, 150, 200]) / np.pi)
    panel.deleteLater()


def test_legacy_language_setting_migrates_to_english_without_language_controls(application):
    settings = QSettings()
    previous = settings.value("language", None)
    settings.setValue("language", "zh_CN")
    window = MainWindow()
    try:
        assert window.language == "en"
        assert settings.value("language") == "en"
        assert window.image_source_label.text() == "Display image"
        assert not hasattr(window, "language_menu")
        window._set_language("zh_CN")
        assert window.language == "en"
        assert window.streamline_action.text() == "Streamlines"
    finally:
        window.close()
        window.deleteLater()
        if previous is None:
            settings.remove("language")
        else:
            settings.setValue("language", previous)


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
        panel.display_min_speed.set_value(100)
        panel._display_changed()
        assert window.viewer_3d._streamline_actor.GetMapper().GetInput().GetNumberOfLines() == 0
        panel.display_min_speed.set_value(0)
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


def test_planar_ui_navigation_uses_cached_slices_and_mask_edits_invalidate(application, tmp_path, monkeypatch):
    window = MainWindow()
    window.streamlines.cache_dir = tmp_path
    sources = [sequence(3, name) for name in ("flow_x", "flow_y", "flow_z")]
    window.images.extend(sources)
    for source in sources:
        window.image_combo.addItem(source.display_name)
    mask = sequence(1, "seg")
    window.masks.append(mask)
    window.mask_combo.addItem(mask.display_name)
    panel = window.streamline_panel
    panel.set_sources(sources)
    panel.seed_mask.setCurrentIndex(panel.seed_mask.findData(id(mask)))
    panel.distribution.setCurrentIndex(panel.distribution.findData("random"))
    panel.seed_count.setValue(3)
    panel.planar_seeds.setValue(3)
    panel.max_steps.setValue(3)
    panel.mode_2d.setCurrentIndex(panel.mode_2d.findData("planar"))
    panel.enabled.setChecked(True)
    window._compute_streamlines()
    try:
        wait_for(application, lambda: len(window.streamlines.frames) == 2)
        assert window.streamlines.frames[0].planar_slices is not None
        def no_work(*_args, **_kwargs):
            raise AssertionError("Slice navigation started planar calculation")
        monkeypatch.setattr(window.streamlines, "request", no_work)
        monkeypatch.setattr(jobs, "trace_planar_slices", no_work)
        signature = window.streamlines.signature
        panel.mode_2d.setCurrentIndex(panel.mode_2d.findData("projection"))
        panel.mode_2d.setCurrentIndex(panel.mode_2d.findData("planar"))
        assert window.streamlines.signature == signature
        assert not window._streamline_timer.isActive()
        for z in range(sources[0].shape_xyz[2]):
            window.cursor[2] = z
            window.refresh_views()
        previous = window._streamline_mask_revision
        changed = []
        monkeypatch.setattr(window, "_streamline_parameters_changed", lambda: changed.append(True))
        window.viewer_3d.set_rendering_enabled(False)
        window._refresh_3d(dirty_values={1})
        assert changed == [True]
        assert window._streamline_mask_revision == previous + 1
        window._refresh_3d()
        assert changed == [True]
    finally:
        window.close()
        window.deleteLater()
