"""Cancelable all-frame precomputation with content-addressed disk caching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import tempfile
from threading import Event
import weakref
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED
from zlib import error as ZlibError

import numpy as np
from PySide6.QtCore import QObject, QStandardPaths, QTimer, Signal

from spatiotemporal_labeler.tools.streamlines import (
    PLANES, PlanarConfig, SliceSegments, StreamlineCancelled, StreamlineConfig, StreamlineFrame,
    project_all_slices, trace_streamlines, trace_planar_slices, validate_sources,
)


CACHE_VERSION = 3
DISK_BUDGET = 2 * 1024**3


def source_digest(source, cancelled) -> str:
    """Hash actual loaded samples, including noncontiguous/reoriented arrays.

    File timestamps alone are insufficient: a file can change while an older
    copy is still loaded. Hashing is done once per loaded sequence, off-thread.
    """
    digest = hashlib.sha256()
    metadata = (source.data.shape, str(source.data.dtype), source.spacing_xyz,
                source.transform.origin_ras, np.asarray(source.transform.direction_ras).tolist())
    digest.update(json.dumps(metadata).encode())
    iterator = np.nditer(source.data, flags=["external_loop", "buffered", "zerosize_ok"],
                         op_flags=["readonly"], order="C", buffersize=262144)
    for chunk in iterator:
        if cancelled():
            raise StreamlineCancelled()
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def _key(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _atomic_save(path: Path, **arrays) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            # Level 1 keeps NPZ interoperability while reducing background
            # cache-write latency; the result arrays themselves are lossless.
            with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
                for name, values in arrays.items():
                    with archive.open(name + ".npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, np.asanyarray(values), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_frame(path: Path, frame: StreamlineFrame) -> None:
    arrays = {"world": frame.world_segments, "speed": frame.speeds}
    for plane, value in frame.slices.items():
        arrays[plane] = value.values
        arrays[plane + "_offsets"] = value.offsets
    _atomic_save(path, **arrays)


def load_frame(path: Path, shape) -> StreamlineFrame:
    with np.load(path, allow_pickle=False) as saved:
        world, speeds = saved["world"], saved["speed"]
        if world.shape != (len(speeds), 2, 3) or speeds.ndim != 1:
            raise ValueError("Invalid cached streamline geometry")
        slices = {}
        for plane, (_, _, fixed) in PLANES.items():
            values, offsets = saved[plane], saved[plane + "_offsets"]
            if (values.ndim != 2 or values.shape[1] != 5
                    or offsets.shape != (shape[fixed] + 1,)
                    or offsets.dtype.kind not in "iu" or offsets[0] != 0
                    or offsets[-1] != len(values) or np.any(np.diff(offsets) < 0)):
                raise ValueError("Invalid cached slice index")
            slices[plane] = SliceSegments(values, offsets)
    return StreamlineFrame(world, speeds, slices)


def save_slice_sets(path: Path, slices) -> None:
    arrays = {}
    for plane, value in slices.items():
        arrays[plane] = value.values
        arrays[plane + "_offsets"] = value.offsets
    _atomic_save(path, **arrays)


def load_slice_sets(path: Path, shape):
    result = {}
    with np.load(path, allow_pickle=False) as saved:
        for plane, (_, _, fixed) in PLANES.items():
            values, offsets = saved[plane], saved[plane + "_offsets"]
            if (values.ndim != 2 or values.shape[1] != 5
                    or offsets.shape != (shape[fixed] + 1,)
                    or offsets.dtype.kind not in "iu" or offsets[0] != 0
                    or offsets[-1] != len(values) or np.any(np.diff(offsets) < 0)):
                raise ValueError("Invalid cached planar slice index")
            result[plane] = SliceSegments(values, offsets)
    return result


class StreamlineController(QObject):
    framesChanged = Signal()
    progressChanged = Signal(int, int, str)

    def __init__(self, parent=None, cache_dir: Path | None = None) -> None:
        super().__init__(parent)
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
        self.cache_dir = cache_dir or Path(base) / "streamlines-v1"
        self.frames: dict[int, StreamlineFrame] = {}
        self.reference = None
        self.frame_count = 0
        self.signature = None
        self._generation = 0
        self._cancel = Event()
        self._queue = Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streamlines")
        self._digests: dict[int, tuple[object, str]] = {}
        self._closed = False
        self._future = None
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._poll)

    @property
    def nbytes(self) -> int:
        return sum(frame.nbytes for frame in self.frames.values())

    def cancel(self, clear: bool = False) -> None:
        self._generation += 1
        self._cancel.set()
        if self._future is not None:
            self._future.cancel()
        self.signature = None
        if clear:
            self.frames.clear()
            self.reference = None
            self.frame_count = 0
            self._digests.clear()
            self.framesChanged.emit()

    def request(self, sources, config: StreamlineConfig, slab_mm: float,
                current_frame: int = 0, force: bool = False, *,
                seed_mask=None, mask_labels=(), mask_revision=0,
                planar: PlanarConfig = PlanarConfig()) -> None:
        if self._closed:
            return
        config.validate()
        planar.validate()
        count = validate_sources(sources)
        if seed_mask is not None:
            if (not sources[0].spatially_compatible_with(seed_mask)
                    or seed_mask.frame_count not in (1, count)):
                raise ValueError("Seed mask must share the flow grid and frame count (or be 3D).")
        signature = (tuple(id(source) for source in sources), config, float(slab_mm),
                     id(seed_mask), tuple(mask_labels), mask_revision, planar)
        if signature == self.signature and not force:
            return
        self.cancel()
        self.signature = signature
        self.reference = sources[0]
        self.frame_count = count
        self.frames.clear()
        self.framesChanged.emit()
        self._cancel = Event()
        generation, cancel = self._generation, self._cancel
        # Snapshot mutable labels once per parameter/edit request. Workers never
        # read live masks while brush gestures or undo modify their arrays.
        mask_snapshot = None
        if seed_mask is not None:
            mask_snapshot = (np.isin(seed_mask.data, mask_labels) if mask_labels
                             else seed_mask.data > 0)
        self.progressChanged.emit(0, count, "")
        order = [int(np.clip(current_frame, 0, count - 1))]
        order += [frame for frame in range(count) if frame != order[0]]
        self._future = self._executor.submit(
            self._compute, generation, cancel, tuple(sources), config, float(slab_mm),
            order, force, mask_snapshot, planar,
        )
        self._timer.start()

    def _compute(self, generation, cancel, sources, config, slab_mm, order, force,
                 mask_snapshot, planar):
        cancelled = cancel.is_set
        disk_warning = ""
        try:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                disk_available = True
            except OSError as error:
                disk_available = False
                disk_warning = str(error)
            digests = []
            for source in sources:
                cached = self._digests.get(id(source))
                if cached is None or cached[0]() is not source:
                    digest = source_digest(source, cancelled)
                    if cancelled():
                        raise StreamlineCancelled()
                    self._digests[id(source)] = (weakref.ref(source), digest)
                digests.append(self._digests[id(source)][1])
            mask_key = (None if mask_snapshot is None else
                        hashlib.sha256(memoryview(np.ascontiguousarray(mask_snapshot))).hexdigest())
            trace_key = _key((CACHE_VERSION, digests, mask_key, asdict(config)))
            view_key = _key((trace_key, slab_mm))
            planar_key = _key((CACHE_VERSION, "planar", digests, mask_key,
                               asdict(replace(config, seed_count=planar.seed_count))))
            reference = sources[0]
            for frame_index in order:
                if cancelled():
                    raise StreamlineCancelled()
                view_path = self.cache_dir / f"view_{view_key}_{frame_index}.npz"
                trace_path = self.cache_dir / f"trace_{trace_key}_{frame_index}.npz"
                planar_path = self.cache_dir / f"planar_{planar_key}_{frame_index}.npz"
                seed_frame = (None if mask_snapshot is None else
                              mask_snapshot[..., min(frame_index, mask_snapshot.shape[3] - 1)])
                frame = None
                if disk_available and view_path.exists() and not force:
                    try:
                        frame = load_frame(view_path, reference.shape_xyz)
                    except (OSError, ValueError, KeyError, EOFError, BadZipFile, ZlibError):
                        pass
                if frame is None:
                    segments = speeds = None
                    if disk_available and trace_path.exists() and not force:
                        try:
                            with np.load(trace_path, allow_pickle=False) as saved:
                                segments, speeds = saved["segments"], saved["speeds"]
                                if segments.shape != (len(speeds), 2, 3) or speeds.ndim != 1:
                                    segments = speeds = None
                        except (OSError, ValueError, KeyError, EOFError, BadZipFile, ZlibError):
                            pass
                    if segments is None:
                        components = [source.data[..., min(frame_index, source.frame_count - 1)]
                                      for source in sources]
                        segments, speeds = trace_streamlines(
                            components, reference.spacing_xyz, reference.transform.direction_ras,
                            config, cancelled, seed_mask=seed_frame,
                        )
                        if disk_available:
                            try:
                                _atomic_save(trace_path, segments=segments, speeds=speeds)
                            except OSError as error:
                                disk_warning = str(error)
                    frame = project_all_slices(
                        segments, speeds, reference.shape_xyz, reference.spacing_xyz,
                        reference.transform.origin_ras, reference.transform.direction_ras,
                        slab_mm, cancelled,
                    )
                    if disk_available:
                        try:
                            save_frame(view_path, frame)
                        except OSError as error:
                            disk_warning = str(error)
                if planar.enabled:
                    planar_slices = None
                    if disk_available and planar_path.exists() and not force:
                        try:
                            planar_slices = load_slice_sets(planar_path, reference.shape_xyz)
                        except (OSError, ValueError, KeyError, EOFError, BadZipFile, ZlibError):
                            pass
                    if planar_slices is None:
                        components = [source.data[..., min(frame_index, source.frame_count - 1)]
                                      for source in sources]
                        planar_slices = trace_planar_slices(
                            components, reference.spacing_xyz, reference.transform.direction_ras,
                            config, planar, cancelled, seed_mask=seed_frame,
                        )
                        if disk_available:
                            try:
                                save_slice_sets(planar_path, planar_slices)
                            except OSError as error:
                                disk_warning = str(error)
                    frame = replace(frame, planar_slices=planar_slices)
                if cancelled():
                    raise StreamlineCancelled()
                self._queue.put((generation, "frame", (frame_index, frame)))
            if disk_available and not cancelled():
                self._prune_disk()
            self._queue.put((generation, "done", disk_warning))
        except StreamlineCancelled:
            pass
        except Exception as error:
            self._queue.put((generation, "error", str(error)))

    def _prune_disk(self) -> None:
        try:
            files = sorted(self.cache_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
            size = sum(path.stat().st_size for path in files)
            for path in files:
                if size <= DISK_BUDGET:
                    break
                size -= path.stat().st_size
                path.unlink(missing_ok=True)
        except OSError:
            pass  # Disk cache maintenance never invalidates completed geometry.

    def _poll(self) -> None:
        changed = False
        while True:
            try:
                generation, kind, value = self._queue.get_nowait()
            except Empty:
                break
            if generation != self._generation or self._closed:
                continue
            if kind == "frame":
                index, frame = value
                self.frames[index] = frame
                changed = True
                self.progressChanged.emit(len(self.frames), self.frame_count, "")
            else:
                self.progressChanged.emit(len(self.frames), self.frame_count, value)
                self._timer.stop()
                if kind == "error":
                    self.signature = None  # A later change/retry may recover.
        if changed:
            self.framesChanged.emit()
        if self._future is not None and self._future.done() and self._queue.empty():
            self._timer.stop()

    def clear(self) -> None:
        self.cancel(clear=True)

    def clear_disk(self) -> None:
        self.clear()
        def remove():
            for path in self.cache_dir.glob("*.npz"):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        self._executor.submit(remove)

    def shutdown(self) -> None:
        self._closed = True
        self.cancel(clear=True)
        self._timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
