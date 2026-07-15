from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import nrrd
import numpy as np
from numpy.typing import NDArray


class SequenceFormatError(ValueError):
    """Raised when a file cannot be interpreted as a spatial 3D+t sequence."""


SUPPORTED_IMAGE_SUFFIXES = (".nrrd", ".nii", ".nii.gz")
NIFTI_LABEL_EXTENSION_PREFIX = "SpatioTemporalLabelerLabels="


def is_supported_image_path(path: str | Path) -> bool:
    return Path(path).name.lower().endswith(SUPPORTED_IMAGE_SUFFIXES)


def _space_to_ras(space: str | None) -> NDArray[np.float64]:
    normalized = (space or "right-anterior-superior").lower().replace("_", "-")
    signs = {
        "right-anterior-superior": (1.0, 1.0, 1.0),
        "left-posterior-superior": (-1.0, -1.0, 1.0),
        "left-anterior-superior": (-1.0, 1.0, 1.0),
        "right-posterior-superior": (1.0, -1.0, 1.0),
        "right-anterior-inferior": (1.0, 1.0, -1.0),
        "left-anterior-inferior": (-1.0, 1.0, -1.0),
        "left-posterior-inferior": (-1.0, -1.0, -1.0),
        "right-posterior-inferior": (1.0, -1.0, -1.0),
    }
    return np.diag(signs.get(normalized, (1.0, 1.0, 1.0)))


def _header_directions(header: Mapping[str, Any], ndim: int) -> list[NDArray[np.float64] | None]:
    raw = header.get("space directions")
    if raw is None:
        return [None] * ndim
    directions: list[NDArray[np.float64] | None] = []
    for value in raw:
        if value is None:
            directions.append(None)
            continue
        vector = np.asarray(value, dtype=float).reshape(-1)
        valid = vector.size == 3 and np.all(np.isfinite(vector)) and np.linalg.norm(vector) > 0
        directions.append(vector if valid else None)
    if len(directions) != ndim:
        raise SequenceFormatError("NRRD 'space directions' count does not match its dimension")
    return directions


def _infer_time_axis(header: Mapping[str, Any], directions: list[NDArray[np.float64] | None]) -> int:
    nonspatial = [axis for axis, vector in enumerate(directions) if vector is None]
    if len(nonspatial) == 1:
        return nonspatial[0]

    kinds = [str(kind).lower() for kind in header.get("kinds", [])]
    candidates = [axis for axis, kind in enumerate(kinds) if kind not in {"domain", "space"}]
    if len(candidates) == 1:
        return candidates[0]
    raise SequenceFormatError(
        "Expected exactly one non-spatial time/list axis; check 'kinds' and 'space directions'"
    )


@dataclass(frozen=True)
class AxisTransform:
    """Reversible mapping between an input NRRD layout and canonical RAS X,Y,Z,T."""

    original_axis_for_canonical: tuple[int, int, int, int]
    flipped_canonical_axes: tuple[bool, bool, bool]
    spacing_xyz: tuple[float, float, float]
    origin_ras: tuple[float, float, float]
    direction_ras: NDArray[np.float64]
    original_ndim: int = 4

    @property
    def time_axis_original(self) -> int:
        return self.original_axis_for_canonical[3]

    def to_canonical(self, data: NDArray[Any]) -> NDArray[Any]:
        if self.original_ndim == 3 and data.ndim == 3:
            data = np.expand_dims(data, axis=self.time_axis_original)
        result = np.transpose(data, self.original_axis_for_canonical)
        for axis, flipped in enumerate(self.flipped_canonical_axes):
            if flipped:
                result = np.flip(result, axis=axis)
        return np.ascontiguousarray(result)

    def to_original(self, data: NDArray[Any]) -> NDArray[Any]:
        result = data
        for axis, flipped in enumerate(self.flipped_canonical_axes):
            if flipped:
                result = np.flip(result, axis=axis)
        inverse_order = tuple(int(axis) for axis in np.argsort(self.original_axis_for_canonical))
        result = np.transpose(result, inverse_order)
        if self.original_ndim == 3:
            result = np.take(result, 0, axis=self.time_axis_original)
        return np.ascontiguousarray(result)

    @classmethod
    def from_header(cls, header: Mapping[str, Any], shape: tuple[int, ...]) -> "AxisTransform":
        original_ndim = len(shape)
        if original_ndim not in (3, 4):
            raise SequenceFormatError(f"Expected a 3D or 4D NRRD, got shape {shape}")

        working_header = copy.deepcopy(dict(header))
        if original_ndim == 3:
            shape = (*shape, 1)
            raw_directions = working_header.get("space directions")
            if raw_directions is not None:
                working_header["space directions"] = [*raw_directions, None]
            kinds = list(working_header.get("kinds", ["domain"] * 3))
            working_header["kinds"] = [*kinds, "list"]
            spacings = list(working_header.get("spacings", [1.0] * 3))
            working_header["spacings"] = [*spacings, np.nan]

        directions_source = _header_directions(working_header, len(shape))
        has_spatial_metadata = sum(vector is not None for vector in directions_source) == 3
        if has_spatial_metadata:
            time_axis = _infer_time_axis(working_header, directions_source)
        else:
            kinds = [str(kind).lower() for kind in working_header.get("kinds", [])]
            candidates = [axis for axis, kind in enumerate(kinds) if kind not in {"domain", "space"}]
            time_axis = candidates[0] if len(candidates) == 1 else 3
            spacings = np.asarray(working_header.get("spacings", np.ones(4)), dtype=float)
            spatial_axes = [axis for axis in range(4) if axis != time_axis]
            directions_source = [None] * 4
            for world_axis, original_axis in enumerate(spatial_axes):
                spacing = spacings[original_axis] if original_axis < len(spacings) else 1.0
                vector = np.zeros(3, dtype=float)
                vector[world_axis] = spacing if np.isfinite(spacing) and spacing > 0 else 1.0
                directions_source[original_axis] = vector

        source_to_ras = _space_to_ras(
            str(working_header.get("space", "right-anterior-superior"))
        )
        directions_ras = [
            None if vector is None else source_to_ras @ vector for vector in directions_source
        ]
        spatial_axes = [axis for axis, vector in enumerate(directions_ras) if vector is not None]
        if len(spatial_axes) != 3:
            raise SequenceFormatError("Expected three spatial axes in the 4D NRRD")

        def alignment(order: tuple[int, int, int]) -> float:
            return sum(
                abs(float(directions_ras[original_axis][world_axis]))
                / float(np.linalg.norm(directions_ras[original_axis]))
                for world_axis, original_axis in enumerate(order)
            )

        spatial_order = max(itertools.permutations(spatial_axes), key=alignment)
        flipped = tuple(
            bool(directions_ras[original_axis][world_axis] < 0)
            for world_axis, original_axis in enumerate(spatial_order)
        )

        origin_source = np.asarray(
            working_header.get("space origin", np.zeros(3)), dtype=float
        )
        origin_ras = source_to_ras @ origin_source
        canonical_vectors: list[NDArray[np.float64]] = []
        for canonical_axis, original_axis in enumerate(spatial_order):
            vector = np.asarray(directions_ras[original_axis], dtype=float)
            if flipped[canonical_axis]:
                origin_ras = origin_ras + (shape[original_axis] - 1) * vector
                vector = -vector
            canonical_vectors.append(vector)

        spacing = tuple(float(np.linalg.norm(vector)) for vector in canonical_vectors)
        direction = np.column_stack(
            [vector / axis_spacing for vector, axis_spacing in zip(canonical_vectors, spacing)]
        )
        return cls(
            original_axis_for_canonical=(*spatial_order, time_axis),
            flipped_canonical_axes=flipped,
            spacing_xyz=spacing,
            origin_ras=tuple(float(value) for value in origin_ras),
            direction_ras=direction,
            original_ndim=original_ndim,
        )


@dataclass(frozen=True)
class NiftiTransform:
    """Reversible mapping between a NIfTI voxel layout and canonical RAS XYZT."""

    to_ras_orientation: NDArray[np.float64]
    to_original_orientation: NDArray[np.float64]
    spacing_xyz: tuple[float, float, float]
    origin_ras: tuple[float, float, float]
    direction_ras: NDArray[np.float64]
    original_ndim: int

    def to_canonical(self, data: NDArray[Any]) -> NDArray[Any]:
        result = nib.orientations.apply_orientation(data, self.to_ras_orientation)
        if self.original_ndim == 3:
            result = result[..., np.newaxis]
        return np.ascontiguousarray(result)

    def to_original(self, data: NDArray[Any]) -> NDArray[Any]:
        result = nib.orientations.apply_orientation(data, self.to_original_orientation)
        if self.original_ndim == 3:
            result = result[..., 0]
        return np.ascontiguousarray(result)

    @classmethod
    def from_affine(
        cls, affine: NDArray[np.float64], shape: tuple[int, ...]
    ) -> "NiftiTransform":
        if len(shape) not in (3, 4):
            raise SequenceFormatError(f"Expected a 3D or 4D NIfTI, got shape {shape}")
        source_orientation = nib.orientations.io_orientation(affine)
        ras_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
        to_ras = nib.orientations.ornt_transform(source_orientation, ras_orientation)
        to_original = nib.orientations.ornt_transform(ras_orientation, source_orientation)
        canonical_affine = affine @ nib.orientations.inv_ornt_aff(to_ras, shape[:3])
        basis = np.asarray(canonical_affine[:3, :3], dtype=float)
        spacing = tuple(float(value) for value in nib.affines.voxel_sizes(canonical_affine))
        direction = np.column_stack(
            [basis[:, axis] / spacing[axis] for axis in range(3)]
        )
        return cls(
            to_ras_orientation=np.asarray(to_ras, dtype=float),
            to_original_orientation=np.asarray(to_original, dtype=float),
            spacing_xyz=spacing,
            origin_ras=tuple(float(value) for value in canonical_affine[:3, 3]),
            direction_ras=direction,
            original_ndim=len(shape),
        )


_NRRD_TYPES: dict[np.dtype[Any], str] = {
    np.dtype("uint8"): "uint8",
    np.dtype("int8"): "int8",
    np.dtype("uint16"): "uint16",
    np.dtype("int16"): "int16",
    np.dtype("uint32"): "uint32",
    np.dtype("int32"): "int32",
    np.dtype("uint64"): "uint64",
    np.dtype("int64"): "int64",
    np.dtype("float32"): "float",
    np.dtype("float64"): "double",
}


@dataclass
class Sequence4D:
    """A 3D+t image in canonical RAS-oriented X,Y,Z,T voxel layout."""

    data: NDArray[Any]
    header: dict[str, Any]
    transform: AxisTransform | NiftiTransform
    path: Path | None = None
    dirty: bool = False
    source_format: str = "nrrd"
    nifti_affine: NDArray[np.float64] | None = None
    nifti_header: Any = None
    nifti_image_class: Any = None
    display_name_hint: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Sequence4D":
        source = Path(path).expanduser().resolve()
        if not is_supported_image_path(source):
            raise SequenceFormatError(f"Unsupported medical image format: {source.name}")
        if source.name.lower().endswith((".nii", ".nii.gz")):
            image = nib.load(str(source))
            data = np.asanyarray(image.dataobj)
            transform = NiftiTransform.from_affine(
                np.asarray(image.affine, dtype=float), data.shape
            )
            metadata: dict[str, Any] = {}
            for extension in image.header.extensions:
                try:
                    content = extension.get_content()
                    text = (
                        content.decode("utf-8", errors="replace")
                        if isinstance(content, bytes)
                        else str(content)
                    )
                except (NotImplementedError, UnicodeError, ValueError):
                    continue
                if text.startswith(NIFTI_LABEL_EXTENSION_PREFIX):
                    metadata["SpatioTemporalLabelerLabels"] = text.removeprefix(
                        NIFTI_LABEL_EXTENSION_PREFIX
                    )
            return cls(
                data=transform.to_canonical(data),
                header=metadata,
                transform=transform,
                path=source,
                source_format="nifti",
                nifti_affine=np.asarray(image.affine, dtype=float).copy(),
                nifti_header=image.header.copy(),
                nifti_image_class=type(image),
            )
        data, header = nrrd.read(str(source), index_order="F")
        transform = AxisTransform.from_header(header, data.shape)
        return cls(
            data=transform.to_canonical(data),
            header=copy.deepcopy(dict(header)),
            transform=transform,
            path=source,
            source_format="nrrd",
        )

    @classmethod
    def blank_mask_from(cls, source: "Sequence4D", dtype: np.dtype[Any] = np.dtype("uint8")) -> "Sequence4D":
        header = copy.deepcopy(source.header)
        header["type"] = _NRRD_TYPES[np.dtype(dtype)]
        header["DataNodeClassName"] = "vtkMRMLLabelMapVolumeNode"
        header["encoding"] = header.get("encoding", "gzip")
        return cls(
            data=np.zeros(source.data.shape, dtype=dtype),
            header=header,
            transform=source.transform,
            path=None,
            dirty=True,
            source_format=source.source_format,
            nifti_affine=(
                None if source.nifti_affine is None else source.nifti_affine.copy()
            ),
            nifti_header=(
                None if source.nifti_header is None else source.nifti_header.copy()
            ),
            nifti_image_class=source.nifti_image_class,
        )

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.data.shape[:3])

    @property
    def frame_count(self) -> int:
        return int(self.data.shape[3])

    @property
    def spacing_xyz(self) -> tuple[float, float, float]:
        return self.transform.spacing_xyz

    @property
    def display_name(self) -> str:
        return self.path.name if self.path else (self.display_name_hint or "Untitled mask")

    def spatially_compatible_with(self, other: "Sequence4D") -> bool:
        return (
            self.shape_xyz == other.shape_xyz
            and np.allclose(self.spacing_xyz, other.spacing_xyz, atol=1e-4)
            and np.allclose(self.transform.origin_ras, other.transform.origin_ras, atol=1e-3)
            and np.allclose(
                self.transform.direction_ras, other.transform.direction_ras, atol=1e-4
            )
        )

    def compatible_with(self, other: "Sequence4D") -> bool:
        return self.frame_count == other.frame_count and self.spatially_compatible_with(other)

    def map_single_frame_to(
        self,
        frame_count: int,
        target_frame: int | None = None,
    ) -> "Sequence4D":
        """Create an unsaved 4D sequence by copying or placing one 3D frame."""
        count = int(frame_count)
        if self.frame_count != 1:
            raise ValueError("Only a single-frame label sequence can be mapped")
        if count < 2:
            raise ValueError("The destination must contain at least two frames")
        if target_frame is not None and not 0 <= int(target_frame) < count:
            raise ValueError(f"Target frame must be between 0 and {count - 1}")

        if target_frame is None:
            data = np.repeat(self.data, count, axis=3)
        else:
            data = np.zeros((*self.shape_xyz, count), dtype=self.data.dtype)
            data[..., int(target_frame)] = self.data[..., 0]

        header = copy.deepcopy(self.header)
        transform = replace(self.transform, original_ndim=4)
        if isinstance(transform, AxisTransform):
            directions = header.get("space directions")
            if directions is not None and len(directions) == 3:
                header["space directions"] = np.vstack(
                    [np.asarray(directions, dtype=float), np.full((1, 3), np.nan)]
                )
            kinds = list(header.get("kinds", ["domain"] * 3))
            if len(kinds) == 3:
                header["kinds"] = [*kinds, "list"]
            spacings = list(header.get("spacings", []))
            if len(spacings) == 3:
                header["spacings"] = [*spacings, np.nan]
            for field in ("thicknesses", "axis mins", "axis maxs"):
                values = list(header.get(field, []))
                if len(values) == 3:
                    header[field] = [*values, np.nan]
            labels = list(header.get("labels", []))
            if len(labels) == 3:
                header["labels"] = [*labels, "time"]
            units = list(header.get("units", []))
            if len(units) == 3:
                header["units"] = [*units, ""]

        source_name = self.path.name if self.path is not None else self.display_name
        return Sequence4D(
            data=np.ascontiguousarray(data),
            header=header,
            transform=transform,
            path=None,
            dirty=True,
            source_format=self.source_format,
            nifti_affine=(
                None if self.nifti_affine is None else self.nifti_affine.copy()
            ),
            nifti_header=(
                None if self.nifti_header is None else self.nifti_header.copy()
            ),
            nifti_image_class=self.nifti_image_class,
            display_name_hint=f"{source_name} (4D)",
        )

    def save(self, path: str | Path | None = None) -> Path:
        destination = Path(path).expanduser().resolve() if path else self.path
        if destination is None:
            raise ValueError("A destination path is required for an untitled sequence")
        destination.parent.mkdir(parents=True, exist_ok=True)

        original = self.transform.to_original(self.data)
        if self.source_format == "nifti":
            if self.nifti_affine is None:
                raise ValueError("A NIfTI affine is required to save this sequence")
            header = None if self.nifti_header is None else self.nifti_header.copy()
            if header is not None:
                retained_extensions = []
                for extension in header.extensions:
                    try:
                        content = extension.get_content()
                        text = (
                            content.decode("utf-8", errors="replace")
                            if isinstance(content, bytes)
                            else str(content)
                        )
                    except (NotImplementedError, UnicodeError, ValueError):
                        text = ""
                    if not text.startswith(NIFTI_LABEL_EXTENSION_PREFIX):
                        retained_extensions.append(extension)
                header.extensions.clear()
                header.extensions.extend(retained_extensions)
                label_metadata = self.header.get("SpatioTemporalLabelerLabels")
                if label_metadata:
                    content = f"{NIFTI_LABEL_EXTENSION_PREFIX}{label_metadata}".encode(
                        "utf-8"
                    )
                    header.extensions.append(nib.nifti1.Nifti1Extension(6, content))
            image_class = self.nifti_image_class or nib.Nifti1Image
            image = image_class(original, self.nifti_affine, header=header)
            image.set_data_dtype(original.dtype)
            nib.save(image, str(destination))
            self.path = destination
            self.dirty = False
            return destination

        header = copy.deepcopy(self.header)
        header.pop("data file", None)
        header["sizes"] = np.asarray(original.shape, dtype=int)
        header["dimension"] = original.ndim
        header["type"] = _NRRD_TYPES.get(original.dtype, str(original.dtype))
        if original.dtype.itemsize == 1:
            header.pop("endian", None)
        nrrd.write(str(destination), original, header=header, index_order="F")
        self.path = destination
        self.header = header
        self.dirty = False
        return destination
