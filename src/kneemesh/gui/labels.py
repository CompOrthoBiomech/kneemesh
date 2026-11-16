"""The GUI's imported labels: reading a LabelMap's values, numbering them, and composing the included ones into a Segmentation.

A label entry is a plain dict so it can live in trame's state: ``id``, the
``path`` of its LabelMap and its file ``name``, the ``file_value`` it has there, the ``structure``
it names (or None), the ``value`` it takes in the composed segmentation, and
whether to ``include`` it.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import SimpleITK as sitk

from kneemesh.config import BONES, STRUCTURES, Structure
from kneemesh.disk import read_label_map, segmentation_labels_path
from kneemesh.segmentation import Interpolator, Segmentation, assemble_label_maps
from kneemesh.vol import extract_isocontour, signed_distance_field, smooth_isocontour

# Background voxels kept around a label cropped for its preview surface, enough for the distance field's Gaussian.
_PREVIEW_PADDING = 4
_PREVIEW_SMOOTHING_ITERATIONS = 20


class LabelEntry(TypedDict):
    """One value of an imported LabelMap, as the GUI's state holds it."""

    id: int
    path: str
    name: str
    file_value: int
    structure: Structure | None
    value: int
    include: bool


def label_values(image: sitk.Image) -> list[int]:
    """The nonzero values in a LabelMap, ascending."""
    return [int(value) for value in np.unique(sitk.GetArrayViewFromImage(image)) if value != 0]


def file_structures(path: str | Path) -> dict[int, Structure]:
    """The structure of each value in a LabelMap that kneemesh wrote, from <stem>_labels.json beside it, or by the file's name.

    A file named for a structure, such as tibia.nii, names its only value
    when it has one; the caller decides that. Unknown names are dropped.
    """
    labels_path = segmentation_labels_path(path)
    if not labels_path.is_file():
        return {}
    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(labels, dict):
        return {}
    return {int(value): name for name, value in labels.items() if name in STRUCTURES and isinstance(value, int)}


def structure_of_name(path: str | Path) -> Structure | None:
    """The structure a file is named for, such as tibia for tibia.nii, or None."""
    stem = Path(path).name.split(".")[0]
    return next((structure for structure in STRUCTURES if structure == stem), None)


def new_entries(path: str | Path, values: Sequence[int], existing: Iterable[LabelEntry]) -> list[LabelEntry]:
    """Entries for every value of a newly imported LabelMap, numbered consistently with the existing entries.

    A LabelMap of one value takes the next free label value. One of several
    values keeps its own value where that is free, and otherwise also takes
    the next free one. Structures come from a <stem>_labels.json beside the
    file, or for a single value from the file's name, and are left unset
    where those name none or a structure another entry already names.
    """
    existing = list(existing)
    used_values = {entry["value"] for entry in existing}
    used_structures = {entry["structure"] for entry in existing if entry["structure"] is not None}
    next_id = max((entry["id"] for entry in existing), default=0) + 1
    named: dict[int, Structure] = file_structures(path)
    if len(values) == 1 and not named and (structure := structure_of_name(path)) is not None:
        named = {values[0]: structure}

    entries: list[LabelEntry] = []
    for file_value in values:
        if len(values) > 1 and file_value not in used_values:
            value = file_value
        else:
            value = next_free_value(used_values)
        used_values.add(value)
        structure: Structure | None = named.get(file_value)
        if structure in used_structures:
            structure = None
        if structure is not None:
            used_structures.add(structure)
        entries.append(
            LabelEntry(
                id=next_id,
                path=str(path),
                name=Path(path).name,
                file_value=file_value,
                structure=structure,
                value=value,
                include=structure is not None,
            )
        )
        next_id += 1
    return entries


def config_entries(
    label_map: str | Path | None,
    labels: Mapping[Structure, int] | None,
    label_maps: Mapping[Structure, str | Path] | None,
    values_of: Callable[[str], Sequence[int]],
) -> list[LabelEntry]:
    """Entries for the segmentation a LabelMapBrep or LabelMapAssembly configuration reads.

    label_map's values keep their own value, the ones labels names included
    under their structures, any others excluded and unassigned. Each of
    label_maps is numbered 1, 2, ... in order, as kneemesh-brep assembles
    them; should one hold several values, its first takes the structure and
    the others are excluded.

    Args:
        label_map: A multilabel LabelMap, with labels naming its values.
        labels: The structure of each of label_map's values.
        label_maps: Single-label LabelMaps by structure, instead of label_map.
        values_of: The nonzero values of the LabelMap at a path.
    """
    entries: list[LabelEntry] = []

    def add(path: str, file_value: int, structure: Structure | None, value: int) -> None:
        entries.append(
            LabelEntry(
                id=len(entries) + 1,
                path=path,
                name=Path(path).name,
                file_value=file_value,
                structure=structure,
                value=value,
                include=structure is not None,
            )
        )

    if label_map is not None:
        names: dict[int, Structure] = {value: structure for structure, value in (labels or {}).items()}
        for file_value in values_of(str(label_map)):
            add(str(label_map), file_value, names.get(file_value), file_value)
        return entries
    extras: list[tuple[str, int]] = []
    for index, (structure, path) in enumerate((label_maps or {}).items()):
        first, *others = values_of(str(path))
        add(str(path), first, structure, index + 1)
        extras.extend((str(path), other) for other in others)
    for path, file_value in extras:
        add(path, file_value, None, next_free_value(entry["value"] for entry in entries))
    return entries


def next_free_value(used: Iterable[int]) -> int:
    """The smallest label value from 1 up that is not used."""
    used = set(used)
    return next(value for value in range(1, len(used) + 2) if value not in used)


def entry_problems(entries: Iterable[LabelEntry]) -> list[str]:
    """What keeps the included entries from forming a segmentation, or an empty list."""
    included = [entry for entry in entries if entry["include"]]
    problems: list[str] = []
    if not included:
        problems.append("Include at least one label")
    if unnamed := [entry for entry in included if entry["structure"] is None]:
        problems.append(f"Choose a structure for {', '.join(_describe(entry) for entry in unnamed)}")
    structures = [entry["structure"] for entry in included if entry["structure"] is not None]
    if duplicates := sorted({structure for structure in structures if structures.count(structure) > 1}):
        problems.append(f"More than one included label names {', '.join(duplicates)}")
    values = [entry["value"] for entry in included]
    if duplicates := sorted({value for value in values if values.count(value) > 1}):
        problems.append(f"More than one included label has value {', '.join(map(str, duplicates))}")
    if bad := sorted({value for value in values if not 0 < value <= 65535}):
        problems.append(f"Label values must lie in 1 to 65535, got {', '.join(map(str, bad))}")
    return problems


def missing_bones(structures: Iterable[Structure], entries: Iterable[LabelEntry]) -> list[str]:
    """Messages for each cartilage among structures whose bone no included entry names."""
    named = {entry["structure"] for entry in entries if entry["include"]}
    return [
        f"{structure} needs its bone, {BONES[structure]}, among the included labels"
        for structure in structures
        if structure in BONES and BONES[structure] not in named
    ]


def _describe(entry: LabelEntry) -> str:
    return f"{Path(entry['path']).name} value {entry['file_value']}"


def compose_segmentation(entries: Iterable[LabelEntry], interpolator: Interpolator = "label_gaussian", padding: int = 4) -> Segmentation:
    """The Segmentation of the included entries, each structure at its entry's value.

    Entries all from one LabelMap are relabelled in place, which keeps the
    grid and every voxel; values not included become background. Entries
    from several LabelMaps are each taken as a single-label LabelMap of the
    voxels at their file_value and assembled onto one grid
    (segmentation.assemble_label_maps), then relabelled.

    Raises:
        ValueError: If the entries have problems (entry_problems), or a LabelMap cannot be read or assembled.
    """
    entries = [entry for entry in entries if entry["include"]]
    if problems := entry_problems(entries):
        raise ValueError("; ".join(problems))
    labels: dict[str, int] = {}
    for entry in entries:
        assert entry["structure"] is not None
        labels[entry["structure"]] = entry["value"]

    images = {path: read_label_map(path) for path in dict.fromkeys(entry["path"] for entry in entries)}
    if len(images) == 1:
        (image,) = images.values()
        changes = dict.fromkeys(map(float, label_values(image)), 0.0)
        changes.update({float(entry["file_value"]): float(entry["value"]) for entry in entries})
        return Segmentation(_relabel(image, changes), labels)

    single_labels = {name: sitk.Equal(images[entry["path"]], entry["file_value"]) for name, entry in zip(labels, entries, strict=True)}
    assembled = assemble_label_maps(single_labels, interpolator, padding)
    changes = {float(assembled.labels[name]): float(value) for name, value in labels.items()}
    return Segmentation(_relabel(assembled.image, changes), labels)


def _relabel(image: sitk.Image, changes: Mapping[float, float]) -> sitk.Image:
    """image with each value changed as changes says, widened to sitkUInt16 if a new value needs it."""
    if max(changes.values(), default=0) > 255 and image.GetPixelID() == sitk.sitkUInt8:
        image = sitk.Cast(image, sitk.sitkUInt16)
    if max(changes.values(), default=0) > 32767 and image.GetPixelID() == sitk.sitkInt16:
        image = sitk.Cast(image, sitk.sitkUInt16)
    return sitk.ChangeLabel(image, changeMap=dict(changes))


def brep_sources(entries: Iterable[LabelEntry]) -> dict[str, Any] | None:
    """The fields of a LabelMapBrep that read the included entries' segmentation as kneemesh-brep would, or None if none can.

    Entries from one LabelMap at their own values become label_map with
    labels. Entries that are each the only value of their own LabelMap become
    label_maps, assembled in entry order; kneemesh-brep numbers them 1, 2, ...
    whatever the entries' values. Anything else, such as several values of
    one file with others, or renumbered values of one file, has no
    equivalent, and the composed segmentation must be written instead.
    """
    entries = [entry for entry in entries if entry["include"]]
    paths = list(dict.fromkeys(entry["path"] for entry in entries))
    if len(paths) == 1 and all(entry["value"] == entry["file_value"] for entry in entries):
        return {"label_map": paths[0], "labels": {entry["structure"]: entry["value"] for entry in entries}}
    if len(paths) == len(entries) and all(len(label_values(read_label_map(entry["path"]))) == 1 for entry in entries):
        return {"label_maps": {entry["structure"]: entry["path"] for entry in entries}}
    return None


def label_surface(image: sitk.Image, value: int) -> Any:
    """A smooth vtkPolyData surface of the voxels at value, in physical coordinates, as the B-rep pipelines extract one.

    The image is cropped around the label first, so large LabelMaps preview quickly.

    Raises:
        ValueError: If the image holds no voxel at value.
    """
    mask = sitk.Equal(image, value)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(mask)
    if not statistics.HasLabel(1):
        raise ValueError(f"The LabelMap holds no voxel at value {value}")
    box = statistics.GetBoundingBox(1)
    size = np.asarray(image.GetSize())
    start = np.maximum(np.asarray(box[:3]) - _PREVIEW_PADDING, 0)
    stop = np.minimum(np.asarray(box[:3]) + np.asarray(box[3:]) + _PREVIEW_PADDING, size)
    cropped = sitk.RegionOfInterest(mask, (stop - start).tolist(), start.tolist())
    surface = extract_isocontour(signed_distance_field(cropped, 1))
    return smooth_isocontour(surface, _PREVIEW_SMOOTHING_ITERATIONS)
