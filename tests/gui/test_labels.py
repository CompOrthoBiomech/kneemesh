import json

import numpy as np
import pytest
import SimpleITK as sitk

from kneemesh.disk import read_label_map, write_label_map
from kneemesh.gui.labels import (
    LabelEntry,
    brep_sources,
    compose_segmentation,
    config_entries,
    entry_problems,
    label_surface,
    label_values,
    missing_bones,
    new_entries,
    next_free_value,
)


def _image(values: np.ndarray, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> sitk.Image:
    image = sitk.GetImageFromArray(values.astype(np.uint8))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin(origin)
    return image


def _blocks() -> np.ndarray:
    """A 10-voxel cube holding values 1 and 2 in two 3-voxel blocks, and 7 in one voxel."""
    values = np.zeros((10, 10, 10), dtype=np.uint8)
    values[2:5, 2:5, 2:5] = 1
    values[5:8, 5:8, 5:8] = 2
    values[0, 0, 0] = 7
    return values


def _entry(entry_id: int, path: str, file_value: int, structure, value: int, include: bool = True) -> LabelEntry:
    return LabelEntry(id=entry_id, path=path, name=path, file_value=file_value, structure=structure, value=value, include=include)


def test_label_values_lists_nonzero_values():
    assert label_values(_image(_blocks())) == [1, 2, 7]


def test_next_free_value_fills_gaps():
    assert next_free_value([]) == 1
    assert next_free_value([1, 2, 4]) == 3
    assert next_free_value([2, 3]) == 1


def test_single_value_takes_next_free_value_and_structure_from_name(tmp_path):
    first = new_entries(tmp_path / "tibia.nii", [1], [])
    assert [(entry["structure"], entry["value"], entry["include"]) for entry in first] == [("tibia", 1, True)]
    # A second single-label file with the same value is renumbered.
    second = new_entries(tmp_path / "lateral_meniscus.nii", [1], first)
    assert [(entry["id"], entry["structure"], entry["value"], entry["name"]) for entry in second] == [
        (2, "lateral_meniscus", 2, "lateral_meniscus.nii")
    ]


def test_unknown_name_leaves_structure_unset_and_excluded(tmp_path):
    (entry,) = new_entries(tmp_path / "scan.nii", [5], [])
    assert entry["structure"] is None
    assert not entry["include"]
    assert entry["value"] == 1


def test_several_values_keep_free_values_and_read_labels_json(tmp_path):
    path = tmp_path / "knee.nrrd"
    (tmp_path / "knee_labels.json").write_text(json.dumps({"femur": 1, "tibia": 2, "unknown": 3}), encoding="utf-8")
    existing = new_entries(tmp_path / "patella.nii", [1], [])
    entries = new_entries(path, [1, 2, 3], existing)
    assert [(entry["file_value"], entry["value"], entry["structure"]) for entry in entries] == [
        (1, 2, "femur"),
        (2, 3, "tibia"),
        (3, 4, None),
    ]


def test_structure_already_named_is_left_unset(tmp_path):
    existing = new_entries(tmp_path / "tibia.nii", [1], [])
    (entry,) = new_entries(tmp_path / "other" / "tibia.nii", [1], existing)
    assert entry["structure"] is None


def test_entry_problems_reports_unnamed_duplicate_and_bad_values():
    entries = [
        _entry(1, "a.nii", 1, "tibia", 1),
        _entry(2, "b.nii", 1, "tibia", 1),
        _entry(3, "c.nii", 1, None, 0),
        _entry(4, "d.nii", 1, "femur", 9, include=False),
    ]
    problems = entry_problems(entries)
    assert any("Choose a structure" in problem for problem in problems)
    assert any("names tibia" in problem for problem in problems)
    assert any("value 1" in problem for problem in problems)
    assert any("1 to 65535" in problem for problem in problems)
    assert entry_problems([entries[0]]) == []
    assert entry_problems([]) == ["Include at least one label"]


def test_missing_bones_names_cartilage_without_its_bone():
    entries = [_entry(1, "a.nii", 1, "medial_tibial_cartilage", 1)]
    assert missing_bones(["medial_tibial_cartilage", "lateral_meniscus"], entries) == [
        "medial_tibial_cartilage needs its bone, tibia, among the included labels"
    ]
    assert missing_bones(["medial_tibial_cartilage"], [*entries, _entry(2, "a.nii", 2, "tibia", 2)]) == []


def test_compose_one_file_relabels_in_place(tmp_path):
    path = str(tmp_path / "knee.nrrd")
    write_label_map(_image(_blocks()), path)
    entries = [
        _entry(1, path, 1, "femur", 2),
        _entry(2, path, 2, "tibia", 300),
        _entry(3, path, 7, "patella", 3, include=False),
    ]
    segmentation = compose_segmentation(entries)
    assert segmentation.labels == {"femur": 2, "tibia": 300}
    values = sitk.GetArrayFromImage(segmentation.image)
    expected = np.zeros_like(values, dtype=np.int64)
    expected[_blocks() == 1] = 2
    expected[_blocks() == 2] = 300
    np.testing.assert_array_equal(values, expected)
    assert segmentation.image.GetPixelID() == sitk.sitkUInt16


def test_compose_several_files_assembles_at_entry_values(tmp_path):
    femur, tibia = str(tmp_path / "femur.nii"), str(tmp_path / "tibia.nii")
    write_label_map(_image((_blocks() == 1).astype(np.uint8) * 5), femur)
    write_label_map(_image((_blocks() == 2).astype(np.uint8), origin=(0.0, 0.0, 0.0)), tibia)
    entries = [_entry(1, femur, 5, "femur", 4), _entry(2, tibia, 1, "tibia", 9)]
    segmentation = compose_segmentation(entries, "nearest_neighbor", padding=2)
    assert segmentation.labels == {"femur": 4, "tibia": 9}
    values = sitk.GetArrayFromImage(segmentation.image)
    assert np.count_nonzero(values == 4) == 27
    assert np.count_nonzero(values == 9) == 27
    assert set(np.unique(values)) == {0, 4, 9}


def test_compose_rejects_problems(tmp_path):
    with pytest.raises(ValueError, match="Choose a structure"):
        compose_segmentation([_entry(1, str(tmp_path / "a.nii"), 1, None, 1)])


def test_brep_sources_by_how_the_entries_can_be_read(tmp_path):
    knee, femur, tibia = str(tmp_path / "knee.nrrd"), str(tmp_path / "femur.nii"), str(tmp_path / "tibia.nii")
    write_label_map(_image(_blocks()), knee)
    write_label_map(_image((_blocks() == 1).astype(np.uint8)), femur)
    write_label_map(_image((_blocks() == 2).astype(np.uint8)), tibia)

    same_values = [_entry(1, knee, 1, "femur", 1), _entry(2, knee, 2, "tibia", 2), _entry(3, knee, 7, None, 3, include=False)]
    assert brep_sources(same_values) == {"label_map": knee, "labels": {"femur": 1, "tibia": 2}}

    single_labels = [_entry(1, femur, 1, "femur", 1), _entry(2, tibia, 1, "tibia", 2)]
    assert brep_sources(single_labels) == {"label_maps": {"femur": femur, "tibia": tibia}}

    renumbered = [_entry(1, knee, 1, "femur", 5)]
    assert brep_sources(renumbered) is None
    mixed = [_entry(1, knee, 1, "femur", 1), _entry(2, tibia, 1, "tibia", 2)]
    assert brep_sources(mixed) is None


def test_config_entries_from_a_label_map_and_from_label_maps():
    values = {"knee.nrrd": [1, 2, 3], "femur.nii": [1], "tibia.nii": [1, 4]}
    entries = config_entries("knee.nrrd", {"femur": 1, "tibia": 3}, None, lambda path: values[path])
    assert [(entry["file_value"], entry["structure"], entry["value"], entry["include"]) for entry in entries] == [
        (1, "femur", 1, True),
        (2, None, 2, False),
        (3, "tibia", 3, True),
    ]
    entries = config_entries(None, None, {"femur": "femur.nii", "tibia": "tibia.nii"}, lambda path: values[path])
    assert [(entry["path"], entry["file_value"], entry["structure"], entry["value"], entry["include"]) for entry in entries] == [
        ("femur.nii", 1, "femur", 1, True),
        ("tibia.nii", 1, "tibia", 2, True),
        ("tibia.nii", 4, None, 3, False),
    ]
    assert len({entry["id"] for entry in entries}) == 3


def test_label_surface_lies_around_the_label(tmp_path):
    path = tmp_path / "knee.nrrd"
    image = _image(_blocks())
    image.SetOrigin((10.0, 20.0, 30.0))
    write_label_map(image, path)
    surface = label_surface(read_label_map(path), 2)
    assert surface.GetNumberOfPolys() > 0
    x_min, x_max, y_min, y_max, z_min, z_max = surface.GetBounds()
    # Voxels 5-7 along each axis span 4.5 to 7.5 in index space; smoothing pulls the surface in a little.
    assert 14.0 < x_min < 15.5 and 16.5 < x_max < 18.0
    assert 24.0 < y_min < 25.5 and 26.5 < y_max < 28.0
    assert 34.0 < z_min < 35.5 and 36.5 < z_max < 38.0
    with pytest.raises(ValueError, match="no voxel"):
        label_surface(image, 3)
