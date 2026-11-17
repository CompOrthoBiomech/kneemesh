import asyncio
import io
import json
import uuid
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk
from trame.app import get_server

from kneemesh.config import BrepMesh, LabelMapBrep, MeniscusBrep
from kneemesh.disk import read_segmentation, write_label_map
from kneemesh.gui.app import KneemeshApp, _parse_config, zip_files


@pytest.fixture
def app() -> KneemeshApp:
    # Each app needs its own server, which get_server keeps by name.
    return KneemeshApp(get_server(f"test-{uuid.uuid4().hex}", client_type="vue3"))


def _write_blocks(path) -> str:
    values = np.zeros((10, 10, 10), dtype=np.uint8)
    values[2:5, 2:5, 2:5] = 1
    values[5:8, 5:8, 5:8] = 2
    write_label_map(sitk.GetImageFromArray(values), path)
    return str(path)


def test_defaults_are_valid(app):
    assert app.state.form_errors == {}
    assert app.state.expected_geometry["femur"] == "femur.stl"
    assert app.state.expected_geometry["lateral_meniscus"] == "lateral_meniscus.brep"


def test_select_shows_a_structure_section(app):
    app.select("mesh:medial_tibial_cartilage")
    assert (app.state.selected_kind, app.state.selected_structure, app.state.selected_section) == ("mesh", "medial_tibial_cartilage", "cartilage")
    assert app.state.selected_shared == ["patellar_cartilage", "lateral_tibial_cartilage"]
    app.select("settings")
    assert (app.state.selected_kind, app.state.selected_structure, app.state.selected_section) == ("settings", None, None)


def test_invalid_settings_are_reported_by_key(app):
    app.state.update({"geo__meniscus__spacing": "wide"})
    app._validate()
    assert "geo__meniscus__spacing" in app.state.form_errors
    app.reset_section("geometry", "meniscus")
    app._validate()
    assert app.state.form_errors == {}


def test_import_label_file_and_edit_entries(app, tmp_path):
    path = _write_blocks(tmp_path / "scan.nrrd")
    asyncio.run(app._load_label_file(path))
    assert [(entry["file_value"], entry["value"], entry["structure"]) for entry in app.state.labels] == [(1, 1, None), (2, 2, None)]
    first, second = (entry["id"] for entry in app.state.labels)
    app.update_label(first, "structure", "tibia")
    app.update_label(first, "include", True)
    app.update_label(second, "value", "7")
    app.update_label(second, "value", "seven")
    assert app.state.labels[0]["structure"] == "tibia"
    assert app.state.labels[1]["value"] == 7
    assert app.state.label_structures == ["tibia"]
    assert app.state.label_problems == []
    app.remove_label(first)
    assert [entry["id"] for entry in app.state.labels] == [second]
    assert app.state.label_problems == ["Include at least one label"]


def test_element_types_and_geometry_paths(app):
    app.set_element_type("lateral_meniscus", "hex8")
    app.set_element_type("medial_meniscus", "")
    assert app.state.element_types == {"lateral_meniscus": "hex8"}
    app.set_brep_path("lateral_meniscus", "elsewhere.brep")
    assert app._geometry_path("lateral_meniscus") == "elsewhere.brep"
    app.set_brep_path("lateral_meniscus", None)
    assert app._geometry_path("lateral_meniscus") == "lateral_meniscus.brep"


def test_parse_config_tells_the_kind():
    assert isinstance(_parse_config(json.dumps({"label_map": "a.nii", "labels": {"tibia": 1}})), LabelMapBrep)
    assert isinstance(_parse_config(json.dumps({"breps": {"tibia": "tibia.stl"}, "element_type": "hex8"})), BrepMesh)
    assert type(_parse_config(json.dumps({"label_maps": {"tibia": "t.nii"}, "output": "knee.nrrd"}))).__name__ == "LabelMapAssembly"
    with pytest.raises(ValueError, match="not a LabelMapBrep"):
        _parse_config(json.dumps({"bogus": 1}))


def test_import_and_export_configurations(app, tmp_path):
    knee = _write_blocks(tmp_path / "knee.nrrd")
    brep_json = tmp_path / "brep.json"
    brep_json.write_text(
        json.dumps({"label_map": knee, "labels": {"femur": 1, "tibia": 2}, "output": "_hex.brep", "meniscus": {"spacing": 3.0}}), encoding="utf-8"
    )
    asyncio.run(app._import_config(str(brep_json)))
    assert app.state.geo__meniscus__spacing == 3.0
    assert app.state.geo__output == "_hex.brep"
    assert [(entry["structure"], entry["value"]) for entry in app.state.labels] == [("femur", 1), ("tibia", 2)]

    # Labels at their own values export as label_map with labels.
    exported = tmp_path / "exported.json"
    asyncio.run(app._export_geometry(str(exported)))
    config = LabelMapBrep.from_json(exported)
    assert (str(config.label_map), config.labels, config.meniscus.spacing) == (knee, {"femur": 1, "tibia": 2}, 3.0)

    # Renumbered labels have no such equivalent, so their segmentation is written beside the configuration.
    app.update_label(app.state.labels[0]["id"], "value", 5)
    asyncio.run(app._export_geometry(str(exported)))
    config = LabelMapBrep.from_json(exported)
    assert config.label_map == tmp_path / "exported_segmentation.nrrd"
    assert read_segmentation(tmp_path / "exported_segmentation.nrrd").labels == {"femur": 5, "tibia": 2}

    app.set_element_type("femur", "")
    app.state.update({"mesh__element_type": "hex8"})
    app.set_brep_path("tibia", "tibia_surface.stl")
    mesh_json = tmp_path / "mesh.json"
    app._export_mesh(str(mesh_json))
    config = BrepMesh.from_json(mesh_json)
    assert config.element_type == "hex8"
    assert {name: str(path) for name, path in (config.breps or {}).items()} == {"femur": "femur.stl", "tibia": "tibia_surface.stl"}

    asyncio.run(app._import_config(str(mesh_json)))
    assert app.state.brep_paths == {"femur": "femur.stl", "tibia": "tibia_surface.stl"}


def test_label_maps_import_previews_each_file(app, tmp_path):
    paths = {}
    for name, offset in (("femur", 0.0), ("tibia", 100.0)):
        image = sitk.GetImageFromArray(np.pad(np.ones((3, 3, 3), dtype=np.uint8), 2))
        image.SetOrigin((offset, 0.0, 0.0))
        paths[name] = str(tmp_path / f"{name}.nii")
        write_label_map(image, paths[name])
    config = tmp_path / "assemble.json"
    config.write_text(json.dumps({"label_maps": paths, "output": str(tmp_path / "knee.nrrd")}), encoding="utf-8")
    asyncio.run(app._import_config(str(config)))
    assert [(entry["structure"], entry["value"]) for entry in app.state.labels] == [("femur", 1), ("tibia", 2)]
    assert app.state.geo__assembled_label_map == str(tmp_path / "knee.nrrd")
    femur, tibia = (app._surfaces[(paths[name], 1)].GetBounds() for name in ("femur", "tibia"))
    assert femur[1] < 10
    assert tibia[0] > 100


def test_own_sections_start_from_the_shared_one_and_export(app, tmp_path):
    app.state.update({"geo__meniscus__spacing": 3.0})
    app.select("geometry:lateral_meniscus")
    app.set_own_section("geometry", "lateral_meniscus", True)
    assert app.state.own_sections["geometry"] == ["lateral_meniscus"]
    assert (app.state.selected_section, app.state.selected_shared) == ("lateral_meniscus", [])
    assert app.state.geo__lateral_meniscus__spacing == 3.0
    app.state.update({"geo__lateral_meniscus__spacing": 2.0})
    app.select("geometry:medial_meniscus")
    assert (app.state.selected_section, app.state.selected_shared) == ("meniscus", [])

    knee = _write_blocks(tmp_path / "knee.nrrd")
    asyncio.run(app._load_label_file(knee))
    for entry, structure in zip(app.state.labels, ("lateral_meniscus", "medial_meniscus"), strict=True):
        app.update_label(entry["id"], "structure", structure)
        app.update_label(entry["id"], "include", True)
    exported = tmp_path / "brep.json"
    asyncio.run(app._export_geometry(str(exported)))
    config = LabelMapBrep.from_json(exported)
    assert (config.section("lateral_meniscus", MeniscusBrep).spacing, config.section("medial_meniscus", MeniscusBrep).spacing) == (2.0, 3.0)

    # Sharing again keeps the own values for later but leaves them out of the configuration.
    app.set_own_section("geometry", "lateral_meniscus", False)
    assert app._geometry_config() == LabelMapBrep.model_validate({"meniscus": {"spacing": 3.0}})
    app.set_own_section("geometry", "lateral_meniscus", True)
    assert app.state.geo__lateral_meniscus__spacing == 2.0

    other = KneemeshApp(get_server(f"test-{uuid.uuid4().hex}", client_type="vue3"))
    asyncio.run(other._import_config(str(exported)))
    assert other.state.own_sections["geometry"] == ["lateral_meniscus"]
    assert other.state.geo__lateral_meniscus__spacing == 2.0


def test_solo_shows_only_one_structure(app):
    app.toggle_visibility("tibia")
    app.toggle_solo("tibia")
    assert (app.state.solo, app.state.hidden) == ("tibia", [])
    app.toggle_solo("tibia")
    assert app.state.solo is None


def test_uploads_are_written_into_the_dialog_directory(app, tmp_path):
    app.add_label()
    app.dialog_open_directory(str(tmp_path))
    app._upload([{"name": "../scan.nrrd", "content": [b"ab", b"cd"]}, {"name": "notes.txt", "content": b"x"}])
    assert (tmp_path / "scan.nrrd").read_bytes() == b"abcd"
    assert (tmp_path / "notes.txt").read_bytes() == b"x"
    assert not (tmp_path.parent / "scan.nrrd").exists()
    # The one file the dialog can open is selected.
    assert (app.state.dialog_selection, app.state.dialog_selection_name) == (str(tmp_path / "scan.nrrd"), "scan.nrrd")
    assert app.state.dialog_upload is None


def test_zip_of_written_files_names_them_from_their_common_directory(tmp_path):
    (tmp_path / "meshes").mkdir()
    (tmp_path / "a.brep").write_text("a", encoding="utf-8")
    (tmp_path / "meshes" / "a.vtu").write_text("b", encoding="utf-8")
    archive = zipfile.ZipFile(io.BytesIO(zip_files([str(tmp_path / "a.brep"), str(tmp_path / "meshes" / "a.vtu"), str(tmp_path / "gone")])))
    assert sorted(archive.namelist()) == ["a.brep", "meshes/a.vtu"]
    assert archive.read("meshes/a.vtu") == b"b"
    assert zipfile.ZipFile(io.BytesIO(zip_files([]))).namelist() == []


def test_exports_are_remembered_for_download(app, tmp_path):
    app._export_mesh(str(tmp_path / "mesh.json"))
    assert list(app._written) == [str((tmp_path / "mesh.json").resolve())]
    assert app.state.written_count == 1
