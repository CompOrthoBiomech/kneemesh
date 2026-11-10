import json

import gmsh
import pytest

from meniscus2brep.config import BrepConfig, MeshConfig
from meniscus2brep.disk import read_brep, write_label_map
from meniscus2brep.geo import gmsh_session
from meniscus2brep.mesh import Algorithm2D, Algorithm3D
from meniscus2brep.pipeline import brep_main, configure, mesh_main


@pytest.fixture
def in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_config(path, values):
    path.write_text(json.dumps(values))
    return path


def test_configs_have_defaults_for_every_field():
    assert BrepConfig().points_per_slice == 32
    assert MeshConfig().algorithm_2d is Algorithm2D.FRONTAL_DELAUNAY


def test_mesh_config_saves_algorithms_by_name_and_reads_names_or_values():
    saved = json.loads(MeshConfig(algorithm_3d=Algorithm3D.HXT).model_dump_json())

    assert saved["algorithm_3d"] == "HXT"
    assert MeshConfig.model_validate({"algorithm_3d": "hxt"}).algorithm_3d is Algorithm3D.HXT
    assert MeshConfig.model_validate({"algorithm_3d": 10}).algorithm_3d is Algorithm3D.HXT


@pytest.mark.parametrize("config_type", [BrepConfig, MeshConfig])
def test_configs_round_trip_through_json(tmp_path, config_type):
    path = tmp_path / "config.json"
    path.write_text(config_type().model_dump_json())

    assert config_type.from_json(path) == config_type()


def test_configure_without_config_saves_timestamped_settings(in_tmp_path):
    config, saved = configure(BrepConfig, "test", ["--points-per-slice", "10", "--cap", "planar"])

    assert config.points_per_slice == 10
    assert config.cap == "planar"
    assert saved.parent == type(saved)(".")
    assert saved.name.startswith("settings") and saved.suffix == ".json"
    assert BrepConfig.from_json(in_tmp_path / saved.name) == config


def test_configure_without_arguments_saves_defaults(in_tmp_path):
    config, saved = configure(MeshConfig, "test", [])

    assert config == MeshConfig()
    assert MeshConfig.from_json(in_tmp_path / saved.name) == MeshConfig()


def test_configure_overrides_config_file_and_saves_modified_copy(tmp_path):
    path = write_config(tmp_path / "run.json", {"spacing": 2.5, "spline_order": 2})

    config, saved = configure(BrepConfig, "test", ["--config", str(path), "--spline-order", "4", "--sigma", "0.5"])

    assert (config.spacing, config.spline_order, config.sigma) == (2.5, 4, 0.5)
    assert saved == tmp_path / "run_modified.json"
    assert BrepConfig.from_json(saved) == config
    assert json.loads(path.read_text()) == {"spacing": 2.5, "spline_order": 2}


def test_configure_with_config_and_no_overrides_saves_nothing(tmp_path):
    path = write_config(tmp_path / "run.json", {"element": "hexahedral"})

    config, saved = configure(MeshConfig, "test", ["--config", str(path)])

    assert config.element == "hexahedral"
    assert saved is None
    assert sorted(item.name for item in tmp_path.iterdir()) == ["run.json"]


def test_configure_reads_none_to_clear_optional_values(tmp_path):
    path = write_config(tmp_path / "run.json", {"max_size": 2.0})

    config, _ = configure(MeshConfig, "test", ["--config", str(path), "--max-size", "none"])

    assert config.max_size is None


@pytest.mark.parametrize(
    "argv",
    [["--points-per-slice", "many"], ["--cap", "round"], ["--not-an-option", "1"]],
)
def test_configure_rejects_invalid_overrides(in_tmp_path, argv):
    with pytest.raises(SystemExit) as error:
        configure(BrepConfig, "test", argv)

    assert error.value.code == 2
    assert list(in_tmp_path.iterdir()) == []


def test_configure_rejects_unknown_key_in_config_file(tmp_path):
    path = write_config(tmp_path / "run.json", {"spacingg": 2.0})

    with pytest.raises(SystemExit):
        configure(BrepConfig, "test", ["--config", str(path)])


def mesh_entity_counts(path):
    with gmsh_session():
        gmsh.merge(str(path))
        return {
            gmsh.model.mesh.getElementProperties(element_type)[0]: len(gmsh.model.mesh.getElementsByType(element_type)[0])
            for element_type in gmsh.model.mesh.getElementTypes(3)
        }


def test_pipelines_build_and_mesh_a_capped_solid(in_tmp_path, meniscus_label, capsys):
    write_label_map(meniscus_label, in_tmp_path / "label.nii")

    assert brep_main(["--label-map", "label.nii", "--output", "solid.brep", "--cap", "smooth"]) == 0
    assert read_brep(in_tmp_path / "solid.brep").text
    assert mesh_main(["--brep", "solid.brep", "--output", "solid.msh", "--max-size", "4", "--element-order", "2"]) == 0

    assert list(mesh_entity_counts(in_tmp_path / "solid.msh")) == ["Tetrahedron 10"]
    assert "Wrote solid.msh" in capsys.readouterr().out
    assert len(list(in_tmp_path.glob("settings*.json"))) >= 1


def test_pipelines_build_and_mesh_hexahedral_blocks(in_tmp_path, meniscus_label):
    write_label_map(meniscus_label, in_tmp_path / "label.nii")
    brep_config = write_config(in_tmp_path / "brep.json", {"label_map": "label.nii", "output": "blocks.brep", "points_per_slice": 10, "hex_blocks": True})
    mesh_config = write_config(in_tmp_path / "mesh.json", {"brep": "blocks.brep", "output": "blocks.vtk", "element": "hexahedral"})

    assert brep_main(["--config", str(brep_config)]) == 0
    assert mesh_main(["--config", str(mesh_config), "--elements-per-edge", "2", "--elements-along", "10"]) == 0

    assert (in_tmp_path / "blocks.vtk").stat().st_size > 0
    assert MeshConfig.from_json(in_tmp_path / "mesh_modified.json").elements_per_edge == 2


def test_pipeline_reports_errors_without_traceback(in_tmp_path, capsys):
    assert mesh_main(["--brep", "missing.brep"]) == 1
    assert "No such file" in capsys.readouterr().err
