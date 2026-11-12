import json

import gmsh
import pytest
from loguru import logger

from kneemesh.brep import gmsh_session
from kneemesh.config import MeniscusBrep, MeniscusMesh, TibialCartilageBrep
from kneemesh.disk import read_brep, write_label_map
from kneemesh.pipeline import STRUCTURES, brep_main, configure, mesh_main
from kneemesh.tet_meshing import Algorithm2D, Algorithm3D
from kneemesh.vol import signed_distance_field


@pytest.fixture
def in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_config(path, values):
    path.write_text(json.dumps(values))
    return path


def test_configs_have_defaults_for_every_field():
    assert MeniscusBrep().points_per_slice == 32
    assert MeniscusMesh().algorithm_2d is Algorithm2D.FRONTAL_DELAUNAY
    assert (MeniscusBrep().smoothing_iterations, MeniscusBrep().smoothing_pass_band) == (50, 0.001)


def test_configure_overrides_smoothing(in_tmp_path):
    config, _, _ = configure({"meniscus": MeniscusBrep}, "test", ["meniscus", "--smoothing-iterations", "0", "--smoothing-pass-band", "0.01"])

    assert (config.smoothing_iterations, config.smoothing_pass_band) == (0, 0.01)


def test_mesh_config_saves_algorithms_by_name_and_reads_them_in_any_case():
    saved = json.loads(MeniscusMesh(algorithm_3d=Algorithm3D.HXT).model_dump_json())

    assert saved["algorithm_3d"] == "hxt"
    assert MeniscusMesh.model_validate({"algorithm_3d": "hxt"}).algorithm_3d is Algorithm3D.HXT
    assert MeniscusMesh.model_validate({"algorithm_3d": "HXT"}).algorithm_3d is Algorithm3D.HXT


def test_mesh_config_rejects_gmsh_numbers_and_unknown_algorithms():
    for value in (10, "rtree"):
        with pytest.raises(ValueError):
            MeniscusMesh.model_validate({"algorithm_3d": value})


@pytest.mark.parametrize("config_type", [MeniscusBrep, MeniscusMesh])
def test_configs_round_trip_through_json(tmp_path, config_type):
    path = tmp_path / "config.json"
    path.write_text(config_type().model_dump_json())

    assert config_type.from_json(path) == config_type()


def test_configure_without_config_saves_timestamped_settings(in_tmp_path):
    config, _, saved = configure({"meniscus": MeniscusBrep}, "test", ["meniscus", "--points-per-slice", "10", "--cap", "planar"])

    assert config.points_per_slice == 10
    assert config.cap == "planar"
    assert saved is not None
    assert saved.parent == type(saved)(".")
    assert saved.name.startswith("settings") and saved.suffix == ".json"
    assert MeniscusBrep.from_json(in_tmp_path / saved.name) == config


def test_configure_without_arguments_saves_defaults(in_tmp_path):
    config, _, saved = configure({"meniscus": MeniscusMesh}, "test", ["meniscus"])

    assert config == MeniscusMesh()
    assert MeniscusMesh.from_json(in_tmp_path / saved.name) == MeniscusMesh()


def test_configure_overrides_config_file_and_saves_modified_copy(tmp_path):
    path = write_config(tmp_path / "run.json", {"spacing": 2.5, "spline_order": 2})

    config, _, saved = configure({"meniscus": MeniscusBrep}, "test", ["meniscus", "--config", str(path), "--spline-order", "4", "--sigma", "0.5"])

    assert (config.spacing, config.spline_order, config.sigma) == (2.5, 4, 0.5)
    assert saved is not None
    assert saved == tmp_path / "run_modified.json"
    assert MeniscusBrep.from_json(saved) == config
    assert json.loads(path.read_text()) == {"spacing": 2.5, "spline_order": 2}


def test_configure_with_config_and_no_overrides_saves_nothing(tmp_path):
    path = write_config(tmp_path / "run.json", {"element": "hexahedral"})

    config, _, saved = configure({"meniscus": MeniscusMesh}, "test", ["meniscus", "--config", str(path)])

    assert config.element == "hexahedral"
    assert saved is None
    assert sorted(item.name for item in tmp_path.iterdir()) == ["run.json"]


def test_configure_reads_none_to_clear_optional_values(tmp_path):
    path = write_config(tmp_path / "run.json", {"max_size": 2.0})

    config, _, _ = configure({"meniscus": MeniscusMesh}, "test", ["meniscus", "--config", str(path), "--max-size", "none"])

    assert config.max_size is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--points-per-slice", "many"],
        ["--cap", "round"],
        ["--not-an-option", "1"],
        ["--smoothing-iterations", "-1"],
        ["--smoothing-pass-band", "0"],
    ],
)
def test_configure_rejects_invalid_overrides(in_tmp_path, argv):
    with pytest.raises(SystemExit) as error:
        configure({"meniscus": MeniscusBrep}, "test", ["meniscus", *argv])

    assert error.value.code == 2
    assert list(in_tmp_path.iterdir()) == []


def test_configure_rejects_unknown_key_in_config_file(tmp_path):
    path = write_config(tmp_path / "run.json", {"spacingg": 2.0})

    with pytest.raises(SystemExit):
        configure({"meniscus": MeniscusBrep}, "test", ["meniscus", "--config", str(path)])


def mesh_entity_counts(path):
    with gmsh_session():
        gmsh.merge(str(path))
        return {
            gmsh.model.mesh.getElementProperties(element_type)[0]: len(gmsh.model.mesh.getElementsByType(element_type)[0])
            for element_type in gmsh.model.mesh.getElementTypes(3)
        }


def test_pipelines_build_and_mesh_a_capped_solid(in_tmp_path, meniscus_label, capsys):
    write_label_map(meniscus_label, in_tmp_path / "label.nii")

    assert brep_main(["meniscus", "--label-map", "label.nii", "--output", "solid.brep", "--cap", "smooth"]) == 0
    assert read_brep(in_tmp_path / "solid.brep").text
    assert mesh_main(["meniscus", "--brep", "solid.brep", "--output", "solid.msh", "--max-size", "4", "--element-order", "2"]) == 0

    assert list(mesh_entity_counts(in_tmp_path / "solid.msh")) == ["Tetrahedron 10"]
    assert "Wrote solid.msh" in capsys.readouterr().err
    assert len(list(in_tmp_path.glob("settings*.json"))) >= 1


def test_pipelines_build_and_mesh_hexahedral_blocks(in_tmp_path, meniscus_label):
    write_label_map(meniscus_label, in_tmp_path / "label.nii")
    brep_config = write_config(
        in_tmp_path / "brep.json", {"label_map": "label.nii", "output": "blocks.brep", "points_per_slice": 10, "hex_blocks": True}
    )
    mesh_config = write_config(in_tmp_path / "mesh.json", {"brep": "blocks.brep", "output": "blocks.vtk", "element": "hexahedral"})

    assert brep_main(["meniscus", "--config", str(brep_config)]) == 0
    assert mesh_main(["meniscus", "--config", str(mesh_config), "--elements-per-edge", "2", "--elements-along", "10"]) == 0

    assert (in_tmp_path / "blocks.vtk").stat().st_size > 0
    assert MeniscusMesh.from_json(in_tmp_path / "mesh_modified.json").elements_per_edge == 2


def test_pipeline_reports_errors_without_traceback(in_tmp_path, capsys):
    assert mesh_main(["meniscus", "--brep", "missing.brep"]) == 1
    assert "No such file" in capsys.readouterr().err


def test_configure_returns_the_config_file(tmp_path):
    path = write_config(tmp_path / "run.json", {})

    assert configure({"meniscus": MeniscusMesh}, "test", ["meniscus", "--config", str(path)]).source == path


def test_configure_names_settings_with_the_given_timestamp(in_tmp_path):
    assert configure({"meniscus": MeniscusMesh}, "test", ["meniscus"], timestamp="20260102_030405").saved.name == "settings20260102_030405.json"


def test_package_logging_is_disabled_by_default(meniscus_label):
    messages = []
    handler = logger.add(messages.append, level="DEBUG")
    try:
        signed_distance_field(meniscus_label)
    finally:
        logger.remove(handler)

    assert messages == []


def test_pipeline_without_config_logs_to_timestamped_file_and_stderr(in_tmp_path, meniscus_label, capsys):
    write_label_map(meniscus_label, in_tmp_path / "label.nii")

    assert brep_main(["meniscus", "--label-map", "label.nii"]) == 0

    logs = list(in_tmp_path.glob("brep*.log"))
    settings = list(in_tmp_path.glob("settings*.json"))
    assert len(logs) == 1 and len(settings) == 1
    assert logs[0].stem.removeprefix("brep") == settings[0].stem.removeprefix("settings")
    log = logs[0].read_text()
    assert "INFO" in log and "Wrote meniscus.brep" in log
    assert "DEBUG" in log and "Centerline has" in log
    stderr = capsys.readouterr().err
    assert "Wrote meniscus.brep" in stderr
    assert "DEBUG" not in stderr


def test_pipeline_with_config_logs_to_file_named_for_it(in_tmp_path, meniscus_solid):
    from kneemesh.disk import write_brep

    write_brep(meniscus_solid, in_tmp_path / "solid.brep")
    config = write_config(in_tmp_path / "coarse.json", {"brep": "solid.brep", "max_size": 4.0})

    assert mesh_main(["meniscus", "--config", str(config)]) == 0

    assert "Meshed" in (in_tmp_path / "coarse.log").read_text()
    assert list(in_tmp_path.glob("mesh*.log")) == []


def test_pipeline_logs_errors_to_file(in_tmp_path, capsys):
    assert mesh_main(["meniscus", "--brep", "missing.brep"]) == 1

    assert "ERROR" in next(in_tmp_path.glob("mesh*.log")).read_text()
    assert "No such file" in capsys.readouterr().err


def test_pipeline_disables_package_logging_when_done(in_tmp_path, meniscus_label):
    mesh_main(["meniscus", "--brep", "missing.brep"])
    messages = []
    handler = logger.add(messages.append, level="DEBUG")
    try:
        signed_distance_field(meniscus_label)
    finally:
        logger.remove(handler)

    assert messages == []


@pytest.mark.parametrize("structure", list(STRUCTURES))
def test_every_structure_has_configs_with_defaults(structure):
    brep_type, mesh_type = STRUCTURES[structure]

    assert brep_type().output.suffix == ".brep"
    assert mesh_type().brep == brep_type().output


def test_configure_picks_the_structure_config(in_tmp_path):
    config_types = {structure: brep for structure, (brep, _) in STRUCTURES.items()}

    config, _, _ = configure(config_types, "test", ["tibial-cartilage", "--name", "lateral", "--label", "3"])

    assert config == TibialCartilageBrep(name="lateral", label=3)


def test_configure_requires_a_structure(in_tmp_path):
    with pytest.raises(SystemExit) as error:
        configure({"meniscus": MeniscusBrep}, "test", ["--spacing", "2"])

    assert error.value.code == 2


def test_configure_rejects_another_structures_options(in_tmp_path):
    config_types = {structure: brep for structure, (brep, _) in STRUCTURES.items()}

    with pytest.raises(SystemExit):
        configure(config_types, "test", ["femur", "--spacing", "2"])


@pytest.mark.parametrize("structure", [structure for structure in STRUCTURES if structure != "meniscus"])
def test_unimplemented_structures_report_an_error(in_tmp_path, structure, capsys):
    assert brep_main([structure]) == 1
    assert mesh_main([structure]) == 1

    assert "not implemented" in capsys.readouterr().err
