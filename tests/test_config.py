from pathlib import Path

import pydantic
import pytest

from kneemesh.config import STRUCTURES, BaseConfig, BrepMesh, LabelMapBrep, MeniscusBrep, MeniscusMesh


class ExampleConfig(BaseConfig):
    name: str
    label: int = 1


def test_from_json_parses_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"name": "medial", "label": 2}')

    assert ExampleConfig.from_json(path) == ExampleConfig(name="medial", label=2)


def test_from_json_rejects_unknown_field(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"name": "medial", "lable": 2}')

    with pytest.raises(pydantic.ValidationError):
        ExampleConfig.from_json(path)


def test_from_json_rejects_malformed_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"name": ')

    with pytest.raises(pydantic.ValidationError):
        ExampleConfig.from_json(path)


def test_labels_and_label_maps_parse_json_text():
    config = LabelMapBrep.model_validate({"label_map": "knee.nii", "labels": '{"tibia": 1, "medial_meniscus": 4}'})
    assembled = LabelMapBrep.model_validate({"label_maps": '{"medial_meniscus": "m.nii", "tibia": "t.nii"}'})

    assert config.labels == {"tibia": 1, "medial_meniscus": 4}
    assert assembled.label_maps == {"medial_meniscus": Path("m.nii"), "tibia": Path("t.nii")}
    assert assembled.structures == ("medial_meniscus", "tibia")
    assert LabelMapBrep.model_validate({"labels": "none"}).labels is None


def test_label_config_round_trips_through_json(tmp_path):
    config = LabelMapBrep(
        label_maps={"tibia": Path("t.nii"), "medial_tibial_cartilage": Path("m.nii")},
        interpolator="nearest_neighbor",
        meniscus=MeniscusBrep(points_per_slice=10),
    )
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())

    assert LabelMapBrep.from_json(path) == config


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"label_map": "knee.nii", "label_maps": {"medial_meniscus": "m.nii"}}, "not both"),
        ({"labels": {"medial_meniscus": 1}, "label_maps": {"medial_meniscus": "m.nii"}}, "named by their keys"),
        ({"label_map": "knee.nii"}, "Give labels"),
        ({"label_map": "knee.nii", "assembled_label_map": "out.nii", "labels": {"tibia": 1}}, "only when assembling"),
        ({"labels": {"meniscus": 1}}, "lateral_meniscus"),
        ({"label_maps": {"rim": "r.nii"}}, "lateral_meniscus"),
        ({"interpolator": "linear"}, "label_gaussian"),
        ({"padding": -1}, "greater than or equal"),
        ({"output": ".msh"}, "extension .brep"),
        ({"output": "/out.brep"}, "cannot hold a directory"),
    ],
)
def test_label_config_rejects_inconsistent_sources(values, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        LabelMapBrep.model_validate(values)


def test_cartilage_needs_its_bone_labeled():
    assert LabelMapBrep(labels={"femur": 1, "femoral_cartilage": 2}).structures == ("femur", "femoral_cartilage")
    with pytest.raises(pydantic.ValidationError, match="needs its bone, femur"):
        LabelMapBrep(labels={"femoral_cartilage": 2})
    with pytest.raises(pydantic.ValidationError, match="needs its bone, tibia"):
        LabelMapBrep(label_maps={"lateral_tibial_cartilage": Path("l.nii")})


def test_brep_output_is_a_suffix_on_each_structure():
    config = LabelMapBrep(labels={"medial_meniscus": 1}, output="_hex.brep", output_directory=Path("out"))

    assert config.output_path("medial_meniscus") == Path("out/medial_meniscus_hex.brep")
    assert LabelMapBrep().output_path("lateral_meniscus") == Path("lateral_meniscus.brep")


@pytest.mark.parametrize(
    ("element_type", "shape", "order", "incomplete"),
    [
        ("hex8", "hexahedral", 1, False),
        ("hex20", "hexahedral", 2, True),
        ("hex27", "hexahedral", 2, False),
        ("tet4", "tetrahedral", 1, False),
        ("tet10", "tetrahedral", 2, False),
    ],
)
def test_mesh_output_defaults_to_the_element_type(element_type, shape, order, incomplete):
    config = BrepMesh(element_type=element_type)

    assert (config.element_shape, config.element_order, config.second_order_incomplete) == (shape, order, incomplete)
    assert config.output_path("lateral_meniscus") == Path(f"lateral_meniscus_{element_type}.msh")


def test_mesh_output_suffix_may_omit_or_reuse_the_element_type():
    assert BrepMesh(output=".vtk").output_path("tibia") == Path("tibia.vtk")
    assert BrepMesh(element_type="hex27", output="_{element_type}.inp").output_path("femur") == Path("femur_hex27.inp")


@pytest.mark.parametrize(("output", "message"), [("_{order}.msh", "only the {element_type}"), ("_mesh", "file extension"), ("_{", "placeholder")])
def test_mesh_config_rejects_bad_suffixes(output, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        BrepMesh(output=output)


def test_mesh_config_restricts_brep_keys_and_element_types():
    assert BrepMesh.model_validate({"breps": '{"medial_meniscus": "m.brep"}'}).breps == {"medial_meniscus": Path("m.brep")}
    with pytest.raises(pydantic.ValidationError):
        BrepMesh.model_validate({"breps": {"meniscus": "m.brep"}})
    with pytest.raises(pydantic.ValidationError):
        BrepMesh.model_validate({"element_type": "hex64"})


def test_mesh_optimization_applies_only_to_hexahedra():
    assert BrepMesh(element_type="hex8", meniscus=MeniscusMesh(optimize=True)).meniscus.optimize
    with pytest.raises(pydantic.ValidationError, match="only to hexahedral"):
        BrepMesh(meniscus=MeniscusMesh(optimize=True))


def test_structures_list_every_name():
    assert STRUCTURES == (
        "lateral_meniscus",
        "medial_meniscus",
        "patellar_cartilage",
        "femoral_cartilage",
        "lateral_tibial_cartilage",
        "medial_tibial_cartilage",
        "femur",
        "tibia",
        "patella",
    )
