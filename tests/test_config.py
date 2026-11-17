from pathlib import Path

import pydantic
import pytest

from kneemesh.config import (
    STRUCTURES,
    BaseConfig,
    BoneBrep,
    BoneMesh,
    BrepMesh,
    CartilageBrep,
    CartilageMesh,
    FemoralCartilageBrep,
    FemoralCartilageMesh,
    LabelMapAssembly,
    LabelMapBrep,
    MeniscusBrep,
    MeniscusMesh,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


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
    assert BrepMesh(output=".vtk").output_path("medial_meniscus") == Path("medial_meniscus.vtk")
    assert BrepMesh(element_type="hex27", output="_{element_type}.inp").output_path("lateral_meniscus") == Path("lateral_meniscus_hex27.inp")


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


def test_meniscus_optimizes_by_default_and_tetrahedra_ignore_it():
    assert MeniscusMesh().optimize
    assert not CartilageMesh().optimize
    # Saved tetrahedral settings hold the meniscus's optimize too, so they must load again.
    assert BrepMesh.model_validate_json(BrepMesh().model_dump_json()).meniscus.optimize


def test_meniscus_clips_its_inner_edge_only_for_swept_hexahedra_by_default():
    assert MeniscusBrep().resolved_inner_clip_ratio == 0.0
    assert MeniscusBrep(hex_blocks=True).resolved_inner_clip_ratio == 0.0
    assert MeniscusBrep(hex_blocks=True, hex_design="swept_quads").resolved_inner_clip_ratio == 0.25
    assert MeniscusBrep(hex_blocks=True, hex_design="swept_quads", inner_clip_ratio=0.0).resolved_inner_clip_ratio == 0.0
    assert MeniscusBrep(hex_blocks=True, hex_design="swept_quads", inner_clip_height=1.5).resolved_inner_clip_ratio == 0.0
    assert MeniscusBrep(inner_clip_ratio=0.3).resolved_inner_clip_ratio == 0.3
    with pytest.raises(pydantic.ValidationError, match="not both"):
        MeniscusBrep(inner_clip_height=1.0, inner_clip_ratio=0.2)
    with pytest.raises(pydantic.ValidationError, match="planar caps or none"):
        MeniscusBrep(inner_clip_ratio=0.2, cap="smooth")


def test_meniscus_hex_blocks_default_to_butterfly_blocks():
    assert MeniscusBrep().hex_design == "butterfly"
    assert MeniscusBrep.model_validate({"hex_blocks": True, "hex_design": "swept_quads"}).hex_design == "swept_quads"
    with pytest.raises(pydantic.ValidationError):
        MeniscusBrep.model_validate({"hex_design": "o_grid"})


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


def test_cartilage_defaults_depend_on_hex_blocks():
    tetrahedral, hexahedral = CartilageBrep(), CartilageBrep(hex_blocks=True)

    assert (tetrahedral.resolved_perimeter_radius, tetrahedral.resolved_min_thickness, tetrahedral.resolved_hole_thickness) == (1.0, 0.1, 0.05)
    assert (hexahedral.resolved_perimeter_radius, hexahedral.resolved_min_thickness, hexahedral.resolved_hole_thickness) == (3.0, 0.2, 0.0)
    assert CartilageBrep(hex_blocks=True, perimeter_radius=2.0, min_thickness=0.3).resolved_perimeter_radius == 2.0


def test_cartilage_holes_must_be_thinner_than_the_minimum():
    with pytest.raises(pydantic.ValidationError, match="must be below min_thickness"):
        CartilageBrep(min_thickness=0.1, hole_thickness=0.1)
    # Hexahedral blocks ignore hole_thickness.
    assert CartilageBrep(hex_blocks=True, min_thickness=0.1, hole_thickness=0.2).resolved_hole_thickness == 0.0


@pytest.mark.parametrize("values", [{"side_cosine": 1.0}, {"perimeter_radius": 0.0}, {"min_thickness": 1.0}, {"perimeter_inset": -1.0}])
def test_cartilage_config_rejects_out_of_range_values(values):
    with pytest.raises(pydantic.ValidationError):
        CartilageBrep.model_validate(values)


def test_cartilage_optimization_is_optional():
    assert BrepMesh(element_type="hex8", cartilage=CartilageMesh(optimize=True)).cartilage.optimize


def test_sections_share_surface_and_meshing_settings():
    assert set(MeniscusBrep.model_fields) & set(CartilageBrep.model_fields) == {
        "sigma",
        "level",
        "smoothing_iterations",
        "smoothing_pass_band",
        "hex_blocks",
    }
    assert {"min_size", "max_size", "optimize", "preserve_volume"} <= set(MeniscusMesh.model_fields) & set(CartilageMesh.model_fields)


def test_meniscus_and_cartilage_weigh_aspect_ratios_by_default():
    assert MeniscusMesh().aspect_weight == 0.02
    assert CartilageMesh().aspect_weight == 0.5
    assert MeniscusMesh().warp_weight == 50.0
    assert CartilageMesh().warp_weight == 0.0
    with pytest.raises(pydantic.ValidationError):
        CartilageMesh(aspect_weight=-1.0)
    with pytest.raises(pydantic.ValidationError):
        MeniscusMesh(warp_weight=-1.0)


def test_femoral_cartilage_sections_extend_the_cartilage_settings():
    brep = LabelMapBrep(labels={"femur": 1, "femoral_cartilage": 2}, femoral_cartilage=FemoralCartilageBrep(hex_blocks=True))

    assert brep.femoral_cartilage.resolved_perimeter_radius == 3.0
    # Its solid is built like the other sheets', so the block partition's ring width is gone.
    with pytest.raises(pydantic.ValidationError):
        FemoralCartilageBrep.model_validate({"ring_width": 0.5})


def test_femoral_tetrahedra_default_to_the_element_size():
    assert FemoralCartilageMesh().resolved_max_size == 2.0
    assert FemoralCartilageMesh(element_size=3.0).resolved_max_size == 3.0
    assert FemoralCartilageMesh(element_size=2.0, max_size=3.0).resolved_max_size == 3.0
    assert FemoralCartilageMesh().aspect_weight == 0.5


def test_cartilage_hexahedra_settings():
    config = CartilageMesh()

    assert (config.element_size, config.elements_through, config.quad_algorithm, config.save_partition) == (2.0, 3, "frontal_delaunay", False)
    assert CartilageMesh.model_validate({"quad_algorithm": "quasi_structured"}).quad_algorithm == "quasi_structured"
    with pytest.raises(pydantic.ValidationError):
        CartilageMesh.model_validate({"quad_algorithm": "butterfly"})
    with pytest.raises(pydantic.ValidationError):
        CartilageMesh.model_validate({"elements_around": 8})
    with pytest.raises(pydantic.ValidationError, match="only to hexahedral"):
        BrepMesh(cartilage=CartilageMesh(save_partition=True))
    assert BrepMesh(element_types={"patellar_cartilage": "hex8"}, cartilage=CartilageMesh(save_partition=True)).cartilage.save_partition
    with pytest.raises(pydantic.ValidationError, match="femoral_cartilage.save_partition"):
        BrepMesh(element_types={"patellar_cartilage": "hex8"}, femoral_cartilage=FemoralCartilageMesh(save_partition=True))
    assert BrepMesh(element_type="hex8").partition_path("patellar_cartilage") == Path("patellar_cartilage_partition.brep")


def test_element_types_override_element_type_for_their_structures():
    config = BrepMesh.model_validate({"element_type": "tet4", "element_types": '{"medial_meniscus": "hex20"}', "output": "_{element_type}.vtu"})

    assert config.structure_element_type("medial_meniscus") == "hex20"
    assert config.structure_element_type("lateral_meniscus") == "tet4"
    assert config.output_path("medial_meniscus") == Path("medial_meniscus_hex20.vtu")
    assert config.output_path("lateral_meniscus") == Path("lateral_meniscus_tet4.vtu")
    medial = config.for_structure("medial_meniscus")
    assert (medial.element_shape, medial.element_order, medial.second_order_incomplete) == ("hexahedral", 2, True)
    assert config.for_structure("lateral_meniscus") is config
    assert BrepMesh().element_types is None
    assert BrepMesh.model_validate({"element_types": "none"}).element_types is None


def test_element_types_reject_bones_and_unknown_types():
    with pytest.raises(pydantic.ValidationError, match="femur.element_type"):
        BrepMesh.model_validate({"element_types": {"femur": "tet4"}})
    with pytest.raises(pydantic.ValidationError):
        BrepMesh.model_validate({"element_types": {"medial_meniscus": "quad4"}})
    with pytest.raises(pydantic.ValidationError):
        BrepMesh.model_validate({"element_types": {"meniscus": "hex8"}})


def test_bones_are_written_as_surfaces():
    config = LabelMapBrep(labels={"tibia": 1, "lateral_tibial_cartilage": 2}, output="_tet.brep", tibia=BoneBrep(output="_bone.stl"))

    assert config.output_path("tibia") == Path("tibia_bone.stl")
    assert config.output_path("lateral_tibial_cartilage") == Path("lateral_tibial_cartilage_tet.brep")
    assert LabelMapBrep().output_path("patella") == Path("patella.stl")
    with pytest.raises(pydantic.ValidationError, match=".stl"):
        BoneBrep(output=".brep")


def test_bone_surface_settings():
    config = BoneBrep()

    assert (config.shaft_cut_fraction, config.triangle_size) == (0.0, 1.0)
    for values in ({"shaft_cut_fraction": 1.0}, {"shaft_cut_fraction": -0.1}, {"triangle_size": 0.0}):
        with pytest.raises(pydantic.ValidationError):
            BoneBrep.model_validate(values)


@pytest.mark.parametrize(
    ("element_type", "shape", "order", "incomplete"),
    [
        ("tri3", "triangle", 1, False),
        ("tri6", "triangle", 2, False),
        ("quad4", "quadrilateral", 1, False),
        ("quad8", "quadrilateral", 2, True),
        ("quad9", "quadrilateral", 2, False),
        ("tet4", "tetrahedral", 1, False),
        ("tet10", "tetrahedral", 2, False),
    ],
)
def test_bones_mesh_with_their_own_element_type(element_type, shape, order, incomplete):
    config = BrepMesh(element_type="hex8", femur=BoneMesh(element_type=element_type))

    bone = config.bone("femur")
    assert (bone.element_shape, bone.element_order, bone.second_order_incomplete) == (shape, order, incomplete)
    assert config.output_path("femur") == Path(f"femur_{element_type}.msh")
    assert config.output_path("lateral_meniscus") == Path("lateral_meniscus_hex8.msh")


def test_bone_mesh_defaults_and_limits():
    config = BoneMesh()

    assert config.element_type == "tri3"
    assert (config.element_size, config.cortical_element_size, config.cortical_distance, config.cancellous_grad, config.max_size) == (
        2.0,
        1.0,
        3.0,
        0.5,
        4.0,
    )
    with pytest.raises(pydantic.ValidationError, match="must be at least cortical_element_size"):
        BoneMesh(cortical_element_size=2.0, max_size=1.0)
    for element_type in ("hex8", "tri10"):
        with pytest.raises(pydantic.ValidationError):
            BoneMesh.model_validate({"element_type": element_type})


# kneemesh-assemble writes <stem>_labels.json beside its output, untracked.
@pytest.mark.parametrize("path", sorted(path for path in EXAMPLES.glob("*.json") if not path.stem.endswith("_labels")), ids=lambda path: path.stem)
def test_example_configurations_are_valid(path):
    model = LabelMapAssembly if path.stem.endswith("_assemble") else LabelMapBrep if "_brep" in path.stem else BrepMesh
    assert isinstance(model.from_json(path), model)


def test_each_bone_has_its_own_settings():
    brep = LabelMapBrep.model_validate({"femur": {"shaft_cut_fraction": 0.4}, "tibia": {"shaft_cut_fraction": 0.3}})
    mesh = BrepMesh.model_validate({"femur": {"element_type": "tet4"}})

    assert [brep.bone(bone).shaft_cut_fraction for bone in ("femur", "tibia", "patella")] == [0.4, 0.3, 0.0]
    assert [mesh.bone(bone).element_type for bone in ("femur", "tibia", "patella")] == ["tet4", "tri3", "tri3"]
    assert mesh.output_path("femur") == Path("femur_tet4.msh")
    assert mesh.output_path("tibia") == Path("tibia_tri3.msh")
    with pytest.raises(ValueError, match="not a bone"):
        mesh.bone("lateral_meniscus")


@pytest.mark.parametrize(("assembled", "valid"), [("knee.vtu", True), ("knee.vtkhdf", True), ("knee.msh", False)])
def test_assembled_mesh_is_vtu_or_vtkhdf(assembled, valid):
    if valid:
        assert BrepMesh.model_validate({"assembled_mesh": assembled}).assembled_mesh == Path(assembled)
    else:
        with pytest.raises(pydantic.ValidationError, match="assembled_mesh needs extension"):
            BrepMesh.model_validate({"assembled_mesh": assembled})


def test_own_sections_layer_over_the_shared_section():
    config = LabelMapBrep.model_validate({"meniscus": {"spacing": 3.0, "points_per_slice": 20}, "lateral_meniscus": {"spacing": 2.0}})

    lateral = config.section("lateral_meniscus", MeniscusBrep)
    assert (lateral.spacing, lateral.points_per_slice) == (2.0, 20)
    assert config.section("medial_meniscus", MeniscusBrep) is config.meniscus
    assert config.own_sections() == ("lateral_meniscus",)
    assert LabelMapBrep.model_validate_json(config.model_dump_json()) == config
    assert config.section("femur", BoneBrep) is config.femur
    assert config.section("femoral_cartilage", CartilageBrep) is config.femoral_cartilage
    with pytest.raises(TypeError, match="CartilageBrep"):
        config.section("patellar_cartilage", MeniscusBrep)
    # Models layer as their dicts do.
    config = LabelMapBrep(meniscus=MeniscusBrep(spacing=3.0), lateral_meniscus=MeniscusBrep(points_per_slice=20))
    assert (config.section("lateral_meniscus", MeniscusBrep).spacing, config.section("lateral_meniscus", MeniscusBrep).points_per_slice) == (3.0, 20)


def test_own_sections_clear_exclusive_shared_fields_with_null():
    with pytest.raises(pydantic.ValidationError, match="not both"):
        LabelMapBrep.model_validate({"meniscus": {"inner_clip_ratio": 0.2}, "medial_meniscus": {"inner_clip_height": 1.0}})
    config = LabelMapBrep.model_validate(
        {"meniscus": {"inner_clip_ratio": 0.2}, "medial_meniscus": {"inner_clip_height": 1.0, "inner_clip_ratio": None}}
    )
    assert config.section("medial_meniscus", MeniscusBrep).inner_clip_height == 1.0


def test_save_partition_follows_the_structures_of_each_section():
    config = BrepMesh.model_validate({"element_types": {"patellar_cartilage": "hex8"}, "patellar_cartilage": {"save_partition": True}})
    assert config.section("patellar_cartilage", CartilageMesh).save_partition
    with pytest.raises(pydantic.ValidationError, match="lateral_tibial_cartilage.save_partition"):
        BrepMesh.model_validate({"element_types": {"patellar_cartilage": "hex8"}, "lateral_tibial_cartilage": {"save_partition": True}})
    # Once the hexahedral structure has its own section, only tetrahedral ones take the shared one.
    with pytest.raises(pydantic.ValidationError, match=r"\bcartilage\.save_partition"):
        BrepMesh.model_validate({"element_types": {"patellar_cartilage": "hex8"}, "patellar_cartilage": {}, "cartilage": {"save_partition": True}})
