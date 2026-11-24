import pytest
from pydantic import ValidationError

from kneemesh.config import STRUCTURE_SECTIONS, STRUCTURES, BrepMesh, LabelMapBrep, MeniscusBrep
from kneemesh.gui import settings


@pytest.mark.parametrize("kind", ["geometry", "mesh"])
def test_default_state_round_trips_to_the_default_config(kind):
    values = settings.config_values(kind, settings.state_values(kind))
    assert settings.MODELS[kind].model_validate(values) == settings.MODELS[kind]()


def test_every_structure_has_a_section_in_both_models():
    for structure in STRUCTURES:
        section = STRUCTURE_SECTIONS[structure]
        assert section in settings.section_names("geometry")
        assert section in settings.section_names("mesh")


def test_managed_fields_have_no_form():
    names = {spec.name for spec in settings.top_fields("geometry")}
    assert names == {"interpolator", "padding", "assembled_label_map", "output"}
    names = {spec.name for spec in settings.top_fields("mesh")}
    assert names == {"element_type", "output", "assembled_mesh"}


def test_widgets_follow_field_types():
    specs = {spec.key: spec for spec in settings.all_fields("geometry") + settings.all_fields("mesh")}
    sigma = specs["geo__meniscus__sigma"]
    assert (sigma.widget, sigma.optional, sigma.integer) == ("number", True, False)
    iterations = specs["geo__meniscus__smoothing_iterations"]
    assert (iterations.widget, iterations.integer, iterations.minimum) == ("number", True, 0.0)
    assert specs["geo__meniscus__smoothing_pass_band"].maximum == 2.0
    assert specs["geo__meniscus__hex_blocks"].widget == "switch"
    assert specs["geo__meniscus__hex_design"].items == ("swept_quads", "butterfly")
    assert specs["geo__interpolator"].items == ("label_gaussian", "nearest_neighbor")
    assert "frontal_delaunay" in specs["mesh__meniscus__algorithm_2d"].items
    assert specs["mesh__femur__element_type"].items[0] == "tri3"
    assert (specs["mesh__assembled_mesh"].widget, specs["mesh__assembled_mesh"].optional) == ("text", True)
    assert specs["mesh__meniscus__transverse_size"].description


def test_config_values_parse_text_and_read_empty_optionals_as_none():
    state = settings.state_values("geometry")
    state |= {"geo__meniscus__spacing": "2.5", "geo__meniscus__points_per_slice": "24", "geo__meniscus__sigma": "", "output_directory": "out"}
    config = LabelMapBrep.model_validate(settings.config_values("geometry", state))
    assert config.meniscus.spacing == 2.5
    assert config.meniscus.points_per_slice == 24
    assert config.meniscus.sigma is None
    assert str(config.output_directory) == "out"


def test_state_values_of_a_config():
    config = BrepMesh.model_validate({"element_type": "hex8", "meniscus": {"transverse_size": 0.5}, "output_directory": "meshes"})
    state = settings.state_values("mesh", config)
    assert state["mesh__element_type"] == "hex8"
    assert state["mesh__meniscus__transverse_size"] == 0.5
    assert state["output_directory"] == "meshes"


def test_validation_errors_by_key():
    state = settings.state_values("geometry") | {
        "geo__meniscus__spacing": "wide",
        "geo__meniscus__inner_clip_height": 1,
        "geo__meniscus__inner_clip_ratio": 0.2,
    }
    with pytest.raises(ValidationError) as caught:
        LabelMapBrep.model_validate(settings.config_values("geometry", state))
    errors = settings.validation_errors("geometry", caught.value)
    assert "geo__meniscus__spacing" in errors

    state = settings.state_values("geometry") | {"geo__meniscus__inner_clip_height": 1, "geo__meniscus__inner_clip_ratio": 0.2}
    with pytest.raises(ValidationError) as caught:
        LabelMapBrep.model_validate(settings.config_values("geometry", state))
    assert "not both" in settings.validation_errors("geometry", caught.value)["geo__meniscus"]

    state = settings.state_values("geometry") | {"geo__output": ".stl"}
    with pytest.raises(ValidationError) as caught:
        LabelMapBrep.model_validate(settings.config_values("geometry", state))
    assert ".brep" in settings.validation_errors("geometry", caught.value)["geo"]


def test_humanize():
    assert settings.humanize("points_per_slice") == "Points per slice"


def test_own_sections_start_from_the_shared_section_and_count_only_when_own():
    config = LabelMapBrep.model_validate({"meniscus": {"spacing": 3.0}, "medial_meniscus": {"spacing": 2.0}})
    state = settings.state_values("geometry", config)
    assert (state["geo__lateral_meniscus__spacing"], state["geo__medial_meniscus__spacing"]) == (3.0, 2.0)
    assert settings.own_sections(config) == ["medial_meniscus"]

    state["geo__lateral_meniscus__spacing"] = 2.5
    assert "lateral_meniscus" not in settings.config_values("geometry", state, ["medial_meniscus"])
    values = settings.config_values("geometry", state, ["medial_meniscus", "lateral_meniscus"])
    rebuilt = LabelMapBrep.model_validate(values)
    assert (rebuilt.section("lateral_meniscus", MeniscusBrep).spacing, rebuilt.section("medial_meniscus", MeniscusBrep).spacing) == (2.5, 2.0)


def test_structure_sections_and_sharing():
    assert settings.structure_section("lateral_meniscus") == "meniscus"
    assert settings.structure_section("lateral_meniscus", ["lateral_meniscus"]) == "lateral_meniscus"
    assert settings.structure_section("femur", ["femur"]) == "femur"
    assert settings.sharing_structures("patellar_cartilage", ["medial_tibial_cartilage"]) == ["lateral_tibial_cartilage"]
    assert settings.sharing_structures("patellar_cartilage", ["patellar_cartilage"]) == []
    assert "geo__patellar_cartilage__side_cosine" in {spec.key for spec in settings.all_fields("geometry")}


def test_overclosure_has_forms_on_the_settings_page_only():
    assert settings.OVERCLOSURE_SECTION in settings.section_names("geometry")
    assert settings.OVERCLOSURE_SECTION not in settings.structure_section_names("geometry")
    specs = {spec.name: spec for spec in settings.section_fields("geometry", settings.OVERCLOSURE_SECTION)}
    assert (specs["enabled"].widget, specs["clearance"].widget, specs["priority"].widget) == ("switch", "number", "order")
    assert sorted(specs["priority"].items) == sorted(STRUCTURES)


def test_overclosure_priority_round_trips_through_the_state():
    priority = [*LabelMapBrep().overclosure.priority]
    priority[4], priority[5] = priority[5], priority[4]
    config = LabelMapBrep.model_validate({"overclosure": {"priority": priority, "clearance": 0.1}})

    state = settings.state_values("geometry", config)

    assert state["geo__overclosure__priority"] == priority
    assert LabelMapBrep.model_validate(settings.config_values("geometry", state)).overclosure == config.overclosure


def test_overclosure_priority_errors_are_reported_on_its_form():
    state = settings.state_values("geometry")
    state["geo__overclosure__priority"] = ["lateral_meniscus", *[name for name in state["geo__overclosure__priority"] if name != "lateral_meniscus"]]

    with pytest.raises(ValidationError) as error:
        LabelMapBrep.model_validate(settings.config_values("geometry", state))

    assert "must come first" in settings.validation_errors("geometry", error.value)[settings.section_key("geometry", "overclosure")]
