import pytest
from pydantic import ValidationError

from kneemesh.config import STRUCTURES, BrepMesh, LabelMapBrep
from kneemesh.gui import settings


@pytest.mark.parametrize("kind", ["geometry", "mesh"])
def test_default_state_round_trips_to_the_default_config(kind):
    values = settings.config_values(kind, settings.state_values(kind))
    assert settings.MODELS[kind].model_validate(values) == settings.MODELS[kind]()


def test_every_structure_has_a_section_in_both_models():
    for structure in STRUCTURES:
        section = settings.STRUCTURE_SECTIONS[structure]
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
