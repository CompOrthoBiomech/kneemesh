import pydantic
import pytest

from kneemesh.config import BaseConfig, FemoralCartilageBrep, MeniscusBrep, TibialCartilageBrep


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
    config = MeniscusBrep(label_map="knee.nii", labels='{"tibia": 1, "meniscus": 4}')
    assembled = MeniscusBrep(label_maps='{"meniscus": "m.nii", "tibia": "t.nii"}')

    assert config.labels == {"tibia": 1, "meniscus": 4}
    assert {name: str(path) for name, path in assembled.label_maps.items()} == {"meniscus": "m.nii", "tibia": "t.nii"}
    assert MeniscusBrep(labels="none").labels is None


def test_label_config_round_trips_through_json(tmp_path):
    config = TibialCartilageBrep(label_maps={"tibia": "t.nii", "medial": "m.nii"}, name="medial", interpolator="nearest_neighbor")
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())

    assert TibialCartilageBrep.from_json(path) == config


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"label_map": "knee.nii", "label_maps": {"meniscus": "m.nii"}}, "not both"),
        ({"labels": {"meniscus": 1}, "label_maps": {"meniscus": "m.nii"}}, "named by their keys"),
        ({"label_map": "knee.nii", "labels": {"tibia": 1}}, "No label named 'meniscus'"),
        ({"label_maps": {"tibia": "t.nii"}}, "No label named 'meniscus'"),
        ({"interpolator": "linear"}, "label_gaussian"),
        ({"padding": -1}, "greater than or equal"),
    ],
)
def test_label_config_rejects_inconsistent_sources(values, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        MeniscusBrep.model_validate(values)


def test_cartilage_config_needs_its_bone_labeled():
    assert FemoralCartilageBrep(labels={"femur": 1, "femoral_cartilage": 2}).bone == "femur"
    with pytest.raises(pydantic.ValidationError, match="No label named 'femur'"):
        FemoralCartilageBrep(labels={"femoral_cartilage": 2})
    with pytest.raises(pydantic.ValidationError, match="different labels"):
        TibialCartilageBrep(labels={"tibia": 1}, name="tibia")
