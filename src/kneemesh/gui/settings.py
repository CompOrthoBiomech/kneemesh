"""Form widgets for the B-rep and mesh configurations, derived from their pydantic models, and their values in the GUI's flat state.

trame tracks changes to top-level state keys only, so every field shown in
a form gets its own key: ``geo__meniscus__spacing`` for a section's field
and ``geo__output`` for a top-level one. ``output_directory`` is one key that
both configurations share. Fields the GUI manages itself, such as labels,
breps, and element_types, have no form.
"""

import types
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeAliasType, get_args, get_origin

from annotated_types import Ge, Gt, Le, Lt
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from kneemesh.config import BaseConfig, BrepMesh, LabelMapBrep, Structure

type Kind = Literal["geometry", "mesh"]
type Widget = Literal["switch", "select", "number", "text"]

MODELS: dict[Kind, type[BaseConfig]] = {"geometry": LabelMapBrep, "mesh": BrepMesh}
PREFIXES: dict[Kind, str] = {"geometry": "geo", "mesh": "mesh"}

OUTPUT_DIRECTORY_KEY = "output_directory"
"""The state key of the output directory, which the B-rep and mesh configurations share."""

# Fields the GUI fills in itself, from the label tree, the structures' own pages, or the shared output directory.
_MANAGED: dict[Kind, frozenset[str]] = {
    "geometry": frozenset({"label_map", "labels", "label_maps", "output_directory"}),
    "mesh": frozenset({"breps", "element_types", "output_directory"}),
}

STRUCTURE_SECTIONS: dict[Structure, str] = {
    "lateral_meniscus": "meniscus",
    "medial_meniscus": "meniscus",
    "lateral_tibial_cartilage": "cartilage",
    "medial_tibial_cartilage": "cartilage",
    "patellar_cartilage": "cartilage",
    "femoral_cartilage": "femoral_cartilage",
    "femur": "femur",
    "tibia": "tibia",
    "patella": "patella",
}
"""The configuration section, in both models, that holds each structure's settings."""


@dataclass(frozen=True)
class FieldSpec:
    """How to show one configuration field.

    Attributes:
        key: The field's state key.
        name: The field's name in its model.
        section: The section holding the field, or None for a top-level field.
        label: The widget's label.
        description: The field's description, shown as a tooltip.
        widget: The widget to edit it with.
        items: The choices of a select.
        optional: Whether the field takes None, shown as an empty field.
        integer: Whether a number field takes integers.
        minimum: The smallest value allowed, or None.
        maximum: The largest value allowed, or None.
    """

    key: str
    name: str
    section: str | None
    label: str
    description: str
    widget: Widget
    items: tuple[str, ...] = ()
    optional: bool = False
    integer: bool = False
    minimum: float | None = None
    maximum: float | None = None


def state_key(kind: Kind, section: str | None, name: str) -> str:
    """The state key of a field of kind's model, in a section or at the top level."""
    if name == OUTPUT_DIRECTORY_KEY and section is None:
        return OUTPUT_DIRECTORY_KEY
    return f"{PREFIXES[kind]}__{section}__{name}" if section is not None else f"{PREFIXES[kind]}__{name}"


def section_key(kind: Kind, section: str) -> str:
    """The key under which errors of a whole section, from its model's validators, are reported."""
    return f"{PREFIXES[kind]}__{section}"


def section_names(kind: Kind) -> list[str]:
    """The sections of kind's model, such as meniscus and femur."""
    return [name for name, field in MODELS[kind].model_fields.items() if _section_model(field.annotation) is not None]


def section_fields(kind: Kind, section: str) -> list[FieldSpec]:
    """The fields of a section of kind's model, in declaration order."""
    model = _section_model(MODELS[kind].model_fields[section].annotation)
    assert model is not None, f"{section} is not a section"
    return [_spec(kind, section, name, field) for name, field in model.model_fields.items()]


def top_fields(kind: Kind) -> list[FieldSpec]:
    """The top-level fields of kind's model with forms: those that are not sections or managed by the GUI."""
    return [
        _spec(kind, None, name, field)
        for name, field in MODELS[kind].model_fields.items()
        if _section_model(field.annotation) is None and name not in _MANAGED[kind]
    ]


def all_fields(kind: Kind) -> list[FieldSpec]:
    """Every field of kind's model with a form."""
    return top_fields(kind) + [spec for section in section_names(kind) for spec in section_fields(kind, section)]


def state_values(kind: Kind, config: BaseConfig | None = None) -> dict[str, Any]:
    """The state of every form field of kind's model, from config or the model's defaults, and the shared output directory."""
    values = (config if config is not None else MODELS[kind]()).model_dump(mode="json")
    state = {OUTPUT_DIRECTORY_KEY: values[OUTPUT_DIRECTORY_KEY]}
    for spec in all_fields(kind):
        state[spec.key] = values[spec.section][spec.name] if spec.section is not None else values[spec.name]
    return state


def config_values(kind: Kind, state: Mapping[str, Any]) -> dict[str, Any]:
    """Unvalidated values of kind's model from the state of its forms, with sections nested and empty optional fields as None.

    Numbers typed into a form arrive as text, which pydantic's lax mode parses.
    """
    values: dict[str, Any] = {OUTPUT_DIRECTORY_KEY: state.get(OUTPUT_DIRECTORY_KEY, ".")}
    for spec in all_fields(kind):
        if spec.key not in state:
            continue
        value = state[spec.key]
        if spec.optional and (value is None or (isinstance(value, str) and not value.strip())):
            value = None
        if spec.section is None:
            values[spec.name] = value
        else:
            values.setdefault(spec.section, {})[spec.name] = value
    return values


def validation_errors(kind: Kind, error: ValidationError) -> dict[str, str]:
    """Each of a ValidationError's messages by the state key it concerns.

    An error of a field goes under its key, an error of a whole section,
    from its model's validators, under section_key, and any other under the
    model's prefix, such as "geo".
    """
    errors: dict[str, str] = {}
    for detail in error.errors():
        location = [str(part) for part in detail["loc"]]
        message = detail["msg"].removeprefix("Value error, ")
        if len(location) >= 2 and location[0] in section_names(kind):
            key = state_key(kind, location[0], location[1])
        elif len(location) == 1 and location[0] in section_names(kind):
            key = section_key(kind, location[0])
        elif len(location) == 1:
            key = state_key(kind, None, location[0])
        else:
            key = PREFIXES[kind]
        errors[key] = f"{errors[key]}; {message}" if key in errors else message
    return errors


def humanize(name: str) -> str:
    """A field or structure name as a label: lateral_meniscus as Lateral meniscus."""
    return name.replace("_", " ").capitalize()


def _section_model(annotation: object) -> type[BaseConfig] | None:
    return annotation if isinstance(annotation, type) and issubclass(annotation, BaseConfig) else None


def _spec(kind: Kind, section: str | None, name: str, field: FieldInfo) -> FieldSpec:
    annotation, optional = _unwrap_optional(field.annotation)
    minimum = maximum = None
    for constraint in field.metadata:
        if isinstance(constraint, Ge | Gt):
            bound = constraint.ge if isinstance(constraint, Ge) else constraint.gt
            minimum = float(bound) if isinstance(bound, int | float) else None
        elif isinstance(constraint, Le | Lt):
            bound = constraint.le if isinstance(constraint, Le) else constraint.lt
            maximum = float(bound) if isinstance(bound, int | float) else None
    key, label, description = state_key(kind, section, name), humanize(name), field.description or ""
    if annotation is bool:
        widget: Widget = "switch"
    elif _choices(annotation) is not None:
        widget = "select"
    elif annotation is int or annotation is float:
        widget = "number"
    elif annotation is str or annotation is Path:
        widget = "text"
    else:
        raise TypeError(f"No widget for {name}: {annotation!r}")
    return FieldSpec(
        key=key,
        name=name,
        section=section,
        label=label,
        description=description,
        widget=widget,
        items=_choices(annotation) or (),
        optional=optional,
        integer=annotation is int,
        minimum=minimum,
        maximum=maximum,
    )


def _unwrap_optional(annotation: object) -> tuple[object, bool]:
    """The annotation without None, and whether it allowed None."""
    if isinstance(annotation, TypeAliasType):
        annotation = annotation.__value__
    if get_origin(annotation) is types.UnionType:
        arguments = [argument for argument in get_args(annotation) if argument is not type(None)]
        if len(arguments) == 1:
            return arguments[0], len(arguments) < len(get_args(annotation))
    return annotation, False


def _choices(annotation: object) -> tuple[str, ...] | None:
    """The values of a Literal or an Enum, or None for other types."""
    if isinstance(annotation, TypeAliasType):
        annotation = annotation.__value__
    if get_origin(annotation) is Literal:
        return tuple(str(argument) for argument in get_args(annotation))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple(str(member.value) for member in annotation)
    return None
