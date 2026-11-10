"""Command-line pipelines for building B-reps and meshing them."""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from meniscus2brep.config import BaseConfig, BrepConfig, MeshConfig
from meniscus2brep.disk import read_brep, read_label_map, write_brep, write_mesh
from meniscus2brep.geo import BRep, contours_to_brep, contours_to_hex_blocks
from meniscus2brep.mesh import Mesh, hexahedral_mesh, tetrahedral_mesh
from meniscus2brep.vol import centerline, extract_isocontour, signed_distance_field, slice_isocontour


def run_brep(config: BrepConfig) -> BRep:
    """Build a B-rep from a LabelMap as config describes and write it to config.output."""
    label_map = read_label_map(config.label_map)
    curve = centerline(label_map, config.label)
    distance_field = signed_distance_field(label_map, config.label, config.sigma)
    slices = slice_isocontour(extract_isocontour(distance_field, config.level), curve, config.spacing, config.points_per_slice)
    if config.hex_blocks:
        brep = contours_to_hex_blocks(slices, config.spline_order)
    else:
        brep = contours_to_brep(slices, config.spline_order, config.cap, config.cap_offset)
    write_brep(brep, config.output)
    return brep


def run_mesh(config: MeshConfig) -> Mesh:
    """Mesh a B-rep as config describes and write it to config.output."""
    brep = read_brep(config.brep)
    if config.element == "hexahedral":
        mesh = hexahedral_mesh(brep, config.elements_per_edge, config.elements_along, config.element_order, config.second_order_incomplete)
    else:
        mesh = tetrahedral_mesh(
            brep,
            config.min_size,
            config.max_size,
            config.elements_per_circle,
            config.algorithm_2d,
            config.algorithm_3d,
            config.element_order,
            config.second_order_incomplete,
        )
    write_mesh(mesh, config.output)
    return mesh


def configure[ConfigType: BaseConfig](
    config_type: type[ConfigType], description: str, argv: Sequence[str] | None = None
) -> tuple[ConfigType, Path | None]:
    """Build a configuration from an optional JSON file and command-line overrides.

    Every field of config_type is an option, such as --points-per-slice 10.
    Values given on the command line override those in --config. The
    configuration is saved so the run can be repeated: as
    settings<timestamp>.json in the working directory when there is no
    --config, or as <config stem>_modified.json beside the file when
    overrides change it. A config file used without overrides is not copied.

    Returns:
        The configuration and the path it was saved to, or None if not saved.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="JSON configuration; options given here override its values")
    defaults = config_type().model_dump(mode="json")
    for name, field in config_type.model_fields.items():
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=argparse.SUPPRESS, help=f"{field.description} (default: {defaults[name]})")
    arguments = vars(parser.parse_args(argv))
    config_path = arguments.pop("config")

    values = json.loads(config_path.read_text(encoding="utf-8")) if config_path is not None else {}
    try:
        config = config_type.model_validate({**values, **arguments})
    except ValidationError as error:
        parser.error(str(error))

    if config_path is None:
        saved = Path(f"settings{datetime.now():%Y%m%d_%H%M%S}.json")
    elif arguments:
        saved = config_path.with_name(f"{config_path.stem}_modified.json")
    else:
        saved = None
    if saved is not None:
        saved.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return config, saved


def brep_main(argv: Sequence[str] | None = None) -> int:
    """Run the B-rep pipeline from the command line."""
    config, saved = configure(BrepConfig, "Build a B-rep of a meniscus from a segmented LabelMap.", argv)
    return _run(lambda: run_brep(config), config.output, saved)


def mesh_main(argv: Sequence[str] | None = None) -> int:
    """Run the meshing pipeline from the command line."""
    config, saved = configure(MeshConfig, "Mesh a meniscus B-rep with tetrahedra or hexahedra.", argv)
    return _run(lambda: run_mesh(config), config.output, saved)


def _run(step, output: Path, saved: Path | None) -> int:
    if saved is not None:
        print(f"Saved configuration to {saved}")
    try:
        step()
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {output}")
    return 0
