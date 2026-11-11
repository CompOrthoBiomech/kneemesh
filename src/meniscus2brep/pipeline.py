"""Command-line pipelines for building B-reps and meshing them."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from loguru import logger
from pydantic import ValidationError

from meniscus2brep.config import BaseConfig, BrepConfig, MeshConfig
from meniscus2brep.disk import read_brep, read_label_map, write_brep, write_mesh
from meniscus2brep.geo import BRep, contours_to_brep, contours_to_hex_blocks
from meniscus2brep.mesh import Mesh, hexahedral_mesh, tetrahedral_mesh
from meniscus2brep.vol import centerline, extract_isocontour, signed_distance_field, slice_isocontour, smooth_isocontour


def run_brep(config: BrepConfig) -> BRep:
    """Build a B-rep from a LabelMap as config describes and write it to config.output."""
    logger.info("Reading LabelMap {}", config.label_map)
    label_map = read_label_map(config.label_map)
    logger.info("Fitting the centerline of label {}", config.label)
    curve = centerline(label_map, config.label)
    logger.info("Computing the signed distance field")
    distance_field = signed_distance_field(label_map, config.label, config.sigma)
    logger.info("Extracting the surface at level {}", config.level)
    surface = extract_isocontour(distance_field, config.level)
    logger.info("Smoothing the surface with {} iterations and pass band {}", config.smoothing_iterations, config.smoothing_pass_band)
    surface = smooth_isocontour(surface, config.smoothing_iterations, config.smoothing_pass_band)
    logger.info("Slicing every {} along the centerline", config.spacing)
    slices = slice_isocontour(surface, curve, config.spacing, config.points_per_slice)
    if config.hex_blocks:
        logger.info("Building hexahedral blocks through {} slices", len(slices))
        brep = contours_to_hex_blocks(slices, config.spline_order)
    else:
        logger.info("Lofting through {} slices with {} caps", len(slices), config.cap)
        brep = contours_to_brep(slices, config.spline_order, config.cap, config.cap_offset)
    write_brep(brep, config.output)
    return brep


def run_mesh(config: MeshConfig) -> Mesh:
    """Mesh a B-rep as config describes and write it to config.output."""
    logger.info("Reading B-rep {}", config.brep)
    brep = read_brep(config.brep)
    logger.info("Meshing with {} elements of order {}", config.element, config.element_order)
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
    counts = ", ".join(f"{len(connectivity)} {name}" for name, connectivity in mesh.elements.items())
    logger.info("Meshed {} nodes and {}", len(mesh.nodes), counts)
    write_mesh(mesh, config.output)
    return mesh


class Configuration[ConfigType: BaseConfig](NamedTuple):
    """A pipeline's configuration and the files it came from and was saved to."""

    config: ConfigType
    source: Path | None
    saved: Path | None


def configure[ConfigType: BaseConfig](
    config_type: type[ConfigType], description: str, argv: Sequence[str] | None = None, timestamp: str | None = None
) -> Configuration[ConfigType]:
    """Build a configuration from an optional JSON file and command-line overrides.

    Every field of config_type is an option, such as --points-per-slice 10.
    Values given on the command line override those in --config. The
    configuration is saved so the run can be repeated: as
    settings<timestamp>.json in the working directory when there is no
    --config, or as <config stem>_modified.json beside the file when
    overrides change it. A config file used without overrides is not copied.

    Args:
        config_type: The configuration model.
        description: Description for the command's help.
        argv: Command-line arguments, by default sys.argv[1:].
        timestamp: Timestamp for the settings file name, by default the current time.

    Returns:
        The configuration, the --config file if any, and the path the
        configuration was saved to, or None if it was not saved.
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
        saved = Path(f"settings{timestamp or _timestamp()}.json")
    elif arguments:
        saved = config_path.with_name(f"{config_path.stem}_modified.json")
    else:
        saved = None
    if saved is not None:
        saved.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return Configuration(config, config_path, saved)


def brep_main(argv: Sequence[str] | None = None) -> int:
    """Run the B-rep pipeline from the command line."""
    timestamp = _timestamp()
    setup = configure(BrepConfig, "Build a B-rep of a meniscus from a segmented LabelMap.", argv, timestamp)
    return _run(lambda: run_brep(setup.config), setup, Path(f"brep{timestamp}.log"))


def mesh_main(argv: Sequence[str] | None = None) -> int:
    """Run the meshing pipeline from the command line."""
    timestamp = _timestamp()
    setup = configure(MeshConfig, "Mesh a meniscus B-rep with tetrahedra or hexahedra.", argv, timestamp)
    return _run(lambda: run_mesh(setup.config), setup, Path(f"mesh{timestamp}.log"))


def _timestamp() -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}"


def _run(step: Callable[[], object], setup: Configuration, default_log: Path) -> int:
    """Run a pipeline step, logging to stderr and to a file named for its configuration.

    The log file is <config stem>.log beside the --config file, or default_log
    in the working directory without one.
    """
    log_path = setup.source.with_suffix(".log") if setup.source is not None else default_log
    logger.remove()
    handlers = [
        logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> {message}"),
        logger.add(log_path, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"),
    ]
    logger.enable("meniscus2brep")
    try:
        logger.info("Logging to {}", log_path)
        if setup.source is not None:
            logger.info("Read configuration from {}", setup.source)
        if setup.saved is not None:
            logger.info("Saved configuration to {}", setup.saved)
        logger.debug("Configuration: {}", setup.config.model_dump_json())
        try:
            step()
        except (ValueError, OSError) as error:
            logger.error("{}", error)
            return 1
        except Exception:
            logger.exception("Pipeline failed unexpectedly")
            raise
        logger.info("Wrote {}", setup.config.output)
        return 0
    finally:
        logger.disable("meniscus2brep")
        for handler in handlers:
            logger.remove(handler)
