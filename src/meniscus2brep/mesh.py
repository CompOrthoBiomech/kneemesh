"""Tetrahedral and hexahedral meshing of B-reps with gmsh."""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import gmsh
import numpy as np

from meniscus2brep.geo import BRep, gmsh_session


class _GmshAlgorithm(StrEnum):
    """A gmsh meshing algorithm, valued by a readable name and carrying gmsh's option number.

    Lookup by value ignores case, so "HXT" and "hxt" both name the same member.
    """

    number: int

    def __new__(cls, name: str, number: int) -> Self:
        member = str.__new__(cls, name)
        member._value_ = name
        member.number = number
        return member

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        if isinstance(value, str):
            return next((member for member in cls if member.value == value.lower()), None)
        return None


class Algorithm2D(_GmshAlgorithm):
    """gmsh surface meshing algorithms that produce triangles (option Mesh.Algorithm).

    gmsh's "initial mesh only" algorithm crashes the process and BAMG fails on
    these shapes, so both are omitted.
    """

    MESH_ADAPT = "mesh_adapt", 1
    AUTOMATIC = "automatic", 2
    DELAUNAY = "delaunay", 5
    FRONTAL_DELAUNAY = "frontal_delaunay", 6


class Algorithm3D(_GmshAlgorithm):
    """gmsh volume meshing algorithms that produce tetrahedra (option Mesh.Algorithm3D).

    gmsh's R-tree algorithm is omitted because it writes a debug file into the
    working directory.
    """

    DELAUNAY = "delaunay", 1
    FRONTAL = "frontal", 4
    MMG3D = "mmg3d", 7
    HXT = "hxt", 10


@dataclass(frozen=True, eq=False)
class Mesh:
    """A volume mesh.

    Attributes:
        nodes: (N, 3) physical node coordinates.
        elements: Connectivity keyed by gmsh element name, such as
            "Tetrahedron 4" or "Hexahedron 27". Each array is (M, nodes per
            element) of 0-based indices into nodes, in gmsh's node ordering.
        element_types: gmsh's element type number for each key of elements.
    """

    nodes: np.ndarray
    elements: dict[str, np.ndarray]
    element_types: dict[str, int]


def tetrahedral_mesh(
    brep: BRep,
    min_size: float = 0.0,
    max_size: float | None = None,
    elements_per_circle: int = 0,
    algorithm_2d: Algorithm2D = Algorithm2D.FRONTAL_DELAUNAY,
    algorithm_3d: Algorithm3D = Algorithm3D.DELAUNAY,
    element_order: int = 1,
    second_order_incomplete: bool = False,
) -> Mesh:
    """Mesh a solid B-rep with tetrahedra.

    Args:
        brep: A solid, such as contours_to_brep with a cap.
        min_size: Smallest element size, in physical units.
        max_size: Largest element size, in physical units. By default gmsh's
            size from the shape's extent applies.
        elements_per_circle: Size elements so that a full circle of the local
            surface curvature would hold this many. 0 disables curvature sizing.
        algorithm_2d: Algorithm for the boundary triangles.
        algorithm_3d: Algorithm for the tetrahedra.
        element_order: Polynomial order of the elements. 2 gives 10-node tetrahedra.
        second_order_incomplete: Omit interior nodes from high-order elements.
            Second-order tetrahedra have none, so this only changes orders 3
            and up, such as 16-node instead of 20-node third-order tetrahedra.

    Returns:
        The tetrahedral mesh.

    Raises:
        ValueError: If a size or count is out of range or brep has no volume.
    """
    if min_size < 0:
        raise ValueError(f"min_size must not be negative, got {min_size}")
    if max_size is not None and max_size <= min_size:
        raise ValueError(f"max_size must exceed min_size, got {max_size} and {min_size}")
    if elements_per_circle < 0:
        raise ValueError(f"elements_per_circle must not be negative, got {elements_per_circle}")
    _require_element_order(element_order)

    options = {
        "Mesh.MeshSizeMin": min_size,
        "Mesh.MeshSizeFromCurvature": elements_per_circle,
        "Mesh.Algorithm": Algorithm2D(algorithm_2d).number,
        "Mesh.Algorithm3D": Algorithm3D(algorithm_3d).number,
        "Mesh.ElementOrder": element_order,
        "Mesh.SecondOrderIncomplete": second_order_incomplete,
    }
    if max_size is not None:
        options["Mesh.MeshSizeMax"] = max_size

    with gmsh_session(), _gmsh_options(options):
        _import_solid(brep)
        gmsh.model.mesh.generate(3)
        return _volume_mesh()


def hexahedral_mesh(
    brep: BRep,
    elements_per_edge: int = 4,
    elements_along: int = 20,
    element_order: int = 1,
    second_order_incomplete: bool = False,
) -> Mesh:
    """Mesh a solid of six-sided blocks with hexahedra, using transfinite surfaces and volumes.

    Every block must have six faces, such as the three blocks from
    contours_to_hex_blocks. Edges on the planar end caps get
    elements_per_edge elements, and the edges running between the caps get
    elements_along, so each block is a grid of elements_per_edge by
    elements_per_edge quadrilaterals swept elements_along times.

    Args:
        brep: A solid of six-sided blocks with planar caps.
        elements_per_edge: Elements along each edge of a cap.
        elements_along: Elements along the solid between the caps.
        element_order: Polynomial order of the elements. 2 gives 27-node hexahedra.
        second_order_incomplete: Omit face and interior nodes from high-order
            elements, giving 20-node instead of 27-node second-order hexahedra.

    Returns:
        The hexahedral mesh.

    Raises:
        ValueError: If a count is below 1, or brep is not a solid of six-sided
            blocks with planar caps.
    """
    if elements_per_edge < 1 or elements_along < 1:
        raise ValueError(f"Element counts must be at least 1, got {elements_per_edge} and {elements_along}")
    _require_element_order(element_order)

    options = {"Mesh.ElementOrder": element_order, "Mesh.SecondOrderIncomplete": second_order_incomplete}
    with gmsh_session(), _gmsh_options(options):
        _import_solid(brep)
        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        caps = [tag for _, tag in gmsh.model.getEntities(2) if gmsh.model.getType(2, tag) == "Plane"]
        six_sided = all(len(gmsh.model.getBoundary([(3, volume)], oriented=False)) == 6 for volume in volumes)
        if not six_sided or len(caps) != 2 * len(volumes):
            raise ValueError("Hexahedral meshing needs a solid of six-sided blocks with planar caps, such as contours_to_hex_blocks gives")

        cap_edges = {abs(tag) for _, tag in gmsh.model.getBoundary([(2, cap) for cap in caps], combined=False, oriented=False)}
        for _, curve in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(curve, (elements_per_edge if curve in cap_edges else elements_along) + 1)
        for _, face in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(face)
            gmsh.model.mesh.setRecombine(2, face)
        for volume in volumes:
            gmsh.model.mesh.setTransfiniteVolume(volume)
        gmsh.model.mesh.generate(3)
        return _volume_mesh()


def _require_element_order(element_order: int) -> None:
    if element_order < 1:
        raise ValueError(f"element_order must be at least 1, got {element_order}")


@contextmanager
def _gmsh_options(options: dict[str, float | bool]) -> Iterator[None]:
    """Set gmsh numeric options for a block, restoring them afterward.

    gmsh options are global, so they would otherwise outlive the session when
    gmsh was already running.
    """
    previous = {name: gmsh.option.getNumber(name) for name in options}
    try:
        for name, value in options.items():
            gmsh.option.setNumber(name, float(value))
        yield
    finally:
        for name, value in previous.items():
            gmsh.option.setNumber(name, value)


def _import_solid(brep: BRep) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shape.brep"
        path.write_text(brep.text, encoding="utf-8")
        gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    if not gmsh.model.getEntities(3):
        raise ValueError("Meshing needs a solid B-rep, such as contours_to_brep with a cap")


def _volume_mesh() -> Mesh:
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    coordinates = np.array(coordinates)
    node_tags = np.array(node_tags)

    index = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
    index[node_tags] = np.arange(len(node_tags))

    elements, element_types = {}, {}
    for element_type in gmsh.model.mesh.getElementTypes(3):
        name, _, _, nodes_per_element, *_ = gmsh.model.mesh.getElementProperties(element_type)
        _, element_nodes = gmsh.model.mesh.getElementsByType(element_type)
        elements[name] = index[element_nodes].reshape(-1, nodes_per_element)
        element_types[name] = int(element_type)
    return Mesh(coordinates.reshape(-1, 3), elements, element_types)
