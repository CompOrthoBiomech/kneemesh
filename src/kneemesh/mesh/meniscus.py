"""Hexahedral meshing of the meniscus, swept or in butterfly blocks, and the meniscus's named sets of faces.

hexahedral_mesh takes either B-rep a meniscus's hexahedra are made from:

- The lofted solid with planar caps (geo.meniscus.contours_to_brep). One
  cross-section is meshed with unstructured quadrilaterals (gmsh's
  Frontal-Delaunay for quads), and its nodes are mapped onto stations
  evenly spaced along the loft and joined into hexahedra.
- The five-block butterfly solid (geo.meniscus.contours_to_hex_blocks),
  whose blocks are meshed with structured, transfinite hexahedra.

Both meshes of a meniscus have the sets "anterior_horn", "posterior_horn",
and "body" besides "surface" (mesh_sets): the faces on the caps at either
end and on the rest of its surface.
"""

import gmsh
import numpy as np
from loguru import logger

from kneemesh.brep import BRep, gmsh_session
from kneemesh.mesh.sheet import quadrangulate, stack_layers
from kneemesh.mesh_sets import boundary_faces
from kneemesh.tet_meshing import Mesh, gmsh_mesh, gmsh_options, import_solid, require_element_order

# Points sampled around each station of the loft, as the polygon interior nodes are mapped by.
_RING_SAMPLES = 400
# Stations along the loft its length is measured through, and samples along each butterfly face's sides.
_LENGTH_SAMPLES = 200
_BLOCK_SAMPLES = 33
_QUADRANGLE_TYPES = {1: 3, 2: 10}


def hexahedral_mesh(
    brep: BRep,
    transverse_size: float = 1.0,
    circumferential_size: float = 2.0,
    element_order: int = 1,
    second_order_incomplete: bool = False,
) -> Mesh:
    """Mesh a meniscus's lofted solid with swept hexahedra, or its butterfly blocks with structured ones.

    Elements are about transverse_size across the meniscus's cross-section
    and circumferential_size along it, its length measured through the
    sections' centroids.

    A solid of one lofted face and two planar caps, as
    geo.meniscus.contours_to_brep makes with planar caps, is swept: its
    middle cross-section is meshed with quadrilaterals of transverse_size,
    and the section's nodes are mapped onto stations evenly spaced in length
    along the loft (see _swept_layers). These hexahedra can start inverted,
    so optimize them (mesh_optimization.optimize_hexahedral_mesh).

    A solid of six-sided blocks with planar caps, such as the five blocks
    from geo.meniscus.contours_to_hex_blocks, is meshed with transfinite
    surfaces and volumes: each edge on the caps gets a quarter of the middle
    section's perimeter over transverse_size, and the edges running between
    the caps the length over circumferential_size, so each block is a grid
    of quadrilaterals swept along the solid.

    Args:
        brep: A meniscus's lofted solid with planar caps, or its butterfly blocks.
        transverse_size: Target length of the elements' edges across the meniscus.
        circumferential_size: Target length of the elements' edges along the meniscus.
        element_order: Polynomial order of the elements. 2 gives 27-node hexahedra.
        second_order_incomplete: Omit face and interior nodes from high-order
            elements, giving 20-node instead of 27-node second-order hexahedra.

    Returns:
        The hexahedral mesh.

    Raises:
        ValueError: If a size is not positive, element_order is out of range,
            or brep is neither a lofted solid with planar caps nor a solid of
            six-sided blocks with planar caps.
    """
    if transverse_size <= 0 or circumferential_size <= 0:
        raise ValueError(f"Element sizes must be positive, got {transverse_size} and {circumferential_size}")
    require_element_order(element_order)

    with gmsh_session():
        import_solid(brep)
        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        caps = [tag for _, tag in gmsh.model.getEntities(2) if gmsh.model.getType(2, tag) == "Plane"]
        others = [tag for _, tag in gmsh.model.getEntities(2) if tag not in caps]
        if len(volumes) == 1 and len(caps) == 2 and len(others) == 1:
            if element_order > 2:
                raise ValueError(f"Swept meniscus hexahedra are linear or quadratic, got element_order {element_order}")
            layers, quadrangles = _swept_layers(others[0], transverse_size, circumferential_size, element_order)
        else:
            return _block_mesh(volumes, caps, transverse_size, circumferential_size, element_order, second_order_incomplete)
    return _swept_mesh(layers, quadrangles, element_order, second_order_incomplete)


def _block_mesh(volumes: list[int], caps: list[int], transverse_size: float, circumferential_size: float, order: int, incomplete: bool) -> Mesh:
    """Mesh the current model's six-sided blocks with transfinite hexahedra of about the sizes given."""
    six_sided = all(len(gmsh.model.getBoundary([(3, volume)], oriented=False)) == 6 for volume in volumes)
    if not six_sided or len(caps) != 2 * len(volumes):
        raise ValueError(
            "Hexahedral meshing needs a lofted solid with planar caps, as contours_to_brep makes, "
            + "or a solid of six-sided blocks with planar caps, as contours_to_hex_blocks makes"
        )
    perimeter, length = _block_dimensions(caps)
    per_edge = max(1, round(perimeter / 4 / transverse_size))
    along = max(1, round(length / circumferential_size))
    logger.debug(
        "Meshing butterfly blocks {} elements per cap edge and {} along, for a middle perimeter of {:.3g} and length {:.3g}",
        per_edge,
        along,
        perimeter,
        length,
    )
    options: dict[str, float | bool] = {"Mesh.ElementOrder": order, "Mesh.SecondOrderIncomplete": incomplete}
    with gmsh_options(options):
        cap_edges = {abs(tag) for _, tag in gmsh.model.getBoundary([(2, cap) for cap in caps], combined=False, oriented=False)}
        for _, curve in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(curve, (per_edge if curve in cap_edges else along) + 1)
        for _, face in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(face)
            gmsh.model.mesh.setRecombine(2, face)
        for volume in volumes:
            gmsh.model.mesh.setTransfiniteVolume(volume)
        gmsh.model.add_physical_group(3, volumes, name="solid")
        groups = meniscus_face_groups()

        gmsh.model.mesh.generate(3)
        return gmsh_mesh(3, groups)


def _block_dimensions(caps: list[int]) -> tuple[float, float]:
    """Return the butterfly blocks' perimeter midway between the caps and their mean length along the outer faces.

    Each outer face spans one arc of the boundary from cap to cap; its
    iso-parameter lines along the meniscus are the longer ones.
    """
    outer = [face for _, face in gmsh.model.getEntities(2) if face not in caps and len(gmsh.model.getAdjacencies(2, face)[0]) == 1]
    samples = np.linspace(0.0, 1.0, _BLOCK_SAMPLES)
    arcs, lengths = [], []
    for face in outer:
        low, high = (np.asarray(bound) for bound in gmsh.model.getParametrizationBounds(2, face))
        grid = np.stack(np.meshgrid(low[0] + samples * (high[0] - low[0]), low[1] + samples * (high[1] - low[1]), indexing="ij"), axis=-1)
        points = np.reshape(gmsh.model.getValue(2, face, grid.reshape(-1).tolist()), (_BLOCK_SAMPLES, _BLOCK_SAMPLES, 3))
        middle = _BLOCK_SAMPLES // 2
        first = float(np.linalg.norm(np.diff(points[:, middle], axis=0), axis=1).sum())
        second = float(np.linalg.norm(np.diff(points[middle], axis=0), axis=1).sum())
        lengths.append(max(first, second))
        arcs.append(min(first, second))
    return float(np.sum(arcs)), float(np.mean(lengths))


def _swept_layers(loft: int, transverse_size: float, circumferential_size: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the swept nodes' positions at every station, (stations, N, 3), and the cross-section's quadrilaterals.

    The stations are evenly spaced in length along the curve through the
    rings' centroids, about circumferential_size apart, with order - 1
    more between each pair for high-order nodes. The middle station's
    section is meshed with quadrilaterals of transverse_size. Its boundary
    nodes keep their parameter around the loft at every station and lie
    exactly on the loft. Its interior nodes keep their mean value
    coordinates in a polygon of _RING_SAMPLES points around the station,
    which reproduce the middle section exactly and deform it smoothly with
    the others, and which lie in a station's plane wherever its polygon
    does: the end stations are the caps' rims.
    """
    around, along = _loft_directions(loft)
    low, high = (np.asarray(bound) for bound in gmsh.model.getParametrizationBounds(2, loft))
    turns = low[around] + (high[around] - low[around]) * np.arange(_RING_SAMPLES) / _RING_SAMPLES

    def rings_at(stations: np.ndarray) -> np.ndarray:
        return _evaluate(loft, around, np.tile(turns, len(stations)), np.repeat(stations, _RING_SAMPLES)).reshape(len(stations), _RING_SAMPLES, 3)

    fine = np.linspace(low[along], high[along], _LENGTH_SAMPLES)
    centroids = rings_at(fine).mean(axis=1)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(centroids, axis=0), axis=1))))
    elements = max(1, round(distances[-1] / circumferential_size))
    stations = np.interp(np.linspace(0.0, distances[-1], elements * order + 1), distances, fine)
    rings = rings_at(stations)
    station_count = len(stations)

    perimeters = np.linalg.norm(np.roll(rings, -1, axis=1) - rings, axis=2).sum(axis=1)
    reference = rings[int(np.argmax(perimeters))]
    center = reference.mean(axis=0)
    axes = np.linalg.svd(reference - center)[2][:2]
    polygon = (reference - center) @ axes.T
    with gmsh_session():
        points, quadrangles, boundary = _section_quadrangles(polygon, transverse_size, order)

    interior = np.setdiff1d(np.arange(len(points)), boundary)
    layers = np.zeros((station_count, len(points), 3))
    layers[:, interior] = np.einsum("np,spd->snd", _mean_value_coordinates(points[interior], polygon), rings)
    fractions = _polygon_fractions(points[boundary], polygon)
    boundary_turns = low[around] + (high[around] - low[around]) * fractions / _RING_SAMPLES
    on_loft = _evaluate(loft, around, np.tile(boundary_turns, station_count), np.repeat(stations, len(boundary)))
    layers[:, boundary] = on_loft.reshape(station_count, len(boundary), 3)
    logger.debug(
        "Meshed the meniscus's cross-section with {} quadrilaterals, {} nodes on its boundary, for {} elements along its length of {:.3g}",
        len(quadrangles),
        len(boundary),
        elements,
        distances[-1],
    )
    return layers, quadrangles


def _loft_directions(loft: int) -> tuple[int, int]:
    """Return the indices of the loft's parameters around and along it: the closed direction is around."""
    low, high = gmsh.model.getParametrizationBounds(2, loft)
    middles = [(low[axis] + high[axis]) / 2 for axis in range(2)]
    gaps = []
    for axis in range(2):
        ends = [[*middles] for _ in range(2)]
        ends[0][axis], ends[1][axis] = low[axis], high[axis]
        values = np.reshape(gmsh.model.getValue(2, loft, [*ends[0], *ends[1]]), (2, 3))
        gaps.append(float(np.linalg.norm(values[0] - values[1])))
    around = int(np.argmin(gaps))
    return around, 1 - around


def _evaluate(loft: int, around: int, turns: np.ndarray, stations: np.ndarray) -> np.ndarray:
    """Evaluate the loft at parameters around and along it, (N, 3)."""
    parameters = np.column_stack((turns, stations) if around == 0 else (stations, turns))
    return np.reshape(gmsh.model.getValue(2, loft, parameters.ravel().tolist()), (-1, 3))


def _section_quadrangles(polygon: np.ndarray, size: float, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh a closed 2D polygon's interior with quadrilaterals in the current model.

    Returns:
        The nodes' (N, 2) coordinates, the (Q, 4 or 9) quadrilaterals in
        gmsh's node ordering, and the indices of the nodes on the boundary.
    """
    occ = gmsh.model.occ
    tags = [occ.addPoint(x, y, 0.0) for x, y in polygon]
    curve = occ.addSpline([*tags, tags[0]])
    surface = occ.addPlaneSurface([occ.addCurveLoop([curve])])
    occ.synchronize()
    quadrangulate([surface], size, "frontal_delaunay", "the meniscus's cross-section")
    if order == 2:
        with gmsh_options({"Mesh.SecondOrderIncomplete": 0}):
            gmsh.model.mesh.setOrder(2)
    element_tags, element_nodes = gmsh.model.mesh.getElementsByType(_QUADRANGLE_TYPES[order], surface)
    # Only the quadrilaterals' nodes: the points the spline passes through are free vertices with nodes of their own.
    used, quadrangles = np.unique(np.asarray(element_nodes, dtype=np.int64), return_inverse=True)
    points = np.reshape([gmsh.model.mesh.getNode(int(tag))[0] for tag in used], (-1, 3))[:, :2]
    boundary = np.flatnonzero(np.isin(used, gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)[0]))
    return points, quadrangles.reshape(len(element_tags), -1), boundary


def _mean_value_coordinates(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Return the (N, P) mean value coordinates of 2D points inside a closed polygon of P vertices.

    Each row sums to 1 and reproduces its point from the polygon's vertices.
    tan(a / 2) is computed as sin(a) / (1 + cos(a)), which stays finite for
    points in line with an edge.
    """
    offsets = polygon[np.newaxis] - points[:, np.newaxis]
    distances = np.linalg.norm(offsets, axis=2)
    following = np.roll(offsets, -1, axis=1)
    sines = offsets[..., 0] * following[..., 1] - offsets[..., 1] * following[..., 0]
    cosines = np.einsum("npd,npd->np", offsets, following)
    half_tangents = sines / (distances * np.roll(distances, -1, axis=1) + cosines)
    weights = (np.roll(half_tangents, 1, axis=1) + half_tangents) / distances
    return weights / weights.sum(axis=1, keepdims=True)


def _polygon_fractions(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Return where points lie around a closed polygon, as fractional vertex indices of their nearest points on it."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = points[:, np.newaxis] - polygon[np.newaxis]
    along = np.clip(np.einsum("npd,pd->np", offsets, edges) / np.einsum("pd,pd->p", edges, edges), 0.0, 1.0)
    distances = np.linalg.norm(offsets - along[..., np.newaxis] * edges, axis=2)
    nearest = np.argmin(distances, axis=1)
    return nearest + along[np.arange(len(points)), nearest]


def _swept_mesh(layers: np.ndarray, quadrangles: np.ndarray, order: int, incomplete: bool) -> Mesh:
    """Join the section's nodes at every station into hexahedra, naming the caps by the stations at either end."""
    mesh, layer = stack_layers(layers, quadrangles, order, incomplete)
    faces = boundary_faces(mesh.elements)
    ends = [np.concatenate([np.all(layer[connectivity] == end, axis=1) for connectivity in faces.values()]) for end in (0, len(layers) - 1)]
    anterior, posterior = sorted(ends, key=lambda on_cap: float(_cap_y(mesh, faces, on_cap)))
    groups = {ANTERIOR_HORN: anterior, POSTERIOR_HORN: posterior, BODY: ~(anterior | posterior)}
    logger.debug("Swept {} hexahedra through {} layers", sum(len(rows) for rows in mesh.elements.values()), (len(layers) - 1) // order)
    return mesh.with_face_sets(groups, faces)


def _cap_y(mesh: Mesh, faces: dict[str, np.ndarray], on_cap: np.ndarray) -> float:
    """Return the mean y of the nodes of the faces on a cap."""
    connectivity = np.concatenate([rows[:, :4] for rows in faces.values()])
    return float(mesh.nodes[np.unique(connectivity[on_cap]), 1].mean())


ANTERIOR_HORN = "anterior_horn"
"""Name of a meniscus mesh's set of the faces on the cap at its anterior end, and of their nodes."""
POSTERIOR_HORN = "posterior_horn"
"""Name of a meniscus mesh's set of the faces on the cap at its posterior end, and of their nodes."""
BODY = "body"
"""Name of a meniscus mesh's set of the faces on its surface between the horns' caps, and of their nodes."""


def meniscus_face_groups() -> dict[str, list[int]]:
    """Name the current model's meniscus faces, for tet_meshing.tetrahedral_mesh's face_groups: its horns' caps and its body.

    The caps are the planar boundary faces, as on hexahedral blocks and
    planar-capped solids, or, with smooth caps, every boundary face but the
    largest, the loft. Caps sharing curves make one horn. The anterior horn
    is the one at lower y: SimpleITK's physical space is LPS, so y grows
    toward the patient's back.

    Raises:
        ValueError: If the caps do not form two horns.
    """
    boundary = [tag for _, tag in gmsh.model.getEntities(2) if len(gmsh.model.getAdjacencies(2, tag)[0]) == 1]
    planar = [face for face in boundary if gmsh.model.getType(2, face) == "Plane"]
    caps = planar or sorted(boundary, key=lambda face: gmsh.model.occ.getMass(2, face))[:-1]
    curves = {face: {abs(tag) for _, tag in gmsh.model.getBoundary([(2, face)], combined=False, oriented=False)} for face in caps}
    horns: list[set[int]] = []
    for face in caps:
        joined = [horn for horn in horns if any(curves[face] & curves[other] for other in horn)]
        horns = [horn for horn in horns if horn not in joined] + [set().union({face}, *joined)]
    if len(horns) != 2:
        raise ValueError(f"A meniscus's caps should form two horns, got {len(horns)}")

    def y(horn: set[int]) -> float:
        masses = np.array([gmsh.model.occ.getMass(2, face) for face in horn])
        centers = np.array([gmsh.model.occ.getCenterOfMass(2, face) for face in horn])
        return float(masses @ centers[:, 1] / masses.sum())

    anterior, posterior = sorted(horns, key=y)
    return {ANTERIOR_HORN: sorted(anterior), POSTERIOR_HORN: sorted(posterior), BODY: sorted(set(boundary) - set(caps))}
