"""Node optimization of hexahedral meshes for the scaled Jacobian and the Frobenius aspect ratio.

Up to three strategies run in turn on the hexahedra's corner nodes:

1. Smart Laplacian smoothing moves each node toward the average of its edge
   neighbors, keeping a move only if it does not lower the worst scaled
   Jacobian of the elements around the node.
2. If any element is still inverted, untangling minimizes the sum of
   max(0, 0.2 - SJ)^2 over every element corner, which stays finite and
   smooth through inverted corners, until none is inverted.
3. Optimization minimizes the sum of 1 / SJ over every element corner, plus
   aspect_weight times the sum of each corner's squared Frobenius aspect
   ratio, by projected gradient descent. Both terms grow without bound as a
   corner degenerates, so no step can invert an element, and both weigh the
   worst corners most. 1 / SJ alone squares corners up by stretching
   elements; the aspect term holds their edge lengths in proportion. A
   third term, warp_weight times the sum over the boundary quadrilaterals of
   their squared normalized out-of-plane volume, flattens warped faces on the
   surface, which corner measures cannot see.

High-order hexahedra are also measured at the points of the Gauss rules
simulations integrate them with, where their curved edges can invert an
element whose corners are valid. Their Jacobians there are barriers in
optimization and targets in untangling, alongside the corners.

Boundary nodes are classified from the mesh's own boundary quadrilaterals.
Where adjacent boundary faces meet at more than a feature angle, their shared
edges are feature edges. Surface nodes slide in their tangent plane and may
leave it by a fraction of the local element size; nodes on a feature edge
slide along it with the same allowance off it; feature corners stay fixed.
The inner nodes of a mesh's straight_lines stay at their fractions along the
segment between the line's ends, which move as their own kind allows.
By default the surface nodes' net offset off their tangent planes is
removed, which keeps the mesh's volume.
Tangential travel is also limited, so the tangent plane stays a good model of
the surface. High-order nodes follow the corners of their element.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import gmsh
import numpy as np
from loguru import logger
from scipy import sparse

from kneemesh.brep import gmsh_session
from kneemesh.mesh_sets import unshared
from kneemesh.tet_meshing import Mesh

# For each corner of a gmsh hexahedron, the three corners it shares an edge
# with, ordered so the corner's Jacobian determinant is positive for a valid element.
_CORNER_NEIGHBORS = np.array([(1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7), (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3)])
_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
# Faces ordered counterclockwise seen from outside a valid element.
_FACES = np.array([(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])

DEFAULT_LAPLACIAN_ITERATIONS = 10
DEFAULT_OPTIMIZATION_ITERATIONS = 200
DEFAULT_SURFACE_DEVIATION = 0.1
DEFAULT_SURFACE_TRAVEL = 0.5
DEFAULT_FEATURE_ANGLE = 30.0

_INTERIOR, _SURFACE, _EDGE, _FIXED, _LINE = 0, 1, 2, 3, 4
_INITIAL_STEP = 0.01
_MIN_STEP = 1e-8
_STEP_GROWTH = 1.5
_CONVERGENCE = 1e-7
_VOLUME_ROUNDS = 3
_UNTANGLE_TARGET = 0.2
# Smallest corner scaled Jacobian untangling stops at. Stopping as soon as none is inverted could leave a corner
# at 1e-11, where 1 / SJ is so steep that optimization's first step backtracks below _MIN_STEP and stalls.
_UNTANGLE_MARGIN = 0.01
# Scaled Jacobian at a Gauss point of a high-order hexahedron below which optimization's barrier on it acts.
_GAUSS_BARRIER = 0.2


@dataclass(frozen=True)
class MeshQuality:
    """Scaled Jacobian, Frobenius aspect ratio, and surface warp statistics of a hexahedral mesh.

    Attributes:
        minimum: Smallest element scaled Jacobian (see scaled_jacobians). Negative means an inverted element.
        mean: Mean element scaled Jacobian.
        inverted: Number of elements with a Jacobian determinant at or below zero at a corner, or for a
            high-order element, of its own Jacobian at a corner node or Gauss point.
        max_aspect: Largest element Frobenius aspect ratio, infinite if any element is inverted.
        mean_aspect: Mean element Frobenius aspect ratio, infinite if any element is inverted.
        max_warp: Largest warp of a boundary quadrilateral, in degrees (see surface_warps).
        warp_99: 99th percentile of the boundary quadrilaterals' warps, in degrees.
    """

    minimum: float
    mean: float
    inverted: int
    max_aspect: float
    mean_aspect: float
    max_warp: float
    warp_99: float


@dataclass(frozen=True, eq=False)
class _GaussSamples:
    """Jacobians of one type of high-order hexahedra at Gauss points, as affine functions of their corners.

    High-order nodes follow their element's corners trilinearly, so at any
    point the Jacobian is the trilinear map's, from the corners, plus a
    constant: the original high-order nodes' offsets from their trilinear
    positions, through their own shape functions' gradients.
    """

    elements: np.ndarray  # (K,) rows of the hexahedra's corners
    gradients: np.ndarray  # (G, 8, 3) trilinear shape functions' gradients at each Gauss point, by corner and local axis
    offsets: np.ndarray  # (K, G, 3, 3) constant part of each Jacobian, one local axis's tangent per row

    def tangents(self, nodes: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Return the Jacobians at the Gauss points, (K, G, 3, 3), with rows dx/du, dx/dv, dx/dw as a corner's edges."""
        # A batched product with the (3G, 8) gradients is several times faster than einsum.
        products = self._matrix() @ nodes[corners[self.elements]]
        return products.reshape(self.offsets.shape) + self.offsets

    def scatter(self, gradient: np.ndarray, corners: np.ndarray, size: int) -> np.ndarray:
        """Return the gradient by node, (size, 3), of a function with the given gradient by Jacobian, (K, G, 3, 3)."""
        values = (self._matrix().T @ gradient.reshape(len(gradient), -1, 3)).reshape(-1, 3)
        indices = corners[self.elements].ravel()
        return np.column_stack([np.bincount(indices, weights=values[:, axis], minlength=size) for axis in range(3)])

    def _matrix(self) -> np.ndarray:
        """The gradients as a (3G, 8) matrix, row g * 3 + i for Gauss point g and local axis i."""
        return np.ascontiguousarray(np.swapaxes(self.gradients, 1, 2).reshape(-1, 8))


def _gauss_samples(mesh: Mesh) -> list[_GaussSamples]:
    """Return the Gauss point Jacobians of each type of high-order hexahedra in a mesh of hexahedra.

    An element is sampled at its corner nodes and the points of its Gauss
    rules (_sample_basis). Linear hexahedra have none.
    """
    samples: list[_GaussSamples] = []
    if all(connectivity.shape[1] == 8 for connectivity in mesh.elements.values()):
        return samples
    start = 0
    with gmsh_session():
        for name, connectivity in mesh.elements.items():
            rows = np.arange(start, start + len(connectivity))
            start += len(connectivity)
            if connectivity.shape[1] == 8:
                continue
            element_type = mesh.element_types[name]
            local = np.reshape(gmsh.model.mesh.getElementProperties(element_type)[4], (-1, 3))
            points, basis = _sample_basis(element_type)
            # Trilinear shape functions (1 + u u_c)(1 + v v_c)(1 + w w_c) / 8 differentiated along each axis.
            factors = 1 + points[:, None, :] * local[None, :8, :]
            gradients = np.stack([local[None, :8, axis] * np.prod(np.delete(factors, axis, axis=-1), axis=-1) / 8 for axis in range(3)], axis=-1)
            nodes = mesh.nodes[connectivity]
            full = np.einsum("gni,knx->kgix", basis, nodes)
            trilinear = np.einsum("gci,kcx->kgix", gradients, nodes[:, :8])
            samples.append(_GaussSamples(rows, gradients, full - trilinear))
    return samples


def _sample_basis(element_type: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the points a high-order volume element is measured at, in local coordinates, (G, 3), and its shape functions' gradients there, (G, n, 3).

    The points are its corner nodes, then the points of the two Gauss rules
    simulations integrate an element of order p with. A hexahedron's are the
    reduced and full rules, p and p + 1 points along each axis, exact to
    orders 2p - 1 and 2p + 1; a tetrahedron's are exact to orders 2p - 2 and
    2p, 4 and 11 points for second order, the stiffness and mass rules.
    Needs a gmsh session.
    """
    properties = gmsh.model.mesh.getElementProperties(element_type)
    name, order, count = str(properties[0]), int(properties[2]), int(properties[3])
    tetrahedron = name.startswith("Tetrahedron")
    corners = np.reshape(properties[4], (-1, 3))[: 4 if tetrahedron else 8]
    degrees = (2 * order - 2, 2 * order) if tetrahedron else (2 * order - 1, 2 * order + 1)
    gauss = [np.reshape(gmsh.model.mesh.getIntegrationPoints(element_type, f"Gauss{degree}")[0], (-1, 3)) for degree in degrees]
    points = np.vstack([corners, *gauss])
    _, basis, _ = gmsh.model.mesh.getBasisFunctions(element_type, points.ravel(), "GradLagrange")
    return points, np.reshape(basis, (len(points), count, 3))


def _tetrahedron_sample_jacobians(nodes: np.ndarray, connectivity: np.ndarray, element_type: int) -> np.ndarray:
    """Return high-order tetrahedra's Jacobians at their corners and Gauss points, (M, G, 3, 3), with rows dx/du, dx/dv, dx/dw.

    The rows are a straight tetrahedron's edges from its first corner, in the
    order _tetrahedron_quality takes them.
    """
    with gmsh_session():
        _, basis = _sample_basis(element_type)
    return np.einsum("gni,knx->kgix", basis, nodes[connectivity])


def _element_scaled_jacobians(nodes: np.ndarray, corners: np.ndarray, samples: Sequence[_GaussSamples] = ()) -> np.ndarray:
    """Return each hexahedron's smallest scaled Jacobian over its corners and Gauss points, (M,)."""
    quality = _corner_scaled_jacobians(nodes, corners).min(axis=1)
    for sample in samples:
        quality[sample.elements] = np.minimum(quality[sample.elements], _scaled_jacobians_of(sample.tangents(nodes, corners)).min(axis=1))
    return quality


def _hexahedron_metrics(nodes: np.ndarray, corners: np.ndarray, samples: Sequence[_GaussSamples] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each hexahedron's scaled Jacobian, Frobenius aspect ratio, and whether it is inverted, as reported, (M,) each.

    A linear hexahedron is measured on its corners' edges. A high-order one is
    measured by its own Jacobian at its corner nodes and Gauss points alone,
    since its straight-sided corners' edges are not its shape: a curved
    element can be valid where they are inverted, and the reverse.
    """
    edges = _corner_edges(nodes, corners)
    scaled = _scaled_jacobians_of(edges).min(axis=1)
    aspects = _corner_aspects(edges).max(axis=1)
    inverted = (np.linalg.det(edges) <= 0).any(axis=1)
    for sample in samples:
        tangents = sample.tangents(nodes, corners)
        scaled[sample.elements] = _scaled_jacobians_of(tangents).min(axis=1)
        aspects[sample.elements] = _corner_aspects(tangents).max(axis=1)
        inverted[sample.elements] = (np.linalg.det(tangents) <= 0).any(axis=1)
    return scaled, aspects, inverted


def _any_inverted(nodes: np.ndarray, corners: np.ndarray, samples: Sequence[_GaussSamples] = ()) -> np.ndarray:
    """Return which hexahedra have a Jacobian determinant at or below zero at a corner or Gauss point, (M,)."""
    inverted = (np.linalg.det(_corner_edges(nodes, corners)) <= 0).any(axis=1)
    for sample in samples:
        inverted[sample.elements] |= (np.linalg.det(sample.tangents(nodes, corners)) <= 0).any(axis=1)
    return inverted


@dataclass(frozen=True, eq=False)
class _Constraints:
    """How each corner node may move from its starting position."""

    kind: np.ndarray  # (N,) one of _INTERIOR, _SURFACE, _EDGE, _FIXED, _LINE
    origin: np.ndarray  # (N, 3) starting positions
    direction: np.ndarray  # (N, 3) unit surface normal, edge tangent, or zero
    deviation: np.ndarray  # (N,) allowed distance off the tangent plane or edge line
    travel: np.ndarray  # (N,) allowed distance along the tangent plane or edge line
    boundary_edges: np.ndarray  # (B, 2) node pairs on the boundary
    feature_edges: np.ndarray  # (F, 2) node pairs on feature edges
    areas: np.ndarray  # (N,) boundary area around each surface node, 0 elsewhere
    edge_areas: np.ndarray  # (N, 3) outward boundary area vector around each feature edge node, 0 elsewhere
    preserve_volume: bool
    lines: np.ndarray  # (L, K) chains of nodes kept straight
    fractions: np.ndarray  # (L, K) each node's fraction of the way from its chain's first node to its last

    def place_lines(self, positions: np.ndarray) -> np.ndarray:
        """Return positions with each straight line's inner nodes at their fractions between its ends."""
        if not len(self.lines):
            return positions
        placed = positions.copy()
        first, last = positions[self.lines[:, :1]], positions[self.lines[:, -1:]]
        placed[self.lines[:, 1:-1]] = ((1 - self.fractions[..., None]) * first + self.fractions[..., None] * last)[:, 1:-1]
        return placed

    def line_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Return the gradient with respect to the free nodes, passing each line's inner nodes' share to its ends."""
        if not len(self.lines):
            return gradient
        inner, fractions = self.lines[:, 1:-1], self.fractions[:, 1:-1, None]
        reduced = gradient.copy()
        np.add.at(reduced, self.lines[:, 0], np.sum((1 - fractions) * gradient[inner], axis=1))
        np.add.at(reduced, self.lines[:, -1], np.sum(fractions * gradient[inner], axis=1))
        reduced[inner.ravel()] = 0.0
        return reduced

    def project(self, positions: np.ndarray) -> np.ndarray:
        offsets = positions - self.origin
        along = np.einsum("ij,ij->i", offsets, self.direction)[:, None] * self.direction
        across = offsets - along

        surface = self.kind == _SURFACE
        edge = self.kind == _EDGE
        # Edge nodes: along is the tangential part, across the part off the line.
        tangential = np.clip(np.einsum("ij,ij->i", offsets, self.direction), -self.travel, self.travel)
        edge_offsets = tangential[:, None] * self.direction + _clip_norm(across, self.deviation)
        # Surface nodes: along is the normal part, across the tangential part.
        normal = np.clip(np.einsum("ij,ij->i", offsets, self.direction), -self.deviation, self.deviation)
        if self.preserve_volume and self.areas.any():
            # Shifting the surface nodes' normal offsets by their area-weighted mean cancels the
            # net volume they and the edge nodes, moving off their lines, add; a few rounds keep
            # the offsets within their limits too.
            edge_volume = float(np.einsum("ij,ij->", self.edge_areas[edge], edge_offsets[edge]))
            for _ in range(_VOLUME_ROUNDS):
                shift = (np.sum(self.areas * normal) + edge_volume) / np.sum(self.areas)
                normal = np.clip(normal - shift * (self.areas > 0), -self.deviation, self.deviation)
        offsets[surface] = (_clip_norm(across, self.travel) + normal[:, None] * self.direction)[surface]
        offsets[edge] = edge_offsets[edge]
        offsets[self.kind == _FIXED] = 0.0
        return self.place_lines(self.origin + offsets)


def scaled_jacobians(mesh: Mesh) -> np.ndarray:
    """Return each hexahedron's scaled Jacobian, the smallest over its eight corners and, if high-order, its Gauss points.

    A corner's scaled Jacobian is the determinant of its three edge vectors
    divided by the product of their lengths: 1 for a right angle corner, 0
    for a flat one, and negative for an inverted one. A high-order element
    is measured instead by its own Jacobian, its tangents along the local
    axes in place of edges, at its corner nodes and the points of its
    reduced and full Gauss rules, p and p + 1 along each axis for order p:
    curved edges can invert it there though its straight-sided corners are
    valid, and leave it valid though they are not.

    Raises:
        ValueError: If the mesh has elements other than hexahedra.
    """
    return _hexahedron_metrics(mesh.nodes, _hexahedron_corners(mesh), _gauss_samples(mesh))[0]


def frobenius_aspects(mesh: Mesh) -> np.ndarray:
    """Return each hexahedron's Frobenius aspect ratio, the largest over its eight corners.

    A corner's Frobenius aspect ratio is ``|E|_F^2 / (3 det(E)^(2/3))`` for the
    matrix E of its three edge vectors, as Verdict's maximum aspect Frobenius
    measures it: 1 for a cube's corner, and larger as the corner's edges
    differ in length or leave right angles. It is infinite for a degenerate
    or inverted corner. High-order elements are measured by their own
    Jacobian at their corner nodes and Gauss points (see scaled_jacobians).

    Raises:
        ValueError: If the mesh has elements other than hexahedra.
    """
    return _hexahedron_metrics(mesh.nodes, _hexahedron_corners(mesh), _gauss_samples(mesh))[1]


def surface_warps(mesh: Mesh) -> np.ndarray:
    """Return each boundary quadrilateral's warp, in degrees, from the hexahedra's corner nodes.

    A quadrilateral's warp is the angle between the normals of the two
    triangles a diagonal splits it into, the larger over its two diagonals:
    0 for a flat face. Scaled Jacobians and aspect ratios, measured at
    element corners, miss it; it is what makes a mesh's surface look
    crumpled.

    Raises:
        ValueError: If the mesh has elements other than hexahedra.
    """
    return _warps(mesh.nodes[_boundary_faces(_hexahedron_corners(mesh))])


def _warps(points: np.ndarray) -> np.ndarray:
    """Warps of (M, 4, 3) quadrilaterals' corners, in degrees."""

    def angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        lengths = np.maximum(np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1), np.finfo(float).tiny)
        return np.degrees(np.arccos(np.clip(np.einsum("...i,...i", first, second) / lengths, -1.0, 1.0)))

    def split(a: int, b: int, c: int, d: int) -> np.ndarray:
        corner = points[:, a]
        return angle(np.cross(points[:, b] - corner, points[:, c] - corner), np.cross(points[:, c] - corner, points[:, d] - corner))

    return np.maximum(split(0, 1, 2, 3), split(1, 2, 3, 0))


def mesh_quality(mesh: Mesh) -> MeshQuality:
    """Summarize a hexahedral mesh's scaled Jacobians, Frobenius aspect ratios, and surface warps.

    Raises:
        ValueError: If the mesh has elements other than hexahedra.
    """
    corners = _hexahedron_corners(mesh)
    quality, aspects, inverted = _hexahedron_metrics(mesh.nodes, corners, _gauss_samples(mesh))
    warps = surface_warps(mesh)
    return MeshQuality(
        float(quality.min()),
        float(quality.mean()),
        int(np.count_nonzero(inverted)),
        float(aspects.max()),
        float(aspects.mean()),
        float(warps.max(initial=0.0)),
        float(np.percentile(warps, 99)) if len(warps) else 0.0,
    )


SCALED_JACOBIAN = "scaled_jacobian"
"""Name of the scaled Jacobian in element_quality."""

FROBENIUS_ASPECT = "frobenius_aspect"
"""Name of the Frobenius aspect ratio in element_quality."""

WARP = "warp"
"""Name of the quadrilaterals' warp in element_quality."""

EDGE_RATIO = "edge_ratio"
"""Name of the edge ratio in element_quality."""

MIN_DIHEDRAL_ANGLE = "min_dihedral_angle"
"""Name of the tetrahedra's smallest dihedral angle in element_quality."""

# Corners of each element shape: high-order elements are measured on these first nodes alone.
_CORNER_COUNTS = {"Tetrahedron": 4, "Hexahedron": 8, "Triangle": 3, "Quadrilateral": 4}
# Edges between the corners of each element shape, in gmsh's corner numbering.
_SHAPE_EDGES = {
    "Tetrahedron": np.array([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]),
    "Hexahedron": _EDGES,
    "Triangle": np.array([(0, 1), (1, 2), (2, 0)]),
    "Quadrilateral": np.array([(0, 1), (1, 2), (2, 3), (3, 0)]),
}

# A regular tetrahedron's edges from its first corner, one per row, with unit length.
_REGULAR_TETRAHEDRON = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]])


def element_quality(mesh: Mesh) -> dict[str, np.ndarray]:
    """Return quality metrics of every element and face of a mesh, measured on corner nodes, by name.

    Each array holds one value per element and face, in their numbering
    (Mesh.elements, then Mesh.faces), NaN where the metric does not apply,
    so meshes mixing element shapes take every metric of any of them.
    Higher-order elements are measured on their corner nodes, as if they
    were first order, except for the scaled Jacobian and Frobenius aspect
    ratio of high-order hexahedra and tetrahedra, which take the element's
    own Jacobian at its corner nodes and Gauss points (see scaled_jacobians
    and _sample_basis): curved edges can invert an element whose
    straight-sided corners are valid, and leave one valid whose corners are
    not.

    - scaled_jacobian, of volume elements, or of a surface mesh's elements:
      the smallest over its corners of the corner's Jacobian determinant over
      the product of its edges' lengths, scaled so the ideal shape (a cube,
      square, regular tetrahedron, or equilateral triangle) is 1. It is 0
      for a degenerate element and negative for an inverted one;
      triangles, which have no orientation of their own, are never negative.
    - frobenius_aspect, of the same: the largest over its corners of
      ``|T|_F^2 / (d det(T)^(2/d))``, in d dimensions, for the map T from the
      ideal shape's corner to the element's, as Verdict's aspect Frobenius:
      1 for the ideal shape, larger as edges differ in length or leave the
      ideal angles, and infinite for an inverted or degenerate element.
    - warp, of quadrilateral faces, or of a surface mesh's quadrilaterals:
      the angle between the two triangles either diagonal splits it into,
      the larger of the two, in degrees (see surface_warps). Present only
      when the mesh has quadrilaterals.
    - edge_ratio, of volume elements, or of a surface mesh's elements: the
      longest edge between corners over the shortest, 1 for the ideal
      shapes and infinite for a collapsed edge. Comparable across shapes.
    - min_dihedral_angle, of tetrahedra: the smallest angle between two
      faces along an edge, in degrees, about 70.5 for the regular
      tetrahedron and 0 for a flat one, a sliver included, which the edge
      ratio misses. Like the edge ratio it does not see orientation, so an
      inverted tetrahedron measures as its mirror image does; its scaled
      Jacobian tells. Present only when the mesh has tetrahedra.

    Raises:
        ValueError: If the mesh has an element shape other than tetrahedra, hexahedra, triangles, and quadrilaterals.
    """
    count = mesh.element_count
    quality = {SCALED_JACOBIAN: np.full(count, np.nan), FROBENIUS_ASPECT: np.full(count, np.nan), EDGE_RATIO: np.full(count, np.nan)}
    warps, dihedrals = np.full(count, np.nan), np.full(count, np.nan)
    start = 0
    for kind, connectivity in [*(("element", item) for item in mesh.elements.items()), *(("face", item) for item in mesh.faces.items())]:
        name, rows = connectivity
        shape = name.split(" ")[0]
        if shape not in _CORNER_COUNTS:
            raise ValueError(f"No quality metrics for {name} elements")
        corners = rows[:, : _CORNER_COUNTS[shape]]
        span = slice(start, start + len(rows))
        start += len(rows)
        if shape == "Quadrilateral":
            warps[span] = _warps(mesh.nodes[corners])
        if kind == "face":
            continue
        quality[EDGE_RATIO][span] = _edge_ratios(mesh.nodes[corners], _SHAPE_EDGES[shape])
        if shape == "Hexahedron":
            samples = _gauss_samples(Mesh(mesh.nodes, {name: rows}, {name: mesh.element_types[name]}))
            quality[SCALED_JACOBIAN][span], quality[FROBENIUS_ASPECT][span], _ = _hexahedron_metrics(mesh.nodes, corners, samples)
        elif shape == "Tetrahedron":
            scaled, aspects = _tetrahedron_quality(mesh.nodes[corners])
            if rows.shape[1] > 4:
                jacobians = _tetrahedron_sample_jacobians(mesh.nodes, rows, mesh.element_types[name])
                # Each point's tangents are the edges of the straight tetrahedron with the same Jacobian.
                points = np.concatenate((np.zeros((*jacobians.shape[:2], 1, 3)), jacobians), axis=2).reshape(-1, 4, 3)
                at_points, aspects_at_points = (values.reshape(jacobians.shape[:2]) for values in _tetrahedron_quality(points))
                scaled, aspects = at_points.min(axis=1), aspects_at_points.max(axis=1)
            quality[SCALED_JACOBIAN][span], quality[FROBENIUS_ASPECT][span] = scaled, aspects
            dihedrals[span] = _min_dihedral_angles(mesh.nodes[corners])
        elif shape == "Triangle":
            quality[SCALED_JACOBIAN][span], quality[FROBENIUS_ASPECT][span] = _triangle_quality(mesh.nodes[corners])
        else:
            quality[SCALED_JACOBIAN][span], quality[FROBENIUS_ASPECT][span] = _quadrilateral_quality(mesh.nodes[corners])
    if not np.isnan(warps).all():
        quality[WARP] = warps
    if not np.isnan(dihedrals).all():
        quality[MIN_DIHEDRAL_ANGLE] = dihedrals
    return quality


def _edge_ratios(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Longest over shortest of the given (E, 2) corner edges of (M, corners, 3) elements, infinite where an edge has collapsed."""
    lengths = np.linalg.norm(points[:, edges[:, 1]] - points[:, edges[:, 0]], axis=-1)
    shortest, longest = lengths.min(axis=1), lengths.max(axis=1)
    positive = shortest > 0
    return np.where(positive, longest / np.where(positive, shortest, 1.0), np.inf)


def _min_dihedral_angles(points: np.ndarray) -> np.ndarray:
    """Smallest interior dihedral angles of (M, 4, 3) tetrahedra, in degrees, 0 for a flat one."""
    angles = []
    for (a, b), (c, d) in zip(_SHAPE_EDGES["Tetrahedron"], [(3, 2), (0, 3), (1, 3), (1, 2), (2, 0), (0, 1)], strict=True):
        # The angle at edge ab between the faces abc and abd: between c's and d's offsets from the edge, normal to it.
        axis = points[:, b] - points[:, a]
        axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), np.finfo(float).tiny)
        first, second = points[:, c] - points[:, a], points[:, d] - points[:, a]
        first -= np.einsum("mi,mi->m", first, axis)[:, None] * axis
        second -= np.einsum("mi,mi->m", second, axis)[:, None] * axis
        lengths = np.maximum(np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1), np.finfo(float).tiny)
        angles.append(np.degrees(np.arccos(np.clip(np.einsum("mi,mi->m", first, second) / lengths, -1.0, 1.0))))
    return np.min(angles, axis=0)


def _tetrahedron_quality(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled Jacobians and Frobenius aspect ratios of (M, 4, 3) tetrahedra's corners."""
    edges = points[:, 1:] - points[:, :1]
    determinants = np.linalg.det(edges)
    lengths = np.linalg.norm(points[:, :, None] - points[:, None, :], axis=-1)
    # Every corner's determinant is the same signed volume; each corner divides it by its own three edges' lengths.
    products = np.stack([lengths[:, k, [m for m in range(4) if m != k]].prod(axis=1) for k in range(4)], axis=1)
    scaled = np.sqrt(2) * determinants / np.maximum(products.max(axis=1), np.finfo(float).tiny)
    # The map from the regular tetrahedron to this one, T = E^T W^-T.
    maps = np.swapaxes(edges, 1, 2) @ np.linalg.inv(_REGULAR_TETRAHEDRON).T
    return scaled, _aspects(maps, determinants / np.linalg.det(_REGULAR_TETRAHEDRON), 3)


def _triangle_quality(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled Jacobians and Frobenius aspect ratios of (M, 3, 3) triangles' corners."""
    doubled_area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=-1)
    lengths = np.linalg.norm(points - np.roll(points, -1, axis=1), axis=-1)
    products = np.stack([lengths[:, k] * lengths[:, k - 1] for k in range(3)], axis=1)
    tiny = np.finfo(float).tiny
    scaled = 2 / np.sqrt(3) * doubled_area / np.maximum(products.max(axis=1), tiny)
    # Verdict's triangle aspect Frobenius, (a^2 + b^2 + c^2) / (4 sqrt(3) area), for which every corner gives the same.
    squared = (lengths**2).sum(axis=1)
    positive = doubled_area > 0
    aspect = np.where(positive, squared / (2 * np.sqrt(3) * np.where(positive, doubled_area, 1.0)), np.inf)
    return scaled, aspect


def _quadrilateral_quality(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled Jacobians and Frobenius aspect ratios of (M, 4, 3) quadrilaterals' corners, in the plane normal to their diagonals."""
    normals = np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 1])
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), np.finfo(float).tiny)
    forward = np.roll(points, -1, axis=1) - points
    backward = np.roll(points, 1, axis=1) - points
    areas = np.einsum("mki,mi->mk", np.cross(forward, backward), normals)
    forward_squared, backward_squared = (forward**2).sum(axis=-1), (backward**2).sum(axis=-1)
    scaled = areas / np.maximum(np.sqrt(forward_squared * backward_squared), np.finfo(float).tiny)
    positive = areas > 0
    aspects = np.where(positive, (forward_squared + backward_squared) / (2 * np.where(positive, areas, 1.0)), np.inf)
    return scaled.min(axis=1), aspects.max(axis=1)


def _aspects(maps: np.ndarray, determinants: np.ndarray, dimension: int) -> np.ndarray:
    """Frobenius aspect ratios |T|_F^2 / (d det(T)^(2/d)) of (..., d, d) maps with the given determinants, infinite where not positive."""
    squared = np.einsum("...ij,...ij->...", maps, maps)
    positive = determinants > 0
    return np.where(positive, squared / (dimension * np.where(positive, determinants, 1.0) ** (2 / dimension)), np.inf)


def optimize_hexahedral_mesh(
    mesh: Mesh,
    laplacian_iterations: int = DEFAULT_LAPLACIAN_ITERATIONS,
    optimization_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
    surface_deviation: float = DEFAULT_SURFACE_DEVIATION,
    surface_travel: float = DEFAULT_SURFACE_TRAVEL,
    feature_angle: float = DEFAULT_FEATURE_ANGLE,
    preserve_volume: bool = True,
    aspect_weight: float = 0.0,
    warp_weight: float = 0.0,
) -> Mesh:
    """Move a hexahedral mesh's nodes to raise its scaled Jacobians, keeping connectivity.

    Smart Laplacian smoothing runs first, then optimization of the sum of
    1 / SJ over every element corner plus aspect_weight times the sum of their
    squared Frobenius aspect ratios, plus warp_weight times the sum of the
    boundary quadrilaterals' squared warp measures (_warp_objective). Optimization needs every element valid,
    so if smoothing leaves any corner's scaled Jacobian below 0.01, inverted
    or nearly, the mesh is untangled first, by raising every corner's toward
    0.2 until all reach 0.01, and optimization is skipped with a warning if
    any stays inverted.

    High-order hexahedra are also measured at their Gauss points (see
    scaled_jacobians), where their curved edges can invert them though their
    corners are valid: smoothing keeps their scaled Jacobians there from
    falling, untangling raises them as it does the corners', and
    optimization adds (1 / SJ - 1 / 0.2)^2 for each below 0.2, a barrier that
    leaves elements well shaped there alone.

    Boundary allowances scale with each node's local size, the mean length
    of the element edges at the node. Feature edges are boundary edges where
    the faces on either side meet at more than feature_angle; nodes where
    feature edges end, branch, or turn by more than feature_angle are fixed.

    Args:
        mesh: Mesh of linear or high-order hexahedra, such as hexahedral_mesh returns.
        laplacian_iterations: Smart Laplacian sweeps. 0 skips smoothing.
        optimization_iterations: Gradient steps of the optimization, and at
            most as many of untangling. 0 skips both.
        surface_deviation: How far a boundary node may move off its starting
            tangent plane, or off its feature edge, as a fraction of its local size.
            0 keeps boundary nodes on their tangent planes and edge lines.
        surface_travel: How far a boundary node may slide along its tangent
            plane or feature edge, as a fraction of its local size.
        feature_angle: Angle in degrees between boundary face normals above
            which their shared edge is a feature edge.
        preserve_volume: Remove the area-weighted mean of the surface nodes'
            offsets off their tangent planes, so they can move out in some
            places and in at others but not swell or shrink the mesh overall.
            Without it, optimization tends to push the whole surface outward,
            adding about 5% to a meniscus at surface_deviation 0.1.
        aspect_weight: Weight of the squared Frobenius aspect ratios against
            1 / SJ. 0 optimizes the scaled Jacobian alone, which can stretch
            elements: it raised the real medial tibial cartilage's largest
            aspect ratio to 42 with its walls free to bend and 11 with them
            straight, against 6.4 at 0.5. The aspect term also penalizes
            deliberately elongated elements, such as the meniscus's along its
            centerline.
        warp_weight: Weight of the boundary quadrilaterals' warp against
            1 / SJ, per quadrilateral. 0 leaves warped faces as they are.

    Returns:
        A new Mesh with the same connectivity and moved nodes. High-order
        nodes move with the trilinear interpolation of their element's corner
        displacements, so curved edges keep their offsets from straight ones.

    Raises:
        ValueError: If the mesh has elements other than hexahedra or an
            argument is out of range.
    """
    if laplacian_iterations < 0 or optimization_iterations < 0:
        raise ValueError(f"Iteration counts must not be negative, got {laplacian_iterations} and {optimization_iterations}")
    if surface_deviation < 0 or surface_travel < 0:
        raise ValueError(f"Surface allowances must not be negative, got {surface_deviation} and {surface_travel}")
    if not 0 < feature_angle < 180:
        raise ValueError(f"feature_angle must be between 0 and 180 degrees, got {feature_angle}")
    if aspect_weight < 0 or warp_weight < 0:
        raise ValueError(f"aspect_weight and warp_weight must not be negative, got {aspect_weight} and {warp_weight}")

    corners = _hexahedron_corners(mesh)
    samples = _gauss_samples(mesh)
    before = mesh_quality(mesh)
    constraints = _constraints(mesh.nodes, corners, surface_deviation, surface_travel, feature_angle, preserve_volume, mesh.straight_lines)
    positions = constraints.place_lines(mesh.nodes)
    if laplacian_iterations:
        positions = _smart_laplacian(positions, corners, constraints, laplacian_iterations, samples)
    if optimization_iterations:
        if _element_scaled_jacobians(positions, corners, samples).min() < _UNTANGLE_MARGIN:
            positions = _untangle(positions, corners, constraints, optimization_iterations, samples)
        if _any_inverted(positions, corners, samples).any():
            logger.warning("Skipping optimization because untangling left inverted elements")
        else:
            faces = _boundary_faces(corners) if warp_weight else None
            positions = _optimize(positions, corners, constraints, optimization_iterations, aspect_weight, faces, warp_weight, samples)

    result = replace(mesh, nodes=_move_high_order_nodes(mesh, corners, positions))
    after = mesh_quality(result)
    logger.debug(
        "Scaled Jacobian minimum {:.3f} -> {:.3f}, mean {:.3f} -> {:.3f}; Frobenius aspect maximum {:.3g} -> {:.3g}; inverted elements {} -> {}",
        before.minimum,
        after.minimum,
        before.mean,
        after.mean,
        before.max_aspect,
        after.max_aspect,
        before.inverted,
        after.inverted,
    )
    return result


def _hexahedron_corners(mesh: Mesh) -> np.ndarray:
    """Return the corner node indices of every hexahedron, (M, 8)."""
    others = [name for name in mesh.elements if not name.startswith("Hexahedron")]
    if others or not mesh.elements:
        raise ValueError(f"Hexahedral mesh optimization needs a mesh of only hexahedra, got {list(mesh.elements)}")
    return np.vstack([connectivity[:, :8] for connectivity in mesh.elements.values()])


def _corner_edges(nodes: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Return the three edge vectors leaving each corner, (M, 8, 3, 3), one vector per row."""
    return nodes[corners[:, _CORNER_NEIGHBORS]] - nodes[corners][:, :, None, :]


def _corner_scaled_jacobians(nodes: np.ndarray, corners: np.ndarray) -> np.ndarray:
    return _scaled_jacobians_of(_corner_edges(nodes, corners))


def _scaled_jacobians_of(edges: np.ndarray) -> np.ndarray:
    """Return the scaled Jacobian of each (..., 3, 3) matrix of edge or tangent rows."""
    lengths = np.linalg.norm(edges, axis=-1).prod(axis=-1)
    return np.linalg.det(edges) / np.maximum(lengths, np.finfo(float).tiny)


def _jacobian_terms(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the cofactor rows, determinants, and squared row lengths of (..., 3, 3) matrices of edge rows."""
    # For rows r0, r1, r2, the rows of E^-T are (r1 x r2, r2 x r0, r0 x r1) / det E.
    cofactors = np.cross(np.roll(edges, -1, axis=-2), np.roll(edges, -2, axis=-2))
    determinants = np.einsum("...i,...i->...", edges[..., 0, :], cofactors[..., 0, :])
    return cofactors, determinants, np.einsum("...i,...i->...", edges, edges)


def _gauss_barrier(tangents: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the sum of (1 / SJ - 1 / _GAUSS_BARRIER)^2 over Gauss points below _GAUSS_BARRIER, and its gradient by tangent.

    The sum is infinite, with a zero gradient, if any determinant is at or
    below zero. With d log SJ / dE as in _objective,
    d f / dE = -2 (1 / SJ - 1 / b) (1 / SJ) d log SJ / dE, which is 0 at
    SJ = b, so the barrier and its gradient are continuous there.
    """
    cofactors, determinants, squared = _jacobian_terms(tangents)
    if not (determinants > 0).all():
        return np.inf, np.zeros_like(tangents)
    inverse_quality = np.sqrt(squared).prod(axis=-1) / determinants
    excess = np.maximum(inverse_quality - 1 / _GAUSS_BARRIER, 0.0)
    log_gradient = cofactors / determinants[..., None, None] - tangents / squared[..., None]
    return float(np.sum(excess**2)), -2 * (excess * inverse_quality)[..., None, None] * log_gradient


def _corner_aspects(edges: np.ndarray) -> np.ndarray:
    """Return the Frobenius aspect ratio of each corner's (..., 3, 3) edge matrix, infinite where it is not positive."""
    determinants = np.linalg.det(edges)
    squared = np.einsum("...ij,...ij->...", edges, edges)
    positive = determinants > 0
    return np.where(positive, squared / (3 * np.where(positive, determinants, 1.0) ** (2 / 3)), np.inf)


def _objective(
    nodes: np.ndarray,
    corners: np.ndarray,
    aspect_weight: float = 0.0,
    faces: np.ndarray | None = None,
    warp_weight: float = 0.0,
    samples: Sequence[_GaussSamples] = (),
) -> tuple[float, np.ndarray]:
    """Return the sum of 1 / SJ plus aspect_weight times the sum of F^2 over all corners, and its gradient.

    The Gauss points of high-order elements add their barrier (_gauss_barrier).
    The sum is infinite, with a zero gradient, if any corner or Gauss point is degenerate or inverted.

    log SJ = log det E - sum log |e_k| for the matrix E whose rows are the
    edge vectors e_k, so d log SJ / dE = E^-T - e_k / |e_k|^2 row by row, and
    d (1 / SJ) = -(1 / SJ) d log SJ. The Frobenius aspect ratio is
    F = |E|_F^2 / (3 det(E)^(2/3)), so d log F / dE = 2 E / |E|_F^2 - (2/3) E^-T
    and d F^2 = 2 F^2 d log F.
    """
    edges = _corner_edges(nodes, corners)
    cofactors, determinants, squared = _jacobian_terms(edges)
    if not (determinants > 0).all():
        return np.inf, np.zeros_like(nodes)
    barriers = [_gauss_barrier(sample.tangents(nodes, corners)) for sample in samples]
    if not all(np.isfinite(barrier) for barrier, _ in barriers):
        return np.inf, np.zeros_like(nodes)
    inverse_quality = np.sqrt(squared).prod(axis=-1) / determinants
    inverse_transpose = cofactors / determinants[..., None, None]
    edge_gradient = -inverse_quality[..., None, None] * (inverse_transpose - edges / squared[..., None])
    energy = float(inverse_quality.sum())
    if aspect_weight:
        frobenius = squared.sum(axis=-1)
        squared_aspects = (frobenius / (3 * determinants ** (2 / 3))) ** 2
        energy += aspect_weight * float(squared_aspects.sum())
        log_aspect = 2 * edges / frobenius[..., None, None] - (2 / 3) * inverse_transpose
        edge_gradient += 2 * aspect_weight * squared_aspects[..., None, None] * log_aspect

    # Each edge vector is its far node minus the corner node.
    indices = np.concatenate((corners[:, _CORNER_NEIGHBORS].ravel(), corners.ravel()))
    values = np.concatenate((edge_gradient.reshape(-1, 3), -edge_gradient.sum(axis=2).reshape(-1, 3)))
    gradient = np.column_stack([np.bincount(indices, weights=values[:, axis], minlength=len(nodes)) for axis in range(3)])
    for sample, (barrier, tangent_gradient) in zip(samples, barriers, strict=True):
        energy += barrier
        gradient += sample.scatter(tangent_gradient, corners, len(nodes))
    if warp_weight and faces is not None:
        warp_energy, warp_gradient = _warp_objective(nodes, faces)
        energy += warp_weight * warp_energy
        gradient += warp_weight * warp_gradient
    return energy, gradient


def _warp_objective(nodes: np.ndarray, faces: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the sum over quadrilaterals of W = V^2 / A^3, and its gradient.

    V = (b - a) . ((c - a) x (d - a)) is six times the volume of the
    tetrahedron of a face's corners, 0 for a flat face, and A half the length
    of the cross product N of its diagonals, its area for a flat one, so W is
    dimensionless and about the square of the warp angle in radians over
    four for small warps. dV/db = (c - a) x (d - a) and cyclically, and for
    the diagonals u = c - a and v = d - b, dA/du = (v x n) / 2 and
    dA/dv = (n x u) / 2 with n = N / |N|.
    """
    a, b, c, d = (nodes[faces[:, index]] for index in range(4))
    volume = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))
    first, second = c - a, d - b
    normal = np.cross(first, second)
    length = np.maximum(np.linalg.norm(normal, axis=1), np.finfo(float).tiny)
    area = 0.5 * length
    unit = normal / length[:, None]
    energy = volume**2 / area**3
    d_volume = 2 * volume / area**3
    d_area = -3 * volume**2 / area**4
    by_volume = {1: np.cross(c - a, d - a), 2: np.cross(d - a, b - a), 3: np.cross(b - a, c - a)}
    by_volume[0] = -(by_volume[1] + by_volume[2] + by_volume[3])
    along_first, along_second = 0.5 * np.cross(second, unit), 0.5 * np.cross(unit, first)
    by_area = {0: -along_first, 1: -along_second, 2: along_first, 3: along_second}
    values = np.concatenate([d_volume[:, None] * by_volume[index] + d_area[:, None] * by_area[index] for index in range(4)])
    indices = faces.T.ravel()
    gradient = np.column_stack([np.bincount(indices, weights=values[:, axis], minlength=len(nodes)) for axis in range(3)])
    return float(energy.sum()), gradient


def _optimize(
    positions: np.ndarray,
    corners: np.ndarray,
    constraints: _Constraints,
    iterations: int,
    aspect_weight: float,
    faces: np.ndarray | None = None,
    warp_weight: float = 0.0,
    samples: Sequence[_GaussSamples] = (),
) -> np.ndarray:
    """Minimize the objective by projected gradient descent with backtracking."""
    # Scaling each node's step by its local size squared makes the step dimensionless.
    scale = _local_sizes(constraints.origin, corners)[:, None] ** 2
    energy, gradient = _objective(positions, corners, aspect_weight, faces, warp_weight, samples)
    step = _INITIAL_STEP
    for iteration in range(iterations):
        direction = scale * constraints.line_gradient(gradient)
        while step >= _MIN_STEP:
            trial = constraints.project(positions - step * direction)
            trial_energy, trial_gradient = _objective(trial, corners, aspect_weight, faces, warp_weight, samples)
            if trial_energy < energy:
                break
            step /= 2
        else:
            logger.debug("Optimization stalled after {} steps", iteration)
            break
        converged = energy - trial_energy < _CONVERGENCE * energy
        positions, energy, gradient = trial, trial_energy, trial_gradient
        step *= _STEP_GROWTH
        if converged:
            logger.debug("Optimization converged after {} steps", iteration + 1)
            break
    return positions


def _untangling_objective(nodes: np.ndarray, corners: np.ndarray, samples: Sequence[_GaussSamples] = ()) -> tuple[float, np.ndarray]:
    """Return the sum of max(0, _UNTANGLE_TARGET - SJ)^2 over all corners and Gauss points, and its gradient.

    SJ = det E / prod |e_k|, so dSJ / de_k = cof_k / prod |e_k| - SJ e_k / |e_k|^2
    for the cofactor rows cof_k of E. Unlike 1 / SJ, it is finite and smooth
    through det E = 0, so it can pull inverted corners back.
    """
    energy, edge_gradient = _untangling_terms(_corner_edges(nodes, corners))
    indices = np.concatenate((corners[:, _CORNER_NEIGHBORS].ravel(), corners.ravel()))
    values = np.concatenate((edge_gradient.reshape(-1, 3), -edge_gradient.sum(axis=2).reshape(-1, 3)))
    gradient = np.column_stack([np.bincount(indices, weights=values[:, axis], minlength=len(nodes)) for axis in range(3)])
    for sample in samples:
        sample_energy, tangent_gradient = _untangling_terms(sample.tangents(nodes, corners))
        energy += sample_energy
        gradient += sample.scatter(tangent_gradient, corners, len(nodes))
    return energy, gradient


def _untangling_terms(edges: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the sum of max(0, _UNTANGLE_TARGET - SJ)^2 over (..., 3, 3) matrices of edge rows, and its gradient by edge."""
    cofactors, determinants, squared = _jacobian_terms(edges)
    squared = np.maximum(squared, np.finfo(float).tiny)
    product = np.sqrt(squared).prod(axis=-1)
    quality = determinants / product
    shortfall = np.maximum(_UNTANGLE_TARGET - quality, 0.0)
    quality_gradient = cofactors / product[..., None, None] - quality[..., None, None] * edges / squared[..., None]
    return float(np.sum(shortfall**2)), -2 * shortfall[..., None, None] * quality_gradient


def _untangle(
    positions: np.ndarray, corners: np.ndarray, constraints: _Constraints, iterations: int, samples: Sequence[_GaussSamples] = ()
) -> np.ndarray:
    """Raise every corner's and Gauss point's scaled Jacobian toward _UNTANGLE_TARGET by projected gradient descent, stopping once all reach _UNTANGLE_MARGIN.

    Only corners and Gauss points below the target contribute, so the rest
    of the mesh moves only as far as the poor ones' nodes pull it.
    """
    scale = _local_sizes(constraints.origin, corners)[:, None] ** 2
    energy, gradient = _untangling_objective(positions, corners, samples)
    step = _INITIAL_STEP
    for iteration in range(iterations):
        if _element_scaled_jacobians(positions, corners, samples).min() >= _UNTANGLE_MARGIN:
            logger.debug("Untangled after {} steps", iteration)
            break
        direction = scale * constraints.line_gradient(gradient)
        while step >= _MIN_STEP:
            trial = constraints.project(positions - step * direction)
            trial_energy, trial_gradient = _untangling_objective(trial, corners, samples)
            if trial_energy < energy:
                break
            step /= 2
        else:
            logger.debug("Untangling stalled after {} steps", iteration)
            break
        positions, energy, gradient = trial, trial_energy, trial_gradient
        step *= _STEP_GROWTH
    return positions


def _smart_laplacian(
    positions: np.ndarray, corners: np.ndarray, constraints: _Constraints, iterations: int, samples: Sequence[_GaussSamples] = ()
) -> np.ndarray:
    """Move nodes toward their neighbors' average, keeping only moves that do not lower nearby element quality.

    An element's quality is its smallest scaled Jacobian over its corners and Gauss points.

    Interior nodes average all edge neighbors, surface nodes their neighbors
    along the boundary, and edge nodes their neighbors along feature edges.
    """
    adjacency = _smoothing_adjacency(corners, constraints)
    counts = np.asarray(adjacency.sum(axis=1)).ravel()
    movable = (constraints.kind != _FIXED) & (counts > 0)
    elements = np.repeat(np.arange(len(corners)), 8)
    for _ in range(iterations):
        quality = _element_scaled_jacobians(positions, corners, samples)
        average = adjacency @ positions / np.maximum(counts, 1)[:, None]
        trial = positions.copy()
        trial[movable] = average[movable]
        trial = constraints.project(trial)
        moved = movable.copy()
        # Undo moves around any element that got worse until none does; each round undoes at least one move.
        while True:
            trial_quality = _element_scaled_jacobians(trial, corners, samples)
            worse = elements[(trial_quality < quality - 1e-12)[elements]]
            undo = np.zeros(len(positions), dtype=bool)
            undo[corners[np.unique(worse)].ravel()] = True
            # A line's inner nodes move with its ends, so undo those.
            if len(constraints.lines):
                bent = undo[constraints.lines[:, 1:-1]].any(axis=1)
                undo[constraints.lines[bent][:, [0, -1]].ravel()] = True
            undo &= moved
            if not undo.any():
                break
            trial[undo] = positions[undo]
            trial = constraints.place_lines(trial)
            moved &= ~undo
        if not moved.any() or np.allclose(trial, positions):
            break
        positions = trial
    return positions


def _smoothing_adjacency(corners: np.ndarray, constraints: _Constraints) -> sparse.csr_array:
    """Return which neighbors each node averages in smoothing."""
    edges = np.unique(np.sort(corners[:, _EDGES].reshape(-1, 2), axis=1), axis=0)
    candidates = {_INTERIOR: edges, _SURFACE: constraints.boundary_edges, _EDGE: constraints.feature_edges}
    rows, columns = [], []
    for node_kind, pairs in candidates.items():
        for a, b in ((0, 1), (1, 0)):
            selected = pairs[constraints.kind[pairs[:, a]] == node_kind]
            rows.append(selected[:, a])
            columns.append(selected[:, b])
    row, column = np.concatenate(rows), np.concatenate(columns)
    size = len(constraints.kind)
    return sparse.csr_array((np.ones(len(row)), (row, column)), shape=(size, size))


def _boundary_faces(corners: np.ndarray) -> np.ndarray:
    """Return the quadrilaterals that belong to exactly one hexahedron, (F, 4), ordered outward."""
    faces = corners[:, _FACES].reshape(-1, 4)
    return faces[unshared(faces)]


def _face_area_vectors(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return each quadrilateral's outward normal scaled by its area, from the cross product of its diagonals."""
    points = positions[faces]
    return 0.5 * np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 1])


def _constraints(
    nodes: np.ndarray,
    corners: np.ndarray,
    deviation: float,
    travel: float,
    feature_angle: float,
    preserve_volume: bool = True,
    lines: np.ndarray | None = None,
) -> _Constraints:
    """Classify corner nodes as interior, surface, feature edge, fixed, or inner nodes of straight lines, with their allowances."""
    kind = np.full(len(nodes), _FIXED)
    kind[np.unique(corners)] = _INTERIOR
    direction = np.zeros_like(nodes)

    faces = _boundary_faces(corners)
    areas = _face_area_vectors(nodes, faces)
    normals = np.zeros_like(nodes)
    np.add.at(normals, faces.ravel(), np.repeat(areas, 4, axis=0))
    boundary = np.unique(faces)
    kind[boundary] = _SURFACE
    direction[boundary] = normals[boundary] / np.linalg.norm(normals[boundary], axis=1, keepdims=True)

    # Group the faces' edges so each boundary edge lists the faces that share it.
    face_edges = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
    boundary_edges, inverse, counts = np.unique(face_edges, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse.ravel(), kind="stable") // 4  # owning face of each occurrence, grouped by edge
    starts = np.cumsum(counts) - counts
    unit = areas / np.linalg.norm(areas, axis=1, keepdims=True)
    # An edge is a feature where its two faces meet at more than feature_angle,
    # or where the boundary is open or non-manifold and it has another number of faces.
    feature = counts != 2
    paired = np.flatnonzero(counts == 2)
    cosines = np.einsum("ij,ij->i", unit[order[starts[paired]]], unit[order[starts[paired] + 1]])
    feature[paired] = cosines < np.cos(np.radians(feature_angle))
    feature_edges = boundary_edges[feature]

    feature_count = np.bincount(feature_edges.ravel(), minlength=len(nodes))
    kind[feature_count == 1] = _FIXED
    kind[feature_count >= 3] = _FIXED
    for node in np.flatnonzero(feature_count == 2):
        ends = feature_edges[(feature_edges == node).any(axis=1)]
        neighbors = ends[ends != node]
        incoming, outgoing = nodes[node] - nodes[neighbors[0]], nodes[neighbors[1]] - nodes[node]
        incoming, outgoing = incoming / np.linalg.norm(incoming), outgoing / np.linalg.norm(outgoing)
        if incoming @ outgoing < np.cos(np.radians(feature_angle)):
            kind[node] = _FIXED
            continue
        kind[node] = _EDGE
        tangent = incoming + outgoing
        direction[node] = tangent / np.linalg.norm(tangent)
    lines = np.zeros((0, 2), dtype=np.int64) if lines is None else lines
    kind[lines[:, 1:-1].ravel()] = _LINE
    direction[(kind == _FIXED) | (kind == _INTERIOR) | (kind == _LINE)] = 0.0
    lengths = np.concatenate((np.zeros((len(lines), 1)), np.cumsum(np.linalg.norm(np.diff(nodes[lines], axis=1), axis=-1), axis=1)), axis=1)
    fractions = lengths / np.maximum(lengths[:, -1:], np.finfo(float).tiny)

    sizes = _local_sizes(nodes, corners)
    logger.debug(
        "Optimizing {} interior, {} surface, and {} feature edge nodes with {} fixed and {} on straight lines",
        *(int(np.count_nonzero(kind[np.unique(corners)] == value)) for value in (_INTERIOR, _SURFACE, _EDGE, _FIXED, _LINE)),
    )
    node_areas = np.where(kind == _SURFACE, np.linalg.norm(normals, axis=1) / 4, 0.0)
    edge_areas = np.where((kind == _EDGE)[:, None], normals / 4, 0.0)
    return _Constraints(
        kind,
        nodes.copy(),
        direction,
        deviation * sizes,
        travel * sizes,
        boundary_edges,
        feature_edges,
        node_areas,
        edge_areas,
        preserve_volume,
        lines,
        fractions,
    )


def _local_sizes(nodes: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Return each node's mean incident element edge length; 0 for nodes on no element."""
    edges = np.unique(np.sort(corners[:, _EDGES].reshape(-1, 2), axis=1), axis=0)
    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    totals = np.bincount(edges.ravel(), weights=np.repeat(lengths, 2), minlength=len(nodes))
    counts = np.bincount(edges.ravel(), minlength=len(nodes))
    return totals / np.maximum(counts, 1)


def _move_high_order_nodes(mesh: Mesh, corners: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return the original nodes with corners at positions and high-order nodes moved trilinearly with them."""
    displacement = positions - mesh.nodes
    result = mesh.nodes.copy()
    corner_nodes = np.unique(corners)
    result[corner_nodes] = positions[corner_nodes]
    with gmsh_session():
        for name, connectivity in mesh.elements.items():
            if connectivity.shape[1] == 8:
                continue
            local = np.reshape(gmsh.model.mesh.getElementProperties(mesh.element_types[name])[4], (-1, 3))
            # Trilinear shape functions of the eight corners at each high-order node's local coordinates.
            weights = np.prod(1 + local[8:, None, :] * local[None, :8, :], axis=-1) / 8
            moved = np.einsum("nc,ecx->enx", weights, displacement[connectivity[:, :8]])
            result[connectivity[:, 8:].ravel()] = mesh.nodes[connectivity[:, 8:].ravel()] + moved.reshape(-1, 3)
    return result


def _clip_norm(vectors: np.ndarray, limits: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    factors = np.where(norms > limits, limits / np.maximum(norms, np.finfo(float).tiny), 1.0)
    return vectors * factors[:, None]
