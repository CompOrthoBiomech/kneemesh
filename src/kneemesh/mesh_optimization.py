"""Node optimization of hexahedral meshes for the scaled Jacobian and the Frobenius aspect ratio.

Two strategies run in turn on the hexahedra's corner nodes:

1. Smart Laplacian smoothing moves each node toward the average of its edge
   neighbors, keeping a move only if it does not lower the worst scaled
   Jacobian of the elements around the node.
2. Optimization minimizes the sum of 1 / SJ over every element corner, plus
   aspect_weight times the sum of each corner's squared Frobenius aspect
   ratio, by projected gradient descent. Both terms grow without bound as a
   corner degenerates, so no step can invert an element, and both weigh the
   worst corners most. 1 / SJ alone squares corners up by stretching
   elements; the aspect term holds their edge lengths in proportion.

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

from dataclasses import dataclass, replace

import gmsh
import numpy as np
from loguru import logger
from scipy import sparse

from kneemesh.brep import gmsh_session
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


@dataclass(frozen=True)
class MeshQuality:
    """Scaled Jacobian and Frobenius aspect ratio statistics of a hexahedral mesh.

    Attributes:
        minimum: Smallest element scaled Jacobian. Negative means an inverted element.
        mean: Mean element scaled Jacobian.
        inverted: Number of elements with a corner Jacobian determinant at or below zero.
        max_aspect: Largest element Frobenius aspect ratio, infinite if any element is inverted.
        mean_aspect: Mean element Frobenius aspect ratio, infinite if any element is inverted.
    """

    minimum: float
    mean: float
    inverted: int
    max_aspect: float
    mean_aspect: float


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
        # Surface nodes: along is the normal part, across the tangential part.
        normal = np.clip(np.einsum("ij,ij->i", offsets, self.direction), -self.deviation, self.deviation)
        if self.preserve_volume:
            # Removing the area-weighted mean normal offset cancels the net volume
            # it adds; a few rounds keep the offsets within their limits too.
            for _ in range(_VOLUME_ROUNDS):
                shift = np.sum(self.areas * normal) / np.sum(self.areas)
                normal = np.clip(normal - shift * (self.areas > 0), -self.deviation, self.deviation)
        offsets[surface] = (_clip_norm(across, self.travel) + normal[:, None] * self.direction)[surface]
        edge = self.kind == _EDGE
        # Edge nodes: along is the tangential part, across the part off the line.
        tangential = np.clip(np.einsum("ij,ij->i", offsets, self.direction), -self.travel, self.travel)
        offsets[edge] = (tangential[:, None] * self.direction + _clip_norm(across, self.deviation))[edge]
        offsets[self.kind == _FIXED] = 0.0
        return self.place_lines(self.origin + offsets)


def scaled_jacobians(mesh: Mesh) -> np.ndarray:
    """Return each hexahedron's scaled Jacobian, the smallest over its eight corners.

    A corner's scaled Jacobian is the determinant of its three edge vectors
    divided by the product of their lengths: 1 for a right angle corner, 0
    for a flat one, and negative for an inverted one. High-order elements are
    measured on their corner nodes.

    Raises:
        ValueError: If the mesh has elements other than hexahedra.
    """
    return _corner_scaled_jacobians(mesh.nodes, _hexahedron_corners(mesh)).min(axis=1)


def frobenius_aspects(mesh: Mesh) -> np.ndarray:
    """Return each hexahedron's Frobenius aspect ratio, the largest over its eight corners.

    A corner's Frobenius aspect ratio is ``|E|_F^2 / (3 det(E)^(2/3))`` for the
    matrix E of its three edge vectors, as Verdict's maximum aspect Frobenius
    measures it: 1 for a cube's corner, and larger as the corner's edges
    differ in length or leave right angles. It is infinite for a degenerate
    or inverted corner. High-order elements are measured on their corner nodes.

    Raises:
        ValueError: If the mesh has elements other than hexahedra.
    """
    return _corner_aspects(_corner_edges(mesh.nodes, _hexahedron_corners(mesh))).max(axis=1)


def mesh_quality(mesh: Mesh) -> MeshQuality:
    """Summarize a hexahedral mesh's scaled Jacobians and Frobenius aspect ratios.

    Raises:
        ValueError: If the mesh has elements other than hexahedra.
    """
    corners = _hexahedron_corners(mesh)
    edges = _corner_edges(mesh.nodes, corners)
    determinants = np.linalg.det(edges)
    quality = _corner_scaled_jacobians(mesh.nodes, corners).min(axis=1)
    aspects = _corner_aspects(edges).max(axis=1)
    return MeshQuality(
        float(quality.min()),
        float(quality.mean()),
        int(np.count_nonzero((determinants <= 0).any(axis=1))),
        float(aspects.max()),
        float(aspects.mean()),
    )


def optimize_hexahedral_mesh(
    mesh: Mesh,
    laplacian_iterations: int = DEFAULT_LAPLACIAN_ITERATIONS,
    optimization_iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
    surface_deviation: float = DEFAULT_SURFACE_DEVIATION,
    surface_travel: float = DEFAULT_SURFACE_TRAVEL,
    feature_angle: float = DEFAULT_FEATURE_ANGLE,
    preserve_volume: bool = True,
    aspect_weight: float = 0.0,
) -> Mesh:
    """Move a hexahedral mesh's nodes to raise its scaled Jacobians, keeping connectivity.

    Smart Laplacian smoothing runs first, then optimization of the sum of
    1 / SJ over every element corner plus aspect_weight times the sum of their
    squared Frobenius aspect ratios. Optimization needs every element valid,
    so it is skipped with a warning if smoothing leaves one inverted.

    Boundary allowances scale with each node's local size, the mean length
    of the element edges at the node. Feature edges are boundary edges where
    the faces on either side meet at more than feature_angle; nodes where
    feature edges end, branch, or turn by more than feature_angle are fixed.

    Args:
        mesh: Mesh of linear or high-order hexahedra, such as hexahedral_mesh returns.
        laplacian_iterations: Smart Laplacian sweeps. 0 skips smoothing.
        optimization_iterations: Gradient steps of the optimization. 0 skips it.
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
    if aspect_weight < 0:
        raise ValueError(f"aspect_weight must not be negative, got {aspect_weight}")

    corners = _hexahedron_corners(mesh)
    before = mesh_quality(mesh)
    constraints = _constraints(mesh.nodes, corners, surface_deviation, surface_travel, feature_angle, preserve_volume, mesh.straight_lines)
    positions = constraints.place_lines(mesh.nodes)
    if laplacian_iterations:
        positions = _smart_laplacian(positions, corners, constraints, laplacian_iterations)
    if optimization_iterations:
        if (np.linalg.det(_corner_edges(positions, corners)) <= 0).any():
            logger.warning("Skipping optimization because the mesh has inverted elements")
        else:
            positions = _optimize(positions, corners, constraints, optimization_iterations, aspect_weight)

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
    edges = _corner_edges(nodes, corners)
    lengths = np.linalg.norm(edges, axis=-1).prod(axis=-1)
    return np.linalg.det(edges) / np.maximum(lengths, np.finfo(float).tiny)


def _corner_aspects(edges: np.ndarray) -> np.ndarray:
    """Return the Frobenius aspect ratio of each corner's (..., 3, 3) edge matrix, infinite where it is not positive."""
    determinants = np.linalg.det(edges)
    squared = np.einsum("...ij,...ij->...", edges, edges)
    positive = determinants > 0
    return np.where(positive, squared / (3 * np.where(positive, determinants, 1.0) ** (2 / 3)), np.inf)


def _objective(nodes: np.ndarray, corners: np.ndarray, aspect_weight: float = 0.0) -> tuple[float, np.ndarray]:
    """Return the sum of 1 / SJ plus aspect_weight times the sum of F^2 over all corners, and its gradient.

    The sum is infinite, with a zero gradient, if any corner is degenerate or inverted.

    log SJ = log det E - sum log |e_k| for the matrix E whose rows are the
    edge vectors e_k, so d log SJ / dE = E^-T - e_k / |e_k|^2 row by row, and
    d (1 / SJ) = -(1 / SJ) d log SJ. The Frobenius aspect ratio is
    F = |E|_F^2 / (3 det(E)^(2/3)), so d log F / dE = 2 E / |E|_F^2 - (2/3) E^-T
    and d F^2 = 2 F^2 d log F.
    """
    edges = _corner_edges(nodes, corners)
    # For rows r0, r1, r2, the rows of E^-T are (r1 x r2, r2 x r0, r0 x r1) / det E.
    cofactors = np.cross(np.roll(edges, -1, axis=-2), np.roll(edges, -2, axis=-2))
    determinants = np.einsum("...i,...i->...", edges[..., 0, :], cofactors[..., 0, :])
    if not (determinants > 0).all():
        return np.inf, np.zeros_like(nodes)
    squared = np.einsum("...i,...i->...", edges, edges)
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
    return energy, gradient


def _optimize(positions: np.ndarray, corners: np.ndarray, constraints: _Constraints, iterations: int, aspect_weight: float) -> np.ndarray:
    """Minimize the objective by projected gradient descent with backtracking."""
    # Scaling each node's step by its local size squared makes the step dimensionless.
    scale = _local_sizes(constraints.origin, corners)[:, None] ** 2
    energy, gradient = _objective(positions, corners, aspect_weight)
    step = _INITIAL_STEP
    for iteration in range(iterations):
        direction = scale * constraints.line_gradient(gradient)
        while step >= _MIN_STEP:
            trial = constraints.project(positions - step * direction)
            trial_energy, trial_gradient = _objective(trial, corners, aspect_weight)
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


def _smart_laplacian(positions: np.ndarray, corners: np.ndarray, constraints: _Constraints, iterations: int) -> np.ndarray:
    """Move nodes toward their neighbors' average, keeping only moves that do not lower nearby element quality.

    Interior nodes average all edge neighbors, surface nodes their neighbors
    along the boundary, and edge nodes their neighbors along feature edges.
    """
    adjacency = _smoothing_adjacency(corners, constraints)
    counts = np.asarray(adjacency.sum(axis=1)).ravel()
    movable = (constraints.kind != _FIXED) & (counts > 0)
    elements = np.repeat(np.arange(len(corners)), 8)
    for _ in range(iterations):
        quality = _corner_scaled_jacobians(positions, corners).min(axis=1)
        average = adjacency @ positions / np.maximum(counts, 1)[:, None]
        trial = positions.copy()
        trial[movable] = average[movable]
        trial = constraints.project(trial)
        moved = movable.copy()
        # Undo moves around any element that got worse until none does; each round undoes at least one move.
        while True:
            trial_quality = _corner_scaled_jacobians(trial, corners).min(axis=1)
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
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.ravel()] == 1]


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
    return _Constraints(
        kind, nodes.copy(), direction, deviation * sizes, travel * sizes, boundary_edges, feature_edges, node_areas, preserve_volume, lines, fractions
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
