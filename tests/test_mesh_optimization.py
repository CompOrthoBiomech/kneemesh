import numpy as np
import pytest
from loguru import logger

from kneemesh.geo.meniscus import centerline, contours_to_hex_blocks, slice_isocontour
from kneemesh.mesh.meniscus import hexahedral_mesh
from kneemesh.mesh_optimization import (
    _constraints,
    _hexahedron_corners,
    _local_sizes,
    _objective,
    _untangling_objective,
    frobenius_aspects,
    mesh_quality,
    optimize_hexahedral_mesh,
    scaled_jacobians,
)
from kneemesh.tet_meshing import Mesh
from kneemesh.vol import extract_isocontour, signed_distance_field, smooth_isocontour

# A hexahedron as six tetrahedra around its 0-6 diagonal.
HEXAHEDRON_TETRAHEDRA = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
UNIT_CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)


def hexahedra(nodes, connectivity):
    return Mesh(np.asarray(nodes, dtype=float), {"Hexahedron 8": np.asarray(connectivity)}, {"Hexahedron 8": 5})


def cube_grid(divisions, perturbation=0.0, seed=0):
    """A unit cube of divisions^3 hexahedra with interior nodes moved randomly by up to perturbation cells."""
    axis = np.linspace(0.0, 1.0, divisions + 1)
    nodes = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)

    def index(i, j, k):
        return (i * (divisions + 1) + j) * (divisions + 1) + k

    cells = [
        [index(i, j, k), index(i + 1, j, k), index(i + 1, j + 1, k), index(i, j + 1, k)]
        + [index(i, j, k + 1), index(i + 1, j, k + 1), index(i + 1, j + 1, k + 1), index(i, j + 1, k + 1)]
        for i in range(divisions)
        for j in range(divisions)
        for k in range(divisions)
    ]
    interior = np.all((nodes > 1e-9) & (nodes < 1 - 1e-9), axis=1)
    nodes[interior] += np.random.default_rng(seed).uniform(-perturbation, perturbation, (int(interior.sum()), 3)) / divisions
    return hexahedra(nodes, cells)


def tangled_grid():
    """A 6^3 grid whose random interior perturbation inverts one element."""
    mesh = cube_grid(6, perturbation=0.3)
    assert mesh_quality(mesh).inverted == 1
    return mesh


def volume(mesh):
    total = 0.0
    for connectivity in mesh.elements.values():
        points = mesh.nodes[connectivity[:, :8]]
        for a, b, c, d in HEXAHEDRON_TETRAHEDRA:
            edges = points[:, b] - points[:, a], points[:, c] - points[:, a], points[:, d] - points[:, a]
            total += np.einsum("ij,ij->i", edges[0], np.cross(edges[1], edges[2])).sum() / 6
    return total


def on_cube_faces(nodes):
    return np.isclose(nodes, 0.0) | np.isclose(nodes, 1.0)


@pytest.fixture(scope="module")
def real_hex_meshes(real_label, real_medial_label):
    """Hexahedral meshes of the real lateral and medial menisci."""
    meshes = []
    for label in (real_label, real_medial_label):
        surface = smooth_isocontour(extract_isocontour(signed_distance_field(label)))
        slices = slice_isocontour(surface, centerline(label), spacing=5.0, points_per_slice=10)
        meshes.append(hexahedral_mesh(contours_to_hex_blocks(slices), elements_per_edge=4, elements_along=20))
    return meshes


@pytest.fixture(scope="module")
def optimized_real_hex_meshes(real_hex_meshes):
    return [optimize_hexahedral_mesh(mesh) for mesh in real_hex_meshes]


def test_scaled_jacobian_is_one_for_a_cube_and_lower_for_a_sheared_one():
    sheared = UNIT_CUBE + np.outer(UNIT_CUBE[:, 2], [0.5, 0.0, 0.0])

    assert scaled_jacobians(hexahedra(UNIT_CUBE, [range(8)])) == pytest.approx([1.0])
    assert scaled_jacobians(hexahedra(sheared, [range(8)])) == pytest.approx([1 / np.sqrt(1.25)])


def test_scaled_jacobian_is_negative_for_an_inverted_hexahedron():
    inverted = UNIT_CUBE.copy()
    inverted[6] = [0.2, 0.2, -0.5]

    quality = mesh_quality(hexahedra(inverted, [range(8)]))

    assert quality.minimum < 0
    assert quality.inverted == 1


def test_objective_gradient_matches_finite_differences():
    mesh = cube_grid(3, perturbation=0.15)
    corners = _hexahedron_corners(mesh)
    _, gradient = _objective(mesh.nodes, corners)
    step = 1e-6

    for node, axis in [(21, 0), (22, 1), (41, 2)]:
        ahead, behind = mesh.nodes.copy(), mesh.nodes.copy()
        ahead[node, axis] += step
        behind[node, axis] -= step
        difference = (_objective(ahead, corners)[0] - _objective(behind, corners)[0]) / (2 * step)
        assert gradient[node, axis] == pytest.approx(difference, rel=1e-5)


def test_frobenius_aspect_is_one_for_a_cube_and_grows_with_stretching_and_shear():
    box = UNIT_CUBE * [4.0, 1.0, 1.0]
    sheared = UNIT_CUBE + np.outer(UNIT_CUBE[:, 2], [1.0, 0.0, 0.0])

    assert frobenius_aspects(hexahedra(UNIT_CUBE, [range(8)])) == pytest.approx([1.0])
    # (16 + 1 + 1) / (3 * 4^(2/3)) for a 4 x 1 x 1 box.
    assert frobenius_aspects(hexahedra(box, [range(8)])) == pytest.approx([18 / (3 * 4 ** (2 / 3))])
    # A unit shear makes one corner's edges 1, 1, and sqrt(2) long with the cube's volume.
    assert frobenius_aspects(hexahedra(sheared, [range(8)])) == pytest.approx([4 / 3])


def test_mesh_quality_reports_aspect_ratios_and_inverted_elements_as_infinite():
    inverted = UNIT_CUBE.copy()
    inverted[6] = [0.2, 0.2, -0.5]

    assert mesh_quality(hexahedra(UNIT_CUBE, [range(8)])).max_aspect == pytest.approx(1.0)
    assert mesh_quality(hexahedra(inverted, [range(8)])).max_aspect == np.inf


def test_objective_gradient_with_aspect_term_matches_finite_differences():
    mesh = cube_grid(3, perturbation=0.15)
    corners = _hexahedron_corners(mesh)
    _, gradient = _objective(mesh.nodes, corners, 0.7)
    step = 1e-6

    for node, axis in [(21, 0), (22, 1), (41, 2)]:
        ahead, behind = mesh.nodes.copy(), mesh.nodes.copy()
        ahead[node, axis] += step
        behind[node, axis] -= step
        difference = (_objective(ahead, corners, 0.7)[0] - _objective(behind, corners, 0.7)[0]) / (2 * step)
        assert gradient[node, axis] == pytest.approx(difference, rel=1e-5)


def z_lines(divisions):
    """Each column of a cube_grid's nodes along z through its interior, from z = 0 to z = 1."""
    columns = [
        [(i * (divisions + 1) + j) * (divisions + 1) + k for k in range(divisions + 1)] for i in range(1, divisions) for j in range(1, divisions)
    ]
    return np.array(columns)


def test_line_gradient_passes_inner_nodes_shares_to_their_ends():
    mesh = cube_grid(3, perturbation=0.15)
    corners = _hexahedron_corners(mesh)
    lines = z_lines(3)
    constraints = _constraints(mesh.nodes, corners, 0.1, 0.5, 30.0, lines=lines)
    placed = constraints.place_lines(mesh.nodes)
    gradient = constraints.line_gradient(_objective(placed, corners)[1])
    step = 1e-6

    assert np.all(gradient[lines[:, 1:-1].ravel()] == 0.0)
    for node, axis in [(lines[0, 0], 0), (lines[1, -1], 2), (lines[2, 0], 1)]:
        ahead, behind = placed.copy(), placed.copy()
        ahead[node, axis] += step
        behind[node, axis] -= step
        difference = (_objective(constraints.place_lines(ahead), corners)[0] - _objective(constraints.place_lines(behind), corners)[0]) / (2 * step)
        assert gradient[node, axis] == pytest.approx(difference, rel=1e-5)


def test_straight_lines_stay_straight_at_their_fractions():
    divisions = 4
    mesh = cube_grid(divisions, perturbation=0.2)
    lines = z_lines(divisions)
    straight = Mesh(mesh.nodes, mesh.elements, mesh.element_types, lines)

    optimized = optimize_hexahedral_mesh(straight)

    points = optimized.nodes[lines]
    fractions = np.linspace(0.0, 1.0, divisions + 1)[None, :, None]
    lengths = np.linalg.norm(np.diff(mesh.nodes[lines], axis=1), axis=-1)
    original = np.concatenate((np.zeros((len(lines), 1)), np.cumsum(lengths, axis=1)), axis=1) / lengths.sum(axis=1, keepdims=True)
    # Inner nodes keep their starting fractions of the way along their lines, which the perturbation made uneven.
    assert points == pytest.approx(points[:, :1] + original[..., None] * (points[:, -1:] - points[:, :1]), abs=1e-12)
    assert not np.allclose(original, fractions[..., 0])
    assert mesh_quality(optimized).minimum > mesh_quality(mesh).minimum


def test_aspect_weight_must_not_be_negative():
    with pytest.raises(ValueError, match="aspect_weight"):
        optimize_hexahedral_mesh(cube_grid(2), aspect_weight=-1.0)


def test_cube_nodes_are_classified_by_their_boundary_features():
    mesh = cube_grid(5)
    constraints = _constraints(mesh.nodes, _hexahedron_corners(mesh), 0.1, 0.5, 30.0)

    # 4^3 interior nodes, 4^2 on each of 6 faces, 4 on each of 12 edges, and 8 corners.
    assert np.bincount(constraints.kind).tolist() == [64, 96, 48, 8]

    # Lines through the interior along z have 4 inner nodes each, all interior.
    with_lines = _constraints(mesh.nodes, _hexahedron_corners(mesh), 0.1, 0.5, 30.0, lines=z_lines(5))
    assert np.bincount(with_lines.kind).tolist() == [0, 96, 48, 8, 64]


def test_optimization_restores_a_distorted_grid():
    mesh = tangled_grid()

    optimized = optimize_hexahedral_mesh(mesh, surface_deviation=0.0)

    quality = mesh_quality(optimized)
    assert quality.inverted == 0
    assert quality.minimum > 0.99


def test_optimization_without_deviation_keeps_boundary_nodes_on_flat_faces():
    mesh = cube_grid(5, perturbation=0.3)
    faces = on_cube_faces(mesh.nodes)

    optimized = optimize_hexahedral_mesh(mesh, surface_deviation=0.0)

    assert np.allclose(optimized.nodes[faces], mesh.nodes[faces])
    corners = faces.all(axis=1)
    assert np.array_equal(optimized.nodes[corners], mesh.nodes[corners])


def test_smart_laplacian_never_lowers_the_worst_element():
    mesh = cube_grid(5, perturbation=0.15, seed=3)

    smoothed = optimize_hexahedral_mesh(mesh, optimization_iterations=0)

    assert mesh_quality(smoothed).minimum >= mesh_quality(mesh).minimum
    assert mesh_quality(smoothed).mean > mesh_quality(mesh).mean


def test_untangling_objective_gradient_matches_finite_differences_through_inverted_corners():
    mesh = tangled_grid()
    corners = _hexahedron_corners(mesh)
    _, gradient = _untangling_objective(mesh.nodes, corners)
    step = 1e-6

    # The inverted element's nodes include corners on both sides of zero and above the target.
    for node in np.unique(corners[scaled_jacobians(mesh) <= 0]):
        for axis in range(3):
            ahead, behind = mesh.nodes.copy(), mesh.nodes.copy()
            ahead[node, axis] += step
            behind[node, axis] -= step
            difference = (_untangling_objective(ahead, corners)[0] - _untangling_objective(behind, corners)[0]) / (2 * step)
            assert gradient[node, axis] == pytest.approx(difference, rel=1e-4, abs=1e-8)


def test_optimization_untangles_inverted_elements_without_smoothing():
    mesh = tangled_grid()

    optimized = optimize_hexahedral_mesh(mesh, laplacian_iterations=0, surface_deviation=0.0)

    quality = mesh_quality(optimized)
    assert quality.inverted == 0
    assert quality.minimum > 0.99


def test_optimization_is_skipped_when_elements_stay_inverted():
    # Every node of a lone hexahedron is a feature corner, so none can move.
    inverted = UNIT_CUBE.copy()
    inverted[6] = [0.2, 0.2, -0.5]
    mesh = hexahedra(inverted, [range(8)])
    messages = []
    logger.enable("kneemesh")
    handler = logger.add(messages.append, level="WARNING")
    try:
        result = optimize_hexahedral_mesh(mesh, laplacian_iterations=0)
    finally:
        logger.remove(handler)
        logger.disable("kneemesh")

    assert np.array_equal(result.nodes, mesh.nodes)
    assert any("inverted" in message for message in messages)


def test_optimization_with_no_iterations_keeps_the_nodes():
    mesh = cube_grid(3, perturbation=0.2)

    result = optimize_hexahedral_mesh(mesh, laplacian_iterations=0, optimization_iterations=0)

    assert np.array_equal(result.nodes, mesh.nodes)
    assert result.elements is mesh.elements


def test_optimization_raises_the_quality_of_real_menisci(real_hex_meshes, optimized_real_hex_meshes):
    for mesh, optimized in zip(real_hex_meshes, optimized_real_hex_meshes):
        before, after = mesh_quality(mesh), mesh_quality(optimized)
        assert after.inverted == 0
        assert after.minimum > before.minimum + 0.1
        assert after.mean > before.mean + 0.05


def test_optimization_keeps_the_volume_of_real_menisci(real_hex_meshes, optimized_real_hex_meshes):
    for mesh, optimized in zip(real_hex_meshes, optimized_real_hex_meshes):
        assert volume(optimized) == pytest.approx(volume(mesh), rel=0.01)


def test_optimization_without_volume_preservation_swells_the_mesh(real_hex_meshes):
    mesh = real_hex_meshes[1]

    swollen = optimize_hexahedral_mesh(mesh, preserve_volume=False)

    assert volume(swollen) > 1.02 * volume(mesh)


def test_surface_nodes_stay_within_their_allowances(real_hex_meshes, optimized_real_hex_meshes):
    for mesh, optimized in zip(real_hex_meshes, optimized_real_hex_meshes):
        corners = _hexahedron_corners(mesh)
        constraints = _constraints(mesh.nodes, corners, 0.1, 0.5, 30.0)
        sizes = _local_sizes(mesh.nodes, corners)
        offsets = optimized.nodes - mesh.nodes
        along = np.einsum("ij,ij->i", offsets, constraints.direction)
        across = np.linalg.norm(offsets - along[:, None] * constraints.direction, axis=1)

        surface, edge, fixed = (constraints.kind == kind for kind in (1, 2, 3))
        assert np.all(np.abs(along[surface]) <= 0.1 * sizes[surface] + 1e-9)
        assert np.all(across[surface] <= 0.5 * sizes[surface] + 1e-9)
        assert np.all(np.abs(along[edge]) <= 0.5 * sizes[edge] + 1e-9)
        assert np.all(across[edge] <= 0.1 * sizes[edge] + 1e-9)
        assert np.array_equal(optimized.nodes[fixed], mesh.nodes[fixed])


@pytest.mark.parametrize("incomplete", [False, True])
def test_high_order_nodes_follow_their_corners(meniscus_hex_blocks, incomplete):
    linear = hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=3, elements_along=12)
    quadratic = hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=3, elements_along=12, element_order=2, second_order_incomplete=incomplete)
    name = next(iter(quadratic.elements))
    connectivity = quadratic.elements[name]

    optimized = optimize_hexahedral_mesh(quadratic)

    # Optimization moves only the corners, as it does in the linear mesh with the same corners.
    expected = mesh_quality(optimize_hexahedral_mesh(linear))
    assert mesh_quality(optimized).minimum == pytest.approx(expected.minimum, abs=1e-6)
    assert mesh_quality(optimized).mean == pytest.approx(expected.mean, abs=1e-6)

    # Node 8 lies on the edge from corner 0 to corner 1, and keeps its offset from that edge's midpoint.
    def offsets(nodes):
        return nodes[connectivity[:, 8]] - (nodes[connectivity[:, 0]] + nodes[connectivity[:, 1]]) / 2

    assert np.allclose(offsets(optimized.nodes), offsets(quadratic.nodes))


def test_optimization_rejects_tetrahedra(meniscus_tetrahedra):
    with pytest.raises(ValueError, match="only hexahedra"):
        optimize_hexahedral_mesh(meniscus_tetrahedra)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"laplacian_iterations": -1}, "Iteration counts"),
        ({"optimization_iterations": -1}, "Iteration counts"),
        ({"surface_deviation": -0.1}, "allowances"),
        ({"surface_travel": -0.1}, "allowances"),
        ({"feature_angle": 0.0}, "feature_angle"),
        ({"feature_angle": 180.0}, "feature_angle"),
    ],
)
def test_optimization_rejects_invalid_arguments(arguments, message):
    with pytest.raises(ValueError, match=message):
        optimize_hexahedral_mesh(cube_grid(2), **arguments)
