from dataclasses import replace

import gmsh
import numpy as np
import pytest
from loguru import logger

from kneemesh.brep import gmsh_session
from kneemesh.geo.meniscus import centerline, contours_to_hex_blocks, slice_isocontour
from kneemesh.mesh.meniscus import hexahedral_mesh
from kneemesh.mesh_optimization import (
    _constraints,
    _corner_scaled_jacobians,
    _gauss_samples,
    _hexahedron_corners,
    _local_sizes,
    _move_high_order_nodes,
    _objective,
    _untangling_objective,
    _warp_objective,
    element_quality,
    frobenius_aspects,
    mesh_quality,
    optimize_hexahedral_mesh,
    scaled_jacobians,
    surface_warps,
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
def real_hex_blocks(real_label, real_medial_label):
    """Butterfly blocks of the real lateral and medial menisci."""
    blocks = []
    for label in (real_label, real_medial_label):
        surface = smooth_isocontour(extract_isocontour(signed_distance_field(label)))
        blocks.append(contours_to_hex_blocks(slice_isocontour(surface, centerline(label), spacing=5.0, points_per_slice=10)))
    return blocks


@pytest.fixture(scope="module")
def real_hex_meshes(real_hex_blocks):
    """Hexahedral meshes of the real lateral and medial menisci."""
    return [hexahedral_mesh(blocks, transverse_size=1.5, circumferential_size=4.0) for blocks in real_hex_blocks]


def gauss_determinants(mesh):
    """Jacobian determinants of every element of one type at the points of its Gauss rules, from gmsh's basis functions.

    The rules are exact to orders 2p - 1 and 2p + 1 for hexahedra of order p, 2p - 2 and 2p for tetrahedra.
    """
    determinants = []
    with gmsh_session():
        for name, connectivity in mesh.elements.items():
            element_type = mesh.element_types[name]
            order = gmsh.model.mesh.getElementProperties(element_type)[2]
            degrees = (2 * order - 2, 2 * order) if name.startswith("Tetrahedron") else (2 * order - 1, 2 * order + 1)
            for rule in (f"Gauss{degree}" for degree in degrees):
                points, _ = gmsh.model.mesh.getIntegrationPoints(element_type, rule)
                _, gradients, _ = gmsh.model.mesh.getBasisFunctions(element_type, points, "GradLagrange")
                gradients = np.reshape(gradients, (len(points) // 3, connectivity.shape[1], 3))
                determinants.append(np.linalg.det(np.einsum("gni,enx->egix", gradients, mesh.nodes[connectivity])))
    return np.concatenate(determinants, axis=1)


def quadratic_hexahedra(mesh, name="Hexahedron 20", gmsh_type=17):
    """A linear hexahedral mesh's elements with straight-sided second-order nodes, each element's its own."""
    corners = mesh.elements["Hexahedron 8"]
    with gmsh_session():
        local = np.reshape(gmsh.model.mesh.getElementProperties(gmsh_type)[4], (-1, 3))
    weights = np.prod(1 + local[8:, None, :] * local[None, :8, :], axis=-1) / 8
    extra = np.einsum("nc,ecx->enx", weights, mesh.nodes[corners]).reshape(-1, 3)
    numbers = len(mesh.nodes) + np.arange(len(extra)).reshape(len(corners), -1)
    return Mesh(np.vstack([mesh.nodes, extra]), {name: np.hstack([corners, numbers])}, {name: gmsh_type})


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


def test_warp_weight_must_not_be_negative():
    with pytest.raises(ValueError, match="warp_weight"):
        optimize_hexahedral_mesh(cube_grid(2), warp_weight=-1.0)


def test_surface_warps_are_zero_for_flat_faces_and_grow_with_a_lifted_corner():
    lifted = UNIT_CUBE.copy()
    lifted[6] += [0.0, 0.0, 0.3]

    assert surface_warps(hexahedra(UNIT_CUBE, [range(8)])) == pytest.approx(np.zeros(6), abs=1e-9)
    warps = surface_warps(hexahedra(lifted, [range(8)]))
    # Raised along z, node 6 stays in the planes of the two sides it lies on, so only the top face warps.
    assert np.count_nonzero(warps > 1.0) == 1
    quality = mesh_quality(hexahedra(lifted, [range(8)]))
    assert quality.max_warp == pytest.approx(warps.max())
    assert 0 < quality.warp_99 <= quality.max_warp


def test_warp_objective_gradient_matches_finite_differences():
    mesh = cube_grid(3, perturbation=0.2)
    rng = np.random.default_rng(1)
    nodes = mesh.nodes + rng.uniform(-0.05, 0.05, mesh.nodes.shape)
    faces = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [16, 17, 21, 20]])
    _, gradient = _warp_objective(nodes, faces)
    step = 1e-6

    for node, axis in [(0, 0), (5, 1), (17, 2), (21, 0)]:
        ahead, behind = nodes.copy(), nodes.copy()
        ahead[node, axis] += step
        behind[node, axis] -= step
        difference = (_warp_objective(ahead, faces)[0] - _warp_objective(behind, faces)[0]) / (2 * step)
        assert gradient[node, axis] == pytest.approx(difference, rel=1e-5, abs=1e-9)
    assert _warp_objective(UNIT_CUBE, np.array([[0, 1, 2, 3]]))[0] == pytest.approx(0.0, abs=1e-12)


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
    linear = hexahedral_mesh(meniscus_hex_blocks, transverse_size=2.4, circumferential_size=6.25)
    quadratic = hexahedral_mesh(
        meniscus_hex_blocks, transverse_size=2.4, circumferential_size=6.25, element_order=2, second_order_incomplete=incomplete
    )
    name = next(iter(quadratic.elements))
    connectivity = quadratic.elements[name]

    optimized = optimize_hexahedral_mesh(quadratic)

    # Only the corners move, as in the linear mesh with the same corners, though the Gauss points' barrier moves them differently.
    linear_corners = _hexahedron_corners(linear)
    assert np.allclose(quadratic.nodes[connectivity[:, :8]], linear.nodes[linear_corners])
    assert not np.allclose(optimized.nodes[connectivity[:, :8]], quadratic.nodes[connectivity[:, :8]])

    # Node 8 lies on the edge from corner 0 to corner 1, and keeps its offset from that edge's midpoint.
    def offsets(nodes):
        return nodes[connectivity[:, 8]] - (nodes[connectivity[:, 0]] + nodes[connectivity[:, 1]]) / 2

    assert np.allclose(offsets(optimized.nodes), offsets(quadratic.nodes))


def test_high_order_hexahedra_are_measured_at_their_gauss_points():
    straight = quadratic_hexahedra(hexahedra(UNIT_CUBE, [range(8)]))
    assert scaled_jacobians(straight) == pytest.approx([1.0])
    assert mesh_quality(straight).inverted == 0

    # Node 8 is the middle of the edge from corner 0 to corner 1; pushed far inward, it folds the element near that edge.
    bent = replace(straight, nodes=straight.nodes.copy())
    bent.nodes[8] += [0.0, 1.5, 0.0]
    corners = _hexahedron_corners(bent)
    assert _corner_scaled_jacobians(bent.nodes, corners).min() == pytest.approx(1.0)
    assert (gauss_determinants(bent) <= 0).any()
    assert scaled_jacobians(bent)[0] < 0
    assert mesh_quality(bent).inverted == 1
    assert frobenius_aspects(bent)[0] == np.inf
    quality = element_quality(bent)
    assert quality["scaled_jacobian"][0] < 0
    assert quality["frobenius_aspect"][0] == np.inf

    # Bent less, it stays valid but measures lower at its Gauss points than at its corners.
    bent.nodes[8] = straight.nodes[8] + [0.0, 0.3, 0.0]
    assert 0 < scaled_jacobians(bent)[0] < 0.95


@pytest.mark.parametrize(("name", "gmsh_type"), [("Hexahedron 20", 17), ("Hexahedron 27", 12)])
def test_gauss_point_jacobians_follow_moved_corners_as_gmsh_computes_them(name, gmsh_type):
    mesh = quadratic_hexahedra(cube_grid(2), name, gmsh_type)
    rng = np.random.default_rng(1)
    mesh.nodes[27:] += rng.uniform(-0.03, 0.03, mesh.nodes[27:].shape)
    corners = _hexahedron_corners(mesh)
    (samples,) = _gauss_samples(mesh)
    moved = mesh.nodes.copy()
    moved[:27] += rng.uniform(-0.05, 0.05, (27, 3))

    expected = gauss_determinants(replace(mesh, nodes=_move_high_order_nodes(mesh, corners, moved)))

    # The samples are the corner nodes, then the Gauss points.
    assert np.allclose(np.linalg.det(samples.tangents(moved, corners))[:, 8:], expected, rtol=1e-10, atol=1e-12)


def bent_quadratic_grid():
    """A perturbed 3^3 grid of 20-node hexahedra whose second-order nodes are moved randomly, leaving two below the barrier, the worst at 0.15."""
    mesh = quadratic_hexahedra(cube_grid(3, perturbation=0.15))
    mesh.nodes[64:] += np.random.default_rng(3).uniform(-0.04, 0.04, mesh.nodes[64:].shape)
    return mesh


def test_objective_gradient_with_gauss_points_matches_finite_differences():
    mesh = bent_quadratic_grid()
    corners = _hexahedron_corners(mesh)
    samples = _gauss_samples(mesh)
    assert 0 < scaled_jacobians(mesh).min() < 0.2
    direction = np.zeros_like(mesh.nodes)
    direction[:64] = np.random.default_rng(3).normal(size=(64, 3))
    step = 1e-6

    def objective(nodes):
        return _objective(nodes, corners, 0.3, None, 0.0, samples)

    def untangling(nodes):
        return _untangling_objective(nodes, corners, samples)

    assert objective(mesh.nodes)[0] > _objective(mesh.nodes, corners, 0.3)[0]
    for function in (objective, untangling):
        difference = (function(mesh.nodes + step * direction)[0] - function(mesh.nodes - step * direction)[0]) / (2 * step)
        assert np.sum(function(mesh.nodes)[1] * direction) == pytest.approx(difference, rel=1e-6)


def test_objective_is_infinite_when_a_gauss_point_is_inverted():
    mesh = quadratic_hexahedra(hexahedra(UNIT_CUBE, [range(8)]))
    mesh.nodes[8] += [0.0, 1.5, 0.0]
    corners = _hexahedron_corners(mesh)

    assert np.isfinite(_objective(mesh.nodes, corners)[0])
    assert _objective(mesh.nodes, corners, samples=_gauss_samples(mesh))[0] == np.inf


@pytest.mark.parametrize("incomplete", [False, True])
def test_optimization_leaves_no_real_meniscus_hexahedron_inverted_at_a_gauss_point(real_hex_blocks, incomplete):
    # Measured on corners alone, optimization left 39 of the real lateral meniscus's 855 20-node hexahedra inverted at
    # Gauss points, though none started so.
    mesh = hexahedral_mesh(real_hex_blocks[0], 1.5, 4.0, element_order=2, second_order_incomplete=incomplete)

    optimized = optimize_hexahedral_mesh(mesh)

    assert (gauss_determinants(optimized) > 0).all()
    assert mesh_quality(optimized).inverted == 0


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


REGULAR_TETRAHEDRON = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]])


def single(nodes, name, gmsh_type, connectivity=None):
    nodes = np.asarray(nodes, dtype=float)
    rows = np.asarray([connectivity if connectivity is not None else range(len(nodes))])
    return Mesh(nodes, {name: rows}, {name: gmsh_type})


def test_element_quality_is_one_for_ideal_elements():
    for mesh in (
        single(UNIT_CUBE, "Hexahedron 8", 5),
        single(REGULAR_TETRAHEDRON, "Tetrahedron 4", 4),
        single(REGULAR_TETRAHEDRON[:3], "Triangle 3", 2),
        single(UNIT_CUBE[:4], "Quadrilateral 4", 3),
    ):
        quality = element_quality(mesh)
        assert np.allclose(quality["scaled_jacobian"], 1.0)
        assert np.allclose(quality["frobenius_aspect"], 1.0)
    assert np.allclose(element_quality(single(UNIT_CUBE[:4], "Quadrilateral 4", 3))["warp"], 0.0)
    assert "warp" not in element_quality(single(REGULAR_TETRAHEDRON, "Tetrahedron 4", 4))


def test_element_quality_of_distorted_and_inverted_elements():
    flat = single(REGULAR_TETRAHEDRON * [1, 1, 0.2], "Tetrahedron 4", 4)
    quality = element_quality(flat)
    assert 0 < quality["scaled_jacobian"][0] < 0.5
    # The map from the regular tetrahedron is diag(1, 1, 0.2).
    assert np.isclose(quality["frobenius_aspect"][0], 2.04 / (3 * 0.2 ** (2 / 3)))
    inverted = element_quality(single(REGULAR_TETRAHEDRON, "Tetrahedron 4", 4, [0, 2, 1, 3]))
    assert np.isclose(inverted["scaled_jacobian"][0], -1.0)
    assert inverted["frobenius_aspect"][0] == np.inf
    right = element_quality(single([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "Triangle 3", 2))
    assert np.isclose(right["scaled_jacobian"][0], np.sqrt(2) / np.sqrt(3))
    assert np.isclose(right["frobenius_aspect"][0], 2 / np.sqrt(3))
    lifted = element_quality(single(UNIT_CUBE[:4] + [[0, 0, 0], [0, 0, 0], [0, 0, 0.3], [0, 0, 0]], "Quadrilateral 4", 3))
    assert lifted["warp"][0] > 10
    with pytest.raises(ValueError, match="Prism 6"):
        element_quality(single(UNIT_CUBE[:6], "Prism 6", 6))


def straight_tetrahedron_10(corners):
    """A 10-node tetrahedron with its middle nodes on its straight edges' midpoints."""
    with gmsh_session():
        local = np.reshape(gmsh.model.mesh.getElementProperties(11)[4], (-1, 3))
    corners = np.asarray(corners, dtype=float)
    return corners[0] + local @ (corners[1:] - corners[0])


def test_element_quality_measures_straight_high_order_tetrahedra_as_linear_ones():
    quality = element_quality(single(straight_tetrahedron_10(REGULAR_TETRAHEDRON), "Tetrahedron 10", 11))
    assert np.allclose(quality["scaled_jacobian"], 1.0)
    assert np.allclose(quality["frobenius_aspect"], 1.0)


def test_element_quality_measures_curved_tetrahedra_at_their_gauss_points():
    nodes = straight_tetrahedron_10(REGULAR_TETRAHEDRON)
    # Node 4, the middle of the edge from corner 0 to corner 1, pulled across the element inverts it near that edge.
    nodes[4] += [0.0, 1.2, 0.0]
    mesh = single(nodes, "Tetrahedron 10", 11)
    assert (gauss_determinants(mesh) <= 0).any()

    quality = element_quality(mesh)

    assert quality["scaled_jacobian"][0] < 0
    assert quality["frobenius_aspect"][0] == np.inf
    # Corner metrics alone see the regular tetrahedron.
    assert quality["edge_ratio"][0] == pytest.approx(1.0)
    assert quality["min_dihedral_angle"][0] == pytest.approx(np.degrees(np.arccos(1 / 3)))
    # Bent less, it stays valid but measures lower at its Gauss points than at its corners.
    nodes[4] = straight_tetrahedron_10(REGULAR_TETRAHEDRON)[4] + [0.0, 0.2, 0.0]
    assert 0 < element_quality(mesh)["scaled_jacobian"][0] < 0.95


def test_edge_ratio_and_dihedral_angle_of_tetrahedra():
    regular = element_quality(single(REGULAR_TETRAHEDRON, "Tetrahedron 4", 4))
    assert regular["edge_ratio"][0] == pytest.approx(1.0)
    assert regular["min_dihedral_angle"][0] == pytest.approx(np.degrees(np.arccos(1 / 3)))

    # A sliver: four nearly coplanar corners of a square, its edges about equal, so only the dihedral angle sees it.
    sliver = element_quality(single([[0, 0, 0], [1, 0, 0], [1, 1, 0.05], [0, 1, 0.05]], "Tetrahedron 4", 4))
    assert sliver["edge_ratio"][0] < 1.5
    assert sliver["min_dihedral_angle"][0] < 5

    # The right-angled corner tetrahedron: edges 1 and sqrt(2), three right dihedral angles and three of arccos(1 / sqrt(3)).
    corner = element_quality(single([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], "Tetrahedron 4", 4))
    assert corner["edge_ratio"][0] == pytest.approx(np.sqrt(2))
    assert corner["min_dihedral_angle"][0] == pytest.approx(np.degrees(np.arccos(1 / np.sqrt(3))))

    # Mirrored, it measures the same; its scaled Jacobian tells it is inverted.
    inverted = element_quality(single(REGULAR_TETRAHEDRON, "Tetrahedron 4", 4, [0, 2, 1, 3]))
    assert inverted["min_dihedral_angle"][0] == pytest.approx(regular["min_dihedral_angle"][0])


def test_edge_ratio_applies_to_every_shape_and_the_dihedral_angle_to_tetrahedra_alone():
    box = element_quality(single(UNIT_CUBE * [3.0, 1.0, 1.0], "Hexahedron 8", 5))
    assert box["edge_ratio"][0] == pytest.approx(3.0)
    assert "min_dihedral_angle" not in box
    assert element_quality(single([[0, 0, 0], [2, 0, 0], [0, 1, 0]], "Triangle 3", 2))["edge_ratio"][0] == pytest.approx(np.sqrt(5))
    assert element_quality(single(UNIT_CUBE[:4] * [1.0, 0.5, 1.0], "Quadrilateral 4", 3))["edge_ratio"][0] == pytest.approx(2.0)

    # A mesh of both takes the dihedral angle for its tetrahedra, NaN for its hexahedra.
    nodes = np.vstack([UNIT_CUBE, REGULAR_TETRAHEDRON + [2.0, 0.0, 0.0]])
    mixed = Mesh(nodes, {"Hexahedron 8": np.arange(8)[None], "Tetrahedron 4": np.arange(8, 12)[None]}, {"Hexahedron 8": 5, "Tetrahedron 4": 4})
    quality = element_quality(mixed)
    assert np.isnan(quality["min_dihedral_angle"][0])
    assert quality["min_dihedral_angle"][1] == pytest.approx(np.degrees(np.arccos(1 / 3)))
    assert np.allclose(quality["edge_ratio"], 1.0)


def test_element_quality_of_collapsed_edges_is_infinite_edge_ratio():
    collapsed = REGULAR_TETRAHEDRON.copy()
    collapsed[1] = collapsed[0]
    assert element_quality(single(collapsed, "Tetrahedron 4", 4))["edge_ratio"][0] == np.inf


def test_element_quality_numbers_faces_after_elements():
    mesh = cube_grid(2).with_face_sets()
    quality = element_quality(mesh)
    assert all(len(values) == mesh.element_count for values in quality.values())
    assert np.allclose(quality["scaled_jacobian"][:8], 1.0)
    assert np.isnan(quality["scaled_jacobian"][8:]).all()
    assert np.isnan(quality["warp"][:8]).all()
    assert np.allclose(quality["warp"][8:], 0.0)
    assert np.allclose(quality["scaled_jacobian"][:8], scaled_jacobians(mesh))
