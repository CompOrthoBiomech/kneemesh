import gmsh
import numpy as np
import pytest

from meniscus2brep.mesh import Algorithm2D, Algorithm3D, Mesh, hexahedral_mesh, tetrahedral_mesh

# Neighbors of each corner of a gmsh hexahedron, ordered so the corner's
# Jacobian determinant is positive for a valid element.
HEXAHEDRON_CORNER_NEIGHBORS = [(1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7), (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3)]
# A hexahedron as six tetrahedra around its 0-6 diagonal.
HEXAHEDRON_TETRAHEDRA = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def tetrahedron_volumes(nodes, corners):
    a, b, c, d = (nodes[corners[:, index]] for index in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6


def hexahedron_volumes(mesh, name) -> np.ndarray:
    corners = mesh.elements[name][:, :8]
    volumes = sum(tetrahedron_volumes(mesh.nodes, corners[:, list(tetrahedron)]) for tetrahedron in HEXAHEDRON_TETRAHEDRA)

    return np.array([volumes]).ravel()


def scaled_corner_jacobians(mesh, name):
    corners = mesh.nodes[mesh.elements[name][:, :8]]
    values = []
    for corner, neighbors in enumerate(HEXAHEDRON_CORNER_NEIGHBORS):
        edges = [corners[:, neighbor] - corners[:, corner] for neighbor in neighbors]
        lengths = np.prod([np.linalg.norm(edge, axis=1) for edge in edges], axis=0)
        values.append(np.einsum("ij,ij->i", edges[0], np.cross(edges[1], edges[2])) / lengths)
    return np.array(values)


def brep_volume(brep, tmp_path):
    from meniscus2brep.disk import write_brep
    from meniscus2brep.geo import gmsh_session

    path = tmp_path / "shape.brep"
    write_brep(brep, path)
    with gmsh_session():
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        return sum(gmsh.model.occ.getMass(3, volume) for _, volume in gmsh.model.getEntities(3))


@pytest.fixture(scope="module")
def meniscus_tetrahedra(meniscus_solid):
    return tetrahedral_mesh(meniscus_solid, max_size=2.0)


@pytest.fixture(scope="module")
def meniscus_hexahedra(meniscus_hex_blocks):
    return hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=4, elements_along=30)


def test_tetrahedral_mesh_is_linear_tetrahedra_by_default(meniscus_tetrahedra):
    assert isinstance(meniscus_tetrahedra, Mesh)
    assert list(meniscus_tetrahedra.elements) == ["Tetrahedron 4"]
    assert meniscus_tetrahedra.nodes.shape[1] == 3
    assert meniscus_tetrahedra.elements["Tetrahedron 4"].max() < len(meniscus_tetrahedra.nodes)


def test_tetrahedral_mesh_fills_the_solid(meniscus_tetrahedra, meniscus_solid, tmp_path):
    volumes = tetrahedron_volumes(meniscus_tetrahedra.nodes, meniscus_tetrahedra.elements["Tetrahedron 4"])

    assert np.all(volumes > 0)
    # Flat-faced tetrahedra cut inside the curved surface, losing a few percent at this size.
    assert volumes.sum() == pytest.approx(brep_volume(meniscus_solid, tmp_path), rel=0.05)


def median_edge_length(mesh):
    corners = mesh.nodes[mesh.elements["Tetrahedron 4"]]
    return np.median(np.concatenate([np.linalg.norm(corners[:, a] - corners[:, b], axis=1) for a in range(4) for b in range(a + 1, 4)]))


@pytest.mark.parametrize("size", [1.5, 3.0])
def test_tetrahedral_mesh_follows_max_size(meniscus_solid, size):
    # gmsh treats sizes as targets for edge length rather than hard limits.
    assert median_edge_length(tetrahedral_mesh(meniscus_solid, max_size=size)) == pytest.approx(size, rel=0.25)


def test_tetrahedral_mesh_min_size_limits_curvature_refinement(meniscus_solid):
    fine = tetrahedral_mesh(meniscus_solid, max_size=4.0, elements_per_circle=12)
    limited = tetrahedral_mesh(meniscus_solid, min_size=2.0, max_size=4.0, elements_per_circle=12)

    assert len(limited.elements["Tetrahedron 4"]) < len(fine.elements["Tetrahedron 4"])
    assert median_edge_length(limited) > median_edge_length(fine)


def test_curvature_sizing_refines_the_mesh(meniscus_solid):
    def count(elements_per_circle):
        return len(tetrahedral_mesh(meniscus_solid, max_size=4.0, elements_per_circle=elements_per_circle).elements["Tetrahedron 4"])

    assert count(0) < count(6) < count(12)


def test_tetrahedral_mesh_second_order_has_10_node_tetrahedra(meniscus_solid):
    mesh = tetrahedral_mesh(meniscus_solid, max_size=3.0, element_order=2)

    assert list(mesh.elements) == ["Tetrahedron 10"]
    assert mesh.elements["Tetrahedron 10"].shape[1] == 10


def test_algorithms_carry_gmsh_option_numbers():
    assert [algorithm.number for algorithm in Algorithm2D] == [1, 2, 5, 6]
    assert [algorithm.number for algorithm in Algorithm3D] == [1, 4, 7, 10]
    assert Algorithm3D("MMG3D") is Algorithm3D.MMG3D


@pytest.mark.parametrize("algorithm_2d", list(Algorithm2D))
def test_every_2d_algorithm_meshes_the_solid(meniscus_solid, algorithm_2d):
    mesh = tetrahedral_mesh(meniscus_solid, max_size=3.0, algorithm_2d=algorithm_2d)

    assert len(mesh.elements["Tetrahedron 4"]) > 0


@pytest.mark.parametrize("algorithm_3d", list(Algorithm3D))
def test_every_3d_algorithm_meshes_the_solid(meniscus_solid, algorithm_3d):
    mesh = tetrahedral_mesh(meniscus_solid, max_size=3.0, algorithm_3d=algorithm_3d)

    assert len(mesh.elements["Tetrahedron 4"]) > 0


def test_tetrahedral_mesh_restores_options_of_running_gmsh(meniscus_solid):
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        before = gmsh.option.getNumber("Mesh.MeshSizeMax")
        tetrahedral_mesh(meniscus_solid, max_size=2.0, element_order=2)
        assert gmsh.option.getNumber("Mesh.MeshSizeMax") == before
        assert gmsh.option.getNumber("Mesh.ElementOrder") == 1
    finally:
        gmsh.finalize()


def test_tetrahedral_mesh_rejects_open_surface(meniscus_surface):
    with pytest.raises(ValueError, match="solid"):
        tetrahedral_mesh(meniscus_surface)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"min_size": -1.0}, "min_size"),
        ({"min_size": 2.0, "max_size": 1.0}, "max_size"),
        ({"elements_per_circle": -1}, "elements_per_circle"),
        ({"element_order": 0}, "element_order"),
    ],
)
def test_tetrahedral_mesh_rejects_invalid_arguments(meniscus_solid, arguments, message):
    with pytest.raises(ValueError, match=message):
        tetrahedral_mesh(meniscus_solid, **arguments)


def test_hexahedral_mesh_is_structured_blocks(meniscus_hexahedra):
    # Three blocks, each 4 x 4 elements per cap swept 30 times.
    assert list(meniscus_hexahedra.elements) == ["Hexahedron 8"]
    assert len(meniscus_hexahedra.elements["Hexahedron 8"]) == 3 * 4 * 4 * 30


def test_hexahedral_mesh_has_no_inverted_elements(meniscus_hexahedra):
    assert scaled_corner_jacobians(meniscus_hexahedra, "Hexahedron 8").min() > 0.05


def test_hexahedral_mesh_fills_the_solid(meniscus_hexahedra, meniscus_hex_blocks, tmp_path):
    volumes = hexahedron_volumes(meniscus_hexahedra, "Hexahedron 8")

    assert volumes.sum() == pytest.approx(brep_volume(meniscus_hex_blocks, tmp_path), rel=0.03)


@pytest.mark.parametrize(("per_edge", "along"), [(2, 15), (8, 60)])
def test_hexahedral_mesh_is_valid_at_other_resolutions(meniscus_hex_blocks, per_edge, along):
    mesh = hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=per_edge, elements_along=along)

    assert len(mesh.elements["Hexahedron 8"]) == 3 * per_edge * per_edge * along
    assert scaled_corner_jacobians(mesh, "Hexahedron 8").min() > 0


def test_hexahedral_mesh_second_order_has_27_node_hexahedra(meniscus_hex_blocks):
    mesh = hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=2, elements_along=10, element_order=2)

    assert list(mesh.elements) == ["Hexahedron 27"]
    assert mesh.elements["Hexahedron 27"].shape == (3 * 2 * 2 * 10, 27)


def test_hexahedral_mesh_second_order_incomplete_has_20_node_hexahedra(meniscus_hex_blocks):
    mesh = hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=2, elements_along=10, element_order=2, second_order_incomplete=True)

    assert list(mesh.elements) == ["Hexahedron 20"]
    assert mesh.elements["Hexahedron 20"].shape == (3 * 2 * 2 * 10, 20)


@pytest.mark.parametrize(
    ("order", "incomplete", "name"), [(2, True, "Tetrahedron 10"), (3, False, "Tetrahedron 20"), (3, True, "Tetrahedron 16")]
)
def test_tetrahedral_mesh_second_order_incomplete_drops_interior_nodes(meniscus_solid, order, incomplete, name):
    mesh = tetrahedral_mesh(meniscus_solid, max_size=4.0, element_order=order, second_order_incomplete=incomplete)

    assert list(mesh.elements) == [name]


def test_mesh_records_gmsh_element_types(meniscus_tetrahedra, meniscus_hexahedra):
    assert meniscus_tetrahedra.element_types == {"Tetrahedron 4": 4}
    assert meniscus_hexahedra.element_types == {"Hexahedron 8": 5}


def test_hexahedral_mesh_rejects_single_block_solid(meniscus_solid):
    with pytest.raises(ValueError, match="six-sided blocks"):
        hexahedral_mesh(meniscus_solid)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [({"elements_per_edge": 0}, "at least 1"), ({"elements_along": 0}, "at least 1"), ({"element_order": 0}, "element_order")],
)
def test_hexahedral_mesh_rejects_invalid_arguments(meniscus_hex_blocks, arguments, message):
    with pytest.raises(ValueError, match=message):
        hexahedral_mesh(meniscus_hex_blocks, **arguments)
