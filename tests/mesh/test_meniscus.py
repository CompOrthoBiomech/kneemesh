
import gmsh
import numpy as np
import pytest

from kneemesh.disk import write_brep
from kneemesh.mesh.meniscus import hexahedral_mesh
from kneemesh.tet_meshing import tetrahedral_mesh
from kneemesh.vol import extract_isocontour, signed_distance_field, smooth_isocontour

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
    from kneemesh.brep import gmsh_session

    path = tmp_path / "shape.brep"
    write_brep(brep, path)
    with gmsh_session():
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        return sum(gmsh.model.occ.getMass(3, volume) for _, volume in gmsh.model.getEntities(3))


def test_hexahedral_mesh_is_structured_blocks(meniscus_hexahedra):
    # Five blocks, each 4 x 4 elements per cap swept 30 times.
    assert list(meniscus_hexahedra.elements) == ["Hexahedron 8"]
    assert len(meniscus_hexahedra.elements["Hexahedron 8"]) == 5 * 4 * 4 * 30


def test_hexahedral_mesh_has_no_inverted_elements(meniscus_hexahedra):
    assert scaled_corner_jacobians(meniscus_hexahedra, "Hexahedron 8").min() > 0.05


def test_hexahedral_mesh_fills_the_solid(meniscus_hexahedra, meniscus_hex_blocks, tmp_path):
    volumes = hexahedron_volumes(meniscus_hexahedra, "Hexahedron 8")

    # Only 16 linear faces span each section's curved boundary, so the chords cut off about 3.5%.
    assert volumes.sum() == pytest.approx(brep_volume(meniscus_hex_blocks, tmp_path), rel=0.05)


@pytest.mark.parametrize(("per_edge", "along"), [(2, 15), (8, 60)])
def test_hexahedral_mesh_is_valid_at_other_resolutions(meniscus_hex_blocks, per_edge, along):
    mesh = hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=per_edge, elements_along=along)

    assert len(mesh.elements["Hexahedron 8"]) == 5 * per_edge * per_edge * along
    assert scaled_corner_jacobians(mesh, "Hexahedron 8").min() > 0


def test_hexahedral_mesh_second_order_has_27_node_hexahedra(meniscus_hex_blocks):
    mesh = hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=2, elements_along=10, element_order=2)

    assert list(mesh.elements) == ["Hexahedron 27"]
    assert mesh.elements["Hexahedron 27"].shape == (5 * 2 * 2 * 10, 27)


def test_hexahedral_mesh_second_order_incomplete_has_20_node_hexahedra(meniscus_hex_blocks):
    mesh = hexahedral_mesh(meniscus_hex_blocks, elements_per_edge=2, elements_along=10, element_order=2, second_order_incomplete=True)

    assert list(mesh.elements) == ["Hexahedron 20"]
    assert mesh.elements["Hexahedron 20"].shape == (5 * 2 * 2 * 10, 20)


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


def test_tetrahedral_mesh_of_real_meniscus(real_label, tmp_path):
    from kneemesh.geo.meniscus import centerline, contours_to_brep, slice_isocontour

    slices = slice_isocontour(extract_isocontour(signed_distance_field(real_label)), centerline(real_label), spacing=5.0)
    brep = contours_to_brep(slices, cap="planar")

    mesh = tetrahedral_mesh(brep, max_size=1.0)

    volumes = tetrahedron_volumes(mesh.nodes, mesh.elements["Tetrahedron 4"])
    assert np.all(volumes > 0)
    assert volumes.sum() == pytest.approx(brep_volume(brep, tmp_path), rel=0.03)


def test_hexahedral_mesh_of_real_meniscus_has_no_inverted_elements(real_label):
    from kneemesh.geo.meniscus import centerline, contours_to_hex_blocks, slice_isocontour

    surface = smooth_isocontour(extract_isocontour(signed_distance_field(real_label)))
    slices = slice_isocontour(surface, centerline(real_label), spacing=5.0, points_per_slice=10)

    mesh = hexahedral_mesh(contours_to_hex_blocks(slices), elements_per_edge=4, elements_along=20)

    assert scaled_corner_jacobians(mesh, "Hexahedron 8").min() > 0.1
