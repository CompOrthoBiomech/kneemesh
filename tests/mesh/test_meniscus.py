import math

import gmsh
import numpy as np
import pytest

from kneemesh.config import DEFAULT_MENISCUS_ASPECT_WEIGHT
from kneemesh.disk import write_brep
from kneemesh.mesh.meniscus import _mean_value_coordinates, hexahedral_mesh, meniscus_face_groups
from kneemesh.mesh_optimization import mesh_quality, optimize_hexahedral_mesh
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


def block_counts(mesh):
    """Return a butterfly mesh's elements per cap edge and along, from its horn's 5 * n * n faces."""
    horn = len(mesh.element_sets["anterior_horn"])
    per_edge = round(math.sqrt(horn / 5))
    assert 5 * per_edge**2 == horn
    total = len(mesh.elements[next(iter(mesh.elements))])
    assert total % horn == 0
    return per_edge, total // horn


def edge_lengths(mesh):
    """Return the mean length of the hexahedra's edges along the meniscus, and the median across it.

    Of each hexahedron's three groups of four parallel edges, the one closest
    to the fixture's tangent about the z axis runs along it.
    """
    corners = mesh.nodes[mesh.elements[next(iter(mesh.elements))][:, :8]]
    groups = [((0, 1), (3, 2), (4, 5), (7, 6)), ((0, 3), (1, 2), (4, 7), (5, 6)), ((0, 4), (1, 5), (2, 6), (3, 7))]
    edges = np.stack([np.stack([corners[:, end] - corners[:, start] for start, end in group], axis=1) for group in groups], axis=1)
    centers = corners.mean(axis=1)
    tangents = np.column_stack((-centers[:, 1], centers[:, 0], np.zeros(len(centers))))
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    alignment = (np.abs(np.einsum("egkd,ed->egk", edges, tangents)) / np.linalg.norm(edges, axis=-1)).mean(axis=2)
    along = np.argmax(alignment, axis=1)
    lengths = np.linalg.norm(edges, axis=-1)
    rows = np.arange(len(corners))
    across = np.concatenate([lengths[rows, (along + shift) % 3] for shift in (1, 2)])
    return float(lengths[rows, along].mean()), float(np.median(across))


def test_hexahedral_mesh_is_structured_blocks(meniscus_hexahedra):
    assert list(meniscus_hexahedra.elements) == ["Hexahedron 8"]
    # Five blocks of n by n elements per cap, swept along: n of 4 for a perimeter of about 29 at 1.8.
    assert block_counts(meniscus_hexahedra)[0] == 4


# The butterfly's boundary edges take transverse_size, but its inner block and short spokes are smaller.
@pytest.mark.parametrize(("mesh", "smallest"), [("meniscus_hexahedra", 0.3), ("swept_hexahedra", 0.6)])
def test_hexahedral_mesh_targets_the_sizes(request, mesh, smallest):
    along, across = edge_lengths(request.getfixturevalue(mesh))

    assert along == pytest.approx(2.5, rel=0.1)
    assert smallest * 1.8 < across < 1.2 * 1.8


def test_hexahedral_mesh_has_no_inverted_elements(meniscus_hexahedra):
    assert scaled_corner_jacobians(meniscus_hexahedra, "Hexahedron 8").min() > 0.05


def test_hexahedral_mesh_fills_the_solid(meniscus_hexahedra, meniscus_hex_blocks, tmp_path):
    volumes = hexahedron_volumes(meniscus_hexahedra, "Hexahedron 8")

    # Only 16 linear faces span each section's curved boundary, so the chords cut off about 3.5%.
    assert volumes.sum() == pytest.approx(brep_volume(meniscus_hex_blocks, tmp_path), rel=0.05)


@pytest.mark.parametrize(("transverse", "circumferential"), [(3.6, 5.0), (0.9, 1.25)])
def test_hexahedral_mesh_is_valid_at_other_sizes(meniscus_hex_blocks, meniscus_hexahedra, transverse, circumferential):
    mesh = hexahedral_mesh(meniscus_hex_blocks, transverse_size=transverse, circumferential_size=circumferential)

    per_edge, along = block_counts(mesh)
    base_per_edge, base_along = block_counts(meniscus_hexahedra)
    assert per_edge == pytest.approx(base_per_edge * 1.8 / transverse, abs=1)
    assert along == pytest.approx(base_along * 2.5 / circumferential, abs=1)
    assert scaled_corner_jacobians(mesh, "Hexahedron 8").min() > 0


@pytest.mark.parametrize(("incomplete", "name", "nodes"), [(False, "Hexahedron 27", 27), (True, "Hexahedron 20", 20)])
def test_hexahedral_mesh_second_order(meniscus_hex_blocks, incomplete, name, nodes):
    linear = hexahedral_mesh(meniscus_hex_blocks, transverse_size=3.6, circumferential_size=7.5)
    mesh = hexahedral_mesh(meniscus_hex_blocks, transverse_size=3.6, circumferential_size=7.5, element_order=2, second_order_incomplete=incomplete)

    assert list(mesh.elements) == [name]
    assert mesh.elements[name].shape == (len(linear.elements["Hexahedron 8"]), nodes)


def test_mesh_records_gmsh_element_types(meniscus_tetrahedra, meniscus_hexahedra):
    assert meniscus_tetrahedra.element_types == {"Tetrahedron 4": 4, "Triangle 3": 2}
    assert meniscus_hexahedra.element_types == {"Hexahedron 8": 5, "Quadrilateral 4": 3}


def test_hexahedral_mesh_rejects_smooth_caps(meniscus_domed_solid):
    with pytest.raises(ValueError, match="planar caps"):
        hexahedral_mesh(meniscus_domed_solid)


@pytest.fixture(scope="module")
def swept_hexahedra(meniscus_solid):
    """Swept hexahedra of the planar-capped meniscus_solid, about 1.8 mm across and 2.5 mm along."""
    return hexahedral_mesh(meniscus_solid, transverse_size=1.8, circumferential_size=2.5)


def test_swept_hexahedral_mesh_stacks_one_section(swept_hexahedra):
    assert swept_hexahedra.element_types == {"Hexahedron 8": 5, "Quadrilateral 4": 3}
    hexahedra = swept_hexahedra.elements["Hexahedron 8"]
    # One section's quadrilaterals per layer, each cap one section.
    section = len(swept_hexahedra.element_sets["anterior_horn"])
    assert len(swept_hexahedra.element_sets["posterior_horn"]) == section
    assert len(hexahedra) % section == 0
    assert len(swept_hexahedra.nodes) % (len(hexahedra) // section + 1) == 0
    # Frontal-Delaunay fills the section with fewer quadrilaterals than the butterfly's 5 * 4 * 4 at the same size.
    assert 8 <= section < 5 * 16


def test_swept_hexahedral_mesh_fills_the_solid(swept_hexahedra, meniscus_solid, tmp_path):
    volumes = hexahedron_volumes(swept_hexahedra, "Hexahedron 8")

    # As for the butterfly, about 16 linear faces span each section's curved boundary, and the chords cut off about 4%.
    assert volumes.sum() == pytest.approx(brep_volume(meniscus_solid, tmp_path), rel=0.05)


def test_swept_hexahedral_mesh_optimizes_to_valid_elements(swept_hexahedra):
    quality = mesh_quality(optimize_hexahedral_mesh(swept_hexahedra))

    assert quality.inverted == 0
    assert quality.minimum > 0.3


def test_swept_hexahedral_mesh_names_the_horns_and_body(swept_hexahedra):
    check_horns(swept_hexahedra)


@pytest.mark.parametrize(("incomplete", "name", "nodes"), [(False, "Hexahedron 27", 27), (True, "Hexahedron 20", 20)])
def test_swept_hexahedral_mesh_second_order(meniscus_solid, incomplete, name, nodes):
    linear = hexahedral_mesh(meniscus_solid, transverse_size=3.6, circumferential_size=7.5)
    mesh = hexahedral_mesh(meniscus_solid, transverse_size=3.6, circumferential_size=7.5, element_order=2, second_order_incomplete=incomplete)

    assert list(mesh.elements) == [name]
    assert mesh.elements[name].shape == (len(linear.elements["Hexahedron 8"]), nodes)
    # The corners are the linear mesh's nodes, and the middle nodes lie between them.
    assert np.isin(np.round(mesh.nodes[mesh.elements[name][:, :8]], 9).reshape(-1, 3), np.round(linear.nodes, 9)).all()
    check_horns(mesh)


def distances_to_brep(brep, points, tmp_path):
    """Distance from each point to its nearest face of a B-rep."""
    from kneemesh.brep import gmsh_session

    path = tmp_path / "shape.brep"
    write_brep(brep, path)
    with gmsh_session():
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        faces = [face for _, face in gmsh.model.getEntities(2)]
        closest = [np.reshape(gmsh.model.getClosestPoint(2, face, points.ravel().tolist())[0], (-1, 3)) for face in faces]
    return np.min([np.linalg.norm(points - near, axis=1) for near in closest], axis=0)


@pytest.mark.parametrize("solid", ["meniscus_solid", "meniscus_clipped_solid"])
def test_swept_hexahedra_lie_on_the_loft(request, solid, tmp_path):
    brep = request.getfixturevalue(solid)
    mesh = hexahedral_mesh(brep, transverse_size=1.8, circumferential_size=2.5)

    surface = mesh.nodes[mesh.node_sets["surface"]]
    assert distances_to_brep(brep, surface, tmp_path).max() < 1e-6


def test_swept_hexahedra_of_a_clipped_loft_have_nodes_on_the_cut_edges(meniscus_clipped_solid, meniscus_clipped_slices):
    mesh = hexahedral_mesh(meniscus_clipped_solid, transverse_size=1.8, circumferential_size=2.5)
    quality = mesh_quality(optimize_hexahedral_mesh(mesh))

    assert quality.inverted == 0
    assert quality.minimum > 0.3
    check_horns(mesh)
    # Each cap's rim has a node at both corners of the cut.
    nodes = mesh.nodes[mesh.node_sets["surface"]]
    for contour in (meniscus_clipped_slices[0], meniscus_clipped_slices[-1]):
        end = contour.straight_from
        assert end is not None
        for corner in (contour.points[0], contour.points[end]):
            assert np.linalg.norm(nodes - corner, axis=1).min() < 1e-6


def test_mean_value_coordinates_reproduce_points():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    points = np.array([[0.5, 0.5], [0.1, 0.8], [0.9, 0.05]])

    weights = _mean_value_coordinates(points, square)

    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(weights @ square, points)
    assert np.all(weights > 0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [({"transverse_size": 0.0}, "positive"), ({"circumferential_size": -1.0}, "positive"), ({"element_order": 0}, "element_order")],
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

    mesh = hexahedral_mesh(contours_to_hex_blocks(slices))

    assert scaled_corner_jacobians(mesh, "Hexahedron 8").min() > 0.1


def test_swept_hexahedral_mesh_of_real_meniscus_optimizes_well(real_medial_label):
    from kneemesh.geo.meniscus import centerline, contours_to_brep, slice_isocontour

    surface = smooth_isocontour(extract_isocontour(signed_distance_field(real_medial_label)))
    slices = slice_isocontour(surface, centerline(real_medial_label), spacing=5.0)

    mesh = hexahedral_mesh(contours_to_brep(slices, cap="planar"))
    quality = mesh_quality(optimize_hexahedral_mesh(mesh, aspect_weight=DEFAULT_MENISCUS_ASPECT_WEIGHT))

    # The section starts with a few barely inverted elements, which untangle.
    assert quality.inverted == 0
    assert quality.minimum > 0.3
    assert quality.mean > 0.88
    assert quality.max_aspect < 10


def face_centers(mesh, name):
    """The centroid of each face in an element set of a mesh with one kind of face."""
    ((kind, faces),) = mesh.faces.items()
    first = sum(len(connectivity) for connectivity in mesh.elements.values())
    return mesh.nodes[faces[mesh.element_sets[name] - first, : 3 if kind.startswith("Triangle") else 4]].mean(axis=1)


def check_horns(mesh):
    surface = set(mesh.element_sets["surface"].tolist())
    anterior, posterior, body = (set(mesh.element_sets[name].tolist()) for name in ("anterior_horn", "posterior_horn", "body"))
    assert list(mesh.element_sets) == ["surface", "anterior_horn", "posterior_horn", "body"]
    assert anterior and posterior and body
    assert anterior | posterior | body == surface
    assert not (anterior & posterior or anterior & body or posterior & body)
    # The horns are the ends: the fixture's arc spans 270 degrees about z with its gap facing -x.
    for name, sign in (("anterior_horn", -1), ("posterior_horn", 1)):
        centers = face_centers(mesh, name)
        assert np.all(sign * centers[:, 1] > 0)
        assert np.all(centers[:, 0] < 0)
        assert set(mesh.node_sets[name].tolist()) <= set(mesh.node_sets["surface"].tolist())


def test_hexahedral_mesh_names_the_horns_and_body(meniscus_hexahedra):
    check_horns(meniscus_hexahedra)
    # Each horn is a cap of five blocks of 4 by 4 faces.
    assert len(meniscus_hexahedra.element_sets["anterior_horn"]) == len(meniscus_hexahedra.element_sets["posterior_horn"]) == 5 * 16


@pytest.mark.parametrize("solid", ["meniscus_solid", "meniscus_domed_solid"])
def test_tetrahedral_mesh_names_the_horns_and_body(request, solid):
    check_horns(tetrahedral_mesh(request.getfixturevalue(solid), max_size=2.0, face_groups=meniscus_face_groups))
