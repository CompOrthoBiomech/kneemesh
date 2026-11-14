import gmsh
import numpy as np
import pytest

from kneemesh.brep import gmsh_session
from kneemesh.mesh_sets import (
    FACE_ELEMENTS,
    _face_nodes,
    boundary_faces,
    face_groups_of,
    face_sets,
    surface_sets,
)
from kneemesh.tet_meshing import Mesh

GMSH_TYPES = {"Tetrahedron 4": 4, "Tetrahedron 10": 11, "Hexahedron 8": 5, "Hexahedron 20": 17, "Hexahedron 27": 12}

# Two unit cubes sharing the face x = 1.
CUBES = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1, 2)], dtype=float)
HEXAHEDRA = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])


def closed(faces):
    """Whether every edge of the corner faces is shared by exactly two faces, once each way."""
    edges = np.concatenate([np.stack([rows, np.roll(rows, -1, axis=1)], axis=2).reshape(-1, 2) for rows in faces])
    directed = {tuple(edge) for edge in edges.tolist()}
    return len(directed) == len(edges) and all((b, a) in directed for a, b in directed)


@pytest.mark.parametrize("name", list(FACE_ELEMENTS))
def test_face_nodes_follow_gmsh_reference_elements(name):
    with gmsh_session():
        properties = gmsh.model.mesh.getElementProperties(GMSH_TYPES[name])
    local = np.reshape(properties[4], (-1, 3))
    corners = properties[5]
    centroid = local[:corners].mean(axis=0)

    for face in _face_nodes(name):
        count = 3 if name.startswith("Tetrahedron") else 4
        points = local[face[:count]]
        # Outward, with each middle node halfway along its edge and the center node at the corners' mean.
        assert np.cross(points[1] - points[0], points[-1] - points[0]) @ (points.mean(axis=0) - centroid) > 0
        for index, node in enumerate(face[count : 2 * count]):
            assert np.allclose(local[node], (points[index] + points[(index + 1) % count]) / 2)
        if len(face) == 9:
            assert np.allclose(local[face[8]], points.mean(axis=0))


def test_boundary_faces_leave_out_shared_faces():
    faces = boundary_faces({"Hexahedron 8": HEXAHEDRA})

    quadrilaterals = faces["Quadrilateral 4"]
    assert list(faces) == ["Quadrilateral 4"]
    assert len(quadrilaterals) == 10
    assert not any(set(face) == {1, 4, 7, 10} for face in quadrilaterals.tolist())
    assert closed([quadrilaterals])


def test_boundary_faces_point_out():
    tetrahedra = np.array([[0, 1, 2, 3]])
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

    triangles = boundary_faces({"Tetrahedron 4": tetrahedra})["Triangle 3"]

    normals = np.cross(nodes[triangles[:, 1]] - nodes[triangles[:, 0]], nodes[triangles[:, 2]] - nodes[triangles[:, 0]])
    assert len(triangles) == 4
    assert np.all(np.einsum("ij,ij->i", normals, nodes[triangles].mean(axis=1) - nodes.mean(axis=0)) > 0)


def test_boundary_faces_reject_surface_elements():
    with pytest.raises(ValueError, match="No boundary faces"):
        boundary_faces({"Triangle 3": np.array([[0, 1, 2]])})


def test_face_groups_match_faces_to_surfaces_by_corners():
    faces = boundary_faces({"Hexahedron 8": HEXAHEDRA})
    end = np.array([[2, 5, 11, 8]])

    on = face_groups_of(faces, {7: end[:, ::-1]})

    assert np.count_nonzero(on == 7) == 1
    assert set(faces["Quadrilateral 4"][on == 7][0]) == {2, 5, 8, 11}
    assert np.count_nonzero(on == -1) == 9


def test_face_sets_number_faces_after_elements():
    faces = boundary_faces({"Hexahedron 8": HEXAHEDRA})
    ends = np.array([set(face) in ({0, 3, 6, 9}, {2, 5, 8, 11}) for face in faces["Quadrilateral 4"].tolist()])

    element_sets, node_sets = face_sets({"Hexahedron 8": HEXAHEDRA}, faces, {"ends": ends})

    assert list(element_sets) == ["surface", "ends"]
    assert element_sets["surface"].tolist() == list(range(2, 12))
    assert element_sets["ends"].tolist() == (2 + np.flatnonzero(ends)).tolist()
    assert node_sets["surface"].tolist() == list(range(12))
    assert node_sets["ends"].tolist() == [0, 2, 3, 5, 6, 8, 9, 11]


def test_surface_sets_hold_a_surface_mesh():
    element_sets, node_sets = surface_sets({"Triangle 3": np.array([[0, 1, 2], [0, 2, 4]])})

    assert element_sets["surface"].tolist() == [0, 1]
    assert node_sets["surface"].tolist() == [0, 1, 2, 4]


def test_mesh_with_face_sets_adds_faces_and_their_types():
    mesh = Mesh(CUBES, {"Hexahedron 8": HEXAHEDRA}, {"Hexahedron 8": 5}).with_face_sets()

    assert mesh.element_types == {"Hexahedron 8": 5, "Quadrilateral 4": 3}
    assert len(mesh.faces["Quadrilateral 4"]) == 10
    assert mesh.element_count == 12
    assert list(mesh.element_sets) == list(mesh.node_sets) == ["surface"]
