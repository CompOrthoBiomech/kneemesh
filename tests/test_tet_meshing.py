import gmsh
import numpy as np
import pytest

from kneemesh import tet_meshing
from kneemesh.brep import gmsh_session
from kneemesh.disk import write_brep
from kneemesh.geo.meniscus import centerline, contours_to_brep, slice_isocontour
from kneemesh.segmentation import assemble_label_maps
from kneemesh.tet_meshing import (
    Algorithm2D,
    Algorithm3D,
    Mesh,
    import_solid,
    oriented_extents,
    tetrahedral_mesh,
)
from kneemesh.vol import extract_isocontour, signed_distance_field, smooth_isocontour


def tetrahedron_volumes(nodes, corners):
    a, b, c, d = (nodes[corners[:, index]] for index in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6


def brep_volume(brep, tmp_path):
    from kneemesh.brep import gmsh_session

    path = tmp_path / "shape.brep"
    write_brep(brep, path)
    with gmsh_session():
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        return sum(gmsh.model.occ.getMass(3, volume) for _, volume in gmsh.model.getEntities(3))


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


@pytest.fixture(scope="module")
def curvature_refined_tetrahedra(meniscus_solid):
    """meniscus_solid's tetrahedra at 12 elements per circle of curvature, with no min_size to stop the refinement."""
    return tetrahedral_mesh(meniscus_solid, min_size=0.0, max_size=4.0, elements_per_circle=12)


def test_tetrahedral_mesh_min_size_limits_curvature_refinement(meniscus_solid, curvature_refined_tetrahedra):
    limited = tetrahedral_mesh(meniscus_solid, min_size=2.0, max_size=4.0, elements_per_circle=12)

    assert len(limited.elements["Tetrahedron 4"]) < len(curvature_refined_tetrahedra.elements["Tetrahedron 4"])
    assert median_edge_length(limited) > median_edge_length(curvature_refined_tetrahedra)


def test_curvature_sizing_refines_the_mesh(meniscus_solid, curvature_refined_tetrahedra):
    def count(elements_per_circle):
        mesh = tetrahedral_mesh(meniscus_solid, min_size=0.0, max_size=4.0, elements_per_circle=elements_per_circle)
        return len(mesh.elements["Tetrahedron 4"])

    assert count(0) < count(6) < len(curvature_refined_tetrahedra.elements["Tetrahedron 4"])


def test_tetrahedral_mesh_second_order_has_10_node_tetrahedra(meniscus_solid):
    mesh = tetrahedral_mesh(meniscus_solid, max_size=3.0, element_order=2)

    assert list(mesh.elements) == ["Tetrahedron 10"]
    assert mesh.elements["Tetrahedron 10"].shape[1] == 10


def gauss_determinants(mesh):
    """Jacobian determinants of every second-order tetrahedron at the points of gmsh's Gauss rules of orders 2 and 4."""
    connectivity = mesh.elements["Tetrahedron 10"]
    determinants = []
    with gmsh_session():
        for rule in ("Gauss2", "Gauss4"):
            points, _ = gmsh.model.mesh.getIntegrationPoints(11, rule)
            _, gradients, _ = gmsh.model.mesh.getBasisFunctions(11, points, "GradLagrange")
            gradients = np.reshape(gradients, (len(points) // 3, 10, 3))
            determinants.append(np.linalg.det(np.einsum("gni,enx->egix", gradients, mesh.nodes[connectivity])))
    return np.concatenate(determinants, axis=1)


def test_second_order_tetrahedra_of_the_real_meniscus_are_valid_at_gauss_points(real_label, monkeypatch):
    surface = smooth_isocontour(extract_isocontour(signed_distance_field(real_label)))
    solid = contours_to_brep(slice_isocontour(surface, centerline(real_label), spacing=5.0), cap="planar")

    assert (gauss_determinants(tetrahedral_mesh(solid, element_order=2)) > 0).all()

    # Without gmsh's high-order optimization, middle nodes curved onto the surface invert thin tetrahedra beside it.
    monkeypatch.setattr(tet_meshing, "_HIGH_ORDER_OPTIMIZE", 0)
    assert (gauss_determinants(tetrahedral_mesh(solid, element_order=2)) <= 0).any(axis=1).sum() > 0


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


def test_default_min_size_limits_curvature_refinement(meniscus_solid, curvature_refined_tetrahedra):
    # 12 elements per circle halve the edges without min_size (0.44 against 0.59 mm); 24 took 30 s to mesh.
    default = tetrahedral_mesh(meniscus_solid, max_size=4.0, elements_per_circle=12)
    offsets = default.nodes - default.nodes.mean(axis=0)
    thickness = np.ptp(offsets @ np.linalg.eigh(offsets.T @ offsets)[1], axis=0).min()

    assert len(default.elements["Tetrahedron 4"]) < len(curvature_refined_tetrahedra.elements["Tetrahedron 4"])
    assert median_edge_length(default) > 0.8 * thickness / 20


def test_oriented_extents_ignore_the_shapes_orientation(meniscus_solid):
    with gmsh_session():
        import_solid(meniscus_solid)
        aligned = oriented_extents()
        volumes = gmsh.model.getEntities(3)
        gmsh.model.occ.rotate(volumes, 0, 0, 0, 1, 1, 0, 0.7)
        gmsh.model.occ.synchronize()
        rotated = oriented_extents()
        x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.getBoundingBox(-1, -1)

    # The fixture's section is 8 high, swept about the z axis; tilted, no axis-aligned side is that thin.
    assert rotated == pytest.approx(aligned, rel=1e-6)
    assert aligned[0] == pytest.approx(8.0, rel=0.1)
    assert min(x_max - x_min, y_max - y_min, z_max - z_min) > 3 * aligned[0]


def test_default_min_size_yields_to_a_smaller_max_size(meniscus_solid):
    assert median_edge_length(tetrahedral_mesh(meniscus_solid, max_size=1.0)) == pytest.approx(1.0, rel=0.25)


def test_default_max_size_is_a_quarter_of_the_thickness(meniscus_solid):
    # The fixture's section is 8 high, so elements are about 2 across.
    assert median_edge_length(tetrahedral_mesh(meniscus_solid)) == pytest.approx(2.0, rel=0.25)


def test_default_max_size_gives_a_larger_min_size_room(meniscus_solid):
    assert median_edge_length(tetrahedral_mesh(meniscus_solid, min_size=3.0)) == pytest.approx(3.0, rel=0.25)


@pytest.mark.parametrize("assembled", [True, False])
def test_default_sizes_mesh_the_thin_real_medial_meniscus(real_medial_label, assembled):
    label = assemble_label_maps({"medial_meniscus": real_medial_label}).image if assembled else real_medial_label
    surface = smooth_isocontour(extract_isocontour(signed_distance_field(label)))
    solid = contours_to_brep(slice_isocontour(surface, centerline(label), spacing=5.0), cap="planar")

    count = len(tetrahedral_mesh(solid).elements["Tetrahedron 4"])

    # gmsh's own default sizes, from the overall extent, are coarser: 200-1000 tetrahedra against about 2000, or, on
    # the medial meniscus's filled-in sections after assembly, overlapping facets. Before contours followed the
    # section's corners, gmsh also failed with them or silently made no tetrahedra.
    try:
        coarse = len(tetrahedral_mesh(solid, max_size=1e3).elements["Tetrahedron 4"])
    except ValueError:
        coarse = 0
    assert count > max(coarse, 1000)


def test_tetrahedral_mesh_rejects_open_surface(meniscus_surface):
    with pytest.raises(ValueError, match="solid"):
        tetrahedral_mesh(meniscus_surface)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"min_size": -1.0}, "min_size"),
        ({"min_size": 2.0, "max_size": 1.0}, "max_size"),
        ({"max_size": 0.0}, "max_size must be positive"),
        ({"elements_per_circle": -1}, "elements_per_circle"),
        ({"element_order": 0}, "element_order"),
    ],
)
def test_tetrahedral_mesh_rejects_invalid_arguments(meniscus_solid, arguments, message):
    with pytest.raises(ValueError, match=message):
        tetrahedral_mesh(meniscus_solid, **arguments)


@pytest.mark.parametrize(("order", "incomplete", "name"), [(2, True, "Tetrahedron 10"), (3, False, "Tetrahedron 20"), (3, True, "Tetrahedron 16")])
def test_tetrahedral_mesh_second_order_incomplete_drops_interior_nodes(meniscus_solid, order, incomplete, name):
    mesh = tetrahedral_mesh(meniscus_solid, max_size=4.0, element_order=order, second_order_incomplete=incomplete)

    assert list(mesh.elements) == [name]
