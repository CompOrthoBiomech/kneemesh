import gmsh
import numpy as np
import pytest

from kneemesh.brep import gmsh_session
from kneemesh.geo.sheet import fit_sheet, sheet_to_brep
from kneemesh.mesh.sheet import _base_complex, hexahedral_mesh, sheet_partition_brep
from kneemesh.mesh_optimization import mesh_quality, optimize_hexahedral_mesh, scaled_jacobians
from kneemesh.tet_meshing import import_solid, tetrahedral_mesh

# A hexahedron as six tetrahedra around its 0-6 diagonal.
HEXAHEDRON_TETRAHEDRA = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def max_bend(mesh):
    """The largest distance of a straight line's node from the segment between its ends, relative to its length."""
    points = mesh.nodes[mesh.straight_lines]
    first, last = points[:, :1], points[:, -1:]
    direction = (last - first) / np.linalg.norm(last - first, axis=-1, keepdims=True)
    offsets = points - first
    across = offsets - np.sum(offsets * direction, axis=-1, keepdims=True) * direction
    return float((np.linalg.norm(across, axis=-1).max(axis=1) / np.linalg.norm(last - first, axis=-1)[:, 0]).max())


def hexahedra_volume(mesh):
    corners = mesh.nodes[mesh.elements["Hexahedron 8"]]
    total = 0.0
    for tetrahedron in HEXAHEDRON_TETRAHEDRA:
        a, b, c, d = (corners[:, index] for index in tetrahedron)
        total += float(np.sum(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)))) / 6
    return total


def tetrahedra_volume(brep):
    """The solid's volume from tetrahedra; occ.getMass is unreliable with ruled walls."""
    mesh = tetrahedral_mesh(brep)
    corners = mesh.nodes[mesh.elements["Tetrahedron 4"]]
    return float(np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])).sum() / 6)


def hexahedron_count(mesh):
    return sum(len(connectivity) for connectivity in mesh.elements.values())


def block_sides(brep):
    """The number of faces of each volume and of sides of each of those faces."""
    with gmsh_session():
        import_solid(brep)
        faces = [gmsh.model.getBoundary([(3, volume)], oriented=False) for _, volume in gmsh.model.getEntities(3)]
        return [sorted(len(gmsh.model.getBoundary([face], combined=False)) for face in block) for block in faces]


def test_hexahedral_mesh_is_valid_and_fills_the_solid(cartilage_hexahedra, cartilage_hex_solid):
    assert cartilage_hexahedra.element_types == {"Hexahedron 8": 5}
    assert scaled_jacobians(cartilage_hexahedra).min() > 0.3
    assert mesh_quality(cartilage_hexahedra).mean > 0.8
    # Linear faces cut chords off the curved perimeter.
    assert hexahedra_volume(cartilage_hexahedra) == pytest.approx(tetrahedra_volume(cartilage_hex_solid), rel=0.03)


def test_hexahedral_mesh_follows_the_size_and_layers(cartilage_hex_solid, cartilage_hexahedra):
    count = hexahedron_count(cartilage_hexahedra)

    fine = hexahedral_mesh(cartilage_hex_solid, 1.0)
    thin = hexahedral_mesh(cartilage_hex_solid, 2.0, elements_through=1)

    assert 3 < hexahedron_count(fine) / count < 5
    assert hexahedron_count(thin) * 3 == count


def test_hexahedral_mesh_keeps_a_straight_column_through_the_wall_at_every_rim_node(cartilage_hexahedra):
    lines = cartilage_hexahedra.straight_lines
    # Rim nodes are on exactly one boundary face of the bottom layer, the wall, besides the bottom side itself.
    corners = cartilage_hexahedra.elements["Hexahedron 8"]
    bottom = np.unique(corners[:, :4])

    assert lines.shape[1] == 4
    assert len(np.unique(lines)) == lines.size
    assert np.isin(lines[:, 0], bottom).all() or np.isin(lines[:, -1], bottom).all()
    assert len(lines) >= 30
    assert max_bend(cartilage_hexahedra) < 1e-9


@pytest.mark.parametrize(("incomplete", "name", "nodes"), [(False, "Hexahedron 27", 27), (True, "Hexahedron 20", 20)])
def test_hexahedral_mesh_second_order(cartilage_hex_solid, incomplete, name, nodes):
    mesh = hexahedral_mesh(cartilage_hex_solid, 3.0, elements_through=1, element_order=2, second_order_incomplete=incomplete)

    connectivity = mesh.elements[name]
    assert connectivity.shape[1] == nodes
    assert scaled_jacobians(mesh).min() > 0
    # Every node is used, and each edge's middle node lies near its ends' midpoint, off it where the sheet curves.
    assert len(np.unique(connectivity)) == len(mesh.nodes)
    first, middle, last = (mesh.nodes[connectivity[:, index]] for index in (0, 8, 1))
    assert (np.linalg.norm(middle - 0.5 * (first + last), axis=1) / np.linalg.norm(last - first, axis=1)).max() < 0.15


def test_hexahedral_mesh_of_a_sheet_with_a_hole(cartilage_defect_sides):
    sheet = fit_sheet(cartilage_defect_sides, perimeter_radius=1.0, min_thickness=0.1, hole_thickness=0.05)
    brep = sheet_to_brep(sheet)

    mesh = hexahedral_mesh(brep, 1.5)
    optimized = optimize_hexahedral_mesh(mesh, aspect_weight=0.5)

    assert len(sheet.holes) == 1
    # One element beside the hole starts barely inverted, and optimization untangles it.
    assert mesh_quality(mesh).minimum > -0.05
    assert mesh_quality(optimized).inverted == 0
    assert hexahedra_volume(optimized) == pytest.approx(tetrahedra_volume(brep), rel=0.05)


@pytest.mark.parametrize("algorithm", ["quasi_structured", "frontal_delaunay"])
def test_hexahedral_mesh_with_either_quadrilateral_algorithm(cartilage_hex_solid, algorithm):
    mesh = hexahedral_mesh(cartilage_hex_solid, 2.0, quad_algorithm=algorithm)

    assert scaled_jacobians(mesh).min() > 0.2


def test_hexahedral_mesh_of_the_tetrahedral_solid(cartilage_solid):
    assert scaled_jacobians(hexahedral_mesh(cartilage_solid, 2.0)).min() > 0


def test_tetrahedral_mesh_of_the_solid(cartilage_solid):
    mesh = tetrahedral_mesh(cartilage_solid)

    assert list(mesh.elements) == ["Tetrahedron 4"]


def test_hexahedral_mesh_needs_a_sheet(meniscus_solid):
    with pytest.raises(ValueError, match="sheet's solid"):
        hexahedral_mesh(meniscus_solid, 2.0)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"element_size": 0.0}, "element_size"),
        ({"elements_through": 0}, "at least 1"),
        ({"element_order": 0}, "element_order"),
        ({"element_order": 3}, "linear or quadratic"),
    ],
)
def test_hexahedral_mesh_rejects_invalid_arguments(cartilage_hex_solid, arguments, message):
    with pytest.raises(ValueError, match=message):
        hexahedral_mesh(cartilage_hex_solid, **({"element_size": 2.0} | arguments))


@pytest.mark.parametrize("structure", ["lateral_tibial_cartilage", "medial_tibial_cartilage", "patellar_cartilage"])
def test_real_cartilage_hexahedra_are_valid_and_optimize(real_cartilage_hex_solids, structure):
    mesh = hexahedral_mesh(real_cartilage_hex_solids[structure], 2.0)

    before = mesh_quality(mesh)
    optimized = optimize_hexahedral_mesh(mesh, aspect_weight=0.5)
    after = mesh_quality(optimized)

    assert before.minimum > 0.1
    assert after.minimum > 0.4
    assert after.mean > before.mean
    assert after.max_aspect < min(before.max_aspect, 4.0)
    assert max_bend(optimized) < 1e-9


def test_scaled_jacobian_alone_stretches_real_cartilage_elements(real_cartilage_hex_solids):
    mesh = hexahedral_mesh(real_cartilage_hex_solids["patellar_cartilage"], 2.0)

    alone = mesh_quality(optimize_hexahedral_mesh(mesh))
    combined = mesh_quality(optimize_hexahedral_mesh(mesh, aspect_weight=0.5))

    # It squares up corners by stretching elements along the rim: 8 against 2.3.
    assert alone.max_aspect > 2 * combined.max_aspect


def test_femoral_hexahedra_are_valid(femoral_hexahedra):
    assert femoral_hexahedra.element_types == {"Hexahedron 8": 5}
    assert scaled_jacobians(femoral_hexahedra).min() > 0.2
    assert max_bend(femoral_hexahedra) < 1e-9


def test_real_femoral_hexahedra_are_valid_and_optimize(real_femoral_hex_solid):
    mesh = hexahedral_mesh(real_femoral_hex_solid, 2.0)

    before = mesh_quality(mesh)
    after = mesh_quality(optimize_hexahedral_mesh(mesh, aspect_weight=0.5))

    assert before.minimum > 0.2
    assert after.minimum > 0.4
    assert after.mean > 0.9
    assert after.max_aspect < min(before.max_aspect, 4.0)


def test_base_complex_of_a_regular_grid_is_one_patch():
    # A 3 by 2 grid of quadrilaterals; every boundary node is regular but the four corners.
    nodes = np.arange(12).reshape(4, 3)
    quads = np.array([[nodes[i, j], nodes[i + 1, j], nodes[i + 1, j + 1], nodes[i, j + 1]] for i in range(3) for j in range(2)])
    rim = np.setdiff1d(nodes, nodes[1:-1, 1:-1])

    grids = _base_complex(quads, rim)

    assert len(grids) == 1
    assert sorted(grids[0].shape) == [3, 4]
    assert sorted(grids[0].ravel().tolist()) == list(range(12))


def test_base_complex_splits_at_an_irregular_node():
    # Three quadrilaterals around a central node of valence 3, as at a triangle's center.
    center, middles, corners = 0, [1, 2, 3], [4, 5, 6]
    quads = np.array([[center, middles[k], corners[k], middles[k - 1]] for k in range(3)])
    rim = np.array(middles + corners)

    grids = _base_complex(quads, rim)

    assert len(grids) == 3
    assert all(grid.shape == (2, 2) for grid in grids)


def test_sheet_partition_is_six_sided_blocks(cartilage_hex_solid):
    partition = sheet_partition_brep(cartilage_hex_solid, 3.0)

    sides = block_sides(partition)
    assert len(sides) >= 1
    assert all(block == [4] * 6 for block in sides)
