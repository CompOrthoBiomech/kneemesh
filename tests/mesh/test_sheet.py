import gmsh
import numpy as np
import pytest

from kneemesh.brep import gmsh_session
from kneemesh.mesh.sheet import hexahedral_mesh
from kneemesh.mesh_optimization import mesh_quality, optimize_hexahedral_mesh, scaled_jacobians
from kneemesh.tet_meshing import import_solid, tetrahedral_mesh


def element_count(around, radial, through):
    # Four outer blocks of around by radial, and the inner block of around by around, each through deep.
    return (4 * radial + around) * around * through


def brep_volume(brep):
    with gmsh_session():
        import_solid(brep)
        return sum(gmsh.model.occ.getMass(3, volume) for _, volume in gmsh.model.getEntities(3))


# A hexahedron as six tetrahedra around its 0-6 diagonal.
HEXAHEDRON_TETRAHEDRA = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def hexahedra_volume(mesh):
    corners = mesh.nodes[mesh.elements["Hexahedron 8"]]
    total = 0.0
    for tetrahedron in HEXAHEDRON_TETRAHEDRA:
        a, b, c, d = (corners[:, index] for index in tetrahedron)
        total += float(np.sum(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)))) / 6
    return total


def test_hexahedral_mesh_is_structured_blocks(cartilage_hexahedra):
    assert cartilage_hexahedra.element_types == {"Hexahedron 8": 5}
    assert len(cartilage_hexahedra.elements["Hexahedron 8"]) == element_count(8, 4, 3)


def test_hexahedral_mesh_is_valid(cartilage_hexahedra):
    assert scaled_jacobians(cartilage_hexahedra).min() > 0.3


def test_hexahedral_mesh_fills_the_solid(cartilage_hexahedra, cartilage_hex_blocks):
    assert mesh_quality(cartilage_hexahedra).mean > 0.8
    # Only 32 linear faces span the perimeter, so their chords cut off about 2.3%.
    assert hexahedra_volume(cartilage_hexahedra) == pytest.approx(brep_volume(cartilage_hex_blocks), rel=0.04)


def test_tetrahedral_mesh_of_the_solid(cartilage_solid):
    mesh = tetrahedral_mesh(cartilage_solid)

    assert list(mesh.elements) == ["Tetrahedron 4"]


@pytest.mark.parametrize("counts", [(4, 2, 1), (12, 6, 2), (6, 3, 5)])
def test_hexahedral_mesh_follows_each_count(cartilage_hex_blocks, counts):
    mesh = hexahedral_mesh(cartilage_hex_blocks, *counts)

    assert len(mesh.elements["Hexahedron 8"]) == element_count(*counts)
    assert scaled_jacobians(mesh).min() > 0


def test_hexahedral_mesh_second_order(cartilage_hex_blocks):
    complete = hexahedral_mesh(cartilage_hex_blocks, 4, 2, 1, element_order=2)
    incomplete = hexahedral_mesh(cartilage_hex_blocks, 4, 2, 1, element_order=2, second_order_incomplete=True)

    assert complete.elements["Hexahedron 27"].shape == (element_count(4, 2, 1), 27)
    assert incomplete.elements["Hexahedron 20"].shape == (element_count(4, 2, 1), 20)


def test_hexahedral_mesh_rejects_a_single_block_solid(cartilage_solid):
    with pytest.raises(ValueError, match="five-block butterfly solid"):
        hexahedral_mesh(cartilage_solid)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [({"elements_around": 0}, "at least 1"), ({"elements_through": 0}, "at least 1"), ({"element_order": 0}, "element_order")],
)
def test_hexahedral_mesh_rejects_invalid_arguments(cartilage_hex_blocks, arguments, message):
    with pytest.raises(ValueError, match=message):
        hexahedral_mesh(cartilage_hex_blocks, **arguments)


@pytest.mark.parametrize("structure", ["lateral_tibial_cartilage", "medial_tibial_cartilage", "patellar_cartilage"])
def test_real_cartilage_hexahedra_are_valid_and_optimize(real_cartilage_hex_blocks, structure):
    mesh = hexahedral_mesh(real_cartilage_hex_blocks[structure])

    before = mesh_quality(mesh)
    after = mesh_quality(optimize_hexahedral_mesh(mesh, aspect_weight=0.5))

    # Without perimeter rounding, the real lateral tibial cartilage's notch inverts elements at these counts.
    assert before.minimum > 0.1
    assert after.minimum > 0.35
    assert after.mean > before.mean
    assert after.max_aspect < min(before.max_aspect, 6.0)


def test_scaled_jacobian_alone_stretches_real_cartilage_elements(real_cartilage_hex_blocks):
    mesh = hexahedral_mesh(real_cartilage_hex_blocks["medial_tibial_cartilage"])

    alone = mesh_quality(optimize_hexahedral_mesh(mesh))
    combined = mesh_quality(optimize_hexahedral_mesh(mesh, aspect_weight=0.5))

    assert alone.max_aspect > 3 * combined.max_aspect
