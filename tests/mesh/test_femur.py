import numpy as np
import pytest

from kneemesh.mesh.femur import hexahedral_mesh
from kneemesh.mesh_optimization import mesh_quality, optimize_hexahedral_mesh, scaled_jacobians


def test_hexahedral_mesh_is_valid(femoral_hexahedra):
    assert femoral_hexahedra.element_types == {"Hexahedron 8": 5}
    assert scaled_jacobians(femoral_hexahedra).min() > 0.2


def test_smaller_elements_make_more_of_them(femoral_hex_blocks, femoral_hexahedra):
    finer = hexahedral_mesh(femoral_hex_blocks, element_size=1.5)
    thinner = hexahedral_mesh(femoral_hex_blocks, elements_through=1)

    count = len(femoral_hexahedra.elements["Hexahedron 8"])
    assert len(finer.elements["Hexahedron 8"]) > 2 * count
    assert len(thinner.elements["Hexahedron 8"]) * 3 == count


def test_hexahedral_mesh_second_order(femoral_hex_blocks):
    mesh = hexahedral_mesh(femoral_hex_blocks, element_size=4.0, elements_through=1, element_order=2, second_order_incomplete=True)

    assert mesh.elements["Hexahedron 20"].shape[1] == 20


def test_hexahedral_mesh_needs_seventeen_blocks(cartilage_hex_blocks):
    with pytest.raises(ValueError, match="seventeen-block"):
        hexahedral_mesh(cartilage_hex_blocks)


@pytest.mark.parametrize(("arguments", "message"), [({"element_size": 0.0}, "element_size"), ({"elements_through": 0}, "at least 1")])
def test_hexahedral_mesh_rejects_invalid_arguments(femoral_hex_blocks, arguments, message):
    with pytest.raises(ValueError, match=message):
        hexahedral_mesh(femoral_hex_blocks, **arguments)


def test_real_femoral_hexahedra_are_valid_and_optimize(real_femoral_hex_blocks):
    mesh = hexahedral_mesh(real_femoral_hex_blocks)

    before = mesh_quality(mesh)
    after = mesh_quality(optimize_hexahedral_mesh(mesh, aspect_weight=0.5))

    # The worst elements before optimizing sit at the notch's corners and the tapered lateral condyle's end.
    assert before.minimum > 0.05
    assert after.minimum > 0.4
    assert after.mean > 0.85
    assert after.max_aspect < min(before.max_aspect, 5.0)


def test_hexahedral_mesh_keeps_its_wall_columns(femoral_hexahedra):
    lines = femoral_hexahedra.straight_lines

    # One column of four nodes, three elements through, at every node around the U's perimeter.
    assert lines.shape[1] == 4
    assert len(lines) > 50
    assert len(np.unique(lines)) == lines.size


def test_scaled_jacobian_alone_stretches_real_femoral_elements(real_femoral_hex_blocks):
    mesh = hexahedral_mesh(real_femoral_hex_blocks)

    alone = mesh_quality(optimize_hexahedral_mesh(mesh))
    combined = mesh_quality(optimize_hexahedral_mesh(mesh, aspect_weight=0.5))

    # Straight walls keep it from stretching much along the rim, as it did without them.
    assert alone.max_aspect > 1.4 * combined.max_aspect
