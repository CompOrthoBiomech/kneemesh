"""Hexahedral meshing of the femoral cartilage's seventeen-block solid. The femur is not implemented yet."""

import gmsh

from kneemesh.brep import BRep, gmsh_session
from kneemesh.config import DEFAULT_FEMORAL_ELEMENT_SIZE
from kneemesh.mesh.sheet import sized_hexahedral_mesh
from kneemesh.tet_meshing import Mesh, import_solid

_BLOCKS = 17


def hexahedral_mesh(
    brep: BRep,
    element_size: float = DEFAULT_FEMORAL_ELEMENT_SIZE,
    elements_through: int = 3,
    element_order: int = 1,
    second_order_incomplete: bool = False,
) -> Mesh:
    """Mesh the femoral cartilage's seventeen-block solid with hexahedra of about element_size across.

    Its five core and twelve ring blocks have more edge classes than the
    butterfly's, so counts follow from element_size (sized_hexahedral_mesh)
    rather than being given per class.

    Args:
        brep: The seventeen-block solid from femoral_sheet_to_hex_blocks.
        element_size: Target length of the elements' edges along the sheet.
        elements_through: Elements through the thickness.
        element_order: Polynomial order of the elements. 2 gives 27-node hexahedra.
        second_order_incomplete: Omit face and interior nodes from high-order
            elements, giving 20-node instead of 27-node second-order hexahedra.

    Raises:
        ValueError: If a size or count is out of range, or brep is not seventeen blocks.
    """
    with gmsh_session():
        import_solid(brep)
        volumes = len(gmsh.model.getEntities(3))
    if volumes != _BLOCKS:
        raise ValueError(
            f"Hexahedral meshing needs the femoral cartilage's seventeen-block solid, such as femoral_sheet_to_hex_blocks gives; got {volumes} volumes"
        )
    return sized_hexahedral_mesh(brep, element_size, elements_through, element_order, second_order_incomplete)
