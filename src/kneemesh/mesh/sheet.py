"""Hexahedral meshing of a cartilage sheet's five-block butterfly solid."""

import gmsh

from kneemesh.brep import BRep, gmsh_session
from kneemesh.tet_meshing import Mesh, gmsh_options, import_solid, require_element_order, volume_mesh

_BLOCKS = 5
_SHAPE_ERROR = "Hexahedral meshing needs the five-block butterfly solid of a cartilage sheet, such as sheet_to_hex_blocks gives"


def hexahedral_mesh(
    brep: BRep,
    elements_around: int = 8,
    elements_radial: int = 4,
    elements_through: int = 3,
    element_order: int = 1,
    second_order_incomplete: bool = False,
) -> Mesh:
    """Mesh a cartilage sheet's five-block solid with hexahedra, using transfinite surfaces and volumes.

    Edge counts follow from the solid's topology alone. Opposite edges of
    each four-sided face must have equal counts, so edges joined through
    faces form a class, and the butterfly has four: two run around the sheet,
    each holding two opposite perimeter arcs and the inner block's sides
    opposite them, one runs through the thickness, and one along the spokes.
    The inner block is the one block sharing all four walls.

    Args:
        brep: The five-block solid from sheet_to_hex_blocks.
        elements_around: Elements along each of the four perimeter arcs and
            each side of the inner block, so 4 * elements_around around the perimeter.
        elements_radial: Elements along each spoke, from the perimeter to the inner block.
        elements_through: Elements through the thickness.
        element_order: Polynomial order of the elements. 2 gives 27-node hexahedra.
        second_order_incomplete: Omit face and interior nodes from high-order
            elements, giving 20-node instead of 27-node second-order hexahedra.

    Returns:
        The hexahedral mesh, (4 * elements_radial + elements_around) *
        elements_around * elements_through elements.

    Raises:
        ValueError: If a count is below 1, or brep is not a five-block sheet solid.
    """
    if min(elements_around, elements_radial, elements_through) < 1:
        raise ValueError(f"Element counts must be at least 1, got {elements_around}, {elements_radial}, and {elements_through}")
    require_element_order(element_order)

    options: dict[str, float | bool] = {"Mesh.ElementOrder": element_order, "Mesh.SecondOrderIncomplete": second_order_incomplete}
    with gmsh_session(), gmsh_options(options):
        import_solid(brep)
        counts = {"around": elements_around, "radial": elements_radial, "through": elements_through}
        for curve, kind in _edge_classes().items():
            gmsh.model.mesh.setTransfiniteCurve(curve, counts[kind] + 1)
        for _, face in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(face)
            gmsh.model.mesh.setRecombine(2, face)
        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        for volume in volumes:
            gmsh.model.mesh.setTransfiniteVolume(volume)
        gmsh.model.add_physical_group(3, volumes, name="solid")

        gmsh.model.mesh.generate(3)
        return volume_mesh()


def _edge_classes() -> dict[int, str]:
    """Return "around", "radial", or "through" for every curve of the current model's five-block sheet solid.

    Raises:
        ValueError: If the model is not five six-sided blocks of four-sided faces in four edge classes.
    """
    volumes = [tag for _, tag in gmsh.model.getEntities(3)]
    if len(volumes) != _BLOCKS:
        raise ValueError(f"{_SHAPE_ERROR}; got {len(volumes)} volumes")
    sides: dict[int, list[int]] = {}
    for _, face in gmsh.model.getEntities(2):
        sides[face] = [abs(tag) for _, tag in gmsh.model.getBoundary([(2, face)], combined=False, oriented=True)]
        if len(sides[face]) != 4:
            raise ValueError(f"{_SHAPE_ERROR}; face {face} has {len(sides[face])} sides")
    for volume in volumes:
        if len(gmsh.model.getBoundary([(3, volume)], oriented=False)) != 6:
            raise ValueError(f"{_SHAPE_ERROR}; volume {volume} does not have six faces")

    curves = [tag for _, tag in gmsh.model.getEntities(1)]
    parent = {curve: curve for curve in curves}

    def root(curve: int) -> int:
        while parent[curve] != curve:
            parent[curve] = parent[parent[curve]]
            curve = parent[curve]
        return curve

    # A face's boundary runs around it in order, so sides 0 and 2, and 1 and 3, are opposite.
    for edges in sides.values():
        for first, second in ((edges[0], edges[2]), (edges[1], edges[3])):
            parent[root(first)] = root(second)
    classes = {curve: root(curve) for curve in curves}
    if len(set(classes.values())) != 4:
        raise ValueError(f"{_SHAPE_ERROR}; its edges fall in {len(set(classes.values()))} classes instead of 4")

    # The inner block shares all four of its walls with the outer blocks,
    # which share three each. The class common to those walls runs through the
    # thickness, and its other classes run around: arcs 0 and 2 with the inner
    # sides opposite them, and arcs 1 and 3 with theirs.
    shared = {face for face in sides if len(gmsh.model.getAdjacencies(2, face)[0]) == 2}
    inner = [volume for volume in volumes if sum(abs(tag) in shared for _, tag in gmsh.model.getBoundary([(3, volume)], oriented=False)) == 4]
    if len(inner) != 1:
        raise ValueError(f"{_SHAPE_ERROR}; {len(inner)} blocks share four faces instead of one")
    inner_faces = [abs(tag) for _, tag in gmsh.model.getBoundary([(3, inner[0])], oriented=False)]
    walls = [{classes[edge] for edge in sides[face]} for face in inner_faces if face in shared]
    through = set.intersection(*walls)
    around = {classes[edge] for face in inner_faces for edge in sides[face]} - through
    if len(through) != 1 or len(around) != 2:
        raise ValueError(f"{_SHAPE_ERROR}; the inner block's edges are not in one through-thickness and two around classes")
    names = {kind: "around" for kind in around} | {through.pop(): "through"}
    return {curve: names.get(kind, "radial") for curve, kind in classes.items()}
