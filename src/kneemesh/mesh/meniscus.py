"""Hexahedral meshing of the meniscus's five-block butterfly solid, and the meniscus's named sets of faces.

Both meshes of a meniscus have the sets "anterior_horn", "posterior_horn",
and "body" besides "surface" (mesh_sets): the faces on the caps at either
end and on the rest of its surface.
"""

import gmsh
import numpy as np

from kneemesh.brep import BRep, gmsh_session
from kneemesh.tet_meshing import Mesh, gmsh_mesh, gmsh_options, import_solid, require_element_order


def hexahedral_mesh(
    brep: BRep,
    elements_per_edge: int = 4,
    elements_along: int = 20,
    element_order: int = 1,
    second_order_incomplete: bool = False,
) -> Mesh:
    """Mesh a solid of six-sided blocks with hexahedra, using transfinite surfaces and volumes.

    Every block must have six faces, such as the five blocks from
    contours_to_hex_blocks. Edges on the planar end caps get
    elements_per_edge elements, and the edges running between the caps get
    elements_along, so each block is a grid of elements_per_edge by
    elements_per_edge quadrilaterals swept elements_along times.

    Args:
        brep: A solid of six-sided blocks with planar caps.
        elements_per_edge: Elements along each edge of a cap.
        elements_along: Elements along the solid between the caps.
        element_order: Polynomial order of the elements. 2 gives 27-node hexahedra.
        second_order_incomplete: Omit face and interior nodes from high-order
            elements, giving 20-node instead of 27-node second-order hexahedra.

    Returns:
        The hexahedral mesh.

    Raises:
        ValueError: If a count is below 1, or brep is not a solid of six-sided
            blocks with planar caps.
    """
    if elements_per_edge < 1 or elements_along < 1:
        raise ValueError(f"Element counts must be at least 1, got {elements_per_edge} and {elements_along}")
    require_element_order(element_order)

    options: dict[str, float | bool] = {"Mesh.ElementOrder": element_order, "Mesh.SecondOrderIncomplete": second_order_incomplete}
    with gmsh_session(), gmsh_options(options):
        import_solid(brep)
        volumes = [tag for _, tag in gmsh.model.getEntities(3)]
        caps = [tag for _, tag in gmsh.model.getEntities(2) if gmsh.model.getType(2, tag) == "Plane"]
        six_sided = all(len(gmsh.model.getBoundary([(3, volume)], oriented=False)) == 6 for volume in volumes)
        if not six_sided or len(caps) != 2 * len(volumes):
            raise ValueError("Hexahedral meshing needs a solid of six-sided blocks with planar caps, such as contours_to_hex_blocks gives")

        cap_edges = {abs(tag) for _, tag in gmsh.model.getBoundary([(2, cap) for cap in caps], combined=False, oriented=False)}
        for _, curve in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(curve, (elements_per_edge if curve in cap_edges else elements_along) + 1)
        for _, face in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(face)
            gmsh.model.mesh.setRecombine(2, face)
        for volume in volumes:
            gmsh.model.mesh.setTransfiniteVolume(volume)
        gmsh.model.add_physical_group(3, volumes, name="solid")
        groups = meniscus_face_groups()

        gmsh.model.mesh.generate(3)
        return gmsh_mesh(3, groups)


ANTERIOR_HORN = "anterior_horn"
"""Name of a meniscus mesh's set of the faces on the cap at its anterior end, and of their nodes."""
POSTERIOR_HORN = "posterior_horn"
"""Name of a meniscus mesh's set of the faces on the cap at its posterior end, and of their nodes."""
BODY = "body"
"""Name of a meniscus mesh's set of the faces on its surface between the horns' caps, and of their nodes."""


def meniscus_face_groups() -> dict[str, list[int]]:
    """Name the current model's meniscus faces, for tet_meshing.tetrahedral_mesh's face_groups: its horns' caps and its body.

    The caps are the planar boundary faces, as on hexahedral blocks and
    planar-capped solids, or, with smooth caps, every boundary face but the
    largest, the loft. Caps sharing curves make one horn. The anterior horn
    is the one at lower y: SimpleITK's physical space is LPS, so y grows
    toward the patient's back.

    Raises:
        ValueError: If the caps do not form two horns.
    """
    boundary = [tag for _, tag in gmsh.model.getEntities(2) if len(gmsh.model.getAdjacencies(2, tag)[0]) == 1]
    planar = [face for face in boundary if gmsh.model.getType(2, face) == "Plane"]
    caps = planar or sorted(boundary, key=lambda face: gmsh.model.occ.getMass(2, face))[:-1]
    curves = {face: {abs(tag) for _, tag in gmsh.model.getBoundary([(2, face)], combined=False, oriented=False)} for face in caps}
    horns: list[set[int]] = []
    for face in caps:
        joined = [horn for horn in horns if any(curves[face] & curves[other] for other in horn)]
        horns = [horn for horn in horns if horn not in joined] + [set().union({face}, *joined)]
    if len(horns) != 2:
        raise ValueError(f"A meniscus's caps should form two horns, got {len(horns)}")

    def y(horn: set[int]) -> float:
        masses = np.array([gmsh.model.occ.getMass(2, face) for face in horn])
        centers = np.array([gmsh.model.occ.getCenterOfMass(2, face) for face in horn])
        return float(masses @ centers[:, 1] / masses.sum())

    anterior, posterior = sorted(horns, key=y)
    return {ANTERIOR_HORN: sorted(anterior), POSTERIOR_HORN: sorted(posterior), BODY: sorted(set(boundary) - set(caps))}
