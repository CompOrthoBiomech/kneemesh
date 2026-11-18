"""Boundary faces of volume meshes, and the named node and element sets built from them.

A Mesh (tet_meshing.Mesh) numbers its elements, then its boundary faces,
each kind in the order of its dict; its element sets hold those numbers,
0-based, and its node sets indices into its nodes. Every structure's mesh
has a "surface" set of its boundary faces, or of its own elements for a
bone's surface mesh, and of their nodes. Structure meshers add named
groups of faces, such as a cartilage's articular side, from the B-rep
faces gmsh meshed them on.
"""

from collections.abc import Mapping

import numpy as np

SURFACE = "surface"
"""Name of the set of every structure's boundary faces, or of a surface mesh's elements, and of their nodes."""

# gmsh's corners of each face of a volume element, ordered so the face's normal points out, and its edges, in the
# order of the middle nodes that follow the corners. Hexahedron 27's face centers follow its edges' nodes in the
# order of its faces.
_TETRAHEDRON_FACES = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (3, 1, 2))
_TETRAHEDRON_EDGES = ((0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1))
_HEXAHEDRON_FACES = ((0, 3, 2, 1), (0, 1, 5, 4), (0, 4, 7, 3), (1, 2, 6, 5), (2, 3, 7, 6), (4, 5, 6, 7))
_HEXAHEDRON_EDGES = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7))

FACE_ELEMENTS: dict[str, tuple[str, int]] = {
    "Tetrahedron 4": ("Triangle 3", 2),
    "Tetrahedron 10": ("Triangle 6", 9),
    "Hexahedron 8": ("Quadrilateral 4", 3),
    "Hexahedron 20": ("Quadrilateral 8", 16),
    "Hexahedron 27": ("Quadrilateral 9", 10),
}
"""The gmsh name and type number of the faces of each volume element that has boundary faces."""


def _face_nodes(name: str) -> list[list[int]]:
    """Each face's local node indices in its volume element, in gmsh's ordering of the face element."""
    tetrahedron = name.startswith("Tetrahedron")
    faces, edges = (_TETRAHEDRON_FACES, _TETRAHEDRON_EDGES) if tetrahedron else (_HEXAHEDRON_FACES, _HEXAHEDRON_EDGES)
    first_middle = 4 if tetrahedron else 8
    nodes = int(name.split()[1])
    middles = {frozenset(edge): first_middle + position for position, edge in enumerate(edges)}
    result = []
    for position, corners in enumerate(faces):
        face = list(corners)
        if nodes > first_middle:
            face += [middles[frozenset((corner, corners[(index + 1) % len(corners)]))] for index, corner in enumerate(corners)]
        if nodes == 27:
            face.append(20 + position)
        result.append(face)
    return result


def boundary_faces(elements: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the faces of volume elements that belong to only one element, keyed by gmsh name, oriented outward.

    Args:
        elements: Connectivity of tetrahedra or hexahedra by gmsh name, in gmsh's node ordering.

    Raises:
        ValueError: If an element type has no face table here.
    """
    candidates: dict[str, list[np.ndarray]] = {}
    for name, connectivity in elements.items():
        if name not in FACE_ELEMENTS:
            raise ValueError(f"No boundary faces for {name} elements; expected one of {list(FACE_ELEMENTS)}")
        local = _face_nodes(name)
        candidates.setdefault(FACE_ELEMENTS[name][0], []).extend(connectivity[:, face] for face in local)
    faces = {}
    for name, parts in candidates.items():
        every = np.concatenate(parts)
        corners = 3 if name.startswith("Triangle") else 4
        faces[name] = every[unshared(every[:, :corners])]
    return faces


def unshared(faces: np.ndarray) -> np.ndarray:
    """Return a mask of the rows of faces (F, corners) whose set of corner nodes no other row has.

    Sorting the rows with lexsort and comparing neighbors is about three times
    as fast as np.unique(..., axis=0), which sorts them as opaque records.
    """
    keys = np.sort(faces, axis=1)
    order = np.lexsort(keys.T[::-1])
    ordered = keys[order]
    starts = np.ones(len(ordered), dtype=bool)
    starts[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    group = np.cumsum(starts) - 1
    mask = np.zeros(len(faces), dtype=bool)
    mask[order] = np.bincount(group)[group] == 1
    return mask


def face_groups_of(faces: Mapping[str, np.ndarray], surfaces: Mapping[int, np.ndarray]) -> np.ndarray:
    """Return the tag of the surface each face lies on, -1 where none, over the faces concatenated in dict order.

    Args:
        faces: Boundary faces by gmsh name, as boundary_faces gives.
        surfaces: Corner node indices of the elements gmsh made on each B-rep surface, by its tag.
    """
    tags: dict[tuple[int, ...], int] = {}
    for tag, corners in surfaces.items():
        tags.update((key, tag) for key in map(tuple, np.sort(corners, axis=1).tolist()))
    keys = [np.sort(connectivity[:, : 3 if name.startswith("Triangle") else 4], axis=1).tolist() for name, connectivity in faces.items()]
    return np.array([tags.get(tuple(key), -1) for rows in keys for key in rows], dtype=np.int64)


def face_sets(
    elements: Mapping[str, np.ndarray], faces: Mapping[str, np.ndarray], groups: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return element and node sets of a volume mesh's boundary faces: "surface", then one per group.

    Args:
        elements: The mesh's volume elements by gmsh name, numbered first.
        faces: Its boundary faces by gmsh name, numbered after the elements.
        groups: Boolean masks over the faces concatenated in dict order, by set name.
    """
    first = sum(len(connectivity) for connectivity in elements.values())
    count = sum(len(connectivity) for connectivity in faces.values())
    offsets = np.cumsum([0, *(len(connectivity) for connectivity in faces.values())])
    element_sets: dict[str, np.ndarray] = {}
    node_sets: dict[str, np.ndarray] = {}
    for name, mask in {SURFACE: np.ones(count, dtype=bool), **groups}.items():
        element_sets[name] = first + np.flatnonzero(mask)
        nodes = [connectivity[mask[start:end]].ravel() for connectivity, start, end in zip(faces.values(), offsets[:-1], offsets[1:], strict=True)]
        node_sets[name] = np.unique(np.concatenate([np.zeros(0, dtype=np.int64), *nodes]))
    return element_sets, node_sets


def surface_sets(elements: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the "surface" element and node sets of a surface mesh: every element and every node it uses."""
    count = sum(len(connectivity) for connectivity in elements.values())
    nodes = [connectivity.ravel() for connectivity in elements.values()]
    return {SURFACE: np.arange(count)}, {SURFACE: np.unique(np.concatenate([np.zeros(0, dtype=np.int64), *nodes]))}
