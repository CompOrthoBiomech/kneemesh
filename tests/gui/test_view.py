from dataclasses import replace

import numpy as np

from kneemesh.disk import write_mesh
from kneemesh.gui.labels import LabelEntry
from kneemesh.gui.view import Scene, read_mesh_preview
from kneemesh.mesh_optimization import element_quality
from kneemesh.tet_meshing import Mesh


def _cube_mesh(path, lift=0.0):
    """Two hexahedra side by side, with their faces and quality, written as a preview; lift raises one top corner."""
    nodes = np.array([[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0)])
    nodes[11, 2] += lift
    corners = [[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]]
    mesh = Mesh(nodes, {"Hexahedron 8": np.array(corners)}, {"Hexahedron 8": 5}).with_face_sets()
    write_mesh(replace(mesh, quality=element_quality(mesh)), path)
    return read_mesh_preview(path)


def _visible(actors):
    return [actor.GetVisibility() for actor in actors]


def test_meshes_are_colored_by_a_metric_with_a_scalar_bar(tmp_path):
    scene = Scene()
    scene.set_mesh("lateral_meniscus", _cube_mesh(tmp_path / "a.vtu"))
    scene.set_mesh("medial_meniscus", _cube_mesh(tmp_path / "b.vtu", lift=0.5))
    assert scene.metrics() == ["scaled_jacobian", "frobenius_aspect", "warp"]

    scene.refresh([], "latest", [], metric="scaled_jacobian")
    assert scene.scalar_bar.GetVisibility()
    low, high = scene.lookup_table.GetTableRange()
    assert low < 1.0
    assert np.isclose(high, 1.0)
    mapper = scene._stages[("mesh", "medial_meniscus")][0].GetMapper()
    assert mapper.GetScalarVisibility()
    # The element metrics are drawn on the elements' outer faces, the warp on the boundary faces.
    assert mapper.GetInput().GetNumberOfCells() == 10

    scene.refresh([], "latest", [], metric="warp")
    assert scene.lookup_table.GetTableRange()[0] == 0.0
    assert scene.lookup_table.GetTableRange()[1] > 10

    # Only the shown meshes set the range.
    scene.refresh([], "latest", [], solo="lateral_meniscus", metric="scaled_jacobian")
    assert np.allclose(scene.lookup_table.GetTableRange()[0], 1.0)

    scene.refresh([], "latest", [])
    assert not scene.scalar_bar.GetVisibility()
    assert not mapper.GetScalarVisibility()


def test_solo_hides_every_other_structure_and_its_labels(tmp_path):
    scene = Scene()
    scene.set_mesh("lateral_meniscus", _cube_mesh(tmp_path / "a.vtu"))
    scene.set_mesh("medial_meniscus", _cube_mesh(tmp_path / "b.vtu"))
    surface = scene._stages[("mesh", "lateral_meniscus")][0].GetMapper().GetInput()
    entries = [
        LabelEntry(id=1, path="a.nii", name="a.nii", file_value=1, structure="tibia", value=1, include=True),
        LabelEntry(id=2, path="a.nii", name="a.nii", file_value=2, structure=None, value=2, include=True),
    ]
    for entry in entries:
        scene.set_label(entry["id"], surface)

    scene.refresh(entries, "latest", [], solo="medial_meniscus")
    assert _visible(scene._stages[("mesh", "lateral_meniscus")] + scene._labels[1] + scene._labels[2]) == [False, False, False]
    assert _visible(scene._stages[("mesh", "medial_meniscus")]) == [True]
    scene.refresh(entries, "latest", ["medial_meniscus"])
    assert _visible(scene._stages[("mesh", "lateral_meniscus")] + scene._labels[1] + scene._labels[2]) == [True, True, True]
